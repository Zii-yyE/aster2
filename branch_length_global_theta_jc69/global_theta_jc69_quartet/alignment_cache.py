"""In-memory FASTA cache for repeated JC69 quartet site-pattern extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .site_pattern_counts import PatternCountSummary, canonicalize_pattern, read_fasta
from .symbolic_substitution_models import build_symbolic_model


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class LoadedAlignment:
    seq_by_taxon: Mapping[str, str]
    length: int
    source: str


class AlignmentCache:
    """Load FASTA alignments once and reuse them across many quartet fits."""

    def __init__(self, alignments: Sequence[LoadedAlignment], state_counts: Mapping[str, int]) -> None:
        self.alignments = list(alignments)
        self._state_counts = dict(state_counts)
        self.model_name = "JC69"
        self._summary_cache: Dict[Tuple[str, str, str, str], PatternCountSummary] = {}
        self._nucleotide_frequency_cache: Optional[Mapping[str, float]] = None

    @classmethod
    def from_paths(cls, alignment_paths: Sequence[str | Path], model_name: str = "JC69") -> "AlignmentCache":
        if model_name.strip().upper() != "JC69":
            raise ValueError("This integration package supports only JC69.")
        loaded = []
        model = build_symbolic_model("JC69")
        state_counts: Counter[str] = Counter()
        for raw_path in alignment_paths:
            path = Path(raw_path)
            records = read_fasta(path)
            seq_by_taxon = {label: seq for label, seq in records}
            lengths = {len(seq) for seq in seq_by_taxon.values()}
            if len(lengths) != 1:
                raise ValueError(f"Alignment {path} does not have equal-length sequences.")
            for seq in seq_by_taxon.values():
                for token in seq:
                    state = model.normalize_observation(token.upper())
                    if state is not None:
                        state_counts[state] += 1
            loaded.append(
                LoadedAlignment(
                    seq_by_taxon=seq_by_taxon,
                    length=next(iter(lengths), 0),
                    source=str(path),
                )
            )
        if not loaded:
            raise ValueError("At least one alignment file is required.")
        return cls(loaded, state_counts)

    def empirical_nucleotide_frequencies(self) -> Mapping[str, float]:
        if self._nucleotide_frequency_cache is not None:
            return self._nucleotide_frequency_cache
        total = sum(self._state_counts.values())
        if total <= 0:
            raise ValueError("Could not compute nucleotide frequencies: no A/C/G/T observations.")
        result = {nt: float(self._state_counts.get(nt, 0)) / float(total) for nt in ("A", "C", "G", "T")}
        self._nucleotide_frequency_cache = result
        return result

    def summarize_quartet(self, taxa_order: Sequence[str]) -> PatternCountSummary:
        if len(taxa_order) != 4:
            raise ValueError("Expected exactly 4 taxa for a quartet summary.")
        key = tuple(taxa_order)
        if key in self._summary_cache:
            return self._summary_cache[key]

        model = build_symbolic_model("JC69")
        raw_counts: Counter[str] = Counter()
        canonical_counts: Counter[str] = Counter()
        pairwise_valid = {
            _pair_key(taxa_order[i], taxa_order[j]): 0.0
            for i in range(4)
            for j in range(i + 1, 4)
        }
        pairwise_mismatches = {pair: 0.0 for pair in pairwise_valid}
        total_sites_used = 0
        skipped_sites = 0

        for alignment in self.alignments:
            missing = [taxon for taxon in taxa_order if taxon not in alignment.seq_by_taxon]
            if missing:
                raise ValueError(
                    f"Alignment {alignment.source} is missing taxa required by the quartet: {missing}."
                )
            seqs = [alignment.seq_by_taxon[taxon] for taxon in taxa_order]
            for idx in range(alignment.length):
                raw_tokens = [seq[idx].upper() for seq in seqs]
                normalized: list[str] = []
                skip_site = False
                for token in raw_tokens:
                    state = model.normalize_observation(token)
                    if state is None:
                        skip_site = True
                        break
                    normalized.append(state)
                if skip_site:
                    skipped_sites += 1
                    continue
                pattern = tuple(normalized)
                raw_key = "".join(pattern)
                raw_counts[raw_key] += 1
                canonical_counts[canonicalize_pattern(pattern)] += 1
                for i in range(4):
                    for j in range(i + 1, 4):
                        pair = _pair_key(taxa_order[i], taxa_order[j])
                        pairwise_valid[pair] += 1.0
                        if pattern[i] != pattern[j]:
                            pairwise_mismatches[pair] += 1.0
                total_sites_used += 1

        summary = PatternCountSummary(
            canonical_counts=dict(canonical_counts),
            raw_counts=dict(raw_counts),
            total_sites_used=total_sites_used,
            skipped_sites=skipped_sites,
            pairwise_valid_sites=pairwise_valid,
            pairwise_mismatches=pairwise_mismatches,
        )
        self._summary_cache[key] = summary
        return summary
