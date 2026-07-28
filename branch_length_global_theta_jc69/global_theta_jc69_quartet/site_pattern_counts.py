"""
FASTA and quartet site-pattern utilities for the JC69 global-theta estimator.

The estimator uses JC69 symmetry classes. For example, raw patterns ``AAAA``,
``CCCC``, ``GGGG``, and ``TTTT`` all canonicalize to ``AAAA``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence, Tuple

from .symbolic_substitution_models import STATE_ORDER, build_symbolic_model


Pattern = Tuple[str, str, str, str]


@dataclass(frozen=True)
class PatternCountSummary:
    canonical_counts: Mapping[str, float]
    raw_counts: Mapping[str, float]
    total_sites_used: int
    skipped_sites: int
    pairwise_valid_sites: Mapping[Tuple[str, str], float]
    pairwise_mismatches: Mapping[Tuple[str, str], float]


def parse_pattern_literal(raw: str) -> Pattern:
    """Parse a quartet site pattern such as ``AGCT`` or ``(A,G,C,T)``."""
    text = raw.strip().upper()
    text = text.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    tokens = [token for token in re.split(r"[,\s]+", text) if token]
    if len(tokens) == 1 and len(tokens[0]) == 4:
        tokens = list(tokens[0])
    if len(tokens) != 4:
        raise ValueError(f"Expected a quartet site pattern such as AGCT. Got '{raw}'.")
    model = build_symbolic_model("JC69")
    normalized: List[str] = []
    for token in tokens:
        state = model.normalize_observation(token)
        if state is None:
            raise ValueError(f"Unsupported observation '{token}' for JC69.")
        normalized.append(state)
    return normalized[0], normalized[1], normalized[2], normalized[3]


def canonicalize_pattern(pattern: Sequence[str] | str, model_name: str = "JC69") -> str:
    """Canonicalize a four-nucleotide pattern under JC69 state-label symmetry."""
    if model_name.strip().upper() != "JC69":
        raise ValueError("This integration package supports only JC69.")
    raw = tuple(pattern) if not isinstance(pattern, str) else tuple(pattern.strip().upper())
    if len(raw) != 4:
        raise ValueError(f"Expected a quartet pattern of length 4. Got {pattern!r}.")
    model = build_symbolic_model("JC69")
    normalized: List[str] = []
    for token in raw:
        state = model.normalize_observation(token)
        if state is None:
            raise ValueError(f"Unsupported observation '{token}' for JC69.")
        normalized.append(state)

    mapping: Dict[str, str] = {}
    next_symbol = 0
    out: List[str] = []
    for nt in normalized:
        if nt not in mapping:
            mapping[nt] = STATE_ORDER[next_symbol]
            next_symbol += 1
        out.append(mapping[nt])
    return "".join(out)


def pattern_class_representatives(model_name: str = "JC69") -> Dict[str, Pattern]:
    if model_name.strip().upper() != "JC69":
        raise ValueError("This integration package supports only JC69.")
    reps: Dict[str, Pattern] = {}
    for pattern in itertools.product(STATE_ORDER, repeat=4):
        code = canonicalize_pattern(pattern)
        reps.setdefault(code, pattern)
    return reps


CANONICAL_REPRESENTATIVES = pattern_class_representatives("JC69")


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_label: str | None = None
    current_chunks: List[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "":
                continue
            if line.startswith(">"):
                if current_label is not None:
                    records.append((current_label, "".join(current_chunks).upper()))
                current_label = line[1:].strip()
                if current_label == "":
                    raise ValueError(f"Encountered an empty FASTA label in {path}.")
                current_chunks = []
                continue
            if current_label is None:
                raise ValueError(f"Malformed FASTA in {path}: sequence before first header.")
            current_chunks.append(line)

    if current_label is not None:
        records.append((current_label, "".join(current_chunks).upper()))
    return records


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _initial_pairwise_maps(
    taxa_order: Sequence[str],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    pairwise_valid: Dict[Tuple[str, str], float] = {
        _pair_key(taxa_order[i], taxa_order[j]): 0.0
        for i in range(4)
        for j in range(i + 1, 4)
    }
    pairwise_mismatches = {key: 0.0 for key in pairwise_valid}
    return pairwise_valid, pairwise_mismatches


def _update_pairwise_counts(
    pattern: Pattern,
    taxa_order: Sequence[str],
    weight: float,
    pairwise_valid: Dict[Tuple[str, str], float],
    pairwise_mismatches: Dict[Tuple[str, str], float],
) -> None:
    for i in range(4):
        for j in range(i + 1, 4):
            key = _pair_key(taxa_order[i], taxa_order[j])
            pairwise_valid[key] += weight
            if pattern[i] != pattern[j]:
                pairwise_mismatches[key] += weight


def summarize_site_patterns_from_count_file(
    counts_path: Path,
    taxa_order: Sequence[str],
) -> PatternCountSummary:
    """Read a simple ``pattern count`` file for one quartet."""
    raw_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    pairwise_valid, pairwise_mismatches = _initial_pairwise_maps(taxa_order)

    with counts_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.lower().startswith("pattern") and "count" in line.lower():
                continue

            match = re.match(r"^(.*?)[,\t ]+([0-9]+(?:\.[0-9]+)?)\s*$", line)
            if match is None:
                raise ValueError(
                    f"Could not parse counts line {line_no} in {counts_path}. Expected '<pattern> <count>'."
                )

            pattern = parse_pattern_literal(match.group(1))
            count = float(match.group(2))
            if count < 0.0:
                raise ValueError(f"Negative site-pattern count on line {line_no} in {counts_path}.")

            key = "".join(pattern)
            raw_counts[key] += count
            canonical_counts[canonicalize_pattern(pattern)] += count
            _update_pairwise_counts(pattern, taxa_order, count, pairwise_valid, pairwise_mismatches)

    return PatternCountSummary(
        canonical_counts=dict(canonical_counts),
        raw_counts=dict(raw_counts),
        total_sites_used=int(round(sum(raw_counts.values()))),
        skipped_sites=0,
        pairwise_valid_sites=pairwise_valid,
        pairwise_mismatches=pairwise_mismatches,
    )
