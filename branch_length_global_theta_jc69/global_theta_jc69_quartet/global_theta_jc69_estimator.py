"""
N-taxon branch-length estimator based on rooted quartet stitching.

This is the minimal JC69/global-theta version intended for integration into
aster2. It assumes:
- a fixed rooted binary species-tree topology,
- one sequence per taxon in one or more aligned FASTA files,
- JC69 substitution,
- one global theta = 4 Ne mu shared by the full species tree.

The code does not evaluate a full n-taxon MSC likelihood. For each non-root
branch it builds edge-preserving witness quartets, fits each unique quartet
under the global-theta quartet likelihood, and stitches the branch estimates
back onto the full topology.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import random
import statistics
import time
from typing import Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - integration fallback
    def tqdm(iterable, **_kwargs):
        return iterable

from .alignment_cache import AlignmentCache
from .quartet_site_pattern_probability import QuartetSitePatternProbabilityEngine
from .quartet_theta_estimator import QuartetThetaEstimator, THETA_SYMBOL, parse_quartet_topology
from .site_pattern_counts import PatternCountSummary
from .topology_utils import (
    TopologyNode,
    clade_label,
    format_newick_with_branch_lengths,
    format_topology_newick,
    induce_rooted_quartet,
    iter_nodes_postorder,
    leaf_names_left_to_right,
    load_topology_arg,
    sibling,
)


GENERIC_QUARTET_TAXA = ("q1", "q2", "q3", "q4")
DEFAULT_OUTPUT_DIR_NAME = "global_theta_jc69_estimate"


def _emit(message: str = "") -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class WitnessSpec:
    """One quartet that preserves one target branch of the full tree."""

    quartet_taxa: Tuple[str, str, str, str]
    target_clade: FrozenSet[str]


@dataclass(frozen=True)
class QuartetFitResult:
    quartet_taxa: Tuple[str, str, str, str]
    quartet_topology: str
    estimated_theta: float
    branch_coalescent: Mapping[FrozenSet[str], float]
    branch_substitution: Mapping[FrozenSet[str], float]
    log_likelihood: float
    total_sites_used: int


@dataclass(frozen=True)
class BranchAggregate:
    clade: FrozenSet[str]
    coalescent: float
    substitution: float
    attempted_witnesses: int
    successful_witnesses: int
    failed_witnesses: int
    quartet_taxa: Tuple[Tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class EstimatedNtaxaResult:
    topology_newick: str
    estimated_theta: float
    nucleotide_frequencies: Mapping[str, float]
    coalescent_newick: str
    substitution_newick: str
    branch_estimates: Mapping[FrozenSet[str], BranchAggregate]
    unique_quartets_attempted: int
    unique_quartets_succeeded: int
    unique_quartets_failed: int
    num_input_alignments: int
    total_alignment_columns: int
    non_root_branch_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    substitution_tree: Path
    coalescent_tree: Path
    theta_summary: Path


def _quartet_sort_key(quartet_taxa: Tuple[str, str, str, str]) -> Tuple[str, str, str, str]:
    return quartet_taxa


def _parse_label_to_clade(text: str) -> FrozenSet[str]:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        raise ValueError(f"Could not parse clade label '{text}'.")
    inner = stripped[1:-1].strip()
    if inner == "":
        return frozenset()
    return frozenset(part.strip() for part in inner.split(",") if part.strip())


def _aggregate(values: Sequence[float], method: str) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty sequence.")
    name = method.strip().lower()
    if name == "mean":
        return float(sum(values) / len(values))
    if name == "median":
        return float(statistics.median(values))
    raise ValueError(f"Unsupported aggregation method '{method}'.")


def _relabel_quartet_topology(node: TopologyNode, taxon_map: Mapping[str, str]) -> TopologyNode:
    if node.is_leaf:
        assert node.name is not None
        name = taxon_map[node.name]
        return TopologyNode(name=name, clade=frozenset((name,)))

    children = [_relabel_quartet_topology(child, taxon_map) for child in node.children]
    relabeled = TopologyNode(children=children)
    for child in children:
        child.parent = relabeled
    relabeled.clade = frozenset().union(*(child.clade for child in children))
    return relabeled


def _relabel_pattern_summary(summary: PatternCountSummary, taxon_map: Mapping[str, str]) -> PatternCountSummary:
    def remap_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
        a, b = pair
        mapped = (taxon_map[a], taxon_map[b])
        return mapped if mapped[0] <= mapped[1] else (mapped[1], mapped[0])

    return PatternCountSummary(
        canonical_counts=dict(summary.canonical_counts),
        raw_counts=dict(summary.raw_counts),
        total_sites_used=summary.total_sites_used,
        skipped_sites=summary.skipped_sites,
        pairwise_valid_sites={remap_pair(pair): value for pair, value in summary.pairwise_valid_sites.items()},
        pairwise_mismatches={remap_pair(pair): value for pair, value in summary.pairwise_mismatches.items()},
    )


def _remap_clade(clade: FrozenSet[str], taxon_map: Mapping[str, str]) -> FrozenSet[str]:
    return frozenset(taxon_map[name] for name in clade)


def _iter_branch_witnesses(node: TopologyNode, all_taxa: FrozenSet[str]) -> Iterator[WitnessSpec]:
    """
    Generate edge-preserving witness quartets for branch parent(node) -> node.

    The induced rooted quartet must retain the target edge itself, not merely a
    longer path containing that edge. For a leaf child, the quartet must contain
    that leaf and at least one leaf from the sibling side. For an internal child,
    the quartet must contain both child sides and at least one sibling-side leaf.
    """
    if node.parent is None:
        return
    sib = sibling(node)
    if sib is None:
        return
    sibling_clade = sib.clade
    ordered_taxa = sorted(all_taxa)

    if node.is_leaf:
        taxon = next(iter(node.clade))
        other_taxa = [name for name in ordered_taxa if name != taxon]
        for trio in itertools.combinations(other_taxa, 3):
            quartet_set = frozenset((taxon, *trio))
            if quartet_set.isdisjoint(sibling_clade):
                continue
            yield WitnessSpec(quartet_taxa=tuple(sorted(quartet_set)), target_clade=node.clade)
        return

    left, right = node.children
    for quartet in itertools.combinations(ordered_taxa, 4):
        quartet_set = frozenset(quartet)
        if quartet_set.isdisjoint(sibling_clade):
            continue
        if quartet_set.isdisjoint(left.clade):
            continue
        if quartet_set.isdisjoint(right.clade):
            continue
        yield WitnessSpec(quartet_taxa=tuple(quartet), target_clade=quartet_set & node.clade)


def _reservoir_sample_witnesses(
    iterator: Iterable[WitnessSpec],
    limit: Optional[int],
    rng: random.Random,
) -> Tuple[List[WitnessSpec], int]:
    if limit is None or limit <= 0:
        materialized = list(iterator)
        return materialized, len(materialized)
    sample: List[WitnessSpec] = []
    total = 0
    for total, item in enumerate(iterator, start=1):
        if len(sample) < limit:
            sample.append(item)
            continue
        idx = rng.randrange(total)
        if idx < limit:
            sample[idx] = item
    return sample, total


def build_witness_map(
    root: TopologyNode,
    max_witness_per_branch: Optional[int],
    seed: int,
) -> Tuple[Dict[FrozenSet[str], List[WitnessSpec]], Dict[FrozenSet[str], int]]:
    rng = random.Random(seed)
    all_taxa = root.clade
    witness_map: Dict[FrozenSet[str], List[WitnessSpec]] = {}
    total_candidates: Dict[FrozenSet[str], int] = {}
    for node in iter_nodes_postorder(root):
        if node.parent is None:
            continue
        sampled, total = _reservoir_sample_witnesses(
            _iter_branch_witnesses(node, all_taxa),
            limit=max_witness_per_branch,
            rng=rng,
        )
        if not sampled:
            raise ValueError(f"Could not generate any witness quartets for branch {clade_label(node.clade)}.")
        witness_map[node.clade] = sampled
        total_candidates[node.clade] = total
    return witness_map, total_candidates


class CachedQuartetFitter:
    """Fit each unique induced quartet once and cache the result."""

    def __init__(
        self,
        full_tree: TopologyNode,
        alignment_cache: AlignmentCache,
        max_iter: int,
        restarts: int,
    ) -> None:
        self.full_tree = full_tree
        self.alignment_cache = alignment_cache
        self.max_iter = max_iter
        self.restarts = restarts
        self._cache: Dict[Tuple[str, str, str, str], QuartetFitResult] = {}
        self._failures: Dict[Tuple[str, str, str, str], str] = {}
        self._engine_cache: Dict[str, QuartetSitePatternProbabilityEngine] = {}

    @property
    def cache(self) -> Mapping[Tuple[str, str, str, str], QuartetFitResult]:
        return self._cache

    @property
    def failures(self) -> Mapping[Tuple[str, str, str, str], str]:
        return self._failures

    def fit_quartet(self, quartet_taxa: Sequence[str]) -> QuartetFitResult:
        key = tuple(sorted(quartet_taxa))
        if key in self._cache:
            return self._cache[key]
        if key in self._failures:
            raise RuntimeError(self._failures[key])

        try:
            induced = induce_rooted_quartet(self.full_tree, key)
            quartet_taxa_order = leaf_names_left_to_right(induced)
            taxon_to_generic = {taxon: generic for taxon, generic in zip(quartet_taxa_order, GENERIC_QUARTET_TAXA)}
            generic_to_taxon = {generic: taxon for taxon, generic in taxon_to_generic.items()}
            generic_induced = _relabel_quartet_topology(induced, taxon_to_generic)
            quartet_topology_newick = format_topology_newick(generic_induced) + ";"

            summary = self.alignment_cache.summarize_quartet(quartet_taxa_order)
            generic_summary = _relabel_pattern_summary(summary, taxon_to_generic)

            probability_engine = self._engine_cache.get(quartet_topology_newick)
            if probability_engine is None:
                parsed = parse_quartet_topology(quartet_topology_newick)
                all_clades = [node.clade for node in iter_nodes_postorder(parsed.root)]
                ne_override = {clade: THETA_SYMBOL / 4 for clade in all_clades}
                mu_override = {clade: 1.0 for clade in all_clades}
                probability_engine = QuartetSitePatternProbabilityEngine(
                    parsed,
                    model_name="JC69",
                    ne_override_by_clade=ne_override,
                    mu_override_by_clade=mu_override,
                    external_symbols=(THETA_SYMBOL,),
                )
                self._engine_cache[quartet_topology_newick] = probability_engine

            estimator = QuartetThetaEstimator(
                probability_engine.species_tree,
                generic_summary,
                model_name="JC69",
                probability_engine=probability_engine,
            )
            result = estimator.fit(max_iter=self.max_iter, restarts=self.restarts)
            fit = QuartetFitResult(
                quartet_taxa=tuple(quartet_taxa_order),
                quartet_topology=format_topology_newick(induced) + ";",
                estimated_theta=result.estimated_theta,
                branch_coalescent={
                    _remap_clade(_parse_label_to_clade(label), generic_to_taxon): value
                    for label, value in result.branch_lengths_coalescent.items()
                },
                branch_substitution={
                    _remap_clade(_parse_label_to_clade(label), generic_to_taxon): value
                    for label, value in result.branch_lengths_substitution.items()
                },
                log_likelihood=result.best_log_likelihood,
                total_sites_used=result.total_sites_used,
            )
            self._cache[key] = fit
            return fit
        except Exception as exc:
            self._failures[key] = str(exc)
            raise


def _collect_unique_quartets(witness_map: Mapping[FrozenSet[str], Sequence[WitnessSpec]]) -> List[Tuple[str, str, str, str]]:
    unique = {tuple(sorted(witness.quartet_taxa)) for witnesses in witness_map.values() for witness in witnesses}
    return sorted(unique, key=_quartet_sort_key)


def _default_output_dir(species_tree_arg: str) -> Path:
    candidate = Path(species_tree_arg)
    if candidate.exists():
        stem = candidate.stem or DEFAULT_OUTPUT_DIR_NAME
        return candidate.resolve().parent / f"{stem}_estimate"
    return Path.cwd() / DEFAULT_OUTPUT_DIR_NAME


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _write_result_files(result: EstimatedNtaxaResult, output_dir: Path, prefix: str = "estimated") -> OutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_prefix = prefix.strip() if prefix.strip() else "estimated"
    substitution_tree = output_dir / f"{clean_prefix}_tree_su.newick"
    coalescent_tree = output_dir / f"{clean_prefix}_tree_cu.newick"
    theta_summary = output_dir / f"{clean_prefix}_theta.txt"

    substitution_tree.write_text(result.substitution_newick + "\n", encoding="utf-8")
    coalescent_tree.write_text(result.coalescent_newick + "\n", encoding="utf-8")
    summary_lines = [
        "Global-theta JC69 n-taxon quartet branch stitching",
        f"Topology: {result.topology_newick}",
        "Substitution model: JC69",
        f"Estimated theta: {result.estimated_theta:.10f}",
        "Nucleotide frequencies: "
        f"A={result.nucleotide_frequencies['A']:.6f}, "
        f"C={result.nucleotide_frequencies['C']:.6f}, "
        f"G={result.nucleotide_frequencies['G']:.6f}, "
        f"T={result.nucleotide_frequencies['T']:.6f}",
        f"Input alignment files: {result.num_input_alignments}",
        f"Total alignment columns across loci: {result.total_alignment_columns}",
        f"Non-root branches: {result.non_root_branch_count}",
        f"Unique quartets attempted: {result.unique_quartets_attempted}",
        f"Unique quartets succeeded: {result.unique_quartets_succeeded}",
        f"Unique quartets failed: {result.unique_quartets_failed}",
        f"Elapsed time: {_format_seconds(result.elapsed_seconds)}",
    ]
    theta_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return OutputPaths(output_dir, substitution_tree, coalescent_tree, theta_summary)


def _print_startup_summary(species_tree: TopologyNode, alignment_cache: AlignmentCache) -> None:
    freqs = alignment_cache.empirical_nucleotide_frequencies()
    _emit(f"Topology: {format_topology_newick(species_tree)};")
    _emit("Substitution model: JC69")
    _emit(f"Total sequence length: {sum(alignment.length for alignment in alignment_cache.alignments)}")
    _emit(
        "Nucleotide frequencies from alignment: "
        f"A={freqs['A']:.4f} C={freqs['C']:.4f} G={freqs['G']:.4f} T={freqs['T']:.4f}"
    )
    _emit()


def estimate_branch_lengths(
    species_tree: TopologyNode,
    alignment_paths: Sequence[str],
    max_witness_per_branch: Optional[int] = 8,
    aggregate: str = "median",
    max_iter: int = 20,
    restarts: int = 1,
    seed: int = 1,
) -> EstimatedNtaxaResult:
    """Estimate SU/CU branch lengths and one global theta from alignments."""
    alignment_cache = AlignmentCache.from_paths(alignment_paths, model_name="JC69")
    witness_map, _candidate_counts = build_witness_map(
        species_tree,
        max_witness_per_branch=max_witness_per_branch,
        seed=seed,
    )
    unique_quartets = _collect_unique_quartets(witness_map)
    return _estimate_branch_lengths_prepared(
        species_tree=species_tree,
        alignment_cache=alignment_cache,
        witness_map=witness_map,
        unique_quartets=unique_quartets,
        aggregate=aggregate,
        max_iter=max_iter,
        restarts=restarts,
    )


def _estimate_branch_lengths_prepared(
    species_tree: TopologyNode,
    alignment_cache: AlignmentCache,
    witness_map: Mapping[FrozenSet[str], Sequence[WitnessSpec]],
    unique_quartets: Sequence[Tuple[str, str, str, str]],
    aggregate: str,
    max_iter: int,
    restarts: int,
) -> EstimatedNtaxaResult:
    start_time = time.perf_counter()
    fitter = CachedQuartetFitter(
        full_tree=species_tree,
        alignment_cache=alignment_cache,
        max_iter=max_iter,
        restarts=restarts,
    )

    for quartet_taxa in tqdm(unique_quartets, desc="Quartet fitting", unit="quartet"):
        try:
            fitter.fit_quartet(quartet_taxa)
        except Exception:
            pass

    branch_summary: Dict[FrozenSet[str], Tuple[List[float], int, List[Tuple[str, str, str, str]]]] = {}
    for clade, witnesses in witness_map.items():
        su_values: List[float] = []
        successful_quartets: List[Tuple[str, str, str, str]] = []
        failed = 0
        for witness in witnesses:
            key = tuple(sorted(witness.quartet_taxa))
            if key in fitter.failures:
                failed += 1
                continue
            fit = fitter.cache[key]
            if witness.target_clade not in fit.branch_substitution:
                failed += 1
                continue
            su_values.append(fit.branch_substitution[witness.target_clade])
            successful_quartets.append(fit.quartet_taxa)
        if not su_values:
            raise RuntimeError(f"All witness quartets failed for branch {clade_label(clade)}.")
        branch_summary[clade] = (su_values, failed, successful_quartets)

    if not fitter.cache:
        raise RuntimeError("No quartet fits succeeded; could not estimate a global theta.")
    estimated_theta = _aggregate([fit.estimated_theta for fit in fitter.cache.values()], aggregate)

    branch_estimates: Dict[FrozenSet[str], BranchAggregate] = {}
    substitution: Dict[FrozenSet[str], float] = {}
    coalescent: Dict[FrozenSet[str], float] = {}
    for clade, witnesses in witness_map.items():
        su_values, failed, successful_quartets = branch_summary[clade]
        su_value = _aggregate(su_values, aggregate)
        cu_value = 2.0 * su_value / estimated_theta
        substitution[clade] = su_value
        coalescent[clade] = cu_value
        branch_estimates[clade] = BranchAggregate(
            clade=clade,
            coalescent=cu_value,
            substitution=su_value,
            attempted_witnesses=len(witnesses),
            successful_witnesses=len(successful_quartets),
            failed_witnesses=failed,
            quartet_taxa=tuple(sorted(set(successful_quartets))),
        )

    elapsed_seconds = time.perf_counter() - start_time
    return EstimatedNtaxaResult(
        topology_newick=format_topology_newick(species_tree) + ";",
        estimated_theta=estimated_theta,
        nucleotide_frequencies=alignment_cache.empirical_nucleotide_frequencies(),
        coalescent_newick=format_newick_with_branch_lengths(species_tree, coalescent) + ";",
        substitution_newick=format_newick_with_branch_lengths(species_tree, substitution) + ";",
        branch_estimates=branch_estimates,
        unique_quartets_attempted=len(unique_quartets),
        unique_quartets_succeeded=len(fitter.cache),
        unique_quartets_failed=len(fitter.failures),
        num_input_alignments=len(alignment_cache.alignments),
        total_alignment_columns=sum(alignment.length for alignment in alignment_cache.alignments),
        non_root_branch_count=len(witness_map),
        elapsed_seconds=elapsed_seconds,
    )


def _print_result(result: EstimatedNtaxaResult) -> None:
    _emit()
    _emit(f"Estimated theta = {result.estimated_theta:.10f}")
    _emit("Estimated tree in substitution units:")
    _emit(result.substitution_newick)
    _emit()
    _emit("Estimated tree in coalescent units:")
    _emit(result.coalescent_newick)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate branch lengths on a fixed rooted n-taxon species-tree topology "
            "by stitching rooted quartet fits under JC69 with one global theta = 4 Ne mu."
        )
    )
    parser.add_argument(
        "--species-tree",
        required=True,
        help="Rooted binary Newick topology or a path to a file containing it. Branch lengths and comments are ignored.",
    )
    parser.add_argument(
        "--alignment",
        action="append",
        required=True,
        help="FASTA alignment file. Use this option multiple times for multiple loci.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for output files. If omitted, a deterministic sibling directory is created.",
    )
    parser.add_argument("--prefix", type=str, default="estimated", help="Prefix used for output filenames.")
    parser.add_argument(
        "--max-witness-per-branch",
        type=int,
        default=8,
        help="Maximum number of witness quartets retained per branch. Use 0 or negative to keep all.",
    )
    parser.add_argument(
        "--aggregate",
        choices=("median", "mean"),
        default="median",
        help="How to aggregate branch-specific SU estimates and quartet theta estimates.",
    )
    parser.add_argument("--max-iter", type=int, default=20, help="Maximum Nelder-Mead iterations for each quartet fit.")
    parser.add_argument("--restarts", type=int, default=1, help="Number of restarts for each quartet fit.")
    parser.add_argument("--seed", type=int, default=1, help="Seed used for witness quartet sampling.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    tree = load_topology_arg(args.species_tree)
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir(args.species_tree)
    alignment_cache = AlignmentCache.from_paths(args.alignment, model_name="JC69")
    _print_startup_summary(tree, alignment_cache)

    witness_map, _ = build_witness_map(tree, args.max_witness_per_branch, args.seed)
    unique_quartets = _collect_unique_quartets(witness_map)
    result = _estimate_branch_lengths_prepared(
        species_tree=tree,
        alignment_cache=alignment_cache,
        witness_map=witness_map,
        unique_quartets=unique_quartets,
        aggregate=args.aggregate,
        max_iter=args.max_iter,
        restarts=args.restarts,
    )
    _write_result_files(result, output_dir, prefix=args.prefix)
    _print_result(result)


if __name__ == "__main__":
    main()
