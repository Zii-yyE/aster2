"""
Rooted-quartet optimizer for the global-theta JC69 setting.

For one rooted quartet, this module optimizes:
- the three internal node ages in substitution units (SU),
- one global theta = 4 Ne mu.

Given theta, quartet coalescent-unit branch lengths are derived by
``CU = 2 * SU / theta`` under the diploid convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import sympy as sp

from .annotated_species_tree import (
    AnnotatedSpeciesNode,
    ParsedQuartetSpeciesTree,
    format_plain_newick_with_branch_lengths,
    iter_nodes_postorder,
    parse_uniform_quartet_species_tree,
)
from .quartet_site_pattern_probability import QuartetSitePatternProbabilityEngine
from .site_pattern_counts import PatternCountSummary


MIN_PROBABILITY = 1e-300
THETA_SYMBOL = sp.Symbol("theta", positive=True)


@dataclass(frozen=True)
class EstimatedQuartetThetaResult:
    best_log_likelihood: float
    optimizer_iterations: int
    estimated_theta: float
    internal_ages_substitution: Mapping[str, float]
    branch_lengths_substitution: Mapping[str, float]
    branch_lengths_coalescent: Mapping[str, float]
    substitution_newick: str
    coalescent_newick: str
    canonical_pattern_probabilities: Mapping[str, float]
    normalization_sum: float
    raw_pattern_space_size: int
    total_sites_used: int
    skipped_sites: int


def _clade_label(clade: FrozenSet[str]) -> str:
    return "{" + ",".join(sorted(clade)) + "}"


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pairwise_jc69_distance(mismatch_fraction: float) -> Optional[float]:
    if mismatch_fraction < 0.0 or mismatch_fraction >= 0.75:
        return None
    remaining = 1.0 - (4.0 * mismatch_fraction / 3.0)
    if remaining <= 0.0:
        return None
    return -0.75 * math.log(remaining)


def _simple_vector_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _simple_vector_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def _simple_vector_scale(a: Sequence[float], scalar: float) -> List[float]:
    return [scalar * x for x in a]


def _simplex_centroid(points: Sequence[Sequence[float]]) -> List[float]:
    n = len(points[0])
    out = [0.0] * n
    for point in points:
        for i, value in enumerate(point):
            out[i] += value
    return [value / float(len(points)) for value in out]


def nelder_mead_optimize(
    objective,
    x0: Sequence[float],
    step: float = 0.5,
    max_iter: int = 200,
    x_tol: float = 1e-4,
    f_tol: float = 1e-6,
) -> Tuple[List[float], float, int]:
    """Small dependency-free Nelder-Mead optimizer for quartet fits."""
    n = len(x0)
    simplex: List[List[float]] = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] += step
        simplex.append(point)

    values = [objective(point) for point in simplex]
    iterations = 0

    while iterations < max_iter:
        order = sorted(range(len(simplex)), key=lambda idx: values[idx])
        simplex = [simplex[idx] for idx in order]
        values = [values[idx] for idx in order]
        best = simplex[0]
        worst = simplex[-1]
        second_worst_value = values[-2]

        max_vertex_span = max(
            max(abs(a - b) for a, b in zip(vertex, best)) for vertex in simplex[1:]
        )
        if max_vertex_span <= x_tol and max(abs(val - values[0]) for val in values[1:]) <= f_tol:
            break

        centroid = _simplex_centroid(simplex[:-1])
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        reflected = _simple_vector_add(
            centroid,
            _simple_vector_scale(_simple_vector_sub(centroid, worst), alpha),
        )
        reflected_value = objective(reflected)

        if values[0] <= reflected_value < second_worst_value:
            simplex[-1] = reflected
            values[-1] = reflected_value
            iterations += 1
            continue

        if reflected_value < values[0]:
            expanded = _simple_vector_add(
                centroid,
                _simple_vector_scale(_simple_vector_sub(reflected, centroid), gamma),
            )
            expanded_value = objective(expanded)
            if expanded_value < reflected_value:
                simplex[-1] = expanded
                values[-1] = expanded_value
            else:
                simplex[-1] = reflected
                values[-1] = reflected_value
            iterations += 1
            continue

        contracted = _simple_vector_add(
            centroid,
            _simple_vector_scale(_simple_vector_sub(worst, centroid), rho),
        )
        contracted_value = objective(contracted)
        if contracted_value < values[-1]:
            simplex[-1] = contracted
            values[-1] = contracted_value
            iterations += 1
            continue

        best = simplex[0]
        new_simplex = [best]
        new_values = [values[0]]
        for vertex in simplex[1:]:
            shrunk = _simple_vector_add(
                best,
                _simple_vector_scale(_simple_vector_sub(vertex, best), sigma),
            )
            new_simplex.append(shrunk)
            new_values.append(objective(shrunk))
        simplex = new_simplex
        values = new_values
        iterations += 1

    order = sorted(range(len(simplex)), key=lambda idx: values[idx])
    simplex = [simplex[idx] for idx in order]
    values = [values[idx] for idx in order]
    return simplex[0], values[0], iterations


class QuartetThetaEstimator:
    def __init__(
        self,
        species_tree: ParsedQuartetSpeciesTree,
        pattern_summary: PatternCountSummary,
        model_name: str = "JC69",
        model_params: Optional[Mapping[str, float]] = None,
        probability_engine: Optional[QuartetSitePatternProbabilityEngine] = None,
    ) -> None:
        if model_name.strip().upper() != "JC69":
            raise ValueError("This integration package supports only JC69.")
        self.species_tree = species_tree
        self.pattern_summary = pattern_summary
        self.model_params = dict(model_params or {})
        self.internal_nodes = tuple(node for node in self._iter_internal_nodes_postorder(species_tree.root))

        all_clades = [node.clade for node in iter_nodes_postorder(species_tree.root)]
        ne_override = {clade: THETA_SYMBOL / 4 for clade in all_clades}
        mu_override = {clade: 1.0 for clade in all_clades}
        self.probability_engine = (
            probability_engine
            if probability_engine is not None
            else QuartetSitePatternProbabilityEngine(
                species_tree,
                model_name="JC69",
                ne_override_by_clade=ne_override,
                mu_override_by_clade=mu_override,
                external_symbols=(THETA_SYMBOL,),
            )
        )

    @property
    def parameter_dimension(self) -> int:
        return len(self.internal_nodes) + 1

    @staticmethod
    def _iter_internal_nodes_postorder(node: AnnotatedSpeciesNode) -> Iterable[AnnotatedSpeciesNode]:
        for child in node.children:
            yield from QuartetThetaEstimator._iter_internal_nodes_postorder(child)
        if not node.is_leaf:
            yield node

    def su_age_map_from_log_deltas(self, log_deltas: Sequence[float]) -> Dict[FrozenSet[str], float]:
        if len(log_deltas) != len(self.internal_nodes):
            raise ValueError(f"Expected {len(self.internal_nodes)} log-delta parameters, got {len(log_deltas)}.")
        age_by_clade: Dict[FrozenSet[str], float] = {}
        for node, log_delta in zip(self.internal_nodes, log_deltas):
            delta = math.exp(log_delta)
            child_ages = [0.0 if child.is_leaf else age_by_clade[child.clade] for child in node.children]
            age_by_clade[node.clade] = max(child_ages) + delta
        return age_by_clade

    def theta_from_log_value(self, log_theta: float) -> float:
        return math.exp(log_theta)

    def _branch_length_maps_from_parameters(
        self,
        age_by_clade: Mapping[FrozenSet[str], float],
        theta: float,
    ) -> Tuple[Dict[FrozenSet[str], float], Dict[FrozenSet[str], float]]:
        substitution: Dict[FrozenSet[str], float] = {}
        coalescent: Dict[FrozenSet[str], float] = {}
        for node in iter_nodes_postorder(self.species_tree.root):
            if node.parent is None:
                continue
            child_age = 0.0 if node.is_leaf else age_by_clade[node.clade]
            parent_age = age_by_clade[node.parent.clade]
            su = parent_age - child_age
            if su <= 0.0:
                raise ValueError(f"Encountered a non-positive SU branch length on species branch {_clade_label(node.clade)}.")
            substitution[node.clade] = su
            coalescent[node.clade] = 2.0 * su / theta
        return substitution, coalescent

    def _unpack_parameters(self, params: Sequence[float]) -> Tuple[Dict[FrozenSet[str], float], float]:
        if len(params) != self.parameter_dimension:
            raise ValueError("Quartet-theta parameter vector has the wrong length.")
        age_params = params[:-1]
        log_theta = params[-1]
        age_by_clade = self.su_age_map_from_log_deltas(age_params)
        theta = self.theta_from_log_value(log_theta)
        return age_by_clade, theta

    def negative_log_likelihood(self, params: Sequence[float]) -> float:
        try:
            age_by_clade, theta = self._unpack_parameters(params)
        except Exception:
            return float("inf")

        total = 0.0
        param_values = {THETA_SYMBOL: theta}
        for canonical_pattern, count in self.pattern_summary.canonical_counts.items():
            try:
                probability = self.probability_engine.canonical_probability(
                    canonical_pattern,
                    age_by_clade,
                    parameter_values=param_values,
                )
            except Exception:
                return float("inf")
            if not math.isfinite(probability) or probability <= 0.0:
                return float("inf")
            total += float(count) * math.log(max(probability, MIN_PROBABILITY))
        return -total

    def _heuristic_initial_log_params(self) -> Tuple[float, ...]:
        pairwise_distances: Dict[Tuple[str, str], float] = {}
        for pair, valid_sites in self.pattern_summary.pairwise_valid_sites.items():
            if valid_sites <= 0.0:
                continue
            mismatches = self.pattern_summary.pairwise_mismatches[pair]
            corrected = _pairwise_jc69_distance(mismatches / valid_sites)
            if corrected is not None:
                pairwise_distances[pair] = corrected

        representative_cache: Dict[FrozenSet[str], str] = {}

        def representative_leaf(node: AnnotatedSpeciesNode) -> str:
            if node.clade in representative_cache:
                return representative_cache[node.clade]
            if node.is_leaf:
                assert node.label is not None
                representative_cache[node.clade] = node.label
                return node.label
            rep = representative_leaf(node.children[0])
            representative_cache[node.clade] = rep
            return rep

        all_distances = list(pairwise_distances.values())
        fallback_scale = max(sum(all_distances) / len(all_distances), 0.05) if all_distances else 0.1

        age_guesses: Dict[FrozenSet[str], float] = {}
        for node in self.internal_nodes:
            left_rep = representative_leaf(node.children[0])
            right_rep = representative_leaf(node.children[1])
            pair = _pair_key(left_rep, right_rep)
            corrected = pairwise_distances.get(pair, fallback_scale)
            approx_age = max(corrected / 2.0, 1e-6)
            child_ages = [0.0 if child.is_leaf else age_guesses[child.clade] for child in node.children]
            age_guesses[node.clade] = max(approx_age, max(child_ages) + 1e-6)

        log_deltas: List[float] = []
        for node in self.internal_nodes:
            child_ages = [0.0 if child.is_leaf else age_guesses[child.clade] for child in node.children]
            delta = max(age_guesses[node.clade] - max(child_ages), 1e-6)
            log_deltas.append(math.log(delta))

        theta_guess = max(fallback_scale, 1e-3)
        return tuple(log_deltas + [math.log(theta_guess)])

    def fit(self, max_iter: int = 200, restarts: int = 5) -> EstimatedQuartetThetaResult:
        base = self._heuristic_initial_log_params()
        age_dim = len(self.internal_nodes)
        theta_scales = [0.01, 0.1, 1.0, 10.0, 100.0]
        if restarts <= 0:
            restarts = 1
        while len(theta_scales) < restarts:
            theta_scales.append(theta_scales[-1] * 10.0)
        theta_scales = theta_scales[:restarts]

        best_point: Optional[List[float]] = None
        best_value = float("inf")
        total_iterations = 0
        for theta_scale in theta_scales:
            start = list(base[:age_dim])
            start.append(base[-1] + math.log(theta_scale))
            point, value, iterations = nelder_mead_optimize(
                self.negative_log_likelihood,
                start,
                step=0.5,
                max_iter=max_iter,
            )
            total_iterations += iterations
            if value < best_value:
                best_value = value
                best_point = point

        assert best_point is not None
        age_by_clade, theta = self._unpack_parameters(best_point)
        param_values = {THETA_SYMBOL: theta}
        substitution, coalescent = self._branch_length_maps_from_parameters(age_by_clade, theta)
        canonical_probs = self.probability_engine.canonical_probability_map(age_by_clade, parameter_values=param_values)
        normalization = self.probability_engine.normalization_sum(age_by_clade, parameter_values=param_values)

        internal_ages = {_clade_label(node.clade): age_by_clade[node.clade] for node in self.internal_nodes}
        branch_substitution = {_clade_label(clade): value for clade, value in substitution.items()}
        branch_coalescent = {_clade_label(clade): value for clade, value in coalescent.items()}

        return EstimatedQuartetThetaResult(
            best_log_likelihood=-best_value,
            optimizer_iterations=total_iterations,
            estimated_theta=theta,
            internal_ages_substitution=internal_ages,
            branch_lengths_substitution=branch_substitution,
            branch_lengths_coalescent=branch_coalescent,
            substitution_newick=format_plain_newick_with_branch_lengths(self.species_tree.root, substitution),
            coalescent_newick=format_plain_newick_with_branch_lengths(self.species_tree.root, coalescent),
            canonical_pattern_probabilities=canonical_probs,
            normalization_sum=normalization,
            raw_pattern_space_size=self.probability_engine.model.state_count ** 4,
            total_sites_used=self.pattern_summary.total_sites_used,
            skipped_sites=self.pattern_summary.skipped_sites,
        )


def parse_quartet_topology(raw: str) -> ParsedQuartetSpeciesTree:
    return parse_uniform_quartet_species_tree(raw, ne=1.0, mu=1.0)
