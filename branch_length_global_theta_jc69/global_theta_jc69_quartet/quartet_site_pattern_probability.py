"""
Quartet MSC + substitution likelihood for site-pattern probabilities.

This module owns the probability-evaluation path:
- rooted quartet gene-tree preparation,
- MSC case integration,
- symbolic-to-numeric compilation for raw quartet site patterns under the
  JC69 substitution model.

It deliberately does not perform optimization. The estimator layer consumes this
engine and handles parameter search over internal species-tree ages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import os
import math
from pathlib import Path
import pickle
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import mpmath as mp
import sympy as sp
from sympy.core.relational import GreaterThan, LessThan, Relational, StrictGreaterThan, StrictLessThan

from .annotated_species_tree import (
    AnnotatedSpeciesNode,
    ParsedQuartetSpeciesTree,
    branch_parameter_maps,
    iter_internal_nodes_postorder,
)
from .msc_density import (
    GeneEvent,
    MSCCase,
    SpeciesBranch,
    enumerate_quartet_msc_cases,
)
from .rooted_gene_trees import (
    enumerate_rooted_labeled_quartets,
    rooted_tree_to_symbolic_newick,
)
from .site_pattern_counts import canonicalize_pattern, pattern_class_representatives
from .symbolic_substitution_models import SymbolicSubstitutionModel, build_symbolic_model


Tree = Union[str, Tuple["Tree", "Tree"]]
G_SYMBOLS = tuple(sp.Symbol(f"g{i}", positive=True) for i in range(1, 4))


@dataclass(frozen=True)
class PreparedGeneTree:
    rooted_tree: Tree
    gene_newick: str
    clade_to_event_id: Mapping[FrozenSet[str], str]
    branches: Tuple[SpeciesBranch, ...]
    events: Tuple[GeneEvent, ...]
    age_constraints: Tuple[sp.Expr, ...]
    cases: Tuple[MSCCase, ...]


def _tree_leafset(node: Tree) -> FrozenSet[str]:
    if isinstance(node, str):
        return frozenset([node])
    left, right = node
    return frozenset(set(_tree_leafset(left)) | set(_tree_leafset(right)))


def _assign_gene_age_symbols(tree: Tree) -> Dict[int, sp.Symbol]:
    counter = [0]
    ages: Dict[int, sp.Symbol] = {}

    def walk(node: Tree) -> None:
        if isinstance(node, str):
            return
        left, right = node
        walk(left)
        walk(right)
        counter[0] += 1
        if counter[0] > 3:
            raise RuntimeError("A rooted quartet gene tree must have exactly 3 internal nodes.")
        ages[id(node)] = G_SYMBOLS[counter[0] - 1]

    walk(tree)
    if counter[0] != 3:
        raise RuntimeError("A rooted quartet gene tree must have exactly 3 internal nodes.")
    return ages


def _gene_age_order_constraints(tree: Tree) -> Tuple[sp.Expr, ...]:
    ages = _assign_gene_age_symbols(tree)
    out: List[sp.Expr] = []

    def walk(node: Tree) -> None:
        if isinstance(node, str):
            return
        node_age = ages[id(node)]
        out.append(StrictGreaterThan(node_age, sp.Integer(0)))
        left, right = node
        if not isinstance(left, str):
            out.append(StrictGreaterThan(node_age, ages[id(left)]))
        if not isinstance(right, str):
            out.append(StrictGreaterThan(node_age, ages[id(right)]))
        walk(left)
        walk(right)

    walk(tree)
    return tuple(out)


def _canonicalize_expr(expr: sp.Expr, symbol_by_name: Mapping[str, sp.Symbol]) -> sp.Expr:
    mapping = {}
    for sym in expr.free_symbols:
        key = str(sym)
        if key in symbol_by_name:
            mapping[sym] = symbol_by_name[key]
    if not mapping:
        return expr
    return expr.xreplace(mapping)


def _condition_dnf_pieces(cond: sp.Expr) -> List[sp.Expr]:
    if cond is sp.false or cond == sp.false:
        return []
    if cond is sp.true or cond == sp.true:
        return [sp.true]

    c = cond
    if c.has(sp.Or):
        c = sp.simplify_logic(c, form="dnf")

    if c is sp.false or c == sp.false:
        return []
    if c is sp.true or c == sp.true:
        return [sp.true]
    if isinstance(c, sp.Or):
        out = []
        for piece in c.args:
            p = sp.simplify(piece)
            if p is not sp.false and p != sp.false:
                out.append(p)
        return out
    return [sp.simplify(c)]


def _add_less_edge(graph: Dict[sp.Expr, Set[sp.Expr]], a: sp.Expr, b: sp.Expr) -> None:
    aa = sp.simplify(a)
    bb = sp.simplify(b)
    graph.setdefault(aa, set()).add(bb)
    graph.setdefault(bb, set())


def _build_less_graph(atoms: Sequence[sp.Expr]) -> Dict[sp.Expr, Set[sp.Expr]]:
    graph: Dict[sp.Expr, Set[sp.Expr]] = {}
    for atom in atoms:
        if not isinstance(atom, Relational):
            continue
        lhs = sp.simplify(atom.lhs)
        rhs = sp.simplify(atom.rhs)
        if isinstance(atom, (StrictLessThan, LessThan)):
            _add_less_edge(graph, lhs, rhs)
        elif isinstance(atom, (StrictGreaterThan, GreaterThan)):
            _add_less_edge(graph, rhs, lhs)
    return graph


def _known_leq(a: sp.Expr, b: sp.Expr, graph: Dict[sp.Expr, Set[sp.Expr]]) -> bool:
    aa = sp.simplify(a)
    bb = sp.simplify(b)
    diff = sp.simplify(aa - bb)
    if diff.is_negative or diff.is_zero:
        return True
    if diff.is_positive:
        return False
    if aa == bb:
        return True
    if aa not in graph:
        return False

    seen: Set[sp.Expr] = set()
    stack = [aa]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in graph.get(cur, set()):
            if nxt == bb:
                return True
            if nxt not in seen:
                stack.append(nxt)
    return False


def _tightest_lower(candidates: Sequence[sp.Expr], graph: Dict[sp.Expr, Set[sp.Expr]]) -> sp.Expr:
    if not candidates:
        return sp.Integer(0)
    unique: List[sp.Expr] = []
    seen = set()
    for candidate in candidates:
        value = sp.simplify(candidate)
        key = sp.srepr(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)

    survivors: List[sp.Expr] = []
    for c in unique:
        dominated = False
        for d in unique:
            if c == d:
                continue
            if _known_leq(c, d, graph):
                dominated = True
                break
        if not dominated:
            survivors.append(c)
    if len(survivors) == 1:
        return survivors[0]
    return sp.Max(*survivors)


def _tightest_upper(candidates: Sequence[sp.Expr], graph: Dict[sp.Expr, Set[sp.Expr]]) -> sp.Expr:
    if not candidates:
        return sp.oo
    unique: List[sp.Expr] = []
    seen = set()
    for candidate in candidates:
        value = sp.simplify(candidate)
        key = sp.srepr(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)

    survivors: List[sp.Expr] = []
    for c in unique:
        dominated = False
        for d in unique:
            if c == d:
                continue
            if _known_leq(d, c, graph):
                dominated = True
                break
        if not dominated:
            survivors.append(c)
    if len(survivors) == 1:
        return survivors[0]
    return sp.Min(*survivors)


def _extract_integration_bounds(
    cond: sp.Expr,
    order: Sequence[sp.Symbol],
    base_constraints: Sequence[sp.Expr],
    extra_constraints: Optional[Sequence[sp.Expr]] = None,
) -> List[Tuple[sp.Symbol, sp.Expr, sp.Expr]]:
    atoms = list(sp.And.make_args(cond)) if isinstance(cond, sp.And) else [cond]
    atoms.extend(base_constraints)
    if extra_constraints:
        atoms.extend(extra_constraints)

    less_graph = _build_less_graph(atoms)
    lower: Dict[sp.Symbol, List[sp.Expr]] = {var: [] for var in order}
    upper: Dict[sp.Symbol, List[sp.Expr]] = {var: [] for var in order}

    for atom in atoms:
        if not isinstance(atom, Relational):
            continue
        lhs = atom.lhs
        rhs = atom.rhs
        for var in order:
            if lhs == var and not rhs.has(var):
                if isinstance(atom, (StrictGreaterThan, GreaterThan)):
                    lower[var].append(rhs)
                elif isinstance(atom, (StrictLessThan, LessThan)):
                    upper[var].append(rhs)
            elif rhs == var and not lhs.has(var):
                if isinstance(atom, (StrictLessThan, LessThan)):
                    lower[var].append(lhs)
                elif isinstance(atom, (StrictGreaterThan, GreaterThan)):
                    upper[var].append(lhs)

    bounds: List[Tuple[sp.Symbol, sp.Expr, sp.Expr]] = []
    integrated: Set[sp.Symbol] = set()
    for var in order:
        ls = [expr for expr in lower[var] if expr.free_symbols.isdisjoint(integrated)]
        us = [expr for expr in upper[var] if expr.free_symbols.isdisjoint(integrated)]
        lower_bound = _tightest_lower(ls, less_graph)
        upper_bound = _tightest_upper(us, less_graph)

        remaining = set(order) - integrated - {var}
        if isinstance(upper_bound, sp.Symbol) and upper_bound in remaining:
            lower[upper_bound].append(lower_bound)
            _add_less_edge(less_graph, lower_bound, upper_bound)
        if isinstance(lower_bound, sp.Symbol) and lower_bound in remaining and upper_bound is not sp.oo:
            upper[lower_bound].append(upper_bound)
            _add_less_edge(less_graph, lower_bound, upper_bound)

        if not lower_bound.has(var):
            _add_less_edge(less_graph, lower_bound, var)
        if upper_bound is not sp.oo and not upper_bound.has(var):
            _add_less_edge(less_graph, var, upper_bound)

        bounds.append((var, sp.simplify(lower_bound), sp.simplify(upper_bound)))
        integrated.add(var)

    return bounds


def _integrate_term_exp_linear(term: sp.Expr, var: sp.Symbol, lower: sp.Expr, upper: sp.Expr) -> sp.Expr:
    def _leq_simple(a: sp.Expr, b: sp.Expr) -> bool:
        diff = sp.simplify(a - b)
        return bool(diff.is_negative or diff.is_zero)

    reduced = term
    repl: Dict[sp.Expr, sp.Expr] = {}
    for expr in list(reduced.atoms(sp.Min)):
        if len(expr.args) != 2:
            continue
        a, b = expr.args
        if a == var and not b.has(var):
            other = b
        elif b == var and not a.has(var):
            other = a
        else:
            continue
        if upper is not sp.oo and _leq_simple(upper, other):
            repl[expr] = var
        elif _leq_simple(other, lower):
            repl[expr] = other

    for expr in list(reduced.atoms(sp.Max)):
        if len(expr.args) != 2:
            continue
        a, b = expr.args
        if a == var and not b.has(var):
            other = b
        elif b == var and not a.has(var):
            other = a
        else:
            continue
        if upper is not sp.oo and _leq_simple(upper, other):
            repl[expr] = other
        elif _leq_simple(other, lower):
            repl[expr] = var

    if repl:
        reduced = sp.simplify(reduced.xreplace(repl))

    term = sp.factor_terms(sp.expand_power_exp(sp.powsimp(reduced, force=True)), var)
    if term == 0:
        return sp.Integer(0)

    coeff = sp.Integer(1)
    exponent = sp.Integer(0)
    factors = term.args if isinstance(term, sp.Mul) else (term,)
    for factor in factors:
        if factor.func == sp.exp:
            exponent += factor.args[0]
        elif isinstance(factor, sp.Pow) and factor.base == sp.E:
            exponent += factor.exp
        else:
            coeff *= factor

    if exponent == 0 and not coeff.has(var):
        return coeff * (upper - lower)

    if coeff.has(var):
        raise ValueError(
            f"Term is not of supported linear-exponential form in {var}: {term}."
        )

    slope = sp.diff(exponent, var)
    if slope.has(var):
        raise ValueError(
            f"Exponential slope still depends on {var}: exponent={exponent}, term={term}."
        )
    constant = exponent - slope * var
    prefactor = coeff * sp.exp(constant)

    if slope == 0:
        return prefactor * (upper - lower)

    def _eval_exp(bound: sp.Expr) -> sp.Expr:
        if bound is sp.oo:
            return sp.limit(sp.exp(slope * var), var, sp.oo)
        if bound is -sp.oo:
            return sp.limit(sp.exp(slope * var), var, -sp.oo)
        return sp.exp(slope * bound)

    return prefactor / slope * (_eval_exp(upper) - _eval_exp(lower))


def _integrate_exponential_sum(
    expr: sp.Expr,
    bounds: Sequence[Tuple[sp.Symbol, sp.Expr, sp.Expr]],
) -> sp.Expr:
    out = sp.expand(expr)
    for var, lower, upper in bounds:
        expanded = sp.expand(out)
        terms = expanded.args if isinstance(expanded, sp.Add) else (expanded,)
        out = sp.expand(sum(_integrate_term_exp_linear(term, var, lower, upper) for term in terms))
    return sp.expand(out)


def _branch_parameter_maps_by_bid(
    branches: Sequence[SpeciesBranch],
    ne_by_clade: Mapping[FrozenSet[str], sp.Expr | float | int],
    mu_by_clade: Mapping[FrozenSet[str], sp.Expr | float | int],
) -> Tuple[Dict[str, sp.Expr], Dict[str, sp.Expr]]:
    ne_by_bid: Dict[str, sp.Expr] = {}
    mu_by_bid: Dict[str, sp.Expr] = {}
    root_clade = max(ne_by_clade, key=len)
    for branch in branches:
        key = root_clade if branch.bid == "root_above" else branch.clade
        ne_by_bid[branch.bid] = sp.sympify(ne_by_clade[key])
        mu_by_bid[branch.bid] = sp.sympify(mu_by_clade[key])
    return ne_by_bid, mu_by_bid


def _build_case_density_expression(
    case: MSCCase,
    branch_by_id: Mapping[str, SpeciesBranch],
    event_by_id: Mapping[str, GeneEvent],
    ne_by_bid: Mapping[str, sp.Expr],
) -> sp.Expr:
    expr = sp.Integer(1)
    for bid, ordered_eids in case.branch_event_order.items():
        branch = branch_by_id[bid]
        ne = ne_by_bid[bid]
        u, _ = case.uv[bid]
        j = u
        prev_local = sp.Integer(0)

        for eid in ordered_eids:
            event = event_by_id[eid]
            local_t = sp.simplify(event.age - branch.lower)
            delta = sp.simplify(local_t - prev_local)
            total_rate = sp.Integer(j * (j - 1)) / (sp.Integer(4) * ne)
            pair_rate = sp.Integer(1) / (sp.Integer(2) * ne)
            expr *= pair_rate * sp.exp(-total_rate * delta)
            j -= 1
            prev_local = local_t

        v = j
        if branch.upper is not None:
            rem = sp.simplify(branch.length - prev_local)
            total_rate = sp.Integer(v * (v - 1)) / (sp.Integer(4) * ne)
            expr *= sp.exp(-total_rate * rem)
        elif v != 1:
            raise ValueError("MSC case left more than one lineage above the root.")
    return sp.expand(expr)


def _species_branch_chain(
    start_bid: str,
    end_bid: str,
    branch_by_id: Mapping[str, SpeciesBranch],
) -> List[str]:
    chain = [start_bid]
    current = start_bid
    visited = {current}
    while current != end_bid:
        parent_bid = branch_by_id[current].parent_bid
        if parent_bid is None:
            raise ValueError(
                f"Species-branch ancestry could not connect {start_bid} to {end_bid}."
            )
        current = parent_bid
        if current in visited:
            raise ValueError("Detected a cycle while following species-branch ancestry.")
        visited.add(current)
        chain.append(current)
    return chain


def _lineage_substitution_length(
    lower_age: sp.Expr,
    upper_age: sp.Expr,
    start_bid: str,
    end_bid: str,
    branch_by_id: Mapping[str, SpeciesBranch],
    mu_by_bid: Mapping[str, sp.Expr],
) -> sp.Expr:
    chain = _species_branch_chain(start_bid, end_bid, branch_by_id)
    if len(chain) == 1:
        return sp.simplify(mu_by_bid[start_bid] * (upper_age - lower_age))

    expr = sp.Integer(0)
    first = branch_by_id[chain[0]]
    if first.upper is None:
        raise ValueError("A lineage cannot leave the root-above branch and re-enter the species tree.")
    expr += mu_by_bid[chain[0]] * (first.upper - lower_age)

    for bid in chain[1:-1]:
        branch = branch_by_id[bid]
        if branch.length is None:
            raise ValueError("Only the terminal branch in an ancestry chain may be root_above.")
        expr += mu_by_bid[bid] * branch.length

    last = branch_by_id[chain[-1]]
    expr += mu_by_bid[chain[-1]] * (upper_age - last.lower)
    return sp.simplify(expr)


def _site_prob_given_gene_tree_case(
    rooted_tree: Tree,
    pattern_map: Mapping[str, str],
    model: SymbolicSubstitutionModel,
    case: MSCCase,
    clade_to_event_id: Mapping[FrozenSet[str], str],
    branch_by_id: Mapping[str, SpeciesBranch],
    leaf_branch_by_clade: Mapping[FrozenSet[str], str],
    mu_by_bid: Mapping[str, sp.Expr],
) -> sp.Expr:
    ages = _assign_gene_age_symbols(rooted_tree)
    clade_by_node: Dict[int, FrozenSet[str]] = {}

    def age(node: Tree) -> sp.Expr:
        if isinstance(node, str):
            return sp.Integer(0)
        return ages[id(node)]

    def clade(node: Tree) -> FrozenSet[str]:
        key = id(node) if not isinstance(node, str) else hash(node)
        if key not in clade_by_node:
            clade_by_node[key] = _tree_leafset(node)
        return clade_by_node[key]

    edge_lengths: Dict[Tuple[FrozenSet[str], FrozenSet[str]], sp.Expr] = {}

    def collect_edge_lengths(node: Tree) -> None:
        if isinstance(node, str):
            return
        parent_clade = clade(node)
        parent_age = age(node)
        parent_branch = case.assign_map[clade_to_event_id[parent_clade]]
        left, right = node
        for child in (left, right):
            child_clade = clade(child)
            child_age = age(child)
            if isinstance(child, str):
                start_bid = leaf_branch_by_clade[child_clade]
            else:
                start_bid = case.assign_map[clade_to_event_id[child_clade]]
            edge_lengths[(child_clade, parent_clade)] = _lineage_substitution_length(
                lower_age=child_age,
                upper_age=parent_age,
                start_bid=start_bid,
                end_bid=parent_branch,
                branch_by_id=branch_by_id,
                mu_by_bid=mu_by_bid,
            )
            collect_edge_lengths(child)

    collect_edge_lengths(rooted_tree)
    memo: Dict[Tuple[int, int], sp.Expr] = {}

    def subtree_likelihood(node: Tree, state_at_node: int) -> sp.Expr:
        node_key = id(node) if not isinstance(node, str) else hash((node, state_at_node))
        memo_key = (node_key, state_at_node)
        if memo_key in memo:
            return memo[memo_key]

        if isinstance(node, str):
            observed = model.state_to_index[pattern_map[node]]
            value = sp.Integer(1) if observed == state_at_node else sp.Integer(0)
            memo[memo_key] = value
            return value

        left, right = node
        left_clade = clade(left)
        right_clade = clade(right)
        parent_clade = clade(node)
        left_len = edge_lengths[(left_clade, parent_clade)]
        right_len = edge_lengths[(right_clade, parent_clade)]
        left_sum = sp.Integer(0)
        right_sum = sp.Integer(0)
        for child_state in range(model.state_count):
            left_sum += model.transition_probability(state_at_node, child_state, left_len) * subtree_likelihood(left, child_state)
            right_sum += model.transition_probability(state_at_node, child_state, right_len) * subtree_likelihood(right, child_state)
        value = sp.expand(left_sum * right_sum)
        memo[memo_key] = value
        return value

    total = sp.Integer(0)
    for root_state, pi in enumerate(model.stationary_distribution):
        total += pi * subtree_likelihood(rooted_tree, root_state)
    return sp.expand(total)


class QuartetSitePatternProbabilityEngine:
    """
    Prepared probability engine for a fixed annotated quartet species tree.

    The engine caches raw site-pattern expressions, so repeated numeric
    evaluations at different internal ages only pay the symbolic setup cost
    once per raw pattern.
    """

    def __init__(
        self,
        species_tree: ParsedQuartetSpeciesTree,
        model_name: str = "JC69",
        model_params: Optional[Mapping[str, sp.Expr | float | int]] = None,
        ne_override_by_clade: Optional[Mapping[FrozenSet[str], sp.Expr | float | int]] = None,
        mu_override_by_clade: Optional[Mapping[FrozenSet[str], sp.Expr | float | int]] = None,
        external_symbols: Sequence[sp.Symbol] = (),
    ) -> None:
        self.species_tree = species_tree
        params = dict(model_params or {})
        params["mu"] = 1
        self.model = build_symbolic_model(model_name, params)
        self.pattern_representatives = pattern_class_representatives(self.model.name)

        self.internal_nodes = tuple(iter_internal_nodes_postorder(species_tree.root))
        self.age_symbols = tuple(
            sp.Symbol(f"s{i}", positive=True) for i in range(1, len(self.internal_nodes) + 1)
        )
        self.age_symbol_by_clade = {
            node.clade: symbol for node, symbol in zip(self.internal_nodes, self.age_symbols)
        }
        self.symbol_by_name = {str(symbol): symbol for symbol in self.age_symbols}
        for symbol in G_SYMBOLS:
            self.symbol_by_name[str(symbol)] = symbol
        self.external_symbols = tuple(external_symbols)
        for symbol in self.external_symbols:
            self.symbol_by_name[str(symbol)] = symbol

        self.species_age_constraints = self._species_age_constraints()
        self.symbolic_species_newick = self._symbolic_species_newick()
        base_ne_by_clade, base_mu_by_clade = branch_parameter_maps(species_tree.root)
        self.ne_by_clade = {
            clade: sp.sympify(
                ne_override_by_clade[clade]
                if ne_override_by_clade is not None and clade in ne_override_by_clade
                else value
            )
            for clade, value in base_ne_by_clade.items()
        }
        self.mu_by_clade = {
            clade: sp.sympify(
                mu_override_by_clade[clade]
                if mu_override_by_clade is not None and clade in mu_override_by_clade
                else value
            )
            for clade, value in base_mu_by_clade.items()
        }
        self.prepared_gene_trees = self._prepare_gene_trees()

        self.pattern_expression_cache: Dict[str, sp.Expr] = {}
        self.pattern_function_cache_math: Dict[str, object] = {}
        self.pattern_function_cache_mpmath: Dict[str, object] = {}
        # Some symbolic pattern expressions contain large cancelling exponentials.
        # Once the fast backend overflows for a pattern, keep using mpmath for it.
        self.pattern_backend_preference: Dict[str, str] = {}
        self.engine_cache_key = self._engine_cache_key()

    def _engine_cache_key(self) -> str:
        def clade_key(clade: FrozenSet[str]) -> Tuple[int, Tuple[str, ...]]:
            return (len(clade), tuple(sorted(clade)))

        def scalar_repr(value: sp.Expr | float | int) -> str:
            expr = sp.sympify(value)
            if expr.free_symbols:
                return sp.srepr(expr)
            return f"{float(expr):.17g}"

        payload: List[str] = [
            self.symbolic_species_newick,
            self.model.name,
            repr(self.model),
            ",".join(self.model.state_labels),
            sp.srepr(sp.Tuple(*self.model.stationary_distribution)),
            ",".join(str(symbol) for symbol in self.external_symbols),
        ]
        payload.extend(
            f"ne:{','.join(sorted(clade))}={scalar_repr(value)}"
            for clade, value in sorted(self.ne_by_clade.items(), key=lambda item: clade_key(item[0]))
        )
        payload.extend(
            f"mu:{','.join(sorted(clade))}={scalar_repr(value)}"
            for clade, value in sorted(self.mu_by_clade.items(), key=lambda item: clade_key(item[0]))
        )
        digest = hashlib.sha256("||".join(payload).encode("utf-8")).hexdigest()
        return digest

    def _expression_cache_path(self, canonical_pattern: str) -> Path:
        cache_root = Path(__file__).resolve().parents[1] / ".cache" / "quartet_probability_expressions"
        return cache_root / self.engine_cache_key / f"{canonical_pattern}.pkl"

    def _load_cached_expression(self, canonical_pattern: str) -> Optional[sp.Expr]:
        path = self._expression_cache_path(canonical_pattern)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                expr = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(expr, sp.Expr):
            return None
        return expr

    def _store_cached_expression(self, canonical_pattern: str, expr: sp.Expr) -> None:
        path = self._expression_cache_path(canonical_pattern)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(expr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)

    def _species_age_constraints(self) -> Tuple[sp.Expr, ...]:
        constraints: List[sp.Expr] = []
        for node in self.internal_nodes:
            node_age = self.age_symbol_by_clade[node.clade]
            constraints.append(StrictGreaterThan(node_age, sp.Integer(0)))
            for child in node.children:
                if child.is_leaf:
                    continue
                constraints.append(
                    StrictGreaterThan(node_age, self.age_symbol_by_clade[child.clade])
                )
        return tuple(constraints)

    def _symbolic_species_newick(self) -> str:
        def emit(node: AnnotatedSpeciesNode) -> str:
            if node.is_leaf:
                assert node.label is not None
                text = node.label
            else:
                text = "(" + ",".join(emit(child) for child in node.children) + ")"
            if node.parent is not None:
                parent_age = self.age_symbol_by_clade[node.parent.clade]
                child_age = sp.Integer(0) if node.is_leaf else self.age_symbol_by_clade[node.clade]
                text += f":{sp.sstr(sp.simplify(parent_age - child_age))}"
            return text

        return emit(self.species_tree.root) + ";"

    def _prepare_gene_trees(self) -> Tuple[PreparedGeneTree, ...]:
        out: List[PreparedGeneTree] = []
        for rooted_tree in enumerate_rooted_labeled_quartets(self.species_tree.taxa_in_order):
            gene_newick = rooted_tree_to_symbolic_newick(rooted_tree, prefix="g")
            case_bundle = enumerate_quartet_msc_cases(self.symbolic_species_newick, gene_newick)
            events: Sequence[GeneEvent] = case_bundle["events"]
            clade_to_event_id = {event.clade: event.eid for event in events}
            out.append(
                PreparedGeneTree(
                    rooted_tree=rooted_tree,
                    gene_newick=gene_newick,
                    clade_to_event_id=clade_to_event_id,
                    branches=tuple(case_bundle["branches"]),
                    events=tuple(events),
                    age_constraints=_gene_age_order_constraints(rooted_tree),
                    cases=tuple(case_bundle["cases"]),
                )
            )
        return tuple(out)

    def canonical_probability_expression(self, canonical_pattern: str) -> sp.Expr:
        """
        Return the symbolic probability expression for one canonical pattern.
        """

        if canonical_pattern in self.pattern_expression_cache:
            return self.pattern_expression_cache[canonical_pattern]

        cached = self._load_cached_expression(canonical_pattern)
        if cached is not None:
            self.pattern_expression_cache[canonical_pattern] = cached
            return cached

        if canonical_pattern not in self.pattern_representatives:
            raise ValueError(f"Unknown canonical pattern '{canonical_pattern}'.")
        representative = self.pattern_representatives[canonical_pattern]
        pattern_map = {
            taxon: nt for taxon, nt in zip(self.species_tree.taxa_in_order, representative)
        }

        total = sp.Integer(0)
        for prepared in self.prepared_gene_trees:
            branches = prepared.branches
            branch_by_id = {branch.bid: branch for branch in branches}
            leaf_branch_by_clade = {
                branch.clade: branch.bid for branch in branches if len(branch.clade) == 1
            }
            ne_by_bid, mu_by_bid = _branch_parameter_maps_by_bid(
                branches, self.ne_by_clade, self.mu_by_clade
            )
            event_by_id = {event.eid: event for event in prepared.events}

            # For a fixed assignment of gene events to species branches, the
            # substitution part is shared across all time-order refinements.
            site_expr_cache: Dict[Tuple[Tuple[str, str], ...], sp.Expr] = {}
            for case in prepared.cases:
                assign_key = tuple(sorted(case.assign_map.items()))
                if assign_key not in site_expr_cache:
                    site_expr = _site_prob_given_gene_tree_case(
                        rooted_tree=prepared.rooted_tree,
                        pattern_map=pattern_map,
                        model=self.model,
                        case=case,
                        clade_to_event_id=prepared.clade_to_event_id,
                        branch_by_id=branch_by_id,
                        leaf_branch_by_clade=leaf_branch_by_clade,
                        mu_by_bid=mu_by_bid,
                    )
                    site_expr_cache[assign_key] = _canonicalize_expr(site_expr, self.symbol_by_name)
                site_expr = site_expr_cache[assign_key]

                density_expr = _canonicalize_expr(
                    _build_case_density_expression(case, branch_by_id, event_by_id, ne_by_bid),
                    self.symbol_by_name,
                )
                condition = _canonicalize_expr(case.condition, self.symbol_by_name)
                integrand = sp.expand(site_expr * density_expr)

                for cond_piece in _condition_dnf_pieces(condition):
                    bounds = _extract_integration_bounds(
                        cond_piece,
                        G_SYMBOLS,
                        base_constraints=self.species_age_constraints,
                        extra_constraints=prepared.age_constraints,
                    )
                    total += _integrate_exponential_sum(integrand, bounds)

        total = sp.factor_terms(sp.expand(total))
        self.pattern_expression_cache[canonical_pattern] = total
        self._store_cached_expression(canonical_pattern, total)
        return total

    def _compile_pattern_probability(self, canonical_pattern: str, backend: str = "math"):
        if backend == "math":
            cache = self.pattern_function_cache_math
            modules = [{"Max": max, "Min": min}, "math"]
        elif backend == "mpmath":
            cache = self.pattern_function_cache_mpmath
            modules = [{"Max": max, "Min": min}, "mpmath"]
        else:
            raise ValueError(f"Unknown backend '{backend}'.")

        if canonical_pattern not in cache:
            expr = self.canonical_probability_expression(canonical_pattern)
            cache[canonical_pattern] = sp.lambdify(
                self.age_symbols + self.external_symbols,
                expr,
                modules=modules,
            )
        return cache[canonical_pattern]

    def canonical_probability(
        self,
        canonical_pattern: str,
        age_by_clade: Mapping[FrozenSet[str], float],
        parameter_values: Optional[Mapping[sp.Symbol | str, float]] = None,
    ) -> float:
        ordered_ages = [age_by_clade[node.clade] for node in self.internal_nodes]
        ordered_params: List[float] = []
        if self.external_symbols:
            if parameter_values is None:
                raise ValueError(
                    "This probability engine requires values for external parameters: "
                    + ", ".join(str(symbol) for symbol in self.external_symbols)
                )
            for symbol in self.external_symbols:
                if symbol in parameter_values:
                    ordered_params.append(float(parameter_values[symbol]))
                elif str(symbol) in parameter_values:
                    ordered_params.append(float(parameter_values[str(symbol)]))
                else:
                    raise ValueError(f"Missing external parameter value for '{symbol}'.")
        args = ordered_ages + ordered_params
        backend = self.pattern_backend_preference.get(canonical_pattern, "math")
        if backend == "math":
            try:
                fast_func = self._compile_pattern_probability(canonical_pattern, backend="math")
                value = fast_func(*args)
                value = float(value)
                if not mp.isfinite(value):
                    raise OverflowError(
                        f"Non-finite math backend value for {canonical_pattern}: {value}"
                    )
            except Exception:
                self.pattern_backend_preference[canonical_pattern] = "mpmath"
                backend = "mpmath"

        if backend == "mpmath":
            safe_func = self._compile_pattern_probability(canonical_pattern, backend="mpmath")
            mp.mp.dps = 80
            value = safe_func(*(mp.mpf(arg) for arg in args))
            if hasattr(value, "imag"):
                imag = abs(value.imag)
                if imag > mp.mpf("1e-20"):
                    raise ValueError(
                        f"Pattern probability became complex for {canonical_pattern}: {value}"
                    )
                value = value.real
            value = float(value)

        if value < 0.0 and value > -1e-12:
            value = 0.0
        if not math.isfinite(value):
            raise ValueError(f"Pattern probability is not finite for {canonical_pattern}: {value}")
        return value

    def canonical_probability_map(
        self,
        age_by_clade: Mapping[FrozenSet[str], float],
        parameter_values: Optional[Mapping[sp.Symbol | str, float]] = None,
        canonical_patterns: Optional[Iterable[str]] = None,
    ) -> Dict[str, float]:
        patterns = (
            sorted(canonical_patterns)
            if canonical_patterns is not None
            else sorted(self.pattern_representatives)
        )
        return {
            code: self.canonical_probability(code, age_by_clade, parameter_values=parameter_values)
            for code in patterns
        }

    def full_pattern_probability_map(
        self,
        age_by_clade: Mapping[FrozenSet[str], float],
        parameter_values: Optional[Mapping[sp.Symbol | str, float]] = None,
    ) -> Dict[str, float]:
        canonical_probs = self.canonical_probability_map(age_by_clade, parameter_values=parameter_values)
        out: Dict[str, float] = {}
        for pattern in itertools.product(self.model.state_labels, repeat=4):
            raw = "".join(pattern)
            code = canonicalize_pattern(raw, model_name=self.model.name)
            out[raw] = canonical_probs[code]
        return out

    def normalization_sum(
        self,
        age_by_clade: Mapping[FrozenSet[str], float],
        parameter_values: Optional[Mapping[sp.Symbol | str, float]] = None,
    ) -> float:
        return float(sum(self.full_pattern_probability_map(age_by_clade, parameter_values=parameter_values).values()))
