"""
Compute symbolic quartet MSC density f((G,t)|(S,tau)) from species/gene Newick trees.

The implementation follows Chifman & Kubatko (2015), Equations (2), (3), and (7):
- branch-level coalescent density in a population
- no-coalescence probability on a branch
- product across species-tree populations

Input trees are rooted, binary, and ultrametric quartets with branch lengths that can
be numeric or symbolic SymPy expressions.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import sympy as sp


@dataclass(eq=False)
class Node:
    name: Optional[str] = None
    children: Optional[List["Node"]] = None
    brlen: Optional[sp.Expr] = None  # length from this node to its parent
    parent: Optional["Node"] = None

    def is_leaf(self) -> bool:
        return not self.children

    def leafset(self) -> FrozenSet[str]:
        if self.is_leaf():
            if self.name is None:
                raise ValueError("Encountered unnamed leaf.")
            return frozenset([self.name])
        out: set[str] = set()
        for ch in self.children or []:
            out |= set(ch.leafset())
        return frozenset(out)


@dataclass
class SpeciesBranch:
    bid: str
    clade: FrozenSet[str]
    lower: sp.Expr
    upper: Optional[sp.Expr]  # None means +infinity (root-above branch)
    length: Optional[sp.Expr]  # upper-lower for finite branches
    child: Optional[Node] = None
    parent_bid: Optional[str] = None


@dataclass
class GeneEvent:
    eid: str
    clade: FrozenSet[str]
    age: sp.Expr
    node: Optional[Node] = None


@dataclass
class MSCCase:
    condition: Any
    assign_map: Dict[str, str]
    branch_event_order: Dict[str, Tuple[str, ...]]
    uv: Dict[str, Tuple[int, int]]


def parse_newick_with_lengths(newick: str) -> Node:
    s = newick.strip()
    if not s.endswith(";"):
        raise ValueError("Newick must end with ';'")
    s = s[:-1]
    i, n = 0, len(s)

    def skip_ws() -> None:
        nonlocal i
        while i < n and s[i].isspace():
            i += 1

    def parse_name() -> Optional[str]:
        nonlocal i
        skip_ws()
        start = i
        while i < n and s[i] not in [",", "(", ")", ":", ";"]:
            i += 1
        name = s[start:i].strip()
        return name if name else None

    def parse_brlen() -> Optional[sp.Expr]:
        nonlocal i
        skip_ws()
        if i < n and s[i] == ":":
            i += 1
            start = i
            depth = 0
            while i < n:
                ch = s[i]
                if ch == "(":
                    depth += 1
                    i += 1
                    continue
                if ch == ")":
                    if depth == 0:
                        break
                    depth -= 1
                    i += 1
                    continue
                if ch in [",", ";"] and depth == 0:
                    break
                i += 1
            raw = s[start:i].strip()
            if not raw:
                raise ValueError("Empty branch length after ':'.")
            try:
                return sp.sympify(raw)
            except Exception as exc:
                raise ValueError(f"Could not parse branch length '{raw}': {exc}") from exc
        return None

    def parse_subtree() -> Node:
        nonlocal i
        skip_ws()
        if i < n and s[i] == "(":
            i += 1
            children: List[Node] = []
            while True:
                children.append(parse_subtree())
                skip_ws()
                if i >= n:
                    raise ValueError("Unexpected end of Newick.")
                if s[i] == ",":
                    i += 1
                    continue
                if s[i] == ")":
                    i += 1
                    break
                raise ValueError(f"Unexpected character '{s[i]}' in Newick.")
            name = parse_name()  # optional internal label
            brlen = parse_brlen()
            return Node(name=name, children=children, brlen=brlen)

        name = parse_name()
        if not name:
            raise ValueError("Encountered unnamed leaf.")
        brlen = parse_brlen()
        return Node(name=name, children=None, brlen=brlen)

    root = parse_subtree()
    skip_ws()
    if i != n:
        raise ValueError(f"Trailing content in Newick: '{s[i:]}'")
    _annotate_parents(root, None)
    return root


def _annotate_parents(node: Node, parent: Optional[Node]) -> None:
    node.parent = parent
    for ch in node.children or []:
        _annotate_parents(ch, node)


def _all_nodes(node: Node) -> List[Node]:
    out = [node]
    for ch in node.children or []:
        out.extend(_all_nodes(ch))
    return out


def _compute_ages_from_present(root: Node) -> Dict[Node, sp.Expr]:
    ages: Dict[Node, sp.Expr] = {}

    def dfs(node: Node) -> sp.Expr:
        if node.is_leaf():
            ages[node] = sp.Integer(0)
            return ages[node]
        if not node.children:
            raise ValueError("Malformed tree: internal node with no children.")
        candidate_ages: List[sp.Expr] = []
        for ch in node.children:
            if ch.brlen is None:
                raise ValueError("All non-root edges must have branch lengths.")
            candidate_ages.append(sp.simplify(dfs(ch) + ch.brlen))
        base = candidate_ages[0]
        for other in candidate_ages[1:]:
            d = sp.simplify(other - base)
            if d != 0 and d.equals(0) is not True:
                raise ValueError(
                    f"Tree is not ultrametric at node '{node.name or 'internal'}': "
                    f"child-derived ages {candidate_ages} disagree."
                )
        ages[node] = base
        return base

    dfs(root)
    return ages


def _validate_rooted_binary_quartet(root: Node, label: str) -> None:
    leaves = sorted(root.leafset())
    if len(leaves) != 4:
        raise ValueError(f"{label} must have exactly 4 leaves (quartet). Found {len(leaves)}.")
    for node in _all_nodes(root):
        if not node.is_leaf():
            if not node.children or len(node.children) != 2:
                raise ValueError(f"{label} must be rooted binary.")


def _root_split(root: Node) -> FrozenSet[FrozenSet[str]]:
    if not root.children or len(root.children) != 2:
        raise ValueError("Root must have exactly 2 children.")
    return frozenset([root.children[0].leafset(), root.children[1].leafset()])


def _split_pretty(split: FrozenSet[FrozenSet[str]]) -> str:
    parts = sorted([tuple(sorted(x)) for x in split])
    return f"{parts[0]} | {parts[1]}"


def _collect_species_branches(root: Node, ages: Dict[Node, sp.Expr]) -> List[SpeciesBranch]:
    out: List[SpeciesBranch] = []
    node_to_bid: Dict[Node, str] = {}

    def walk(node: Node) -> None:
        for ch in node.children or []:
            upper = ages[node]
            lower = ages[ch]
            bid = f"edge_{len(out)}"
            out.append(
                SpeciesBranch(
                    bid=bid,
                    clade=ch.leafset(),
                    lower=lower,
                    upper=upper,
                    length=sp.simplify(upper - lower),
                    child=ch,
                )
            )
            node_to_bid[ch] = bid
            walk(ch)

    walk(root)
    out.append(
        SpeciesBranch(
            bid="root_above",
            clade=root.leafset(),
            lower=ages[root],
            upper=None,
            length=None,
            child=None,
        )
    )
    node_to_bid[root] = "root_above"

    for br in out:
        if br.bid == "root_above" or br.child is None:
            br.parent_bid = None
            continue
        parent_node = br.child.parent
        if parent_node is None:
            br.parent_bid = "root_above"
            continue
        br.parent_bid = node_to_bid[parent_node]
    return out


def _collect_gene_events(root: Node, ages: Dict[Node, sp.Expr]) -> List[GeneEvent]:
    out: List[GeneEvent] = []
    for node in _all_nodes(root):
        if not node.is_leaf():
            out.append(
                GeneEvent(
                    eid=f"g{len(out)+1}",
                    clade=node.leafset(),
                    age=ages[node],
                    node=node,
                )
            )
    return out


def _is_true(x: Any) -> bool:
    return x is True or x == sp.true


def _is_false(x: Any) -> bool:
    return x is False or x == sp.false


def _gt(a: sp.Expr, b: sp.Expr) -> Any:
    d = sp.simplify(a - b)
    if d.is_positive:
        return sp.true
    if d.is_zero or d.is_negative:
        return sp.false
    return sp.StrictGreaterThan(a, b)


def _lt(a: sp.Expr, b: sp.Expr) -> Any:
    d = sp.simplify(a - b)
    if d.is_negative:
        return sp.true
    if d.is_zero or d.is_positive:
        return sp.false
    return sp.StrictLessThan(a, b)


def _le(a: sp.Expr, b: sp.Expr) -> Any:
    d = sp.simplify(a - b)
    if d.is_negative or d.is_zero:
        return sp.true
    if d.is_positive:
        return sp.false
    return sp.LessThan(a, b)


def _and_all(conds: Sequence[Any]) -> Any:
    cur: Any = sp.true
    for c in conds:
        if _is_false(c):
            return sp.false
        if _is_true(c):
            continue
        if _is_true(cur):
            cur = c
        else:
            cur = sp.And(cur, c)
    return cur


def _age_in_branch_condition(age: sp.Expr, branch: SpeciesBranch) -> Any:
    conds: List[Any] = [_gt(age, branch.lower)]
    if branch.upper is not None:
        conds.append(_le(age, branch.upper))
    return _and_all(conds)


def _species_branch_mapping_lines(branches: List[SpeciesBranch]) -> List[str]:
    lines = ["Species populations (from Newick edges):"]
    for b in branches:
        if b.upper is None:
            lines.append(f"- {b.bid}: clade={sorted(b.clade)}, interval=({b.lower}, +inf)")
        else:
            lines.append(
                f"- {b.bid}: clade={sorted(b.clade)}, interval=({b.lower}, {b.upper}], length={b.length}"
            )
    return lines


def enumerate_quartet_msc_cases(
    species_newick: str,
    gene_newick: str,
) -> Dict[str, Any]:
    """
    Enumerate quartet MSC cases for a rooted binary species/gene-tree pair.

    Returns a dict with:
    - cases: list of mutually exclusive MSC cases
    - branches: species-tree branches with time intervals and ancestry links
    - events: gene-tree coalescent events with clades/ages
    - species_split / gene_split: root splits for quick topology inspection
    - mapping_lines: species branch/population mapping with time intervals
    """
    S = parse_newick_with_lengths(species_newick)
    G = parse_newick_with_lengths(gene_newick)
    _validate_rooted_binary_quartet(S, "Species tree")
    _validate_rooted_binary_quartet(G, "Gene tree")

    species_taxa = S.leafset()
    gene_taxa = G.leafset()
    if species_taxa != gene_taxa:
        raise ValueError(
            f"Species/Gene leaf labels differ.\nSpecies: {sorted(species_taxa)}\nGene: {sorted(gene_taxa)}"
        )

    ages_S = _compute_ages_from_present(S)
    ages_G = _compute_ages_from_present(G)

    branches = _collect_species_branches(S, ages_S)
    events = _collect_gene_events(G, ages_G)

    # Candidate branch assignments for each gene coalescent event.
    event_candidates: Dict[str, List[Tuple[str, Any]]] = {}
    for ev in events:
        cands: List[Tuple[str, Any]] = []
        for br in branches:
            if ev.clade.issubset(br.clade):
                cond = _age_in_branch_condition(ev.age, br)
                if not _is_false(cond):
                    cands.append((br.bid, cond))
        if not cands:
            raise ValueError(
                f"No valid population interval for event {ev.eid} (clade={sorted(ev.clade)}, age={ev.age})."
            )
        event_candidates[ev.eid] = cands

    event_by_id = {ev.eid: ev for ev in events}
    branch_by_id = {br.bid: br for br in branches}

    def branch_is_ancestor_or_same(ancestor_bid: str, descendant_bid: str) -> bool:
        cur = descendant_bid
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            if cur == ancestor_bid:
                return True
            seen.add(cur)
            cur = branch_by_id[cur].parent_bid
        return False

    event_parent: Dict[str, str] = {}
    node_to_event_id = {
        ev.node: ev.eid for ev in events if ev.node is not None
    }
    for ev in events:
        if ev.node is None or ev.node.parent is None:
            continue
        parent_node = ev.node.parent
        if parent_node in node_to_event_id:
            event_parent[ev.eid] = node_to_event_id[parent_node]

    candidate_lists = [
        [(ev.eid, bid, cond) for (bid, cond) in event_candidates[ev.eid]]
        for ev in events
    ]
    raw_assignments = itertools.product(*candidate_lists)

    all_cases: List[MSCCase] = []

    # Used to enforce ancestry-consistent event ordering when multiple events land on one branch.
    def clade_ancestor(x: GeneEvent, y: GeneEvent) -> bool:
        return x.clade != y.clade and x.clade.issuperset(y.clade)

    for assignment in raw_assignments:
        assign_map: Dict[str, str] = {}
        assign_cond_parts: List[Any] = []
        branch_events: Dict[str, List[GeneEvent]] = {b.bid: [] for b in branches}
        for eid, bid, cond in assignment:
            assign_map[eid] = bid
            assign_cond_parts.append(cond)
            branch_events[bid].append(event_by_id[eid])
        assign_cond = _and_all(assign_cond_parts)
        if _is_false(assign_cond):
            continue

        ancestry_valid = True
        for child_eid, parent_eid in event_parent.items():
            if not branch_is_ancestor_or_same(assign_map[parent_eid], assign_map[child_eid]):
                ancestry_valid = False
                break
        if not ancestry_valid:
            continue

        # Compute entering/leaving lineage counts (u,v) on each species branch from leaves upward.
        uv: Dict[str, Tuple[int, int]] = {}
        finite_by_child: Dict[Node, SpeciesBranch] = {
            b.child: b for b in branches if b.upper is not None and b.child is not None
        }

        valid = True

        def recurse_v(node: Node) -> int:
            nonlocal valid
            if node.is_leaf():
                u = 1
            else:
                u = sum(recurse_v(ch) for ch in node.children or [])
            br = finite_by_child.get(node)
            if br is None:
                # root node; no finite branch above.
                return u
            m = len(branch_events[br.bid])
            v = u - m
            if v < 1 or v > u:
                valid = False
            uv[br.bid] = (u, v)
            return v

        root_entering = recurse_v(S)
        root_branch = branch_by_id["root_above"]
        m_root = len(branch_events[root_branch.bid])
        v_root = root_entering - m_root
        uv[root_branch.bid] = (root_entering, v_root)
        if v_root != 1:
            valid = False
        if not valid:
            continue

        # Enumerate event orders within each branch (needed if order is ambiguous symbolically).
        branch_order_options: Dict[str, List[Tuple[List[GeneEvent], Any]]] = {}
        for bid, evs in branch_events.items():
            if len(evs) <= 1:
                branch_order_options[bid] = [(evs, sp.true)]
                continue

            opts: List[Tuple[List[GeneEvent], Any]] = []
            for perm in itertools.permutations(evs):
                # Younger events first on a branch.
                ok = True
                conds: List[Any] = []

                # Ancestor clades cannot appear before descendant clades.
                for i in range(len(perm)):
                    for j in range(i + 1, len(perm)):
                        if clade_ancestor(perm[i], perm[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                # Enforce strictly increasing ages along the chosen order.
                for i in range(len(perm) - 1):
                    conds.append(_lt(perm[i].age, perm[i + 1].age))
                cond = _and_all(conds)
                if _is_false(cond):
                    continue
                opts.append((list(perm), cond))

            if not opts:
                valid = False
                break
            branch_order_options[bid] = opts
        if not valid:
            continue

        # Build branch factors using Equation (2)/(3), then combine.
        branch_ids = [b.bid for b in branches]
        per_branch_opts = [branch_order_options[bid] for bid in branch_ids]
        for one_choice in itertools.product(*per_branch_opts):
            cond_parts = [assign_cond]
            feasible = True
            branch_event_order: Dict[str, Tuple[str, ...]] = {}
            for bid, (ordered_events, order_cond) in zip(branch_ids, one_choice):
                cond_parts.append(order_cond)
                if _is_false(order_cond):
                    feasible = False
                    break
                branch_event_order[bid] = tuple(ev.eid for ev in ordered_events)

            if not feasible:
                continue

            full_cond = _and_all(cond_parts)
            if _is_false(full_cond):
                continue
            all_cases.append(
                MSCCase(
                    condition=sp.simplify(full_cond),
                    assign_map=dict(assign_map),
                    branch_event_order=branch_event_order,
                    uv={bid: tuple(values) for bid, values in uv.items()},
                )
            )

    return {
        "cases": all_cases,
        "branches": branches,
        "events": events,
        "species_split": _split_pretty(_root_split(S)),
        "gene_split": _split_pretty(_root_split(G)),
        "mapping_lines": _species_branch_mapping_lines(branches),
        "species_newick": species_newick,
        "gene_newick": gene_newick,
    }


def msc_density_quartet_from_newicks(
    species_newick: str,
    gene_newick: str,
    theta_symbol: str = "theta",
    simplify_piecewise: bool = True,
) -> Dict[str, Any]:
    """
    Symbolically compute f((G,t)|(S,tau)) for a rooted binary quartet.

    Returns a dict with:
    - f: SymPy expression (possibly Piecewise) for the conditional density
    - theta: the SymPy population-size symbol
    - species_split / gene_split: root splits for quick topology inspection
    - mapping_lines: species branch/population mapping with time intervals
    """
    theta = sp.Symbol(theta_symbol, positive=True)
    enumerated = enumerate_quartet_msc_cases(species_newick, gene_newick)
    branches: List[SpeciesBranch] = enumerated["branches"]
    events: List[GeneEvent] = enumerated["events"]
    cases: List[MSCCase] = enumerated["cases"]
    branch_by_id = {br.bid: br for br in branches}
    event_by_id = {ev.eid: ev for ev in events}

    all_cases: List[Tuple[sp.Expr, Any]] = []
    for case in cases:
        expr = sp.Integer(1)
        feasible = True
        for bid, ordered_eids in case.branch_event_order.items():
            br = branch_by_id[bid]
            u, _ = case.uv[bid]
            j = u
            prev_local = sp.Integer(0)

            for eid in ordered_eids:
                ev = event_by_id[eid]
                local_t = sp.simplify(ev.age - br.lower)
                delta = sp.simplify(local_t - prev_local)
                expr *= (sp.Integer(2) / theta) * sp.exp(-sp.Integer(j * (j - 1)) * delta / theta)
                j -= 1
                prev_local = local_t

            v = j
            if br.upper is not None:
                rem = sp.simplify(br.length - prev_local)
                expr *= sp.exp(-sp.Integer(v * (v - 1)) * rem / theta)
            elif v != 1:
                feasible = False
                break

        if feasible:
            all_cases.append((sp.simplify(expr), case.condition))

    if not all_cases:
        f_expr: sp.Expr = sp.Integer(0)
    elif len(all_cases) == 1 and _is_true(all_cases[0][1]):
        f_expr = all_cases[0][0]
    else:
        # Keep first-seen order; conditions are intended to be mutually exclusive.
        f_expr = sp.Piecewise(*all_cases, (sp.Integer(0), True))
        if simplify_piecewise:
            f_expr = sp.simplify(f_expr)

    return {
        "f": f_expr,
        "theta": theta,
        "species_split": enumerated["species_split"],
        "gene_split": enumerated["gene_split"],
        "mapping_lines": enumerated["mapping_lines"],
        "species_newick": species_newick,
        "gene_newick": gene_newick,
    }


def msc_density_balanced_from_newicks(species_newick: str, gene_newick: str) -> Dict[str, Any]:
    """
    Backward-compatible wrapper kept for existing callers.
    """
    out = msc_density_quartet_from_newicks(species_newick, gene_newick)
    out["relation"] = (
        "concordant" if out["species_split"] == out["gene_split"] else "discordant"
    )
    return out


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute symbolic quartet MSC density f((G,t)|(S,tau)) from species and gene Newick trees."
    )
    p.add_argument("--species", required=True, help="Rooted binary quartet species tree Newick with branch lengths.")
    p.add_argument("--gene", required=True, help="Rooted binary quartet gene tree Newick with branch lengths.")
    p.add_argument("--theta", default="theta", help="Population-size symbol name (default: theta).")
    return p


if __name__ == "__main__":
    args = _build_cli().parse_args()
    out = msc_density_quartet_from_newicks(args.species, args.gene, theta_symbol=args.theta)
    print("Species Newick:", out["species_newick"])
    print("Gene    Newick:", out["gene_newick"])
    print("Species split :", out["species_split"])
    print("Gene split    :", out["gene_split"])
    print("\nSpecies-population mapping:")
    for line in out["mapping_lines"]:
        print(line)
    print("\nSymbolic density f((G,t)|(S,tau)):")
    print(out["f"])
