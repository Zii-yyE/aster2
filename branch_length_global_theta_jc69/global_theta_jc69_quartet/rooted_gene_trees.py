"""
Utilities for rooted labeled quartet gene-tree enumeration and symbolic Newick.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple, Union


Tree = Union[str, Tuple["Tree", "Tree"]]


def _canon_tree(t: Tree) -> Tree:
    if isinstance(t, str):
        return t
    a, b = t
    ca, cb = _canon_tree(a), _canon_tree(b)
    return (ca, cb) if str(ca) <= str(cb) else (cb, ca)


def enumerate_rooted_labeled_quartets(labels: Sequence[str]) -> List[Tree]:
    """
    Enumerate all rooted binary trees on the input labels.
    For 4 labels, this returns 15 rooted labeled quartets.
    """
    labels = tuple(sorted(labels))
    memo: Dict[Tuple[str, ...], List[Tree]] = {}

    def rec(xs: Tuple[str, ...]) -> List[Tree]:
        if len(xs) == 1:
            return [xs[0]]
        if xs in memo:
            return memo[xs]

        out: List[Tree] = []
        first = xs[0]
        n = len(xs)
        for r in range(1, n):
            for a_idx in itertools.combinations(range(1, n), r - 1):
                a = (first,) + tuple(xs[i] for i in a_idx)
                b = tuple(x for x in xs if x not in a)
                for ta in rec(tuple(sorted(a))):
                    for tb in rec(tuple(sorted(b))):
                        out.append(_canon_tree((ta, tb)))

        uniq: List[Tree] = []
        seen = set()
        for t in out:
            key = str(t)
            if key not in seen:
                seen.add(key)
                uniq.append(t)
        memo[xs] = uniq
        return uniq

    return rec(labels)


def rooted_tree_to_symbolic_newick(tree: Tree, prefix: str = "g") -> str:
    """
    Convert rooted topology to ultrametric symbolic Newick:
    - internal node ages are assigned as g1, g2, g3 (postorder)
    - each edge length is parent_age - child_age (leaf age = 0)
    """
    counter = [0]
    ages: Dict[int, str] = {}

    def assign(node: Tree) -> None:
        if isinstance(node, str):
            return
        a, b = node
        assign(a)
        assign(b)
        counter[0] += 1
        ages[id(node)] = f"{prefix}{counter[0]}"

    def age(node: Tree) -> str:
        return "0" if isinstance(node, str) else ages[id(node)]

    def sub_expr(parent_age: str, child_age: str) -> str:
        if child_age == "0":
            return parent_age
        return f"({parent_age}-({child_age}))"

    def emit(node: Tree) -> str:
        if isinstance(node, str):
            return node
        a, b = node
        p_age = age(node)
        a_br = sub_expr(p_age, age(a))
        b_br = sub_expr(p_age, age(b))
        return f"({emit(a)}:{a_br},{emit(b)}:{b_br})"

    assign(tree)
    return emit(tree) + ";"

