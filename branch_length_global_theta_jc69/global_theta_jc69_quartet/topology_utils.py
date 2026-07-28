"""
Generic rooted-tree topology utilities for branch-centric quartet stitching.

This module intentionally ignores branch lengths and annotations in the input
Newick. It keeps only the rooted binary topology and taxon labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Iterator, List, Optional, Sequence, Tuple


@dataclass(eq=False)
class TopologyNode:
    name: Optional[str] = None
    children: List["TopologyNode"] = field(default_factory=list)
    parent: Optional["TopologyNode"] = None
    clade: FrozenSet[str] = frozenset()

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


def _parse_comment(text: str, i: int) -> int:
    if i >= len(text) or text[i] != "[":
        return i
    depth = 0
    while i < len(text):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("Malformed Newick comment: missing closing ']'.")


def parse_rooted_topology_newick(raw: str) -> TopologyNode:
    """
    Parse a rooted binary Newick topology.

    Supported input features:
    - leaf labels
    - optional internal labels
    - optional branch lengths (ignored)
    - optional square-bracket comments/annotations (ignored)
    """

    text = raw.strip()
    if not text.endswith(";"):
        raise ValueError("Newick must end with ';'.")
    text = text[:-1]
    i = 0
    n = len(text)

    def skip_ws() -> None:
        nonlocal i
        while i < n and text[i].isspace():
            i += 1

    def parse_label() -> Optional[str]:
        nonlocal i
        skip_ws()
        start = i
        while i < n and text[i] not in "[,:();":
            i += 1
        label = text[start:i].strip().strip("'\"")
        return label or None

    def skip_annotations() -> None:
        nonlocal i
        skip_ws()
        while i < n and text[i] == "[":
            i = _parse_comment(text, i)
            skip_ws()

    def skip_branch_length() -> None:
        nonlocal i
        skip_ws()
        if i >= n or text[i] != ":":
            return
        i += 1
        skip_ws()
        while i < n and text[i] not in ",);":
            if text[i] == "[":
                i = _parse_comment(text, i)
                continue
            i += 1

    def parse_subtree() -> TopologyNode:
        nonlocal i
        skip_ws()
        if i < n and text[i] == "(":
            i += 1
            children: List[TopologyNode] = []
            while True:
                children.append(parse_subtree())
                skip_ws()
                if i >= n:
                    raise ValueError("Unexpected end of Newick.")
                if text[i] == ",":
                    i += 1
                    continue
                if text[i] == ")":
                    i += 1
                    break
                raise ValueError(f"Unexpected character '{text[i]}' in Newick.")
            label = parse_label()
            skip_annotations()
            skip_branch_length()
            node = TopologyNode(name=label, children=children)
            for child in children:
                child.parent = node
            return node

        label = parse_label()
        if label is None:
            raise ValueError("Encountered an unnamed leaf in the species tree topology.")
        skip_annotations()
        skip_branch_length()
        return TopologyNode(name=label)

    root = parse_subtree()
    skip_ws()
    if i != n:
        raise ValueError(f"Trailing content in Newick: '{text[i:]}'")
    _annotate_clades(root, parent=None)
    _validate_rooted_binary_tree(root)
    return root


def _annotate_clades(node: TopologyNode, parent: Optional[TopologyNode]) -> FrozenSet[str]:
    node.parent = parent
    if node.is_leaf:
        if node.name is None:
            raise ValueError("Encountered unnamed leaf.")
        node.clade = frozenset([node.name])
        return node.clade
    if len(node.children) != 2:
        raise ValueError("Species tree must be rooted and binary.")
    clade: set[str] = set()
    for child in node.children:
        clade |= set(_annotate_clades(child, node))
    node.clade = frozenset(clade)
    return node.clade


def _validate_rooted_binary_tree(root: TopologyNode) -> None:
    seen: set[str] = set()
    leaves: List[str] = []
    for node in iter_nodes_postorder(root):
        if node.is_leaf:
            if node.name is None:
                raise ValueError("Encountered unnamed leaf.")
            if node.name in seen:
                raise ValueError(f"Duplicate taxon label '{node.name}' in species tree.")
            seen.add(node.name)
            leaves.append(node.name)
        elif len(node.children) != 2:
            raise ValueError("Species tree must be rooted and binary.")
    if len(leaves) < 4:
        raise ValueError("This implementation requires at least 4 taxa.")


def iter_nodes_postorder(node: TopologyNode) -> Iterator[TopologyNode]:
    for child in node.children:
        yield from iter_nodes_postorder(child)
    yield node


def iter_leaves_left_to_right(node: TopologyNode) -> Iterator[TopologyNode]:
    if node.is_leaf:
        yield node
        return
    for child in node.children:
        yield from iter_leaves_left_to_right(child)


def leaf_names_left_to_right(node: TopologyNode) -> Tuple[str, ...]:
    return tuple(leaf.name for leaf in iter_leaves_left_to_right(node) if leaf.name is not None)


def sibling(node: TopologyNode) -> Optional[TopologyNode]:
    if node.parent is None:
        return None
    left, right = node.parent.children
    return right if left is node else left


def format_topology_newick(node: TopologyNode) -> str:
    if node.is_leaf:
        assert node.name is not None
        return node.name
    return "(" + ",".join(format_topology_newick(child) for child in node.children) + ")"


def format_newick_with_branch_lengths(
    node: TopologyNode,
    branch_lengths_by_clade: dict[FrozenSet[str], float],
) -> str:
    if node.is_leaf:
        assert node.name is not None
        text = node.name
    else:
        text = "(" + ",".join(format_newick_with_branch_lengths(child, branch_lengths_by_clade) for child in node.children) + ")"
    if node.parent is not None:
        value = branch_lengths_by_clade[node.clade]
        text += f":{value:.10f}"
    return text


def clade_label(clade: FrozenSet[str]) -> str:
    return "{" + ",".join(sorted(clade)) + "}"


def prune_to_taxa(node: TopologyNode, selected: FrozenSet[str]) -> Optional[TopologyNode]:
    if node.is_leaf:
        if node.name in selected:
            return TopologyNode(name=node.name)
        return None
    kept_children: List[TopologyNode] = []
    for child in node.children:
        pruned = prune_to_taxa(child, selected)
        if pruned is not None:
            kept_children.append(pruned)
    if not kept_children:
        return None
    if len(kept_children) == 1:
        return kept_children[0]
    if len(kept_children) != 2:
        raise RuntimeError("Pruned rooted subtree is not binary.")
    new_node = TopologyNode(children=kept_children)
    for child in kept_children:
        child.parent = new_node
    _annotate_clades(new_node, parent=None)
    return new_node


def induce_rooted_quartet(root: TopologyNode, quartet_taxa: Sequence[str]) -> TopologyNode:
    selected = frozenset(quartet_taxa)
    if len(selected) != 4:
        raise ValueError("Expected exactly 4 distinct taxa for an induced quartet.")
    pruned = prune_to_taxa(root, selected)
    if pruned is None:
        raise RuntimeError("Failed to induce quartet from the species tree topology.")
    _annotate_clades(pruned, parent=None)
    taxa = set(leaf_names_left_to_right(pruned))
    if taxa != selected:
        raise RuntimeError(
            f"Induced quartet taxa {sorted(taxa)} do not match requested taxa {sorted(selected)}."
        )
    return pruned


def format_annotated_quartet_newick(
    node: TopologyNode,
    ne: float,
    mu: float,
    is_root: bool = True,
) -> str:
    annotation = f"[&Ne={ne:.15g},mu={mu:.15g}]"
    if node.is_leaf:
        assert node.name is not None
        return f"{node.name}{annotation}"
    inner = ",".join(
        format_annotated_quartet_newick(child, ne=ne, mu=mu, is_root=False)
        for child in node.children
    )
    return f"({inner}){annotation}"


def load_topology_arg(raw: str) -> TopologyNode:
    from pathlib import Path

    path = Path(raw)
    text = path.read_text(encoding="utf-8") if path.exists() else raw
    return parse_rooted_topology_newick(text)
