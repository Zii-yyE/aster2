"""
Minimal rooted-quartet species-tree utilities for the current runtime workflow.

This module now supports only the topology-only quartet construction path used
by the global-theta JC69 estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Mapping, Optional, Tuple

from .topology_utils import TopologyNode, leaf_names_left_to_right, parse_rooted_topology_newick


@dataclass
class AnnotatedSpeciesNode:
    label: Optional[str]
    ne: float
    mu: float
    children: List["AnnotatedSpeciesNode"] = field(default_factory=list)
    parent: Optional["AnnotatedSpeciesNode"] = None
    clade: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass(frozen=True)
class ParsedQuartetSpeciesTree:
    raw_newick: str
    root: AnnotatedSpeciesNode
    taxa_in_order: Tuple[str, str, str, str]
    topology_newick: str


def _from_topology_node(node: TopologyNode, ne: float, mu: float) -> AnnotatedSpeciesNode:
    if node.is_leaf:
        return AnnotatedSpeciesNode(label=node.name, ne=ne, mu=mu, clade=node.clade)
    children = [_from_topology_node(child, ne, mu) for child in node.children]
    out = AnnotatedSpeciesNode(label=node.name, ne=ne, mu=mu, children=children, clade=node.clade)
    for child in children:
        child.parent = out
    return out


def iter_nodes_postorder(node: AnnotatedSpeciesNode) -> Iterator[AnnotatedSpeciesNode]:
    for child in node.children:
        yield from iter_nodes_postorder(child)
    yield node


def iter_internal_nodes_postorder(node: AnnotatedSpeciesNode) -> Iterator[AnnotatedSpeciesNode]:
    for item in iter_nodes_postorder(node):
        if not item.is_leaf:
            yield item


def branch_parameter_maps(
    root: AnnotatedSpeciesNode,
) -> Tuple[Dict[FrozenSet[str], float], Dict[FrozenSet[str], float]]:
    ne_by_clade: Dict[FrozenSet[str], float] = {}
    mu_by_clade: Dict[FrozenSet[str], float] = {}
    for node in iter_nodes_postorder(root):
        ne_by_clade[node.clade] = float(node.ne)
        mu_by_clade[node.clade] = float(node.mu)
    return ne_by_clade, mu_by_clade


def format_topology_newick(root: AnnotatedSpeciesNode) -> str:
    if root.is_leaf:
        assert root.label is not None
        return root.label
    return "(" + ",".join(format_topology_newick(child) for child in root.children) + ")"


def format_plain_newick_with_branch_lengths(
    root: AnnotatedSpeciesNode,
    branch_lengths_by_clade: Mapping[FrozenSet[str], float],
) -> str:
    def emit(node: AnnotatedSpeciesNode) -> str:
        if node.is_leaf:
            assert node.label is not None
            text = node.label
        else:
            text = "(" + ",".join(emit(child) for child in node.children) + ")"
        if node.parent is not None:
            value = branch_lengths_by_clade[node.clade]
            text += f":{value:.10f}"
        return text

    return emit(root)


def parse_uniform_quartet_species_tree(
    raw: str,
    ne: float = 1.0,
    mu: float = 1.0,
) -> ParsedQuartetSpeciesTree:
    if ne <= 0.0:
        raise ValueError("Uniform Ne must be positive.")
    if mu <= 0.0:
        raise ValueError("Uniform mu must be positive.")

    topology_root = parse_rooted_topology_newick(raw)
    taxa = leaf_names_left_to_right(topology_root)
    if len(taxa) != 4:
        raise ValueError(f"Expected a rooted quartet species tree with 4 taxa. Found {len(taxa)}.")
    if len(set(taxa)) != 4:
        raise ValueError("Taxon labels must be unique in the species tree.")

    root = _from_topology_node(topology_root, ne=float(ne), mu=float(mu))
    return ParsedQuartetSpeciesTree(
        raw_newick=raw.strip(),
        root=root,
        taxa_in_order=tuple(taxa),
        topology_newick=format_topology_newick(root),
    )
