"""Core KnowledgeGraph class backed by a NetworkX DiGraph."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import networkx as nx


class KnowledgeGraph:
    """A directed knowledge graph that stores entities as nodes and relations as edges.

    Internally wraps a :class:`networkx.DiGraph` and provides a domain-oriented
    API for entity/relation management, path finding, and community operations.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._community_map: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def add_entity(
        self,
        id: str,
        label: str,
        type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update an entity node in the graph."""
        if properties is None:
            properties = {}
        self._graph.add_node(id, label=label, type=type, **properties)

    def get_entity(self, id: str) -> Dict[str, Any]:
        """Return all attributes of a node, or an empty dict if not found."""
        if id not in self._graph:
            return {}
        data = dict(self._graph.nodes[id])
        data["id"] = id
        return data

    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Return all entities whose *type* attribute matches *entity_type*."""
        result: List[Dict[str, Any]] = []
        for node_id, attrs in self._graph.nodes(data=True):
            if attrs.get("type") == entity_type:
                entity = dict(attrs)
                entity["id"] = node_id
                result.append(entity)
        return result

    # ------------------------------------------------------------------
    # Relation operations
    # ------------------------------------------------------------------

    def add_relation(
        self,
        src: str,
        dst: str,
        relation_type: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update a directed relation edge from *src* to *dst*."""
        if properties is None:
            properties = {}
        self._graph.add_edge(src, dst, relation_type=relation_type, weight=weight, **properties)

    def get_relation(self, src: str, dst: str) -> Dict[str, Any]:
        """Return all attributes of an edge, or an empty dict if not found."""
        if not self._graph.has_edge(src, dst):
            return {}
        data = dict(self._graph.edges[src, dst])
        data["src"] = src
        data["dst"] = dst
        return data

    # ------------------------------------------------------------------
    # Neighbour / subgraph operations
    # ------------------------------------------------------------------

    def get_neighbors(self, node: str, direction: str = "both") -> List[str]:
        """Return neighbour node IDs.

        Parameters
        ----------
        node:
            The node whose neighbours to retrieve.
        direction:
            ``'out'`` – successors only; ``'in'`` – predecessors only;
            ``'both'`` – union of both sets.
        """
        if direction == "out":
            return list(self._graph.successors(node))
        if direction == "in":
            return list(self._graph.predecessors(node))
        out = set(self._graph.successors(node))
        in_ = set(self._graph.predecessors(node))
        return list(out | in_)

    def subgraph(self, nodes: List[str]) -> "KnowledgeGraph":
        """Return a new KnowledgeGraph containing only *nodes* and edges between them."""
        sg = self._graph.subgraph(nodes)
        new_kg = KnowledgeGraph()
        for node_id, attrs in sg.nodes(data=True):
            new_kg._graph.add_node(node_id, **attrs)
        for src, dst, attrs in sg.edges(data=True):
            new_kg._graph.add_edge(src, dst, **attrs)
        for node_id in nodes:
            if node_id in self._community_map:
                new_kg._community_map[node_id] = self._community_map[node_id]
        return new_kg

    def get_subgraph_around(self, node_id: str, hops: int = 2) -> "KnowledgeGraph":
        """Return a new KnowledgeGraph containing all nodes within *hops* of *node_id*.

        The search considers both in-edges and out-edges (undirected BFS).
        """
        visited: set = set()
        frontier: set = {node_id}
        for _ in range(hops):
            next_frontier: set = set()
            for n in frontier:
                if n not in visited:
                    visited.add(n)
                    next_frontier.update(self.get_neighbors(n, direction="both"))
            frontier = next_frontier - visited
        visited.update(frontier)
        return self.subgraph(list(visited))

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------

    def find_paths(
        self, src: str, dst: str, max_hops: int = 3
    ) -> List[List[str]]:
        """Find all simple paths from *src* to *dst* up to *max_hops* edges.

        Returns a list of node-ID lists, each representing one path.
        """
        if src not in self._graph or dst not in self._graph:
            return []
        try:
            paths = list(
                nx.all_simple_paths(self._graph, source=src, target=dst, cutoff=max_hops)
            )
        except nx.NetworkXNoPath:
            paths = []
        return paths

    # ------------------------------------------------------------------
    # Community operations
    # ------------------------------------------------------------------

    def assign_communities(self, community_map: Dict[str, int]) -> None:
        """Bulk-assign node -> community_id mappings."""
        self._community_map.update(community_map)

    def get_community(self, node_id: str) -> Optional[int]:
        """Return the community ID for *node_id*, or ``None`` if unassigned."""
        return self._community_map.get(node_id)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the graph to a plain-Python dict (JSON-safe)."""
        nodes = []
        for node_id, attrs in self._graph.nodes(data=True):
            entry = {"id": node_id, **attrs}
            if node_id in self._community_map:
                entry["community"] = self._community_map[node_id]
            nodes.append(entry)

        edges = []
        for src, dst, attrs in self._graph.edges(data=True):
            edges.append({"src": src, "dst": dst, **attrs})

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Reconstruct a KnowledgeGraph from the dict produced by :meth:`to_dict`."""
        kg = cls()
        for node in data.get("nodes", []):
            node = dict(node)
            node_id = node.pop("id")
            community = node.pop("community", None)
            label = node.pop("label", node_id)
            entity_type = node.pop("type", "UNKNOWN")
            kg.add_entity(node_id, label=label, type=entity_type, properties=node)
            if community is not None:
                kg._community_map[node_id] = community

        for edge in data.get("edges", []):
            edge = dict(edge)
            src = edge.pop("src")
            dst = edge.pop("dst")
            relation_type = edge.pop("relation_type", "RELATED_TO")
            weight = edge.pop("weight", 1.0)
            kg.add_relation(src, dst, relation_type=relation_type, weight=weight, properties=edge)

        return kg

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[str]:
        """Return all node IDs."""
        return list(self._graph.nodes())

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Return all (src, dst) edge tuples."""
        return list(self._graph.edges())

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._graph

    def __len__(self) -> int:
        return self.num_nodes

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(nodes={self.num_nodes}, edges={self.num_edges})"
        )

    # ------------------------------------------------------------------
    # Internal accessor (used by algorithm modules)
    # ------------------------------------------------------------------

    @property
    def _nx(self) -> nx.DiGraph:
        """Expose the underlying NetworkX graph for algorithm use."""
        return self._graph
