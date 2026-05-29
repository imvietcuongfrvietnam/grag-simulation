"""G-Retriever simulation.

Based on:
  He, X. et al. (2024). "G-Retriever: Retrieval-Augmented Generation for
  Textual Graph Understanding and Question Answering."

G-Retriever uses a Prize-Collecting Steiner Tree (PCST) algorithm to find the
minimal connected subgraph of a KG that answers a query.  Each node and edge
has a prize (relevance score) and a cost; PCST finds the highest net-value
connected subgraph.

All LLM calls are replaced by deterministic mock responses.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from grag.core.document import SimpleEntityExtractor, TextChunk
from grag.core.embeddings import MockEmbedder
from grag.core.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PCSTNode:
    """Internal node representation for PCST."""

    node_id: str
    prize: float
    cost: float = 0.0  # node activation cost (usually 0 for nodes)
    selected: bool = False


@dataclass
class PCSTEdge:
    """Internal edge representation for PCST."""

    src: str
    dst: str
    cost: float
    selected: bool = False


# ---------------------------------------------------------------------------
# GRetriever
# ---------------------------------------------------------------------------


class GRetriever:
    """G-Retriever: GNN-based subgraph retrieval via PCST.

    Architecture
    ------------
    - Index: build KG from documents, embed nodes and edges.
    - Retrieve: assign prizes to nodes based on query similarity, run PCST
      approximation to find the minimal subgraph that covers the query.

    The PCST approximation used here is a greedy prize-weighted Steiner tree
    heuristic:
      1. Score each node with a relevance prize.
      2. Sort by prize descending; greedily add nodes while connecting them
         to the current tree via shortest paths, subject to cost budget.

    Parameters
    ----------
    prize_weight:
        Multiplier applied to query-similarity scores to derive node prizes.
    cost_weight:
        Multiplier applied to edge weights to derive traversal costs (lower
        cost = more likely to include the edge).
    """

    def __init__(self, prize_weight: float = 1.0, cost_weight: float = 0.1) -> None:
        self.prize_weight = prize_weight
        self.cost_weight = cost_weight

        self._extractor = SimpleEntityExtractor()
        self._embedder = MockEmbedder(dim=128)

        self._graph: KnowledgeGraph = KnowledgeGraph()
        self._node_embeddings: Dict[str, np.ndarray] = {}
        self._passages: List[str] = []
        self._passage_embeddings: Optional[np.ndarray] = None
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        """Build the KG index with node embeddings.

        Parameters
        ----------
        documents:
            Raw text strings to index.
        """
        self._graph = KnowledgeGraph()
        self._node_embeddings = {}
        self._passages = list(documents)

        for doc_idx, doc_text in enumerate(documents):
            entities = self._extractor.extract_entities(doc_text)
            relations = self._extractor.extract_relations(doc_text, entities)

            for ent in entities:
                self._graph.add_entity(
                    id=ent["id"],
                    label=ent["text"],
                    type=ent["type"],
                    properties={"source_doc": doc_idx},
                )

            for rel in relations:
                for nid in (rel["src"], rel["dst"]):
                    if nid not in self._graph:
                        self._graph.add_entity(nid, label=nid, type="CONCEPT")
                self._graph.add_relation(
                    src=rel["src"],
                    dst=rel["dst"],
                    relation_type=rel["relation"],
                    weight=1.0,
                    properties={"source_doc": doc_idx},
                )

        # Embed all nodes by their label
        for node_id in self._graph.nodes:
            label = self._graph.get_entity(node_id).get("label", node_id)
            self._node_embeddings[node_id] = self._embedder.embed(label)

        # Embed passages
        self._passage_embeddings = self._embedder.embed_batch(documents)
        self._indexed = True

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Find the minimal subgraph answering *query* via PCST.

        Parameters
        ----------
        query:
            Natural language query string.

        Returns
        -------
        Dict with keys:
          ``answer`` – mock generated answer.
          ``subgraph_nodes`` – node IDs in the PCST subgraph.
          ``subgraph_edges`` – edges in the PCST subgraph as (src, dst, type) tuples.
          ``node_prizes`` – per-node relevance prizes.
          ``total_prize`` – total prize collected.
          ``total_cost`` – total edge cost incurred.
        """
        self._require_index()

        if self._graph.num_nodes == 0:
            return {"answer": "No entities indexed.", "subgraph_nodes": [],
                    "subgraph_edges": [], "node_prizes": {}, "total_prize": 0.0,
                    "total_cost": 0.0}

        query_emb = self._embedder.embed(query)

        # Assign prizes to all nodes
        prizes: Dict[str, float] = self._compute_prizes(query_emb)

        # Assign costs to all edges
        costs: Dict[Tuple[str, str], float] = self._compute_edge_costs()

        # Run PCST approximation
        selected_nodes, selected_edges = self._pcst_approx(self._graph, prizes, costs)

        # Build result subgraph
        subgraph = self._graph.subgraph(selected_nodes)

        # Edge details for output
        edge_details = [
            {
                "src": src,
                "dst": dst,
                "relation": self._graph.get_relation(src, dst).get("relation_type", "?"),
            }
            for src, dst in selected_edges
            if self._graph._nx.has_edge(src, dst)
        ]

        total_prize = sum(prizes.get(n, 0.0) for n in selected_nodes)
        total_cost = sum(costs.get((s, d), 0.0) for s, d in selected_edges)

        answer = self._mock_answer(query, selected_nodes, edge_details, prizes)

        return {
            "answer": answer,
            "subgraph_nodes": selected_nodes,
            "subgraph_edges": edge_details,
            "subgraph_size": len(selected_nodes),
            "node_prizes": {n: round(prizes.get(n, 0.0), 4) for n in selected_nodes},
            "total_prize": round(total_prize, 4),
            "total_cost": round(total_cost, 4),
            "pcst": {
                "total_prize": round(total_prize, 4),
                "total_cost": round(total_cost, 4),
                "net_value": round(total_prize - total_cost, 4),
            },
        }

    def _pcst_approx(
        self,
        graph: KnowledgeGraph,
        prizes: Dict[str, float],
        costs: Dict[Tuple[str, str], float],
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Approximate Prize-Collecting Steiner Tree.

        Greedy PCST heuristic:
        1. Sort nodes by prize descending.
        2. Maintain a growing "tree" set of selected nodes.
        3. For each candidate node, find its cheapest connection path to the
           current tree using Dijkstra over edge costs.
        4. Accept the node+path if prize(node) > cost(path).
        5. Stop when no remaining node improves the objective.

        Parameters
        ----------
        graph:
            The full knowledge graph.
        prizes:
            Node ID -> relevance prize (non-negative float).
        costs:
            (src, dst) -> traversal cost (non-negative float).

        Returns
        -------
        Tuple of (selected_node_ids, selected_edge_pairs).
        """
        if graph.num_nodes == 0:
            return [], []

        # Seed with highest-prize node
        sorted_nodes = sorted(prizes.keys(), key=lambda n: prizes[n], reverse=True)
        if not sorted_nodes:
            return [], []

        tree_nodes: Set[str] = {sorted_nodes[0]}
        tree_edges: Set[Tuple[str, str]] = set()

        for candidate in sorted_nodes[1:]:
            if candidate in tree_nodes:
                continue
            prize = prizes.get(candidate, 0.0)
            if prize <= 0.0:
                break  # no more positive-prize nodes

            # Find cheapest path from candidate to any node in tree_nodes
            path_cost, path = self._cheapest_path(graph, candidate, tree_nodes, costs)

            net_gain = prize - path_cost
            if net_gain > 0 or len(tree_nodes) == 1:
                # Accept: add candidate and the connecting path
                tree_nodes.add(candidate)
                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    tree_nodes.add(src)
                    tree_nodes.add(dst)
                    # Store edge in canonical form (check actual graph direction)
                    if graph._nx.has_edge(src, dst):
                        tree_edges.add((src, dst))
                    elif graph._nx.has_edge(dst, src):
                        tree_edges.add((dst, src))

        return list(tree_nodes), list(tree_edges)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_prizes(self, query_emb: np.ndarray) -> Dict[str, float]:
        """Compute node prizes as scaled cosine similarity to the query."""
        prizes: Dict[str, float] = {}
        for node_id in self._graph.nodes:
            node_emb = self._node_embeddings.get(node_id)
            if node_emb is None:
                prizes[node_id] = 0.0
                continue
            sim = MockEmbedder.cosine_similarity(query_emb, node_emb)
            # Normalise to [0, 1] and apply prize weight
            prizes[node_id] = max(0.0, (sim + 1.0) / 2.0) * self.prize_weight
        return prizes

    def _compute_edge_costs(self) -> Dict[Tuple[str, str], float]:
        """Compute edge costs as (1 / weight) * cost_weight."""
        costs: Dict[Tuple[str, str], float] = {}
        for src, dst in self._graph.edges:
            rel = self._graph.get_relation(src, dst)
            w = max(rel.get("weight", 1.0), 1e-6)
            costs[(src, dst)] = (1.0 / w) * self.cost_weight
        return costs

    def _cheapest_path(
        self,
        graph: KnowledgeGraph,
        start: str,
        targets: Set[str],
        edge_costs: Dict[Tuple[str, str], float],
    ) -> Tuple[float, List[str]]:
        """Dijkstra from *start* to any node in *targets*.

        Returns (total_cost, path).  If no path exists, returns (inf, []).
        """
        if start in targets:
            return 0.0, [start]

        # Dijkstra
        dist: Dict[str, float] = defaultdict(lambda: float("inf"))
        prev: Dict[str, Optional[str]] = {}
        dist[start] = 0.0
        heap: List[Tuple[float, str]] = [(0.0, start)]

        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            if node in targets:
                # Reconstruct path
                path: List[str] = []
                cur: Optional[str] = node
                while cur is not None:
                    path.append(cur)
                    cur = prev.get(cur)
                path.reverse()
                return dist[node], path

            for nb in graph.get_neighbors(node, direction="both"):
                # Use undirected cost (try both directions)
                c = edge_costs.get((node, nb), edge_costs.get((nb, node), self.cost_weight))
                new_dist = dist[node] + c
                if new_dist < dist[nb]:
                    dist[nb] = new_dist
                    prev[nb] = node
                    heapq.heappush(heap, (new_dist, nb))

        return float("inf"), []

    def _mock_answer(
        self,
        query: str,
        selected_nodes: List[str],
        edge_details: List[Dict[str, Any]],
        prizes: Dict[str, float],
    ) -> str:
        """Generate a mock answer from the PCST subgraph."""
        if not selected_nodes:
            return f"PCST found no relevant subgraph for: {query}"

        # Get labels for top-prize nodes
        top_nodes_by_prize = sorted(
            selected_nodes, key=lambda n: prizes.get(n, 0.0), reverse=True
        )[:4]
        labels = [
            self._graph.get_entity(n).get("label", n) for n in top_nodes_by_prize
        ]
        label_str = ", ".join(labels)

        edge_types = list(dict.fromkeys(e["relation"] for e in edge_details[:5]))
        edge_type_str = ", ".join(edge_types) if edge_types else "various relations"

        return (
            f"G-Retriever answer for '{query}': PCST identified a minimal subgraph "
            f"with {len(selected_nodes)} nodes and {len(edge_details)} edges. "
            f"Key entities: {label_str}. "
            f"Connecting relations: {edge_type_str}. "
            f"This subgraph represents the minimum-cost connected structure "
            f"that maximally covers the query's relevant entities."
        )

    def _require_index(self) -> None:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")
