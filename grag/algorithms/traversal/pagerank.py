"""Personalized PageRank for KnowledgeGraph traversal."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from grag.core.graph import KnowledgeGraph


class PersonalizedPageRank:
    """Personalized PageRank (PPR) on a KnowledgeGraph.

    Runs the power-iteration PPR algorithm seeded from a set of source nodes,
    which is the core retrieval mechanism used by HippoRAG and related
    graph-based retrieval systems.

    The computation is performed on the *undirected* projection of the graph so
    that relevance flows in both directions along every edge.
    """

    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> None:
        """
        Parameters
        ----------
        alpha:
            Damping factor (teleport probability = 1 - alpha).
        max_iter:
            Maximum power-iteration steps before forced convergence.
        tol:
            L1 convergence tolerance.
        """
        self._alpha = alpha
        self._max_iter = max_iter
        self._tol = tol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        graph: KnowledgeGraph,
        seed_nodes: List[str],
        seed_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute PPR scores for all nodes, seeded from *seed_nodes*.

        Parameters
        ----------
        graph:
            The knowledge graph to run PPR on.
        seed_nodes:
            List of node IDs to use as teleport targets.
        seed_weights:
            Optional per-seed teleport weights (will be L1-normalised).
            If None, uniform weights are used.

        Returns
        -------
        A dict mapping every node ID to its PPR score.
        """
        G = graph._nx.to_undirected()
        if G.number_of_nodes() == 0:
            return {}

        nodes = list(G.nodes())
        n = len(nodes)
        node_idx: Dict[str, int] = {node: i for i, node in enumerate(nodes)}

        # Build personalisation vector
        teleport = np.zeros(n, dtype=np.float64)
        for s in seed_nodes:
            if s in node_idx:
                w = (seed_weights or {}).get(s, 1.0)
                teleport[node_idx[s]] += w
        if teleport.sum() == 0:
            teleport[:] = 1.0 / n
        else:
            teleport /= teleport.sum()

        # Build row-normalised adjacency matrix
        A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        A /= row_sums  # row-stochastic

        # Power iteration
        r = teleport.copy()
        for _ in range(self._max_iter):
            r_new = self._alpha * (A.T @ r) + (1.0 - self._alpha) * teleport
            if np.abs(r_new - r).sum() < self._tol:
                r = r_new
                break
            r = r_new

        return {nodes[i]: float(r[i]) for i in range(n)}

    def top_k(
        self,
        graph: KnowledgeGraph,
        seed_nodes: List[str],
        k: int = 10,
        exclude_seeds: bool = False,
    ) -> List[Tuple[str, float]]:
        """Return the top-*k* nodes by PPR score.

        Parameters
        ----------
        graph:
            The knowledge graph to rank.
        seed_nodes:
            Seed nodes for personalisation.
        k:
            Number of results to return.
        exclude_seeds:
            If True, seed nodes are removed from the returned ranking.

        Returns
        -------
        List of ``(node_id, score)`` tuples sorted descending by score.
        """
        scores = self.run(graph, seed_nodes)
        if exclude_seeds:
            seed_set = set(seed_nodes)
            scores = {n: s for n, s in scores.items() if n not in seed_set}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def subgraph_scores(
        self,
        graph: KnowledgeGraph,
        seed_nodes: List[str],
        score_threshold: float = 0.001,
    ) -> KnowledgeGraph:
        """Return a subgraph of nodes with PPR score above *score_threshold*."""
        scores = self.run(graph, seed_nodes)
        relevant = [n for n, s in scores.items() if s >= score_threshold]
        return graph.subgraph(relevant)

    # ------------------------------------------------------------------
    # Specification-compatible interface
    # ------------------------------------------------------------------

    def compute(
        self,
        graph: KnowledgeGraph,
        personalization: Optional[Dict[str, float]] = None,
        alpha: float = 0.85,
        max_iter: int = 100,
    ) -> Dict[str, float]:
        """Compute PageRank scores for all nodes.

        Parameters
        ----------
        graph:
            The knowledge graph to rank.
        personalization:
            Optional dict mapping node IDs to teleport weights.  When None,
            uniform PageRank (no personalisation) is computed.
        alpha:
            Damping factor.
        max_iter:
            Maximum power-iteration steps.

        Returns
        -------
        A dict mapping every node ID to its PageRank score.
        """
        saved_alpha, saved_max_iter = self._alpha, self._max_iter
        self._alpha = alpha
        self._max_iter = max_iter
        try:
            if personalization is None:
                seed_nodes = list(graph.nodes)
                scores = self.run(graph, seed_nodes)
            else:
                seed_nodes = list(personalization.keys())
                scores = self.run(graph, seed_nodes, seed_weights=personalization)
        finally:
            self._alpha = saved_alpha
            self._max_iter = saved_max_iter
        return scores

    def ppr_for_query(
        self,
        graph: KnowledgeGraph,
        query_nodes: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Run Personalized PageRank seeded from *query_nodes* and return top results.

        Parameters
        ----------
        graph:
            The knowledge graph to retrieve from.
        query_nodes:
            Node IDs corresponding to query concepts.
        top_k:
            Number of top-scoring nodes to return.

        Returns
        -------
        List of ``(node_id, score)`` tuples sorted descending by PPR score.
        """
        return self.top_k(graph, query_nodes, k=top_k)
