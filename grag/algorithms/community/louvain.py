"""Louvain community detection for KnowledgeGraph instances."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from grag.core.graph import KnowledgeGraph


class LouvainDetector:
    """Louvain method for community detection on a KnowledgeGraph.

    Implements two phases iteratively:
    1. Local optimisation – each node is greedily moved to the neighbouring
       community that maximises modularity gain.
    2. Aggregation – each community is collapsed into a single super-node and
       phase 1 is repeated on the aggregated graph.

    The result is a flat partition of the original nodes.
    """

    def __init__(self, random_state: int = 42) -> None:
        self._rng = random.Random(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        graph: KnowledgeGraph,
        resolution: float = 1.0,
    ) -> Dict[str, int]:
        """Run Louvain community detection and return a node->community mapping.

        Parameters
        ----------
        graph:
            The knowledge graph to partition.
        resolution:
            Resolution parameter (gamma).  Higher values favour smaller, more
            granular communities.

        Returns
        -------
        A dict mapping each node ID (str) to an integer community ID.
        """
        # Try to use python-louvain when available for better accuracy;
        # fall back to a self-contained greedy implementation otherwise.
        try:
            import community as community_louvain  # python-louvain

            undirected = graph._nx.to_undirected()
            if undirected.number_of_nodes() == 0:
                return {}
            partition = community_louvain.best_partition(
                undirected, resolution=resolution, random_state=self._rng.randint(0, 2**31)
            )
            return partition
        except ImportError:
            pass

        return self._greedy_louvain(graph, resolution)

    def get_communities(self, graph: KnowledgeGraph) -> Dict[int, List[str]]:
        """Detect communities and return a community->node_list mapping."""
        partition = self.detect(graph)
        result: Dict[int, List[str]] = defaultdict(list)
        for node, comm in partition.items():
            result[comm].append(node)
        return dict(result)

    def modularity(
        self, graph: KnowledgeGraph, communities: Dict[int, List[str]]
    ) -> float:
        """Compute the modularity Q of a given community assignment.

        Uses the standard Newman-Girvan formula on the undirected projection
        of *graph*.
        """
        undirected = graph._nx.to_undirected()
        if undirected.number_of_edges() == 0:
            return 0.0
        # Build a node -> community mapping for nx.algorithms.community
        node_to_comm: Dict[str, int] = {}
        for comm_id, members in communities.items():
            for n in members:
                node_to_comm[n] = comm_id
        community_sets = [set(members) for members in communities.values()]
        try:
            return nx.algorithms.community.modularity(undirected, community_sets)
        except Exception:
            return self._manual_modularity(undirected, node_to_comm)

    # ------------------------------------------------------------------
    # Internal greedy Louvain implementation
    # ------------------------------------------------------------------

    def _greedy_louvain(
        self, graph: KnowledgeGraph, resolution: float
    ) -> Dict[str, int]:
        """Self-contained greedy Louvain on the undirected projection."""
        G = graph._nx.to_undirected()
        nodes = list(G.nodes())
        if not nodes:
            return {}

        # Phase 0 – initialise each node to its own community
        partition: Dict[str, int] = {n: i for i, n in enumerate(nodes)}
        total_weight = sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True)) or 1.0

        improved = True
        while improved:
            improved = self._local_optimisation(G, partition, total_weight, resolution)

        # Normalise community IDs to consecutive integers
        return self._normalise_partition(partition)

    def _local_optimisation(
        self,
        G: nx.Graph,
        partition: Dict[str, int],
        total_weight: float,
        resolution: float,
    ) -> bool:
        """One pass of local optimisation.  Returns True if any node moved."""
        moved = False
        nodes = list(G.nodes())
        self._rng.shuffle(nodes)

        comm_weights: Dict[int, float] = defaultdict(float)
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1.0)
            comm_weights[partition[u]] += w
            if u != v:
                comm_weights[partition[v]] += w

        for node in nodes:
            current_comm = partition[node]
            node_degree = sum(
                d.get("weight", 1.0) for _, _, d in G.edges(node, data=True)
            )

            # Weights connecting node to each neighbouring community
            neighbour_comm_w: Dict[int, float] = defaultdict(float)
            for nb in G.neighbors(node):
                edge_data = G.get_edge_data(node, nb) or {}
                w = edge_data.get("weight", 1.0)
                neighbour_comm_w[partition[nb]] += w

            # Remove node from its current community
            comm_weights[current_comm] -= node_degree

            best_comm = current_comm
            best_gain = 0.0

            for cand_comm, k_in in neighbour_comm_w.items():
                if cand_comm == current_comm:
                    continue
                gain = (
                    k_in / total_weight
                    - resolution
                    * node_degree
                    * comm_weights[cand_comm]
                    / (2.0 * total_weight ** 2)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_comm = cand_comm

            partition[node] = best_comm
            comm_weights[best_comm] += node_degree
            if best_comm != current_comm:
                moved = True

        return moved

    @staticmethod
    def _normalise_partition(partition: Dict[str, int]) -> Dict[str, int]:
        remap: Dict[int, int] = {}
        counter = 0
        result: Dict[str, int] = {}
        for node, comm in partition.items():
            if comm not in remap:
                remap[comm] = counter
                counter += 1
            result[node] = remap[comm]
        return result

    @staticmethod
    def _manual_modularity(
        G: nx.Graph, node_to_comm: Dict[str, int]
    ) -> float:
        m = G.number_of_edges()
        if m == 0:
            return 0.0
        q = 0.0
        for u, v, d in G.edges(data=True):
            if node_to_comm.get(u) == node_to_comm.get(v):
                w = d.get("weight", 1.0)
                q += w - G.degree(u) * G.degree(v) / (2.0 * m)
        return q / (2.0 * m)
