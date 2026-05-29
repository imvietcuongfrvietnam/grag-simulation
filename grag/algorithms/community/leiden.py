"""Leiden community detection for KnowledgeGraph instances.

Implements a simplified Leiden algorithm following Traag et al. (2019).
The Leiden algorithm improves on Louvain by guaranteeing well-connected
communities through a refinement phase between the local move and aggregation
phases.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx

from grag.core.graph import KnowledgeGraph


class LeidenDetector:
    """Leiden community detection algorithm.

    Phases per iteration:
    1. Local node movement (same greedy optimisation as Louvain).
    2. Refinement – sub-communities within each community are split to ensure
       each community is internally well-connected.
    3. Aggregation – communities are collapsed into super-nodes.

    Returns a flat node -> community_id mapping for the original nodes.
    """

    def __init__(self, random_state: int = 42, theta: float = 0.01) -> None:
        """
        Parameters
        ----------
        random_state:
            Seed for the internal RNG so results are reproducible.
        theta:
            Refinement randomness parameter.  Larger values allow more random
            splits during the refinement phase, improving community quality at
            the cost of determinism.
        """
        self._rng = random.Random(random_state)
        self._theta = theta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        graph: KnowledgeGraph,
        resolution: float = 1.0,
        max_iterations: int = 10,
    ) -> Dict[str, int]:
        """Run Leiden and return a node -> community_id dict.

        Parameters
        ----------
        graph:
            The knowledge graph to partition.
        resolution:
            Resolution parameter (gamma).  Higher values produce finer-grained
            communities.
        max_iterations:
            Maximum number of full Leiden passes (local move + refine +
            aggregate).

        Returns
        -------
        Mapping from node ID (str) to integer community ID.
        """
        G = graph._nx.to_undirected()
        nodes = list(G.nodes())
        if not nodes:
            return {}

        # Flat partition: node -> community
        partition: Dict[str, int] = {n: i for i, n in enumerate(nodes)}

        total_weight = (
            sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True)) or 1.0
        )

        for _ in range(max_iterations):
            moved = self._local_move(G, partition, total_weight, resolution)
            self._refine(G, partition, total_weight, resolution)
            if not moved:
                break

        return self._normalise(partition)

    def get_communities(self, graph: KnowledgeGraph) -> Dict[int, List[str]]:
        """Detect communities and return a community -> node_list mapping."""
        partition = self.detect(graph)
        result: Dict[int, List[str]] = defaultdict(list)
        for node, comm in partition.items():
            result[comm].append(node)
        return dict(result)

    def community_count(self, graph: KnowledgeGraph) -> int:
        """Return the number of communities detected."""
        return len(set(self.detect(graph).values()))

    # ------------------------------------------------------------------
    # Public phase aliases (match the specification interface)
    # ------------------------------------------------------------------

    def _move_nodes_fast(
        self,
        graph: KnowledgeGraph,
        partition: Dict[str, int],
    ) -> bool:
        """Public alias for _local_move that accepts a KnowledgeGraph directly.

        Returns True if at least one node changed community.
        """
        G = graph._nx.to_undirected()
        total_weight = (
            sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True)) or 1.0
        )
        return self._local_move(G, partition, total_weight, resolution=1.0)

    def _refine_partition(
        self,
        graph: KnowledgeGraph,
        partition: Dict[str, int],
    ) -> Dict[str, int]:
        """Public alias for _refine that accepts a KnowledgeGraph and returns the partition."""
        G = graph._nx.to_undirected()
        total_weight = (
            sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True)) or 1.0
        )
        self._refine(G, partition, total_weight, resolution=1.0)
        return self._normalise(partition)

    def _aggregate_graph(
        self,
        graph: KnowledgeGraph,
        partition: Dict[str, int],
    ) -> "KnowledgeGraph":
        """Collapse each community into a single super-node.

        Returns a new KnowledgeGraph where each node represents one community
        and edge weights are the sum of inter-community edge weights in the
        original graph.
        """
        from collections import defaultdict as _dd

        G = graph._nx.to_undirected()
        agg = KnowledgeGraph()

        for comm_id in set(partition.values()):
            agg.add_entity(
                id=str(comm_id),
                label=f"community_{comm_id}",
                type="COMMUNITY",
            )

        inter_weights: Dict[str, float] = _dd(float)
        for u, v, d in G.edges(data=True):
            cu = str(partition.get(u, u))
            cv = str(partition.get(v, v))
            if cu != cv:
                key = f"{min(cu, cv)}||{max(cu, cv)}"
                inter_weights[key] += d.get("weight", 1.0)

        for key, w in inter_weights.items():
            cu, cv = key.split("||")
            agg.add_relation(cu, cv, relation_type="INTER_COMMUNITY", weight=w)

        return agg

    # ------------------------------------------------------------------
    # Phase 1 – Local node movement
    # ------------------------------------------------------------------

    def _local_move(
        self,
        G: nx.Graph,
        partition: Dict[str, int],
        total_weight: float,
        resolution: float,
    ) -> bool:
        """Greedy local node movement; return True if any node changed community."""
        moved = False
        nodes = list(G.nodes())
        self._rng.shuffle(nodes)

        # Community -> sum of internal edge weights
        comm_weight: Dict[int, float] = defaultdict(float)
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1.0)
            comm_weight[partition[u]] += w
            if u != v:
                comm_weight[partition[v]] += w

        for node in nodes:
            current_comm = partition[node]
            node_deg = sum(d.get("weight", 1.0) for _, _, d in G.edges(node, data=True))

            # Accumulate weights to each neighbouring community
            nb_comm_w: Dict[int, float] = defaultdict(float)
            for nb in G.neighbors(node):
                nb_comm_w[partition[nb]] += G.get_edge_data(node, nb, {}).get("weight", 1.0)

            comm_weight[current_comm] -= node_deg

            best_comm = current_comm
            best_gain = 0.0
            for cand_comm, k_in in nb_comm_w.items():
                if cand_comm == current_comm:
                    continue
                gain = (
                    k_in / total_weight
                    - resolution * node_deg * comm_weight[cand_comm] / (2.0 * total_weight ** 2)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_comm = cand_comm

            partition[node] = best_comm
            comm_weight[best_comm] += node_deg
            if best_comm != current_comm:
                moved = True

        return moved

    # ------------------------------------------------------------------
    # Phase 2 – Refinement
    # ------------------------------------------------------------------

    def _refine(
        self,
        G: nx.Graph,
        partition: Dict[str, int],
        total_weight: float,
        resolution: float,
    ) -> None:
        """Refine communities: attempt to split poorly-connected sub-communities.

        For each community we compute an internal sub-partition and accept
        splits that improve the CPM quality function with probability
        proportional to exp(delta / theta).
        """
        # Group nodes by current community
        comm_nodes: Dict[int, List[str]] = defaultdict(list)
        for node, comm in partition.items():
            comm_nodes[comm].append(node)

        for comm_id, members in comm_nodes.items():
            if len(members) < 3:
                continue  # nothing to split

            sub_G = G.subgraph(members)
            sub_partition: Dict[str, int] = {n: i for i, n in enumerate(members)}
            sub_total = (
                sum(d.get("weight", 1.0) for _, _, d in sub_G.edges(data=True)) or 1.0
            )

            # One round of local moves within the sub-graph
            self._local_move(sub_G, sub_partition, sub_total, resolution)

            # Accept the split only if it creates at least 2 distinct sub-communities
            sub_comms = set(sub_partition.values())
            if len(sub_comms) < 2:
                continue

            # Leiden acceptance: probabilistic merge-back with theta
            delta = self._cpm_delta(sub_G, sub_partition, resolution)
            import math
            accept_prob = min(1.0, math.exp(delta / max(self._theta, 1e-9)))
            if self._rng.random() > accept_prob:
                continue

            # Re-label sub-partition within the global partition namespace
            max_global_comm = max(partition.values()) + 1
            for node, sub_c in sub_partition.items():
                partition[node] = max_global_comm + sub_c

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cpm_delta(
        G: nx.Graph, partition: Dict[str, int], resolution: float
    ) -> float:
        """Compute the CPM (Constant Potts Model) quality delta for a partition."""
        comm_edges: Dict[int, float] = defaultdict(float)
        comm_sizes: Dict[int, int] = defaultdict(int)
        for n, c in partition.items():
            comm_sizes[c] += 1
        for u, v, d in G.edges(data=True):
            if partition[u] == partition[v]:
                comm_edges[partition[u]] += d.get("weight", 1.0)
        q = 0.0
        for c in set(partition.values()):
            n = comm_sizes[c]
            q += comm_edges[c] - resolution * n * (n - 1) / 2.0
        return q

    @staticmethod
    def _normalise(partition: Dict[str, int]) -> Dict[str, int]:
        remap: Dict[int, int] = {}
        counter = 0
        result: Dict[str, int] = {}
        for node, comm in partition.items():
            if comm not in remap:
                remap[comm] = counter
                counter += 1
            result[node] = remap[comm]
        return result
