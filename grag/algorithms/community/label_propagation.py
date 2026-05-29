"""Label Propagation community detection for KnowledgeGraph instances."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import networkx as nx

from grag.core.graph import KnowledgeGraph


class LabelPropagationDetector:
    """Synchronous weighted Label Propagation Algorithm (LPA) for community detection.

    Each node is initialised with a unique label.  In every iteration each
    node adopts the label that is most common among its neighbours (ties are
    broken randomly).  The algorithm converges when no node changes its label.

    Edge weights are taken into account: when tallying neighbour labels the
    weight of each edge is used as the vote strength for that neighbour's label.
    """

    def __init__(self, random_state: int = 42, max_iterations: int = 100) -> None:
        self._rng = random.Random(random_state)
        self._max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, graph: KnowledgeGraph) -> Dict[str, int]:
        """Run Label Propagation and return a node -> community_id mapping.

        Parameters
        ----------
        graph:
            The knowledge graph to partition.

        Returns
        -------
        A dict mapping each node ID (str) to an integer community ID.
        """
        G = graph._nx.to_undirected()
        nodes = list(G.nodes())
        if not nodes:
            return {}

        # Initialise: each node gets its own unique label
        labels: Dict[str, str] = {n: n for n in nodes}

        for iteration in range(self._max_iterations):
            shuffled = list(nodes)
            self._rng.shuffle(shuffled)
            changed = False

            for node in shuffled:
                new_label = self._dominant_neighbour_label(G, node, labels)
                if new_label != labels[node]:
                    labels[node] = new_label
                    changed = True

            if not changed:
                break

        return self._labels_to_communities(labels)

    def get_communities(self, graph: KnowledgeGraph) -> Dict[int, List[str]]:
        """Detect communities and return a community -> node_list mapping."""
        partition = self.detect(graph)
        result: Dict[int, List[str]] = defaultdict(list)
        for node, comm in partition.items():
            result[comm].append(node)
        return dict(result)

    def iterations_to_converge(self, graph: KnowledgeGraph) -> int:
        """Return the number of iterations required to reach convergence."""
        G = graph._nx.to_undirected()
        nodes = list(G.nodes())
        if not nodes:
            return 0

        labels: Dict[str, str] = {n: n for n in nodes}
        for iteration in range(self._max_iterations):
            shuffled = list(nodes)
            self._rng.shuffle(shuffled)
            changed = False
            for node in shuffled:
                new_label = self._dominant_neighbour_label(G, node, labels)
                if new_label != labels[node]:
                    labels[node] = new_label
                    changed = True
            if not changed:
                return iteration + 1
        return self._max_iterations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dominant_neighbour_label(
        self, G: nx.Graph, node: str, labels: Dict[str, str]
    ) -> str:
        """Return the weighted-majority label among *node*'s neighbours.

        If the node has no neighbours its current label is preserved.
        Ties are broken by random choice.
        """
        neighbours = list(G.neighbors(node))
        if not neighbours:
            return labels[node]

        # Weighted vote tally
        tally: Dict[str, float] = defaultdict(float)
        for nb in neighbours:
            w = (G.get_edge_data(node, nb) or {}).get("weight", 1.0)
            tally[labels[nb]] += w

        max_score = max(tally.values())
        candidates = [lbl for lbl, score in tally.items() if score == max_score]
        return self._rng.choice(candidates)

    @staticmethod
    def _labels_to_communities(labels: Dict[str, str]) -> Dict[str, int]:
        """Map string labels to consecutive integer community IDs."""
        label_to_id: Dict[str, int] = {}
        counter = 0
        result: Dict[str, int] = {}
        for node, lbl in labels.items():
            if lbl not in label_to_id:
                label_to_id[lbl] = counter
                counter += 1
            result[node] = label_to_id[lbl]
        return result
