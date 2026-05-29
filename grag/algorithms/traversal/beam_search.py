"""Beam-search retrieval over a KnowledgeGraph."""

from __future__ import annotations

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

from grag.core.graph import KnowledgeGraph


class BeamSearchRetriever:
    """Graph beam-search for path-based retrieval.

    Starting from a set of seed nodes, expands the beam by following edges and
    scoring each newly reached node with a user-supplied scoring function (or a
    default degree-based heuristic).  Returns the *k* best terminal nodes along
    with the paths that lead to them.
    """

    def __init__(self, beam_width: int = 5, max_depth: int = 4) -> None:
        """
        Parameters
        ----------
        beam_width:
            Number of candidates kept at each depth level.
        max_depth:
            Maximum number of hops to explore.
        """
        self._beam_width = beam_width
        self._max_depth = max_depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        graph: KnowledgeGraph,
        start_nodes: List[str],
        beam_width: int = 3,
        max_depth: int = 4,
        score_fn: Optional[Callable[[str, KnowledgeGraph], float]] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """Run beam search from *start_nodes* and return scored paths.

        Parameters
        ----------
        graph:
            The knowledge graph to traverse.
        start_nodes:
            Starting node IDs (seed nodes).
        beam_width:
            Number of candidate paths to keep at each depth level.
        max_depth:
            Maximum number of hops to explore from any seed node.
        score_fn:
            A callable ``(node_id, graph) -> float`` used to score each node
            during expansion.  When None a default heuristic based on node
            degree is used.
        direction:
            Edge direction to follow: ``'out'``, ``'in'``, or ``'both'``.

        Returns
        -------
        List of result dicts, each with keys: ``node``, ``path``, ``score``.
        Sorted descending by ``score``.
        """
        effective_beam_width = beam_width if beam_width > 0 else self._beam_width
        effective_max_depth = max_depth if max_depth > 0 else self._max_depth

        if score_fn is None:
            score_fn = self._default_score

        # Beam state: (neg_score, path)  — min-heap, so negate score
        beam: List[Tuple[float, List[str]]] = []
        for seed in start_nodes:
            if seed in graph:
                s = score_fn(seed, graph)
                heapq.heappush(beam, (-s, [seed]))

        visited_paths: List[Tuple[float, List[str]]] = []
        seen_nodes: set = set(start_nodes)

        for _ in range(effective_max_depth):
            if not beam:
                break
            candidates: List[Tuple[float, List[str]]] = []
            while beam:
                neg_score, path = heapq.heappop(beam)
                current = path[-1]
                neighbours = graph.get_neighbors(current, direction=direction)
                for nb in neighbours:
                    if nb in seen_nodes:
                        continue
                    new_path = path + [nb]
                    nb_score = score_fn(nb, graph)
                    candidates.append((-nb_score, new_path))
                    seen_nodes.add(nb)
                visited_paths.append((neg_score, path))

            # Keep only the best beam_width candidates
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:effective_beam_width]

        # Collect remaining beam entries
        visited_paths.extend(beam)

        results = []
        for neg_score, path in visited_paths:
            results.append(
                {"node": path[-1], "path": path, "score": -neg_score}
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:effective_beam_width]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_score(node_id: str, graph: KnowledgeGraph) -> float:
        """Degree-based scoring heuristic."""
        return float(len(graph.get_neighbors(node_id, direction="both")))
