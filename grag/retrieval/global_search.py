"""Global search retriever using community-level map-reduce (Microsoft GraphRAG style).

The algorithm:
1. Rate relevance of each community summary to the query using embedding similarity.
2. Map phase: for each top-k community, extract key points relevant to the query.
3. Reduce phase: aggregate ranked points into a coherent synthesised answer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.embeddings import MockEmbedder


class GlobalSearchRetriever:
    """Community-level map-reduce search (as in Microsoft GraphRAG).

    1. Rate relevance of each community summary to query.
    2. Map: score each community, extract relevant points.
    3. Reduce: aggregate points into a coherent answer.

    Parameters
    ----------
    graph:
        The knowledge graph that has been community-annotated.
    community_summaries:
        Mapping of community_id -> summary text produced by the indexing pipeline.
    embedder:
        Embedding model used for semantic scoring.  Defaults to a fresh
        :class:`~grag.core.embeddings.MockEmbedder`.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        community_summaries: Dict[int, str],
        embedder: Optional[MockEmbedder] = None,
    ) -> None:
        self._graph = graph
        self._community_summaries = community_summaries
        self._embedder: MockEmbedder = embedder or MockEmbedder()
        # Pre-compute community summary embeddings
        self._summary_ids: List[int] = list(community_summaries.keys())
        self._summary_texts: List[str] = [community_summaries[cid] for cid in self._summary_ids]
        if self._summary_texts:
            self._summary_embs: np.ndarray = self._embedder.embed_batch(self._summary_texts)
        else:
            self._summary_embs = np.empty((0, self._embedder.DIM), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k_communities: int = 5) -> Dict[str, Any]:
        """Run global map-reduce search.

        Parameters
        ----------
        query:
            Natural language query string.
        top_k_communities:
            Number of highest-scoring communities to include in the map phase.

        Returns
        -------
        A dict with keys:
        ``answer`` – synthesised string answer,
        ``points`` – ranked list of extracted key points,
        ``community_scores`` – dict mapping community_id to relevance score,
        ``map_results`` – per-community extraction results.
        """
        if not self._summary_ids:
            return {
                "answer": "No community summaries available.",
                "points": [],
                "community_scores": {},
                "map_results": [],
            }

        # Step 1 – score community summaries against the query
        community_scores = self._score_communities(query)

        # Step 2 – select top-k communities
        sorted_communities = sorted(community_scores.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_communities[:top_k_communities]

        # Step 3 – MAP: extract relevant points from each selected community
        map_results = self._map_phase(query, selected)

        # Step 4 – REDUCE: aggregate and synthesise
        points = self._reduce_phase(query, map_results)

        answer = self._synthesise_answer(query, points)

        return {
            "answer": answer,
            "points": points,
            "community_scores": community_scores,
            "map_results": map_results,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_communities(self, query: str) -> Dict[int, float]:
        """Compute semantic relevance of each community summary to *query*."""
        if self._summary_embs.shape[0] == 0:
            return {}
        query_emb = self._embedder.embed(query)
        scores: Dict[int, float] = {}
        for i, comm_id in enumerate(self._summary_ids):
            score = self._embedder.cosine_similarity(query_emb, self._summary_embs[i])
            # Normalise to [0, 1] from cosine range [-1, 1]
            scores[comm_id] = (score + 1.0) / 2.0
        return scores

    def _map_phase(
        self,
        query: str,
        selected_communities: List[Tuple[int, float]],
    ) -> List[Dict[str, Any]]:
        """Extract key points from each selected community.

        For each community, sentence-level relevance is scored and the most
        relevant sentences are kept as points.
        """
        query_emb = self._embedder.embed(query)
        results: List[Dict[str, Any]] = []

        for comm_id, comm_score in selected_communities:
            summary = self._community_summaries.get(comm_id, "")
            if not summary:
                continue

            # Split into sentences / clauses for finer granularity
            sentences = self._split_sentences(summary)
            if not sentences:
                continue

            sent_embs = self._embedder.embed_batch(sentences)
            sent_scores: List[float] = []
            for i in range(len(sentences)):
                s = self._embedder.cosine_similarity(query_emb, sent_embs[i])
                sent_scores.append((s + 1.0) / 2.0)

            # Build ranked points for this community
            ranked_sentences = sorted(
                zip(sentences, sent_scores), key=lambda x: x[1], reverse=True
            )

            # Collect member nodes to provide entity context
            member_nodes = [
                n for n in self._graph.nodes
                if self._graph.get_community(n) == comm_id
            ]

            results.append(
                {
                    "community_id": comm_id,
                    "community_score": comm_score,
                    "points": [
                        {"text": sent, "score": sc}
                        for sent, sc in ranked_sentences
                    ],
                    "member_count": len(member_nodes),
                    "member_nodes": member_nodes[:10],  # cap for readability
                }
            )

        return results

    def _reduce_phase(
        self, query: str, map_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Aggregate points from all communities, ranked globally by score.

        Points are deduplicated by exact text and re-ranked using a combined
        score that weights both sentence relevance and parent community score.
        """
        seen_texts: set = set()
        all_points: List[Dict[str, Any]] = []

        for community_result in map_results:
            comm_score = community_result["community_score"]
            for point in community_result["points"]:
                text = point["text"].strip()
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                # Combined score: geometric mean of sentence and community scores
                combined = math.sqrt(point["score"] * comm_score)
                all_points.append(
                    {
                        "text": text,
                        "score": combined,
                        "sentence_score": point["score"],
                        "community_id": community_result["community_id"],
                        "community_score": comm_score,
                    }
                )

        all_points.sort(key=lambda x: x["score"], reverse=True)
        return all_points

    def _synthesise_answer(self, query: str, points: List[Dict[str, Any]]) -> str:
        """Produce a human-readable synthesised answer from ranked points.

        In a production system this would call an LLM.  Here we build a
        deterministic summary from the highest-scoring points.
        """
        if not points:
            return f"No relevant information found for query: '{query}'."

        top_points = points[:5]
        num_communities = len({p["community_id"] for p in points})

        lines: List[str] = [
            f"Based on analysis of {num_communities} community cluster(s), "
            f"the following key findings are relevant to '{query}':",
            "",
        ]
        for i, point in enumerate(top_points, 1):
            lines.append(f"{i}. {point['text']} (relevance: {point['score']:.3f})")

        if len(points) > 5:
            lines.append(f"\n[{len(points) - 5} additional supporting point(s) available.]")

        return "\n".join(lines)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Naively split text into sentences by common punctuation."""
        import re
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 10]
