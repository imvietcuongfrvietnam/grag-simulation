"""Hybrid search retriever combining graph traversal and dense vector search.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from:
1. Graph-based retrieval (Personalized PageRank from seed entities).
2. Dense vector search (cosine similarity over node label embeddings).

Reference:
  Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009).
  "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods."
  SIGIR 2009.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.embeddings import MockEmbedder
from grag.algorithms.traversal.pagerank import PersonalizedPageRank


class HybridSearchRetriever:
    """Combines graph traversal and dense vector search via Reciprocal Rank Fusion.

    Two independent ranked lists are produced for a query:
    1. **Graph list** – nodes ranked by Personalized PageRank, seeded from the
       entities whose labels are closest to the query embedding.
    2. **Vector list** – nodes ranked by cosine similarity of their label
       embedding to the query embedding.

    RRF merges these lists into a single ranking without requiring score
    calibration between the two systems.

    Parameters
    ----------
    graph:
        A populated :class:`~grag.core.graph.KnowledgeGraph`.
    embedder:
        Embedding model for semantic retrieval.  Defaults to a fresh
        :class:`~grag.core.embeddings.MockEmbedder`.
    alpha:
        Default weight given to the graph-based scores in the combined
        normalised-score fusion (0 = pure vector, 1 = pure graph).
        RRF itself is rank-based and does not use alpha; this is used only
        in the score-weighted variant ``_score_fusion``.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        embedder: Optional[MockEmbedder] = None,
        alpha: float = 0.5,
    ) -> None:
        self._graph = graph
        self._embedder: MockEmbedder = embedder or MockEmbedder()
        self._alpha = alpha
        self._ppr = PersonalizedPageRank(alpha=0.85)

        # Pre-build node label index
        self._node_ids: List[str] = list(graph.nodes)
        self._node_labels: List[str] = [
            graph.get_entity(n).get("label", n) for n in self._node_ids
        ]

        # Pre-compute node label embeddings
        if self._node_labels:
            self._node_embs: np.ndarray = self._embedder.embed_batch(self._node_labels)
        else:
            self._node_embs = np.empty((0, self._embedder.DIM), dtype=np.float32)

        # Build a fast label -> node_id lookup
        self._label_to_id: Dict[str, str] = {
            label: nid for label, nid in zip(self._node_labels, self._node_ids)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        alpha: Optional[float] = None,
        top_k: int = 10,
        seed_k: int = 5,
    ) -> Dict[str, Any]:
        """Run hybrid search and return fused results.

        Parameters
        ----------
        query:
            Natural language query string.
        alpha:
            Weight for graph vs vector score (overrides constructor default).
            ``alpha=1.0`` gives full graph weight; ``alpha=0.0`` gives full
            vector weight.  Used only in the normalised-score variant; RRF
            results are always returned under ``fused_ranking``.
        top_k:
            Maximum number of results to return.
        seed_k:
            Number of seed entities for PPR initialisation.

        Returns
        -------
        Dict with keys:
            ``answer`` – synthesised string answer,
            ``fused_ranking`` – list of entity dicts ranked by RRF,
            ``graph_ranking`` – list of entity dicts from PPR alone,
            ``vector_ranking`` – list of entity dicts from vector search alone,
            ``rrf_scores`` – raw RRF scores per node_id,
            ``alpha_used`` – the alpha value applied.
        """
        if not self._node_ids:
            return {
                "answer": "Knowledge graph is empty.",
                "fused_ranking": [],
                "graph_ranking": [],
                "vector_ranking": [],
                "rrf_scores": {},
                "alpha_used": alpha if alpha is not None else self._alpha,
            }

        effective_alpha = alpha if alpha is not None else self._alpha

        # ---- Vector search ----
        vector_results = self._vector_search(query, top_k=max(top_k * 2, 20))

        # ---- Graph search ----
        # Seed PPR from the top-seed_k vector-closest entities
        seed_ids = [r["id"] for r in vector_results[:seed_k]]
        graph_results = self._graph_search(query, seed_ids, top_k=max(top_k * 2, 20))

        # ---- RRF fusion ----
        fused_ids, rrf_scores = self._rrf_fusion(
            graph_results, vector_results, k=60
        )

        # ---- Build output entity list ----
        fused_ranking = self._build_entity_list(fused_ids[:top_k], rrf_scores)
        graph_ranking_out = graph_results[:top_k]
        vector_ranking_out = vector_results[:top_k]

        answer = self._synthesise_answer(query, fused_ranking, effective_alpha)

        return {
            "answer": answer,
            "fused_ranking": fused_ranking,
            "graph_ranking": graph_ranking_out,
            "vector_ranking": vector_ranking_out,
            "rrf_scores": {nid: round(rrf_scores.get(nid, 0.0), 6) for nid in fused_ids[:top_k]},
            "alpha_used": effective_alpha,
        }

    # ------------------------------------------------------------------
    # Component search methods
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Return nodes ranked by cosine similarity of label to query."""
        if self._node_embs.shape[0] == 0:
            return []

        query_emb = self._embedder.embed(query)
        k = min(top_k, len(self._node_ids))
        indices, scores = self._embedder.top_k_similar(query_emb, self._node_embs, k=k)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            node_id = self._node_ids[int(idx)]
            entity = dict(self._graph.get_entity(node_id))
            entity["vector_score"] = round((float(score) + 1.0) / 2.0, 6)
            entity["vector_rank"] = rank + 1
            results.append(entity)
        return results

    def _graph_search(
        self, query: str, seed_ids: List[str], top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Return nodes ranked by Personalised PageRank from seed entities."""
        if not seed_ids:
            return []

        # Filter to seeds that actually exist in the graph
        valid_seeds = [s for s in seed_ids if s in self._graph]
        if not valid_seeds:
            return []

        ppr_scores = self._ppr.run(self._graph, seed_nodes=valid_seeds)
        ranked = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, (node_id, score) in enumerate(ranked):
            entity = dict(self._graph.get_entity(node_id))
            entity["graph_score"] = round(float(score), 6)
            entity["graph_rank"] = rank + 1
            results.append(entity)
        return results

    # ------------------------------------------------------------------
    # Fusion methods
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        graph_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Reciprocal Rank Fusion (RRF) over two ranked lists.

        RRF score for a document d:
            RRF(d) = sum_r [ 1 / (k + rank_r(d)) ]

        where rank_r(d) is the 1-based rank of d in result list r (or infinity
        if d does not appear in that list, effectively contributing 0).

        Parameters
        ----------
        graph_results:
            Entities ranked by graph traversal.  Must contain ``'id'`` key.
        vector_results:
            Entities ranked by vector search.  Must contain ``'id'`` key.
        k:
            RRF smoothing constant (default 60 per Cormack et al.).

        Returns
        -------
        Tuple of (sorted_node_ids, rrf_score_dict).
        """
        rrf_scores: Dict[str, float] = defaultdict(float)

        for rank, entity in enumerate(graph_results, start=1):
            node_id = entity.get("id", "")
            if node_id:
                rrf_scores[node_id] += 1.0 / (k + rank)

        for rank, entity in enumerate(vector_results, start=1):
            node_id = entity.get("id", "")
            if node_id:
                rrf_scores[node_id] += 1.0 / (k + rank)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_ids, dict(rrf_scores)

    def _score_fusion(
        self,
        graph_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        alpha: float,
    ) -> List[Tuple[str, float]]:
        """Normalised score fusion: alpha * graph_score + (1-alpha) * vector_score.

        Both scores are first min-max normalised to [0, 1] within each list.
        This is an alternative to RRF that is sensitive to score magnitudes.

        Parameters
        ----------
        graph_results:
            Entities ranked by graph traversal with ``'graph_score'`` key.
        vector_results:
            Entities ranked by vector search with ``'vector_score'`` key.
        alpha:
            Weight for graph score (1-alpha is applied to vector score).

        Returns
        -------
        List of (node_id, fused_score) sorted descending.
        """
        graph_map: Dict[str, float] = {
            e["id"]: e.get("graph_score", 0.0) for e in graph_results if "id" in e
        }
        vector_map: Dict[str, float] = {
            e["id"]: e.get("vector_score", 0.0) for e in vector_results if "id" in e
        }

        # Min-max normalise
        def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
            if not scores:
                return {}
            mn, mx = min(scores.values()), max(scores.values())
            rng = mx - mn or 1.0
            return {k: (v - mn) / rng for k, v in scores.items()}

        graph_norm = _minmax(graph_map)
        vector_norm = _minmax(vector_map)

        all_ids = set(graph_norm) | set(vector_norm)
        fused: List[Tuple[str, float]] = []
        for nid in all_ids:
            gs = graph_norm.get(nid, 0.0)
            vs = vector_norm.get(nid, 0.0)
            fused.append((nid, alpha * gs + (1.0 - alpha) * vs))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_entity_list(
        self,
        node_ids: List[str],
        rrf_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Fetch entity attributes and attach RRF scores."""
        result: List[Dict[str, Any]] = []
        for rank, node_id in enumerate(node_ids, start=1):
            entity = dict(self._graph.get_entity(node_id))
            entity["rrf_score"] = round(rrf_scores.get(node_id, 0.0), 6)
            entity["hybrid_rank"] = rank
            result.append(entity)
        return result

    def _synthesise_answer(
        self,
        query: str,
        fused_ranking: List[Dict[str, Any]],
        alpha: float,
    ) -> str:
        """Produce a readable summary from the hybrid-ranked entities."""
        if not fused_ranking:
            return f"No results found for query: '{query}'."

        top = fused_ranking[:5]
        entity_names = [e.get("label", e.get("id", "?")) for e in top]
        entity_types = list({e.get("type", "UNKNOWN") for e in fused_ranking})
        graph_pct = int(alpha * 100)
        vector_pct = 100 - graph_pct

        lines: List[str] = [
            f"Hybrid search for '{query}' (graph={graph_pct}% / vector={vector_pct}%) "
            f"retrieved {len(fused_ranking)} entity/entities across "
            f"{len(entity_types)} type(s): {', '.join(entity_types)}.",
            "",
            "Top results by Reciprocal Rank Fusion:",
        ]
        for i, entity in enumerate(top, 1):
            label = entity.get("label", entity.get("id", "?"))
            etype = entity.get("type", "UNKNOWN")
            rrf = entity.get("rrf_score", 0.0)
            lines.append(f"  {i}. {label} ({etype}) — RRF={rrf:.5f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility: rank correlation diagnostics
    # ------------------------------------------------------------------

    def rank_correlation(
        self,
        graph_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
    ) -> float:
        """Compute Spearman rank correlation between graph and vector rankings.

        Returns a value in [-1, 1]; values close to 1 indicate the two
        systems agree strongly; values near 0 indicate complementary signals.
        """
        graph_rank = {e["id"]: i for i, e in enumerate(graph_results) if "id" in e}
        vector_rank = {e["id"]: i for i, e in enumerate(vector_results) if "id" in e}

        common = set(graph_rank) & set(vector_rank)
        if len(common) < 2:
            return 0.0

        gr = [graph_rank[nid] for nid in common]
        vr = [vector_rank[nid] for nid in common]

        # Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
        n = len(common)
        d_sq = sum((g - v) ** 2 for g, v in zip(gr, vr))
        denom = n * (n ** 2 - 1)
        if denom == 0:
            return 1.0
        return 1.0 - 6.0 * d_sq / denom
