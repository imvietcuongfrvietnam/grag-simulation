"""Local search retriever using entity-centric subgraph expansion (Microsoft GraphRAG style).

The algorithm:
1. Find seed entities that best match the query via embedding similarity.
2. Expand the seed set to a k-hop neighbourhood in the KnowledgeGraph.
3. Rank all nodes in the expanded subgraph by relevance to the query.
4. Combine the structured KG subgraph with unstructured text context
   for a rich, grounded answer.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.embeddings import MockEmbedder


class LocalSearchRetriever:
    """Entity-centric local subgraph retrieval (as in Microsoft GraphRAG).

    1. Find seed entities matching query.
    2. Expand to k-hop neighbourhood.
    3. Rank nodes by relevance.
    4. Combine structured KG + unstructured text context.

    Parameters
    ----------
    graph:
        A populated :class:`~grag.core.graph.KnowledgeGraph`.
    embedder:
        Embedding model for semantic entity matching.  Defaults to a fresh
        :class:`~grag.core.embeddings.MockEmbedder`.
    text_chunks:
        Optional list of text-chunk dicts (must contain ``'text'`` key and
        optionally an ``'embedding'`` key).  When provided, the retriever
        augments KG context with the most relevant passages.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        embedder: Optional[MockEmbedder] = None,
        text_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._graph = graph
        self._embedder: MockEmbedder = embedder or MockEmbedder()
        self._text_chunks: List[Dict[str, Any]] = text_chunks or []

        # Pre-build node label index for entity matching
        self._node_labels: List[str] = []
        self._node_ids: List[str] = []
        for node_id in graph.nodes:
            entity = graph.get_entity(node_id)
            label = entity.get("label", node_id)
            self._node_labels.append(label)
            self._node_ids.append(node_id)

        # Pre-compute node label embeddings (batch for efficiency)
        if self._node_labels:
            self._node_embs: np.ndarray = self._embedder.embed_batch(self._node_labels)
        else:
            self._node_embs = np.empty((0, self._embedder.DIM), dtype=np.float32)

        # Pre-compute text chunk embeddings if not already present
        self._chunk_texts: List[str] = []
        self._chunk_embs: Optional[np.ndarray] = None
        if self._text_chunks:
            self._chunk_texts = [c.get("text", "") for c in self._text_chunks]
            existing_embs = [c.get("embedding") for c in self._text_chunks]
            if all(e is not None for e in existing_embs):
                self._chunk_embs = np.stack(existing_embs, axis=0)
            else:
                self._chunk_embs = self._embedder.embed_batch(self._chunk_texts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 10,
        seed_k: int = 3,
    ) -> Dict[str, Any]:
        """Run local subgraph retrieval.

        Parameters
        ----------
        query:
            Natural language query.
        max_hops:
            Maximum BFS depth from seed entities when expanding the subgraph.
        top_k:
            Maximum number of entities to include in the ranked results.
        seed_k:
            Number of seed entities identified from the query.

        Returns
        -------
        A dict with keys:
        ``answer`` – synthesised string answer,
        ``subgraph`` – :class:`~grag.core.graph.KnowledgeGraph` of the local neighbourhood,
        ``entities`` – ranked list of entity dicts,
        ``context_window`` – combined structured + unstructured context string.
        """
        if not self._node_ids:
            return {
                "answer": "Knowledge graph is empty.",
                "subgraph": KnowledgeGraph(),
                "entities": [],
                "context_window": "",
            }

        # Step 1 – identify seed entities
        seeds = self._find_seed_entities(query, k=seed_k)

        # Step 2 – expand to k-hop neighbourhood
        subgraph_nodes = self._expand_neighbourhood(
            [s["id"] for s in seeds], max_hops=max_hops
        )

        # Step 3 – build local subgraph
        local_kg = self._graph.subgraph(list(subgraph_nodes))

        # Step 4 – rank all nodes in the subgraph
        ranked_entities = self._rank_entities(query, list(subgraph_nodes), top_k=top_k)

        # Step 5 – compose context window
        context_window = self._build_context_window(query, ranked_entities, local_kg)

        # Step 6 – synthesise answer
        answer = self._synthesise_answer(query, ranked_entities, context_window)

        return {
            "answer": answer,
            "subgraph": local_kg,
            "entities": ranked_entities,
            "context_window": context_window,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_seed_entities(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return the *k* graph entities most semantically similar to *query*."""
        if self._node_embs.shape[0] == 0:
            return []
        query_emb = self._embedder.embed(query)
        indices, scores = self._embedder.top_k_similar(query_emb, self._node_embs, k=k)
        seeds: List[Dict[str, Any]] = []
        for idx, score in zip(indices, scores):
            node_id = self._node_ids[int(idx)]
            entity = self._graph.get_entity(node_id)
            entity["seed_score"] = float(score)
            seeds.append(entity)
        return seeds

    def _expand_neighbourhood(self, seed_ids: List[str], max_hops: int) -> Set[str]:
        """BFS from seed nodes up to *max_hops* hops (undirected)."""
        visited: Set[str] = set()
        queue: deque = deque()
        for seed in seed_ids:
            if seed in self._graph:
                queue.append((seed, 0))
                visited.add(seed)

        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self._graph.get_neighbors(node, direction="both"):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return visited

    def _rank_entities(
        self, query: str, node_ids: List[str], top_k: int
    ) -> List[Dict[str, Any]]:
        """Score each entity in *node_ids* against *query* and return top-k."""
        if not node_ids:
            return []

        query_emb = self._embedder.embed(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for node_id in node_ids:
            entity = self._graph.get_entity(node_id)
            label = entity.get("label", node_id)
            label_emb = self._embedder.embed(label)
            sim = self._embedder.cosine_similarity(query_emb, label_emb)
            relevance = (sim + 1.0) / 2.0  # normalise to [0,1]

            # Boost by node centrality (degree) as a proxy for importance
            degree = len(self._graph.get_neighbors(node_id, direction="both"))
            centrality_boost = 1.0 + 0.1 * min(degree, 10) / 10.0

            combined_score = relevance * centrality_boost

            entity["relevance_score"] = round(relevance, 4)
            entity["degree"] = degree
            entity["combined_score"] = round(combined_score, 4)
            scored.append((combined_score, entity))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def _build_context_window(
        self,
        query: str,
        ranked_entities: List[Dict[str, Any]],
        local_kg: KnowledgeGraph,
    ) -> str:
        """Construct the context string combining KG triples and text passages."""
        lines: List[str] = []

        # Structured context: entity attributes and relations
        lines.append("=== Structured Knowledge Graph Context ===")
        for entity in ranked_entities[:5]:
            eid = entity.get("id", "?")
            label = entity.get("label", eid)
            etype = entity.get("type", "UNKNOWN")
            score = entity.get("combined_score", 0.0)
            lines.append(f"  [{etype}] {label} (score={score:.3f})")

            # Outgoing relations
            for neighbor in local_kg.get_neighbors(eid, direction="out"):
                rel_data = local_kg.get_relation(eid, neighbor)
                rel_type = rel_data.get("relation_type", "RELATED_TO")
                neighbor_label = local_kg.get_entity(neighbor).get("label", neighbor)
                lines.append(f"    --[{rel_type}]--> {neighbor_label}")

            # Incoming relations
            for neighbor in local_kg.get_neighbors(eid, direction="in"):
                rel_data = local_kg.get_relation(neighbor, eid)
                rel_type = rel_data.get("relation_type", "RELATED_TO")
                neighbor_label = local_kg.get_entity(neighbor).get("label", neighbor)
                lines.append(f"    <--[{rel_type}]-- {neighbor_label}")

        # Unstructured context: top relevant text chunks
        if self._chunk_embs is not None and len(self._chunk_texts) > 0:
            lines.append("")
            lines.append("=== Relevant Text Passages ===")
            query_emb = self._embedder.embed(query)
            indices, scores = self._embedder.top_k_similar(
                query_emb, self._chunk_embs, k=3
            )
            for idx, score in zip(indices, scores):
                chunk = self._text_chunks[int(idx)]
                text = chunk.get("text", "")[:300]
                norm_score = (float(score) + 1.0) / 2.0
                lines.append(f"  [score={norm_score:.3f}] {text}")
                if len(chunk.get("text", "")) > 300:
                    lines.append("  ...")

        return "\n".join(lines)

    def _synthesise_answer(
        self,
        query: str,
        ranked_entities: List[Dict[str, Any]],
        context_window: str,
    ) -> str:
        """Produce a grounded answer string from ranked entities.

        In production this would call an LLM with *context_window* as context.
        Here we produce a deterministic summary.
        """
        if not ranked_entities:
            return f"No relevant entities found for query: '{query}'."

        top = ranked_entities[:3]
        entity_names = ", ".join(e.get("label", e.get("id", "?")) for e in top)
        entity_types = list({e.get("type", "UNKNOWN") for e in ranked_entities})

        answer_lines: List[str] = [
            f"Local search for '{query}' identified {len(ranked_entities)} relevant "
            f"entities spanning types: {', '.join(entity_types)}.",
            "",
            f"Most relevant entities: {entity_names}.",
            "",
        ]

        for entity in top:
            label = entity.get("label", entity.get("id", "?"))
            etype = entity.get("type", "UNKNOWN")
            degree = entity.get("degree", 0)
            answer_lines.append(
                f"- {label} ({etype}): connected to {degree} neighbouring node(s), "
                f"relevance={entity.get('combined_score', 0):.3f}."
            )

        return "\n".join(answer_lines)
