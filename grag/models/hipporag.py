"""HippoRAG simulation.

Based on:
  Gutierrez, B. J. et al. (2024). "HippoRAG: Neurologically Inspired Long-Term
  Memory for Large Language Models."  NeurIPS 2024.

Architecture inspired by the hippocampal memory system:
  - Neocortex  ≈ LLM (semantic pattern completion)
  - Hippocampus ≈ Knowledge graph index
  - Parahippocampal region ≈ Retrieval encoder (dense embeddings)

The pipeline builds an OpenIE-style phrase-level KG, then at query time seeds
Personalized PageRank from query entities to retrieve the most relevant
passages.  All LLM calls are replaced by mock deterministic responses.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grag.core.document import SimpleEntityExtractor, TextChunk
from grag.core.embeddings import MockEmbedder
from grag.core.graph import KnowledgeGraph
from grag.algorithms.traversal.pagerank import PersonalizedPageRank


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Passage:
    """A passage in the HippoRAG index."""

    passage_id: str
    text: str
    entities: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"Passage(id={self.passage_id!r}, entities={self.entities[:3]})"


# ---------------------------------------------------------------------------
# HippoRAG
# ---------------------------------------------------------------------------


class HippoRAG:
    """Simulates HippoRAG (Hippocampus-inspired RAG).

    Inspired by hippocampal memory:
      - Neocortex       = LLM  (pattern separation / completion)
      - Hippocampus     = Knowledge graph index
      - Parahippocampal = Retrieval encoder (MockEmbedder)

    Pipeline
    --------
    1. Extract named entities from corpus (OpenIE-style triples).
    2. Build phrase-level knowledge graph.
    3. At query time: extract query entities.
    4. Run Personalized PageRank (PPR) from query entities.
    5. Rank and retrieve passages by PPR score + dense similarity.

    Parameters
    ----------
    ppr_alpha:
        Damping factor for the Personalized PageRank algorithm.
    top_k_passages:
        Default number of passages to return.
    """

    def __init__(self, ppr_alpha: float = 0.85, top_k_passages: int = 5) -> None:
        self.ppr_alpha = ppr_alpha
        self.top_k_passages = top_k_passages

        self._extractor = SimpleEntityExtractor()
        self._embedder = MockEmbedder(dim=128)
        self._ppr = PersonalizedPageRank(alpha=ppr_alpha)

        self._graph: KnowledgeGraph = KnowledgeGraph()
        self._passages: List[Passage] = []
        # node_id -> list of passage_ids containing that entity
        self._entity_to_passages: Dict[str, List[str]] = defaultdict(list)
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, passages: List[str]) -> None:
        """Index a list of text passages into the hippocampal knowledge graph.

        For each passage:
        1. Extract entity triples (OpenIE-style via regex).
        2. Add entities as nodes and relations as edges to the KG.
        3. Record which entities appear in which passage.

        Parameters
        ----------
        passages:
            List of raw text strings to index.
        """
        self._graph = KnowledgeGraph()
        self._passages = []
        self._entity_to_passages = defaultdict(list)

        for idx, text in enumerate(passages):
            passage_id = f"p{idx:04d}"
            entities = self._extractor.extract_entities(text)
            relations = self._extractor.extract_relations(text, entities)

            passage_emb = self._embedder.embed(text)
            entity_ids = [e["id"] for e in entities]

            passage = Passage(
                passage_id=passage_id,
                text=text,
                entities=entity_ids,
                embedding=passage_emb,
            )
            self._passages.append(passage)

            # Add entities to the KG (nodes)
            for ent in entities:
                self._graph.add_entity(
                    id=ent["id"],
                    label=ent["text"],
                    type=ent["type"],
                    properties={"source_passages": [passage_id]},
                )
                self._entity_to_passages[ent["id"]].append(passage_id)

            # Add triples as edges
            for rel in relations:
                for nid in (rel["src"], rel["dst"]):
                    if nid not in self._graph:
                        self._graph.add_entity(nid, label=nid, type="CONCEPT")
                self._graph.add_relation(
                    src=rel["src"],
                    dst=rel["dst"],
                    relation_type=rel["relation"],
                    weight=1.0,
                    properties={"source_passage": passage_id},
                )

            # Add co-occurrence edges (simulates OpenIE soft-linking)
            for i in range(len(entity_ids)):
                for j in range(i + 1, min(i + 4, len(entity_ids))):
                    a, b = entity_ids[i], entity_ids[j]
                    if a in self._graph and b in self._graph:
                        if not self._graph._nx.has_edge(a, b):
                            self._graph.add_relation(a, b, "CO_OCCURS", weight=0.3)

        self._indexed = True

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve ranked passages using PPR over the hippocampal KG.

        Steps:
        1. Extract entities from the query (parahippocampal binding).
        2. Run PPR from those entities over the KG (hippocampal activation).
        3. Score each passage by the maximum PPR score among its entities.
        4. Combine with dense cosine similarity for final ranking.

        Parameters
        ----------
        query:
            Natural language query string.

        Returns
        -------
        List of passage dicts, each with keys:
          ``passage_id``, ``text``, ``ppr_score``, ``dense_score``,
          ``combined_score``, ``matched_entities``.
        Sorted descending by ``combined_score``.
        """
        self._require_index()

        query_entities = self._extractor.extract_entities(query)
        query_entity_ids = [e["id"] for e in query_entities if e["id"] in self._graph]

        # Fall back to dense retrieval if no query entities found in graph
        if not query_entity_ids:
            return self._dense_fallback(query)

        # PPR from query entities over the hippocampal KG
        ppr_scores = self._ppr.run(self._graph, seed_nodes=query_entity_ids)

        query_emb = self._embedder.embed(query)

        # Score each passage
        results: List[Dict[str, Any]] = []
        for passage in self._passages:
            # Max PPR score among passage entities (hippocampal recall)
            ppr = max(
                (ppr_scores.get(eid, 0.0) for eid in passage.entities),
                default=0.0,
            )

            # Dense cosine similarity (parahippocampal retrieval)
            dense = 0.0
            if passage.embedding is not None:
                dense = (MockEmbedder.cosine_similarity(query_emb, passage.embedding) + 1.0) / 2.0

            combined = 0.6 * ppr + 0.4 * dense

            matched = [
                eid for eid in passage.entities if ppr_scores.get(eid, 0.0) > 1e-4
            ]

            results.append(
                {
                    "passage_id": passage.passage_id,
                    "text": passage.text,
                    "ppr_score": round(ppr, 6),
                    "dense_score": round(dense, 4),
                    "combined_score": round(combined, 4),
                    "matched_entities": matched[:5],
                }
            )

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[: self.top_k_passages]

    def get_recognition_score(self, query: str, passage: str) -> float:
        """Simulate LLM-based recognition scoring.

        The neocortex (LLM) determines whether *passage* is recognisably
        relevant to *query* based on entity overlap and semantic similarity.

        Parameters
        ----------
        query:
            Query string.
        passage:
            Candidate passage text.

        Returns
        -------
        A float in [0, 1] representing recognition confidence.
        """
        # Entity overlap component
        query_ents = {e["id"] for e in self._extractor.extract_entities(query)}
        passage_ents = {e["id"] for e in self._extractor.extract_entities(passage)}
        if query_ents or passage_ents:
            overlap = len(query_ents & passage_ents) / max(len(query_ents | passage_ents), 1)
        else:
            overlap = 0.0

        # Dense similarity component
        q_emb = self._embedder.embed(query)
        p_emb = self._embedder.embed(passage)
        dense = (MockEmbedder.cosine_similarity(q_emb, p_emb) + 1.0) / 2.0

        # Token-level word overlap (simulates BM25 / lexical memory)
        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        p_words = set(re.findall(r"\b\w{3,}\b", passage.lower()))
        lexical = len(q_words & p_words) / max(len(q_words), 1)

        # Weighted combination mimicking LLM recognition
        score = 0.4 * overlap + 0.35 * dense + 0.25 * lexical
        return round(min(1.0, score), 4)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Return statistics about the hippocampal knowledge graph."""
        if not self._indexed:
            return {"indexed": False}
        return {
            "indexed": True,
            "num_passages": len(self._passages),
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "ppr_alpha": self.ppr_alpha,
            "avg_entities_per_passage": round(
                float(np.mean([len(p.entities) for p in self._passages])), 2
            ) if self._passages else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Pure dense retrieval fallback when no query entities are in the graph."""
        query_emb = self._embedder.embed(query)
        results = []
        for passage in self._passages:
            dense = 0.0
            if passage.embedding is not None:
                dense = (MockEmbedder.cosine_similarity(query_emb, passage.embedding) + 1.0) / 2.0
            results.append(
                {
                    "passage_id": passage.passage_id,
                    "text": passage.text,
                    "ppr_score": 0.0,
                    "dense_score": round(dense, 4),
                    "combined_score": round(dense, 4),
                    "matched_entities": [],
                }
            )
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[: self.top_k_passages]

    def _require_index(self) -> None:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")
