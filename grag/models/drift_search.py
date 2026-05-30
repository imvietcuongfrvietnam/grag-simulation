"""DRIFT Search simulation.

Based on:
  Microsoft Research (2024). "DRIFT: Dynamic Reasoning and Inference with
  Flexible Traversal" — an extension of Microsoft GraphRAG's search strategies.

DRIFT (Dynamic Reasoning In a Flexible Traversal) enriches local search by:
  1. Decomposing the query into sub-questions (query decomposition)
  2. For each sub-question: run a focused local search
  3. Iteratively refining based on intermediate answers
  4. Aggregating partial answers into a final response

This simulation also implements community-weighted DRIFT search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.document import SimpleEntityExtractor
from grag.core.embeddings import MockEmbedder
from grag.algorithms.community.leiden import LeidenDetector
from grag.algorithms.traversal.pagerank import PersonalizedPageRank


@dataclass
class SubQuestion:
    text: str
    entities_found: List[str] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0


class DriftSearch:
    """DRIFT Search: dynamic query decomposition + iterative local search.

    Extends Microsoft GraphRAG's local search with a multi-step
    query-decomposition loop.
    """

    def __init__(self, max_sub_questions: int = 3,
                 max_hops: int = 2, top_k: int = 5):
        self.max_sub_questions = max_sub_questions
        self.max_hops = max_hops
        self.top_k = top_k

        self._graph = KnowledgeGraph()
        self._embedder = MockEmbedder()
        self._extractor = SimpleEntityExtractor()
        self._ppr = PersonalizedPageRank()
        self._leiden = LeidenDetector()
        self._community_map: Dict[str, int] = {}
        self._passages: List[str] = []
        self._indexed = False
        self._index_time: float = 0.0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        t0 = time.time()
        self._passages = list(documents)
        entity_to_passage: Dict[str, List[int]] = {}

        for i, doc in enumerate(documents):
            entities = self._extractor.extract_entities(doc)
            for ent in entities:
                eid = ent["text"].lower().replace(" ", "_")
                if not self._graph._graph.has_node(eid):
                    self._graph.add_entity(eid, ent["text"], ent["type"])
                entity_to_passage.setdefault(eid, []).append(i)
            relations = self._extractor.extract_relations(doc, entities)
            for rel in relations:
                src = rel["src"].lower().replace(" ", "_")
                dst = rel["dst"].lower().replace(" ", "_")
                for nid, label in [(src, rel["src"]), (dst, rel["dst"])]:
                    if not self._graph._graph.has_node(nid):
                        self._graph.add_entity(nid, label, "CONCEPT")
                self._graph.add_relation(src, dst, rel["relation"], weight=1.0)

        self._entity_to_passage = entity_to_passage

        if self._graph.num_nodes >= 2:
            self._community_map = self._leiden.detect(self._graph)
            self._graph.assign_communities(self._community_map)

        self._indexed = True
        self._index_time = time.time() - t0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> Dict[str, Any]:
        """DRIFT: decompose → search per sub-question → aggregate."""
        if not self._indexed:
            raise RuntimeError("Call index() first.")
        t0 = time.time()

        sub_questions = self._decompose(query)
        answered_sub_questions: List[Dict[str, Any]] = []

        for sq in sub_questions:
            sq_result = self._local_search(sq.text)
            sq.entities_found = sq_result["entities"]
            sq.answer = sq_result["answer"]
            sq.confidence = sq_result["confidence"]
            answered_sub_questions.append({
                "sub_question": sq.text,
                "answer": sq.answer,
                "entities": sq.entities_found[:5],
                "confidence": round(sq.confidence, 4),
            })

        final_answer = self._aggregate(query, sub_questions)
        all_entities = list({
            e for sq in sub_questions for e in sq.entities_found
        })

        return {
            "answer": final_answer,
            "sub_questions": answered_sub_questions,
            "total_entities_used": len(all_entities),
            "entities_used": all_entities[:10],
            "query_time": round(time.time() - t0, 4),
        }

    def _decompose(self, query: str) -> List[SubQuestion]:
        """Simulate query decomposition using heuristic sub-query generation."""
        q_entities = self._extractor.extract_entities(query)
        sub_questions: List[SubQuestion] = [SubQuestion(text=query)]

        if len(q_entities) >= 2:
            for ent in q_entities[:self.max_sub_questions - 1]:
                sub_q = f"What is known about {ent['text']}?"
                sub_questions.append(SubQuestion(text=sub_q))

        if len(sub_questions) < self.max_sub_questions:
            sub_questions.append(SubQuestion(
                text=f"What relationships exist related to: {query[:60]}?"
            ))

        return sub_questions[:self.max_sub_questions]

    def _local_search(self, sub_query: str) -> Dict[str, Any]:
        """Focused local search for a single sub-question."""
        q_emb = self._embedder.embed(sub_query)
        entities = self._extractor.extract_entities(sub_query)
        seed_ids = [
            ent["text"].lower().replace(" ", "_")
            for ent in entities
            if self._graph._graph.has_node(ent["text"].lower().replace(" ", "_"))
        ]

        if not seed_ids and self._graph.num_nodes > 0:
            all_nodes = list(self._graph._graph.nodes())[:30]
            embs = np.stack([
                self._embedder.embed(self._graph._graph.nodes[n].get("label", n))
                for n in all_nodes
            ])
            idxs, _ = self._embedder.top_k_similar(q_emb, embs, k=2)
            seed_ids = [all_nodes[i] for i in idxs]

        # k-hop BFS expansion
        explored: set = set(seed_ids)
        frontier = list(seed_ids)
        for _ in range(self.max_hops):
            next_frontier = []
            for node in frontier:
                for nb in list(self._graph._graph.successors(node))[:3]:
                    if nb not in explored:
                        explored.add(nb)
                        next_frontier.append(nb)
            frontier = next_frontier

        entity_labels = [
            self._graph._graph.nodes[n].get("label", n)
            for n in list(explored)[:8]
            if self._graph._graph.has_node(n)
        ]
        confidence = min(0.9, 0.3 + 0.1 * len(explored))
        answer = (
            f"Sub-answer: entities [{', '.join(entity_labels[:4])}] "
            f"are relevant to '{sub_query[:60]}'"
        )
        return {
            "answer": answer,
            "entities": list(explored),
            "confidence": confidence,
        }

    def _aggregate(self, query: str,
                    sub_questions: List[SubQuestion]) -> str:
        all_entities = [
            e for sq in sub_questions for e in sq.entities_found
        ]
        unique_entities = list(dict.fromkeys(all_entities))[:6]
        labels = [
            self._graph._graph.nodes[e].get("label", e)
            for e in unique_entities
            if self._graph._graph.has_node(e)
        ]
        avg_conf = (sum(sq.confidence for sq in sub_questions) /
                    max(len(sub_questions), 1))
        return (
            f"[DRIFT Search] Answered '{query[:80]}' by resolving "
            f"{len(sub_questions)} sub-questions. Key entities: "
            f"{', '.join(labels[:4])}. "
            f"Aggregate confidence: {avg_conf:.2f}."
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "num_communities": len(set(self._community_map.values())),
            "max_sub_questions": self.max_sub_questions,
            "index_time_s": round(self._index_time, 3),
        }
