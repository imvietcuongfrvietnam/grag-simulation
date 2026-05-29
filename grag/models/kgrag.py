"""Knowledge Graph RAG (KG-RAG) simulation.

Classic KG-augmented retrieval-augmented generation:
  1. Build a structured knowledge graph from documents.
  2. At query time, identify key entities and perform a SPARQL-like structured
     lookup over the KG.
  3. Combine KG context with dense passage retrieval.
  4. Generate an answer using mock LLM reasoning.

All LLM calls are replaced by deterministic mock responses.
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Triple:
    """A subject-predicate-object triple extracted from text."""

    subject: str
    predicate: str
    obj: str
    source_text: str = ""
    confidence: float = 1.0

    def to_natural(self) -> str:
        return f"{self.subject} {self.predicate.replace('_', ' ').lower()} {self.obj}"


# ---------------------------------------------------------------------------
# KGRAG
# ---------------------------------------------------------------------------


class KGRAG:
    """Classic Knowledge Graph-augmented RAG.

    Architecture
    ------------
    - Build structured KG from documents (entity + relation extraction).
    - Query → SPARQL-like entity lookup over the KG.
    - KG context + dense retrieval → mock LLM generation.

    Parameters
    ----------
    max_hops:
        Maximum graph hops for neighbourhood expansion during retrieval.
    top_k_passages:
        Number of passages to retrieve for the dense component.
    """

    def __init__(self, max_hops: int = 2, top_k_passages: int = 5) -> None:
        self.max_hops = max_hops
        self.top_k_passages = top_k_passages

        self._extractor = SimpleEntityExtractor()
        self._embedder = MockEmbedder(dim=128)

        self._graph: KnowledgeGraph = KnowledgeGraph()
        self._triples: List[Triple] = []
        self._passages: List[str] = []
        self._passage_embeddings: Optional[np.ndarray] = None
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_kg(self, documents: List[str]) -> KnowledgeGraph:
        """Build a structured knowledge graph from *documents*.

        Parameters
        ----------
        documents:
            Raw text strings to process.

        Returns
        -------
        The constructed KnowledgeGraph.
        """
        self._graph = KnowledgeGraph()
        self._triples = []
        self._passages = list(documents)

        for doc_idx, doc_text in enumerate(documents):
            entities = self._extractor.extract_entities(doc_text)
            relations = self._extractor.extract_relations(doc_text, entities)

            for ent in entities:
                self._graph.add_entity(
                    id=ent["id"],
                    label=ent["text"],
                    type=ent["type"],
                    properties={"source_doc": doc_idx},
                )

            for rel in relations:
                for nid in (rel["src"], rel["dst"]):
                    if nid not in self._graph:
                        self._graph.add_entity(nid, label=nid, type="CONCEPT")

                self._graph.add_relation(
                    src=rel["src"],
                    dst=rel["dst"],
                    relation_type=rel["relation"],
                    weight=1.0,
                    properties={"source_doc": doc_idx},
                )
                self._triples.append(
                    Triple(
                        subject=rel["src"],
                        predicate=rel["relation"],
                        obj=rel["dst"],
                        source_text=doc_text[:100],
                    )
                )

        # Pre-compute passage embeddings for dense retrieval
        self._passage_embeddings = self._embedder.embed_batch(documents)
        self._indexed = True
        return self._graph

    def query(self, question: str) -> Dict[str, Any]:
        """Answer a question using KG context + dense retrieval.

        Steps:
        1. Extract question entities.
        2. SPARQL-like KG lookup for structured facts.
        3. Dense passage retrieval for supporting context.
        4. Mock LLM generation with reasoning chain.

        Parameters
        ----------
        question:
            Natural language question.

        Returns
        -------
        Dict with keys:
          ``answer`` – generated text answer.
          ``kg_context`` – list of KG triples as dicts.
          ``retrieved_passages`` – top retrieved passage snippets.
          ``reasoning_chain`` – step-by-step mock reasoning.
        """
        self._require_index()

        # Step 1 – extract question entities
        q_entities = self._extractor.extract_entities(question)
        q_entity_ids = [e["id"] for e in q_entities if e["id"] in self._graph]

        # Step 2 – structured KG lookup
        kg_context: List[Dict[str, Any]] = []
        for eid in q_entity_ids:
            kg_context.extend(self._sparql_like_lookup(eid))

        # Step 3 – dense passage retrieval
        retrieved_passages = self._dense_retrieve(question)

        # Step 4 – mock answer generation
        reasoning_chain = self._build_reasoning_chain(
            question, q_entities, kg_context, retrieved_passages
        )
        answer = self._mock_generate(question, kg_context, retrieved_passages, reasoning_chain)

        return {
            "answer": answer,
            "kg_context": kg_context[:10],
            "retrieved_passages": [p["text"][:150] for p in retrieved_passages],
            "reasoning_chain": reasoning_chain,
        }

    def _sparql_like_lookup(self, entity: str) -> List[Dict[str, Any]]:
        """Simulate a structured SPARQL-like query over the KG.

        For a given entity, retrieves:
        - All outgoing triples  (entity ?rel ?obj)
        - All incoming triples  (?subj ?rel entity)
        - All properties of the entity node

        Parameters
        ----------
        entity:
            Node ID to look up in the KG.

        Returns
        -------
        List of triple dicts with keys: ``subject``, ``predicate``, ``object``,
        ``direction``.
        """
        results: List[Dict[str, Any]] = []

        if entity not in self._graph:
            return results

        entity_data = self._graph.get_entity(entity)

        # Outgoing edges  (entity -> dst)
        for dst in self._graph.get_neighbors(entity, direction="out"):
            rel_data = self._graph.get_relation(entity, dst)
            results.append(
                {
                    "subject": entity_data.get("label", entity),
                    "predicate": rel_data.get("relation_type", "RELATED_TO"),
                    "object": self._graph.get_entity(dst).get("label", dst),
                    "direction": "out",
                    "weight": rel_data.get("weight", 1.0),
                }
            )

        # Incoming edges  (src -> entity)
        for src in self._graph.get_neighbors(entity, direction="in"):
            rel_data = self._graph.get_relation(src, entity)
            results.append(
                {
                    "subject": self._graph.get_entity(src).get("label", src),
                    "predicate": rel_data.get("relation_type", "RELATED_TO"),
                    "object": entity_data.get("label", entity),
                    "direction": "in",
                    "weight": rel_data.get("weight", 1.0),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Dense cosine-similarity retrieval over indexed passages."""
        if self._passage_embeddings is None or len(self._passages) == 0:
            return []

        q_emb = self._embedder.embed(question)
        k = min(self.top_k_passages, len(self._passages))
        top_idx, scores = self._embedder.top_k_similar(q_emb, self._passage_embeddings, k=k)

        results = []
        for i, idx in enumerate(top_idx):
            results.append(
                {
                    "passage_idx": int(idx),
                    "text": self._passages[idx],
                    "score": round(float(scores[i]), 4),
                }
            )
        return results

    def _build_reasoning_chain(
        self,
        question: str,
        q_entities: List[Dict[str, Any]],
        kg_context: List[Dict[str, Any]],
        passages: List[Dict[str, Any]],
    ) -> List[str]:
        """Simulate a step-by-step reasoning chain."""
        chain: List[str] = []

        entity_labels = [e["text"] for e in q_entities[:4]]
        chain.append(
            f"Step 1: Identified question entities: {', '.join(entity_labels) or 'none detected'}."
        )

        if kg_context:
            facts = [
                f"{t['subject']} {t['predicate'].replace('_', ' ').lower()} {t['object']}"
                for t in kg_context[:3]
            ]
            chain.append(f"Step 2: KG lookup returned {len(kg_context)} facts. Key facts: {'; '.join(facts)}.")
        else:
            chain.append("Step 2: No direct KG facts found for query entities.")

        chain.append(
            f"Step 3: Dense retrieval found {len(passages)} relevant passages "
            f"(top score: {passages[0]['score'] if passages else 'N/A'})."
        )

        chain.append("Step 4: Synthesising KG facts and passage context to generate answer.")
        return chain

    def _mock_generate(
        self,
        question: str,
        kg_context: List[Dict[str, Any]],
        passages: List[Dict[str, Any]],
        reasoning_chain: List[str],
    ) -> str:
        """Mock LLM answer generation using KG and dense context."""
        # Build answer from KG triples
        if kg_context:
            facts_str = "; ".join(
                f"{t['subject']} {t['predicate'].replace('_', ' ').lower()} {t['object']}"
                for t in kg_context[:3]
            )
            kg_part = f"According to the knowledge graph: {facts_str}. "
        else:
            kg_part = "No structured KG facts were found. "

        # Add passage context
        if passages:
            passage_snippet = passages[0]["text"][:120].replace("\n", " ")
            passage_part = f"Supporting passage context: '{passage_snippet}...' "
        else:
            passage_part = ""

        return (
            f"Answer to '{question}': {kg_part}"
            f"{passage_part}"
            f"This answer integrates {len(kg_context)} KG triples and "
            f"{len(passages)} retrieved passages through structured-unstructured fusion."
        )

    def _require_index(self) -> None:
        if not self._indexed:
            raise RuntimeError("Call build_kg() before query().")
