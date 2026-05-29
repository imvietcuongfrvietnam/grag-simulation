"""Edge-RAG simulation.

Edge-conditioned RAG: relationship types (edge labels) are first-class
retrieval signals.  Edges are embedded and used to find relevant subgraphs,
enabling relation-type filtering during retrieval.

All LLM calls are replaced by deterministic mock responses.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from grag.core.document import SimpleEntityExtractor, TextChunk
from grag.core.embeddings import MockEmbedder
from grag.core.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EdgeRecord:
    """A KG edge with its embedding."""

    src: str
    dst: str
    relation_type: str
    weight: float
    text_description: str  # natural language description of this edge
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    source_passage: str = ""

    def to_triple_text(self) -> str:
        pred = self.relation_type.replace("_", " ").lower()
        return f"{self.src} {pred} {self.dst}"


# ---------------------------------------------------------------------------
# EdgeRAG
# ---------------------------------------------------------------------------


class EdgeRAG:
    """Edge-RAG: relationship types as first-class retrieval signals.

    Architecture
    ------------
    - Index: extract triples, embed each edge as a (subject, predicate, object)
      natural language string.
    - Retrieve: embed the query, find the most similar edges, then expand the
      subgraph around those edges.
    - Optional relation_filter restricts retrieval to specific edge types.

    Parameters
    ----------
    top_k_edges:
        Default number of edges to use as retrieval seeds.
    subgraph_hops:
        Number of hops to expand around retrieved edges.
    """

    def __init__(self, top_k_edges: int = 10, subgraph_hops: int = 1) -> None:
        self.top_k_edges = top_k_edges
        self.subgraph_hops = subgraph_hops

        self._extractor = SimpleEntityExtractor()
        self._embedder = MockEmbedder(dim=128)

        self._graph: KnowledgeGraph = KnowledgeGraph()
        self._edge_records: List[EdgeRecord] = []
        self._edge_embeddings: Optional[np.ndarray] = None
        self._passages: List[str] = []
        self._passage_embeddings: Optional[np.ndarray] = None
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        """Build the edge-centric KG index from *documents*.

        For each document:
        1. Extract entity-relation triples.
        2. Build the KG.
        3. Embed each edge as a natural-language triple string.

        Parameters
        ----------
        documents:
            Raw text strings to index.
        """
        self._graph = KnowledgeGraph()
        self._edge_records = []
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

                # Create an EdgeRecord with a text description
                src_label = self._graph.get_entity(rel["src"]).get("label", rel["src"])
                dst_label = self._graph.get_entity(rel["dst"]).get("label", rel["dst"])
                pred = rel["relation"].replace("_", " ").lower()
                edge_text = f"{src_label} {pred} {dst_label}"

                self._edge_records.append(
                    EdgeRecord(
                        src=rel["src"],
                        dst=rel["dst"],
                        relation_type=rel["relation"],
                        weight=1.0,
                        text_description=edge_text,
                        source_passage=doc_text[:200],
                    )
                )

        # Pre-compute edge embeddings
        if self._edge_records:
            edge_texts = [e.text_description for e in self._edge_records]
            edge_embs = self._embedder.embed_batch(edge_texts)
            self._edge_embeddings = edge_embs
            for rec, emb in zip(self._edge_records, edge_embs):
                rec.embedding = emb

        # Pre-compute passage embeddings
        self._passage_embeddings = self._embedder.embed_batch(documents)

        self._indexed = True

    def retrieve(
        self, query: str, relation_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Retrieve a relevant subgraph by edge similarity.

        Parameters
        ----------
        query:
            Natural language query string.
        relation_filter:
            Optional list of relation type strings to restrict retrieval.
            E.g. ``['WORKS_AT', 'FOUNDED']``.  If None, all relation types are
            considered.

        Returns
        -------
        Dict with keys:
          ``answer`` – mock generated answer.
          ``matched_edges`` – list of top edge dicts.
          ``subgraph_nodes`` – node IDs in the retrieved subgraph.
          ``subgraph_edges`` – edge count in the retrieved subgraph.
          ``relation_types_used`` – distinct relation types in the result.
          ``dense_passages`` – top dense-retrieval passage snippets.
        """
        self._require_index()

        query_emb = self._embedder.embed(query)

        # Filter edge records by relation type if requested
        candidate_edges = self._edge_records
        if relation_filter:
            filter_set: Set[str] = {r.upper() for r in relation_filter}
            candidate_edges = [
                e for e in self._edge_records if e.relation_type.upper() in filter_set
            ]

        # Score edges by embedding similarity to query
        top_edge_records = self._score_edges(query_emb, candidate_edges)

        # Collect seed nodes from top edges
        seed_nodes: List[str] = []
        for rec, _ in top_edge_records:
            seed_nodes.extend([rec.src, rec.dst])
        seed_nodes = list(dict.fromkeys(seed_nodes))  # deduplicate preserving order

        # Expand subgraph around seed nodes
        subgraph = self._expand_subgraph(seed_nodes)

        # Dense passage retrieval
        dense = self._dense_retrieve(query_emb)

        # Collect distinct relation types
        rel_types = list(dict.fromkeys(r.relation_type for r, _ in top_edge_records))

        matched_edges = [
            {
                "src": r.src,
                "dst": r.dst,
                "relation": r.relation_type,
                "description": r.text_description,
                "score": round(float(s), 4),
            }
            for r, s in top_edge_records
        ]

        answer = self._mock_answer(query, top_edge_records, subgraph)

        return {
            "answer": answer,
            "matched_edges": matched_edges,
            "subgraph_nodes": subgraph.nodes,
            "subgraph_edges": subgraph.num_edges,
            "relation_types_used": rel_types,
            "dense_passages": [p["text"][:150] for p in dense[:3]],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_edges(
        self,
        query_emb: np.ndarray,
        candidates: List[EdgeRecord],
    ) -> List[Tuple[EdgeRecord, float]]:
        """Score *candidates* by cosine similarity to *query_emb*."""
        if not candidates:
            return []

        embs = np.stack([e.embedding for e in candidates if e.embedding is not None])
        if len(embs) == 0:
            return []

        k = min(self.top_k_edges, len(candidates))
        top_idx, scores = self._embedder.top_k_similar(query_emb, embs, k=k)
        result = []
        for i, idx in enumerate(top_idx):
            result.append((candidates[idx], float(scores[i])))
        return result

    def _expand_subgraph(self, seed_nodes: List[str]) -> KnowledgeGraph:
        """Expand the seed nodes to a *subgraph_hops*-hop neighbourhood."""
        visited: Set[str] = set(seed_nodes)
        frontier: Set[str] = set(seed_nodes)
        for _ in range(self.subgraph_hops):
            next_frontier: Set[str] = set()
            for n in frontier:
                if n in self._graph:
                    for nb in self._graph.get_neighbors(n, direction="both"):
                        if nb not in visited:
                            next_frontier.add(nb)
            visited.update(next_frontier)
            frontier = next_frontier
        return self._graph.subgraph(list(visited))

    def _dense_retrieve(self, query_emb: np.ndarray) -> List[Dict[str, Any]]:
        if self._passage_embeddings is None or len(self._passages) == 0:
            return []
        k = min(5, len(self._passages))
        top_idx, scores = self._embedder.top_k_similar(query_emb, self._passage_embeddings, k=k)
        return [
            {"passage_idx": int(idx), "text": self._passages[idx], "score": round(float(scores[i]), 4)}
            for i, idx in enumerate(top_idx)
        ]

    def _mock_answer(
        self,
        query: str,
        top_edges: List[Tuple[EdgeRecord, float]],
        subgraph: KnowledgeGraph,
    ) -> str:
        if not top_edges:
            return f"No relevant edges found for query: {query}"

        edge_descs = [r.text_description for r, _ in top_edges[:3]]
        desc_str = "; ".join(edge_descs)
        rel_types = list(dict.fromkeys(r.relation_type for r, _ in top_edges[:5]))
        rel_str = ", ".join(rel_types[:3])

        return (
            f"Edge-RAG answer for '{query}': Retrieved {subgraph.num_nodes} entities "
            f"via {len(top_edges)} matched edges (relation types: {rel_str}). "
            f"Key relationships: {desc_str}. "
            f"The answer is grounded in {subgraph.num_edges} edges of the retrieved subgraph."
        )

    def _require_index(self) -> None:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")
