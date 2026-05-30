"""SubgraphRAG simulation.

Based on:
  Li, J. et al. (2024). "Simple is Effective: The Roles of Graphs and Large
  Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation."
  arXiv:2410.20724.

SubgraphRAG proposes that a simple triple-scoring approach—without complex
GNN architectures—can match or outperform more elaborate methods by:
  1. Retrieving the top-k triples by embedding similarity
  2. Constructing a minimal subgraph from those triples
  3. Linearising the subgraph as LLM context
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.document import SimpleEntityExtractor
from grag.core.embeddings import MockEmbedder


class SubgraphRAG:
    """Simple triple-retrieval RAG without complex GNN scoring.

    Approach: embed each triple as natural language, retrieve top-k
    by cosine similarity to query, build minimal connecting subgraph.
    """

    def __init__(self, top_k_triples: int = 10, linearise: bool = True):
        self.top_k_triples = top_k_triples
        self.linearise = linearise

        self._graph = KnowledgeGraph()
        self._embedder = MockEmbedder()
        self._extractor = SimpleEntityExtractor()

        self._triple_texts: List[str] = []
        self._triple_embs: Optional[np.ndarray] = None
        self._triple_meta: List[Tuple[str, str, str]] = []

        self._indexed = False
        self._index_time: float = 0.0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        t0 = time.time()
        for doc in documents:
            entities = self._extractor.extract_entities(doc)
            for ent in entities:
                eid = ent["text"].lower().replace(" ", "_")
                if not self._graph._graph.has_node(eid):
                    self._graph.add_entity(eid, ent["text"], ent["type"])
            relations = self._extractor.extract_relations(doc, entities)
            for rel in relations:
                src = rel["src"].lower().replace(" ", "_")
                dst = rel["dst"].lower().replace(" ", "_")
                for nid, label in [(src, rel["src"]), (dst, rel["dst"])]:
                    if not self._graph._graph.has_node(nid):
                        self._graph.add_entity(nid, label, "CONCEPT")
                self._graph.add_relation(src, dst, rel["relation"], weight=1.0)
                triple_text = f"{rel['src']} {rel['relation']} {rel['dst']}"
                self._triple_texts.append(triple_text)
                self._triple_meta.append((src, rel["relation"], dst))

        if self._triple_texts:
            self._triple_embs = np.stack([
                self._embedder.embed(t) for t in self._triple_texts
            ])

        self._indexed = True
        self._index_time = time.time() - t0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve top-k triples and build minimal subgraph."""
        if not self._indexed:
            raise RuntimeError("Call index() first.")
        t0 = time.time()

        if not self._triple_texts:
            return {"answer": "No triples indexed.", "triples": [],
                    "subgraph": {}, "linearised": ""}

        q_emb = self._embedder.embed(query)
        idxs, scores_arr = self._embedder.top_k_similar(
            q_emb, self._triple_embs,
            k=min(self.top_k_triples, len(self._triple_texts))
        )

        retrieved_triples: List[Dict[str, Any]] = []
        subgraph_nodes: set = set()
        subgraph_edges: List[Tuple[str, str, str]] = []

        for rank, idx in enumerate(idxs):
            src, rel, dst = self._triple_meta[idx]
            score = float(np.dot(q_emb, self._triple_embs[idx]) /
                          (np.linalg.norm(q_emb) *
                           np.linalg.norm(self._triple_embs[idx]) + 1e-8))
            retrieved_triples.append({
                "triple": self._triple_texts[idx],
                "src": src, "relation": rel, "dst": dst,
                "score": round(score, 4),
                "rank": rank + 1,
            })
            subgraph_nodes.add(src)
            subgraph_nodes.add(dst)
            subgraph_edges.append((src, rel, dst))

        linearised = self._linearise(retrieved_triples) if self.linearise else ""
        answer = self._generate_answer(query, retrieved_triples, linearised)

        return {
            "answer": answer,
            "top_triples": retrieved_triples,
            "subgraph": {
                "nodes": list(subgraph_nodes),
                "edges": subgraph_edges,
                "num_nodes": len(subgraph_nodes),
                "num_edges": len(subgraph_edges),
            },
            "linearised_context": linearised[:500] if linearised else "",
            "query_time": round(time.time() - t0, 4),
        }

    def _linearise(self, triples: List[Dict[str, Any]]) -> str:
        """Convert triples to a readable linear format for LLM input."""
        lines = ["Knowledge graph context:"]
        for t in triples:
            src_label = self._graph._graph.nodes.get(
                t["src"], {}).get("label", t["src"])
            dst_label = self._graph._graph.nodes.get(
                t["dst"], {}).get("label", t["dst"])
            lines.append(f"  ({src_label}) --[{t['relation']}]--> ({dst_label})")
        return "\n".join(lines)

    def _generate_answer(self, query: str,
                          triples: List[Dict], linearised: str) -> str:
        top_triple = triples[0]["triple"] if triples else "N/A"
        return (
            f"[SubgraphRAG] Retrieved {len(triples)} triples (top: '{top_triple}') "
            f"forming a {len(set(t['src'] for t in triples) | set(t['dst'] for t in triples))}-node "
            f"subgraph to answer: '{query[:80]}'"
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "num_triples_indexed": len(self._triple_texts),
            "top_k_triples": self.top_k_triples,
            "index_time_s": round(self._index_time, 3),
        }
