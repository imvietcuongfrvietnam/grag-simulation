"""LightRAG simulation.

Based on:
  Guo, Z. et al. (2024). "LightRAG: Simple and Fast Retrieval-Augmented
  Generation." arXiv:2410.05779.

LightRAG uses a dual-level retrieval strategy over a graph index:
  - Low-level  → specific entity search (find entities matching query terms)
  - High-level → abstract concept search (traverse community-level summaries)
  - Hybrid     → combines both with configurable alpha weighting

Indexing builds TWO complementary indexes from the same KG:
  1. Entity index  : each entity embedded for nearest-neighbour lookup
  2. Relation index: each triple embedded as natural language for edge lookup
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


@dataclass
class LightRAGResult:
    mode: str
    answer: str
    low_level_hits: List[Dict[str, Any]] = field(default_factory=list)
    high_level_hits: List[Dict[str, Any]] = field(default_factory=list)
    entities_used: List[str] = field(default_factory=list)
    query_time: float = 0.0


class LightRAG:
    """LightRAG: dual-level graph retrieval (low + high level).

    Modes
    -----
    local   : entity-level retrieval (specific facts)
    global  : community-level retrieval (abstract themes)
    hybrid  : weighted fusion of local + global
    naive   : plain embedding similarity (baseline comparison)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        alpha: float = 0.5,
        top_k: int = 5,
        community_resolution: float = 1.0,
    ):
        if mode not in ("local", "global", "hybrid", "naive"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self.alpha = alpha
        self.top_k = top_k
        self.community_resolution = community_resolution

        self._graph = KnowledgeGraph()
        self._embedder = MockEmbedder()
        self._extractor = SimpleEntityExtractor()
        self._leiden = LeidenDetector()

        # Entity embedding index
        self._entity_ids: List[str] = []
        self._entity_embs: Optional[np.ndarray] = None

        # Relation (triple) embedding index
        self._triple_strs: List[str] = []
        self._triple_embs: Optional[np.ndarray] = None
        self._triple_meta: List[Dict[str, str]] = []

        # Community summaries (high-level index)
        self._community_summaries: Dict[int, str] = {}
        self._community_embs: Optional[np.ndarray] = None
        self._community_ids: List[int] = []

        # Raw documents for naive retrieval
        self._doc_texts: List[str] = []
        self._doc_embs: Optional[np.ndarray] = None

        self._indexed = False
        self._index_time: float = 0.0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        """Full LightRAG indexing: KG construction + dual-level embedding."""
        t0 = time.time()
        self._doc_texts = documents

        for i, doc in enumerate(documents):
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
                triple_str = f"{rel['src']} {rel['relation']} {rel['dst']}"
                self._triple_strs.append(triple_str)
                self._triple_meta.append({"src": src, "dst": dst,
                                           "relation": rel["relation"]})

        # Build entity embedding index (low-level)
        self._entity_ids = list(self._graph._graph.nodes())
        if self._entity_ids:
            self._entity_embs = np.stack([
                self._embedder.embed(
                    self._graph._graph.nodes[n].get("label", n)
                )
                for n in self._entity_ids
            ])

        # Build triple embedding index
        if self._triple_strs:
            self._triple_embs = np.stack([
                self._embedder.embed(t) for t in self._triple_strs
            ])

        # Build community summaries (high-level)
        if self._graph.num_nodes >= 2:
            partition = self._leiden.detect(
                self._graph, resolution=self.community_resolution
            )
            self._graph.assign_communities(partition)
            communities = self._leiden.get_communities(self._graph)
            for cid, members in communities.items():
                labels = [
                    self._graph._graph.nodes[m].get("label", m)
                    for m in list(members)[:8]
                ]
                self._community_summaries[cid] = (
                    f"Community {cid} covers: {', '.join(labels)}. "
                    f"Total {len(members)} entities related through "
                    f"shared context in the knowledge graph."
                )
            self._community_ids = list(self._community_summaries.keys())
            if self._community_ids:
                self._community_embs = np.stack([
                    self._embedder.embed(self._community_summaries[c])
                    for c in self._community_ids
                ])

        # Document embeddings (naive mode)
        if documents:
            self._doc_embs = np.stack([
                self._embedder.embed(d) for d in documents
            ])

        self._indexed = True
        self._index_time = time.time() - t0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, mode: Optional[str] = None) -> LightRAGResult:
        """Retrieve using local, global, hybrid, or naive mode."""
        if not self._indexed:
            raise RuntimeError("Call index() first.")
        mode = mode or self.mode
        t0 = time.time()

        low_hits: List[Dict[str, Any]] = []
        high_hits: List[Dict[str, Any]] = []

        if mode in ("local", "hybrid"):
            low_hits = self._low_level_search(query)
        if mode in ("global", "hybrid"):
            high_hits = self._high_level_search(query)
        if mode == "naive":
            low_hits = self._naive_search(query)

        entities_used = list({h["entity"] for h in low_hits if "entity" in h})

        answer = self._generate_answer(query, mode, low_hits, high_hits)
        return LightRAGResult(
            mode=mode,
            answer=answer,
            low_level_hits=low_hits[:self.top_k],
            high_level_hits=high_hits[:self.top_k],
            entities_used=entities_used,
            query_time=time.time() - t0,
        )

    def _low_level_search(self, query: str) -> List[Dict[str, Any]]:
        """Entity + triple-level search."""
        q_emb = self._embedder.embed(query)
        results: List[Dict[str, Any]] = []

        if self._entity_embs is not None and len(self._entity_ids) > 0:
            idxs, _ = self._embedder.top_k_similar(
                q_emb, self._entity_embs, k=min(self.top_k, len(self._entity_ids))
            )
            for idx in idxs:
                eid = self._entity_ids[idx]
                node = self._graph._graph.nodes[eid]
                neighbors = list(self._graph._graph.successors(eid))[:4]
                results.append({
                    "entity": node.get("label", eid),
                    "type": node.get("type", "UNKNOWN"),
                    "neighbors": neighbors,
                    "score": float(np.dot(q_emb, self._entity_embs[idx]) /
                                   (np.linalg.norm(q_emb) *
                                    np.linalg.norm(self._entity_embs[idx]) + 1e-8)),
                    "level": "low",
                })

        if self._triple_embs is not None and len(self._triple_strs) > 0:
            tidxs, _ = self._embedder.top_k_similar(
                q_emb, self._triple_embs,
                k=min(self.top_k, len(self._triple_strs))
            )
            for idx in tidxs:
                results.append({
                    "triple": self._triple_strs[idx],
                    "meta": self._triple_meta[idx],
                    "score": float(np.dot(q_emb, self._triple_embs[idx]) /
                                   (np.linalg.norm(q_emb) *
                                    np.linalg.norm(self._triple_embs[idx]) + 1e-8)),
                    "level": "low",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.top_k]

    def _high_level_search(self, query: str) -> List[Dict[str, Any]]:
        """Community-level abstract search."""
        if self._community_embs is None or not self._community_ids:
            return []
        q_emb = self._embedder.embed(query)
        idxs, _ = self._embedder.top_k_similar(
            q_emb, self._community_embs,
            k=min(self.top_k, len(self._community_ids))
        )
        results = []
        for idx in idxs:
            cid = self._community_ids[idx]
            results.append({
                "community_id": cid,
                "summary": self._community_summaries[cid],
                "score": float(np.dot(q_emb, self._community_embs[idx]) /
                               (np.linalg.norm(q_emb) *
                                np.linalg.norm(self._community_embs[idx]) + 1e-8)),
                "level": "high",
            })
        return results

    def _naive_search(self, query: str) -> List[Dict[str, Any]]:
        """Plain embedding similarity over raw documents."""
        if self._doc_embs is None:
            return []
        q_emb = self._embedder.embed(query)
        idxs, _ = self._embedder.top_k_similar(
            q_emb, self._doc_embs, k=min(self.top_k, len(self._doc_texts))
        )
        return [
            {"text": self._doc_texts[i][:200], "score": 0.0, "level": "naive"}
            for i in idxs
        ]

    def _generate_answer(self, query: str, mode: str,
                          low: List[Dict], high: List[Dict]) -> str:
        low_entities = [h.get("entity", h.get("triple", "")) for h in low[:3]]
        high_comms = [f"community {h['community_id']}" for h in high[:2]]
        parts = []
        if low_entities:
            parts.append(f"specific entities: {', '.join(str(e) for e in low_entities)}")
        if high_comms:
            parts.append(f"abstract themes from {', '.join(high_comms)}")
        context = " and ".join(parts) if parts else "the knowledge graph"
        return (
            f"[LightRAG/{mode}] Synthesised answer for '{query[:80]}' "
            f"using {context}."
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "num_triples_indexed": len(self._triple_strs),
            "num_communities": len(self._community_summaries),
            "index_time_s": round(self._index_time, 3),
        }
