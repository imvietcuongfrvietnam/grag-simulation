"""FastGraphRAG simulation.

Based on:
  circlemind-ai (2024). "Fast GraphRAG: Efficient, low-cost Graph RAG that
  adapts to your data." GitHub: circlemind-ai/fast-graphrag.

FastGraphRAG is a lightweight alternative to Microsoft GraphRAG that:
  - Uses PageRank (instead of full Leiden + summarisation) for entity importance
  - Supports incremental graph updates without full re-indexing
  - Retrieves via a single passage of PPR-ranked context
  - Targets lower latency and cost at the expense of global coherence
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.document import SimpleEntityExtractor
from grag.core.embeddings import MockEmbedder
from grag.algorithms.traversal.pagerank import PersonalizedPageRank


class FastGraphRAG:
    """Lightweight Graph RAG using PageRank-based entity ranking.

    Compared to MicrosoftGraphRAG:
      - No community detection step (uses PageRank importance instead)
      - No map-reduce summarisation (single-pass ranked context window)
      - Supports incremental document ingestion
    """

    def __init__(self, top_k_entities: int = 10, ppr_alpha: float = 0.85,
                 context_hops: int = 2):
        self.top_k_entities = top_k_entities
        self.ppr_alpha = ppr_alpha
        self.context_hops = context_hops

        self._graph = KnowledgeGraph()
        self._embedder = MockEmbedder()
        self._extractor = SimpleEntityExtractor()
        self._ppr = PersonalizedPageRank(alpha=ppr_alpha)

        # Global PageRank scores (computed after each index/update)
        self._global_pr: Dict[str, float] = {}
        # Entity → passage mapping
        self._entity_to_passages: Dict[str, List[int]] = {}
        self._passages: List[str] = []
        self._passage_embs: Optional[np.ndarray] = None

        self._indexed = False
        self._index_time: float = 0.0
        self._num_incremental_updates: int = 0

    # ------------------------------------------------------------------
    # Indexing (supports incremental updates)
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        """Initial indexing of a document corpus."""
        t0 = time.time()
        self._passages = list(documents)
        self._ingest_documents(documents, offset=0)
        self._refresh_global_pagerank()
        self._passage_embs = np.stack([
            self._embedder.embed(d) for d in self._passages
        ]) if self._passages else None
        self._indexed = True
        self._index_time = time.time() - t0

    def update(self, new_documents: List[str]) -> None:
        """Incrementally add new documents without full re-indexing."""
        if not self._indexed:
            self.index(new_documents)
            return
        offset = len(self._passages)
        self._passages.extend(new_documents)
        self._ingest_documents(new_documents, offset=offset)
        self._refresh_global_pagerank()
        self._passage_embs = np.stack([
            self._embedder.embed(d) for d in self._passages
        ])
        self._num_incremental_updates += 1

    def _ingest_documents(self, documents: List[str], offset: int) -> None:
        for i, doc in enumerate(documents):
            pid = offset + i
            entities = self._extractor.extract_entities(doc)
            for ent in entities:
                eid = ent["text"].lower().replace(" ", "_")
                if not self._graph._graph.has_node(eid):
                    self._graph.add_entity(eid, ent["text"], ent["type"])
                self._entity_to_passages.setdefault(eid, []).append(pid)
            relations = self._extractor.extract_relations(doc, entities)
            for rel in relations:
                src = rel["src"].lower().replace(" ", "_")
                dst = rel["dst"].lower().replace(" ", "_")
                for nid, label in [(src, rel["src"]), (dst, rel["dst"])]:
                    if not self._graph._graph.has_node(nid):
                        self._graph.add_entity(nid, label, "CONCEPT")
                self._graph.add_relation(src, dst, rel["relation"], weight=1.0)

    def _refresh_global_pagerank(self) -> None:
        """Recompute global PageRank scores — used as prior importance."""
        if self._graph.num_nodes == 0:
            return
        self._global_pr = self._ppr.compute(self._graph)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Fast retrieval: PPR from query entities → ranked context window."""
        if not self._indexed:
            raise RuntimeError("Call index() first.")
        t0 = time.time()

        # Find seed entities
        q_entities = self._extractor.extract_entities(query)
        seed_ids = [
            ent["text"].lower().replace(" ", "_")
            for ent in q_entities
            if self._graph._graph.has_node(
                ent["text"].lower().replace(" ", "_")
            )
        ]

        # If no seeds found, fall back to dense retrieval
        if not seed_ids and self._passage_embs is not None:
            q_emb = self._embedder.embed(query)
            idxs, _ = self._embedder.top_k_similar(
                q_emb, self._passage_embs, k=self.top_k_entities
            )
            passages = [
                {"passage_id": i, "text": self._passages[i][:200],
                 "score": 0.5, "method": "dense_fallback"}
                for i in idxs
            ]
            return {
                "answer": self._mock_answer(query, [], passages),
                "seed_entities": [],
                "ranked_entities": [],
                "context_passages": passages,
                "method": "dense_fallback",
                "query_time": round(time.time() - t0, 4),
            }

        # PPR from seed entities
        ppr_scores = self._ppr.ppr_for_query(
            self._graph, seed_ids, top_k=self.top_k_entities
        )
        ranked_entities = [
            {
                "entity": self._graph._graph.nodes[n].get("label", n),
                "ppr_score": round(score, 4),
                "global_pr": round(self._global_pr.get(n, 0.0), 4),
                "combined_score": round(
                    0.7 * score + 0.3 * self._global_pr.get(n, 0.0), 4
                ),
            }
            for n, score in ppr_scores
        ]
        ranked_entities.sort(key=lambda x: x["combined_score"], reverse=True)

        # Collect passages from top-ranked entities
        seen_passages: Set[int] = set()
        context_passages: List[Dict[str, Any]] = []
        for ent_info in ranked_entities[:self.top_k_entities]:
            label_lower = ent_info["entity"].lower().replace(" ", "_")
            for pid in self._entity_to_passages.get(label_lower, []):
                if pid not in seen_passages and pid < len(self._passages):
                    seen_passages.add(pid)
                    context_passages.append({
                        "passage_id": pid,
                        "text": self._passages[pid][:200],
                        "score": ent_info["combined_score"],
                        "method": "ppr",
                    })

        return {
            "answer": self._mock_answer(query, ranked_entities, context_passages),
            "seed_entities": seed_ids,
            "ranked_entities": ranked_entities[:self.top_k_entities],
            "context_passages": context_passages[:self.top_k_entities],
            "method": "ppr",
            "query_time": round(time.time() - t0, 4),
        }

    def _mock_answer(self, query: str,
                      entities: List[Dict], passages: List[Dict]) -> str:
        top_ents = [e["entity"] for e in entities[:3]]
        return (
            f"[FastGraphRAG] Answer for '{query[:80]}' "
            f"using {len(entities)} PageRank-ranked entities "
            f"({', '.join(top_ents[:2]) if top_ents else 'N/A'}) "
            f"and {len(passages)} context passages."
        )

    def get_stats(self) -> Dict[str, Any]:
        top_pr = sorted(self._global_pr.items(), key=lambda x: x[1],
                         reverse=True)[:3]
        return {
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "num_passages": len(self._passages),
            "incremental_updates": self._num_incremental_updates,
            "top_pagerank_entities": [
                self._graph._graph.nodes[n].get("label", n) for n, _ in top_pr
            ],
            "index_time_s": round(self._index_time, 3),
        }
