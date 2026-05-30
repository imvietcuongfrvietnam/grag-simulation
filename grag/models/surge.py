"""SURGE simulation.

Based on:
  Kang, M. et al. (2023). "SURGE: Iterative Retrieval with Graph Reranking for
  Knowledge-Grounded Dialogue." EACL 2023.

SURGE selects an evidence subgraph by:
  1. Identifying relevant graph nodes/edges (entity linking)
  2. Iteratively expanding and reranking paths using GNN-style scoring
  3. Extracting the most evidential subgraph for generation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.document import SimpleEntityExtractor
from grag.core.embeddings import MockEmbedder


@dataclass
class EvidenceSubgraph:
    nodes: List[str]
    edges: List[Tuple[str, str, str]]  # (src, relation, dst)
    score: float
    evidence_passages: List[str] = field(default_factory=list)


class SURGE:
    """SURGE: Subgraph Retrieval Enhanced Generation.

    Simulates the iterative graph reranking approach for
    knowledge-grounded generation.
    """

    def __init__(self, max_iterations: int = 3,
                 expansion_factor: int = 3,
                 top_k_subgraph: int = 8):
        self.max_iterations = max_iterations
        self.expansion_factor = expansion_factor
        self.top_k_subgraph = top_k_subgraph

        self._graph = KnowledgeGraph()
        self._embedder = MockEmbedder()
        self._extractor = SimpleEntityExtractor()
        self._entity_to_passages: Dict[str, List[str]] = {}
        self._passages: List[str] = []
        self._indexed = False
        self._index_time: float = 0.0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        t0 = time.time()
        self._passages = list(documents)
        for i, doc in enumerate(documents):
            entities = self._extractor.extract_entities(doc)
            for ent in entities:
                eid = ent["text"].lower().replace(" ", "_")
                if not self._graph._graph.has_node(eid):
                    self._graph.add_entity(eid, ent["text"], ent["type"])
                self._entity_to_passages.setdefault(eid, []).append(str(i))
            relations = self._extractor.extract_relations(doc, entities)
            for rel in relations:
                src = rel["src"].lower().replace(" ", "_")
                dst = rel["dst"].lower().replace(" ", "_")
                for nid, label in [(src, rel["src"]), (dst, rel["dst"])]:
                    if not self._graph._graph.has_node(nid):
                        self._graph.add_entity(nid, label, "CONCEPT")
                self._graph.add_relation(src, dst, rel["relation"], weight=1.0)
        self._indexed = True
        self._index_time = time.time() - t0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Iteratively expand and rerank subgraph evidence."""
        if not self._indexed:
            raise RuntimeError("Call index() first.")
        t0 = time.time()

        q_emb = self._embedder.embed(query)
        seed_nodes = self._entity_link(query)

        candidate_nodes: Set[str] = set(seed_nodes)
        iteration_logs: List[Dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            # Expand: add k-hop neighbors of current candidates
            expanded = self._expand_candidates(candidate_nodes)
            candidate_nodes.update(expanded)

            # GNN-style rerank: score each node
            node_scores = self._gnn_rerank(list(candidate_nodes), q_emb)

            # Prune: keep top-k by score
            sorted_nodes = sorted(node_scores.items(),
                                   key=lambda x: x[1], reverse=True)
            candidate_nodes = {n for n, _ in sorted_nodes[:self.top_k_subgraph]}

            iteration_logs.append({
                "iteration": iteration + 1,
                "num_candidates": len(candidate_nodes),
                "top_nodes": [
                    self._graph._graph.nodes[n].get("label", n)
                    for n in list(candidate_nodes)[:3]
                    if self._graph._graph.has_node(n)
                ],
                "avg_score": round(
                    sum(node_scores.get(n, 0.0) for n in candidate_nodes)
                    / max(len(candidate_nodes), 1), 4
                ),
            })

        # Build final evidence subgraph
        subgraph = self._build_evidence_subgraph(
            list(candidate_nodes), q_emb, node_scores
        )
        answer = self._generate_answer(query, subgraph)

        return {
            "answer": answer,
            "seed_entities": seed_nodes,
            "evidence_subgraph": {
                "nodes": [
                    self._graph._graph.nodes[n].get("label", n)
                    for n in subgraph.nodes if self._graph._graph.has_node(n)
                ],
                "edges": subgraph.edges[:10],
                "score": round(subgraph.score, 4),
            },
            "iterations": iteration_logs,
            "query_time": round(time.time() - t0, 4),
        }

    def _entity_link(self, query: str) -> List[str]:
        q_entities = self._extractor.extract_entities(query)
        found = []
        for ent in q_entities:
            eid = ent["text"].lower().replace(" ", "_")
            if self._graph._graph.has_node(eid):
                found.append(eid)
        if not found and self._graph.num_nodes > 0:
            q_emb = self._embedder.embed(query)
            nodes = list(self._graph._graph.nodes())[:50]
            embs = np.stack([
                self._embedder.embed(self._graph._graph.nodes[n].get("label", n))
                for n in nodes
            ])
            idxs, _ = self._embedder.top_k_similar(
                q_emb, embs, k=min(3, len(nodes))
            )
            found = [nodes[i] for i in idxs]
        return found[:5]

    def _expand_candidates(self, nodes: Set[str]) -> Set[str]:
        expanded: Set[str] = set()
        for node in nodes:
            if not self._graph._graph.has_node(node):
                continue
            for neighbor in list(self._graph._graph.successors(node))[:self.expansion_factor]:
                expanded.add(neighbor)
            for neighbor in list(self._graph._graph.predecessors(node))[:self.expansion_factor]:
                expanded.add(neighbor)
        return expanded - nodes

    def _gnn_rerank(self, nodes: List[str],
                     q_emb: np.ndarray) -> Dict[str, float]:
        """Simulate GNN message passing by averaging neighbor scores."""
        base_scores: Dict[str, float] = {}
        for n in nodes:
            if not self._graph._graph.has_node(n):
                continue
            label = self._graph._graph.nodes[n].get("label", n)
            n_emb = self._embedder.embed(label)
            sim = float(np.dot(q_emb, n_emb) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(n_emb) + 1e-8))
            base_scores[n] = max(0.0, (sim + 1.0) / 2.0)

        # Message passing: add 0.2 * avg neighbor score
        final_scores: Dict[str, float] = {}
        for n in nodes:
            if not self._graph._graph.has_node(n):
                continue
            neighbors = (
                list(self._graph._graph.successors(n)) +
                list(self._graph._graph.predecessors(n))
            )
            neighbor_scores = [base_scores.get(nb, 0.0) for nb in neighbors
                                if nb in base_scores]
            avg_neighbor = (sum(neighbor_scores) / len(neighbor_scores)
                             if neighbor_scores else 0.0)
            final_scores[n] = base_scores.get(n, 0.0) + 0.2 * avg_neighbor
        return final_scores

    def _build_evidence_subgraph(
        self,
        nodes: List[str],
        q_emb: np.ndarray,
        scores: Dict[str, float],
    ) -> EvidenceSubgraph:
        node_set = set(nodes)
        edges: List[Tuple[str, str, str]] = []
        for n in nodes:
            if not self._graph._graph.has_node(n):
                continue
            for _, dst, data in self._graph._graph.out_edges(n, data=True):
                if dst in node_set:
                    rel = data.get("relation", "related_to")
                    edges.append((n, rel, dst))
        total_score = sum(scores.get(n, 0.0) for n in nodes) / max(len(nodes), 1)
        return EvidenceSubgraph(
            nodes=nodes,
            edges=edges,
            score=total_score,
        )

    def _generate_answer(self, query: str,
                          subgraph: EvidenceSubgraph) -> str:
        top_nodes = [
            self._graph._graph.nodes[n].get("label", n)
            for n in subgraph.nodes[:4]
            if self._graph._graph.has_node(n)
        ]
        return (
            f"[SURGE] Evidence subgraph ({len(subgraph.nodes)} nodes, "
            f"{len(subgraph.edges)} edges) supports the answer to "
            f"'{query[:80]}'. Key entities: {', '.join(top_nodes)}."
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "num_passages": len(self._passages),
            "max_iterations": self.max_iterations,
            "index_time_s": round(self._index_time, 3),
        }
