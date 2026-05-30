"""Think-on-Graph (ToG) simulation.

Based on:
  Sun, J. et al. (2023). "Think-on-Graph: Deep and Responsible Reasoning of
  Large Language Model on Knowledge Graph." ICLR 2024.

ToG performs iterative beam search over a KG guided by an LLM oracle:
  1. Extract topic entities from the query.
  2. Expand: enumerate one-hop relations from each beam entity.
  3. Prune:  LLM scores each relation path; keep top-W beams.
  4. Reason: LLM judges if the current beam set is sufficient to answer.
  5. Repeat until answer found or max depth reached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.document import SimpleEntityExtractor
from grag.core.embeddings import MockEmbedder


@dataclass
class BeamPath:
    """A single path explored during ToG beam search."""
    nodes: List[str]
    relations: List[str]
    score: float = 0.0
    reasoning: str = ""


class ThinkOnGraph:
    """Think-on-Graph: LLM-guided iterative beam search over a KG.

    The LLM oracle is simulated by embedding-based relation scoring.
    """

    def __init__(self, beam_width: int = 3, max_depth: int = 4,
                 prune_threshold: float = 0.2):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.prune_threshold = prune_threshold

        self._graph = KnowledgeGraph()
        self._embedder = MockEmbedder()
        self._extractor = SimpleEntityExtractor()
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
        self._indexed = True
        self._index_time = time.time() - t0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Run ToG beam search and return the reasoning chain + answer."""
        if not self._indexed:
            raise RuntimeError("Call index() first.")
        t0 = time.time()

        # Step 1: topic entity detection
        seed_entities = self._detect_topic_entities(query)
        if not seed_entities:
            return self._no_entity_fallback(query)

        # Initialise beams — one per seed entity
        beams: List[BeamPath] = [
            BeamPath(nodes=[e], relations=[], score=1.0) for e in seed_entities
        ]

        all_iterations: List[Dict[str, Any]] = []
        final_answer_found = False

        for depth in range(1, self.max_depth + 1):
            # Step 2: expand each beam by one hop
            candidates: List[BeamPath] = []
            for beam in beams:
                expansions = self._expand(beam, query)
                candidates.extend(expansions)

            if not candidates:
                break

            # Step 3: LLM prune — keep top-W by score
            candidates.sort(key=lambda b: b.score, reverse=True)
            beams = [c for c in candidates[:self.beam_width]
                     if c.score >= self.prune_threshold]

            # Step 4: LLM reason — check if answer is reachable
            iteration_info = {
                "depth": depth,
                "beams": [
                    {
                        "path": " → ".join(
                            self._graph._graph.nodes[n].get("label", n)
                            for n in b.nodes if self._graph._graph.has_node(n)
                        ),
                        "score": round(b.score, 4),
                        "reasoning": b.reasoning,
                    }
                    for b in beams
                ],
            }
            all_iterations.append(iteration_info)

            if beams and self._llm_judge(query, beams):
                final_answer_found = True
                break

        answer = self._generate_answer(query, beams)
        return {
            "answer": answer,
            "seed_entities": seed_entities,
            "final_beams": [
                {
                    "path": [
                        self._graph._graph.nodes[n].get("label", n)
                        for n in b.nodes if self._graph._graph.has_node(n)
                    ],
                    "relations": b.relations,
                    "score": round(b.score, 4),
                }
                for b in beams
            ],
            "iterations": all_iterations,
            "depth_reached": len(all_iterations),
            "answer_found": final_answer_found,
            "query_time": round(time.time() - t0, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_topic_entities(self, query: str) -> List[str]:
        """Find KG entities that best match the query."""
        q_entities = self._extractor.extract_entities(query)
        found = []
        for ent in q_entities:
            eid = ent["text"].lower().replace(" ", "_")
            if self._graph._graph.has_node(eid):
                found.append(eid)

        if not found and self._graph.num_nodes > 0:
            q_emb = self._embedder.embed(query)
            nodes = list(self._graph._graph.nodes())
            embs = np.stack([
                self._embedder.embed(self._graph._graph.nodes[n].get("label", n))
                for n in nodes
            ])
            idxs, _ = self._embedder.top_k_similar(
                q_emb, embs, k=min(self.beam_width, len(nodes))
            )
            found = [nodes[i] for i in idxs]
        return found[:self.beam_width]

    def _expand(self, beam: BeamPath, query: str) -> List[BeamPath]:
        """Expand a beam by one hop along all outgoing relations."""
        tail = beam.nodes[-1]
        if not self._graph._graph.has_node(tail):
            return []
        q_emb = self._embedder.embed(query)
        expansions = []
        for _, dst, data in self._graph._graph.out_edges(tail, data=True):
            relation = data.get("relation", "related_to")
            path_text = f"{tail} {relation} {dst}"
            path_emb = self._embedder.embed(path_text)
            sim = float(np.dot(q_emb, path_emb) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(path_emb) + 1e-8))
            score = beam.score * 0.9 * max(0.0, (sim + 1.0) / 2.0)
            expansions.append(BeamPath(
                nodes=beam.nodes + [dst],
                relations=beam.relations + [relation],
                score=score,
                reasoning=f"Followed '{relation}' from "
                           f"'{tail}' to '{dst}' (sim={sim:.3f})",
            ))
        return expansions

    def _llm_judge(self, query: str, beams: List[BeamPath]) -> bool:
        """Simulate LLM deciding if the current beams can answer the query."""
        if not beams:
            return False
        max_score = max(b.score for b in beams)
        return max_score > 0.5 and len(beams[0].nodes) >= 2

    def _generate_answer(self, query: str, beams: List[BeamPath]) -> str:
        if not beams:
            return f"No sufficient reasoning path found for: {query[:80]}"
        best = beams[0]
        path_labels = [
            self._graph._graph.nodes[n].get("label", n)
            for n in best.nodes if self._graph._graph.has_node(n)
        ]
        return (
            f"[Think-on-Graph] After {len(best.nodes)-1} reasoning hops "
            f"along path [{' → '.join(path_labels)}], "
            f"the answer to '{query[:80]}' was derived "
            f"(confidence: {best.score:.3f})."
        )

    def _no_entity_fallback(self, query: str) -> Dict[str, Any]:
        return {
            "answer": f"No topic entities detected in KG for: {query[:80]}",
            "seed_entities": [],
            "final_beams": [],
            "iterations": [],
            "depth_reached": 0,
            "answer_found": False,
            "query_time": 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "index_time_s": round(self._index_time, 3),
        }
