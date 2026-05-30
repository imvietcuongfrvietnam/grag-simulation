"""GraphReader simulation.

Based on:
  Li, M. et al. (2024). "GraphReader: Building Graph-based Agent to Enhance
  Long-Context Abilities of Large Language Models." arXiv:2406.14550.

GraphReader converts long documents into a graph and uses an agent to
navigate it via a rational plan:
  1. Build a document structure graph (section → paragraph → entity)
  2. Agent iteratively: selects nodes → reads → asks follow-up → navigates
  3. Stop when sufficient information collected

Simulated here with an embedding-guided multi-hop navigator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.document import SimpleEntityExtractor
from grag.core.embeddings import MockEmbedder


@dataclass
class AgentStep:
    step_id: int
    selected_node: str
    node_label: str
    read_text: str
    follow_up: str
    relevance_score: float


class GraphReader:
    """GraphReader: agent-based navigation of a document structure graph.

    The agent maintains a reading notebook and iteratively selects
    the most relevant graph node to read next.
    """

    def __init__(self, max_steps: int = 5, context_budget: int = 2000):
        self.max_steps = max_steps
        self.context_budget = context_budget

        self._doc_graph = KnowledgeGraph()
        self._embedder = MockEmbedder()
        self._extractor = SimpleEntityExtractor()
        self._node_to_text: Dict[str, str] = {}
        self._indexed = False
        self._index_time: float = 0.0

    # ------------------------------------------------------------------
    # Indexing — build document structure graph
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        """Convert documents into a paragraph/entity graph."""
        t0 = time.time()
        for doc_id, doc in enumerate(documents):
            words = doc.split()
            # Create paragraph nodes (every 60 words)
            para_ids = []
            for i, start in enumerate(range(0, len(words), 60)):
                para_text = " ".join(words[start:start + 60])
                pid = f"doc{doc_id}_para{i}"
                self._doc_graph.add_entity(pid, f"Para {doc_id}.{i}", "PARAGRAPH")
                self._node_to_text[pid] = para_text
                para_ids.append(pid)

            # Chain paragraphs sequentially
            for i in range(len(para_ids) - 1):
                self._doc_graph.add_relation(
                    para_ids[i], para_ids[i + 1], "next_paragraph", weight=1.0
                )

            # Link entities to paragraphs
            entities = self._extractor.extract_entities(doc)
            for ent in entities:
                eid = ent["text"].lower().replace(" ", "_")
                if not self._doc_graph._graph.has_node(eid):
                    self._doc_graph.add_entity(eid, ent["text"], ent["type"])
                    self._node_to_text[eid] = ent["text"]
                # Link to closest paragraph
                if para_ids:
                    self._doc_graph.add_relation(
                        para_ids[0], eid, "mentions", weight=0.5
                    )

            relations = self._extractor.extract_relations(doc, entities)
            for rel in relations:
                src = rel["src"].lower().replace(" ", "_")
                dst = rel["dst"].lower().replace(" ", "_")
                for nid, label in [(src, rel["src"]), (dst, rel["dst"])]:
                    if not self._doc_graph._graph.has_node(nid):
                        self._doc_graph.add_entity(nid, label, "CONCEPT")
                        self._node_to_text[nid] = label
                self._doc_graph.add_relation(
                    src, dst, rel["relation"], weight=1.0
                )

        self._indexed = True
        self._index_time = time.time() - t0

    # ------------------------------------------------------------------
    # Retrieval — agent-based navigation
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Agent navigates the document graph to collect relevant content."""
        if not self._indexed:
            raise RuntimeError("Call index() first.")
        t0 = time.time()

        q_emb = self._embedder.embed(query)
        notebook: List[str] = []
        steps: List[AgentStep] = []
        visited: set = set()
        total_chars = 0

        # Start from the highest-relevance node
        current_node = self._select_start(q_emb, visited)

        for step_id in range(self.max_steps):
            if current_node is None:
                break

            visited.add(current_node)
            node_text = self._node_to_text.get(current_node, "")
            node_label = self._doc_graph._graph.nodes.get(
                current_node, {}
            ).get("label", current_node)

            # Compute relevance and decide whether to take notes
            n_emb = self._embedder.embed(node_text or node_label)
            sim = float(np.dot(q_emb, n_emb) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(n_emb) + 1e-8))
            relevance = max(0.0, (sim + 1.0) / 2.0)

            follow_up = self._generate_follow_up(query, node_label, relevance)

            if relevance > 0.3:
                notebook.append(node_text[:200] if node_text else node_label)
                total_chars += len(node_text or node_label)

            steps.append(AgentStep(
                step_id=step_id + 1,
                selected_node=current_node,
                node_label=node_label,
                read_text=(node_text[:100] + "...") if len(node_text) > 100
                           else node_text,
                follow_up=follow_up,
                relevance_score=round(relevance, 4),
            ))

            if total_chars >= self.context_budget:
                break

            # Navigate to next best node
            current_node = self._navigate_next(current_node, q_emb, visited)

        answer = self._synthesise(query, notebook)
        return {
            "answer": answer,
            "steps": [
                {
                    "step": s.step_id,
                    "node": s.node_label,
                    "relevance": s.relevance_score,
                    "follow_up": s.follow_up,
                }
                for s in steps
            ],
            "notebook_entries": len(notebook),
            "total_chars_read": total_chars,
            "nodes_visited": len(visited),
            "query_time": round(time.time() - t0, 4),
        }

    def _select_start(self, q_emb: np.ndarray,
                       visited: set) -> Optional[str]:
        candidates = [
            n for n in self._doc_graph._graph.nodes() if n not in visited
        ]
        if not candidates:
            return None
        embs = np.stack([
            self._embedder.embed(
                self._doc_graph._graph.nodes[n].get("label", n)
            )
            for n in candidates
        ])
        idxs, _ = self._embedder.top_k_similar(q_emb, embs, k=1)
        return candidates[idxs[0]] if idxs else None

    def _navigate_next(self, current: str, q_emb: np.ndarray,
                        visited: set) -> Optional[str]:
        neighbors = (
            list(self._doc_graph._graph.successors(current)) +
            list(self._doc_graph._graph.predecessors(current))
        )
        candidates = [n for n in neighbors if n not in visited]
        if not candidates:
            return self._select_start(q_emb, visited)
        embs = np.stack([
            self._embedder.embed(
                self._doc_graph._graph.nodes[n].get("label", n)
            )
            for n in candidates
        ])
        idxs, _ = self._embedder.top_k_similar(q_emb, embs, k=1)
        return candidates[idxs[0]]

    def _generate_follow_up(self, query: str, node_label: str,
                              relevance: float) -> str:
        if relevance > 0.5:
            return f"This is highly relevant. Continue exploring nodes connected to '{node_label}'."
        if relevance > 0.3:
            return f"Partially relevant. Check neighbours of '{node_label}' for more details."
        return f"Low relevance at '{node_label}'. Navigate to a different branch."

    def _synthesise(self, query: str, notebook: List[str]) -> str:
        n = len(notebook)
        preview = notebook[0][:100] if notebook else "no content collected"
        return (
            f"[GraphReader] After navigating {n} relevant nodes, "
            f"synthesised answer for '{query[:80]}'. "
            f"Context preview: \"{preview}...\""
        )

    def get_stats(self) -> Dict[str, Any]:
        node_types: Dict[str, int] = {}
        for n, data in self._doc_graph._graph.nodes(data=True):
            t = data.get("type", "UNKNOWN")
            node_types[t] = node_types.get(t, 0) + 1
        return {
            "num_graph_nodes": self._doc_graph.num_nodes,
            "num_graph_edges": self._doc_graph.num_edges,
            "node_types": node_types,
            "max_steps": self.max_steps,
            "index_time_s": round(self._index_time, 3),
        }
