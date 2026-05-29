"""MindMap-RAG simulation.

A mindmap-style hierarchical graph with concept clustering and multi-strategy
hierarchical retrieval.  The root node represents the broadest topic; each
level of branches represents progressively more specific sub-concepts.

All LLM calls are replaced by deterministic mock responses.
"""

from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from grag.core.document import SimpleEntityExtractor, TextChunk
from grag.core.embeddings import MockEmbedder
from grag.core.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MindMapNode:
    """A node in the mindmap hierarchy."""

    node_id: str
    concept: str
    level: int  # 0 = root
    children: List["MindMapNode"] = field(default_factory=list)
    passages: List[str] = field(default_factory=list)  # passage texts attached here
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    parent_id: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:
        return f"MindMapNode(id={self.node_id!r}, concept={self.concept!r}, level={self.level})"


# ---------------------------------------------------------------------------
# MindMapRAG
# ---------------------------------------------------------------------------


class MindMapRAG:
    """MindMap-RAG: hierarchical graph with concept clustering and retrieval.

    Architecture
    ------------
    - Root node = overall topic (inferred from corpus).
    - Level-1 branches = major concepts.
    - Level-2+ branches = sub-concepts (up to ``max_depth``).
    - Leaf nodes = individual passages attached to the most specific concept.

    Parameters
    ----------
    max_depth:
        Maximum depth of the mindmap tree (root = level 0).
    branching_factor:
        Target number of child branches per node.
    """

    def __init__(self, max_depth: int = 3, branching_factor: int = 4) -> None:
        self.max_depth = max_depth
        self.branching_factor = branching_factor

        self._extractor = SimpleEntityExtractor()
        self._embedder = MockEmbedder(dim=128)

        self._root: Optional[MindMapNode] = None
        self._all_nodes: List[MindMapNode] = []
        self._node_counter: int = 0
        self._built: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_mindmap(self, documents: List[str]) -> Dict[str, Any]:
        """Build a mindmap-style hierarchical knowledge structure.

        Parameters
        ----------
        documents:
            List of raw text strings.

        Returns
        -------
        JSON-serialisable dict representing the full mindmap tree.
        """
        self._all_nodes = []
        self._node_counter = 0

        # Extract concepts from all documents
        concepts = self._extract_concepts(documents)

        # Compute concept embeddings
        concept_list = list(concepts.keys())
        if not concept_list:
            concept_list = ["general"]
        concept_embs = self._embedder.embed_batch(concept_list)

        # Build root
        root_concept = self._infer_root_concept(documents)
        self._root = self._make_node(root_concept, level=0)
        self._all_nodes.append(self._root)

        # Recursively build the tree
        self._build_subtree(
            parent=self._root,
            concept_list=concept_list,
            concept_embs=concept_embs,
            concept_to_passages=concepts,
            depth_remaining=self.max_depth,
        )

        # Attach passages to leaf nodes
        self._attach_passages(documents)

        self._built = True
        return self._to_dict(self._root)

    def retrieve(
        self, query: str, strategy: str = "hierarchical"
    ) -> Dict[str, Any]:
        """Retrieve relevant nodes from the mindmap.

        Parameters
        ----------
        query:
            Natural language query string.
        strategy:
            Retrieval strategy:
            - ``'hierarchical'`` – top-down traversal following best branch.
            - ``'breadth_first'`` – BFS over all levels, collect top-k.
            - ``'depth_first'`` – DFS with backtracking on score drop.

        Returns
        -------
        Dict with keys:
          ``strategy``, ``results`` (list of node dicts with ``score``),
          ``answer``, ``nodes_explored``.
        """
        if not self._built or self._root is None:
            raise RuntimeError("Call build_mindmap() first.")

        strategy = strategy.lower()
        if strategy == "hierarchical":
            results, explored = self._hierarchical_retrieve(query)
        elif strategy == "breadth_first":
            results, explored = self._bfs_retrieve(query)
        elif strategy == "depth_first":
            results, explored = self._dfs_retrieve(query)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'hierarchical', 'breadth_first', or 'depth_first'.")

        answer = self._mock_answer(query, results)
        return {
            "strategy": strategy,
            "results": results[:5],
            "answer": answer,
            "nodes_explored": explored,
        }

    def render_mindmap(self) -> str:
        """Render an ASCII mindmap visualisation.

        Returns
        -------
        Multi-line string suitable for printing.
        """
        if self._root is None:
            return "(empty mindmap)"
        lines: List[str] = [f"[ROOT] {self._root.concept}"]
        for child in self._root.children:
            self._render_subtree(child, lines, prefix="  ", is_last=False)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal – concept extraction
    # ------------------------------------------------------------------

    def _extract_concepts(self, documents: List[str]) -> Dict[str, List[str]]:
        """Extract concepts from documents.

        Returns a dict mapping concept_id -> list of source passage texts.
        """
        concept_to_passages: Dict[str, List[str]] = defaultdict(list)
        for doc in documents:
            entities = self._extractor.extract_entities(doc)
            for ent in entities:
                concept_to_passages[ent["id"]].append(doc[:200])
        return dict(concept_to_passages)

    def _infer_root_concept(self, documents: List[str]) -> str:
        """Infer an overall topic name from the documents."""
        all_text = " ".join(documents[:3])
        entities = self._extractor.extract_entities(all_text)
        if entities:
            labels = [e["text"] for e in entities[:3]]
            return "Topic: " + ", ".join(labels)
        words = all_text.split()[:5]
        return "Topic: " + " ".join(words)

    # ------------------------------------------------------------------
    # Internal – tree construction
    # ------------------------------------------------------------------

    def _build_subtree(
        self,
        parent: MindMapNode,
        concept_list: List[str],
        concept_embs: np.ndarray,
        concept_to_passages: Dict[str, List[str]],
        depth_remaining: int,
    ) -> None:
        if depth_remaining <= 0 or not concept_list:
            return

        k = min(self.branching_factor, len(concept_list))
        if k == 0:
            return

        # Cluster concepts by embedding similarity
        clusters = self._cluster_concepts(concept_embs, k)
        cluster_map: Dict[int, List[int]] = defaultdict(list)
        for idx, cid in enumerate(clusters):
            cluster_map[int(cid)].append(idx)

        for cid, indices in sorted(cluster_map.items()):
            # Concept label for this branch = most frequent in cluster
            cluster_concepts = [concept_list[i] for i in indices]
            label = self._label_cluster(cluster_concepts)

            child = self._make_node(label, level=parent.level + 1)
            child.parent_id = parent.node_id
            parent.children.append(child)
            self._all_nodes.append(child)

            if depth_remaining > 1 and len(indices) > 1:
                sub_concepts = [concept_list[i] for i in indices]
                sub_embs = concept_embs[indices]
                sub_to_passages = {c: concept_to_passages.get(c, []) for c in sub_concepts}
                self._build_subtree(
                    child, sub_concepts, sub_embs, sub_to_passages, depth_remaining - 1
                )

    def _cluster_concepts(self, embs: np.ndarray, k: int) -> np.ndarray:
        """Cluster embedding vectors into k groups (simple k-means)."""
        n = len(embs)
        if k >= n:
            return np.arange(n)

        rng = np.random.default_rng(0)
        centroids = embs[rng.choice(n, k, replace=False)]
        labels = np.zeros(n, dtype=int)

        for _ in range(10):
            dists = cdist(embs, centroids, metric="cosine")
            labels = np.argmin(dists, axis=1)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = embs[mask].mean(axis=0)
                    norm = np.linalg.norm(centroids[c])
                    if norm > 1e-12:
                        centroids[c] /= norm

        return labels

    def _label_cluster(self, concepts: List[str]) -> str:
        """Generate a human-readable label for a cluster of concepts."""
        # Use the shortest concept as the label (usually most general)
        sorted_c = sorted(concepts, key=len)
        return sorted_c[0].replace("_", " ").title()

    def _attach_passages(self, documents: List[str]) -> None:
        """Attach document passages to the nearest leaf nodes by embedding."""
        leaf_nodes = [n for n in self._all_nodes if n.is_leaf]
        if not leaf_nodes:
            return

        leaf_embs = np.stack([n.embedding for n in leaf_nodes])  # type: ignore
        for doc in documents:
            doc_emb = self._embedder.embed(doc)
            sims = leaf_embs @ doc_emb / (
                np.linalg.norm(leaf_embs, axis=1) * np.linalg.norm(doc_emb) + 1e-12
            )
            best_leaf_idx = int(np.argmax(sims))
            leaf_nodes[best_leaf_idx].passages.append(doc[:200])

    # ------------------------------------------------------------------
    # Internal – node factory
    # ------------------------------------------------------------------

    def _make_node(self, concept: str, level: int) -> MindMapNode:
        self._node_counter += 1
        node_id = f"mm_{self._node_counter}_L{level}"
        emb = self._embedder.embed(concept)
        return MindMapNode(node_id=node_id, concept=concept, level=level, embedding=emb)

    # ------------------------------------------------------------------
    # Internal – retrieval strategies
    # ------------------------------------------------------------------

    def _score_node(self, query_emb: np.ndarray, node: MindMapNode) -> float:
        if node.embedding is None:
            return 0.0
        sim = MockEmbedder.cosine_similarity(query_emb, node.embedding)
        return (sim + 1.0) / 2.0  # normalise to [0, 1]

    def _hierarchical_retrieve(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Top-down: at each level pick best child, collect all nodes on path."""
        assert self._root is not None
        q_emb = self._embedder.embed(query)
        results = []
        explored = 0
        current = self._root

        while True:
            explored += 1
            score = self._score_node(q_emb, current)
            results.append({"node_id": current.node_id, "concept": current.concept,
                            "level": current.level, "score": round(score, 4),
                            "passages": current.passages[:2]})
            if not current.children:
                break
            best_child = max(current.children, key=lambda n: self._score_node(q_emb, n))
            current = best_child

        results.sort(key=lambda x: x["score"], reverse=True)
        return results, explored

    def _bfs_retrieve(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """BFS: score all nodes, return globally ranked list."""
        assert self._root is not None
        q_emb = self._embedder.embed(query)
        results = []
        queue = [self._root]
        explored = 0

        while queue:
            node = queue.pop(0)
            explored += 1
            score = self._score_node(q_emb, node)
            results.append({"node_id": node.node_id, "concept": node.concept,
                            "level": node.level, "score": round(score, 4),
                            "passages": node.passages[:2]})
            queue.extend(node.children)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results, explored

    def _dfs_retrieve(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """DFS with backtracking: descend into best child, backtrack if score drops."""
        assert self._root is not None
        q_emb = self._embedder.embed(query)
        results: List[Dict[str, Any]] = []
        explored = 0
        stack: List[MindMapNode] = [self._root]

        while stack:
            node = stack.pop()
            explored += 1
            score = self._score_node(q_emb, node)
            results.append({"node_id": node.node_id, "concept": node.concept,
                            "level": node.level, "score": round(score, 4),
                            "passages": node.passages[:2]})
            # Push children sorted so highest-score child is processed first
            children_scored = sorted(
                node.children, key=lambda n: self._score_node(q_emb, n)
            )
            stack.extend(children_scored)  # best last = explored first (LIFO)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results, explored

    # ------------------------------------------------------------------
    # Internal – answer generation
    # ------------------------------------------------------------------

    def _mock_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a plausible answer from retrieved mindmap nodes."""
        if not results:
            return f"No relevant mindmap nodes found for: {query}"

        top_concepts = [r["concept"] for r in results[:3]]
        concept_str = ", ".join(top_concepts)
        passages = []
        for r in results[:2]:
            passages.extend(r.get("passages", []))

        passage_hint = ""
        if passages:
            snippet = passages[0][:100].replace("\n", " ")
            passage_hint = f" Passage evidence: '{snippet}...'."

        return (
            f"MindMap answer for '{query}': The most relevant concepts are {concept_str}. "
            f"These were identified through {results[0]['level']}-level mindmap traversal.{passage_hint}"
        )

    # ------------------------------------------------------------------
    # Internal – serialisation
    # ------------------------------------------------------------------

    def _to_dict(self, node: MindMapNode) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "concept": node.concept,
            "level": node.level,
            "num_children": len(node.children),
            "num_passages": len(node.passages),
            "children": [self._to_dict(c) for c in node.children],
        }

    def _render_subtree(
        self,
        node: MindMapNode,
        lines: List[str],
        prefix: str,
        is_last: bool,
    ) -> None:
        connector = "└─ " if is_last else "├─ "
        passage_hint = f" [{len(node.passages)}p]" if node.passages else ""
        lines.append(f"{prefix}{connector}{node.concept}{passage_hint}")
        child_prefix = prefix + ("   " if is_last else "│  ")
        for i, child in enumerate(node.children):
            self._render_subtree(child, lines, child_prefix, i == len(node.children) - 1)
