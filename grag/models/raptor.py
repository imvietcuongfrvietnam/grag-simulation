"""RAPTOR simulation.

Based on:
  Sarthi, P. et al. (2024). "RAPTOR: Recursive Abstractive Processing for
  Tree-Organized Retrieval."  ICLR 2024.

RAPTOR builds a tree of progressively more abstract summaries.  Leaf nodes
are raw text chunks; each internal node is a mock-LLM summary of a cluster
of its children.  Retrieval can either traverse the tree top-down or search
all nodes collapsed into a flat index.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from grag.core.document import SimpleEntityExtractor, TextChunk
from grag.core.embeddings import MockEmbedder


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in the RAPTOR tree."""

    node_id: str
    text: str
    level: int  # 0 = leaf
    children: List["TreeNode"] = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    cluster_id: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        return self.level == 0

    def __repr__(self) -> str:
        return f"TreeNode(id={self.node_id!r}, level={self.level}, children={len(self.children)})"


# ---------------------------------------------------------------------------
# RAPTOR
# ---------------------------------------------------------------------------


class RAPTOR:
    """Simulates RAPTOR tree construction and retrieval.

    Pipeline
    --------
    1. Embed leaf chunks.
    2. Cluster similar chunks with soft GMM-style assignment.
    3. Summarise each cluster -> parent node (mock LLM).
    4. Recursively build tree until root or until ``max_levels`` is reached.
    5. Retrieval via tree traversal (top-down) or collapsed (flat) retrieval.

    Parameters
    ----------
    max_levels:
        Maximum depth of the tree (leaf level = 0).
    cluster_size:
        Target number of leaves per cluster at each level.
    embedding_dim:
        Dimensionality of the MockEmbedder vectors.
    """

    def __init__(
        self,
        max_levels: int = 4,
        cluster_size: int = 5,
        embedding_dim: int = 128,
    ) -> None:
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.embedding_dim = embedding_dim

        self._embedder = MockEmbedder(dim=embedding_dim)
        self._extractor = SimpleEntityExtractor()

        self._root: Optional[TreeNode] = None
        self._all_nodes: List[TreeNode] = []
        self._node_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_tree(self, texts: List[str]) -> Dict[str, Any]:
        """Build the RAPTOR tree from a list of text strings.

        Parameters
        ----------
        texts:
            Raw text strings (documents or pre-chunked passages).

        Returns
        -------
        A JSON-serialisable dict describing the full tree structure.
        """
        self._all_nodes = []
        self._node_counter = 0

        # Level 0 – leaf nodes
        leaves = self._make_leaf_nodes(texts)
        self._all_nodes.extend(leaves)

        current_level_nodes = leaves
        for level in range(1, self.max_levels + 1):
            if len(current_level_nodes) <= 1:
                # Already at the root, nothing more to cluster
                break
            parent_nodes = self._build_level(current_level_nodes, level)
            self._all_nodes.extend(parent_nodes)
            current_level_nodes = parent_nodes

        # The last level becomes the root (possibly virtual)
        if len(current_level_nodes) == 1:
            self._root = current_level_nodes[0]
        else:
            # Create a virtual root summarising everything
            root_text = self._mock_summary(current_level_nodes, level=self.max_levels + 1)
            self._root = self._make_node(root_text, level=self.max_levels + 1, children=current_level_nodes)
            self._all_nodes.append(self._root)

        return self._tree_to_dict(self._root)

    def retrieve_tree_traversal(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Top-down tree traversal retrieval.

        At each level, select the most similar node to the query and descend
        into its children.  Continues until leaf nodes are reached.

        Parameters
        ----------
        query:
            Natural language query string.
        top_k:
            Number of leaf (or near-leaf) nodes to return.

        Returns
        -------
        List of result dicts with keys: ``node_id``, ``text``, ``level``,
        ``score``.
        """
        if self._root is None:
            return []

        query_emb = self._embedder.embed(query)

        # Descend from root, always keeping the best child at each level
        current_nodes = [self._root]
        path_nodes: List[TreeNode] = []

        while current_nodes:
            # Score all current nodes against query
            candidates = self._score_nodes(query_emb, current_nodes)
            # Pick top-1 at each level, then expand its children
            best_node, _best_score = candidates[0]
            path_nodes.append(best_node)
            if best_node.is_leaf or not best_node.children:
                break
            current_nodes = best_node.children

        # Collect leaves reachable from the best path node
        leaves = self._collect_leaves(path_nodes[-1])
        if not leaves:
            leaves = [path_nodes[-1]]

        scored_leaves = self._score_nodes(query_emb, leaves)
        return [
            {
                "node_id": n.node_id,
                "text": n.text[:200],
                "level": n.level,
                "score": round(float(s), 4),
            }
            for n, s in scored_leaves[:top_k]
        ]

    def retrieve_collapsed(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Flat retrieval across all tree levels.

        All nodes (leaves and internal) are scored against the query and the
        top-*k* are returned regardless of level.

        Parameters
        ----------
        query:
            Natural language query string.
        top_k:
            Number of results to return.

        Returns
        -------
        List of result dicts with keys: ``node_id``, ``text``, ``level``,
        ``score``.
        """
        if not self._all_nodes:
            return []

        query_emb = self._embedder.embed(query)
        scored = self._score_nodes(query_emb, self._all_nodes)

        return [
            {
                "node_id": n.node_id,
                "text": n.text[:200],
                "level": n.level,
                "score": round(float(s), 4),
            }
            for n, s in scored[:top_k]
        ]

    def visualize_tree(self) -> str:
        """Return an ASCII tree visualisation of the RAPTOR tree.

        Returns
        -------
        A multi-line string suitable for printing to a terminal.
        """
        if self._root is None:
            return "(empty tree)"
        lines: List[str] = []
        self._ascii_tree(self._root, lines, prefix="", is_last=True)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal – tree construction
    # ------------------------------------------------------------------

    def _make_leaf_nodes(self, texts: List[str]) -> List[TreeNode]:
        leaves: List[TreeNode] = []
        for i, text in enumerate(texts):
            node = self._make_node(text, level=0)
            leaves.append(node)
        return leaves

    def _build_level(
        self, nodes: List[TreeNode], level: int
    ) -> List[TreeNode]:
        """Cluster *nodes* and create parent summary nodes for the next level."""
        if not nodes:
            return []

        embeddings = np.stack([n.embedding for n in nodes if n.embedding is not None])
        if len(embeddings) == 0:
            return nodes  # degenerate

        k = max(1, len(nodes) // self.cluster_size)
        k = min(k, len(nodes))  # cannot have more clusters than nodes

        clusters = self._gmm_cluster(embeddings, k)

        cluster_map: Dict[int, List[TreeNode]] = defaultdict(list)
        for node, cluster_id in zip(nodes, clusters):
            node.cluster_id = int(cluster_id)
            cluster_map[int(cluster_id)].append(node)

        parent_nodes: List[TreeNode] = []
        for cluster_id, children in sorted(cluster_map.items()):
            summary = self._mock_summary(children, level=level)
            parent = self._make_node(summary, level=level, children=children)
            parent_nodes.append(parent)

        return parent_nodes

    def _make_node(
        self,
        text: str,
        level: int,
        children: Optional[List[TreeNode]] = None,
    ) -> TreeNode:
        self._node_counter += 1
        node_id = f"n{self._node_counter}_L{level}"
        emb = self._embedder.embed(text)
        node = TreeNode(
            node_id=node_id,
            text=text,
            level=level,
            children=children or [],
            embedding=emb,
        )
        return node

    # ------------------------------------------------------------------
    # Internal – clustering (GMM-style soft assignment simplified to k-means)
    # ------------------------------------------------------------------

    def _gmm_cluster(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        """Simplified GMM clustering using k-means initialisation + iteration.

        For simulation purposes we run 10 iterations of Lloyd's algorithm with
        cosine distance, which approximates the behaviour of GMM clustering.
        """
        n = len(embeddings)
        if k >= n:
            return np.arange(n)

        # k-means++ initialisation
        rng = np.random.default_rng(42)
        centroids = [embeddings[rng.integers(n)]]
        for _ in range(k - 1):
            dists = np.min(
                cdist(embeddings, np.array(centroids), metric="cosine"), axis=1
            )
            probs = dists / (dists.sum() + 1e-12)
            centroids.append(embeddings[rng.choice(n, p=probs)])

        centroids_arr = np.array(centroids)

        # Lloyd iterations
        labels = np.zeros(n, dtype=int)
        for _ in range(10):
            dists = cdist(embeddings, centroids_arr, metric="cosine")
            labels = np.argmin(dists, axis=1)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids_arr[c] = embeddings[mask].mean(axis=0)
                    norm = np.linalg.norm(centroids_arr[c])
                    if norm > 1e-12:
                        centroids_arr[c] /= norm

        return labels

    # ------------------------------------------------------------------
    # Internal – mock summarisation
    # ------------------------------------------------------------------

    def _mock_summary(self, nodes: List[TreeNode], level: int) -> str:
        """Simulate an LLM abstractive summary from child node texts."""
        snippets = []
        for node in nodes[:4]:
            words = node.text.split()
            snippets.append(" ".join(words[:15]))

        combined = "; ".join(snippets)
        truncated = textwrap.shorten(combined, width=120, placeholder="...")

        n_children = len(nodes)
        leaf_count = sum(1 for n in nodes if n.is_leaf)

        return (
            f"[Level-{level} Summary] Abstractive synthesis of {n_children} sources "
            f"({leaf_count} leaf, {n_children - leaf_count} internal): {truncated}"
        )

    # ------------------------------------------------------------------
    # Internal – retrieval helpers
    # ------------------------------------------------------------------

    def _score_nodes(
        self, query_emb: np.ndarray, nodes: List[TreeNode]
    ) -> List[Tuple[TreeNode, float]]:
        """Score *nodes* by cosine similarity to *query_emb*, sorted descending."""
        scored = []
        for node in nodes:
            if node.embedding is not None:
                sim = MockEmbedder.cosine_similarity(query_emb, node.embedding)
                scored.append((node, sim))
            else:
                scored.append((node, 0.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _collect_leaves(self, node: TreeNode) -> List[TreeNode]:
        """Return all leaf descendants of *node*."""
        if node.is_leaf:
            return [node]
        leaves: List[TreeNode] = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves

    # ------------------------------------------------------------------
    # Internal – serialisation / visualisation
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the built tree."""
        from collections import Counter
        level_counts: Dict[int, int] = Counter(n.level for n in self._all_nodes)
        leaf_count = level_counts.get(0, 0)
        return {
            "num_leaf_nodes": leaf_count,
            "total_nodes": len(self._all_nodes),
            "num_levels": max(level_counts.keys(), default=0) + 1,
            "nodes_per_level": dict(sorted(level_counts.items())),
            "max_levels_cfg": self.max_levels,
            "cluster_size": self.cluster_size,
        }

    def _tree_to_dict(self, node: TreeNode) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "level": node.level,
            "text_snippet": node.text[:100],
            "num_children": len(node.children),
            "children": [self._tree_to_dict(c) for c in node.children],
        }

    def _ascii_tree(
        self,
        node: TreeNode,
        lines: List[str],
        prefix: str,
        is_last: bool,
    ) -> None:
        connector = "└── " if is_last else "├── "
        snippet = node.text[:60].replace("\n", " ")
        lines.append(f"{prefix}{connector}[L{node.level}] {node.node_id}: {snippet!r}")
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            self._ascii_tree(child, lines, child_prefix, i == len(node.children) - 1)
