#!/usr/bin/env python3
"""
Detailed RAPTOR simulation:
  - Build tree step by step showing levels
  - Show cluster assignments at each level
  - Compare tree traversal vs collapsed retrieval
  - Print ASCII tree
"""

import sys
import os
import time
import math
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.text import Text
from rich.tree import Tree
from rich import box

from grag.core.embeddings import MockEmbedder
from grag.data.sample_corpus import SAMPLE_DOCUMENTS, SAMPLE_QUERIES

console = Console()
rng = random.Random(42)


# ---------------------------------------------------------------------------
# RAPTOR data structures
# ---------------------------------------------------------------------------

class RaptorNode:
    """A node in the RAPTOR tree (leaf = original chunk, internal = summary)."""

    def __init__(
        self,
        node_id: str,
        text: str,
        level: int,
        cluster_id: int,
        children: Optional[List["RaptorNode"]] = None,
    ):
        self.node_id = node_id
        self.text = text
        self.level = level
        self.cluster_id = cluster_id
        self.children: List["RaptorNode"] = children or []
        self.embedding = None  # populated lazily


class RaptorTree:
    """Complete RAPTOR tree with levels."""

    def __init__(self):
        self.levels: Dict[int, List[RaptorNode]] = {}  # level -> nodes
        self.root: Optional[RaptorNode] = None
        self.all_nodes: List[RaptorNode] = []


# ---------------------------------------------------------------------------
# Tree construction helpers
# ---------------------------------------------------------------------------

def _chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 150) -> List[Dict[str, Any]]:
    """Split documents into fixed-size word chunks."""
    chunks = []
    for doc in documents:
        words = doc["content"].split()
        for i, start in enumerate(range(0, len(words), chunk_size)):
            chunk_words = words[start: start + chunk_size]
            chunks.append({
                "id": f"{doc['id']}_chunk{i}",
                "text": " ".join(chunk_words),
                "doc_id": doc["id"],
                "title": doc["title"],
            })
    return chunks


def _cluster_nodes(nodes: List[RaptorNode], num_clusters: int, embedder: MockEmbedder) -> Dict[int, List[RaptorNode]]:
    """Assign nodes to clusters using a deterministic hash-based approach (simulates GMM)."""
    clusters: Dict[int, List[RaptorNode]] = defaultdict(list)
    for node in nodes:
        emb = embedder.embed(node.text[:80])
        # Use the sign of the first `num_clusters` dimensions as cluster bits
        bits = [1 if emb[d] > 0 else 0 for d in range(min(num_clusters.bit_length(), len(emb)))]
        cluster_id = sum(b * (2 ** i) for i, b in enumerate(bits)) % num_clusters
        node.cluster_id = cluster_id
        clusters[cluster_id].append(node)
    # Ensure all cluster IDs 0..num_clusters-1 are present
    for c in range(num_clusters):
        if c not in clusters:
            clusters[c] = []
    return clusters


def _summarize_cluster(nodes: List[RaptorNode], cluster_id: int, level: int) -> str:
    """Produce a deterministic extractive summary for a cluster."""
    if not nodes:
        return f"[Empty cluster {cluster_id} at level {level}]"
    # Pick leading words from each child text
    parts = []
    for node in nodes[:3]:
        lead = " ".join(node.text.split()[:20])
        parts.append(lead)
    combined = ". ".join(parts)
    return f"[L{level}-C{cluster_id} Summary] {combined[:250]}"


def build_raptor_tree(
    documents: List[Dict[str, Any]],
    embedder: MockEmbedder,
    max_levels: int = 3,
    chunks_per_cluster: int = 4,
) -> Tuple[RaptorTree, Dict[str, Any]]:
    """Build a RAPTOR hierarchical tree from documents."""
    stats: Dict[str, Any] = {"levels": {}, "total_nodes": 0, "build_time": 0.0}
    t0 = time.perf_counter()

    tree = RaptorTree()
    node_counter = [0]

    def make_id(prefix: str) -> str:
        node_counter[0] += 1
        return f"{prefix}_{node_counter[0]}"

    # Level 0: leaf chunks
    raw_chunks = _chunk_documents(documents)
    level0_nodes = []
    for chunk in raw_chunks:
        node = RaptorNode(
            node_id=make_id("L0"),
            text=chunk["text"],
            level=0,
            cluster_id=0,
        )
        level0_nodes.append(node)

    tree.levels[0] = level0_nodes
    tree.all_nodes.extend(level0_nodes)
    stats["levels"][0] = {"nodes": len(level0_nodes), "clusters": 1}

    current_nodes = level0_nodes

    for level in range(1, max_levels + 1):
        if len(current_nodes) <= 2:
            break  # Tree fully condensed

        num_clusters = max(2, len(current_nodes) // chunks_per_cluster)
        clusters = _cluster_nodes(current_nodes, num_clusters, embedder)

        parent_nodes = []
        for cluster_id, children in clusters.items():
            if not children:
                continue
            summary_text = _summarize_cluster(children, cluster_id, level)
            parent = RaptorNode(
                node_id=make_id(f"L{level}"),
                text=summary_text,
                level=level,
                cluster_id=cluster_id,
                children=children,
            )
            parent_nodes.append(parent)

        tree.levels[level] = parent_nodes
        tree.all_nodes.extend(parent_nodes)
        stats["levels"][level] = {
            "nodes": len(parent_nodes),
            "clusters": len(set(n.cluster_id for n in parent_nodes)),
        }
        current_nodes = parent_nodes

    # Root is the single top-level node or a synthetic root
    top_level = max(tree.levels.keys())
    if len(tree.levels[top_level]) == 1:
        tree.root = tree.levels[top_level][0]
    else:
        root_text = _summarize_cluster(tree.levels[top_level], 0, top_level + 1)
        tree.root = RaptorNode(
            node_id=make_id("ROOT"),
            text=root_text,
            level=top_level + 1,
            cluster_id=0,
            children=tree.levels[top_level],
        )
        tree.levels[top_level + 1] = [tree.root]

    stats["total_nodes"] = len(tree.all_nodes) + 1  # +1 for root
    stats["build_time"] = time.perf_counter() - t0
    stats["num_levels"] = len(tree.levels)
    stats["num_leaves"] = len(tree.levels[0])
    return tree, stats


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def tree_traversal_retrieval(
    tree: RaptorTree,
    query: str,
    embedder: MockEmbedder,
    top_k: int = 5,
) -> Tuple[List[RaptorNode], float]:
    """Navigate from root to leaves, selecting best-matching child at each level."""
    t0 = time.perf_counter()
    query_emb = embedder.embed(query)

    def best_child(node: RaptorNode) -> Optional[RaptorNode]:
        if not node.children:
            return None
        scored = []
        for child in node.children:
            child_emb = embedder.embed(child.text[:80])
            sim = embedder.cosine_similarity(query_emb, child_emb)
            scored.append((sim, child))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    retrieved = []
    current = tree.root
    while current is not None and len(retrieved) < top_k:
        retrieved.append(current)
        current = best_child(current)

    elapsed = time.perf_counter() - t0
    return retrieved[:top_k], elapsed


def collapsed_retrieval(
    tree: RaptorTree,
    query: str,
    embedder: MockEmbedder,
    top_k: int = 5,
) -> Tuple[List[RaptorNode], float]:
    """Treat all tree nodes as a flat pool and select top-k by cosine similarity."""
    t0 = time.perf_counter()
    query_emb = embedder.embed(query)

    scored = []
    for node in tree.all_nodes:
        node_emb = embedder.embed(node.text[:80])
        sim = embedder.cosine_similarity(query_emb, node_emb)
        scored.append((sim, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    elapsed = time.perf_counter() - t0
    return [n for _, n in scored[:top_k]], elapsed


# ---------------------------------------------------------------------------
# ASCII Tree printer
# ---------------------------------------------------------------------------

def print_ascii_tree(tree: RaptorTree, max_nodes_per_level: int = 5) -> None:
    """Print an ASCII representation of the RAPTOR tree structure."""
    rich_tree = Tree("[bold cyan]RAPTOR Tree Root[/bold cyan]")

    def add_subtree(rich_node, raptor_node: RaptorNode, depth: int) -> None:
        if depth > 3:
            return
        for i, child in enumerate(raptor_node.children[:max_nodes_per_level]):
            preview = child.text[:50].replace("\n", " ")
            label = f"[L{child.level}-C{child.cluster_id}] {preview}..."
            child_branch = rich_node.add(label)
            if child.children:
                add_subtree(child_branch, child, depth + 1)
        remaining = len(raptor_node.children) - max_nodes_per_level
        if remaining > 0:
            rich_node.add(f"[dim]... {remaining} more children[/dim]")

    if tree.root:
        add_subtree(rich_tree, tree.root, 0)

    console.print(rich_tree)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation() -> Dict[str, Any]:
    console.print(Panel.fit(
        "[bold magenta]RAPTOR Simulation[/bold magenta]\n"
        "[dim]Recursive Abstractive Processing for Tree-Organized Retrieval[/dim]",
        border_style="magenta",
    ))

    results: Dict[str, Any] = {}
    embedder = MockEmbedder(dim=128, seed=42)
    docs = SAMPLE_DOCUMENTS[:12]

    # ------------------------------------------------------------------ #
    # PHASE 1: Build the Tree
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 1: Building the RAPTOR Tree[/bold yellow]")
    console.rule()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking documents and building tree...", total=None)
        tree, build_stats = build_raptor_tree(
            docs, embedder, max_levels=3, chunks_per_cluster=4
        )
        progress.stop()

    results["build_time"] = build_stats["build_time"]
    results["total_nodes"] = build_stats["total_nodes"]
    results["num_levels"] = build_stats["num_levels"]

    # Level summary table
    level_table = Table(title="RAPTOR Tree Level Statistics", box=box.ROUNDED)
    level_table.add_column("Level", style="cyan", justify="center")
    level_table.add_column("Type", style="dim")
    level_table.add_column("Nodes", style="green", justify="right")
    level_table.add_column("Clusters", style="yellow", justify="right")
    level_table.add_column("Avg text length", style="blue", justify="right")

    for lvl in sorted(tree.levels.keys()):
        nodes_at_level = tree.levels[lvl]
        level_type = "Leaf chunks" if lvl == 0 else (
            "Root" if lvl == max(tree.levels.keys()) else f"Summary (L{lvl})"
        )
        avg_len = (
            sum(len(n.text.split()) for n in nodes_at_level) // max(len(nodes_at_level), 1)
        )
        num_clusters = len(set(n.cluster_id for n in nodes_at_level))
        level_table.add_row(str(lvl), level_type, str(len(nodes_at_level)),
                            str(num_clusters), f"{avg_len} words")

    console.print(level_table)
    console.print(
        f"[green]Tree built: {build_stats['total_nodes']} total nodes, "
        f"{build_stats['num_levels']} levels, "
        f"build time: {build_stats['build_time']:.3f}s[/green]"
    )

    # ------------------------------------------------------------------ #
    # PHASE 2: Cluster Assignments
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 2: Cluster Assignments Per Level[/bold yellow]")
    console.rule()

    for lvl in sorted(tree.levels.keys()):
        if lvl == 0:
            continue
        nodes_at_level = tree.levels[lvl]
        cluster_dist: Dict[int, int] = defaultdict(int)
        for node in nodes_at_level:
            cluster_dist[node.cluster_id] += 1
        dist_str = " | ".join(f"C{c}:{cnt}" for c, cnt in sorted(cluster_dist.items())[:8])
        console.print(f"  [cyan]Level {lvl}:[/cyan] {dist_str}")

    # ------------------------------------------------------------------ #
    # PHASE 3: ASCII Tree Visualization
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 3: Tree Structure Visualization[/bold yellow]")
    console.rule()
    print_ascii_tree(tree, max_nodes_per_level=3)

    # ------------------------------------------------------------------ #
    # PHASE 4: Retrieval Comparison
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 4: Tree Traversal vs Collapsed Retrieval[/bold yellow]")
    console.rule()

    queries = SAMPLE_QUERIES[:4]
    comparison_rows = []

    for query in queries:
        console.print(f"\n[bold]Query:[/bold] [italic]{query[:70]}[/italic]")

        trav_nodes, trav_time = tree_traversal_retrieval(tree, query, embedder, top_k=5)
        coll_nodes, coll_time = collapsed_retrieval(tree, query, embedder, top_k=5)

        trav_levels = sorted(set(n.level for n in trav_nodes))
        coll_levels = sorted(set(n.level for n in coll_nodes))

        console.print(
            f"  [cyan]Tree traversal:[/cyan] retrieved {len(trav_nodes)} nodes "
            f"from levels {trav_levels}, time={trav_time:.4f}s"
        )
        console.print(
            f"  [yellow]Collapsed retrieval:[/yellow] retrieved {len(coll_nodes)} nodes "
            f"from levels {coll_levels}, time={coll_time:.4f}s"
        )

        comparison_rows.append({
            "query": query[:50],
            "trav_nodes": len(trav_nodes),
            "trav_levels": trav_levels,
            "trav_time": trav_time,
            "coll_nodes": len(coll_nodes),
            "coll_levels": coll_levels,
            "coll_time": coll_time,
        })

    results["retrieval_comparison"] = comparison_rows

    # Summary comparison table
    cmp_table = Table(title="Tree Traversal vs Collapsed Retrieval Summary", box=box.ROUNDED)
    cmp_table.add_column("Dimension", style="cyan")
    cmp_table.add_column("Tree Traversal", style="green")
    cmp_table.add_column("Collapsed Retrieval", style="yellow")

    avg_trav = sum(r["trav_time"] for r in comparison_rows) / max(len(comparison_rows), 1)
    avg_coll = sum(r["coll_time"] for r in comparison_rows) / max(len(comparison_rows), 1)

    cmp_table.add_row("Strategy", "Root-to-leaf greedy descent", "Flat pool similarity search")
    cmp_table.add_row("Nodes scanned", "O(levels)", f"O({build_stats['total_nodes']})")
    cmp_table.add_row("Levels accessed", "Top-down single path", "All levels simultaneously")
    cmp_table.add_row("Avg query time", f"{avg_trav:.4f}s", f"{avg_coll:.4f}s")
    cmp_table.add_row("Best for", "Hierarchical / thematic queries", "Specific fact retrieval")
    cmp_table.add_row("Index size", f"{build_stats['total_nodes']} nodes", f"{build_stats['total_nodes']} nodes")
    console.print(cmp_table)

    # Final stats
    console.print("\n[bold yellow]SIMULATION SUMMARY[/bold yellow]")
    console.rule()
    summary = Table(title="RAPTOR Simulation Results", box=box.DOUBLE)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="bold green", justify="right")
    summary.add_row("Documents indexed", str(len(docs)))
    summary.add_row("Leaf chunks (Level 0)", str(len(tree.levels[0])))
    summary.add_row("Total tree nodes", str(build_stats["total_nodes"]))
    summary.add_row("Tree levels", str(build_stats["num_levels"]))
    summary.add_row("Build time", f"{build_stats['build_time']:.3f}s")
    summary.add_row("Avg traversal query time", f"{avg_trav:.4f}s")
    summary.add_row("Avg collapsed query time", f"{avg_coll:.4f}s")
    console.print(summary)

    results["summary"] = {
        "docs": len(docs),
        "leaf_chunks": len(tree.levels[0]),
        "total_nodes": build_stats["total_nodes"],
        "num_levels": build_stats["num_levels"],
        "build_time": build_stats["build_time"],
        "avg_trav_time": avg_trav,
        "avg_coll_time": avg_coll,
    }
    return results


if __name__ == "__main__":
    run_simulation()
