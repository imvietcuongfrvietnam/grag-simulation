#!/usr/bin/env python3
"""Detailed RAPTOR simulation.

Step-by-step walkthrough:
  1. Embed leaf chunks
  2. GMM clustering at each level
  3. Summarise clusters → parent nodes
  4. Build tree recursively
  5. Compare tree-traversal vs collapsed retrieval
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from grag.data.sample_corpus import SAMPLE_DOCUMENTS, SAMPLE_QUERIES
from grag.models.raptor import RAPTOR

console = Console()
CORPUS = [doc["content"] for doc in SAMPLE_DOCUMENTS]


def main():
    console.print(Panel.fit(
        "[bold blue]RAPTOR — Step-by-Step Simulation[/bold blue]\n"
        "Paper: Sarthi et al. (2024) 'RAPTOR: Recursive Abstractive Processing\n"
        "        for Tree-Organized Retrieval'  (ICLR 2024)",
        border_style="blue",
    ))

    raptor = RAPTOR(max_levels=4, cluster_size=5)

    # ── Build tree ──────────────────────────────────────────────────
    console.print("\n[bold yellow]Phase 1: Building RAPTOR Tree[/bold yellow]")
    t0 = time.time()
    tree = raptor.build_tree(CORPUS)
    elapsed = time.time() - t0

    stats = raptor.get_stats()
    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column("key", style="bold")
    info.add_column("value")
    info.add_row("Leaf nodes (raw chunks)", str(stats.get("num_leaf_nodes", "—")))
    info.add_row("Total tree nodes", str(stats.get("total_nodes", "—")))
    info.add_row("Tree levels", str(stats.get("num_levels", "—")))
    info.add_row("Build time", f"{elapsed:.3f}s")
    console.print(info)

    # ── ASCII tree ──────────────────────────────────────────────────
    console.print("\n[bold yellow]Phase 2: Tree Structure (ASCII)[/bold yellow]")
    console.print(raptor.visualize_tree())

    # ── Retrieval comparison ────────────────────────────────────────
    console.print("\n[bold yellow]Phase 3: Retrieval Comparison[/bold yellow]")
    queries = SAMPLE_QUERIES[:3]
    comp = Table(title="Tree Traversal vs Collapsed Retrieval", show_lines=True)
    comp.add_column("Query", width=40)
    comp.add_column("Tree Traversal", width=50)
    comp.add_column("Collapsed", width=50)
    comp.add_column("Traversal\nTime (s)", justify="right")
    comp.add_column("Collapsed\nTime (s)", justify="right")

    for q in queries:
        t1 = time.time()
        trav = raptor.retrieve_tree_traversal(q, top_k=3)
        tt = time.time() - t1

        t2 = time.time()
        coll = raptor.retrieve_collapsed(q, top_k=3)
        ct = time.time() - t2

        trav_preview = trav[0].get("text", "")[:80] + "…" if trav else "(none)"
        coll_preview = coll[0].get("text", "")[:80] + "…" if coll else "(none)"
        comp.add_row(q[:38] + "…", trav_preview, coll_preview,
                     f"{tt:.3f}", f"{ct:.3f}")

    console.print(comp)

    # ── Level distribution ──────────────────────────────────────────
    console.print("\n[bold yellow]Phase 4: Node Count per Level[/bold yellow]")
    level_table = Table(show_header=True)
    level_table.add_column("Level", style="bold")
    level_table.add_column("Node Count", justify="right")
    level_table.add_column("Description")

    level_counts = stats.get("nodes_per_level", {})
    descriptions = {
        0: "Raw document chunks (leaves)",
        1: "First-level cluster summaries",
        2: "Second-level cluster summaries",
        3: "Third-level cluster summaries",
        4: "Root summary",
    }
    for level in sorted(level_counts.keys()):
        level_table.add_row(
            str(level),
            str(level_counts[level]),
            descriptions.get(level, f"Level-{level} summaries"),
        )
    console.print(level_table)

    console.print("\n[bold green]RAPTOR simulation complete.[/bold green]")


if __name__ == "__main__":
    main()
