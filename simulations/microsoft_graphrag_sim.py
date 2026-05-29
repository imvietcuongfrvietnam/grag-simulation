#!/usr/bin/env python3
"""Detailed Microsoft GraphRAG simulation.

Step-by-step walkthrough of the full Microsoft GraphRAG pipeline:
  1. Document chunking
  2. Entity & relation extraction
  3. Knowledge graph construction
  4. Leiden community detection
  5. Community summarisation
  6. Global search (map-reduce)
  7. Local search (entity-centric)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

from grag.data.sample_corpus import SAMPLE_DOCUMENTS, SAMPLE_QUERIES
from grag.models.microsoft_graphrag import MicrosoftGraphRAG

console = Console()
CORPUS = [doc["content"] for doc in SAMPLE_DOCUMENTS]
TITLES = {doc["content"]: doc["title"] for doc in SAMPLE_DOCUMENTS}


def main():
    console.print(Panel.fit(
        "[bold blue]Microsoft GraphRAG — Step-by-Step Simulation[/bold blue]\n"
        "Paper: Edge et al. (2024) 'From Local to Global: A Graph RAG Approach\n"
        "        to Query-Focused Summarization'",
        border_style="blue",
    ))

    rag = MicrosoftGraphRAG(chunk_size=300, overlap=50, community_level=2)

    # ── Step 1-4: Indexing ──────────────────────────────────────────
    console.print("\n[bold yellow]Phase 1: Indexing Pipeline[/bold yellow]")
    t0 = time.time()
    graph = rag.index(CORPUS)
    elapsed = time.time() - t0

    stats = rag.get_stats()
    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column("key", style="bold")
    info.add_column("value")
    info.add_row("Documents indexed", str(len(CORPUS)))
    info.add_row("Total chunks", str(stats.get("num_chunks", "—")))
    info.add_row("KG nodes (entities)", str(stats.get("num_entities", "—")))
    info.add_row("KG edges (relations)", str(stats.get("num_relations", "—")))
    info.add_row("Communities detected", str(stats.get("num_communities", "—")))
    info.add_row("Community levels", str(stats.get("community_levels", "—")))
    info.add_row("Index time", f"{elapsed:.3f}s")
    console.print(info)

    # ── Step 5: Community summaries ─────────────────────────────────
    console.print("\n[bold yellow]Phase 2: Community Summaries[/bold yellow]")
    summaries = rag._community_summaries if hasattr(rag, "_community_summaries") else {}
    if summaries:
        for cid, summary in list(summaries.items())[:4]:
            preview = summary[:160].replace("\n", " ")
            console.print(f"  [cyan]Community {cid}:[/cyan] {preview}...")
    else:
        console.print("  (community summaries stored internally)")

    # ── Step 6: Global search ───────────────────────────────────────
    console.print("\n[bold yellow]Phase 3: Global Search (Map-Reduce)[/bold yellow]")
    queries_global = SAMPLE_QUERIES[:2]
    for q in queries_global:
        console.print(f"\n  [bold]Q:[/bold] {q}")
        t1 = time.time()
        result = rag.global_search(q)
        qt = time.time() - t1
        console.print(f"  [bold]A:[/bold] {result.get('answer', '')[:300]}")
        console.print(
            f"  [dim]communities used: {result.get('communities_used', '?')}  "
            f"time: {qt:.3f}s[/dim]"
        )

    # ── Step 7: Local search ────────────────────────────────────────
    console.print("\n[bold yellow]Phase 4: Local Search (Entity-Centric)[/bold yellow]")
    queries_local = SAMPLE_QUERIES[2:4]
    for q in queries_local:
        console.print(f"\n  [bold]Q:[/bold] {q}")
        t2 = time.time()
        result = rag.local_search(q)
        qt = time.time() - t2
        console.print(f"  [bold]A:[/bold] {result.get('answer', '')[:300]}")
        entities = result.get("entities_used", [])[:5]
        console.print(
            f"  [dim]entities: {entities}  "
            f"subgraph size: {result.get('subgraph_size', '?')}  "
            f"time: {qt:.3f}s[/dim]"
        )

    # ── Global vs Local comparison ──────────────────────────────────
    console.print("\n[bold yellow]Phase 5: Global vs Local Comparison[/bold yellow]")
    test_q = SAMPLE_QUERIES[4]
    console.print(f"\n  [bold]Query:[/bold] {test_q}")
    g_res = rag.global_search(test_q)
    l_res = rag.local_search(test_q)
    comp = Table(title="Global vs Local", show_lines=True)
    comp.add_column("Metric", style="bold")
    comp.add_column("Global Search", style="green")
    comp.add_column("Local Search", style="cyan")
    comp.add_row("Answer length", str(len(g_res.get("answer", ""))),
                 str(len(l_res.get("answer", ""))))
    comp.add_row("Context source",
                 f"{g_res.get('communities_used', '?')} communities",
                 f"{len(l_res.get('entities_used', []))} entities")
    comp.add_row("Answer preview",
                 g_res.get("answer", "")[:80] + "…",
                 l_res.get("answer", "")[:80] + "…")
    console.print(comp)

    console.print("\n[bold green]Microsoft GraphRAG simulation complete.[/bold green]")


if __name__ == "__main__":
    main()
