#!/usr/bin/env python3
"""Master simulation runner.

Runs all 7 Graph RAG algorithms on the sample corpus and prints
a side-by-side comparison table using the ``rich`` library.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from grag.data.sample_corpus import SAMPLE_DOCUMENTS, SAMPLE_QUERIES
from grag.models.microsoft_graphrag import MicrosoftGraphRAG
from grag.models.raptor import RAPTOR
from grag.models.hipporag import HippoRAG
from grag.models.kgrag import KGRAG
from grag.models.mindmap_rag import MindMapRAG
from grag.models.edge_rag import EdgeRAG
from grag.models.g_retriever import GRetriever

console = Console()

DEMO_QUERIES = SAMPLE_QUERIES[:3]
CORPUS = [doc["content"] for doc in SAMPLE_DOCUMENTS]


def run_system(name: str, system, index_fn, query_fn) -> dict:
    console.print(f"  [cyan]Indexing[/cyan] {name}...", end=" ")
    t0 = time.time()
    index_fn()
    index_time = time.time() - t0
    console.print(f"[green]done[/green] ({index_time:.2f}s)")

    query_results = []
    total_query_time = 0.0
    for q in DEMO_QUERIES:
        t1 = time.time()
        result = query_fn(q)
        qt = time.time() - t1
        total_query_time += qt
        query_results.append({"query": q, "result": result, "time": qt})

    stats = system.get_stats() if hasattr(system, "get_stats") else {}
    return {
        "name": name,
        "index_time": index_time,
        "avg_query_time": total_query_time / len(DEMO_QUERIES),
        "query_results": query_results,
        "stats": stats,
    }


def main():
    console.print(Panel.fit(
        "[bold blue]Graph RAG Algorithm Simulation[/bold blue]\n"
        "Comparing 7 Graph RAG approaches on a fictional AI tech corpus",
        border_style="blue",
    ))

    console.print(f"\n[bold]Corpus:[/bold] {len(SAMPLE_DOCUMENTS)} documents  "
                  f"[bold]Queries:[/bold] {len(DEMO_QUERIES)}\n")

    systems = []

    # ── Microsoft GraphRAG ──────────────────────────────────────────
    ms_rag = MicrosoftGraphRAG(chunk_size=300, community_level=2)
    systems.append(run_system(
        "Microsoft GraphRAG", ms_rag,
        lambda: ms_rag.index(CORPUS),
        lambda q: ms_rag.global_search(q),
    ))

    # ── RAPTOR ──────────────────────────────────────────────────────
    raptor = RAPTOR(max_levels=3)
    systems.append(run_system(
        "RAPTOR", raptor,
        lambda: raptor.build_tree(CORPUS),
        lambda q: {"results": raptor.retrieve_collapsed(q)},
    ))

    # ── HippoRAG ────────────────────────────────────────────────────
    hippo = HippoRAG(ppr_alpha=0.85)
    systems.append(run_system(
        "HippoRAG", hippo,
        lambda: hippo.index(CORPUS),
        lambda q: {"results": hippo.retrieve(q)},
    ))

    # ── KGRAG ───────────────────────────────────────────────────────
    kg = KGRAG(hop_depth=2)
    systems.append(run_system(
        "KG-RAG", kg,
        lambda: kg.build_kg(CORPUS),
        lambda q: kg.query(q),
    ))

    # ── MindMap RAG ─────────────────────────────────────────────────
    mm = MindMapRAG(max_depth=3)
    systems.append(run_system(
        "MindMap-RAG", mm,
        lambda: mm.build_mindmap(CORPUS),
        lambda q: mm.retrieve(q),
    ))

    # ── Edge-RAG ────────────────────────────────────────────────────
    edge = EdgeRAG()
    systems.append(run_system(
        "Edge-RAG", edge,
        lambda: edge.index(CORPUS),
        lambda q: edge.retrieve(q),
    ))

    # ── G-Retriever ─────────────────────────────────────────────────
    gret = GRetriever(prize_weight=1.0, cost_weight=0.1)
    systems.append(run_system(
        "G-Retriever", gret,
        lambda: gret.index(CORPUS),
        lambda q: gret.retrieve(q),
    ))

    # ── Comparison table ────────────────────────────────────────────
    console.print("\n")
    table = Table(title="Algorithm Comparison", show_lines=True)
    table.add_column("Algorithm", style="bold cyan", width=18)
    table.add_column("Index (s)", justify="right")
    table.add_column("Avg Query (s)", justify="right")
    table.add_column("KG Nodes", justify="right")
    table.add_column("KG Edges", justify="right")
    table.add_column("Communities", justify="right")
    table.add_column("Extra", style="dim")

    for r in systems:
        s = r["stats"]
        nodes = str(s.get("num_entities", s.get("num_nodes", "-")))
        edges = str(s.get("num_relations", s.get("num_edges", "-")))
        comms = str(s.get("num_communities", s.get("tree_nodes", "-")))
        extra_keys = [k for k in s if k not in (
            "num_entities", "num_nodes", "num_relations",
            "num_edges", "num_communities", "tree_nodes",
            "index_time_s",
        )]
        extra = ", ".join(f"{k}={s[k]}" for k in extra_keys[:2])
        table.add_row(
            r["name"],
            f"{r['index_time']:.3f}",
            f"{r['avg_query_time']:.3f}",
            nodes, edges, comms, extra,
        )

    console.print(table)

    # ── Sample answer from first query ──────────────────────────────
    console.print(f"\n[bold]Sample query:[/bold] {DEMO_QUERIES[0]}\n")
    for r in systems[:3]:
        qr = r["query_results"][0]["result"]
        answer = (qr.get("answer") or
                  str(qr.get("results", ["(no answer)"])[0])[:120])
        console.print(f"[bold]{r['name']}:[/bold]")
        console.print(f"  {answer[:200]}\n")

    console.print("[bold green]Simulation complete.[/bold green]")


if __name__ == "__main__":
    main()
