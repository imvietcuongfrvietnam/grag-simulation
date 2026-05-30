#!/usr/bin/env python3
"""Simulation of the 8 advanced Graph RAG algorithms.

Covers:
  LightRAG       (Guo et al. 2024)       -- dual-level local/global retrieval
  Think-on-Graph (Sun et al. 2024 ICLR)  -- LLM-guided beam search on KG
  FastGraphRAG   (circlemind 2024)        -- PageRank-based lightweight GraphRAG
  SURGE          (Kang et al. 2023 EACL)  -- iterative subgraph reranking
  StructRAG      (Li et al. 2024 NeurIPS) -- inference-time structure selection
  SubgraphRAG    (Li et al. 2024)         -- simple triple retrieval baseline
  GraphReader    (Li et al. 2024)         -- agent-based document graph navigation
  DRIFT Search   (Microsoft 2024)         -- dynamic query decomposition + local search
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

from grag.data.sample_corpus import SAMPLE_DOCUMENTS, SAMPLE_QUERIES
from grag.models.lightrag import LightRAG
from grag.models.think_on_graph import ThinkOnGraph
from grag.models.fastgraphrag import FastGraphRAG
from grag.models.surge import SURGE
from grag.models.struct_rag import StructRAG
from grag.models.subgraph_rag import SubgraphRAG
from grag.models.graph_reader import GraphReader
from grag.models.drift_search import DriftSearch

console = Console()
CORPUS = [doc["content"] for doc in SAMPLE_DOCUMENTS]
DEMO_QUERIES = SAMPLE_QUERIES[:4]


def section(title: str) -> None:
    console.print(f"\n[bold yellow]{title}[/bold yellow]")
    console.rule(style="yellow")


def run_and_show(name: str, model, query: str) -> dict:
    t0 = time.time()
    result = model.retrieve(query)
    elapsed = time.time() - t0
    ans = result.answer if hasattr(result, "answer") else result.get("answer", "?")
    console.print(f"  [cyan]{name}[/cyan]: {ans[:180]}")
    console.print(f"  [dim]({elapsed:.3f}s)[/dim]")
    return result if isinstance(result, dict) else result.__dict__


def main():
    console.print(Panel.fit(
        "[bold blue]Advanced Graph RAG Models — Simulation[/bold blue]\n"
        "LightRAG · Think-on-Graph · FastGraphRAG · SURGE\n"
        "StructRAG · SubgraphRAG · GraphReader · DRIFT Search",
        border_style="blue",
    ))

    console.print(
        f"\nCorpus: [cyan]{len(CORPUS)}[/cyan] documents  "
        f"Queries: [cyan]{len(DEMO_QUERIES)}[/cyan]\n"
    )

    # ── Instantiate & index all models ──────────────────────────────────────
    section("Indexing Phase")

    models = {}
    index_times = {}

    for name, cls, kwargs, index_method in [
        ("LightRAG (hybrid)",     LightRAG,      {"mode": "hybrid"},          "index"),
        ("LightRAG (local)",      LightRAG,      {"mode": "local"},           "index"),
        ("LightRAG (global)",     LightRAG,      {"mode": "global"},          "index"),
        ("Think-on-Graph",        ThinkOnGraph,  {"beam_width": 3, "max_depth": 4}, "index"),
        ("FastGraphRAG",          FastGraphRAG,  {"top_k_entities": 10},      "index"),
        ("SURGE",                 SURGE,         {"max_iterations": 3},       "index"),
        ("StructRAG",             StructRAG,     {},                          "index"),
        ("SubgraphRAG",           SubgraphRAG,   {"top_k_triples": 10},       "index"),
        ("GraphReader",           GraphReader,   {"max_steps": 5},            "index"),
        ("DRIFT Search",          DriftSearch,   {"max_sub_questions": 3},    "index"),
    ]:
        console.print(f"  Indexing [bold]{name}[/bold]...", end=" ")
        obj = cls(**kwargs)
        t0 = time.time()
        getattr(obj, index_method)(CORPUS)
        elapsed = time.time() - t0
        index_times[name] = elapsed
        models[name] = obj
        stats = obj.get_stats()
        nodes = stats.get("num_entities", stats.get("num_graph_nodes", "?"))
        edges = stats.get("num_relations", stats.get("num_graph_edges", "?"))
        console.print(
            f"[green]done[/green] ({elapsed:.2f}s) "
            f"| nodes={nodes} edges={edges}"
        )

    # ── LightRAG: compare three modes ────────────────────────────────────────
    section("LightRAG: Local vs Global vs Hybrid")
    q = DEMO_QUERIES[0]
    console.print(f"  Query: [italic]{q}[/italic]\n")
    for mode in ("local", "global", "hybrid"):
        key = f"LightRAG ({mode})"
        run_and_show(key, models[key], q)

    # ── Think-on-Graph: show reasoning chain ─────────────────────────────────
    section("Think-on-Graph: Beam Search Trace")
    q = DEMO_QUERIES[1]
    console.print(f"  Query: [italic]{q}[/italic]\n")
    tog_result = models["Think-on-Graph"].retrieve(q)
    console.print(f"  Seed entities: {tog_result.get('seed_entities', [])[:5]}")
    for it in tog_result.get("iterations", [])[:3]:
        console.print(
            f"  Depth {it['depth']}: "
            f"{len(it['beams'])} beams — "
            f"top path: {it['beams'][0]['path'] if it['beams'] else 'none'}"
        )
    console.print(f"  [bold]Answer:[/bold] {tog_result.get('answer','')[:200]}")

    # ── FastGraphRAG: incremental update demo ────────────────────────────────
    section("FastGraphRAG: Incremental Update")
    fgr = models["FastGraphRAG"]
    console.print(f"  Before update: {fgr.get_stats()['num_entities']} entities")
    new_docs = [
        "OpenGraph AI was founded by Dr. Elena Vasquez after she left NeuralTech "
        "to start a new venture focused on open-source knowledge graph tools.",
        "The OpenGraph Toolkit integrates with both CloudBase Inc and TechCogni "
        "Group for enterprise deployments across financial services.",
    ]
    fgr.update(new_docs)
    console.print(f"  After  update: {fgr.get_stats()['num_entities']} entities "
                  f"(+{fgr.get_stats()['incremental_updates']} incremental update)")
    r = fgr.retrieve("Who founded OpenGraph AI?")
    console.print(f"  Query result: {r['answer'][:160]}")

    # ── StructRAG: structure routing demo ────────────────────────────────────
    section("StructRAG: Structure Type Routing")
    struct_queries = [
        ("Who works at NeuralTech Corporation?",          "graph"),
        ("Compare TechFlow and CogniSystems technologies.", "table"),
        ("What types of AI companies are in the ecosystem?", "catalogue"),
        ("How does the Leiden community detection work?",   "algorithm"),
    ]
    sr = models["StructRAG"]
    route_table = Table(show_lines=True)
    route_table.add_column("Query", width=45)
    route_table.add_column("Selected Structure", style="cyan")
    route_table.add_column("Top Route Score", justify="right")
    for query, expected in struct_queries:
        result = sr.retrieve(query)
        selected = result["structure_type"]
        top_score = max(result["routing_scores"].values())
        match = "[green]✓[/green]" if selected == expected else "[yellow]~[/yellow]"
        route_table.add_row(
            query[:43] + "…",
            f"{selected} {match}",
            f"{top_score:.3f}",
        )
    console.print(route_table)

    # ── SURGE: iterative reranking trace ─────────────────────────────────────
    section("SURGE: Iterative Subgraph Reranking")
    q = DEMO_QUERIES[2]
    console.print(f"  Query: [italic]{q}[/italic]\n")
    surge_result = models["SURGE"].retrieve(q)
    for it in surge_result.get("iterations", []):
        console.print(
            f"  Iter {it['iteration']}: {it['num_candidates']} candidates "
            f"| avg score: {it['avg_score']:.4f} "
            f"| top: {it['top_nodes'][:2]}"
        )
    console.print(f"  [bold]Final subgraph:[/bold] "
                  f"{surge_result['evidence_subgraph']['nodes'][:4]}")

    # ── GraphReader: agent navigation steps ──────────────────────────────────
    section("GraphReader: Agent Navigation Steps")
    q = DEMO_QUERIES[3]
    console.print(f"  Query: [italic]{q}[/italic]\n")
    gr_result = models["GraphReader"].retrieve(q)
    step_table = Table(show_lines=True)
    step_table.add_column("Step", justify="center", width=5)
    step_table.add_column("Node Visited", width=30)
    step_table.add_column("Relevance", justify="right")
    step_table.add_column("Follow-up", width=50)
    for step in gr_result.get("steps", [])[:5]:
        step_table.add_row(
            str(step["step"]),
            str(step["node"])[:28],
            f"{step['relevance']:.4f}",
            step["follow_up"][:48] + "…",
        )
    console.print(step_table)
    console.print(f"  Nodes visited: {gr_result['nodes_visited']} | "
                  f"Chars read: {gr_result['total_chars_read']}")

    # ── DRIFT Search: sub-question decomposition ──────────────────────────────
    section("DRIFT Search: Query Decomposition")
    q = DEMO_QUERIES[0]
    console.print(f"  Query: [italic]{q}[/italic]\n")
    drift_result = models["DRIFT Search"].retrieve(q)
    for sq in drift_result.get("sub_questions", []):
        console.print(
            f"  [cyan]Sub-Q:[/cyan] {sq['sub_question'][:60]}\n"
            f"  [dim]→ {sq['answer'][:80]} (conf={sq['confidence']:.3f})[/dim]"
        )
    console.print(f"\n  [bold]Final answer:[/bold] {drift_result['answer'][:200]}")

    # ── Summary table ─────────────────────────────────────────────────────────
    section("Summary: All Advanced Models")
    summary = Table(title="Advanced Graph RAG — Benchmark Summary", show_lines=True)
    summary.add_column("Model", style="bold cyan", width=20)
    summary.add_column("Paper", width=35)
    summary.add_column("Key Idea", width=35)
    summary.add_column("Index (s)", justify="right")
    summary.add_column("KG Nodes", justify="right")

    rows = [
        ("LightRAG", "Guo et al. 2024",
         "Dual-level local+global retrieval", "LightRAG (hybrid)"),
        ("Think-on-Graph", "Sun et al. ICLR 2024",
         "Beam search guided by LLM oracle", "Think-on-Graph"),
        ("FastGraphRAG", "circlemind 2024",
         "PageRank prior, incremental updates", "FastGraphRAG"),
        ("SURGE", "Kang et al. EACL 2023",
         "Iterative GNN-style reranking", "SURGE"),
        ("StructRAG", "Li et al. NeurIPS 2024",
         "Inference-time structure routing", "StructRAG"),
        ("SubgraphRAG", "Li et al. 2024",
         "Simple triple similarity baseline", "SubgraphRAG"),
        ("GraphReader", "Li et al. 2024",
         "Agent navigates document graph", "GraphReader"),
        ("DRIFT Search", "Microsoft 2024",
         "Query decomposition + local search", "DRIFT Search"),
    ]
    for model_name, paper, idea, key in rows:
        m = models[key]
        s = m.get_stats()
        nodes = s.get("num_entities", s.get("num_graph_nodes", "?"))
        summary.add_row(
            model_name, paper, idea,
            f"{index_times.get(key, 0):.3f}",
            str(nodes),
        )
    console.print(summary)

    console.print("\n[bold green]Advanced models simulation complete.[/bold green]")


if __name__ == "__main__":
    main()
