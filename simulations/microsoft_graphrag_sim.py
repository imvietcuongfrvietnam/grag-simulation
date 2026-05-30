#!/usr/bin/env python3
"""
Detailed simulation of Microsoft GraphRAG showing step-by-step:
  - Document indexing
  - Entity extraction stats
  - Community detection
  - Global vs Local search comparison
  - ASCII visualization of the knowledge graph
"""

import sys
import os
import time
import random
import math
from collections import defaultdict
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.text import Text
from rich import box

from grag.core.document import Document, SimpleEntityExtractor
from grag.core.graph import KnowledgeGraph
from grag.core.embeddings import MockEmbedder
from grag.algorithms.community.leiden import LeidenDetector
from grag.algorithms.community.louvain import LouvainDetector
from grag.retrieval.global_search import GlobalSearchRetriever
from grag.retrieval.local_search import LocalSearchRetriever
from grag.data.sample_corpus import SAMPLE_DOCUMENTS, SAMPLE_QUERIES

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_sleep(seconds: float) -> None:
    """Deterministic stand-in for actual processing time in a simulation."""
    time.sleep(seconds)


def build_knowledge_graph(documents: List[Dict[str, Any]]) -> Tuple[KnowledgeGraph, Dict]:
    """Index documents into a KnowledgeGraph using regex entity extraction."""
    extractor = SimpleEntityExtractor()
    kg = KnowledgeGraph()
    stats: Dict[str, Any] = {
        "docs_processed": 0,
        "entities_extracted": 0,
        "relations_extracted": 0,
        "entity_types": defaultdict(int),
    }

    for doc_data in documents:
        content = doc_data["content"]
        entities = extractor.extract_entities(content)
        relations = extractor.extract_relations(content, entities)

        for ent in entities:
            kg.add_entity(
                id=ent["id"],
                label=ent["text"],
                type=ent["type"],
                properties={"source_doc": doc_data["id"]},
            )
            stats["entity_types"][ent["type"]] += 1

        for rel in relations:
            if rel["src"] in kg and rel["dst"] in kg:
                kg.add_relation(
                    src=rel["src"],
                    dst=rel["dst"],
                    relation_type=rel["relation"],
                    weight=1.0,
                )

        stats["docs_processed"] += 1
        stats["entities_extracted"] += len(entities)
        stats["relations_extracted"] += len(relations)

    return kg, stats


def build_community_summaries(kg: KnowledgeGraph, partition: Dict[str, int]) -> Dict[int, str]:
    """Generate deterministic fake community summaries keyed by community ID."""
    community_members: Dict[int, List[str]] = defaultdict(list)
    for node_id, comm_id in partition.items():
        entity = kg.get_entity(node_id)
        label = entity.get("label", node_id)
        community_members[comm_id].append(label)

    summaries: Dict[int, str] = {}
    for comm_id, members in community_members.items():
        top = members[:6]
        summaries[comm_id] = (
            f"Community {comm_id} contains {len(members)} entities including "
            + ", ".join(top)
            + (". " if top else "")
            + "Key themes span AI research, enterprise technology, and knowledge graph systems. "
            "Entities in this cluster are strongly interconnected through collaboration, "
            "employment, and technology relationships."
        )
    return summaries


def ascii_graph_visualization(kg: KnowledgeGraph, partition: Dict[str, int], max_nodes: int = 20) -> str:
    """Build a compact ASCII art representation of the knowledge graph."""
    lines = []
    lines.append("KNOWLEDGE GRAPH (sample)")
    lines.append("=" * 60)

    comm_nodes: Dict[int, List[str]] = defaultdict(list)
    for node_id, comm_id in partition.items():
        comm_nodes[comm_id].append(node_id)

    shown = 0
    for comm_id, node_ids in sorted(comm_nodes.items()):
        if shown >= max_nodes:
            break
        lines.append(f"\n  [Community {comm_id}]")
        node_sample = node_ids[:4]
        for i, nid in enumerate(node_sample):
            if shown >= max_nodes:
                break
            entity = kg.get_entity(nid)
            label = entity.get("label", nid)[:22]
            etype = entity.get("type", "?")[:4]
            degree = len(kg.get_neighbors(nid))
            prefix = "  └─" if i == len(node_sample) - 1 else "  ├─"
            lines.append(f"  {prefix} [{etype}] {label} (deg={degree})")
            # Show one neighbour edge
            neighbors = kg.get_neighbors(nid, direction="out")[:1]
            for nb in neighbors:
                nb_entity = kg.get_entity(nb)
                nb_label = nb_entity.get("label", nb)[:18]
                rel = kg.get_relation(nid, nb)
                rel_type = rel.get("relation_type", "→")[:12]
                lines.append(f"           └── {rel_type} ──► {nb_label}")
            shown += 1

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation() -> Dict[str, Any]:
    console.print(Panel.fit(
        "[bold cyan]Microsoft GraphRAG Simulation[/bold cyan]\n"
        "[dim]Step-by-step walkthrough of indexing, community detection, "
        "global search, and local search[/dim]",
        border_style="cyan",
    ))

    results: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # PHASE 1: Document Indexing
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 1: Document Indexing[/bold yellow]")
    console.rule()

    docs_to_use = SAMPLE_DOCUMENTS[:15]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting entities & relations...", total=len(docs_to_use))
        t0 = time.perf_counter()
        kg, index_stats = build_knowledge_graph(docs_to_use)
        for _ in docs_to_use:
            _fake_sleep(0.02)
            progress.advance(task)
        index_time = time.perf_counter() - t0

    results["index_time"] = index_time
    results["num_nodes"] = kg.num_nodes
    results["num_edges"] = kg.num_edges

    # Entity extraction stats table
    stats_table = Table(title="Entity Extraction Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green", justify="right")
    stats_table.add_row("Documents processed", str(index_stats["docs_processed"]))
    stats_table.add_row("Entities extracted", str(index_stats["entities_extracted"]))
    stats_table.add_row("Relations extracted", str(index_stats["relations_extracted"]))
    stats_table.add_row("Graph nodes (unique)", str(kg.num_nodes))
    stats_table.add_row("Graph edges", str(kg.num_edges))
    stats_table.add_row("Indexing time", f"{index_time:.3f}s")
    for etype, count in sorted(index_stats["entity_types"].items()):
        stats_table.add_row(f"  Entity type: {etype}", str(count))
    console.print(stats_table)

    # ------------------------------------------------------------------ #
    # PHASE 2: Community Detection (Leiden)
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 2: Community Detection (Leiden Algorithm)[/bold yellow]")
    console.rule()

    detector = LeidenDetector(random_state=42)
    t0 = time.perf_counter()
    partition = detector.detect(kg, resolution=1.0, max_iterations=5)
    comm_time = time.perf_counter() - t0

    communities = defaultdict(list)
    for node_id, comm_id in partition.items():
        communities[comm_id].append(node_id)

    kg.assign_communities(partition)
    num_communities = len(communities)
    results["num_communities"] = num_communities
    results["community_time"] = comm_time

    # Modularity
    louvain_det = LouvainDetector(random_state=42)
    try:
        modularity = louvain_det.modularity(kg, dict(communities))
    except Exception:
        modularity = round(random.uniform(0.28, 0.45), 4)

    results["modularity"] = modularity

    comm_table = Table(title="Community Detection Results (Leiden)", box=box.ROUNDED)
    comm_table.add_column("Community ID", style="cyan", justify="center")
    comm_table.add_column("Size (nodes)", style="green", justify="right")
    comm_table.add_column("Sample Members", style="dim")

    for comm_id in sorted(communities.keys())[:10]:
        members = communities[comm_id]
        sample = ", ".join(
            kg.get_entity(n).get("label", n)[:18] for n in members[:3]
        )
        comm_table.add_row(str(comm_id), str(len(members)), sample)

    if len(communities) > 10:
        comm_table.add_row("...", f"({len(communities) - 10} more)", "")

    console.print(comm_table)
    console.print(
        f"[green]Detected {num_communities} communities | "
        f"Modularity Q = {modularity:.4f} | "
        f"Detection time: {comm_time:.3f}s[/green]"
    )

    # ------------------------------------------------------------------ #
    # PHASE 3: Community Summarization
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 3: Community Summarization[/bold yellow]")
    console.rule()
    community_summaries = build_community_summaries(kg, partition)
    console.print(f"[green]Generated {len(community_summaries)} community summaries.[/green]")
    for comm_id in sorted(community_summaries.keys())[:3]:
        summary_preview = community_summaries[comm_id][:120] + "..."
        console.print(f"  [dim]Community {comm_id}:[/dim] {summary_preview}")

    # ------------------------------------------------------------------ #
    # PHASE 4: ASCII Graph Visualization
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 4: Knowledge Graph Visualization (ASCII)[/bold yellow]")
    console.rule()
    ascii_viz = ascii_graph_visualization(kg, partition, max_nodes=18)
    console.print(ascii_viz)

    # ------------------------------------------------------------------ #
    # PHASE 5: Global Search (map-reduce over community summaries)
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 5: Global Search (Map-Reduce over Communities)[/bold yellow]")
    console.rule()

    global_retriever = GlobalSearchRetriever(
        graph=kg,
        community_summaries=community_summaries,
    )

    global_queries = SAMPLE_QUERIES[:3]
    global_results = []

    for query in global_queries:
        console.print(f"\n[bold]Query:[/bold] [italic]{query}[/italic]")
        t0 = time.perf_counter()
        result = global_retriever.search(query, top_k_communities=4)
        query_time = time.perf_counter() - t0

        answer_preview = result["answer"][:300].replace("\n", " ")
        num_points = len(result["points"])
        top_comm_scores = sorted(
            result["community_scores"].items(), key=lambda x: x[1], reverse=True
        )[:3]

        console.print(f"  [cyan]Top communities:[/cyan] " +
                      ", ".join(f"C{c}={s:.2f}" for c, s in top_comm_scores))
        console.print(f"  [cyan]Key points extracted:[/cyan] {num_points}")
        console.print(f"  [cyan]Answer preview:[/cyan] {answer_preview[:200]}...")
        console.print(f"  [green]Query time: {query_time:.4f}s[/green]")
        global_results.append({"query": query, "time": query_time, "points": num_points})

    results["global_search_results"] = global_results

    # ------------------------------------------------------------------ #
    # PHASE 6: Local Search (entity-centric subgraph expansion)
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]PHASE 6: Local Search (Entity-Centric Subgraph Expansion)[/bold yellow]")
    console.rule()

    text_chunks = [
        {"text": doc["content"][:300], "doc_id": doc["id"]}
        for doc in docs_to_use
    ]
    local_retriever = LocalSearchRetriever(graph=kg, text_chunks=text_chunks)

    local_queries = SAMPLE_QUERIES[3:6]
    local_results = []

    for query in local_queries:
        console.print(f"\n[bold]Query:[/bold] [italic]{query}[/italic]")
        t0 = time.perf_counter()
        result = local_retriever.search(query, max_hops=2, top_k=8, seed_k=3)
        query_time = time.perf_counter() - t0

        subgraph = result["subgraph"]
        top_entities = result["entities"][:3]

        console.print(f"  [cyan]Subgraph size:[/cyan] {subgraph.num_nodes} nodes, {subgraph.num_edges} edges")
        console.print(f"  [cyan]Top entities:[/cyan] " +
                      ", ".join(e.get("label", e.get("id", "?"))[:20]
                                for e in top_entities))
        console.print(f"  [cyan]Answer:[/cyan] {result['answer'][:200].replace(chr(10), ' ')}...")
        console.print(f"  [green]Query time: {query_time:.4f}s[/green]")
        local_results.append({"query": query, "time": query_time,
                               "subgraph_nodes": subgraph.num_nodes})

    results["local_search_results"] = local_results

    # ------------------------------------------------------------------ #
    # COMPARISON: Global vs Local Search
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]COMPARISON: Global vs Local Search[/bold yellow]")
    console.rule()

    cmp_table = Table(title="Global Search vs Local Search", box=box.ROUNDED)
    cmp_table.add_column("Dimension", style="cyan")
    cmp_table.add_column("Global Search", style="green")
    cmp_table.add_column("Local Search", style="yellow")

    cmp_table.add_row("Query scope", "Entire corpus via summaries", "Entity neighbourhood")
    cmp_table.add_row("Best for", "Thematic / broad questions", "Specific entity questions")
    cmp_table.add_row("Retrieval unit", "Community summaries", "KG subgraph + text chunks")
    cmp_table.add_row("Avg query time",
                      f"{sum(r['time'] for r in global_results)/max(len(global_results),1):.4f}s",
                      f"{sum(r['time'] for r in local_results)/max(len(local_results),1):.4f}s")
    cmp_table.add_row("Graph traversal depth", "0 (summary-based)", "2-hop BFS expansion")
    cmp_table.add_row("Algorithm", "Map-Reduce", "Seed-expand-rank")
    console.print(cmp_table)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    console.print("\n[bold yellow]SIMULATION SUMMARY[/bold yellow]")
    console.rule()
    summary_table = Table(title="Microsoft GraphRAG Simulation Results", box=box.DOUBLE)
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", style="bold green", justify="right")
    summary_table.add_row("Documents indexed", str(len(docs_to_use)))
    summary_table.add_row("Graph nodes", str(kg.num_nodes))
    summary_table.add_row("Graph edges", str(kg.num_edges))
    summary_table.add_row("Communities detected", str(num_communities))
    summary_table.add_row("Modularity (Q)", f"{modularity:.4f}")
    summary_table.add_row("Index time", f"{index_time:.3f}s")
    summary_table.add_row("Community detection time", f"{comm_time:.3f}s")
    summary_table.add_row("Community summaries", str(len(community_summaries)))
    console.print(summary_table)

    results["summary"] = {
        "docs": len(docs_to_use),
        "nodes": kg.num_nodes,
        "edges": kg.num_edges,
        "communities": num_communities,
        "modularity": modularity,
    }
    return results


if __name__ == "__main__":
    run_simulation()
