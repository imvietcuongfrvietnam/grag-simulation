#!/usr/bin/env python3
"""Community detection algorithm comparison simulation.

Compares three algorithms on the same knowledge graph:
  - Louvain (Blondel et al., 2008)
  - Leiden  (Traag et al., 2019)
  - Label Propagation (Raghavan et al., 2007)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from grag.data.sample_corpus import SAMPLE_DOCUMENTS
from grag.core.document import SimpleEntityExtractor
from grag.core.graph import KnowledgeGraph
from grag.algorithms.community.louvain import LouvainDetector
from grag.algorithms.community.leiden import LeidenDetector
from grag.algorithms.community.label_propagation import LabelPropagationDetector

console = Console()


def build_graph_from_corpus() -> KnowledgeGraph:
    extractor = SimpleEntityExtractor()
    graph = KnowledgeGraph()
    for doc in SAMPLE_DOCUMENTS:
        text = doc["content"]
        entities = extractor.extract_entities(text)
        for ent in entities:
            eid = ent["text"].lower().replace(" ", "_")
            if not graph._graph.has_node(eid):
                graph.add_entity(eid, ent["text"], ent["type"])
        relations = extractor.extract_relations(text, entities)
        for rel in relations:
            src = rel["src"].lower().replace(" ", "_")
            dst = rel["dst"].lower().replace(" ", "_")
            for nid, label in [(src, rel["src"]), (dst, rel["dst"])]:
                if not graph._graph.has_node(nid):
                    graph.add_entity(nid, label, "CONCEPT")
            graph.add_relation(src, dst, rel["relation"], weight=1.0)
    return graph


def main():
    console.print(Panel.fit(
        "[bold blue]Community Detection Algorithm Comparison[/bold blue]\n"
        "Louvain · Leiden · Label Propagation\n"
        "Applied to the same knowledge graph",
        border_style="blue",
    ))

    console.print("\n[bold yellow]Building Knowledge Graph from Corpus…[/bold yellow]")
    graph = build_graph_from_corpus()
    console.print(
        f"  Nodes: [cyan]{graph.num_nodes}[/cyan]  "
        f"Edges: [cyan]{graph.num_edges}[/cyan]"
    )

    algorithms = [
        ("Louvain", LouvainDetector()),
        ("Leiden", LeidenDetector()),
        ("Label Propagation", LabelPropagationDetector()),
    ]

    results = []
    for name, detector in algorithms:
        console.print(f"\n[bold yellow]Running {name}…[/bold yellow]")
        t0 = time.time()
        partition = detector.detect(graph)
        elapsed = time.time() - t0
        communities = detector.get_communities(graph)
        modularity = detector.modularity(graph, communities) if hasattr(
            detector, "modularity") else None

        sizes = sorted([len(v) for v in communities.values()], reverse=True)
        results.append({
            "name": name,
            "num_communities": len(communities),
            "modularity": modularity,
            "largest": sizes[0] if sizes else 0,
            "smallest": sizes[-1] if sizes else 0,
            "avg_size": round(sum(sizes) / max(len(sizes), 1), 1),
            "time": elapsed,
        })
        console.print(f"  Communities found: [cyan]{len(communities)}[/cyan]  "
                      f"Time: {elapsed:.4f}s")

        top3 = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        for cid, members in top3:
            labels = [
                graph._graph.nodes[m].get("label", m)
                for m in list(members)[:4]
            ]
            console.print(f"    Community {cid} ({len(members)} nodes): "
                          f"{', '.join(labels)}")

    # ── Comparison table ────────────────────────────────────────────
    console.print("\n")
    table = Table(title="Community Detection Comparison", show_lines=True)
    table.add_column("Algorithm", style="bold cyan")
    table.add_column("# Communities", justify="right")
    table.add_column("Modularity", justify="right")
    table.add_column("Largest", justify="right")
    table.add_column("Smallest", justify="right")
    table.add_column("Avg Size", justify="right")
    table.add_column("Time (s)", justify="right")

    for r in results:
        mod = f"{r['modularity']:.4f}" if r["modularity"] is not None else "—"
        table.add_row(
            r["name"],
            str(r["num_communities"]),
            mod,
            str(r["largest"]),
            str(r["smallest"]),
            str(r["avg_size"]),
            f"{r['time']:.4f}",
        )
    console.print(table)

    console.print("\n[bold]Key Differences:[/bold]")
    console.print(
        "  • [cyan]Louvain[/cyan]: Fast, greedy modularity maximisation. "
        "May produce disconnected communities.\n"
        "  • [cyan]Leiden[/cyan]: Guarantees well-connected communities via "
        "a refinement phase. Preferred in GraphRAG.\n"
        "  • [cyan]Label Propagation[/cyan]: Near-linear time, "
        "non-deterministic, good for large sparse graphs."
    )
    console.print("\n[bold green]Community detection simulation complete.[/bold green]")


if __name__ == "__main__":
    main()
