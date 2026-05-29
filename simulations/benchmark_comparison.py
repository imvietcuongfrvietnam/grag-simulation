#!/usr/bin/env python3
"""Full benchmark comparing all 7 Graph RAG algorithms.

Metrics: indexing_time, avg_query_time, graph_nodes, graph_edges,
num_communities, subgraph_size, retrieval_depth.

Results are printed as a rich table and saved to benchmark_results.json.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from grag.data.sample_corpus import SAMPLE_DOCUMENTS, SAMPLE_QUERIES
from grag.evaluation.benchmark import GraphRAGBenchmark
from grag.models.microsoft_graphrag import MicrosoftGraphRAG
from grag.models.raptor import RAPTOR
from grag.models.hipporag import HippoRAG
from grag.models.kgrag import KGRAG
from grag.models.mindmap_rag import MindMapRAG
from grag.models.edge_rag import EdgeRAG
from grag.models.g_retriever import GRetriever

console = Console()
CORPUS = [doc["content"] for doc in SAMPLE_DOCUMENTS]


def make_wrapper(model, index_method, query_method):
    """Wrap a model so GraphRAGBenchmark can call index() and query()."""

    class _Wrapper:
        def index(self, documents):
            return getattr(model, index_method)(documents)

        def query(self, question):
            return getattr(model, query_method)(question)

        def get_stats(self):
            return model.get_stats() if hasattr(model, "get_stats") else {}

    return _Wrapper()


def main():
    console.print(Panel.fit(
        "[bold blue]Graph RAG Full Benchmark[/bold blue]\n"
        "7 algorithms · 20 documents · 10 queries",
        border_style="blue",
    ))

    benchmark = GraphRAGBenchmark(CORPUS, SAMPLE_QUERIES)

    ms_rag = MicrosoftGraphRAG(chunk_size=300)
    benchmark.register_system("Microsoft GraphRAG",
                               make_wrapper(ms_rag, "index", "global_search"))

    raptor = RAPTOR(max_levels=3)
    benchmark.register_system("RAPTOR",
                               make_wrapper(raptor, "build_tree",
                                            "retrieve_collapsed"))

    hippo = HippoRAG(ppr_alpha=0.85)
    benchmark.register_system("HippoRAG",
                               make_wrapper(hippo, "index", "retrieve"))

    kg = KGRAG(hop_depth=2)
    benchmark.register_system("KG-RAG",
                               make_wrapper(kg, "build_kg", "query"))

    mm = MindMapRAG(max_depth=3)
    benchmark.register_system("MindMap-RAG",
                               make_wrapper(mm, "build_mindmap", "retrieve"))

    edge = EdgeRAG()
    benchmark.register_system("Edge-RAG",
                               make_wrapper(edge, "index", "retrieve"))

    gret = GRetriever(prize_weight=1.0, cost_weight=0.1)
    benchmark.register_system("G-Retriever",
                               make_wrapper(gret, "index", "retrieve"))

    console.print("\n[bold yellow]Running benchmark…[/bold yellow]\n")
    all_results = benchmark.run_all()

    console.print(benchmark.report())

    out_path = os.path.join(os.path.dirname(__file__), "..",
                             "benchmark_results.json")
    benchmark.save_results(out_path)
    console.print(f"\n[dim]Results saved to {os.path.abspath(out_path)}[/dim]")
    console.print("\n[bold green]Benchmark complete.[/bold green]")


if __name__ == "__main__":
    main()
