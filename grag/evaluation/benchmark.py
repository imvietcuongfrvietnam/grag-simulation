"""Benchmarking harness for comparing multiple Graph RAG systems.

Usage
-----
    corpus  = ["Doc 1 text...", "Doc 2 text..."]
    queries = ["What is X?", "Who is Y?"]

    bm = GraphRAGBenchmark(corpus, queries)
    bm.register_system("GraphRAG", MicrosoftGraphRAG())
    results = bm.run_all()
    print(bm.report())
    bm.save_results("benchmark_results.json")
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Resource tracking helpers (cross-platform, no psutil required)
# ---------------------------------------------------------------------------


def _memory_mb() -> float:
    """Return current process RSS in megabytes (Linux /proc, fallback 0)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# GraphRAGBenchmark
# ---------------------------------------------------------------------------


class GraphRAGBenchmark:
    """Benchmark harness for multiple Graph RAG systems on the same corpus.

    Each registered system must implement:

    * ``index(documents: list[str])`` – build the internal index and return a
      knowledge graph (or any truthy value on success).
    * ``global_search(query: str, **kwargs)`` or ``search(query: str, **kwargs)``
      – answer a query and return a dict with at least an ``'answer'`` key.

    Tracked metrics per system
    --------------------------
    * ``indexing_time_s``    – wall-clock seconds for ``index()``.
    * ``indexing_memory_mb`` – approximate RSS increase during indexing.
    * ``avg_query_time_s``   – mean wall-clock seconds per query.
    * ``total_query_time_s`` – sum of all query times.
    * ``graph_nodes``        – number of nodes in the resulting graph.
    * ``graph_edges``        – number of edges in the resulting graph.
    * ``num_communities``    – number of communities (if graph has the attr).
    * ``answer_length_avg``  – average character length of returned answers.

    Parameters
    ----------
    corpus:
        List of raw document strings used for indexing.
    queries:
        List of query strings to run during benchmarking.
    search_method:
        Name of the method to call for querying (default ``'global_search'``).
        Falls back to ``'search'`` if the named method is not found.
    search_kwargs:
        Extra keyword arguments forwarded to every search call.
    """

    def __init__(
        self,
        corpus: List[str],
        queries: List[str],
        search_method: str = "global_search",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.corpus = corpus
        self.queries = queries
        self._search_method = search_method
        self._search_kwargs: Dict[str, Any] = search_kwargs or {}

        self._systems: Dict[str, Any] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._benchmark_timestamp: Optional[float] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_system(self, name: str, system: Any) -> None:
        """Register a system for benchmarking.

        Parameters
        ----------
        name:
            A short unique label (used in reports and output files).
        system:
            An object with ``index()`` and a search method.
        """
        self._systems[name] = system

    def unregister_system(self, name: str) -> None:
        """Remove a previously registered system."""
        self._systems.pop(name, None)
        self._results.pop(name, None)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_all(self, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """Run the full benchmark for all registered systems.

        Parameters
        ----------
        verbose:
            Print progress to stdout.

        Returns
        -------
        Dict mapping system_name -> result dict (same format as :meth:`run_one`).
        """
        self._benchmark_timestamp = time.time()
        self._results = {}

        for name, system in self._systems.items():
            if verbose:
                print(f"[Benchmark] Running: {name} ...")
            result = self.run_one(name, system, verbose=verbose)
            self._results[name] = result

        return self._results

    def run_one(
        self,
        name: str,
        system: Any,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run the benchmark for a single system and return its result dict.

        Parameters
        ----------
        name:
            System label.
        system:
            System object to benchmark.
        verbose:
            If True, print query-level progress.

        Returns
        -------
        Dict with keys:
            ``system_name``, ``indexing_time_s``, ``indexing_memory_mb``,
            ``graph_stats``, ``query_results``, ``avg_query_time_s``,
            ``total_query_time_s``, ``answer_length_avg``, ``errors``.
        """
        result: Dict[str, Any] = {
            "system_name": name,
            "indexing_time_s": 0.0,
            "indexing_memory_mb": 0.0,
            "graph_stats": {},
            "query_results": [],
            "avg_query_time_s": 0.0,
            "total_query_time_s": 0.0,
            "answer_length_avg": 0.0,
            "errors": [],
        }

        # ---- Indexing phase ----
        mem_before = _memory_mb()
        t0 = time.perf_counter()
        try:
            graph = system.index(self.corpus)
            result["indexing_time_s"] = round(time.perf_counter() - t0, 4)
            result["indexing_memory_mb"] = round(max(0.0, _memory_mb() - mem_before), 2)
            result["graph_stats"] = self._extract_graph_stats(system, graph)
        except Exception as exc:
            result["indexing_time_s"] = round(time.perf_counter() - t0, 4)
            error_msg = f"Indexing failed: {exc}"
            result["errors"].append(error_msg)
            if verbose:
                print(f"  [ERROR] {error_msg}")
            return result

        # ---- Query phase ----
        query_times: List[float] = []
        answer_lengths: List[int] = []
        search_fn = self._resolve_search_method(system)

        for q_idx, query in enumerate(self.queries):
            t1 = time.perf_counter()
            try:
                response = search_fn(query, **self._search_kwargs)
                elapsed = time.perf_counter() - t1

                answer = response.get("answer", "") if isinstance(response, dict) else str(response)
                query_times.append(elapsed)
                answer_lengths.append(len(answer))

                result["query_results"].append(
                    {
                        "query": query,
                        "answer": answer,
                        "time_s": round(elapsed, 4),
                        "response": response if isinstance(response, dict) else {"answer": answer},
                    }
                )

                if verbose:
                    print(f"  [{q_idx + 1}/{len(self.queries)}] {query[:50]!r} -> {elapsed:.3f}s")

            except Exception as exc:
                elapsed = time.perf_counter() - t1
                error_msg = f"Query {q_idx} failed ({query[:40]!r}): {exc}"
                result["errors"].append(error_msg)
                query_times.append(elapsed)
                result["query_results"].append(
                    {
                        "query": query,
                        "answer": "",
                        "time_s": round(elapsed, 4),
                        "error": str(exc),
                    }
                )
                if verbose:
                    print(f"  [ERROR] {error_msg}")

        # Aggregate query stats
        if query_times:
            result["avg_query_time_s"] = round(float(np.mean(query_times)), 4)
            result["total_query_time_s"] = round(float(np.sum(query_times)), 4)
        if answer_lengths:
            result["answer_length_avg"] = round(float(np.mean(answer_lengths)), 1)

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return a human-readable performance report for all run systems.

        Returns
        -------
        Multi-line string suitable for printing to a terminal.
        """
        if not self._results:
            return "No benchmark results available. Call run_all() first."

        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  Graph RAG Benchmark Report")
        lines.append(f"  Corpus size  : {len(self.corpus)} document(s)")
        lines.append(f"  Queries      : {len(self.queries)}")
        lines.append("=" * 72)

        for name, res in self._results.items():
            lines.append("")
            lines.append(f"System : {name}")
            lines.append("-" * 50)

            gs = res.get("graph_stats", {})
            lines.append(f"  Graph nodes       : {gs.get('nodes', 'N/A')}")
            lines.append(f"  Graph edges       : {gs.get('edges', 'N/A')}")
            lines.append(f"  Communities       : {gs.get('communities', 'N/A')}")
            lines.append(f"  Indexing time     : {res.get('indexing_time_s', 0):.3f} s")
            lines.append(f"  Indexing memory   : {res.get('indexing_memory_mb', 0):.1f} MB (approx)")
            lines.append(f"  Avg query time    : {res.get('avg_query_time_s', 0):.4f} s")
            lines.append(f"  Total query time  : {res.get('total_query_time_s', 0):.3f} s")
            lines.append(f"  Avg answer length : {res.get('answer_length_avg', 0):.0f} chars")

            errors = res.get("errors", [])
            if errors:
                lines.append(f"  Errors            : {len(errors)}")
                for e in errors[:3]:
                    lines.append(f"    - {e}")

        lines.append("")
        lines.append("=" * 72)

        # Comparison table
        lines.append(self._comparison_table())
        lines.append("=" * 72)

        return "\n".join(lines)

    def _comparison_table(self) -> str:
        """Render a side-by-side comparison of key metrics."""
        if not self._results:
            return ""

        metrics_to_show = [
            ("nodes", "Graph Nodes"),
            ("edges", "Graph Edges"),
            ("indexing_time_s", "Index Time (s)"),
            ("avg_query_time_s", "Avg Query (s)"),
            ("answer_length_avg", "Avg Answer (chars)"),
        ]

        col_w = max(len(name) for name in self._results) + 2
        header_col = 22
        sep = " | ".join(["-" * header_col] + ["-" * col_w] * len(self._results))
        h_names = " | ".join([" " * header_col] + [n.ljust(col_w) for n in self._results])

        lines = ["", "  Metric Comparison", sep, h_names, sep]

        for attr_key, label in metrics_to_show:
            row_parts = [label.ljust(header_col)]
            for res in self._results.values():
                gs = res.get("graph_stats", {})
                if attr_key in ("nodes", "edges"):
                    val = gs.get(attr_key, "N/A")
                else:
                    val = res.get(attr_key, "N/A")
                if isinstance(val, float):
                    row_parts.append(f"{val:.4f}".ljust(col_w))
                else:
                    row_parts.append(str(val).ljust(col_w))
            lines.append(" | ".join(row_parts))

        lines.append(sep)
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Return benchmark results as a pandas DataFrame.

        Returns
        -------
        :class:`pandas.DataFrame` with systems as rows.
        Raises :exc:`ImportError` if pandas is not installed.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe(). pip install pandas")

        rows = []
        for name, res in self._results.items():
            gs = res.get("graph_stats", {})
            row = {
                "system": name,
                "graph_nodes": gs.get("nodes", 0),
                "graph_edges": gs.get("edges", 0),
                "communities": gs.get("communities", 0),
                "indexing_time_s": res.get("indexing_time_s", 0.0),
                "indexing_memory_mb": res.get("indexing_memory_mb", 0.0),
                "avg_query_time_s": res.get("avg_query_time_s", 0.0),
                "total_query_time_s": res.get("total_query_time_s", 0.0),
                "answer_length_avg": res.get("answer_length_avg", 0.0),
                "num_errors": len(res.get("errors", [])),
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("system")
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, path: str) -> None:
        """Serialise all benchmark results to a JSON file.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created if needed.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Make the results JSON-safe (strip non-serialisable objects)
        safe_results: Dict[str, Any] = {}
        for name, res in self._results.items():
            safe_res = {k: v for k, v in res.items() if k != "query_results"}
            # Simplify query results to only serialisable fields
            safe_queries = []
            for qr in res.get("query_results", []):
                safe_qr = {
                    "query": qr.get("query", ""),
                    "answer": qr.get("answer", ""),
                    "time_s": qr.get("time_s", 0.0),
                }
                if "error" in qr:
                    safe_qr["error"] = qr["error"]
                safe_queries.append(safe_qr)
            safe_res["query_results"] = safe_queries
            safe_results[name] = safe_res

        payload = {
            "benchmark_timestamp": self._benchmark_timestamp,
            "corpus_size": len(self.corpus),
            "num_queries": len(self.queries),
            "queries": self.queries,
            "results": safe_results,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def load_results(cls, path: str) -> "GraphRAGBenchmark":
        """Load benchmark results from a JSON file (for offline reporting).

        Returns a :class:`GraphRAGBenchmark` with ``_results`` populated but
        no registered live systems.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        corpus = [""] * payload.get("corpus_size", 0)
        queries = payload.get("queries", [])
        bm = cls(corpus=corpus, queries=queries)
        bm._results = payload.get("results", {})
        bm._benchmark_timestamp = payload.get("benchmark_timestamp")
        return bm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_search_method(self, system: Any) -> Callable:
        """Return the appropriate search callable from a system object."""
        preferred = self._search_method
        if hasattr(system, preferred):
            return getattr(system, preferred)
        for fallback in ("global_search", "search", "query", "retrieve"):
            if hasattr(system, fallback):
                return getattr(system, fallback)
        raise AttributeError(
            f"System {system!r} has no recognised search method. "
            "Expected one of: global_search, search, query, retrieve."
        )

    @staticmethod
    def _extract_graph_stats(system: Any, graph: Any) -> Dict[str, Any]:
        """Extract graph statistics from a system or its returned graph object."""
        stats: Dict[str, Any] = {}

        # Try system.get_stats() first (MicrosoftGraphRAG exposes this)
        if hasattr(system, "get_stats") and callable(system.get_stats):
            try:
                sys_stats = system.get_stats()
                stats["nodes"] = sys_stats.get("num_entities", 0)
                stats["edges"] = sys_stats.get("num_relations", 0)
                stats["communities"] = sys_stats.get("num_communities", 0)
                stats["avg_community_size"] = sys_stats.get("avg_community_size", 0.0)
                return stats
            except Exception:
                pass

        # Fall back to direct graph attributes
        if graph is not None:
            stats["nodes"] = getattr(graph, "num_nodes", len(getattr(graph, "nodes", [])))
            stats["edges"] = getattr(graph, "num_edges", len(getattr(graph, "edges", [])))

        return stats
