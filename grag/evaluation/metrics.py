"""Retrieval and RAG evaluation metrics.

Covers standard IR metrics (Precision@k, Recall@k, MRR, NDCG@k) as well as
RAG-specific metrics (faithfulness, graph coverage) and a unified
:class:`RAGEvaluator` that compares multiple systems.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Standard IR metrics (module-level functions)
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
    """Precision at rank k.

    Fraction of the top-k retrieved items that are relevant.

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved item IDs (most relevant first).
    relevant:
        Collection of ground-truth relevant item IDs.
    k:
        Cut-off rank.

    Returns
    -------
    P@k in [0, 1].
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
    """Recall at rank k.

    Fraction of relevant items that appear in the top-k retrieved.

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved item IDs.
    relevant:
        Collection of ground-truth relevant item IDs.
    k:
        Cut-off rank.

    Returns
    -------
    R@k in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved: List[Any], relevant: List[Any]) -> float:
    """Mean Reciprocal Rank (MRR).

    The reciprocal of the rank at which the first relevant item appears.
    Returns 0 if no relevant item is found.

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved item IDs.
    relevant:
        Collection of ground-truth relevant item IDs.

    Returns
    -------
    MRR in [0, 1].
    """
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k (binary relevance).

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved item IDs.
    relevant:
        Collection of ground-truth relevant item IDs (all equally relevant).
    k:
        Cut-off rank.

    Returns
    -------
    nDCG@k in [0, 1].
    """
    if k <= 0 or not relevant:
        return 0.0

    relevant_set = set(relevant)
    top_k = retrieved[:k]

    # Actual DCG
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG – all relevant items retrieved first
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def faithfulness_score(answer: str, context: List[str]) -> float:
    """Estimate how faithfully an answer is grounded in the provided context.

    This is a lightweight, LLM-free approximation:
    for each sentence in *answer*, we check what fraction of its content words
    appear in any context passage.  The overall score is the average over
    answer sentences.

    In a production system this would be replaced by an NLI-based check
    (e.g. using a cross-encoder or a prompted LLM).

    Parameters
    ----------
    answer:
        The generated answer string.
    context:
        List of context passage strings used to produce the answer.

    Returns
    -------
    Faithfulness score in [0, 1].  Higher is more grounded.
    """
    import re

    if not answer or not context:
        return 0.0

    # Build a set of all content words appearing in context
    stop_words = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "has", "have", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "to", "of",
        "in", "on", "at", "by", "for", "with", "about", "as", "from",
        "that", "this", "it", "its", "not", "no", "so", "if", "then",
        "than", "also", "which", "who", "what", "when", "where", "how",
        "i", "we", "you", "he", "she", "they", "their", "our", "your",
    }

    context_words: set = set()
    for passage in context:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", passage.lower())
        context_words.update(t for t in tokens if t not in stop_words)

    if not context_words:
        return 0.0

    # Split answer into sentences and score each
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    sentence_scores: List[float] = []

    for sent in sentences:
        sent_words = re.findall(r"\b[a-zA-Z]{3,}\b", sent.lower())
        content_words = [w for w in sent_words if w not in stop_words]
        if not content_words:
            continue
        hits = sum(1 for w in content_words if w in context_words)
        sentence_scores.append(hits / len(content_words))

    if not sentence_scores:
        return 0.0
    return float(np.mean(sentence_scores))


def graph_coverage(subgraph: Any, full_graph: Any) -> float:
    """Fraction of full-graph nodes covered by the retrieved subgraph.

    Both arguments should be :class:`~grag.core.graph.KnowledgeGraph` instances
    or any object with a ``nodes`` property and a ``num_nodes`` attribute.

    Parameters
    ----------
    subgraph:
        The retrieved local subgraph.
    full_graph:
        The complete knowledge graph.

    Returns
    -------
    Coverage in [0, 1].
    """
    total = getattr(full_graph, "num_nodes", len(getattr(full_graph, "nodes", [])))
    if total == 0:
        return 0.0
    sub_nodes = set(getattr(subgraph, "nodes", []))
    full_nodes = set(getattr(full_graph, "nodes", []))
    overlap = len(sub_nodes & full_nodes)
    return overlap / total


# ---------------------------------------------------------------------------
# RAGEvaluator – unified evaluation class
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """Evaluate one or more RAG systems against a gold standard.

    Gold standard format
    --------------------
    A dict mapping query strings to dicts with:
        ``relevant_ids`` (list[str]) – IDs of ground-truth relevant entities/nodes.
        ``answer``        (str, optional) – Reference answer text.

    Example
    -------
    >>> gold = {
    ...     "Who founded OpenAI?": {
    ...         "relevant_ids": ["elon_musk", "sam_altman", "openai"],
    ...         "answer": "OpenAI was co-founded by Elon Musk and Sam Altman.",
    ...     }
    ... }
    >>> evaluator = RAGEvaluator(gold_standard=gold)
    >>> results = [{"query": "Who founded OpenAI?", "retrieved_ids": ["sam_altman", "openai"], "answer": "..."}]
    >>> metrics = evaluator.evaluate("my_system", results)
    """

    _DEFAULT_K = (1, 3, 5, 10)

    def __init__(self, gold_standard: Dict[str, Any]) -> None:
        """
        Parameters
        ----------
        gold_standard:
            Mapping of query -> {``relevant_ids``: list, ``answer``: str (opt)}.
        """
        self._gold = gold_standard
        self._system_results: Dict[str, List[Dict[str, Any]]] = {}
        self._system_metrics: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, system_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate *results* from a single system and store them.

        Parameters
        ----------
        system_name:
            A label for the system (used in comparison tables).
        results:
            List of result dicts.  Each dict must have:
              - ``query`` (str): the query string.
              - ``retrieved_ids`` (list[str]): ordered list of retrieved entity IDs.
              - ``answer`` (str, optional): the generated answer text.
              - ``context`` (list[str], optional): context passages used.

        Returns
        -------
        Dict with aggregate metric values (mean over all queries).
        """
        self._system_results[system_name] = results

        per_query: List[Dict[str, float]] = []

        for result in results:
            query = result.get("query", "")
            retrieved_ids = result.get("retrieved_ids", [])
            answer = result.get("answer", "")
            context = result.get("context", [])

            gold_entry = self._gold.get(query, {})
            relevant_ids = gold_entry.get("relevant_ids", [])
            gold_answer = gold_entry.get("answer", "")

            row: Dict[str, float] = {}

            # IR metrics at multiple k
            for k in self._DEFAULT_K:
                row[f"P@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
                row[f"R@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
                row[f"nDCG@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

            row["MRR"] = mrr(retrieved_ids, relevant_ids)

            # Faithfulness – use gold_answer as context if no explicit context
            ctx = context if context else ([gold_answer] if gold_answer else [])
            row["faithfulness"] = faithfulness_score(answer, ctx)

            per_query.append(row)

        if not per_query:
            metrics: Dict[str, Any] = {"system": system_name, "num_queries": 0}
        else:
            # Aggregate: mean over all queries
            all_keys = per_query[0].keys()
            metrics = {
                "system": system_name,
                "num_queries": len(per_query),
            }
            for key in all_keys:
                values = [q[key] for q in per_query]
                metrics[key] = round(float(np.mean(values)), 4)

        self._system_metrics[system_name] = metrics
        return metrics

    def compare_systems(self, all_results: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Any:
        """Compare all registered (or provided) systems in a single table.

        Parameters
        ----------
        all_results:
            Optional dict mapping system_name -> results list.  When provided,
            each system is evaluated fresh before comparison.  When None, uses
            previously :meth:`evaluate`-d systems.

        Returns
        -------
        A :class:`pandas.DataFrame` (if pandas is available) or a plain dict
        mapping system_name -> metrics dict.
        """
        if all_results:
            for name, results in all_results.items():
                self.evaluate(name, results)

        if not self._system_metrics:
            if _HAS_PANDAS:
                return pd.DataFrame()
            return {}

        if _HAS_PANDAS:
            rows = list(self._system_metrics.values())
            df = pd.DataFrame(rows).set_index("system")
            # Sort by nDCG@10 descending if that column exists
            if "nDCG@10" in df.columns:
                df = df.sort_values("nDCG@10", ascending=False)
            return df

        return dict(self._system_metrics)

    def summary_report(self) -> str:
        """Return a human-readable text summary of all evaluated systems."""
        if not self._system_metrics:
            return "No systems have been evaluated yet."

        lines = ["=" * 60, "RAG Evaluation Summary", "=" * 60]
        for system_name, metrics in self._system_metrics.items():
            lines.append(f"\nSystem: {system_name}")
            lines.append(f"  Queries evaluated: {metrics.get('num_queries', 0)}")
            for key, value in sorted(metrics.items()):
                if key in ("system", "num_queries"):
                    continue
                lines.append(f"  {key:<15} {value:.4f}")

        return "\n".join(lines)

    def per_query_scores(
        self, system_name: str, results: List[Dict[str, Any]], k: int = 10
    ) -> List[Dict[str, Any]]:
        """Return per-query metric breakdown for a single system.

        Parameters
        ----------
        system_name:
            Label for the system.
        results:
            List of result dicts (same format as :meth:`evaluate`).
        k:
            Primary cut-off rank for P@k, R@k, nDCG@k display.

        Returns
        -------
        List of dicts, one per query, with ``query``, ``P@k``, ``R@k``,
        ``nDCG@k``, ``MRR``, and ``faithfulness`` keys.
        """
        output: List[Dict[str, Any]] = []
        for result in results:
            query = result.get("query", "")
            retrieved_ids = result.get("retrieved_ids", [])
            answer = result.get("answer", "")
            context = result.get("context", [])

            gold_entry = self._gold.get(query, {})
            relevant_ids = gold_entry.get("relevant_ids", [])
            gold_answer = gold_entry.get("answer", "")
            ctx = context if context else ([gold_answer] if gold_answer else [])

            output.append(
                {
                    "query": query,
                    f"P@{k}": round(precision_at_k(retrieved_ids, relevant_ids, k), 4),
                    f"R@{k}": round(recall_at_k(retrieved_ids, relevant_ids, k), 4),
                    f"nDCG@{k}": round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4),
                    "MRR": round(mrr(retrieved_ids, relevant_ids), 4),
                    "faithfulness": round(faithfulness_score(answer, ctx), 4),
                    "num_retrieved": len(retrieved_ids),
                    "num_relevant": len(relevant_ids),
                }
            )
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_table(self, data: Dict[str, Dict[str, Any]]) -> str:
        """Format a metrics dict as a plain-text ASCII table."""
        if not data:
            return "(empty)"
        headers = ["System"] + [k for k in next(iter(data.values())).keys()
                                  if k not in ("system",)]
        col_widths = [max(len(h), 12) for h in headers]
        sep = "  ".join("-" * w for w in col_widths)
        header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        lines = [sep, header_row, sep]
        for system_name, metrics in data.items():
            vals = [system_name] + [
                f"{metrics.get(k, 0):.4f}" if isinstance(metrics.get(k), float) else str(metrics.get(k, ""))
                for k in headers[1:]
            ]
            lines.append("  ".join(str(v).ljust(w) for v, w in zip(vals, col_widths)))
        lines.append(sep)
        return "\n".join(lines)
