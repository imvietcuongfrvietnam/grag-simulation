# Graph RAG Simulation

A pure-Python simulation of **Graph Retrieval-Augmented Generation (Graph RAG)** algorithms.  
All LLM calls are replaced with deterministic mock functions so the full pipeline runs  
without any API keys, GPU, or external models.

---

## Overview

Graph RAG augments language model queries with structured knowledge extracted from  
documents and stored in a knowledge graph.  This project implements the core algorithms  
from several influential papers and lets you compare them on the same corpus and query set.

### Why Graph RAG?

Standard dense-retrieval RAG treats a corpus as a flat bag of chunks.  Graph RAG goes  
further by:

- Extracting entities and relations to build a **knowledge graph**.
- Running **community detection** to cluster related entities.
- Enabling **global queries** ("What are the main themes?") via map-reduce over communities.
- Enabling **local queries** ("Who is Alice?") via entity-centric subgraph expansion.
- Combining **graph traversal + vector search** for hybrid retrieval.

---

## Architecture

```
grag-simulation/
│
├── grag/                         Core package
│   ├── core/
│   │   ├── graph.py              KnowledgeGraph (NetworkX-backed)
│   │   ├── document.py           Document, TextChunk, SimpleEntityExtractor
│   │   └── embeddings.py         MockEmbedder (deterministic hash vectors)
│   │
│   ├── algorithms/
│   │   ├── community/
│   │   │   ├── louvain.py        Louvain community detection
│   │   │   ├── leiden.py         Leiden algorithm (improved Louvain)
│   │   │   └── label_propagation.py  Label Propagation
│   │   └── traversal/
│   │       ├── pagerank.py       Personalized PageRank (PPR)
│   │       └── beam_search.py    Beam search on graph
│   │
│   ├── models/
│   │   └── microsoft_graphrag.py Full GraphRAG pipeline
│   │
│   ├── retrieval/
│   │   ├── global_search.py      Community map-reduce search
│   │   ├── local_search.py       Entity-centric subgraph retrieval
│   │   └── hybrid_search.py      Graph + vector fusion via RRF
│   │
│   ├── evaluation/
│   │   ├── metrics.py            P@k, R@k, MRR, nDCG, faithfulness
│   │   └── benchmark.py          Multi-system benchmarking harness
│   │
│   └── visualization/
│       └── graph_viz.py          Matplotlib + ASCII graph/tree plots
│
├── simulations/                  Example simulation scripts
├── requirements.txt
├── setup.py
└── README.md
```

### Data flow

```
Documents
    │
    ▼  chunk_documents()
Text Chunks
    │
    ▼  SimpleEntityExtractor
Entities + Relations
    │
    ▼  KnowledgeGraph.add_entity / add_relation
KnowledgeGraph (NetworkX DiGraph)
    │
    ├──▶  LeidenDetector / LouvainDetector ──▶ Communities
    │         │
    │         ▼  mock community summaries
    │     Community Summaries
    │
    ├──▶  GlobalSearchRetriever (map-reduce)
    ├──▶  LocalSearchRetriever  (entity BFS + PPR)
    └──▶  HybridSearchRetriever (PPR + cosine + RRF)
```

---

## Algorithms

### 1. Microsoft GraphRAG (`grag/models/microsoft_graphrag.py`)

Based on **Edge et al. (2024) "From Local to Global: A Graph RAG Approach to  
Query-Focused Summarization"** (arXiv:2404.16130).

| Phase | Description |
|-------|-------------|
| Chunking | Fixed-size overlapping word windows |
| Extraction | Regex entity/relation extractor (`SimpleEntityExtractor`) |
| Graph build | Co-occurrence + explicit relation edges |
| Community detection | Leiden algorithm with configurable resolution |
| Community summarisation | Mock LLM based on member entity labels |
| Global search | Map-reduce: score summaries, extract points, reduce |
| Local search | PPR-seeded entity expansion, subgraph context |

### 2. Louvain Community Detection (`grag/algorithms/community/louvain.py`)

Classic two-phase greedy modularity optimisation (Blondel et al., 2008).  
Each iteration alternates between local node movement and graph aggregation  
until no further modularity gain is possible.

**Key parameters:** `resolution` (higher → more, smaller communities).

### 3. Leiden Algorithm (`grag/algorithms/community/leiden.py`)

An extension of Louvain that guarantees well-connected communities by adding  
a **refinement phase** between local movement and aggregation (Traag et al., 2019).  
Prevents the "arbitrarily badly connected" communities that Louvain can produce.

**Key parameters:** `resolution`, `theta` (refinement randomness).

### 4. Label Propagation (`grag/algorithms/community/label_propagation.py`)

Near-linear time community detection: each node adopts the label held by the  
majority of its neighbours, iterated until convergence.

### 5. Personalized PageRank (`grag/algorithms/traversal/pagerank.py`)

Power-iteration PPR with a configurable teleport set and seed weights.  
Used by HippoRAG and the local/hybrid retrievers to rank nodes by contextual  
relevance to a query.

**Key parameters:** `alpha` (damping factor, default 0.85), `max_iter`, `tol`.

### 6. Global Search (`grag/retrieval/global_search.py`)

Community-level map-reduce search mirroring Microsoft GraphRAG:

1. Embed query; compute cosine similarity to each community summary.
2. **Map**: for each top-k community, score individual sentences.
3. **Reduce**: deduplicate and re-rank points using geometric mean of sentence  
   and community scores.
4. Synthesise a human-readable answer.

### 7. Local Search (`grag/retrieval/local_search.py`)

Entity-centric retrieval:

1. Embed query; find seed entities by cosine similarity to node labels.
2. BFS expansion to k-hop neighbourhood.
3. Score all subgraph nodes (embedding similarity + degree centrality boost).
4. Build a context window combining KG triples and relevant text passages.

### 8. RAPTOR (`grag/models/raptor.py`)

Based on **Sarthi et al. (2024) "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"** (ICLR 2024).

Builds a multi-level summary tree over text chunks via GMM-inspired k-means clustering.
Two retrieval modes: **tree traversal** (top-down) and **collapsed** (all levels at once).

| Phase | Description |
|-------|-------------|
| Level 0 | Raw text chunks (leaves) |
| Level 1..N | k-means cluster → mock abstractive summary |
| Root | Global summary of all clusters |
| Retrieval | Cosine similarity at each level; traverse or flatten |

### 9. HippoRAG (`grag/models/hipporag.py`)

Based on **Gutierrez et al. (2024) "HippoRAG: Neurologically Inspired Long-Term Memory for Large Language Models"** (NeurIPS 2024).

Inspired by the human hippocampal memory system:

| Component | Analogy |
|-----------|---------|
| LLM | Neocortex (pattern recognition) |
| Knowledge Graph | Hippocampus (associative index) |
| Retrieval Encoder | Parahippocampal region |

Indexing extracts OpenIE-style triples → phrase-level KG.
Retrieval runs PPR from query-matched seed entities, then re-ranks by recognition scoring.

### 10. KG-RAG (`grag/models/kgrag.py`)

Classic Knowledge Graph-augmented RAG combining structured SPARQL-like entity lookups
with dense passage retrieval.

1. Build KG from documents
2. Detect named entities in query → 1-hop neighborhood lookup
3. Dense retrieval for supporting passages
4. Combine structured triples + passages → generate answer

### 11. MindMap-RAG (`grag/models/mindmap_rag.py`)

Based on **Nair et al. (2024) "MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models"** (ACL 2024).

Builds a hierarchical concept map with branching factor up to 4.
Supports three traversal strategies:
- **Hierarchical**: top-down guided by relevance
- **Breadth-first**: broad coverage at each level
- **Depth-first**: deep focus on most relevant branch

### 12. Edge-RAG (`grag/models/edge_rag.py`)

Treats graph **edges (relations)** as first-class retrieval signals.
Each triple is embedded as natural language ("X relation Y") and indexed.
At query time, relevant edges are retrieved first, then their endpoint entities
and connected passages are ranked.

Particularly effective for relation-specific queries: "Who works at X?", "What technology does Y use?"

### 13. G-Retriever (`grag/models/g_retriever.py`)

Based on **He et al. (2024) "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering"** (NeurIPS 2024).

Formulates subgraph retrieval as a **Prize-Collecting Steiner Tree (PCST)** problem:
- **Node prizes**: proportional to embedding similarity to query
- **Edge costs**: inversely proportional to relation weight
- **PCST approximation**: greedy algorithm finds minimal connected subgraph

### 14. Hybrid Search (`grag/retrieval/hybrid_search.py`)

Fuses graph and vector rankings using **Reciprocal Rank Fusion (RRF)**  
(Cormack et al., SIGIR 2009):

```
RRF(d) = Σ_r  1 / (k + rank_r(d))
```

where `k = 60` by default.  A normalised-score fusion variant is also  
provided for comparison.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/example/grag-simulation.git
cd grag-simulation

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e .

# Or install only the runtime requirements
pip install -r requirements.txt
```

### Optional extras

```bash
pip install -e ".[viz]"       # matplotlib + seaborn for graph plotting
pip install -e ".[notebooks]" # Jupyter notebook support
pip install -e ".[dev]"       # pytest, mypy, black, etc.
```

---

## Usage Examples

### Basic Graph RAG Pipeline

```python
from grag.models.microsoft_graphrag import MicrosoftGraphRAG

corpus = [
    "Alice works at Acme Corp, which is located in San Francisco. "
    "Bob collaborated with Alice on the Quantum project.",
    "Acme Corp was founded by Dr. Carol White in 2005. "
    "The company is part of the TechGroup consortium.",
    "Bob is led by Alice in the AI research division. "
    "The division is headquartered in New York.",
]

model = MicrosoftGraphRAG(chunk_size=50, overlap=10, community_level=2)
graph = model.index(corpus)

print(model.get_stats())
# {'indexed': True, 'num_documents': 3, 'num_entities': 12, ...}

result = model.global_search("What organizations are mentioned?")
print(result["answer"])

result = model.local_search("Who is Alice?")
print(result["answer"])
```

### Community Detection

```python
from grag.core.graph import KnowledgeGraph
from grag.algorithms.community.leiden import LeidenDetector
from grag.algorithms.community.louvain import LouvainDetector

kg = KnowledgeGraph()
# ... populate kg ...

leiden = LeidenDetector(random_state=42)
partition = leiden.detect(kg, resolution=1.0)  # {node_id: community_id}

communities = leiden.get_communities(kg)  # {community_id: [node_id, ...]}
print(f"Detected {len(communities)} communities")
```

### Global Search

```python
from grag.retrieval.global_search import GlobalSearchRetriever

# community_summaries: dict mapping community_id -> summary text
retriever = GlobalSearchRetriever(graph=kg, community_summaries=summaries)
result = retriever.search("What are the main research themes?", top_k_communities=5)

print(result["answer"])
print(result["community_scores"])   # {community_id: relevance_score}
```

### Local Search

```python
from grag.retrieval.local_search import LocalSearchRetriever
from grag.core.embeddings import MockEmbedder

retriever = LocalSearchRetriever(
    graph=kg,
    embedder=MockEmbedder(),
    text_chunks=[{"text": chunk} for chunk in raw_chunks],
)
result = retriever.search("Who founded the company?", max_hops=2, top_k=10)

print(result["answer"])
print(result["subgraph"])           # KnowledgeGraph of local neighbourhood
print(result["context_window"])     # Structured + unstructured context
```

### Hybrid Search (Graph + Vector + RRF)

```python
from grag.retrieval.hybrid_search import HybridSearchRetriever

retriever = HybridSearchRetriever(graph=kg, alpha=0.5)
result = retriever.search("machine learning research", top_k=10)

print(result["answer"])
print(result["fused_ranking"])      # Entities ranked by RRF
print(result["rrf_scores"])         # Raw RRF scores

# Rank correlation between the two component systems
corr = retriever.rank_correlation(
    result["graph_ranking"],
    result["vector_ranking"]
)
print(f"Rank correlation: {corr:.3f}")
```

### Evaluation

```python
from grag.evaluation.metrics import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k,
    faithfulness_score, graph_coverage, RAGEvaluator,
)

gold_standard = {
    "Who founded Acme Corp?": {
        "relevant_ids": ["dr._carol_white", "acme_corp"],
        "answer": "Acme Corp was founded by Dr. Carol White.",
    },
}

evaluator = RAGEvaluator(gold_standard)
results = [
    {
        "query": "Who founded Acme Corp?",
        "retrieved_ids": ["dr._carol_white", "acme_corp", "bob"],
        "answer": "Dr. Carol White founded Acme Corp in 2005.",
    }
]

metrics = evaluator.evaluate("my_system", results)
print(metrics)
# {'system': 'my_system', 'P@10': 0.3, 'R@10': 1.0, 'MRR': 1.0, ...}

# Compare multiple systems
comparison_df = evaluator.compare_systems()
print(comparison_df)
```

### Benchmarking

```python
from grag.evaluation.benchmark import GraphRAGBenchmark
from grag.models.microsoft_graphrag import MicrosoftGraphRAG

corpus  = [...]   # list of document strings
queries = [...]   # list of query strings

bm = GraphRAGBenchmark(corpus, queries)
bm.register_system("GraphRAG-v1", MicrosoftGraphRAG(community_level=1))
bm.register_system("GraphRAG-v2", MicrosoftGraphRAG(community_level=2))

results = bm.run_all(verbose=True)
print(bm.report())
bm.save_results("results/benchmark.json")
```

### Visualization

```python
import matplotlib.pyplot as plt
from grag.visualization.graph_viz import (
    plot_knowledge_graph,
    plot_communities,
    plot_raptor_tree,
    ascii_graph,
    ascii_tree,
)

# Plot KG coloured by entity type
fig, ax = plt.subplots(figsize=(14, 10))
plot_knowledge_graph(kg, title="My Knowledge Graph", color_by="type", ax=ax)
plt.tight_layout()
plt.savefig("kg.png", dpi=150)

# Plot with community colours
communities = leiden.get_communities(kg)
plot_communities(kg, communities, title="Community Structure")
plt.savefig("communities.png", dpi=150)

# ASCII (no matplotlib needed)
print(ascii_graph(kg, max_nodes=15))
```

---

## Benchmark Results

Results on a synthetic 5-document / 10-query corpus (all numbers are  
placeholder values — run `GraphRAGBenchmark` on your own corpus for real figures).

| System | Graph Nodes | Graph Edges | Index Time (s) | Avg Query (s) | MRR | nDCG@10 |
|--------|-------------|-------------|----------------|---------------|-----|---------|
| GraphRAG community_level=1 | — | — | — | — | — | — |
| GraphRAG community_level=2 | — | — | — | — | — | — |
| LocalSearch max_hops=1 | — | — | — | — | — | — |
| LocalSearch max_hops=2 | — | — | — | — | — | — |
| HybridSearch alpha=0.3 | — | — | — | — | — | — |
| HybridSearch alpha=0.7 | — | — | — | — | — | — |

Run `GraphRAGBenchmark.run_all()` and `GraphRAGBenchmark.to_dataframe()` to  
populate this table with real measurements.

---

## Project Structure at a Glance

```
grag/
  core/         Foundational data structures (graph, embeddings, documents)
  algorithms/   Graph algorithms (community detection, traversal)
  models/       End-to-end RAG pipeline implementations
  retrieval/    Search strategies (global, local, hybrid)
  evaluation/   Metrics and benchmarking
  visualization/ Plotting utilities
simulations/    Ready-to-run simulation scripts
```

---

## References

1. **Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024).**  
   *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.*  
   arXiv:2404.16130.

2. **Blondel, V.D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).**  
   *Fast unfolding of communities in large networks.*  
   Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.

3. **Traag, V.A., Waltman, L., & van Eck, N.J. (2019).**  
   *From Louvain to Leiden: guaranteeing well-connected communities.*  
   Scientific Reports, 9(1), 5233.

4. **Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009).**  
   *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.*  
   SIGIR 2009, pp. 758–759.

5. **Page, L., Brin, S., Motwani, R., & Winograd, T. (1999).**  
   *The PageRank Citation Ranking: Bringing Order to the Web.*  
   Stanford InfoLab Technical Report.

6. **Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C.D. (2024).**  
   *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.*  
   arXiv:2401.18059.

7. **Gutierrez, B.J., et al. (2024).**  
   *HippoRAG: Neurologically Inspired Long-Term Memory for Large Language Models.*  
   arXiv:2405.14831.

---

## License

MIT License.  See `LICENSE` for details.
