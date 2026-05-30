"""Graph RAG model implementations.

All models are simulation-only: LLM calls are replaced by deterministic mock
responses derived from graph structure and embeddings.  Every model is
importable directly from this package.

Available models
----------------
MicrosoftGraphRAG
    Microsoft's GraphRAG pipeline (Edge et al. 2024).  Global search via
    map-reduce over Leiden communities; local search via PPR subgraph expansion.

RAPTOR
    Recursive Abstractive Processing for Tree-Organized Retrieval
    (Sarthi et al. 2024).  Hierarchical summary tree with tree-traversal and
    collapsed retrieval modes.

HippoRAG
    Hippocampus-inspired RAG (Gutierrez et al. 2024).  Builds a phrase-level
    KG and retrieves passages via Personalized PageRank.

KGRAG
    Classic Knowledge Graph-augmented RAG.  SPARQL-like entity lookup combined
    with dense passage retrieval.

MindMapRAG
    Mindmap-style hierarchical concept graph with hierarchical, BFS, and DFS
    retrieval strategies.

EdgeRAG
    Edge-conditioned RAG: relation types are first-class retrieval signals;
    edges are embedded and used to find relevant subgraphs.

GRetriever
    GNN-based retrieval via a Prize-Collecting Steiner Tree approximation that
    finds the minimal connected subgraph covering a query.
"""

from grag.models.microsoft_graphrag import MicrosoftGraphRAG
from grag.models.raptor import RAPTOR
from grag.models.hipporag import HippoRAG
from grag.models.kgrag import KGRAG
from grag.models.mindmap_rag import MindMapRAG
from grag.models.edge_rag import EdgeRAG
from grag.models.g_retriever import GRetriever
from grag.models.lightrag import LightRAG
from grag.models.think_on_graph import ThinkOnGraph
from grag.models.fastgraphrag import FastGraphRAG
from grag.models.surge import SURGE
from grag.models.struct_rag import StructRAG
from grag.models.subgraph_rag import SubgraphRAG
from grag.models.graph_reader import GraphReader
from grag.models.drift_search import DriftSearch

__all__ = [
    "MicrosoftGraphRAG",
    "RAPTOR",
    "HippoRAG",
    "KGRAG",
    "MindMapRAG",
    "EdgeRAG",
    "GRetriever",
    "LightRAG",
    "ThinkOnGraph",
    "FastGraphRAG",
    "SURGE",
    "StructRAG",
    "SubgraphRAG",
    "GraphReader",
    "DriftSearch",
]
