"""StructRAG simulation.

Based on:
  Li, Z. et al. (2024). "StructRAG: Boosting Knowledge Intensive Reasoning of
  LLMs via Inference-time Hybrid Information Structurization."
  NeurIPS 2024.

StructRAG converts unstructured documents into the most suitable structured
format for a given query — choosing among: graph, table, catalogue, or
algorithm (pseudocode). A router selects the best format at inference time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grag.core.graph import KnowledgeGraph
from grag.core.document import SimpleEntityExtractor
from grag.core.embeddings import MockEmbedder


class StructureType(str, Enum):
    GRAPH = "graph"
    TABLE = "table"
    CATALOGUE = "catalogue"
    ALGORITHM = "algorithm"
    CHUNK = "chunk"  # plain RAG baseline


@dataclass
class StructuredKnowledge:
    structure_type: StructureType
    content: Any
    routing_score: float
    description: str = ""


class StructRAG:
    """StructRAG: inference-time hybrid information structurisation.

    Router selects the best knowledge structure type per query:
      - GRAPH      → entity-relation graph (for relational queries)
      - TABLE      → attribute table (for comparison queries)
      - CATALOGUE  → hierarchical list (for enumeration queries)
      - ALGORITHM  → step-by-step procedure (for process queries)
      - CHUNK      → plain text chunks (fallback)
    """

    def __init__(self, default_structure: Optional[str] = None):
        self.default_structure = (
            StructureType(default_structure) if default_structure else None
        )
        self._embedder = MockEmbedder()
        self._extractor = SimpleEntityExtractor()

        self._graph = KnowledgeGraph()
        self._tables: List[Dict[str, Any]] = []
        self._catalogues: List[Dict[str, Any]] = []
        self._chunks: List[str] = []

        # Structure-type routing: query pattern → structure
        self._routing_keywords: Dict[StructureType, List[str]] = {
            StructureType.GRAPH: [
                "relationship", "related", "who", "works", "founded",
                "between", "connection", "link", "partner"
            ],
            StructureType.TABLE: [
                "compare", "comparison", "difference", "versus", "list",
                "all", "each", "attribute", "property"
            ],
            StructureType.CATALOGUE: [
                "enumerate", "what are", "types", "kinds", "examples",
                "give me", "mention", "name"
            ],
            StructureType.ALGORITHM: [
                "how", "steps", "process", "procedure", "algorithm",
                "workflow", "pipeline", "sequence"
            ],
        }

        self._indexed = False
        self._index_time: float = 0.0

    # ------------------------------------------------------------------
    # Indexing — build all structure types
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> None:
        t0 = time.time()
        self._chunks = list(documents)

        for i, doc in enumerate(documents):
            entities = self._extractor.extract_entities(doc)
            for ent in entities:
                eid = ent["text"].lower().replace(" ", "_")
                if not self._graph._graph.has_node(eid):
                    self._graph.add_entity(eid, ent["text"], ent["type"])
            relations = self._extractor.extract_relations(doc, entities)
            for rel in relations:
                src = rel["src"].lower().replace(" ", "_")
                dst = rel["dst"].lower().replace(" ", "_")
                for nid, label in [(src, rel["src"]), (dst, rel["dst"])]:
                    if not self._graph._graph.has_node(nid):
                        self._graph.add_entity(nid, label, "CONCEPT")
                self._graph.add_relation(src, dst, rel["relation"], weight=1.0)

            # Build table structure: entity × attribute
            table_rows = [
                {
                    "entity": ent["text"],
                    "type": ent["type"],
                    "doc_id": i,
                }
                for ent in entities[:5]
            ]
            if table_rows:
                self._tables.append({"doc_id": i, "rows": table_rows})

            # Build catalogue structure: hierarchical entity list
            types: Dict[str, List[str]] = {}
            for ent in entities:
                types.setdefault(ent["type"], []).append(ent["text"])
            if types:
                self._catalogues.append({"doc_id": i, "catalogue": types})

        self._indexed = True
        self._index_time = time.time() - t0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str,
                  force_structure: Optional[str] = None) -> Dict[str, Any]:
        """Route the query to the best structure and retrieve."""
        if not self._indexed:
            raise RuntimeError("Call index() first.")
        t0 = time.time()

        if force_structure:
            struct_type = StructureType(force_structure)
            routing_scores = {st: 0.0 for st in StructureType}
            routing_scores[struct_type] = 1.0
        elif self.default_structure:
            struct_type = self.default_structure
            routing_scores = {st: 0.0 for st in StructureType}
            routing_scores[struct_type] = 1.0
        else:
            routing_scores = self._route(query)
            struct_type = max(routing_scores.items(), key=lambda x: x[1])[0]

        structured_knowledge = self._build_structure(query, struct_type)
        answer = self._generate_answer(query, structured_knowledge, struct_type)

        return {
            "answer": answer,
            "structure_type": struct_type.value,
            "routing_scores": {k.value: round(v, 4)
                                for k, v in routing_scores.items()},
            "structured_knowledge": structured_knowledge,
            "query_time": round(time.time() - t0, 4),
        }

    def _route(self, query: str) -> Dict[StructureType, float]:
        """Keyword + embedding-based structure router."""
        q_lower = query.lower()
        scores: Dict[StructureType, float] = {}

        for stype, keywords in self._routing_keywords.items():
            keyword_score = sum(1.0 for kw in keywords if kw in q_lower)
            keyword_score /= max(len(keywords), 1)
            scores[stype] = keyword_score

        # Normalise
        total = sum(scores.values()) + 1e-8
        for k in scores:
            scores[k] /= total

        # Default fallback
        if scores.get(StructureType.GRAPH, 0) < 0.05:
            scores[StructureType.CHUNK] = max(
                scores.get(StructureType.CHUNK, 0.0), 0.1
            )
        return scores

    def _build_structure(self, query: str,
                          struct_type: StructureType) -> Any:
        if struct_type == StructureType.GRAPH:
            return self._build_graph_structure(query)
        if struct_type == StructureType.TABLE:
            return self._build_table_structure(query)
        if struct_type == StructureType.CATALOGUE:
            return self._build_catalogue_structure(query)
        if struct_type == StructureType.ALGORITHM:
            return self._build_algorithm_structure(query)
        return self._build_chunk_structure(query)

    def _build_graph_structure(self, query: str) -> Dict[str, Any]:
        q_emb = self._embedder.embed(query)
        nodes = list(self._graph._graph.nodes())[:20]
        embs = np.stack([
            self._embedder.embed(self._graph._graph.nodes[n].get("label", n))
            for n in nodes
        ]) if nodes else np.zeros((0, 128))
        top_nodes: List[str] = []
        if len(nodes) > 0:
            idxs, _ = self._embedder.top_k_similar(
                q_emb, embs, k=min(6, len(nodes))
            )
            top_nodes = [nodes[i] for i in idxs]
        edges = [
            (src, data.get("relation", "?"), dst)
            for n in top_nodes
            for _, dst, data in self._graph._graph.out_edges(n, data=True)
            if dst in top_nodes
        ]
        return {"type": "graph", "nodes": top_nodes, "edges": edges[:10]}

    def _build_table_structure(self, query: str) -> Dict[str, Any]:
        rows = []
        for tbl in self._tables[:5]:
            rows.extend(tbl["rows"])
        return {"type": "table", "columns": ["entity", "type", "doc_id"],
                "rows": rows[:20]}

    def _build_catalogue_structure(self, query: str) -> Dict[str, Any]:
        catalogue: Dict[str, List[str]] = {}
        for cat in self._catalogues[:5]:
            for etype, labels in cat["catalogue"].items():
                catalogue.setdefault(etype, []).extend(labels)
        # Deduplicate
        catalogue = {k: list(set(v))[:5] for k, v in catalogue.items()}
        return {"type": "catalogue", "categories": catalogue}

    def _build_algorithm_structure(self, query: str) -> Dict[str, Any]:
        steps = [
            "1. Extract relevant entities from the knowledge base",
            "2. Retrieve relation triples connecting query entities",
            "3. Traverse the knowledge graph to collect evidence",
            "4. Rank evidence by relevance to query",
            "5. Synthesise final answer from ranked evidence",
        ]
        return {"type": "algorithm", "steps": steps}

    def _build_chunk_structure(self, query: str) -> Dict[str, Any]:
        q_emb = self._embedder.embed(query)
        if not self._chunks:
            return {"type": "chunk", "chunks": []}
        embs = np.stack([self._embedder.embed(c) for c in self._chunks])
        idxs, _ = self._embedder.top_k_similar(q_emb, embs, k=min(3, len(self._chunks)))
        return {
            "type": "chunk",
            "chunks": [self._chunks[i][:200] for i in idxs],
        }

    def _generate_answer(self, query: str, knowledge: Any,
                          struct_type: StructureType) -> str:
        if struct_type == StructureType.GRAPH:
            n = len(knowledge.get("nodes", []))
            e = len(knowledge.get("edges", []))
            return (f"[StructRAG/graph] Using a graph of {n} entities and "
                    f"{e} relations to answer: '{query[:80]}'")
        if struct_type == StructureType.TABLE:
            r = len(knowledge.get("rows", []))
            return (f"[StructRAG/table] Comparing {r} entity-attribute rows "
                    f"to answer: '{query[:80]}'")
        if struct_type == StructureType.CATALOGUE:
            cats = list(knowledge.get("categories", {}).keys())
            return (f"[StructRAG/catalogue] Categories [{', '.join(cats[:4])}] "
                    f"relevant to: '{query[:80]}'")
        if struct_type == StructureType.ALGORITHM:
            return (f"[StructRAG/algorithm] Step-by-step procedure to "
                    f"answer: '{query[:80]}'")
        chunks = knowledge.get("chunks", [])
        return (f"[StructRAG/chunk] Retrieved {len(chunks)} passages for: "
                f"'{query[:80]}'")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "num_tables": len(self._tables),
            "num_catalogues": len(self._catalogues),
            "num_chunks": len(self._chunks),
            "index_time_s": round(self._index_time, 3),
        }
