"""Microsoft GraphRAG simulation.

Based on:
  Edge, D. et al. (2024). "From Local to Global: A Graph RAG Approach to
  Query-Focused Summarization."  arXiv:2404.16130.

All LLM calls are replaced by deterministic mock responses derived from the
actual graph data so the pipeline produces plausible-looking output without
any external API access.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grag.core.document import SimpleEntityExtractor, TextChunk
from grag.core.embeddings import MockEmbedder
from grag.core.graph import KnowledgeGraph
from grag.algorithms.community.leiden import LeidenDetector
from grag.algorithms.traversal.pagerank import PersonalizedPageRank


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


class _Community:
    """Represents a detected community with a generated summary."""

    def __init__(
        self,
        community_id: int,
        members: List[str],
        summary: str,
        embedding: np.ndarray,
    ) -> None:
        self.id = community_id
        self.members = members
        self.summary = summary
        self.embedding = embedding

    def __repr__(self) -> str:
        return f"Community(id={self.id}, size={len(self.members)})"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MicrosoftGraphRAG:
    """Simulates Microsoft's GraphRAG pipeline.

    Pipeline
    --------
    1. Document chunking
    2. Entity & relation extraction
    3. Knowledge graph construction
    4. Community detection (Leiden algorithm)
    5. Community summarization (mock LLM)
    6. Global search – map-reduce over communities
    7. Local search – entity-centric with relationships

    Parameters
    ----------
    chunk_size:
        Number of words per text chunk.
    overlap:
        Number of words of overlap between consecutive chunks.
    community_level:
        Leiden resolution multiplier that controls community granularity.
        Higher values produce more, smaller communities.
    """

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 50,
        community_level: int = 2,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.community_level = community_level

        self._extractor = SimpleEntityExtractor()
        self._embedder = MockEmbedder(dim=128)
        self._leiden = LeidenDetector(random_state=42)
        self._ppr = PersonalizedPageRank(alpha=0.85)

        self._graph: Optional[KnowledgeGraph] = None
        self._chunks: List[TextChunk] = []
        self._communities: List[_Community] = []
        self._chunk_embeddings: Optional[np.ndarray] = None

        # Index counters (set after indexing)
        self._num_docs: int = 0
        self._index_built: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, documents: List[str]) -> KnowledgeGraph:
        """Run the full GraphRAG indexing pipeline.

        Parameters
        ----------
        documents:
            Raw text strings to index.

        Returns
        -------
        The constructed KnowledgeGraph.
        """
        self._num_docs = len(documents)

        # Step 1 – Chunk documents
        self._chunks = self._chunk_documents(documents)

        # Step 2 & 3 – Extract entities/relations and build KG
        self._graph = self._build_knowledge_graph(self._chunks)

        # Step 4 – Community detection
        resolution = 0.5 + 0.25 * self.community_level
        partition = self._leiden.detect(self._graph, resolution=resolution)
        self._graph.assign_communities(partition)

        # Step 5 – Summarise communities (mock LLM)
        self._communities = self._summarise_communities(partition)

        # Pre-compute chunk embeddings for local search
        chunk_texts = [c.text for c in self._chunks]
        self._chunk_embeddings = self._embedder.embed_batch(chunk_texts)

        self._index_built = True
        return self._graph

    def global_search(self, query: str, top_k_communities: int = 5) -> Dict[str, Any]:
        """Global search using map-reduce over community summaries.

        Map phase: each community rates its relevance to the query.
        Reduce phase: aggregate the most relevant communities into an answer.

        Parameters
        ----------
        query:
            Natural language question.
        top_k_communities:
            Number of communities to use in the reduce phase.

        Returns
        -------
        Dict with keys:
            ``answer`` – synthesised text answer.
            ``communities_used`` – list of community IDs that contributed.
            ``confidence`` – float in [0, 1].
            ``map_scores`` – per-community relevance scores.
        """
        self._require_index()

        query_emb = self._embedder.embed(query)

        # Map phase: score each community by embedding similarity
        map_scores: List[Tuple[int, float]] = []
        for comm in self._communities:
            score = MockEmbedder.cosine_similarity(query_emb, comm.embedding)
            # Normalise to [0, 1]
            score = (score + 1.0) / 2.0
            map_scores.append((comm.id, score))

        map_scores.sort(key=lambda x: x[1], reverse=True)
        top_comms = map_scores[: top_k_communities]

        # Reduce phase: aggregate top community summaries into an answer
        used_communities = [cid for cid, _ in top_comms]
        community_texts = [
            self._get_community_by_id(cid).summary
            for cid in used_communities
            if self._get_community_by_id(cid)
        ]
        answer = self._mock_reduce(query, community_texts)
        confidence = float(np.mean([s for _, s in top_comms])) if top_comms else 0.0

        return {
            "answer": answer,
            "communities_used": used_communities,
            "confidence": round(confidence, 4),
            "map_scores": {cid: round(s, 4) for cid, s in map_scores},
        }

    def local_search(self, query: str, top_k_entities: int = 10) -> Dict[str, Any]:
        """Local search using entity-centric subgraph expansion.

        Finds the most relevant entities to the query, then expands to their
        local neighbourhood and uses that subgraph context to answer.

        Parameters
        ----------
        query:
            Natural language question.
        top_k_entities:
            How many seed entities to start expansion from.

        Returns
        -------
        Dict with keys:
            ``answer`` – text answer.
            ``entities_used`` – list of entity IDs.
            ``subgraph_size`` – number of nodes in the local subgraph.
            ``paths`` – representative reasoning paths (list of node-id lists).
        """
        self._require_index()
        assert self._graph is not None

        query_emb = self._embedder.embed(query)

        # Score entities by embedding similarity to query
        entity_nodes = self._graph.nodes
        if not entity_nodes:
            return {"answer": "No entities indexed.", "entities_used": [],
                    "subgraph_size": 0, "paths": []}

        entity_labels = [
            self._graph.get_entity(n).get("label", n) for n in entity_nodes
        ]
        entity_embs = self._embedder.embed_batch(entity_labels)
        top_idx, top_scores = self._embedder.top_k_similar(
            query_emb, entity_embs, k=min(top_k_entities, len(entity_nodes))
        )
        seed_entities = [entity_nodes[i] for i in top_idx]

        # PPR-based subgraph expansion
        ppr_scores = self._ppr.run(self._graph, seed_nodes=seed_entities)
        relevant_nodes = [
            n for n, s in sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
            if s > 1e-4
        ][:30]  # cap subgraph size for answer generation

        # Extract representative paths between top-2 seed entities
        paths: List[List[str]] = []
        if len(seed_entities) >= 2:
            paths = self._graph.find_paths(seed_entities[0], seed_entities[1], max_hops=3)

        subgraph = self._graph.subgraph(relevant_nodes)
        answer = self._mock_local_answer(query, subgraph, seed_entities, paths)

        return {
            "answer": answer,
            "entities_used": seed_entities,
            "subgraph_size": subgraph.num_nodes,
            "paths": paths[:5],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        if not self._index_built:
            return {"indexed": False}

        assert self._graph is not None
        community_sizes = [len(c.members) for c in self._communities]
        return {
            "indexed": True,
            "num_documents": self._num_docs,
            "num_chunks": len(self._chunks),
            "num_entities": self._graph.num_nodes,
            "num_relations": self._graph.num_edges,
            "num_communities": len(self._communities),
            "avg_community_size": round(float(np.mean(community_sizes)), 2) if community_sizes else 0,
            "max_community_size": max(community_sizes) if community_sizes else 0,
            "community_level": self.community_level,
        }

    # ------------------------------------------------------------------
    # Internal helpers – chunking
    # ------------------------------------------------------------------

    def _chunk_documents(self, documents: List[str]) -> List[TextChunk]:
        """Split documents into overlapping word-level chunks."""
        chunks: List[TextChunk] = []
        for doc_idx, doc_text in enumerate(documents):
            words = doc_text.split()
            step = max(1, self.chunk_size - self.overlap)
            for start in range(0, max(1, len(words) - self.overlap), step):
                end = start + self.chunk_size
                chunk_words = words[start:end]
                if not chunk_words:
                    break
                chunk_text = " ".join(chunk_words)
                chunk_id = _stable_id(f"doc{doc_idx}-{start}")
                chunks.append(
                    TextChunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata={"doc_idx": doc_idx, "start_word": start},
                    )
                )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers – KG construction
    # ------------------------------------------------------------------

    def _build_knowledge_graph(self, chunks: List[TextChunk]) -> KnowledgeGraph:
        """Extract entities and relations from all chunks and build a KG."""
        kg = KnowledgeGraph()
        for chunk in chunks:
            entities = self._extractor.extract_entities(chunk.text)
            relations = self._extractor.extract_relations(chunk.text, entities)

            for ent in entities:
                kg.add_entity(
                    id=ent["id"],
                    label=ent["text"],
                    type=ent["type"],
                    properties={"source_chunk": chunk.id},
                )

            for rel in relations:
                # Ensure both nodes exist (they may appear only in relations)
                for node_id in (rel["src"], rel["dst"]):
                    if node_id not in kg:
                        kg.add_entity(node_id, label=node_id, type="CONCEPT")
                kg.add_relation(
                    src=rel["src"],
                    dst=rel["dst"],
                    relation_type=rel["relation"],
                    weight=1.0,
                )

        # Add co-occurrence edges for entities in the same chunk
        for chunk in chunks:
            ents = self._extractor.extract_entities(chunk.text)
            ent_ids = [e["id"] for e in ents]
            for i in range(len(ent_ids)):
                for j in range(i + 1, min(i + 5, len(ent_ids))):
                    a, b = ent_ids[i], ent_ids[j]
                    if a in kg and b in kg:
                        if not kg._nx.has_edge(a, b):
                            kg.add_relation(a, b, "CO_OCCURS", weight=0.5)

        return kg

    # ------------------------------------------------------------------
    # Internal helpers – community summarisation
    # ------------------------------------------------------------------

    def _summarise_communities(
        self, partition: Dict[str, int]
    ) -> List[_Community]:
        """Generate mock LLM summaries for each community."""
        comm_map: Dict[int, List[str]] = defaultdict(list)
        for node, comm_id in partition.items():
            comm_map[comm_id].append(node)

        communities: List[_Community] = []
        for comm_id, members in sorted(comm_map.items()):
            summary = self._mock_community_summary(comm_id, members)
            emb = self._embedder.embed(summary)
            communities.append(_Community(comm_id, members, summary, emb))

        return communities

    def _mock_community_summary(self, comm_id: int, members: List[str]) -> str:
        """Generate a plausible community summary from member entity labels."""
        assert self._graph is not None
        labels = [
            self._graph.get_entity(m).get("label", m) for m in members[:8]
        ]
        entity_types = [
            self._graph.get_entity(m).get("type", "CONCEPT") for m in members[:8]
        ]
        type_counter: Dict[str, int] = defaultdict(int)
        for t in entity_types:
            type_counter[t] += 1
        dominant_type = max(type_counter, key=type_counter.get)  # type: ignore

        label_str = ", ".join(labels[:5])
        extra = f" and {len(members) - 5} more entities" if len(members) > 5 else ""

        type_phrases = {
            "PERSON": "key individuals including",
            "ORG": "organizations such as",
            "LOCATION": "locations including",
            "EVENT": "events involving",
            "CONCEPT": "concepts related to",
        }
        phrase = type_phrases.get(dominant_type, "entities such as")

        return (
            f"Community {comm_id} contains {len(members)} entities. "
            f"This community groups {phrase} {label_str}{extra}. "
            f"The members are interconnected through shared relationships "
            f"and co-occurrence patterns in the source documents."
        )

    # ------------------------------------------------------------------
    # Internal helpers – answer generation (mock LLM)
    # ------------------------------------------------------------------

    def _mock_reduce(self, query: str, community_summaries: List[str]) -> str:
        """Simulate the map-reduce aggregation phase to produce a final answer."""
        if not community_summaries:
            return f"No relevant community context found for: {query}"

        # Extract key entity tokens from summaries for answer construction
        all_labels: List[str] = []
        for summary in community_summaries:
            # Pull entity names from the summary text
            matches = re.findall(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b", summary)
            all_labels.extend(matches[:3])

        unique_labels = list(dict.fromkeys(all_labels))[:6]
        label_str = ", ".join(unique_labels) if unique_labels else "various entities"

        query_lower = query.lower()
        if any(w in query_lower for w in ("who", "person", "people")):
            focus = "individuals and their roles"
        elif any(w in query_lower for w in ("where", "location", "place")):
            focus = "locations and geographic context"
        elif any(w in query_lower for w in ("what", "describe", "explain")):
            focus = "key concepts and relationships"
        elif any(w in query_lower for w in ("how", "process", "mechanism")):
            focus = "processes and mechanisms"
        else:
            focus = "the main themes"

        return (
            f"Based on analysis of {len(community_summaries)} community summaries, "
            f"the answer to '{query}' focuses on {focus}. "
            f"The most relevant entities identified are: {label_str}. "
            f"These entities form interconnected clusters that collectively address the query "
            f"through their shared relationships and contextual proximity in the knowledge graph."
        )

    def _mock_local_answer(
        self,
        query: str,
        subgraph: KnowledgeGraph,
        seed_entities: List[str],
        paths: List[List[str]],
    ) -> str:
        """Generate a plausible local-search answer from the subgraph context."""
        assert self._graph is not None
        seed_labels = [
            self._graph.get_entity(e).get("label", e) for e in seed_entities[:4]
        ]
        seed_str = ", ".join(seed_labels) if seed_labels else "unknown entities"

        path_desc = ""
        if paths:
            path_labels = [
                " -> ".join(
                    self._graph.get_entity(n).get("label", n) for n in path
                )
                for path in paths[:2]
            ]
            path_desc = f" Reasoning paths: {'; '.join(path_labels)}."

        return (
            f"Local search for '{query}' identified {subgraph.num_nodes} relevant entities "
            f"in a {subgraph.num_edges}-edge subgraph. "
            f"Key entities: {seed_str}.{path_desc} "
            f"The local context suggests these entities are directly relevant "
            f"to the query based on their neighbourhood structure."
        )

    # ------------------------------------------------------------------
    # Internal helpers – lookup utilities
    # ------------------------------------------------------------------

    def _get_community_by_id(self, comm_id: int) -> Optional[_Community]:
        for c in self._communities:
            if c.id == comm_id:
                return c
        return None

    def _require_index(self) -> None:
        if not self._index_built:
            raise RuntimeError("Call index() before searching.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _stable_id(text: str) -> str:
    """Return a short stable hex ID for *text*."""
    return hashlib.md5(text.encode()).hexdigest()[:12]
