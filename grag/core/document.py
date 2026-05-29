"""Document primitives and a regex-based mock entity/relation extractor."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A contiguous slice of a document with optional dense embedding."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Document:
    """A full document composed of text chunks."""

    id: str
    title: str
    content: str
    chunks: List[TextChunk] = field(default_factory=list)

    @classmethod
    def from_text(cls, title: str, content: str, chunk_size: int = 256) -> "Document":
        """Construct a Document by splitting *content* into fixed-size chunks."""
        doc_id = str(uuid.uuid4())
        doc = cls(id=doc_id, title=title, content=content)
        words = content.split()
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk = TextChunk(
                id=f"{doc_id}-chunk-{i // chunk_size}",
                text=chunk_text,
                metadata={"doc_id": doc_id, "chunk_index": i // chunk_size, "title": title},
            )
            doc.chunks.append(chunk)
        return doc


# ---------------------------------------------------------------------------
# Entity / relation extraction
# ---------------------------------------------------------------------------

_RELATION_PATTERNS: List[Dict[str, Any]] = [
    {
        "pattern": re.compile(
            r"(?P<src>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)"
            r"\s+(?:works?\s+at|employed\s+by|joined)\s+"
            r"(?P<dst>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            re.IGNORECASE,
        ),
        "relation": "WORKS_AT",
    },
    {
        "pattern": re.compile(
            r"(?P<src>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)"
            r"\s+(?:located\s+in|based\s+in|headquartered\s+in|situated\s+in)\s+"
            r"(?P<dst>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            re.IGNORECASE,
        ),
        "relation": "LOCATED_IN",
    },
    {
        "pattern": re.compile(
            r"(?P<src>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)"
            r"\s+(?:part\s+of|belongs?\s+to|member\s+of)\s+"
            r"(?P<dst>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            re.IGNORECASE,
        ),
        "relation": "PART_OF",
    },
    {
        "pattern": re.compile(
            r"(?P<src>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)"
            r"\s+(?:founded|created|established|started)\s+"
            r"(?P<dst>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            re.IGNORECASE,
        ),
        "relation": "FOUNDED",
    },
    {
        "pattern": re.compile(
            r"(?P<src>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)"
            r"\s+(?:led\s+by|directed\s+by|managed\s+by|headed\s+by)\s+"
            r"(?P<dst>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            re.IGNORECASE,
        ),
        "relation": "LED_BY",
    },
    {
        "pattern": re.compile(
            r"(?P<src>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)"
            r"\s+(?:collaborated?\s+with|partnered?\s+with|worked\s+with)\s+"
            r"(?P<dst>[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            re.IGNORECASE,
        ),
        "relation": "COLLABORATED_WITH",
    },
]

_TYPE_HINTS: Dict[str, str] = {
    "Inc": "ORG",
    "Corp": "ORG",
    "Ltd": "ORG",
    "LLC": "ORG",
    "University": "ORG",
    "Institute": "ORG",
    "Foundation": "ORG",
    "Conference": "EVENT",
    "Summit": "EVENT",
    "Forum": "EVENT",
    "City": "LOCATION",
    "Town": "LOCATION",
    "Street": "LOCATION",
    "Avenue": "LOCATION",
    "Road": "LOCATION",
    "Boulevard": "LOCATION",
}

_PERSON_NAME_RE = re.compile(
    r"\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
)
_CAPS_TOKEN_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})(?:\s+[A-Z][a-zA-Z]{2,})*\b")


def _guess_entity_type(text: str) -> str:
    """Heuristically determine entity type from surface form."""
    for suffix, etype in _TYPE_HINTS.items():
        if suffix.lower() in text.lower():
            return etype
    if _PERSON_NAME_RE.search(text):
        return "PERSON"
    tokens = text.split()
    if len(tokens) == 2 and all(t[0].isupper() for t in tokens):
        return "PERSON"
    if len(tokens) == 1 and text[0].isupper() and text.islower() is False:
        return "CONCEPT"
    return "CONCEPT"


class SimpleEntityExtractor:
    """Regex-driven entity and relation extractor that requires no LLM.

    All outputs are deterministic given the same input text.  Entity IDs are
    stable slugs derived from the surface form so that duplicate mentions map
    to the same node.
    """

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from *text*.

        Returns a list of dicts with keys: ``id``, ``text``, ``type``,
        ``start``, ``end``.
        """
        seen: Dict[str, Dict[str, Any]] = {}

        for match in _PERSON_NAME_RE.finditer(text):
            surface = match.group(0).strip()
            entity_id = self._make_id(surface)
            if entity_id not in seen:
                seen[entity_id] = {
                    "id": entity_id,
                    "text": surface,
                    "type": "PERSON",
                    "start": match.start(),
                    "end": match.end(),
                }

        for match in _CAPS_TOKEN_RE.finditer(text):
            surface = match.group(0).strip()
            if len(surface) < 3:
                continue
            entity_id = self._make_id(surface)
            if entity_id not in seen:
                etype = _guess_entity_type(surface)
                seen[entity_id] = {
                    "id": entity_id,
                    "text": surface,
                    "type": etype,
                    "start": match.start(),
                    "end": match.end(),
                }

        return list(seen.values())

    def extract_relations(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relations between entities already found in *text*.

        Returns a list of dicts with keys: ``src``, ``dst``, ``relation``.
        """
        entity_index: Dict[str, str] = {e["text"].lower(): e["id"] for e in entities}
        results: List[Dict[str, Any]] = []
        seen_pairs: set = set()

        for spec in _RELATION_PATTERNS:
            for match in spec["pattern"].finditer(text):
                src_text = match.group("src").strip()
                dst_text = match.group("dst").strip()
                src_id = entity_index.get(src_text.lower()) or self._make_id(src_text)
                dst_id = entity_index.get(dst_text.lower()) or self._make_id(dst_text)
                pair_key = (src_id, dst_id, spec["relation"])
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    results.append(
                        {"src": src_id, "dst": dst_id, "relation": spec["relation"]}
                    )

        return results

    @staticmethod
    def _make_id(surface: str) -> str:
        """Produce a stable slug ID from an entity surface form."""
        return re.sub(r"\s+", "_", surface.strip().lower())
