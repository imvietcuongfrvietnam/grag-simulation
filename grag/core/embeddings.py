"""Deterministic mock embeddings that require no external model or API."""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np


class MockEmbedder:
    """Hash-based fake embedder that produces deterministic 128-d vectors.

    Embeddings are reproducible across runs and process boundaries: the same
    text always produces the same vector.  Cosine similarity between
    semantically unrelated strings will be near-random, but the interface is
    identical to a real dense retrieval model, making it useful for pipeline
    smoke-tests and algorithm simulations.
    """

    DIM: int = 128

    def __init__(self, dim: int = 128, seed: int = 42) -> None:
        self._dim = dim
        self._seed = seed

    # ------------------------------------------------------------------
    # Core embedding methods
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a unit-norm vector of shape ``(dim,)``."""
        raw = self._hash_to_floats(text, self._dim)
        return self._normalise(raw)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings, returning an array of shape ``(N, dim)``."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        rows = [self.embed(t) for t in texts]
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Return the cosine similarity between two 1-D arrays.

        Both vectors are re-normalised internally so pre-normalisation is not
        required.
        """
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    def top_k_similar(
        self,
        query_emb: np.ndarray,
        corpus_embs: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the *k* most similar vectors from *corpus_embs*.

        Parameters
        ----------
        query_emb:
            1-D array of shape ``(dim,)``.
        corpus_embs:
            2-D array of shape ``(N, dim)``.
        k:
            Number of results to return.

        Returns
        -------
        indices:
            1-D integer array of length ``min(k, N)``.
        scores:
            Corresponding cosine similarity scores.
        """
        if corpus_embs.ndim != 2 or corpus_embs.shape[0] == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        norms = np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-12
        normed = corpus_embs / norms
        scores = normed @ q

        k = min(k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return top_indices, scores[top_indices]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_to_floats(self, text: str, n: int) -> np.ndarray:
        """Deterministically derive *n* floats in ``[-1, 1]`` from *text*."""
        encoded = text.encode("utf-8")
        floats: List[float] = []
        counter = 0
        while len(floats) < n:
            digest = hashlib.sha256(encoded + counter.to_bytes(4, "little")).digest()
            for i in range(0, len(digest) - 1, 2):
                val = (digest[i] << 8 | digest[i + 1]) / 32767.5 - 1.0
                floats.append(val)
                if len(floats) == n:
                    break
            counter += 1
        return np.array(floats[:n], dtype=np.float32)

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return (v / norm).astype(np.float32)
