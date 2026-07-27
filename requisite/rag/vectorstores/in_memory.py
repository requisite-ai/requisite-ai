"""
In-memory vector store: the zero-dependency default, mirroring
``InProcessMemory``'s role for conversation memory.

Stores chunks and their embeddings as plain Python lists, held in the
process's own memory. Cosine similarity is computed in pure Python (no
numpy dependency) -- fine at the scale this default is meant for (a few
thousand chunks); use Pinecone/Weaviate for anything larger or that
needs to survive a restart.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

from requisite.rag.base import BaseVectorStore, Chunk, ScoredChunk


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(BaseVectorStore):
    """Dict-backed vector store, scoped to the current process.

    Examples
    --------
    >>> store = InMemoryVectorStore()
    >>> store.add([Chunk(id="1", text="cats are great", embedding=[1.0, 0.0])])
    >>> results = store.search([1.0, 0.0], top_k=1)
    >>> results[0].chunk.text
    'cats are great'
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "in_memory"

    def add(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk

    def search(self, query_embedding: Sequence[float], *, top_k: int = 5) -> list[ScoredChunk]:
        with self._lock:
            candidates = list(self._chunks.values())

        scored = [
            ScoredChunk(chunk=chunk, score=_cosine_similarity(query_embedding, chunk.embedding))
            for chunk in candidates
            if chunk.embedding is not None
        ]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        with self._lock:
            for chunk_id in chunk_ids:
                self._chunks.pop(chunk_id, None)
