"""
Core RAG interfaces and data models.

RAG decomposes into independent extension points rather than one
interface (per ADR-0001's note on this), mirrored here:

- :class:`BaseEmbeddingProvider` -- text -> vector
- :class:`BaseVectorStore` -- store/search vectors
- :class:`BaseRetriever` -- the thing an application actually calls; the
  shipped :class:`~requisite.rag.retriever.Retriever` composes an
  embedding provider and a vector store (dense), but the interface makes
  no assumption about how retrieval works --
  :class:`~requisite.rag.bm25.BM25Retriever` (keyword-only, no embedding
  provider) and :class:`~requisite.rag.hybrid_retriever.HybridRetriever`
  (dense + BM25, fused) are the anticipated case this independence was
  designed to allow.
- :class:`BaseReranker` -- an optional, standalone post-processing step
  over an already-retrieved candidate list; see
  :class:`~requisite.rag.reranker.LLMReranker`.

See ``docs/adr/0005-rag-integration.md`` and ``docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md``
for the full design rationale.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrievable unit of text.

    Parameters
    ----------
    id:
        Unique identifier within its vector store.
    text:
        The chunk's text content.
    metadata:
        Arbitrary application-supplied metadata (source document, page
        number, section heading, ...) -- never interpreted by the
        framework itself, just carried through.
    embedding:
        The chunk's vector embedding, once computed. ``None`` before
        embedding; vector stores populate this internally.
    """

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None


class ScoredChunk(BaseModel):
    """A :class:`Chunk` returned from a search, with its similarity score."""

    chunk: Chunk
    score: float

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.chunk.text


class BaseEmbeddingProvider(ABC):
    """Abstract interface for turning text into vectors.

    Examples
    --------
    >>> provider = SomeEmbeddingProvider()  # doctest: +SKIP
    >>> vector = provider.embed_one("hello world")  # doctest: +SKIP
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique, lowercase identifier for this provider (e.g. ``"openai"``)."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input, in order."""

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        """Async counterpart to :meth:`embed`. Default: thread-wrapped."""
        return await asyncio.to_thread(self.embed, texts)

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single string."""
        return self.embed([text])[0]

    async def aembed_one(self, text: str) -> list[float]:
        """Async counterpart to :meth:`embed_one`."""
        return (await self.aembed([text]))[0]


def matches_filter(metadata: dict[str, Any], filter: Optional[dict[str, Any]]) -> bool:  # noqa: A002
    """Whether ``metadata`` satisfies every key/value pair in ``filter`` (exact equality).

    ``filter=None`` (or empty) always matches. Shared by
    :class:`~requisite.rag.vectorstores.in_memory.InMemoryVectorStore` and
    :class:`~requisite.rag.vectorstores.weaviate.WeaviateVectorStore`, the
    two concrete stores that implement :meth:`BaseVectorStore.search`'s
    ``filter`` parameter themselves rather than delegating to a vector
    database's own native filtering (which
    :class:`~requisite.rag.vectorstores.pinecone.PineconeVectorStore` does).
    """
    if not filter:
        return True
    return all(metadata.get(key) == value for key, value in filter.items())


class BaseVectorStore(ABC):
    """Abstract interface for storing and searching embedded chunks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique, lowercase identifier for this store (e.g. ``"in_memory"``)."""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk]) -> None:
        """Store chunks, each of which must already have ``embedding`` set."""

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,  # noqa: A002
    ) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks most similar to ``query_embedding``, best first.

        ``filter``, if given, restricts candidates to chunks whose
        ``metadata`` matches every key/value pair (exact equality).
        ``None`` (default) searches the whole store -- unchanged behavior
        from before this parameter existed.
        """

    @abstractmethod
    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks by id. No-op for any id that doesn't exist."""

    async def aadd(self, chunks: Sequence[Chunk]) -> None:
        """Async counterpart to :meth:`add`. Default: thread-wrapped."""
        await asyncio.to_thread(self.add, chunks)

    async def asearch(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,  # noqa: A002
    ) -> list[ScoredChunk]:
        """Async counterpart to :meth:`search`. Default: thread-wrapped."""
        return await asyncio.to_thread(self.search, query_embedding, top_k=top_k, filter=filter)

    async def adelete(self, chunk_ids: Sequence[str]) -> None:
        """Async counterpart to :meth:`delete`. Default: thread-wrapped."""
        await asyncio.to_thread(self.delete, chunk_ids)


class BaseRetriever(ABC):
    """Abstract interface for the thing an application actually calls to
    get relevant context for a query.

    Deliberately independent of :class:`BaseEmbeddingProvider` /
    :class:`BaseVectorStore` at the interface level -- the shipped
    :class:`~requisite.rag.retriever.Retriever` composes both (dense
    retrieval), but a future hybrid or keyword-based retriever might not
    use an embedding provider at all.
    """

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks most relevant to ``query``, best first."""

    async def aretrieve(self, query: str, *, top_k: int = 5) -> list[ScoredChunk]:
        """Async counterpart to :meth:`retrieve`. Default: thread-wrapped."""
        return await asyncio.to_thread(self.retrieve, query, top_k=top_k)


class BaseReranker(ABC):
    """Abstract interface for re-scoring an already-retrieved candidate list.

    Deliberately not wired into any retriever's constructor -- re-ranking
    is a standalone, composable post-processing step applied to whatever
    a retriever already returned:

    >>> reranker = SomeReranker()  # doctest: +SKIP
    >>> candidates = retriever.retrieve(query, top_k=20)  # doctest: +SKIP
    >>> best = reranker.rerank(query, candidates, top_k=5)  # doctest: +SKIP

    This works identically regardless of which :class:`BaseRetriever`
    produced ``candidates``.
    """

    @abstractmethod
    def rerank(
        self, query: str, results: Sequence[ScoredChunk], *, top_k: Optional[int] = None
    ) -> list[ScoredChunk]:
        """Re-score and re-order ``results`` for relevance to ``query``.

        ``top_k`` truncates the output; ``None`` (default) returns every
        input result, just re-ordered.
        """

    async def arerank(
        self, query: str, results: Sequence[ScoredChunk], *, top_k: Optional[int] = None
    ) -> list[ScoredChunk]:
        """Async counterpart to :meth:`rerank`. Default: thread-wrapped."""
        return await asyncio.to_thread(self.rerank, query, results, top_k=top_k)


class BaseCompressor(ABC):
    """Abstract interface for compressing an already-retrieved candidate
    list's text content.

    Like :class:`BaseReranker`, deliberately not wired into any
    retriever's constructor -- a standalone, composable post-processing
    step applied to whatever a retriever (and optionally a reranker)
    already returned:

    >>> compressor = SomeCompressor()  # doctest: +SKIP
    >>> candidates = retriever.retrieve(query, top_k=20)  # doctest: +SKIP
    >>> compressed = compressor.compress(query, candidates)  # doctest: +SKIP

    This is a text-*reduction* step, not a re-scoring one: it shrinks
    each chunk's ``text`` to whatever is relevant to the query, and
    drops chunks with nothing relevant left -- it does not reorder or
    truncate by count the way :meth:`BaseReranker.rerank`'s ``top_k``
    does. Compose the two directly when both are wanted:
    ``compressor.compress(query, reranker.rerank(query, results, top_k=N))``.
    """

    @abstractmethod
    def compress(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Compress each result's text to the content relevant to ``query``.

        Returns a new list of :class:`ScoredChunk` with ``chunk.text``
        replaced by its compressed form and ``score`` unchanged. A
        result with nothing relevant to ``query`` is dropped from the
        output entirely, rather than returned with empty text.
        """

    async def acompress(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Async counterpart to :meth:`compress`. Default: thread-wrapped."""
        return await asyncio.to_thread(self.compress, query, results)
