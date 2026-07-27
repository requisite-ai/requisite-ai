"""
Core RAG interfaces and data models.

RAG decomposes into independent extension points rather than one
interface (per ADR-0001's note on this), mirrored here:

- :class:`BaseEmbeddingProvider` -- text -> vector
- :class:`BaseVectorStore` -- store/search vectors
- :class:`BaseRetriever` -- the thing an application actually calls;
  the shipped :class:`~requisite.rag.retriever.Retriever` composes an
  embedding provider and a vector store, but the interface itself makes
  no assumption about how retrieval works (a future hybrid/BM25
  retriever doesn't need an embedding provider at all).

See ``docs/adr/0005-rag-integration.md`` for the full design rationale.
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
    def search(self, query_embedding: Sequence[float], *, top_k: int = 5) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks most similar to ``query_embedding``, best first."""

    @abstractmethod
    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks by id. No-op for any id that doesn't exist."""

    async def aadd(self, chunks: Sequence[Chunk]) -> None:
        """Async counterpart to :meth:`add`. Default: thread-wrapped."""
        await asyncio.to_thread(self.add, chunks)

    async def asearch(
        self, query_embedding: Sequence[float], *, top_k: int = 5
    ) -> list[ScoredChunk]:
        """Async counterpart to :meth:`search`. Default: thread-wrapped."""
        return await asyncio.to_thread(self.search, query_embedding, top_k=top_k)

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
