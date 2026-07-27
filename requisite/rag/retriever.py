"""
Retriever: the concrete, composed dense retriever most applications use.

Per the roadmap-planning decision, a retriever is exposed to agents as a
:class:`~requisite.capabilities.registry.CapabilityProvider`
(``agent.requires("knowledge_base")``), *not* as a new ``Agent``
constructor parameter -- reusing the mechanism that already exists for
every other kind of capability rather than adding a fourth one. See
``docs/adr/0005-rag-integration.md``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.rag.base import (
    BaseEmbeddingProvider,
    BaseRetriever,
    BaseVectorStore,
    Chunk,
    ScoredChunk,
)
from requisite.rag.chunking import chunk_text
from requisite.tools.base import Tool


class Retriever(BaseRetriever):
    """Dense retrieval: embed the query, search a vector store for nearest chunks.

    Parameters
    ----------
    embedding_provider:
        Turns text into vectors -- both documents (via :meth:`add_texts`)
        and queries (via :meth:`retrieve`).
    vector_store:
        Where embedded chunks are stored and searched.
    top_k:
        Default number of chunks to return; overridable per call.

    Examples
    --------
    >>> from requisite.rag.embeddings import OpenAIEmbeddingProvider
    >>> from requisite.rag.vectorstores import InMemoryVectorStore
    >>> retriever = Retriever(
    ...     embedding_provider=OpenAIEmbeddingProvider(api_key="sk-..."),
    ...     vector_store=InMemoryVectorStore(),
    ... )  # doctest: +SKIP
    >>> retriever.add_texts(["Paris is the capital of France."])  # doctest: +SKIP
    >>> retriever.retrieve("What is the capital of France?")  # doctest: +SKIP

    Exposing it as a capability:

    >>> from requisite.capabilities import default_registry as capabilities
    >>> capabilities.register("knowledge_base", retriever.as_tool())  # doctest: +SKIP
    >>> agent.requires("knowledge_base")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        top_k: int = 5,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """Chunk, embed, and store each text. Returns the stored chunk ids.

        Parameters
        ----------
        texts:
            Documents to add -- each is independently chunked.
        metadatas:
            Optional per-document metadata, applied to every chunk
            derived from that document. Must be the same length as
            ``texts`` if given.
        chunk_size, chunk_overlap:
            Passed to :func:`~requisite.rag.chunking.chunk_text`.
        """
        if metadatas is not None and len(metadatas) != len(texts):
            raise ConfigurationException("metadatas must be the same length as texts.")

        all_pieces: list[str] = []
        piece_metadata: list[dict[str, Any]] = []
        for index, text in enumerate(texts):
            pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_pieces.extend(pieces)
            metadata = dict(metadatas[index]) if metadatas is not None else {}
            piece_metadata.extend([metadata] * len(pieces))

        if not all_pieces:
            return []

        embeddings = self.embedding_provider.embed(all_pieces)
        chunk_ids = [str(uuid.uuid4()) for _ in all_pieces]
        chunks = [
            Chunk(id=chunk_id, text=piece, metadata=metadata, embedding=embedding)
            for chunk_id, piece, metadata, embedding in zip(
                chunk_ids, all_pieces, piece_metadata, embeddings, strict=True
            )
        ]
        self.vector_store.add(chunks)
        return chunk_ids

    async def aadd_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """Async counterpart to :meth:`add_texts`."""
        if metadatas is not None and len(metadatas) != len(texts):
            raise ConfigurationException("metadatas must be the same length as texts.")

        all_pieces: list[str] = []
        piece_metadata: list[dict[str, Any]] = []
        for index, text in enumerate(texts):
            pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_pieces.extend(pieces)
            metadata = dict(metadatas[index]) if metadatas is not None else {}
            piece_metadata.extend([metadata] * len(pieces))

        if not all_pieces:
            return []

        embeddings = await self.embedding_provider.aembed(all_pieces)
        chunk_ids = [str(uuid.uuid4()) for _ in all_pieces]
        chunks = [
            Chunk(id=chunk_id, text=piece, metadata=metadata, embedding=embedding)
            for chunk_id, piece, metadata, embedding in zip(
                chunk_ids, all_pieces, piece_metadata, embeddings, strict=True
            )
        ]
        await self.vector_store.aadd(chunks)
        return chunk_ids

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> list[ScoredChunk]:
        query_embedding = self.embedding_provider.embed_one(query)
        return self.vector_store.search(query_embedding, top_k=top_k or self.top_k)

    async def aretrieve(self, query: str, *, top_k: Optional[int] = None) -> list[ScoredChunk]:
        query_embedding = await self.embedding_provider.aembed_one(query)
        return await self.vector_store.asearch(query_embedding, top_k=top_k or self.top_k)

    def as_tool(
        self,
        *,
        name: str = "knowledge_base",
        description: str = "Search the knowledge base for information relevant to a query.",
        top_k: Optional[int] = None,
    ) -> Tool:
        """Expose this retriever as a :class:`~requisite.tools.base.Tool`.

        The returned tool is what typically gets registered as a
        capability (``capabilities.register("knowledge_base", retriever.as_tool())``)
        so ``agent.requires("knowledge_base")`` resolves to it.
        """
        resolved_top_k = top_k or self.top_k

        def _search(query: str) -> str:
            """Search the knowledge base for information relevant to a query."""
            results = self.retrieve(query, top_k=resolved_top_k)
            if not results:
                return "No relevant information found."
            return "\n\n".join(f"[score={r.score:.3f}] {r.chunk.text}" for r in results)

        return Tool.from_function(_search, name=name, description=description)
