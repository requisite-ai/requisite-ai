"""
Hybrid retrieval: dense (embedding + vector store) fused with sparse
(BM25 keyword) results via Reciprocal Rank Fusion.

Composes an embedding provider and a vector store the same way
:class:`~requisite.rag.retriever.Retriever` does, plus an internal
:class:`~requisite.rag.bm25.BM25Index` -- not a separate
:class:`~requisite.rag.bm25.BM25Retriever` instance, so both sides index
the exact same :class:`~requisite.rag.base.Chunk` objects (same ids),
which fusion-by-id depends on. See
``docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md`` for the full
design rationale, including why Reciprocal Rank Fusion rather than a
normalized weighted-score blend.
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
from requisite.rag.bm25 import BM25Index
from requisite.rag.chunking import chunk_text
from requisite.tools.base import Tool

_DEFAULT_RRF_K = 60


def _reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]], *, top_k: int, k: int = _DEFAULT_RRF_K
) -> list[ScoredChunk]:
    """Merge several ranked result lists into one, by rank position only.

    ``score(chunk) = sum(1 / (k + rank))`` across every list the chunk
    appears in (1-indexed rank; absent from a list contributes nothing).
    Deliberately ignores each list's own ``ScoredChunk.score`` -- cosine
    similarity and BM25 scores aren't on comparable scales, so only
    relative rank position is used to fuse them.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, scored_chunk in enumerate(results, start=1):
            chunk_id = scored_chunk.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk_id, scored_chunk.chunk)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [ScoredChunk(chunk=chunks[cid], score=scores[cid]) for cid in ranked_ids[:top_k]]


class HybridRetriever(BaseRetriever):
    """Dense + BM25 retrieval, fused with Reciprocal Rank Fusion.

    Parameters
    ----------
    embedding_provider:
        Turns text into vectors for the dense half.
    vector_store:
        Where embedded chunks are stored and searched for the dense half.
    top_k:
        Default number of fused chunks to return; overridable per call.
    candidate_pool_size:
        How many results to fetch from *each* side (dense and BM25)
        before fusing -- larger than ``top_k`` so fusion has real
        candidates to blend from both sides rather than just whichever
        side's top few happen to dominate.
    rrf_k:
        The Reciprocal Rank Fusion constant -- see
        :func:`_reciprocal_rank_fusion`.
    bm25_k1, bm25_b:
        Passed to the internal :class:`~requisite.rag.bm25.BM25Index`.

    Examples
    --------
    >>> from requisite.rag.embeddings import OpenAIEmbeddingProvider
    >>> from requisite.rag.vectorstores import InMemoryVectorStore
    >>> retriever = HybridRetriever(
    ...     embedding_provider=OpenAIEmbeddingProvider(api_key="sk-..."),
    ...     vector_store=InMemoryVectorStore(),
    ... )  # doctest: +SKIP
    >>> retriever.add_texts(["Paris is the capital of France."])  # doctest: +SKIP
    >>> retriever.retrieve("capital of France")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        top_k: int = 5,
        candidate_pool_size: int = 20,
        rrf_k: int = _DEFAULT_RRF_K,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.rrf_k = rrf_k
        self._bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)

    @property
    def name(self) -> str:
        return "hybrid"

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """Chunk, embed, and index each text into both the dense and BM25
        sides using the same chunk ids. Returns the stored chunk ids.
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
        self._bm25_index.add(chunks)
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
        self._bm25_index.add(chunks)
        return chunk_ids

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> list[ScoredChunk]:
        resolved_top_k = top_k if top_k is not None else self.top_k
        pool = max(self.candidate_pool_size, resolved_top_k)
        query_embedding = self.embedding_provider.embed_one(query)
        dense_results = self.vector_store.search(query_embedding, top_k=pool)
        bm25_results = self._bm25_index.search(query, top_k=pool)
        return _reciprocal_rank_fusion(
            [dense_results, bm25_results], top_k=resolved_top_k, k=self.rrf_k
        )

    async def aretrieve(self, query: str, *, top_k: Optional[int] = None) -> list[ScoredChunk]:
        """Async counterpart to :meth:`retrieve`."""
        resolved_top_k = top_k if top_k is not None else self.top_k
        pool = max(self.candidate_pool_size, resolved_top_k)
        query_embedding = await self.embedding_provider.aembed_one(query)
        dense_results = await self.vector_store.asearch(query_embedding, top_k=pool)
        bm25_results = self._bm25_index.search(query, top_k=pool)
        return _reciprocal_rank_fusion(
            [dense_results, bm25_results], top_k=resolved_top_k, k=self.rrf_k
        )

    def as_tool(
        self,
        *,
        name: str = "knowledge_base",
        description: str = "Search the knowledge base for information relevant to a query.",
        top_k: Optional[int] = None,
    ) -> Tool:
        """Expose this retriever as a :class:`~requisite.tools.base.Tool`."""
        resolved_top_k = top_k if top_k is not None else self.top_k

        def _search(query: str) -> str:
            """Search the knowledge base for information relevant to a query."""
            results = self.retrieve(query, top_k=resolved_top_k)
            if not results:
                return "No relevant information found."
            return "\n\n".join(f"[score={r.score:.3f}] {r.chunk.text}" for r in results)

        return Tool.from_function(_search, name=name, description=description)
