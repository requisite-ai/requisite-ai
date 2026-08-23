"""
BM25 (Okapi) keyword retrieval: the zero-dependency sparse counterpart to
``InMemoryVectorStore``'s dense cosine similarity.

Implemented in pure Python (no ``rank-bm25``/numpy dependency) -- the
scoring formula is simple enough that a small dependency isn't justified,
matching ``InMemoryVectorStore``'s own "pure Python is fine at this
scale" precedent. Tokenization is deliberately simple (lowercase + word
splitting, no stemming/stopword removal); a smarter tokenizer is a
follow-up if a real need for one emerges, mirroring how ``chunk_text()``
defers token-aware chunking the same way.

See ``docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md`` for the
full design rationale, including why this doesn't use ``rank-bm25``.
"""

from __future__ import annotations

import math
import re
import threading
import uuid
from collections import Counter
from collections.abc import Sequence
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.rag.base import BaseRetriever, Chunk, ScoredChunk
from requisite.rag.chunking import chunk_text
from requisite.tools.base import Tool

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Pure-Python Okapi BM25 scorer over an in-memory corpus of :class:`Chunk`s.

    Internal building block -- :class:`BM25Retriever` wraps this as a
    standalone :class:`~requisite.rag.base.BaseRetriever`;
    :class:`~requisite.rag.hybrid_retriever.HybridRetriever` uses one
    directly so it can index the exact same ``Chunk`` objects (same ids)
    it also sends to a vector store.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter. Higher values let repeated
        term occurrences keep contributing more before saturating.
    b:
        Length-normalization strength, in ``[0, 1]``. ``0`` disables
        document-length normalization entirely; ``1`` fully normalizes.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: dict[str, Chunk] = {}
        self._tokens: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def add(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk
                self._tokens[chunk.id] = _tokenize(chunk.text)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        with self._lock:
            for chunk_id in chunk_ids:
                self._chunks.pop(chunk_id, None)
                self._tokens.pop(chunk_id, None)

    def search(self, query: str, *, top_k: int = 5) -> list[ScoredChunk]:
        if top_k <= 0:
            return []

        with self._lock:
            chunk_ids = list(self._tokens)
            doc_tokens = {cid: self._tokens[cid] for cid in chunk_ids}
            chunks = dict(self._chunks)

        query_terms = _tokenize(query)
        if not chunk_ids or not query_terms:
            return []

        n_docs = len(chunk_ids)
        doc_lengths = {cid: len(doc_tokens[cid]) for cid in chunk_ids}
        avgdl = sum(doc_lengths.values()) / n_docs

        unique_query_terms = set(query_terms)
        df = {
            term: sum(1 for cid in chunk_ids if term in doc_tokens[cid])
            for term in unique_query_terms
        }
        idf = {
            term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
            for term in unique_query_terms
        }

        scored: list[ScoredChunk] = []
        for cid in chunk_ids:
            term_counts = Counter(doc_tokens[cid])
            dl = doc_lengths[cid]
            score = 0.0
            for term in query_terms:
                freq = term_counts.get(term, 0)
                if freq == 0:
                    continue
                numerator = freq * (self._k1 + 1)
                denominator = freq + self._k1 * (1 - self._b + self._b * dl / avgdl)
                score += idf[term] * numerator / denominator
            if score > 0:
                scored.append(ScoredChunk(chunk=chunks[cid], score=score))

        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]


class BM25Retriever(BaseRetriever):
    """Standalone keyword (BM25) retrieval -- no embedding provider needed.

    Same public shape as :class:`~requisite.rag.retriever.Retriever`
    (``add_texts``/``aadd_texts``/``retrieve``/``aretrieve``/``as_tool``),
    useful on its own for exact keyword/id/code search, or as a cheap
    fallback when there's no embedding API budget. See
    :class:`~requisite.rag.hybrid_retriever.HybridRetriever` to combine
    this with dense retrieval.

    Parameters
    ----------
    top_k:
        Default number of chunks to return; overridable per call.
    k1, b:
        Passed to the underlying :class:`BM25Index`.

    Examples
    --------
    >>> retriever = BM25Retriever()
    >>> retriever.add_texts(["Paris is the capital of France."])  # doctest: +SKIP
    >>> retriever.retrieve("capital of France")  # doctest: +SKIP
    """

    def __init__(self, *, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> None:
        self.top_k = top_k
        self._index = BM25Index(k1=k1, b=b)

    @property
    def name(self) -> str:
        return "bm25"

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """Chunk and index each text. Returns the stored chunk ids."""
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

        chunk_ids = [str(uuid.uuid4()) for _ in all_pieces]
        chunks = [
            Chunk(id=chunk_id, text=piece, metadata=metadata)
            for chunk_id, piece, metadata in zip(chunk_ids, all_pieces, piece_metadata, strict=True)
        ]
        self._index.add(chunks)
        return chunk_ids

    async def aadd_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """Async counterpart to :meth:`add_texts`. Pure CPU work, no I/O to offload."""
        return self.add_texts(
            texts, metadatas=metadatas, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> list[ScoredChunk]:
        resolved_top_k = top_k if top_k is not None else self.top_k
        return self._index.search(query, top_k=resolved_top_k)

    async def aretrieve(self, query: str, *, top_k: Optional[int] = None) -> list[ScoredChunk]:
        """Async counterpart to :meth:`retrieve`. Pure CPU work, no I/O to offload."""
        return self.retrieve(query, top_k=top_k)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks by id. No-op for any id that doesn't exist."""
        self._index.delete(chunk_ids)

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
