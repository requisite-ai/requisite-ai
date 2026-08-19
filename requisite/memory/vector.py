"""
Vector-database-backed conversation memory: similarity-scoped recall on
top of plain chronological storage.

Composes two things rather than inventing one bundled backend:

- A chronological :class:`~requisite.memory.base.BaseMemory` delegate
  (any existing backend -- defaults to
  :class:`~requisite.memory.in_process.InProcessMemory`) satisfies
  :class:`BaseMemory`'s exact ``load``/``append``/``clear`` contract, so
  :class:`~requisite.agents.agent.Agent`'s existing call sites work
  against :class:`VectorMemory` unchanged.
- A :class:`~requisite.rag.base.BaseEmbeddingProvider` +
  :class:`~requisite.rag.base.BaseVectorStore` pair -- the same
  composition :class:`~requisite.rag.retriever.Retriever` already uses --
  which every ``append()`` also feeds, enabling the new
  :meth:`VectorMemory.load_relevant` for semantic top-k recall.
  Deliberately *not* part of :class:`BaseMemory`'s own contract -- see
  that module's docstring ("not retrieval, ranking, or similarity
  search") and ``docs/adr/0022-vector-memory.md``.

Install the embedding provider / vector store of your choice the same
way you would for RAG (``pip install requisite-ai[rag]``, or
``pinecone``/``weaviate-client`` for those backends) -- ``VectorMemory``
itself has no hard optional dependency.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

from requisite.core.interfaces import Message, Role
from requisite.memory.base import BaseMemory
from requisite.memory.in_process import InProcessMemory
from requisite.rag.base import BaseEmbeddingProvider, BaseVectorStore, Chunk


class VectorMemory(BaseMemory):
    """Conversation memory with semantic recall, backed by a vector store.

    Parameters
    ----------
    embedding_provider:
        Turns message content into vectors, both on write (``append``)
        and on read (``load_relevant``).
    vector_store:
        Where embedded messages are stored and searched.
    history_backend:
        The :class:`BaseMemory` backend used for plain chronological
        storage (``load``/``append``/``clear``). Defaults to a fresh
        :class:`~requisite.memory.in_process.InProcessMemory`; pass
        :class:`~requisite.memory.sqlite.SQLiteMemory` or
        :class:`~requisite.memory.redis.RedisMemory` for persistence.
    top_k:
        Default number of messages :meth:`load_relevant` returns;
        overridable per call.

    Notes
    -----
    Each session's next chunk id is tracked in an in-memory counter
    (seeded once from ``history_backend`` on first use, not re-derived on
    every call), guarded by a lock so concurrent ``append``/``clear``
    calls on *this instance* can't collide or race each other. Sync calls
    are guarded by a :class:`threading.Lock`; async calls by a separate
    :class:`asyncio.Lock` -- mixing sync and async calls concurrently on
    the same instance isn't guarded (an unusual usage pattern this
    codebase doesn't defend against elsewhere either). The counter is
    purely an id-allocation cache, not required to equal the backend's
    true message count -- see :meth:`append`.

    Examples
    --------
    >>> from requisite.memory import VectorMemory
    >>> from requisite.rag.embeddings import OpenAIEmbeddingProvider
    >>> from requisite.rag.vectorstores import InMemoryVectorStore
    >>> memory = VectorMemory(
    ...     embedding_provider=OpenAIEmbeddingProvider(api_key="sk-..."),
    ...     vector_store=InMemoryVectorStore(),
    ... )  # doctest: +SKIP
    >>> memory.append("session-1", Message.user("My favorite color is blue."))  # doctest: +SKIP
    >>> memory.load("session-1")  # plain chronological history  # doctest: +SKIP
    >>> memory.load_relevant("session-1", "What color do I like?")  # semantic recall  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        history_backend: Optional[BaseMemory] = None,
        top_k: int = 5,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self._history = history_backend or InProcessMemory()
        self.top_k = top_k
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._alock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "vector"

    @staticmethod
    def _chunk_id(session_id: str, turn_index: int) -> str:
        return f"{session_id}:{turn_index}"

    def _chunk(
        self, session_id: str, turn_index: int, message: Message, embedding: list[float]
    ) -> Chunk:
        return Chunk(
            id=self._chunk_id(session_id, turn_index),
            text=message.content,
            metadata={"session_id": session_id, "role": message.role.value},
            embedding=embedding,
        )

    def load(self, session_id: str) -> list[Message]:
        return self._history.load(session_id)

    def append(self, session_id: str, message: Message) -> None:
        with self._lock:
            turn_index = self._counts.get(session_id)
            if turn_index is None:
                turn_index = len(self._history.load(session_id))
            self._counts[session_id] = turn_index + 1
        # Embed + store *before* committing to chronological history -- a
        # failed embedding call then leaves no partial state (nothing
        # appended anywhere) instead of a message that's durably logged
        # but permanently unreachable via load_relevant. The allocated
        # turn_index is simply skipped on failure, which is harmless: ids
        # only need to be unique, not contiguous.
        if message.content:
            embedding = self.embedding_provider.embed_one(message.content)
            self.vector_store.add([self._chunk(session_id, turn_index, message, embedding)])
        self._history.append(session_id, message)

    def clear(self, session_id: str) -> None:
        with self._lock:
            count = self._counts.get(session_id)
            if count is None:
                count = len(self._history.load(session_id))
            self.vector_store.delete([self._chunk_id(session_id, i) for i in range(count)])
            self._history.clear(session_id)
            self._counts[session_id] = 0

    async def aload(self, session_id: str) -> list[Message]:
        return await self._history.aload(session_id)

    async def aappend(self, session_id: str, message: Message) -> None:
        async with self._alock:
            turn_index = self._counts.get(session_id)
            if turn_index is None:
                turn_index = len(await self._history.aload(session_id))
            self._counts[session_id] = turn_index + 1
        if message.content:
            embedding = await self.embedding_provider.aembed_one(message.content)
            await self.vector_store.aadd([self._chunk(session_id, turn_index, message, embedding)])
        await self._history.aappend(session_id, message)

    async def aclear(self, session_id: str) -> None:
        async with self._alock:
            count = self._counts.get(session_id)
            if count is None:
                count = len(await self._history.aload(session_id))
            await self.vector_store.adelete([self._chunk_id(session_id, i) for i in range(count)])
            await self._history.aclear(session_id)
            self._counts[session_id] = 0

    def load_relevant(
        self, session_id: str, query: str, *, top_k: Optional[int] = None
    ) -> list[Message]:
        """Semantic recall: the messages stored for ``session_id`` most
        relevant to ``query``, best match first -- **not** chronological
        (see :meth:`load` for that). Beyond :class:`BaseMemory`'s own
        contract by design; see ``docs/adr/0022-vector-memory.md``.
        """
        resolved_top_k = top_k if top_k is not None else self.top_k
        query_embedding = self.embedding_provider.embed_one(query)
        results = self.vector_store.search(
            query_embedding, top_k=resolved_top_k, filter={"session_id": session_id}
        )
        return [
            Message(role=Role(r.chunk.metadata.get("role", Role.USER.value)), content=r.chunk.text)
            for r in results
        ]

    async def aload_relevant(
        self, session_id: str, query: str, *, top_k: Optional[int] = None
    ) -> list[Message]:
        """Async counterpart to :meth:`load_relevant`."""
        resolved_top_k = top_k if top_k is not None else self.top_k
        query_embedding = await self.embedding_provider.aembed_one(query)
        results = await self.vector_store.asearch(
            query_embedding, top_k=resolved_top_k, filter={"session_id": session_id}
        )
        return [
            Message(role=Role(r.chunk.metadata.get("role", Role.USER.value)), content=r.chunk.text)
            for r in results
        ]
