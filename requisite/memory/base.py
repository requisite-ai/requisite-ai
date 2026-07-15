"""
Memory interface.

Scoped narrowly and deliberately: memory is *conversation-shaped
storage* -- an ordered list of :class:`~requisite.core.interfaces.Message`
per session -- not retrieval, ranking, or similarity search (that's RAG's
job, tracked separately in ``ROADMAP.md``). An :class:`~requisite.agents.agent.Agent`
reads and appends messages through this interface; it never reaches into
storage internals.

See ``docs/adr/0001-core-architecture-and-interfaces.md`` for the
rationale behind this shape, and
``docs/adr/0002-provider-kwargs-and-memory-integration.md`` for how it's
wired into :class:`~requisite.agents.agent.Agent`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from requisite.core.interfaces import Message


class BaseMemory(ABC):
    """Abstract interface for conversation memory backends.

    A "session" is an opaque, caller-chosen string identifying one
    conversation thread -- a chat ID, a user ID, a request ID, whatever
    makes sense for the application. Implementations key storage by this
    string; they don't need to know anything about what it represents.

    Examples
    --------
    >>> from requisite.memory import InProcessMemory
    >>> memory = InProcessMemory()
    >>> memory.append("session-1", Message.user("hello"))
    >>> memory.load("session-1")
    [Message(role=<Role.USER: 'user'>, content='hello', name=None, tool_call_id=None, tool_calls=[])]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique, lowercase identifier for this backend (e.g. ``"in_process"``)."""

    @abstractmethod
    def load(self, session_id: str) -> list[Message]:
        """Return the stored conversation history for a session, oldest first.

        Returns an empty list for a session with no stored history yet --
        never raises for an unknown session.
        """

    @abstractmethod
    def append(self, session_id: str, message: Message) -> None:
        """Persist one new message to a session's history."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete a session's stored history. No-op if the session doesn't exist."""

    async def aload(self, session_id: str) -> list[Message]:
        """Async counterpart to :meth:`load`.

        Default implementation runs :meth:`load` in a worker thread so a
        synchronous backend remains usable from async code without
        blocking the event loop. Override for a true async implementation
        (e.g. a backend using an async database driver).
        """
        return await asyncio.to_thread(self.load, session_id)

    async def aappend(self, session_id: str, message: Message) -> None:
        """Async counterpart to :meth:`append`. Default: thread-wrapped."""
        await asyncio.to_thread(self.append, session_id, message)

    async def aclear(self, session_id: str) -> None:
        """Async counterpart to :meth:`clear`. Default: thread-wrapped."""
        await asyncio.to_thread(self.clear, session_id)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(name={self.name!r})"
