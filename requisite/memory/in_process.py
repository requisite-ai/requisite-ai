"""
In-process memory: the zero-dependency default conversation memory backend.

Stores each session's history as a plain Python list, held in the
process's own memory. Lost on restart -- appropriate for development,
single-process applications, and as the framework's default so
``Agent(memory=...)`` is demonstrable with no external service to set up.
For anything that needs to survive a restart or be shared across
processes, use a persistent backend (SQLite, Redis, ... -- see
``ROADMAP.md``) once one ships.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict

from requisite.core.interfaces import Message
from requisite.memory.base import BaseMemory

logger = logging.getLogger("requisite.memory.in_process")


class InProcessMemory(BaseMemory):
    """Dict-backed conversation memory, scoped to the current process.

    Thread-safe: a single lock guards all reads/writes, which is more than
    sufficient for this backend's expected use (occasional appends/loads,
    not a hot path) and keeps the implementation simple and obviously
    correct rather than fine-grained and subtly racy.

    Examples
    --------
    >>> memory = InProcessMemory()
    >>> memory.append("session-1", Message.user("hi"))
    >>> memory.append("session-1", Message.assistant("hello!"))
    >>> [m.content for m in memory.load("session-1")]
    ['hi', 'hello!']
    >>> memory.clear("session-1")
    >>> memory.load("session-1")
    []
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = defaultdict(list)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "in_process"

    def load(self, session_id: str) -> list[Message]:
        with self._lock:
            return list(self._sessions[session_id])

    def append(self, session_id: str, message: Message) -> None:
        with self._lock:
            self._sessions[session_id].append(message)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
