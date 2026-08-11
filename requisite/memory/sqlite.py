"""
SQLite-backed conversation memory.

Zero extra dependency -- uses the standard library's ``sqlite3`` module --
so a persistent, single-file backend is available with no external
service to run, unlike :class:`~requisite.memory.redis.RedisMemory`. Each
session's messages are rows in one ``messages`` table, ordered by
insertion; :class:`~requisite.core.interfaces.Message` is a Pydantic
model, so rows store ``message.model_dump_json()`` and reconstruct via
``Message.model_validate_json()``.
"""

from __future__ import annotations

import sqlite3
import threading

from requisite.core.exceptions import MemoryException
from requisite.core.interfaces import Message
from requisite.memory.base import BaseMemory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


class SQLiteMemory(BaseMemory):
    """Conversation memory persisted to a local SQLite file.

    Unlike :class:`~requisite.memory.in_process.InProcessMemory`, history
    survives a process restart -- construct a new instance pointed at the
    same ``db_path`` and prior sessions are still there. Thread-safe: one
    connection guarded by a single lock, matching ``InProcessMemory``'s
    locking approach -- this backend isn't a hot path, so simplicity wins
    over fine-grained locking.

    Async methods are not overridden: they inherit
    :class:`~requisite.memory.base.BaseMemory`'s default thread-wrapped
    behavior, since the standard library has no async SQLite driver and
    adding one (``aiosqlite``) isn't justified for this backend.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Required -- deliberately no
        default, so a persistent backend never creates a file somewhere
        surprising.

    Examples
    --------
    >>> import tempfile, os
    >>> path = os.path.join(tempfile.mkdtemp(), "memory.db")
    >>> memory = SQLiteMemory(db_path=path)
    >>> memory.append("session-1", Message.user("hello"))
    >>> memory.load("session-1")
    [Message(role=<Role.USER: 'user'>, content='hello', name=None, tool_call_id=None, tool_calls=[])]
    >>> # A second instance pointed at the same file sees the same history:
    >>> SQLiteMemory(db_path=path).load("session-1") == memory.load("session-1")
    True
    """

    def __init__(self, *, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            with self._lock:
                self._conn.executescript(_SCHEMA)
                self._conn.commit()
        except sqlite3.Error as exc:
            raise MemoryException(
                f"Failed to open/initialize SQLite database at '{db_path}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc

    @property
    def name(self) -> str:
        return "sqlite"

    def load(self, session_id: str) -> list[Message]:
        try:
            with self._lock:
                cursor = self._conn.execute(
                    "SELECT payload FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                )
                rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise MemoryException(
                f"SQLite load failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc
        return [Message.model_validate_json(row[0]) for row in rows]

    def append(self, session_id: str, message: Message) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO messages (session_id, payload) VALUES (?, ?)",
                    (session_id, message.model_dump_json()),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            raise MemoryException(
                f"SQLite append failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc

    def clear(self, session_id: str) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session_id,),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            raise MemoryException(
                f"SQLite clear failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc
