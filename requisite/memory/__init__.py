"""
Conversation memory: pluggable per-session message history.

``default_registry`` ships pre-populated with ``"in_process"`` (the
zero-dependency default) so ``Agent(memory=..., session_id=...)`` works
with no external service to configure, plus ``"sqlite"`` (also
zero-dependency, persistent), ``"redis"`` (requires ``pip install
requisite-ai[redis]``, shared across processes), and ``"vector"``
(similarity-scoped recall on top of chronological storage -- see
``docs/adr/0022-vector-memory.md``). ``SQLiteMemory``/``VectorMemory``
are exported here like ``InProcessMemory`` since neither has a *hard*
optional dependency of its own (``VectorMemory``'s embedding
provider/vector store are supplied by the caller, the same way
``Retriever`` works); ``RedisMemory`` is only reachable via
``requisite.memory.redis.RedisMemory`` or
``default_registry.create("redis", ...)`` so importing this package never
requires the ``redis`` package to be installed. See
``docs/adr/0001-core-architecture-and-interfaces.md`` for the interface's
design rationale.
"""

from requisite.memory.base import BaseMemory
from requisite.memory.factory import MemoryRegistry, default_registry
from requisite.memory.in_process import InProcessMemory
from requisite.memory.policies import BaseConversationPolicy, MessageCountPolicy, SummarizingPolicy
from requisite.memory.sqlite import SQLiteMemory
from requisite.memory.vector import VectorMemory

__all__ = [
    "BaseConversationPolicy",
    "BaseMemory",
    "InProcessMemory",
    "MemoryRegistry",
    "MessageCountPolicy",
    "SQLiteMemory",
    "SummarizingPolicy",
    "VectorMemory",
    "default_registry",
]
