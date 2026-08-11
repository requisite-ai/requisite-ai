"""
Conversation memory: pluggable per-session message history.

``default_registry`` ships pre-populated with ``"in_process"`` (the
zero-dependency default) so ``Agent(memory=..., session_id=...)`` works
with no external service to configure, plus ``"sqlite"`` (also
zero-dependency, persistent) and ``"redis"`` (requires ``pip install
requisite-ai[redis]``, shared across processes). ``SQLiteMemory`` is
exported here like ``InProcessMemory`` since it has no optional
dependency; ``RedisMemory`` is only reachable via
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

__all__ = [
    "BaseConversationPolicy",
    "BaseMemory",
    "InProcessMemory",
    "MemoryRegistry",
    "MessageCountPolicy",
    "SQLiteMemory",
    "SummarizingPolicy",
    "default_registry",
]
