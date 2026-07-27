"""
Conversation memory: pluggable per-session message history.

``default_registry`` ships pre-populated with ``"in_process"`` (the
zero-dependency default) so ``Agent(memory=..., session_id=...)`` works
with no external service to configure. See
``docs/adr/0001-core-architecture-and-interfaces.md`` for the interface's
design rationale.
"""

from requisite.memory.base import BaseMemory
from requisite.memory.factory import MemoryRegistry, default_registry
from requisite.memory.in_process import InProcessMemory
from requisite.memory.policies import BaseConversationPolicy, MessageCountPolicy, SummarizingPolicy

__all__ = [
    "BaseConversationPolicy",
    "BaseMemory",
    "InProcessMemory",
    "MemoryRegistry",
    "MessageCountPolicy",
    "SummarizingPolicy",
    "default_registry",
]
