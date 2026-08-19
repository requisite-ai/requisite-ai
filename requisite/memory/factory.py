"""
Memory registry.

Mirrors :class:`requisite.providers.factory.ProviderRegistry`'s shape: a
plain, instantiable class (not a singleton) mapping backend names to
constructors, pre-populated with the framework's built-in ``"in_process"``,
``"sqlite"``, and ``"redis"`` backends. Third-party packages register
their own memory backend (a vector store, a cloud database, ...) the
same way new providers are added.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from requisite.core.exceptions import ConfigurationException
from requisite.memory.base import BaseMemory

logger = logging.getLogger("requisite.memory.factory")

MemoryBuilder = Callable[..., BaseMemory]


class MemoryRegistry:
    """A registry mapping memory backend names to constructors."""

    def __init__(self) -> None:
        self._builders: dict[str, MemoryBuilder] = {}

    def register(self, name: str, builder: MemoryBuilder) -> None:
        """Register a memory backend constructor under ``name``."""
        key = name.lower().strip()
        if not key:
            raise ConfigurationException("Memory backend name must be a non-empty string.")
        self._builders[key] = builder
        logger.debug("Registered memory backend '%s'", key, extra={"memory_backend": key})

    def available(self) -> list[str]:
        """Return the sorted list of currently registered backend names."""
        return sorted(self._builders)

    def create(self, name: str, **kwargs: Any) -> BaseMemory:
        """Instantiate the memory backend registered under ``name``.

        Raises
        ------
        ConfigurationException
            If no backend is registered under ``name``.
        """
        key = name.lower().strip()
        builder = self._builders.get(key)
        if builder is None:
            raise ConfigurationException(
                f"Unknown memory backend '{name}'. Available: {self.available()}",
            )
        return builder(**kwargs)


def _register_builtin_backends(registry: MemoryRegistry) -> None:
    def _build_in_process(**kwargs: Any) -> BaseMemory:
        from requisite.memory.in_process import InProcessMemory

        return InProcessMemory(**kwargs)

    def _build_sqlite(**kwargs: Any) -> BaseMemory:
        from requisite.memory.sqlite import SQLiteMemory

        return SQLiteMemory(**kwargs)

    def _build_redis(**kwargs: Any) -> BaseMemory:
        from requisite.memory.redis import RedisMemory

        return RedisMemory(**kwargs)

    def _build_vector(**kwargs: Any) -> BaseMemory:
        from requisite.memory.vector import VectorMemory

        return VectorMemory(**kwargs)

    registry.register("in_process", _build_in_process)
    registry.register("sqlite", _build_sqlite)
    registry.register("redis", _build_redis)
    registry.register("vector", _build_vector)


default_registry = MemoryRegistry()
_register_builtin_backends(default_registry)
