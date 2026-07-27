"""
Registries for embedding providers and vector stores.

Same shape as every other registry in the framework: plain, instantiable
classes, not singletons. ``default_embedding_registry`` ships
pre-populated with ``"openai"`` and ``"gemini"`` (lazy-imported, so
neither SDK needs to be installed just to import this module).
``default_vector_store_registry`` ships pre-populated with
``"in_memory"`` -- the zero-dependency default -- with Pinecone/Weaviate
to be registered here once implemented (see ``ROADMAP.md``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from requisite.core.exceptions import ConfigurationException
from requisite.rag.base import BaseEmbeddingProvider, BaseVectorStore

logger = logging.getLogger("requisite.rag.factory")

EmbeddingBuilder = Callable[..., BaseEmbeddingProvider]
VectorStoreBuilder = Callable[..., BaseVectorStore]


class EmbeddingRegistry:
    """A registry mapping embedding provider names to constructors."""

    def __init__(self) -> None:
        self._builders: dict[str, EmbeddingBuilder] = {}

    def register(self, name: str, builder: EmbeddingBuilder) -> None:
        key = name.lower().strip()
        if not key:
            raise ConfigurationException("Embedding provider name must be a non-empty string.")
        self._builders[key] = builder
        logger.debug("Registered embedding provider '%s'", key, extra={"embedding_provider": key})

    def available(self) -> list[str]:
        return sorted(self._builders)

    def create(self, name: str, **kwargs: Any) -> BaseEmbeddingProvider:
        key = name.lower().strip()
        builder = self._builders.get(key)
        if builder is None:
            raise ConfigurationException(
                f"Unknown embedding provider '{name}'. Available: {self.available()}",
            )
        return builder(**kwargs)


class VectorStoreRegistry:
    """A registry mapping vector store names to constructors."""

    def __init__(self) -> None:
        self._builders: dict[str, VectorStoreBuilder] = {}

    def register(self, name: str, builder: VectorStoreBuilder) -> None:
        key = name.lower().strip()
        if not key:
            raise ConfigurationException("Vector store name must be a non-empty string.")
        self._builders[key] = builder
        logger.debug("Registered vector store '%s'", key, extra={"vector_store": key})

    def available(self) -> list[str]:
        return sorted(self._builders)

    def create(self, name: str, **kwargs: Any) -> BaseVectorStore:
        key = name.lower().strip()
        builder = self._builders.get(key)
        if builder is None:
            raise ConfigurationException(
                f"Unknown vector store '{name}'. Available: {self.available()}",
            )
        return builder(**kwargs)


def _register_builtin_embeddings(registry: EmbeddingRegistry) -> None:
    def _build_openai(**kwargs: Any) -> BaseEmbeddingProvider:
        from requisite.rag.embeddings.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(**kwargs)

    def _build_gemini(**kwargs: Any) -> BaseEmbeddingProvider:
        from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider(**kwargs)

    registry.register("openai", _build_openai)
    registry.register("gemini", _build_gemini)


def _register_builtin_vector_stores(registry: VectorStoreRegistry) -> None:
    def _build_in_memory(**kwargs: Any) -> BaseVectorStore:
        from requisite.rag.vectorstores.in_memory import InMemoryVectorStore

        return InMemoryVectorStore(**kwargs)

    registry.register("in_memory", _build_in_memory)


default_embedding_registry = EmbeddingRegistry()
_register_builtin_embeddings(default_embedding_registry)

default_vector_store_registry = VectorStoreRegistry()
_register_builtin_vector_stores(default_vector_store_registry)
