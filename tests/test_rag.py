"""Unit tests for requisite.rag.

Embedding providers are exercised via a deterministic fake
(no real OpenAI/Gemini API calls); the OpenAI/Gemini embedding provider
*classes* are additionally tested against faked SDK modules via
sys.modules injection, matching the pattern used for the chat providers.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from requisite.capabilities.registry import CapabilityRegistry
from requisite.core.exceptions import ConfigurationException
from requisite.rag.base import BaseEmbeddingProvider, Chunk
from requisite.rag.chunking import chunk_text
from requisite.rag.factory import (
    EmbeddingRegistry,
    VectorStoreRegistry,
    default_embedding_registry,
    default_vector_store_registry,
)
from requisite.rag.retriever import Retriever
from requisite.rag.vectorstores.in_memory import InMemoryVectorStore, _cosine_similarity

# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_short_text_returns_single_chunk() -> None:
    assert chunk_text("hello world", chunk_size=1000, chunk_overlap=100) == ["hello world"]


def test_chunk_text_empty_returns_empty_list() -> None:
    assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_chunk_text_splits_long_text() -> None:
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    # every chunk should respect the size budget reasonably
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_overlap_shares_content_between_chunks() -> None:
    text = "sentence number " + " ".join(f"item{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    # last words of chunk[0] should reappear near the start of chunk[1]
    tail_words = set(chunks[0].split()[-3:])
    head_words = set(chunks[1].split()[:10])
    assert tail_words & head_words


def test_chunk_text_rejects_invalid_sizes() -> None:
    with pytest.raises(ConfigurationException):
        chunk_text("x", chunk_size=0)
    with pytest.raises(ConfigurationException):
        chunk_text("x" * 100, chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ConfigurationException):
        chunk_text("x" * 100, chunk_size=10, chunk_overlap=10)


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_in_memory_vector_store_add_and_search() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            Chunk(id="1", text="cats are great", embedding=[1.0, 0.0]),
            Chunk(id="2", text="dogs are great", embedding=[0.9, 0.1]),
            Chunk(id="3", text="totally unrelated", embedding=[0.0, 1.0]),
        ]
    )
    results = store.search([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == "1"
    assert results[0].score >= results[1].score


def test_in_memory_vector_store_delete() -> None:
    store = InMemoryVectorStore()
    store.add([Chunk(id="1", text="hi", embedding=[1.0])])
    store.delete(["1"])
    assert store.search([1.0], top_k=5) == []


def test_in_memory_vector_store_delete_missing_is_noop() -> None:
    store = InMemoryVectorStore()
    store.delete(["does-not-exist"])  # no error


def test_in_memory_vector_store_ignores_chunks_without_embedding() -> None:
    store = InMemoryVectorStore()
    store.add([Chunk(id="1", text="no embedding")])
    assert store.search([1.0], top_k=5) == []


def test_in_memory_vector_store_name() -> None:
    assert InMemoryVectorStore().name == "in_memory"


@pytest.mark.asyncio
async def test_in_memory_vector_store_async_methods() -> None:
    store = InMemoryVectorStore()
    await store.aadd([Chunk(id="1", text="hi", embedding=[1.0, 0.0])])
    results = await store.asearch([1.0, 0.0], top_k=1)
    assert results[0].chunk.id == "1"
    await store.adelete(["1"])
    assert await store.asearch([1.0, 0.0], top_k=1) == []


# ---------------------------------------------------------------------------
# Retriever (with a deterministic fake embedding provider)
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, dependency-free embedding: character-frequency vector."""

    @property
    def name(self) -> str:
        return "fake"

    def embed(self, texts):
        return [[float(t.lower().count(c)) for c in "abcde"] for t in texts]


def test_retriever_add_texts_and_retrieve() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    retriever.add_texts(
        [
            "Paris is the capital of France.",
            "Bananas are a yellow fruit.",
        ]
    )
    results = retriever.retrieve("capital of France", top_k=1)
    assert len(results) == 1
    assert "Paris" in results[0].chunk.text


def test_retriever_add_texts_with_metadata() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    retriever.add_texts(["short doc"], metadatas=[{"source": "doc1.txt"}])
    results = retriever.retrieve("short doc", top_k=1)
    assert results[0].chunk.metadata == {"source": "doc1.txt"}


def test_retriever_add_texts_mismatched_metadata_length_raises() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    with pytest.raises(ConfigurationException):
        retriever.add_texts(["a", "b"], metadatas=[{"x": 1}])


def test_retriever_add_texts_empty_list_returns_empty() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    assert retriever.add_texts([]) == []


def test_retriever_as_tool_returns_formatted_results() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    retriever.add_texts(["Paris is the capital of France."])
    tool = retriever.as_tool()
    assert tool.name == "knowledge_base"
    output = tool.execute(query="capital of France")
    assert "Paris" in output
    assert "score=" in output


def test_retriever_as_tool_no_results() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    tool = retriever.as_tool()
    assert tool.execute(query="anything") == "No relevant information found."


def test_retriever_as_tool_custom_name() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    tool = retriever.as_tool(name="docs_search", description="Search internal docs.")
    assert tool.name == "docs_search"
    assert tool.description == "Search internal docs."


@pytest.mark.asyncio
async def test_retriever_async_add_and_retrieve() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    await retriever.aadd_texts(["Paris is the capital of France."])
    results = await retriever.aretrieve("capital of France")
    assert "Paris" in results[0].chunk.text


# ---------------------------------------------------------------------------
# Capability bridge
# ---------------------------------------------------------------------------


def test_retriever_registers_as_capability() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    retriever.add_texts(["Paris is the capital of France."])

    capabilities = CapabilityRegistry()
    capabilities.register("knowledge_base", retriever.as_tool())

    resolved = capabilities.resolve("knowledge_base")
    assert "Paris" in resolved.execute(query="capital of France")


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def test_default_embedding_registry_has_builtins() -> None:
    assert "openai" in default_embedding_registry.available()
    assert "gemini" in default_embedding_registry.available()


def test_default_vector_store_registry_has_in_memory() -> None:
    assert "in_memory" in default_vector_store_registry.available()
    store = default_vector_store_registry.create("in_memory")
    assert store.name == "in_memory"


def test_embedding_registry_unknown_raises() -> None:
    registry = EmbeddingRegistry()
    with pytest.raises(ConfigurationException):
        registry.create("nope")


def test_embedding_registry_register_empty_name_raises() -> None:
    registry = EmbeddingRegistry()
    with pytest.raises(ConfigurationException):
        registry.register("", lambda **kw: FakeEmbeddingProvider())


def test_vector_store_registry_unknown_raises() -> None:
    registry = VectorStoreRegistry()
    with pytest.raises(ConfigurationException):
        registry.create("nope")


def test_vector_store_registry_register_empty_name_raises() -> None:
    registry = VectorStoreRegistry()
    with pytest.raises(ConfigurationException):
        registry.register("", lambda **kw: InMemoryVectorStore())


# ---------------------------------------------------------------------------
# OpenAI / Gemini embedding providers (SDK faked via sys.modules)
# ---------------------------------------------------------------------------


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingItem(v) for v in vectors]


class _FakeOpenAIEmbeddingClient:
    def __init__(self, **kwargs: Any) -> None:
        self.embeddings = types.SimpleNamespace(create=self._create)

    def _create(self, *, input, model):
        return _FakeEmbeddingResponse([[float(len(t))] for t in input])


def test_openai_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAIEmbeddingClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    from requisite.rag.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(api_key="sk-test")
    vectors = provider.embed(["ab", "abcd"])
    assert vectors == [[2.0], [4.0]]


def test_openai_embedding_provider_missing_key_raises() -> None:
    from requisite.rag.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(api_key=None)
    with pytest.raises(ConfigurationException):
        provider.embed(["hi"])


class _FakeGeminiEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeGeminiEmbedResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = [_FakeGeminiEmbedding(v) for v in vectors]


class _FakeGeminiEmbeddingClient:
    def __init__(self, **kwargs: Any) -> None:
        self.models = types.SimpleNamespace(embed_content=self._embed_content)

    def _embed_content(self, *, model, contents):
        return _FakeGeminiEmbedResponse([[float(len(t))] for t in contents])


def test_gemini_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = _FakeGeminiEmbeddingClient  # type: ignore[attr-defined]
    fake_google_module = types.ModuleType("google")
    fake_google_module.genai = fake_genai_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)

    from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider(api_key="g-test")
    vectors = provider.embed(["ab", "abcd"])
    assert vectors == [[2.0], [4.0]]


def test_gemini_embedding_provider_missing_key_raises() -> None:
    from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider(api_key=None)
    with pytest.raises(ConfigurationException):
        provider.embed(["hi"])
