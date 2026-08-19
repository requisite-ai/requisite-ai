"""Unit tests for requisite.rag.

Embedding providers are exercised via a deterministic fake
(no real OpenAI/Gemini API calls); the OpenAI/Gemini embedding provider
*classes* are additionally tested against faked SDK modules via
sys.modules injection, matching the pattern used for the chat providers.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

import pytest

from requisite.ai import AI
from requisite.capabilities.registry import CapabilityRegistry
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry
from requisite.rag.base import BaseEmbeddingProvider, Chunk, ScoredChunk
from requisite.rag.bm25 import BM25Retriever
from requisite.rag.chunking import chunk_text
from requisite.rag.factory import (
    EmbeddingRegistry,
    VectorStoreRegistry,
    default_embedding_registry,
    default_vector_store_registry,
)
from requisite.rag.hybrid_retriever import HybridRetriever
from requisite.rag.reranker import LLMReranker
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


def test_in_memory_vector_store_search_filters_by_metadata() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            Chunk(id="1", text="a", embedding=[1.0, 0.0], metadata={"session_id": "s1"}),
            Chunk(id="2", text="b", embedding=[1.0, 0.0], metadata={"session_id": "s2"}),
        ]
    )
    results = store.search([1.0, 0.0], top_k=5, filter={"session_id": "s1"})
    assert [r.chunk.id for r in results] == ["1"]


def test_in_memory_vector_store_search_unfiltered_returns_everything() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            Chunk(id="1", text="a", embedding=[1.0, 0.0], metadata={"session_id": "s1"}),
            Chunk(id="2", text="b", embedding=[1.0, 0.0], metadata={"session_id": "s2"}),
        ]
    )
    results = store.search([1.0, 0.0], top_k=5)
    assert {r.chunk.id for r in results} == {"1", "2"}


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


def test_default_vector_store_registry_has_pinecone_and_weaviate() -> None:
    assert "pinecone" in default_vector_store_registry.available()
    assert "weaviate" in default_vector_store_registry.available()


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
# BM25Retriever
# ---------------------------------------------------------------------------


def test_bm25_retriever_ranks_keyword_matches_higher() -> None:
    retriever = BM25Retriever()
    retriever.add_texts(
        [
            "The quick brown fox jumps over the lazy dog.",
            "Bananas are a yellow tropical fruit.",
        ]
    )
    results = retriever.retrieve("quick fox", top_k=2)
    assert len(results) == 1
    assert "fox" in results[0].chunk.text.lower()


def test_bm25_retriever_no_matching_terms_returns_empty() -> None:
    retriever = BM25Retriever()
    retriever.add_texts(["completely unrelated text"])
    assert retriever.retrieve("zzznotpresentzzz") == []


def test_bm25_retriever_empty_corpus_returns_empty() -> None:
    assert BM25Retriever().retrieve("anything") == []


def test_bm25_retriever_delete_removes_from_results() -> None:
    retriever = BM25Retriever()
    ids = retriever.add_texts(["the quick brown fox"])
    retriever.delete(ids)
    assert retriever.retrieve("quick fox") == []


def test_bm25_retriever_accepts_k1_and_b() -> None:
    retriever = BM25Retriever(k1=1.2, b=0.5)
    retriever.add_texts(["the quick brown fox jumps"])
    assert len(retriever.retrieve("fox")) == 1


def test_bm25_retriever_add_texts_mismatched_metadata_length_raises() -> None:
    retriever = BM25Retriever()
    with pytest.raises(ConfigurationException):
        retriever.add_texts(["a", "b"], metadatas=[{"x": 1}])


def test_bm25_retriever_add_texts_empty_list_returns_empty() -> None:
    assert BM25Retriever().add_texts([]) == []


def test_bm25_retriever_name() -> None:
    assert BM25Retriever().name == "bm25"


def test_bm25_retriever_as_tool_returns_formatted_results() -> None:
    retriever = BM25Retriever()
    retriever.add_texts(["Paris is the capital of France."])
    tool = retriever.as_tool()
    output = tool.execute(query="capital of France")
    assert "Paris" in output
    assert "score=" in output


def test_bm25_retriever_as_tool_no_results() -> None:
    tool = BM25Retriever().as_tool()
    assert tool.execute(query="anything") == "No relevant information found."


@pytest.mark.asyncio
async def test_bm25_retriever_async_add_and_retrieve() -> None:
    retriever = BM25Retriever()
    await retriever.aadd_texts(["the quick brown fox jumps over the lazy dog"])
    results = await retriever.aretrieve("quick fox")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# HybridRetriever (dense via FakeEmbeddingProvider + BM25, fused via RRF)
# ---------------------------------------------------------------------------


def test_hybrid_retriever_fuses_dense_and_bm25_winners() -> None:
    # doc_dense: no words in common with the query, but its FakeEmbeddingProvider
    # a-e char-count vector points the same direction as the query's.
    # doc_bm25: shares real words with the query (BM25-favorable), and still has
    # *some* dense similarity (real English words share letters incidentally).
    # doc_noise: shares nothing with the query on either signal -- the control.
    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    retriever.add_texts(
        [
            "aabbccddee aabbccddee",  # doc_dense
            "xylophone information about music",  # doc_bm25
            "zzz qqq www ppp",  # doc_noise -- zero a-e chars, zero shared words
        ]
    )

    results = retriever.retrieve("find the xylophone information", top_k=2)

    result_texts = [r.chunk.text for r in results]
    assert len(results) == 2
    assert "xylophone information about music" in result_texts
    assert "aabbccddee aabbccddee" in result_texts
    assert "zzz qqq www ppp" not in result_texts


def test_hybrid_retriever_add_texts_shares_chunk_ids_across_both_sides() -> None:
    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    ids = retriever.add_texts(["the quick brown fox"])

    dense_hit = retriever.vector_store.search(
        retriever.embedding_provider.embed_one("quick fox"), top_k=1
    )
    bm25_hit = retriever._bm25_index.search("quick fox", top_k=1)  # noqa: SLF001

    assert dense_hit[0].chunk.id == ids[0]
    assert bm25_hit[0].chunk.id == ids[0]


def test_hybrid_retriever_add_texts_empty_list_returns_empty() -> None:
    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    assert retriever.add_texts([]) == []


def test_hybrid_retriever_add_texts_mismatched_metadata_length_raises() -> None:
    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    with pytest.raises(ConfigurationException):
        retriever.add_texts(["a", "b"], metadatas=[{"x": 1}])


def test_hybrid_retriever_name() -> None:
    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    assert retriever.name == "hybrid"


def test_hybrid_retriever_as_tool_returns_formatted_results() -> None:
    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    retriever.add_texts(["Paris is the capital of France."])
    tool = retriever.as_tool()
    output = tool.execute(query="capital of France")
    assert "Paris" in output
    assert "score=" in output


@pytest.mark.asyncio
async def test_hybrid_retriever_async_add_and_retrieve() -> None:
    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    await retriever.aadd_texts(["Paris is the capital of France."])
    results = await retriever.aretrieve("capital of France")
    assert "Paris" in results[0].chunk.text


# ---------------------------------------------------------------------------
# LLMReranker (scripted fake provider, no real LLM calls)
# ---------------------------------------------------------------------------


class ScriptedRerankProvider(BaseProvider):
    """Returns a fixed response_model payload, regardless of the prompt."""

    def __init__(self, parsed: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self._parsed = parsed

    @property
    def name(self) -> str:
        return "scripted_rerank"

    def chat(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> ChatResponse:
        return ChatResponse(content="", model=self._model, provider=self.name, parsed=self._parsed)

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> ChatResponse:
        return self.chat(
            messages,
            model=model,
            temperature=temperature,
            tools=tools,
            response_model=response_model,
            **kwargs,
        )

    def stream(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ):  # pragma: no cover
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


def _reranker_with_scores(parsed_scores: dict[str, Any]) -> LLMReranker:
    registry = ProviderRegistry()
    registry.register("scripted_rerank", lambda **kw: ScriptedRerankProvider(parsed_scores))
    settings = Settings(default_provider="scripted_rerank", model="fake-model")
    return LLMReranker(ai=AI(settings=settings, registry=registry))


def test_llm_reranker_reorders_by_relevance() -> None:
    reranker = _reranker_with_scores(
        {"scores": [{"chunk_id": "c1", "relevance": 2.0}, {"chunk_id": "c2", "relevance": 9.0}]}
    )
    candidates = [
        ScoredChunk(chunk=Chunk(id="c1", text="first"), score=0.9),
        ScoredChunk(chunk=Chunk(id="c2", text="second"), score=0.1),
    ]
    result = reranker.rerank("query", candidates)
    assert [r.chunk.id for r in result] == ["c2", "c1"]
    assert result[0].score == 9.0


def test_llm_reranker_missing_score_falls_back_to_zero() -> None:
    reranker = _reranker_with_scores({"scores": [{"chunk_id": "c1", "relevance": 5.0}]})
    candidates = [
        ScoredChunk(chunk=Chunk(id="c1", text="first"), score=0.1),
        ScoredChunk(chunk=Chunk(id="c2", text="second"), score=0.9),
    ]
    result = reranker.rerank("query", candidates)
    assert result[-1].chunk.id == "c2"
    assert result[-1].score == 0.0


def test_llm_reranker_top_k_truncates() -> None:
    reranker = _reranker_with_scores(
        {"scores": [{"chunk_id": "c1", "relevance": 1.0}, {"chunk_id": "c2", "relevance": 2.0}]}
    )
    candidates = [
        ScoredChunk(chunk=Chunk(id="c1", text="first"), score=0.0),
        ScoredChunk(chunk=Chunk(id="c2", text="second"), score=0.0),
    ]
    result = reranker.rerank("query", candidates, top_k=1)
    assert len(result) == 1
    assert result[0].chunk.id == "c2"


def test_llm_reranker_empty_results_returns_empty() -> None:
    reranker = _reranker_with_scores({"scores": []})
    assert reranker.rerank("query", []) == []


@pytest.mark.asyncio
async def test_llm_reranker_arerank() -> None:
    reranker = _reranker_with_scores(
        {"scores": [{"chunk_id": "c1", "relevance": 1.0}, {"chunk_id": "c2", "relevance": 9.0}]}
    )
    candidates = [
        ScoredChunk(chunk=Chunk(id="c1", text="first"), score=0.0),
        ScoredChunk(chunk=Chunk(id="c2", text="second"), score=0.0),
    ]
    result = await reranker.arerank("query", candidates)
    assert [r.chunk.id for r in result] == ["c2", "c1"]


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


def test_openai_embedding_provider_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from requisite.rag.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(api_key=None)
    with pytest.raises(ConfigurationException):
        provider.embed(["hi"])


def test_openai_embedding_provider_falls_back_to_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAIEmbeddingClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    from requisite.rag.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider()
    assert provider.embed(["ab"]) == [[2.0]]


def test_openai_embedding_provider_empty_string_key_does_not_fall_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicitly-passed "" is a deliberate override, not "not provided" --
    # it must still fail even if a real key is ambient in the environment.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    from requisite.rag.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(api_key="")
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


def test_gemini_embedding_provider_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider(api_key=None)
    with pytest.raises(ConfigurationException):
        provider.embed(["hi"])


def test_gemini_embedding_provider_falls_back_to_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-from-env")
    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = _FakeGeminiEmbeddingClient  # type: ignore[attr-defined]
    fake_google_module = types.ModuleType("google")
    fake_google_module.genai = fake_genai_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)

    from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider()
    assert provider.embed(["ab"]) == [[2.0]]


def test_gemini_embedding_provider_empty_string_key_does_not_fall_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicitly-passed "" is a deliberate override, not "not provided" --
    # it must still fail even if a real key is ambient in the environment.
    monkeypatch.setenv("GEMINI_API_KEY", "g-from-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-from-env-2")
    from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider(api_key="")
    with pytest.raises(ConfigurationException):
        provider.embed(["hi"])


# ---------------------------------------------------------------------------
# Pinecone vector store (SDK faked via sys.modules)
# ---------------------------------------------------------------------------


class _FakePineconeIndex:
    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def upsert(self, *, vectors: Any, namespace: str = "") -> None:
        for chunk_id, values, metadata in vectors:
            self._vectors[chunk_id] = (list(values), dict(metadata))

    def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str = "",
        include_metadata: bool = False,
        filter: Optional[dict[str, Any]] = None,  # noqa: A002
    ) -> Any:
        candidates = self._vectors.items()
        if filter:
            candidates = [
                (chunk_id, (values, metadata))
                for chunk_id, (values, metadata) in candidates
                if all(metadata.get(k) == v for k, v in filter.items())
            ]
        scored = [
            (chunk_id, _cosine_similarity(vector, values), metadata)
            for chunk_id, (values, metadata) in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        matches = [
            types.SimpleNamespace(id=chunk_id, score=score, metadata=metadata)
            for chunk_id, score, metadata in scored[:top_k]
        ]
        return types.SimpleNamespace(matches=matches)

    def delete(self, *, ids: list[str], namespace: str = "") -> None:
        for chunk_id in ids:
            self._vectors.pop(chunk_id, None)


class _FakePineconeClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._indexes: dict[str, _FakePineconeIndex] = {}

    def has_index(self, name: str) -> bool:
        return name in self._indexes

    def create_index(self, *, name: str, spec: Any, dimension: int, metric: str) -> None:
        self._indexes[name] = _FakePineconeIndex()

    def index(self, *, name: str = "", host: str = "") -> _FakePineconeIndex:
        return self._indexes[name]


@pytest.fixture
def fake_pinecone_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_module = types.ModuleType("pinecone")
    fake_module.Pinecone = _FakePineconeClient  # type: ignore[attr-defined]
    fake_module.ServerlessSpec = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pinecone", fake_module)
    return fake_module


def test_pinecone_vector_store_add_and_search(fake_pinecone_module: types.ModuleType) -> None:
    from requisite.rag.vectorstores.pinecone import PineconeVectorStore

    store = PineconeVectorStore(api_key="pc-test", index_name="demo", dimension=2)
    store.add(
        [
            Chunk(id="1", text="cats are great", embedding=[1.0, 0.0], metadata={"topic": "pets"}),
            Chunk(id="2", text="dogs are great", embedding=[0.0, 1.0]),
        ]
    )
    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "1"
    assert results[0].chunk.text == "cats are great"
    assert results[0].chunk.metadata == {"topic": "pets"}


def test_pinecone_vector_store_search_filters_by_metadata(
    fake_pinecone_module: types.ModuleType,
) -> None:
    from requisite.rag.vectorstores.pinecone import PineconeVectorStore

    store = PineconeVectorStore(api_key="pc-test", index_name="demo", dimension=2)
    store.add(
        [
            Chunk(id="1", text="a", embedding=[1.0, 0.0], metadata={"session_id": "s1"}),
            Chunk(id="2", text="b", embedding=[1.0, 0.0], metadata={"session_id": "s2"}),
        ]
    )
    results = store.search([1.0, 0.0], top_k=5, filter={"session_id": "s1"})
    assert [r.chunk.id for r in results] == ["1"]


def test_pinecone_vector_store_delete(fake_pinecone_module: types.ModuleType) -> None:
    from requisite.rag.vectorstores.pinecone import PineconeVectorStore

    store = PineconeVectorStore(api_key="pc-test", index_name="demo", dimension=2)
    store.add([Chunk(id="1", text="cats", embedding=[1.0, 0.0])])
    store.delete(["1"])
    assert store.search([1.0, 0.0], top_k=5) == []


def test_pinecone_vector_store_requires_dimension_to_create(
    fake_pinecone_module: types.ModuleType,
) -> None:
    from requisite.rag.vectorstores.pinecone import PineconeVectorStore

    store = PineconeVectorStore(api_key="pc-test", index_name="demo")
    with pytest.raises(ConfigurationException, match="dimension"):
        store.add([Chunk(id="1", text="cats", embedding=[1.0, 0.0])])


def test_pinecone_vector_store_missing_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pinecone", None)
    from requisite.rag.vectorstores.pinecone import PineconeVectorStore

    store = PineconeVectorStore(api_key="pc-test", index_name="demo", dimension=2)
    with pytest.raises(ConfigurationException):
        store.add([Chunk(id="1", text="cats", embedding=[1.0, 0.0])])


# ---------------------------------------------------------------------------
# Weaviate vector store (SDK faked via sys.modules)
# ---------------------------------------------------------------------------


class _FakeWeaviateObject:
    def __init__(self, properties: dict[str, Any], distance: float) -> None:
        self.properties = properties
        self.metadata = types.SimpleNamespace(distance=distance)


class _FakeWeaviateDataObject:
    def __init__(
        self, *, properties: dict[str, Any], uuid: str, vector: Optional[list[float]] = None
    ) -> None:
        self.properties = properties
        self.uuid = uuid
        self.vector = vector


class _FakeWeaviateData:
    def __init__(self, store: dict[str, tuple[dict[str, Any], list[float]]]) -> None:
        self._store = store

    def insert_many(self, objects: Any) -> Any:
        for obj in objects:
            self._store[obj.uuid] = (obj.properties, obj.vector or [])
        return types.SimpleNamespace()

    def delete_by_id(self, uuid: str) -> bool:
        return self._store.pop(uuid, None) is not None


class _FakeWeaviateQuery:
    def __init__(self, store: dict[str, tuple[dict[str, Any], list[float]]]) -> None:
        self._store = store

    def near_vector(
        self,
        *,
        near_vector: list[float],
        limit: int,
        offset: int = 0,
        return_metadata: Any = None,
        return_properties: Any = None,
    ) -> Any:
        scored = [
            (1.0 - _cosine_similarity(near_vector, vector), properties)
            for properties, vector in self._store.values()
        ]
        scored.sort(key=lambda item: item[0])
        page = scored[offset : offset + limit]
        objects = [_FakeWeaviateObject(props, distance) for distance, props in page]
        return types.SimpleNamespace(objects=objects)


class _FakeWeaviateCollection:
    def __init__(self) -> None:
        self._store: dict[str, tuple[dict[str, Any], list[float]]] = {}
        self.data = _FakeWeaviateData(self._store)
        self.query = _FakeWeaviateQuery(self._store)


class _FakeWeaviateCollections:
    def __init__(self) -> None:
        self._collections: dict[str, _FakeWeaviateCollection] = {}

    def exists(self, name: str) -> bool:
        return name in self._collections

    def create(self, *, name: str, **kwargs: Any) -> _FakeWeaviateCollection:
        collection = _FakeWeaviateCollection()
        self._collections[name] = collection
        return collection

    def get(self, name: str) -> _FakeWeaviateCollection:
        return self._collections[name]


class _FakeWeaviateClient:
    def __init__(self) -> None:
        self.collections = _FakeWeaviateCollections()


@pytest.fixture
def fake_weaviate_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_module = types.ModuleType("weaviate")
    fake_module.connect_to_local = lambda **kwargs: _FakeWeaviateClient()  # type: ignore[attr-defined]
    fake_module.connect_to_weaviate_cloud = (  # type: ignore[attr-defined]
        lambda **kwargs: _FakeWeaviateClient()
    )

    fake_classes_module = types.ModuleType("weaviate.classes")
    fake_config_module = types.ModuleType("weaviate.classes.config")
    fake_config_module.Configure = types.SimpleNamespace(  # type: ignore[attr-defined]
        Vectorizer=types.SimpleNamespace(none=lambda: None),
        VectorIndex=types.SimpleNamespace(hnsw=lambda **kwargs: None),
    )
    fake_config_module.Property = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    fake_config_module.DataType = types.SimpleNamespace(TEXT="text")  # type: ignore[attr-defined]
    fake_config_module.VectorDistances = types.SimpleNamespace(COSINE="cosine")  # type: ignore[attr-defined]

    fake_data_module = types.ModuleType("weaviate.classes.data")
    fake_data_module.DataObject = _FakeWeaviateDataObject  # type: ignore[attr-defined]

    fake_query_module = types.ModuleType("weaviate.classes.query")
    fake_query_module.MetadataQuery = lambda **kwargs: kwargs  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "weaviate", fake_module)
    monkeypatch.setitem(sys.modules, "weaviate.classes", fake_classes_module)
    monkeypatch.setitem(sys.modules, "weaviate.classes.config", fake_config_module)
    monkeypatch.setitem(sys.modules, "weaviate.classes.data", fake_data_module)
    monkeypatch.setitem(sys.modules, "weaviate.classes.query", fake_query_module)
    return fake_module


def test_weaviate_vector_store_add_and_search(fake_weaviate_module: types.ModuleType) -> None:
    from requisite.rag.vectorstores.weaviate import WeaviateVectorStore

    store = WeaviateVectorStore(url="https://example.weaviate.network", api_key="w-test")
    store.add(
        [
            Chunk(id="1", text="cats are great", embedding=[1.0, 0.0], metadata={"topic": "pets"}),
            Chunk(id="2", text="dogs are great", embedding=[0.0, 1.0]),
        ]
    )
    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "1"
    assert results[0].chunk.text == "cats are great"
    assert results[0].chunk.metadata == {"topic": "pets"}


def test_weaviate_vector_store_search_filters_by_metadata(
    fake_weaviate_module: types.ModuleType,
) -> None:
    from requisite.rag.vectorstores.weaviate import WeaviateVectorStore

    store = WeaviateVectorStore(url="https://example.weaviate.network", api_key="w-test")
    store.add(
        [
            Chunk(id="1", text="a", embedding=[1.0, 0.0], metadata={"session_id": "s1"}),
            Chunk(id="2", text="b", embedding=[1.0, 0.0], metadata={"session_id": "s2"}),
        ]
    )
    results = store.search([1.0, 0.0], top_k=5, filter={"session_id": "s1"})
    assert [r.chunk.id for r in results] == ["1"]


def test_weaviate_vector_store_search_paginates_past_first_page_to_find_filter_match(
    fake_weaviate_module: types.ModuleType,
) -> None:
    from requisite.rag.vectorstores.weaviate import WeaviateVectorStore

    store = WeaviateVectorStore(url="https://example.weaviate.network", api_key="w-test")
    # 200 chunks that are a perfect vector match but belong to a different
    # session -- they fill the entire first page. The one chunk that
    # actually matches the filter is vector-dissimilar, so it only appears
    # on the second page. Proves search() pages past the first window
    # instead of silently missing a match outside it (the original bug).
    noise = [
        Chunk(
            id=f"noise-{i}",
            text="noise",
            embedding=[1.0, 0.0],
            metadata={"session_id": "other"},
        )
        for i in range(200)
    ]
    target = Chunk(id="target", text="target", embedding=[0.0, 1.0], metadata={"session_id": "s1"})
    store.add([*noise, target])

    results = store.search([1.0, 0.0], top_k=5, filter={"session_id": "s1"})
    assert [r.chunk.id for r in results] == ["target"]


def test_weaviate_vector_store_search_top_k_zero_returns_empty(
    fake_weaviate_module: types.ModuleType,
) -> None:
    from requisite.rag.vectorstores.weaviate import WeaviateVectorStore

    store = WeaviateVectorStore(url="https://example.weaviate.network", api_key="w-test")
    store.add([Chunk(id="1", text="a", embedding=[1.0, 0.0], metadata={"session_id": "s1"})])

    assert store.search([1.0, 0.0], top_k=0, filter={"session_id": "s1"}) == []
    assert store.search([1.0, 0.0], top_k=0) == []


def test_weaviate_vector_store_delete(fake_weaviate_module: types.ModuleType) -> None:
    from requisite.rag.vectorstores.weaviate import WeaviateVectorStore

    store = WeaviateVectorStore(url="https://example.weaviate.network", api_key="w-test")
    store.add([Chunk(id="1", text="cats", embedding=[1.0, 0.0])])
    store.delete(["1"])
    assert store.search([1.0, 0.0], top_k=5) == []


def test_weaviate_vector_store_local_connection_needs_no_api_key(
    fake_weaviate_module: types.ModuleType,
) -> None:
    from requisite.rag.vectorstores.weaviate import WeaviateVectorStore

    store = WeaviateVectorStore()
    store.add([Chunk(id="1", text="cats", embedding=[1.0, 0.0])])
    assert store.search([1.0, 0.0], top_k=1)[0].chunk.id == "1"


def test_weaviate_vector_store_cloud_requires_api_key(
    fake_weaviate_module: types.ModuleType,
) -> None:
    from requisite.rag.vectorstores.weaviate import WeaviateVectorStore

    store = WeaviateVectorStore(url="https://example.weaviate.network")
    with pytest.raises(ConfigurationException, match="api_key"):
        store.add([Chunk(id="1", text="cats", embedding=[1.0, 0.0])])


def test_weaviate_vector_store_missing_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "weaviate", None)
    from requisite.rag.vectorstores.weaviate import WeaviateVectorStore

    store = WeaviateVectorStore()
    with pytest.raises(ConfigurationException):
        store.add([Chunk(id="1", text="cats", embedding=[1.0, 0.0])])
