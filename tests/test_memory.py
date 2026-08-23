"""Unit tests for the memory package and Agent's memory/session_id integration."""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from requisite.agents.agent import Agent
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException, MemoryException
from requisite.core.interfaces import ChatResponse, Message, StreamChunk
from requisite.memory.factory import MemoryRegistry, default_registry
from requisite.memory.in_process import InProcessMemory
from requisite.memory.sqlite import SQLiteMemory
from requisite.memory.vector import VectorMemory
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry
from requisite.rag.base import BaseEmbeddingProvider
from requisite.rag.vectorstores import InMemoryVectorStore


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic fake: a one-hot-ish vector over a small fixed
    vocabulary, so cosine similarity meaningfully discriminates topics
    without any real embedding model."""

    _VOCAB = ("blue", "color", "pizza", "food")

    @property
    def name(self) -> str:
        return "fake"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0 if word in text.lower() else 0.0 for word in self._VOCAB] for text in texts]


# ---------------------------------------------------------------------------
# InProcessMemory / MemoryRegistry
# ---------------------------------------------------------------------------


def test_in_process_memory_load_append_clear() -> None:
    memory = InProcessMemory()
    assert memory.load("s1") == []

    memory.append("s1", Message.user("hi"))
    memory.append("s1", Message.assistant("hello!"))
    assert [m.content for m in memory.load("s1")] == ["hi", "hello!"]

    memory.clear("s1")
    assert memory.load("s1") == []


def test_in_process_memory_sessions_are_isolated() -> None:
    memory = InProcessMemory()
    memory.append("s1", Message.user("for session 1"))
    memory.append("s2", Message.user("for session 2"))
    assert [m.content for m in memory.load("s1")] == ["for session 1"]
    assert [m.content for m in memory.load("s2")] == ["for session 2"]


def test_in_process_memory_load_does_not_leak_storage_for_probed_sessions() -> None:
    """Regression test: load() on a session id that was never appended to
    must not permanently allocate storage for it -- indexing a
    defaultdict auto-vivifies an entry, unboundedly leaking one dict
    entry per distinct probed id over a long-running process. See
    docs/adr/0031-code-review-fixes.md."""
    memory = InProcessMemory()
    for i in range(5):
        assert memory.load(f"probe-session-{i}") == []
    assert len(memory._sessions) == 0  # noqa: SLF001


def test_in_process_memory_name() -> None:
    assert InProcessMemory().name == "in_process"


@pytest.mark.asyncio
async def test_in_process_memory_async_methods() -> None:
    memory = InProcessMemory()
    await memory.aappend("s1", Message.user("hi"))
    assert [m.content for m in await memory.aload("s1")] == ["hi"]
    await memory.aclear("s1")
    assert await memory.aload("s1") == []


def test_default_memory_registry_has_in_process() -> None:
    assert "in_process" in default_registry.available()
    memory = default_registry.create("in_process")
    assert memory.name == "in_process"


def test_memory_registry_unknown_backend_raises() -> None:
    registry = MemoryRegistry()
    with pytest.raises(ConfigurationException):
        registry.create("does-not-exist")


def test_memory_registry_register_empty_name_raises() -> None:
    registry = MemoryRegistry()
    with pytest.raises(ConfigurationException):
        registry.register("", InProcessMemory)


def test_default_memory_registry_has_sqlite(tmp_path: Path) -> None:
    assert "sqlite" in default_registry.available()
    memory = default_registry.create("sqlite", db_path=str(tmp_path / "registry.db"))
    assert memory.name == "sqlite"


def test_default_memory_registry_has_redis() -> None:
    # Construction alone never imports the `redis` package -- the SDK is
    # only touched lazily on first load/append/clear -- so this needs no
    # fake module, unlike the RedisMemory tests below.
    assert "redis" in default_registry.available()
    memory = default_registry.create("redis")
    assert memory.name == "redis"


def test_default_memory_registry_has_vector() -> None:
    assert "vector" in default_registry.available()
    memory = default_registry.create(
        "vector", embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    assert memory.name == "vector"


# ---------------------------------------------------------------------------
# SQLiteMemory
# ---------------------------------------------------------------------------


def test_sqlite_memory_load_append_clear(tmp_path: Path) -> None:
    memory = SQLiteMemory(db_path=str(tmp_path / "memory.db"))
    assert memory.load("s1") == []

    memory.append("s1", Message.user("hi"))
    memory.append("s1", Message.assistant("hello!"))
    assert [m.content for m in memory.load("s1")] == ["hi", "hello!"]

    memory.clear("s1")
    assert memory.load("s1") == []


def test_sqlite_memory_sessions_are_isolated(tmp_path: Path) -> None:
    memory = SQLiteMemory(db_path=str(tmp_path / "memory.db"))
    memory.append("s1", Message.user("for session 1"))
    memory.append("s2", Message.user("for session 2"))
    assert [m.content for m in memory.load("s1")] == ["for session 1"]
    assert [m.content for m in memory.load("s2")] == ["for session 2"]


def test_sqlite_memory_name(tmp_path: Path) -> None:
    assert SQLiteMemory(db_path=str(tmp_path / "memory.db")).name == "sqlite"


def test_sqlite_memory_persists_across_instances(tmp_path: Path) -> None:
    db_path = str(tmp_path / "memory.db")
    first = SQLiteMemory(db_path=db_path)
    first.append("s1", Message.user("remember this"))

    second = SQLiteMemory(db_path=db_path)
    assert [m.content for m in second.load("s1")] == ["remember this"]


@pytest.mark.asyncio
async def test_sqlite_memory_async_methods(tmp_path: Path) -> None:
    memory = SQLiteMemory(db_path=str(tmp_path / "memory.db"))
    await memory.aappend("s1", Message.user("hi"))
    assert [m.content for m in await memory.aload("s1")] == ["hi"]
    await memory.aclear("s1")
    assert await memory.aload("s1") == []


# ---------------------------------------------------------------------------
# VectorMemory
# ---------------------------------------------------------------------------


def make_vector_memory(**kwargs: Any) -> VectorMemory:
    return VectorMemory(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore(), **kwargs
    )


def test_vector_memory_load_append_clear() -> None:
    memory = make_vector_memory()
    assert memory.load("s1") == []

    memory.append("s1", Message.user("hi"))
    memory.append("s1", Message.assistant("hello!"))
    assert [m.content for m in memory.load("s1")] == ["hi", "hello!"]

    memory.clear("s1")
    assert memory.load("s1") == []


def test_vector_memory_name() -> None:
    assert make_vector_memory().name == "vector"


def test_vector_memory_load_relevant_ranks_by_semantic_similarity() -> None:
    memory = make_vector_memory()
    memory.append("s1", Message.user("My favorite color is blue."))
    memory.append("s1", Message.assistant("Got it, blue is a great color!"))
    memory.append("s1", Message.user("My favorite food is pizza."))

    color_results = memory.load_relevant("s1", "What color do I like?", top_k=1)
    assert "blue" in color_results[0].content.lower()

    food_results = memory.load_relevant("s1", "What food do I like?", top_k=1)
    assert food_results[0].content == "My favorite food is pizza."


def test_vector_memory_load_relevant_is_scoped_to_session() -> None:
    memory = make_vector_memory()
    memory.append("s1", Message.user("My favorite color is blue."))
    memory.append("s2", Message.user("Unrelated session message about blue too."))

    results = memory.load_relevant("s1", "blue", top_k=5)
    assert [r.content for r in results] == ["My favorite color is blue."]


def test_vector_memory_clear_removes_vector_store_entries_too() -> None:
    memory = make_vector_memory()
    memory.append("s1", Message.user("My favorite color is blue."))
    memory.clear("s1")

    assert memory.load_relevant("s1", "blue", top_k=5) == []


def test_vector_memory_uses_default_top_k() -> None:
    memory = make_vector_memory(top_k=1)
    memory.append("s1", Message.user("My favorite color is blue."))
    memory.append("s1", Message.user("My favorite food is pizza."))

    assert len(memory.load_relevant("s1", "color")) == 1


def test_vector_memory_skips_embedding_empty_content() -> None:
    memory = make_vector_memory()
    memory.append("s1", Message.assistant(""))
    # Doesn't raise, and produces no searchable chunk.
    assert memory.load("s1")[0].content == ""
    assert memory.load_relevant("s1", "anything") == []


def test_vector_memory_defaults_to_an_in_process_history_backend() -> None:
    # No history_backend= given -- history shouldn't outlive this instance
    # or be shared with a second, independently-constructed instance.
    memory = VectorMemory(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    memory.append("s1", Message.user("hi"))

    other = VectorMemory(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    assert other.load("s1") == []


def test_vector_memory_accepts_a_persistent_history_backend(tmp_path: Path) -> None:
    memory = VectorMemory(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
        history_backend=SQLiteMemory(db_path=str(tmp_path / "memory.db")),
    )
    memory.append("s1", Message.user("hi"))
    assert [m.content for m in memory.load("s1")] == ["hi"]


@pytest.mark.asyncio
async def test_vector_memory_async_methods() -> None:
    memory = make_vector_memory()
    await memory.aappend("s1", Message.user("My favorite color is blue."))
    await memory.aappend("s1", Message.user("My favorite food is pizza."))

    assert [m.content for m in await memory.aload("s1")] == [
        "My favorite color is blue.",
        "My favorite food is pizza.",
    ]

    results = await memory.aload_relevant("s1", "What color do I like?", top_k=1)
    assert results[0].content == "My favorite color is blue."

    await memory.aclear("s1")
    assert await memory.aload("s1") == []
    assert await memory.aload_relevant("s1", "blue") == []


def test_vector_memory_load_relevant_top_k_zero_returns_empty() -> None:
    # top_k=0 is an explicit request for zero results -- falsy-but-given,
    # must not be silently replaced with the instance default.
    memory = make_vector_memory(top_k=5)
    memory.append("s1", Message.user("My favorite color is blue."))

    assert memory.load_relevant("s1", "blue", top_k=0) == []


@pytest.mark.asyncio
async def test_vector_memory_aload_relevant_top_k_zero_returns_empty() -> None:
    memory = make_vector_memory(top_k=5)
    await memory.aappend("s1", Message.user("My favorite color is blue."))

    assert await memory.aload_relevant("s1", "blue", top_k=0) == []


def test_vector_memory_clear_then_append_does_not_leak_old_content() -> None:
    memory = make_vector_memory()
    memory.append("s1", Message.user("My favorite color is blue."))
    memory.clear("s1")
    memory.append("s1", Message.user("My favorite food is pizza."))

    # Only the post-clear message should ever be recallable -- the
    # pre-clear "blue" content must not resurface under the reused id.
    results = memory.load_relevant("s1", "blue", top_k=5)
    assert [r.content for r in results] == ["My favorite food is pizza."]
    assert [m.content for m in memory.load("s1")] == ["My favorite food is pizza."]


def test_vector_memory_concurrent_appends_produce_unique_chunk_ids() -> None:
    import threading

    memory = make_vector_memory()
    thread_count = 50

    def append_one(i: int) -> None:
        memory.append("s1", Message.user(f"message number {i}"))

    threads = [threading.Thread(target=append_one, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    history = memory.load("s1")
    assert len(history) == thread_count
    # No chunk id collisions -- every concurrently-allocated turn_index was unique.
    assert len(memory.vector_store._chunks) == thread_count  # noqa: SLF001


class SlowEmbeddingProvider(BaseEmbeddingProvider):
    """A fake embedding provider whose embed() blocks until explicitly
    released -- lets a test park an append() call mid-flight (past its
    counter allocation, before its write commits) to exercise races
    against a concurrent clear()."""

    def __init__(self, release_event: threading.Event) -> None:
        self._release_event = release_event

    @property
    def name(self) -> str:
        return "slow"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self._release_event.wait(timeout=5.0)
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_vector_memory_clear_wins_over_concurrent_in_flight_append() -> None:
    """Regression test for the append-vs-clear race: an append() parked
    mid-embed (its turn_index already allocated, but its write not yet
    committed) when a concurrent clear() runs to completion must not
    resurrect the cleared session once its embed call finally returns.
    See docs/adr/0031-code-review-fixes.md."""
    release_event = threading.Event()
    memory = VectorMemory(
        embedding_provider=SlowEmbeddingProvider(release_event), vector_store=InMemoryVectorStore()
    )
    append_done = threading.Event()

    def do_append() -> None:
        memory.append("s1", Message.user("racy message"))
        append_done.set()

    t = threading.Thread(target=do_append)
    t.start()
    try:
        # Give the append time to allocate its turn_index and start
        # blocking inside embed() before clear() runs.
        assert not append_done.wait(timeout=0.2)

        memory.clear("s1")
    finally:
        release_event.set()
        t.join(timeout=5.0)

    assert memory.load("s1") == []
    assert dict(memory.vector_store._chunks) == {}  # noqa: SLF001


class FailingEmbeddingProvider(BaseEmbeddingProvider):
    """A fake embedding provider whose embed() always raises, to exercise
    VectorMemory's failure-ordering guarantee."""

    @property
    def name(self) -> str:
        return "failing"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding failure")


def test_vector_memory_append_does_not_persist_history_on_embedding_failure() -> None:
    memory = VectorMemory(
        embedding_provider=FailingEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    with pytest.raises(RuntimeError):
        memory.append("s1", Message.user("this should not be persisted"))

    # Embedding failed *before* history was touched -- nothing partially saved.
    assert memory.load("s1") == []


@pytest.mark.asyncio
async def test_vector_memory_aappend_does_not_persist_history_on_embedding_failure() -> None:
    memory = VectorMemory(
        embedding_provider=FailingEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    with pytest.raises(RuntimeError):
        await memory.aappend("s1", Message.user("this should not be persisted"))

    assert await memory.aload("s1") == []


# ---------------------------------------------------------------------------
# RedisMemory (SDK faked via sys.modules)
# ---------------------------------------------------------------------------


class _FakeSyncRedisClient:
    def __init__(self, store: dict[str, list[str]]) -> None:
        self._store = store

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "_FakeSyncRedisClient":
        return cls(_FAKE_REDIS_STORE)

    def rpush(self, key: str, value: str) -> None:
        self._store.setdefault(key, []).append(value)

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self._store.get(key, [])
        return list(values) if end == -1 else list(values[start : end + 1])

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def expire(self, key: str, seconds: int) -> None:
        pass


class _FakeAsyncRedisClient:
    def __init__(self, store: dict[str, list[str]]) -> None:
        self._store = store

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "_FakeAsyncRedisClient":
        return cls(_FAKE_REDIS_STORE)

    async def rpush(self, key: str, value: str) -> None:
        self._store.setdefault(key, []).append(value)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self._store.get(key, [])
        return list(values) if end == -1 else list(values[start : end + 1])

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def expire(self, key: str, seconds: int) -> None:
        pass


class _FailingRpushRedisClient(_FakeSyncRedisClient):
    def rpush(self, key: str, value: str) -> None:
        raise RuntimeError("connection refused")


_FAKE_REDIS_STORE: dict[str, list[str]] = {}


@pytest.fixture(autouse=False)
def fake_redis_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    _FAKE_REDIS_STORE.clear()

    fake_redis = types.ModuleType("redis")
    fake_redis.Redis = _FakeSyncRedisClient  # type: ignore[attr-defined]

    fake_redis_asyncio = types.ModuleType("redis.asyncio")
    fake_redis_asyncio.Redis = _FakeAsyncRedisClient  # type: ignore[attr-defined]
    fake_redis.asyncio = fake_redis_asyncio  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake_redis_asyncio)
    return fake_redis


def test_redis_memory_load_append_clear(fake_redis_module: types.ModuleType) -> None:
    from requisite.memory.redis import RedisMemory

    memory = RedisMemory()
    assert memory.load("s1") == []

    memory.append("s1", Message.user("hi"))
    memory.append("s1", Message.assistant("hello!"))
    assert [m.content for m in memory.load("s1")] == ["hi", "hello!"]

    memory.clear("s1")
    assert memory.load("s1") == []


def test_redis_memory_sessions_are_isolated_by_key_prefix(
    fake_redis_module: types.ModuleType,
) -> None:
    from requisite.memory.redis import RedisMemory

    a = RedisMemory(key_prefix="app-a:")
    b = RedisMemory(key_prefix="app-b:")
    a.append("s1", Message.user("for app a"))
    b.append("s1", Message.user("for app b"))
    assert [m.content for m in a.load("s1")] == ["for app a"]
    assert [m.content for m in b.load("s1")] == ["for app b"]


def test_redis_memory_name(fake_redis_module: types.ModuleType) -> None:
    from requisite.memory.redis import RedisMemory

    assert RedisMemory().name == "redis"


@pytest.mark.asyncio
async def test_redis_memory_async_methods_use_native_async_client(
    fake_redis_module: types.ModuleType,
) -> None:
    from requisite.memory.redis import RedisMemory

    memory = RedisMemory()
    await memory.aappend("s1", Message.user("hi"))
    assert [m.content for m in await memory.aload("s1")] == ["hi"]
    await memory.aclear("s1")
    assert await memory.aload("s1") == []


def test_redis_memory_missing_sdk_raises_configuration_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "redis", None)
    from requisite.memory.redis import RedisMemory

    memory = RedisMemory()
    with pytest.raises(ConfigurationException):
        memory.append("s1", Message.user("hi"))


def test_redis_memory_wraps_backend_errors_in_memory_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FAKE_REDIS_STORE.clear()
    fake_redis = types.ModuleType("redis")
    fake_redis.Redis = _FailingRpushRedisClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis", fake_redis)

    from requisite.memory.redis import RedisMemory

    memory = RedisMemory()
    with pytest.raises(MemoryException):
        memory.append("s1", Message.user("hi"))


# ---------------------------------------------------------------------------
# Agent + memory integration
# ---------------------------------------------------------------------------


class RecordingEchoProvider(BaseProvider):
    """Fake provider that echoes the full conversation length back, so tests
    can verify prior history was actually loaded and passed to the model.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self.last_messages: list[Message] = []

    @property
    def name(self) -> str:
        return "recording_echo"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model=None,
        temperature=None,
        tools=None,
        response_model=None,
        **kwargs,
    ) -> ChatResponse:
        self.last_messages = list(messages)
        last_user = next((m.content for m in reversed(messages) if m.role.value == "user"), "")
        return ChatResponse(
            content=f"reply to '{last_user}' (history length {len(messages)})",
            model=self._model,
            provider=self.name,
        )

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> Iterator[StreamChunk]:  # pragma: no cover
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


@pytest.fixture
def provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("recording_echo", RecordingEchoProvider)
    return registry


@pytest.fixture
def settings() -> Settings:
    return Settings(default_provider="recording_echo", model="fake-model")


def test_agent_requires_session_id_when_memory_given(
    provider_registry: ProviderRegistry, settings: Settings
) -> None:
    with pytest.raises(ConfigurationException, match="session_id"):
        Agent(
            name="A",
            provider="recording_echo",
            settings=settings,
            registry=provider_registry,
            memory=InProcessMemory(),
        )


def test_agent_with_memory_persists_and_recalls_across_calls(
    provider_registry: ProviderRegistry, settings: Settings
) -> None:
    memory = InProcessMemory()
    agent = Agent(
        name="A",
        provider="recording_echo",
        settings=settings,
        registry=provider_registry,
        memory=memory,
        session_id="user-1",
    )

    first = agent.run("My name is Alex.")
    assert "history length 1" in first.content  # just this turn's user message

    second = agent.run("What's my name?")
    # history now includes: turn 1 user + turn 1 assistant + turn 2 user = 3
    assert "history length 3" in second.content

    stored = memory.load("user-1")
    assert [m.content for m in stored] == [
        "My name is Alex.",
        "reply to 'My name is Alex.' (history length 1)",
        "What's my name?",
        "reply to 'What's my name?' (history length 3)",
    ]


def test_agent_with_memory_rejects_message_sequence_prompt(
    provider_registry: ProviderRegistry, settings: Settings
) -> None:
    agent = Agent(
        name="A",
        provider="recording_echo",
        settings=settings,
        registry=provider_registry,
        memory=InProcessMemory(),
        session_id="user-1",
    )
    with pytest.raises(ConfigurationException, match="plain string"):
        agent.run([Message.user("hi")])


@pytest.mark.asyncio
async def test_agent_arun_with_memory_persists_and_recalls(
    provider_registry: ProviderRegistry, settings: Settings
) -> None:
    memory = InProcessMemory()
    agent = Agent(
        name="A",
        provider="recording_echo",
        settings=settings,
        registry=provider_registry,
        memory=memory,
        session_id="user-1",
    )
    await agent.arun("hello")
    await agent.arun("again")
    stored = await memory.aload("user-1")
    assert len(stored) == 4


def test_agent_without_memory_does_not_touch_any_backend(
    provider_registry: ProviderRegistry, settings: Settings
) -> None:
    agent = Agent(
        name="A", provider="recording_echo", settings=settings, registry=provider_registry
    )
    assert agent.memory is None
    result = agent.run("hi")
    assert "history length 1" in result.content
