"""Unit tests for the memory package and Agent's memory/session_id integration."""

from __future__ import annotations

import sys
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
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry

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
