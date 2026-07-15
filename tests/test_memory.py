"""Unit tests for the memory package and Agent's memory/session_id integration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import pytest

from requisite.agents.agent import Agent
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse, Message, StreamChunk
from requisite.memory.factory import MemoryRegistry, default_registry
from requisite.memory.in_process import InProcessMemory
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
