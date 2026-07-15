"""Unit tests for requisite.memory.policies (conversation management)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import pytest

from requisite.agents.agent import Agent
from requisite.ai import AI
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse, Message, Role, StreamChunk
from requisite.memory.policies import MessageCountPolicy, SummarizingPolicy
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry

# ---------------------------------------------------------------------------
# MessageCountPolicy
# ---------------------------------------------------------------------------


def test_message_count_policy_noop_when_under_limit() -> None:
    policy = MessageCountPolicy(max_messages=10)
    history = [Message.user("1"), Message.assistant("2")]
    assert policy.apply(history) == history


def test_message_count_policy_trims_to_most_recent() -> None:
    policy = MessageCountPolicy(max_messages=2, keep_system=False)
    history = [Message.user("1"), Message.assistant("2"), Message.user("3")]
    trimmed = policy.apply(history)
    assert [m.content for m in trimmed] == ["2", "3"]


def test_message_count_policy_always_keeps_system_messages() -> None:
    policy = MessageCountPolicy(max_messages=2, keep_system=True)
    history = [
        Message.system("be terse"),
        Message.user("1"),
        Message.assistant("2"),
        Message.user("3"),
    ]
    trimmed = policy.apply(history)
    assert trimmed[0].role == Role.SYSTEM
    assert [m.content for m in trimmed] == ["be terse", "3"]


def test_message_count_policy_rejects_invalid_max_messages() -> None:
    with pytest.raises(ConfigurationException):
        MessageCountPolicy(max_messages=0)


@pytest.mark.asyncio
async def test_message_count_policy_aapply() -> None:
    policy = MessageCountPolicy(max_messages=1, keep_system=False)
    history = [Message.user("1"), Message.user("2")]
    trimmed = await policy.aapply(history)
    assert [m.content for m in trimmed] == ["2"]


# ---------------------------------------------------------------------------
# SummarizingPolicy
# ---------------------------------------------------------------------------


class StubSummarizingProvider(BaseProvider):
    """Fake provider that returns a fixed summary string, recording what it was asked to summarize."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self.last_prompt: str = ""

    @property
    def name(self) -> str:
        return "stub_summarizer"

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
        self.last_prompt = messages[-1].content
        return ChatResponse(content="SUMMARY", model=self._model, provider=self.name)

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


def make_ai() -> tuple[AI, StubSummarizingProvider]:
    registry = ProviderRegistry()
    registry.register("stub_summarizer", StubSummarizingProvider)
    settings = Settings(default_provider="stub_summarizer", model="fake-model")
    provider = StubSummarizingProvider()
    ai = AI(provider=provider, settings=settings, registry=registry)
    return ai, provider


def test_summarizing_policy_noop_when_under_limit() -> None:
    ai, _ = make_ai()
    policy = SummarizingPolicy(ai, max_messages=10, keep_recent=2)
    history = [Message.user("1"), Message.assistant("2")]
    assert policy.apply(history) == history


def test_summarizing_policy_collapses_old_messages() -> None:
    ai, provider = make_ai()
    policy = SummarizingPolicy(ai, max_messages=4, keep_recent=2)
    history = [
        Message.user("1"),
        Message.assistant("2"),
        Message.user("3"),
        Message.assistant("4"),
        Message.user("5"),
    ]
    result = policy.apply(history)
    # summary message + the last 2 kept verbatim
    assert len(result) == 3
    assert result[0].role == Role.SYSTEM
    assert "SUMMARY" in result[0].content
    assert [m.content for m in result[1:]] == ["4", "5"]
    # the summarization call was given the messages that got collapsed
    assert "1" in provider.last_prompt and "3" in provider.last_prompt


def test_summarizing_policy_rejects_keep_recent_gte_max_messages() -> None:
    ai, _ = make_ai()
    with pytest.raises(ConfigurationException):
        SummarizingPolicy(ai, max_messages=4, keep_recent=4)


def test_summarizing_policy_rejects_prompt_without_placeholder() -> None:
    ai, _ = make_ai()
    with pytest.raises(ConfigurationException):
        SummarizingPolicy(
            ai, max_messages=4, keep_recent=1, summarization_prompt="no placeholder here"
        )


@pytest.mark.asyncio
async def test_summarizing_policy_aapply() -> None:
    ai, _ = make_ai()
    policy = SummarizingPolicy(ai, max_messages=2, keep_recent=1)
    history = [Message.user("1"), Message.user("2"), Message.user("3")]
    result = await policy.aapply(history)
    assert len(result) == 2
    assert "SUMMARY" in result[0].content
    assert result[1].content == "3"


# ---------------------------------------------------------------------------
# Agent + conversation_policy integration
# ---------------------------------------------------------------------------


class RecordingEchoProvider(BaseProvider):
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
        return ChatResponse(
            content=f"seen {len(messages)} messages", model=self._model, provider=self.name
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


def test_agent_applies_conversation_policy_before_calling_provider() -> None:
    provider_registry = ProviderRegistry()
    provider_registry.register("recording_echo", RecordingEchoProvider)
    settings = Settings(default_provider="recording_echo", model="fake-model")

    agent = Agent(
        name="A",
        provider="recording_echo",
        settings=settings,
        registry=provider_registry,
        conversation_policy=MessageCountPolicy(max_messages=1, keep_system=False),
    )
    history = [Message.user("old 1"), Message.assistant("old 2")]
    result = agent.run(history)
    assert result.content == "seen 1 messages"


@pytest.mark.asyncio
async def test_agent_arun_applies_conversation_policy() -> None:
    provider_registry = ProviderRegistry()
    provider_registry.register("recording_echo", RecordingEchoProvider)
    settings = Settings(default_provider="recording_echo", model="fake-model")

    agent = Agent(
        name="A",
        provider="recording_echo",
        settings=settings,
        registry=provider_registry,
        conversation_policy=MessageCountPolicy(max_messages=1, keep_system=False),
    )
    history = [Message.user("old 1"), Message.assistant("old 2")]
    result = await agent.arun(history)
    assert result.content == "seen 1 messages"


def test_agent_without_conversation_policy_leaves_history_untouched() -> None:
    provider_registry = ProviderRegistry()
    provider_registry.register("recording_echo", RecordingEchoProvider)
    settings = Settings(default_provider="recording_echo", model="fake-model")

    agent = Agent(
        name="A", provider="recording_echo", settings=settings, registry=provider_registry
    )
    assert agent.conversation_policy is None
    history = [Message.user("old 1"), Message.assistant("old 2")]
    result = agent.run(history)
    assert result.content == "seen 2 messages"
