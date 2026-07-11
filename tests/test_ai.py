"""Unit tests for :class:`requisite.ai.AI` using a fully in-memory fake provider.

These tests never touch the network or any provider SDK -- they exercise
the facade's message-building, configuration-resolution, and dispatch
logic in isolation, per the "mock providers / mock LLM responses"
testing standard.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

import pytest

from requisite.ai import AI
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse, Message, Role, StreamChunk, Usage
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry


class FakeProvider(BaseProvider):
    """A deterministic in-memory provider used purely for testing the facade."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))
        self.last_messages: list[Message] = []
        self.last_temperature: Optional[float] = None

    @property
    def name(self) -> str:
        return "fake"

    def chat(
        self, messages: Sequence[Message], *, model=None, temperature=None, **kwargs
    ) -> ChatResponse:
        self.last_messages = list(messages)
        self.last_temperature = temperature
        return ChatResponse(
            content="fake response",
            model=model or self._model,
            provider=self.name,
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    async def achat(
        self, messages: Sequence[Message], *, model=None, temperature=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, model=model, temperature=temperature, **kwargs)

    def stream(
        self, messages: Sequence[Message], *, model=None, temperature=None, **kwargs
    ) -> Iterator[StreamChunk]:
        self.last_messages = list(messages)
        for token in ["fa", "ke", " stream"]:
            yield StreamChunk(delta=token)
        yield StreamChunk(delta="", is_final=True)

    async def astream(
        self, messages: Sequence[Message], *, model=None, temperature=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self.stream(messages, model=model, temperature=temperature, **kwargs):
            yield chunk


@pytest.fixture
def registry_with_fake() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("fake", FakeProvider)
    return registry


@pytest.fixture
def settings() -> Settings:
    return Settings(default_provider="fake", model="fake-model", temperature=0.5)


def test_chat_returns_text(registry_with_fake: ProviderRegistry, settings: Settings) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    assert ai.chat("hello") == "fake response"


def test_chat_response_carries_usage(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    response = ai.chat_response("hello")
    assert response.usage.total_tokens == 3
    assert response.provider == "fake"


def test_system_prompt_is_prepended(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(
        provider="fake",
        settings=settings,
        registry=registry_with_fake,
        system_prompt="You are terse.",
    )
    ai.chat("hello")
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_messages[0].role == Role.SYSTEM
    assert provider.last_messages[0].content == "You are terse."
    assert provider.last_messages[1].content == "hello"


def test_per_call_system_prompt_overrides_instance_default(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(
        provider="fake",
        settings=settings,
        registry=registry_with_fake,
        system_prompt="default",
    )
    ai.chat("hello", system_prompt="override")
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_messages[0].content == "override"


def test_default_temperature_comes_from_settings(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    ai.chat("hello")
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_temperature == 0.5


def test_per_call_temperature_overrides_default(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    ai.chat("hello", temperature=0.9)
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_temperature == 0.9


def test_stream_yields_text_chunks(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    assert list(ai.stream("hello")) == ["fa", "ke", " stream"]


@pytest.mark.asyncio
async def test_astream_yields_text_chunks(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    chunks = [c async for c in ai.astream("hello")]
    assert chunks == ["fa", "ke", " stream"]


@pytest.mark.asyncio
async def test_achat_returns_text(registry_with_fake: ProviderRegistry, settings: Settings) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    assert await ai.achat("hello") == "fake response"


def test_conversation_history_is_passed_through(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    history = [Message.user("hi"), Message.assistant("hello!"), Message.user("how are you?")]
    ai.chat(history)
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_messages == history


def test_passing_provider_instance_directly_bypasses_registry(settings: Settings) -> None:
    provider = FakeProvider()
    ai = AI(provider=provider, settings=settings, registry=ProviderRegistry())
    assert ai.provider is provider


def test_unknown_provider_name_raises_configuration_exception(settings: Settings) -> None:
    with pytest.raises(ConfigurationException):
        AI(provider="does-not-exist", settings=settings, registry=ProviderRegistry())


def test_missing_default_provider_raises(registry_with_fake: ProviderRegistry) -> None:
    empty_settings = Settings(default_provider="")
    with pytest.raises(ConfigurationException):
        AI(settings=empty_settings, registry=registry_with_fake)
