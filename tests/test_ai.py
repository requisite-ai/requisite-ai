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
from requisite.core.interfaces import ChatResponse, Message, Role, StreamChunk, ToolCall, Usage
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry


class FakeProvider(BaseProvider):
    """A deterministic in-memory provider used purely for testing the facade."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))
        self.last_messages: list[Message] = []
        self.last_temperature: Optional[float] = None
        self.last_tools: Optional[list[Any]] = None
        self.next_stream_tool_calls: Optional[list[ToolCall]] = None

    @property
    def name(self) -> str:
        return "fake"

    def chat(
        self, messages: Sequence[Message], *, model=None, temperature=None, tools=None, **kwargs
    ) -> ChatResponse:
        self.last_messages = list(messages)
        self.last_temperature = temperature
        self.last_tools = list(tools) if tools is not None else None
        return ChatResponse(
            content="fake response",
            model=model or self._model,
            provider=self.name,
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    async def achat(
        self, messages: Sequence[Message], *, model=None, temperature=None, tools=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, model=model, temperature=temperature, tools=tools, **kwargs)

    def stream(
        self, messages: Sequence[Message], *, model=None, temperature=None, tools=None, **kwargs
    ) -> Iterator[StreamChunk]:
        self.last_messages = list(messages)
        self.last_tools = list(tools) if tools is not None else None
        for token in ["fa", "ke", " stream"]:
            yield StreamChunk(delta=token)
        tool_calls = self.next_stream_tool_calls or []
        self.next_stream_tool_calls = None
        yield StreamChunk(delta="", is_final=True, tool_calls=tool_calls)

    async def astream(
        self, messages: Sequence[Message], *, model=None, temperature=None, tools=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self.stream(
            messages, model=model, temperature=temperature, tools=tools, **kwargs
        ):
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


def test_stream_accepts_tools_but_still_yields_only_text(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    from requisite.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"sunny in {city}"

    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    provider.next_stream_tool_calls = [ToolCall(id="call_1", name="get_weather", arguments={})]

    assert list(ai.stream("weather?", tools=[get_weather])) == ["fa", "ke", " stream"]
    assert provider.last_tools is not None
    assert provider.last_tools[0].name == "get_weather"


def test_stream_response_yields_full_chunks_with_tool_calls(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    provider.next_stream_tool_calls = [ToolCall(id="call_1", name="get_weather", arguments={})]

    chunks = list(ai.stream_response("weather?"))
    assert [c.delta for c in chunks] == ["fa", "ke", " stream", ""]
    assert chunks[-1].is_final
    assert chunks[-1].has_tool_calls
    assert chunks[-1].tool_calls[0].name == "get_weather"
    # Intermediate chunks never carry tool calls -- only the final one.
    assert all(not c.has_tool_calls for c in chunks[:-1])


@pytest.mark.asyncio
async def test_astream_response_yields_full_chunks_with_tool_calls(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    provider.next_stream_tool_calls = [ToolCall(id="call_1", name="get_weather", arguments={})]

    chunks = [c async for c in ai.astream_response("weather?")]
    assert chunks[-1].has_tool_calls
    assert chunks[-1].tool_calls[0].name == "get_weather"


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


# ---------------------------------------------------------------------------
# tools= accepts Tool instances, @tool-decorated functions, and plain functions
# ---------------------------------------------------------------------------


def test_chat_response_tools_accepts_tool_decorated_function(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    from requisite.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"sunny in {city}"

    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    ai.chat_response("weather?", tools=[get_weather])

    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_tools is not None
    assert provider.last_tools[0].name == "get_weather"
    assert provider.last_tools[0].to_openai_schema()["function"]["name"] == "get_weather"


def test_chat_response_tools_accepts_plain_function(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"sunny in {city}"

    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    ai.chat_response("weather?", tools=[get_weather])

    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_tools[0].name == "get_weather"  # type: ignore[index]


def test_chat_response_tools_accepts_tool_instance_directly(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    from requisite.tools.base import Tool

    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    built_tool = Tool.from_function(get_weather)
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    ai.chat_response("weather?", tools=[built_tool])

    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_tools == [built_tool]


def test_chat_response_without_tools_passes_none(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    ai.chat_response("hello")
    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_tools is None


@pytest.mark.asyncio
async def test_achat_response_tools_accepts_tool_decorated_function(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    from requisite.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"sunny in {city}"

    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    await ai.achat_response("weather?", tools=[get_weather])

    provider: FakeProvider = ai.provider  # type: ignore[assignment]
    assert provider.last_tools is not None
    assert provider.last_tools[0].name == "get_weather"


# ---------------------------------------------------------------------------
# Rate limiter wiring
# ---------------------------------------------------------------------------


def test_ai_has_no_rate_limiter_by_default(registry_with_fake: ProviderRegistry) -> None:
    # rate_limit_rpm explicitly overridden to None rather than relying on the
    # shared `settings` fixture: a real .env in the repo/cwd may legitimately
    # set RATE_LIMIT_RPM (Settings(_env_file=None) wouldn't help either, since
    # this framework's convention -- see tests/test_settings.py -- is that it
    # only disables the .env file, not real process environment variables).
    settings = Settings(default_provider="fake", model="fake-model", rate_limit_rpm=None)
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    assert ai.rate_limiter is None


def test_ai_builds_default_rate_limiter_from_settings(
    registry_with_fake: ProviderRegistry,
) -> None:
    settings = Settings(default_provider="fake", model="fake-model", rate_limit_rpm=5)
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    assert ai.rate_limiter is not None


def test_ai_explicit_rate_limiter_overrides_settings(
    registry_with_fake: ProviderRegistry,
) -> None:
    from requisite.core.rate_limiter import RateLimiter

    settings = Settings(default_provider="fake", model="fake-model", rate_limit_rpm=5)
    explicit = RateLimiter(requests_per_minute=99)
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake, rate_limiter=explicit)
    assert ai.rate_limiter is explicit


def test_shared_rate_limiter_serializes_calls_across_two_ai_instances(
    monkeypatch: pytest.MonkeyPatch, registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    """Two AI instances sharing one RateLimiter draw from one combined budget."""
    from requisite.core import rate_limiter as rate_limiter_module
    from requisite.core.rate_limiter import RateLimiter

    monkeypatch.setattr(rate_limiter_module, "_WINDOW_SECONDS", 0.2)
    shared = RateLimiter(requests_per_minute=1)

    first = AI(provider="fake", settings=settings, registry=registry_with_fake, rate_limiter=shared)
    second = AI(
        provider="fake", settings=settings, registry=registry_with_fake, rate_limiter=shared
    )

    import time

    first.chat("hello")  # claims the one slot in the shared budget
    start = time.monotonic()
    second.chat("hello")  # must wait for the shared slot to free up
    elapsed = time.monotonic() - start

    assert elapsed >= 0.1


# ---------------------------------------------------------------------------
# Cost limiter wiring
# ---------------------------------------------------------------------------


def test_ai_has_no_cost_limiter_by_default(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake)
    assert ai.cost_limiter is None


def test_ai_cost_limiter_records_real_usage_after_a_successful_call(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    from requisite.core.cost_limiter import CostLimiter, cost_per_token

    # FakeProvider.chat always returns Usage(prompt_tokens=1, completion_tokens=2, ...).
    limiter = CostLimiter(
        budget_usd=100.0,
        cost_fn=cost_per_token(prompt_rate_per_1k=1000.0, completion_rate_per_1k=1000.0),
    )
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake, cost_limiter=limiter)

    ai.chat("hello")

    # 1 prompt token * $1/token + 2 completion tokens * $1/token = $3.
    assert limiter.spent_usd == pytest.approx(3.0)


def test_ai_cost_limiter_blocks_the_call_once_budget_exhausted(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    from requisite.core.cost_limiter import CostLimiter, cost_per_token
    from requisite.core.exceptions import CostLimitException

    limiter = CostLimiter(
        budget_usd=1.0,
        cost_fn=cost_per_token(prompt_rate_per_1k=1000.0, completion_rate_per_1k=1000.0),
    )
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake, cost_limiter=limiter)
    fake_provider = ai.provider
    assert isinstance(fake_provider, FakeProvider)

    ai.chat("hello")  # spends $3, already over the $1 budget
    calls_before = len(fake_provider.last_messages)

    with pytest.raises(CostLimitException, match="budget"):
        ai.chat("hello again")
    # The provider must never have been reached for the blocked call.
    assert fake_provider.last_messages == fake_provider.last_messages[:calls_before]


@pytest.mark.asyncio
async def test_ai_cost_limiter_works_on_the_async_path(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    from requisite.core.cost_limiter import CostLimiter, cost_per_token
    from requisite.core.exceptions import CostLimitException

    limiter = CostLimiter(
        budget_usd=1.0,
        cost_fn=cost_per_token(prompt_rate_per_1k=1000.0, completion_rate_per_1k=1000.0),
    )
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake, cost_limiter=limiter)

    await ai.achat("hello")  # spends $3
    with pytest.raises(CostLimitException):
        await ai.achat("hello again")


def test_ai_cost_limiter_has_no_effect_on_streaming(
    registry_with_fake: ProviderRegistry, settings: Settings
) -> None:
    from requisite.core.cost_limiter import CostLimiter, cost_per_token

    # Budget already fully spent -- if stream() checked cost_limiter, this would raise.
    limiter = CostLimiter(
        budget_usd=0.0001,
        cost_fn=cost_per_token(prompt_rate_per_1k=1000.0, completion_rate_per_1k=1000.0),
    )
    ai = AI(provider="fake", settings=settings, registry=registry_with_fake, cost_limiter=limiter)

    result = "".join(ai.stream("hello"))

    assert result == "fake stream"
    assert limiter.spent_usd == 0.0  # streaming never records either
