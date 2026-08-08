"""Unit tests for :class:`requisite.agents.agent.Agent`.

Uses a scripted fake provider that returns a tool call on its first
invocation and a final answer on its second, to exercise the full
tool-calling loop without any network access.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import pytest

from requisite.agents.agent import Agent, AgentResult
from requisite.agents.registry import AgentRegistry
from requisite.config.settings import Settings
from requisite.core.exceptions import AgentException
from requisite.core.interfaces import ChatResponse, Message, StreamChunk, ToolCall, Usage
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry
from requisite.tools.decorator import tool


class ScriptedToolCallingProvider(BaseProvider):
    """A fake provider that requests one tool call, then returns a final answer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))
        self._call_count = 0

    @property
    def name(self) -> str:
        return "scripted"

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
        self._call_count += 1
        if self._call_count == 1:
            return ChatResponse(
                content="",
                model=self._model,
                provider=self.name,
                tool_calls=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})],
            )
        return ChatResponse(
            content="It's sunny in Paris.",
            model=self._model,
            provider=self.name,
            usage=Usage(total_tokens=10),
        )

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
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> Iterator[StreamChunk]:  # pragma: no cover
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


class AlwaysToolCallProvider(BaseProvider):
    """A fake provider that always requests a tool call -- used to test max_iterations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")

    @property
    def name(self) -> str:
        return "loopy"

    def chat(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> ChatResponse:
        return ChatResponse(
            content="",
            model=self._model,
            provider=self.name,
            tool_calls=[ToolCall(id="call_x", name="noop", arguments={})],
        )

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ):  # pragma: no cover
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


@pytest.fixture
def registry_with_scripted() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("scripted", ScriptedToolCallingProvider)
    registry.register("loopy", AlwaysToolCallProvider)
    return registry


@pytest.fixture
def settings() -> Settings:
    return Settings(default_provider="scripted", model="fake-model")


def make_weather_tool():
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"sunny in {city}"

    return get_weather


def test_agent_executes_tool_and_returns_final_answer(
    registry_with_scripted: ProviderRegistry, settings: Settings
) -> None:
    agent = Agent(
        name="Weather Agent",
        provider="scripted",
        tools=[make_weather_tool()],
        settings=settings,
        registry=registry_with_scripted,
    )
    result = agent.run("What's the weather in Paris?")
    assert isinstance(result, AgentResult)
    assert result.content == "It's sunny in Paris."
    assert result.tool_calls_executed == ["get_weather"]
    assert result.iterations == 2
    assert result.agent_name == "Weather Agent"


@pytest.mark.asyncio
async def test_agent_arun_executes_tool_and_returns_final_answer(
    registry_with_scripted: ProviderRegistry, settings: Settings
) -> None:
    agent = Agent(
        name="Weather Agent",
        provider="scripted",
        tools=[make_weather_tool()],
        settings=settings,
        registry=registry_with_scripted,
    )
    result = await agent.arun("What's the weather in Paris?")
    assert result.content == "It's sunny in Paris."
    assert result.tool_calls_executed == ["get_weather"]


def test_agent_raises_when_max_iterations_exceeded(
    registry_with_scripted: ProviderRegistry,
) -> None:
    settings = Settings(default_provider="loopy", model="fake-model")

    @tool
    def noop() -> str:
        return "noop"

    agent = Agent(
        name="Loopy",
        provider="loopy",
        tools=[noop],
        settings=settings,
        registry=registry_with_scripted,
        max_iterations=2,
    )
    with pytest.raises(AgentException):
        agent.run("do something")


def test_agent_forwards_rate_limiter_to_its_internal_ai(
    registry_with_scripted: ProviderRegistry, settings: Settings
) -> None:
    from requisite.core.rate_limiter import RateLimiter

    shared_limiter = RateLimiter(requests_per_minute=99)
    agent = Agent(
        name="Weather Agent",
        provider="scripted",
        settings=settings,
        registry=registry_with_scripted,
        rate_limiter=shared_limiter,
    )
    assert agent.ai.rate_limiter is shared_limiter


def test_agent_registry_register_and_get(
    registry_with_scripted: ProviderRegistry, settings: Settings
) -> None:
    agent = Agent(
        name="Weather Agent",
        provider="scripted",
        settings=settings,
        registry=registry_with_scripted,
    )
    agent_registry = AgentRegistry()
    agent_registry.register(agent)
    assert agent_registry.get("Weather Agent") is agent
    assert agent_registry.list_names() == ["Weather Agent"]


def test_agent_registry_missing_raises() -> None:
    agent_registry = AgentRegistry()
    from requisite.core.exceptions import AgentException as AE

    with pytest.raises(AE):
        agent_registry.get("nope")
