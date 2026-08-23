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


class MultiToolCallProvider(BaseProvider):
    """A fake provider that requests three tool calls in one turn, then answers."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))
        self._call_count = 0

    @property
    def name(self) -> str:
        return "multi"

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
                tool_calls=[
                    ToolCall(id="call_1", name="slow_echo", arguments={"value": "a"}),
                    ToolCall(id="call_2", name="slow_echo", arguments={"value": "b"}),
                    ToolCall(id="call_3", name="slow_echo", arguments={"value": "c"}),
                ],
            )
        return ChatResponse(
            content="done",
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
    ):  # pragma: no cover
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_agent_arun_executes_independent_tool_calls_concurrently() -> None:
    import asyncio
    import time

    registry = ProviderRegistry()
    registry.register("multi", MultiToolCallProvider)
    settings = Settings(default_provider="multi", model="fake-model")

    @tool
    async def slow_echo(value: str) -> str:
        """Sleep briefly, then echo the given value."""
        await asyncio.sleep(0.2)
        return f"echo:{value}"

    agent = Agent(
        name="Multi Tool Agent",
        provider="multi",
        tools=[slow_echo],
        settings=settings,
        registry=registry,
    )

    start = time.monotonic()
    result = await agent.arun("run all three")
    elapsed = time.monotonic() - start

    # Three tools each sleeping 0.2s would take >=0.6s run sequentially;
    # run concurrently they should complete in well under that.
    assert elapsed < 0.4
    assert result.content == "done"
    assert result.tool_calls_executed == ["slow_echo", "slow_echo", "slow_echo"]


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


class UnknownThenAnswerProvider(BaseProvider):
    """Requests an unregistered tool name, then -- on the next turn, having
    seen the resulting error tool_result -- returns a final answer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self._call_count = 0

    @property
    def name(self) -> str:
        return "unknown-then-answer"

    def chat(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> ChatResponse:
        self._call_count += 1
        if self._call_count == 1:
            return ChatResponse(
                content="",
                model=self._model,
                provider=self.name,
                tool_calls=[ToolCall(id="call_1", name="totally_made_up_tool", arguments={})],
            )
        return ChatResponse(content="recovered", model=self._model, provider=self.name)

    async def achat(self, messages, **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


class AlwaysUnknownToolProvider(BaseProvider):
    """Always requests the same unregistered tool name -- for testing that
    a persistently-hallucinating model still bounds cleanly at
    max_iterations, rather than crashing on the first bad call."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self.call_count = 0

    @property
    def name(self) -> str:
        return "always-unknown"

    def chat(self, messages, **kwargs) -> ChatResponse:
        self.call_count += 1
        return ChatResponse(
            content="",
            model=self._model,
            provider=self.name,
            tool_calls=[ToolCall(id="call_x", name="totally_made_up_tool", arguments={})],
        )

    async def achat(self, messages, **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


def test_agent_run_recovers_from_unknown_tool_call() -> None:
    """A single hallucinated tool call must not abort the whole run --
    the model sees the failure as a tool_result and can recover on a
    later iteration, using max_iterations as an actual retry budget."""
    registry = ProviderRegistry()
    registry.register("unknown-then-answer", UnknownThenAnswerProvider)
    settings = Settings(default_provider="unknown-then-answer", model="fake-model")
    agent = Agent(
        name="Recovering",
        provider="unknown-then-answer",
        tools=[make_weather_tool()],
        settings=settings,
        registry=registry,
        max_iterations=5,
    )

    result = agent.run("do something")

    assert result.content == "recovered"
    assert result.tool_calls_executed == ["totally_made_up_tool"]
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_agent_arun_recovers_from_unknown_tool_call() -> None:
    registry = ProviderRegistry()
    registry.register("unknown-then-answer", UnknownThenAnswerProvider)
    settings = Settings(default_provider="unknown-then-answer", model="fake-model")
    agent = Agent(
        name="Recovering",
        provider="unknown-then-answer",
        tools=[make_weather_tool()],
        settings=settings,
        registry=registry,
        max_iterations=5,
    )

    result = await agent.arun("do something")

    assert result.content == "recovered"
    assert result.tool_calls_executed == ["totally_made_up_tool"]


def test_agent_run_bounds_persistent_unknown_tool_at_max_iterations() -> None:
    """A model that never self-corrects still bounds cleanly at
    max_iterations, using the full retry budget rather than crashing on
    the very first hallucinated call."""
    registry = ProviderRegistry()
    provider_instance = AlwaysUnknownToolProvider()
    registry.register("always-unknown", lambda **kwargs: provider_instance)
    settings = Settings(default_provider="always-unknown", model="fake-model")
    agent = Agent(
        name="NeverLearns",
        provider="always-unknown",
        tools=[make_weather_tool()],
        settings=settings,
        registry=registry,
        max_iterations=3,
    )

    with pytest.raises(AgentException, match="max_iterations"):
        agent.run("do something")

    assert provider_instance.call_count == 3


class SpyingUnknownThenAnswerProvider(BaseProvider):
    """Same script as UnknownThenAnswerProvider, but records every
    messages list it's called with, so the test can inspect the actual
    tool_result content fed back after the failed call."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self._call_count = 0
        self.seen_messages: list[list[Message]] = []

    @property
    def name(self) -> str:
        return "spying-unknown-then-answer"

    def chat(self, messages, **kwargs) -> ChatResponse:
        self._call_count += 1
        self.seen_messages.append(list(messages))
        if self._call_count == 1:
            return ChatResponse(
                content="",
                model=self._model,
                provider=self.name,
                tool_calls=[ToolCall(id="call_1", name="totally_made_up_tool", arguments={})],
            )
        return ChatResponse(content="recovered", model=self._model, provider=self.name)

    async def achat(self, messages, **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


def test_agent_run_tool_result_includes_error_message() -> None:
    """The recovery tool_result content should actually surface the
    error, not swallow it into an opaque generic string."""
    registry = ProviderRegistry()
    provider_instance = SpyingUnknownThenAnswerProvider()
    registry.register("spying-unknown-then-answer", lambda **kwargs: provider_instance)
    settings = Settings(default_provider="spying-unknown-then-answer", model="fake-model")
    agent = Agent(
        name="Recovering",
        provider="spying-unknown-then-answer",
        tools=[make_weather_tool()],
        settings=settings,
        registry=registry,
        max_iterations=5,
    )

    result = agent.run("do something")

    assert result.content == "recovered"
    second_call_messages = provider_instance.seen_messages[1]
    tool_result = second_call_messages[-1]
    assert (
        "totally_made_up_tool" in tool_result.content or "No tool registered" in tool_result.content
    )


class ConcurrentPartialFailureProvider(BaseProvider):
    """Requests two independent tool calls in one turn -- one that will
    fail, one that will succeed -- then a final answer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self._call_count = 0

    @property
    def name(self) -> str:
        return "concurrent-partial-failure"

    def chat(self, messages, **kwargs) -> ChatResponse:
        self._call_count += 1
        if self._call_count == 1:
            return ChatResponse(
                content="",
                model=self._model,
                provider=self.name,
                tool_calls=[
                    ToolCall(id="call_bad", name="totally_made_up_tool", arguments={}),
                    ToolCall(id="call_good", name="get_weather", arguments={"city": "Paris"}),
                ],
            )
        return ChatResponse(content="done", model=self._model, provider=self.name)

    async def achat(self, messages, **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_agent_arun_one_failing_concurrent_tool_call_does_not_abort_the_others() -> None:
    """Regression test for the asyncio.gather orphaned-task leak: when one
    of several concurrently-executed tool calls fails, arun() must not
    raise and abandon the others -- every call gets a tool_result, and
    the run can still reach a final answer."""
    registry = ProviderRegistry()
    registry.register("concurrent-partial-failure", ConcurrentPartialFailureProvider)
    settings = Settings(default_provider="concurrent-partial-failure", model="fake-model")
    agent = Agent(
        name="PartialFailure",
        provider="concurrent-partial-failure",
        tools=[make_weather_tool()],
        settings=settings,
        registry=registry,
        max_iterations=5,
    )

    result = await agent.arun("do something")

    assert result.content == "done"
    assert set(result.tool_calls_executed) == {"totally_made_up_tool", "get_weather"}


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


# ---------------------------------------------------------------------------
# as_tool -- exposing an agent as a single Tool (mirrors BaseSkill.as_tool)
# ---------------------------------------------------------------------------


class FixedAnswerProvider(BaseProvider):
    """A fake provider that always returns one fixed final answer, no tool calls."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))

    @property
    def name(self) -> str:
        return "fixed"

    def chat(self, messages, *, model=None, temperature=None, tools=None, **kwargs) -> ChatResponse:
        last = messages[-1].content if messages else ""
        return ChatResponse(
            content=f"answer to: {last}",
            model=self._model,
            provider=self.name,
            usage=Usage(total_tokens=1),
        )

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, model=model, temperature=temperature, tools=tools, **kwargs)

    def stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


def test_agent_as_tool_shape() -> None:
    registry = ProviderRegistry()
    registry.register("fixed", FixedAnswerProvider)
    agent = Agent(name="assistant", provider="fixed", registry=registry)

    tool = agent.as_tool()
    assert tool.name == "assistant"
    assert tool.parameters_schema["required"] == ["prompt"]
    assert "prompt" in tool.parameters_schema["properties"]


def test_agent_as_tool_execute_runs_the_agent() -> None:
    registry = ProviderRegistry()
    registry.register("fixed", FixedAnswerProvider)
    agent = Agent(name="assistant", provider="fixed", registry=registry)

    tool = agent.as_tool()
    assert tool.execute(prompt="hello") == "answer to: hello"


@pytest.mark.asyncio
async def test_agent_as_tool_aexecute_runs_the_agent() -> None:
    registry = ProviderRegistry()
    registry.register("fixed", FixedAnswerProvider)
    agent = Agent(name="assistant", provider="fixed", registry=registry)

    tool = agent.as_tool()
    assert await tool.aexecute(prompt="hi") == "answer to: hi"
