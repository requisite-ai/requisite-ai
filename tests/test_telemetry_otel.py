"""Unit tests for :mod:`requisite.telemetry.otel` and the ``AI``/``Agent``
instrumentation it powers.

Split into two groups:

- No-op fallback behavior (``opentelemetry-api`` "not installed",
  simulated via monkeypatching ``_OTEL_AVAILABLE``) -- never raises.
- Real span/metric emission, verified against actual OpenTelemetry SDK
  in-memory exporters/readers (``pytest.importorskip`` so the suite
  still passes for anyone running tests without the ``dev`` extra's
  ``opentelemetry-sdk``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import pytest

from requisite.telemetry import otel as otel_module

# ---------------------------------------------------------------------------
# No-op fallback (opentelemetry-api "not installed")
# ---------------------------------------------------------------------------


def test_get_tracer_falls_back_to_noop_without_opentelemetry(monkeypatch: Any) -> None:
    monkeypatch.setattr(otel_module, "_OTEL_AVAILABLE", False)
    tracer = otel_module.get_tracer("x")
    with tracer.start_as_current_span("s", attributes={"a": 1}) as span:
        span.set_attribute("b", 2)
        span.record_exception(ValueError("boom"))
    # Reaching here without raising is the assertion.


def test_get_meter_falls_back_to_noop_without_opentelemetry(monkeypatch: Any) -> None:
    monkeypatch.setattr(otel_module, "_OTEL_AVAILABLE", False)
    meter = otel_module.get_meter("x")
    meter.create_counter("c").add(1, {"a": "b"})
    meter.create_histogram("h").record(1.5, {"a": "b"})
    # Reaching here without raising is the assertion.


# ---------------------------------------------------------------------------
# Real span/metric emission
# ---------------------------------------------------------------------------

pytest.importorskip("opentelemetry")

from opentelemetry import metrics as otel_metrics  # noqa: E402
from opentelemetry import trace as otel_trace  # noqa: E402
from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from requisite.agents.agent import Agent  # noqa: E402
from requisite.ai import AI  # noqa: E402
from requisite.config.settings import Settings  # noqa: E402
from requisite.core.exceptions import ProviderException  # noqa: E402
from requisite.core.interfaces import ChatResponse, Message, StreamChunk, ToolCall, Usage  # noqa: E402
from requisite.providers.base import BaseProvider  # noqa: E402
from requisite.providers.factory import ProviderRegistry  # noqa: E402
from requisite.tools.decorator import tool  # noqa: E402


class FakeProvider(BaseProvider):
    """A deterministic in-memory provider used purely for exercising instrumentation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))

    @property
    def name(self) -> str:
        return "fake"

    def chat(
        self, messages: Sequence[Message], *, model=None, temperature=None, tools=None, **kwargs
    ) -> ChatResponse:
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
        for token in ["fa", "ke"]:
            yield StreamChunk(delta=token)
        yield StreamChunk(delta="", is_final=True)

    async def astream(
        self, messages: Sequence[Message], *, model=None, temperature=None, tools=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self.stream(messages, model=model, temperature=temperature, tools=tools):
            yield chunk


class FailingProvider(BaseProvider):
    """A fake provider whose calls always raise, to exercise the error path."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")

    @property
    def name(self) -> str:
        return "failing"

    def chat(self, messages, *, model=None, temperature=None, tools=None, **kwargs) -> ChatResponse:
        raise ProviderException("simulated provider failure")

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ) -> ChatResponse:
        raise ProviderException("simulated provider failure")

    def stream(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ) -> Iterator[StreamChunk]:  # pragma: no cover - unused
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:  # pragma: no cover - unused
        raise NotImplementedError
        yield  # pragma: no cover


class ScriptedToolCallingProvider(BaseProvider):
    """A fake provider that requests one tool call, then returns a final answer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))
        self._call_count = 0

    @property
    def name(self) -> str:
        return "scripted"

    def chat(self, messages, *, model=None, temperature=None, tools=None, **kwargs) -> ChatResponse:
        self._call_count += 1
        if self._call_count == 1:
            return ChatResponse(
                content="",
                model=self._model,
                provider=self.name,
                tool_calls=[ToolCall(id="call_1", name="echo", arguments={"value": "hi"})],
            )
        return ChatResponse(content="done", model=self._model, provider=self.name)

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, model=model, temperature=temperature, tools=tools, **kwargs)

    def stream(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ) -> Iterator[StreamChunk]:  # pragma: no cover - unused
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:  # pragma: no cover - unused
        raise NotImplementedError
        yield  # pragma: no cover


def make_ai(provider_cls: type[BaseProvider], name: str = "fake") -> AI:
    registry = ProviderRegistry()
    registry.register(name, provider_cls)
    settings = Settings(default_provider=name, model="fake-model")
    return AI(settings=settings, registry=registry)


def make_agent(provider_cls: type[BaseProvider], name: str = "scripted", **kwargs: Any) -> Agent:
    registry = ProviderRegistry()
    registry.register(name, provider_cls)
    settings = Settings(default_provider=name, model="fake-model")
    return Agent(name="Test Agent", provider=name, settings=settings, registry=registry, **kwargs)


@pytest.fixture(scope="session")
def span_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Only the first call to set_tracer_provider in a process takes effect --
    # this fixture is session-scoped specifically so it only runs once.
    otel_trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture(scope="session")
def metric_reader() -> InMemoryMetricReader:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel_metrics.set_meter_provider(provider)
    return reader


def _spans_by_name(exporter: InMemorySpanExporter, name: str) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _sum_counter(reader: InMemoryMetricReader, metric_name: str, **attrs: str) -> float:
    data = reader.get_metrics_data()
    total = 0.0
    if data is None:
        return total
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != metric_name:
                    continue
                for point in metric.data.data_points:
                    if all(point.attributes.get(k) == v for k, v in attrs.items()):
                        total += point.value
    return total


# ---------------------------------------------------------------------------
# AI facade
# ---------------------------------------------------------------------------


def test_ai_chat_response_emits_a_span(span_exporter: InMemorySpanExporter) -> None:
    span_exporter.clear()
    ai = make_ai(FakeProvider)
    ai.chat_response("hi")

    spans = _spans_by_name(span_exporter, "requisite.ai.chat_response")
    assert len(spans) == 1
    assert spans[0].attributes["requisite.provider"] == "fake"
    assert spans[0].attributes["requisite.model"] == "fake-model"
    assert spans[0].status.status_code.name == "UNSET"  # success -- no error status


@pytest.mark.asyncio
async def test_ai_achat_response_emits_a_span(span_exporter: InMemorySpanExporter) -> None:
    span_exporter.clear()
    ai = make_ai(FakeProvider)
    await ai.achat_response("hi")

    spans = _spans_by_name(span_exporter, "requisite.ai.achat_response")
    assert len(spans) == 1
    assert spans[0].attributes["requisite.provider"] == "fake"


def test_ai_chat_response_records_exception_on_provider_error(
    span_exporter: InMemorySpanExporter,
) -> None:
    span_exporter.clear()
    ai = make_ai(FailingProvider, name="failing")
    with pytest.raises(ProviderException):
        ai.chat_response("hi")

    spans = _spans_by_name(span_exporter, "requisite.ai.chat_response")
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"
    assert any(event.name == "exception" for event in spans[0].events)


def test_ai_stream_emits_a_span(span_exporter: InMemorySpanExporter) -> None:
    span_exporter.clear()
    ai = make_ai(FakeProvider)
    list(ai.stream("hi"))

    spans = _spans_by_name(span_exporter, "requisite.ai.stream")
    assert len(spans) == 1


def test_ai_chat_response_increments_request_and_token_metrics(
    metric_reader: InMemoryMetricReader,
) -> None:
    ai = make_ai(FakeProvider)
    before_requests = _sum_counter(
        metric_reader,
        "requisite.ai.requests",
        **{"requisite.provider": "fake", "requisite.status": "success"},
    )
    before_prompt_tokens = _sum_counter(
        metric_reader,
        "requisite.ai.tokens",
        **{"requisite.provider": "fake", "requisite.token_type": "prompt"},
    )

    ai.chat_response("hi")

    after_requests = _sum_counter(
        metric_reader,
        "requisite.ai.requests",
        **{"requisite.provider": "fake", "requisite.status": "success"},
    )
    after_prompt_tokens = _sum_counter(
        metric_reader,
        "requisite.ai.tokens",
        **{"requisite.provider": "fake", "requisite.token_type": "prompt"},
    )

    assert after_requests - before_requests == 1
    assert after_prompt_tokens - before_prompt_tokens == 1  # FakeProvider reports prompt_tokens=1


def test_ai_chat_response_error_increments_error_status_metric(
    metric_reader: InMemoryMetricReader,
) -> None:
    ai = make_ai(FailingProvider, name="failing")
    before = _sum_counter(
        metric_reader,
        "requisite.ai.requests",
        **{"requisite.provider": "failing", "requisite.status": "error"},
    )

    with pytest.raises(ProviderException):
        ai.chat_response("hi")

    after = _sum_counter(
        metric_reader,
        "requisite.ai.requests",
        **{"requisite.provider": "failing", "requisite.status": "error"},
    )
    assert after - before == 1


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@tool
def echo(value: str) -> str:
    """Echo a value back."""
    return f"echo:{value}"


def test_agent_run_emits_nested_spans(span_exporter: InMemorySpanExporter) -> None:
    span_exporter.clear()
    agent = make_agent(ScriptedToolCallingProvider, tools=[echo])
    agent.run("do something")

    run_spans = _spans_by_name(span_exporter, "requisite.agent.run")
    assert len(run_spans) == 1
    run_span = run_spans[0]
    assert run_span.attributes["requisite.agent_name"] == "Test Agent"

    tool_spans = _spans_by_name(span_exporter, "requisite.agent.tool_call")
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["requisite.tool_name"] == "echo"
    assert tool_spans[0].parent.span_id == run_span.context.span_id

    ai_spans = _spans_by_name(span_exporter, "requisite.ai.chat_response")
    assert len(ai_spans) == 2  # one tool-call round-trip + one final answer
    assert all(s.parent.span_id == run_span.context.span_id for s in ai_spans)


@pytest.mark.asyncio
async def test_agent_arun_emits_nested_spans_for_concurrent_tool_calls(
    span_exporter: InMemorySpanExporter,
) -> None:
    span_exporter.clear()
    agent = make_agent(ScriptedToolCallingProvider, tools=[echo])
    await agent.arun("do something")

    run_spans = _spans_by_name(span_exporter, "requisite.agent.run")
    assert len(run_spans) == 1
    run_span = run_spans[0]

    tool_spans = _spans_by_name(span_exporter, "requisite.agent.tool_call")
    assert len(tool_spans) == 1
    assert tool_spans[0].parent.span_id == run_span.context.span_id


def test_agent_run_increments_run_and_tool_call_metrics(
    metric_reader: InMemoryMetricReader,
) -> None:
    agent = make_agent(ScriptedToolCallingProvider, tools=[echo])
    before_runs = _sum_counter(
        metric_reader,
        "requisite.agent.runs",
        **{"requisite.agent_name": "Test Agent", "requisite.status": "success"},
    )
    before_tool_calls = _sum_counter(
        metric_reader,
        "requisite.agent.tool_calls",
        **{"requisite.agent_name": "Test Agent", "requisite.tool_name": "echo"},
    )

    agent.run("do something")

    after_runs = _sum_counter(
        metric_reader,
        "requisite.agent.runs",
        **{"requisite.agent_name": "Test Agent", "requisite.status": "success"},
    )
    after_tool_calls = _sum_counter(
        metric_reader,
        "requisite.agent.tool_calls",
        **{"requisite.agent_name": "Test Agent", "requisite.tool_name": "echo"},
    )

    assert after_runs - before_runs == 1
    assert after_tool_calls - before_tool_calls == 1
