"""
Optional OpenTelemetry tracing + metrics instrumentation.

Unlike every other optional-dependency integration in this framework
(:class:`~requisite.memory.redis.RedisMemory`,
:class:`~requisite.rag.vectorstores.pinecone.PineconeVectorStore`, etc.
-- lazy import, ``ImportError`` -> a helpful ``ConfigurationException``),
this instrumentation lives directly in the call path of
:meth:`~requisite.ai.AI.chat_response` and
:meth:`~requisite.agents.agent.Agent.run`, which every user of the
library calls constantly whether or not they want tracing. Raising on
a missing dependency there would break basic usage for anyone without
``opentelemetry-api`` installed. So :func:`get_tracer`/:func:`get_meter`
**never raise** -- without the package installed, they return small
no-op stand-ins with the same method surface.

Install with ``pip install requisite-ai[otel]`` (``opentelemetry-api``
only -- the *application* installs ``opentelemetry-sdk`` plus whichever
exporter it wants and configures the provider; Requisite never calls
``opentelemetry.trace.set_tracer_provider(...)`` or the metrics
equivalent itself). Until an application does that, OpenTelemetry's own
API already returns safe no-op tracer/meter objects -- the same
"opt-in, never automatic" convention :func:`~requisite.telemetry.logging.configure_logging`
established for structured logging, extended here for free by
OpenTelemetry's own API/SDK split rather than anything Requisite-specific.
"""

from __future__ import annotations

from typing import Any

try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the dep
    _OTEL_AVAILABLE = False


class _NoOpSpan:
    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NoOpTracer:
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpCounter:
    def add(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NoOpHistogram:
    def record(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NoOpMeter:
    def create_counter(self, *args: Any, **kwargs: Any) -> _NoOpCounter:
        return _NoOpCounter()

    def create_histogram(self, *args: Any, **kwargs: Any) -> _NoOpHistogram:
        return _NoOpHistogram()


# OpenTelemetry GenAI semantic conventions (status: Development upstream, so
# attribute names may still change). Every name lives in this block so a rename
# is a one-place edit. Span-only: metric attributes are left untouched to keep
# metric cardinality unchanged.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Requisite provider name -> the convention's well-known gen_ai.provider.name
# value. Names with no well-known value pass through unchanged.
_GENAI_PROVIDER_NAMES = {"gemini": "gcp.gemini"}


def genai_request_attributes(provider: str, model: str) -> dict[str, str]:
    """Span attributes known before a chat call is made."""
    return {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_PROVIDER_NAME: _GENAI_PROVIDER_NAMES.get(provider, provider),
        GEN_AI_REQUEST_MODEL: model,
    }


def set_genai_response_attributes(span: Any, response: Any) -> None:
    """Record the model that answered and its token usage on ``span``."""
    span.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, response.usage.prompt_tokens)
    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, response.usage.completion_tokens)


def get_tracer(name: str) -> Any:
    """Return an OpenTelemetry tracer for ``name``, or a safe no-op stand-in.

    Never raises, regardless of whether ``opentelemetry-api`` is
    installed or whether the application has configured a real
    ``TracerProvider``.
    """
    if not _OTEL_AVAILABLE:
        return _NoOpTracer()
    return _otel_trace.get_tracer(name)


def get_meter(name: str) -> Any:
    """Return an OpenTelemetry meter for ``name``, or a safe no-op stand-in.

    Never raises, regardless of whether ``opentelemetry-api`` is
    installed or whether the application has configured a real
    ``MeterProvider``.
    """
    if not _OTEL_AVAILABLE:
        return _NoOpMeter()
    return _otel_metrics.get_meter(name)
