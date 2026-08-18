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
