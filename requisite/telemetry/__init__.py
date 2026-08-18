"""
Telemetry: structured logging, tracing, and metrics.

See ``requisite.telemetry.logging`` for :class:`JSONFormatter` and
:func:`configure_logging`, and ``requisite.telemetry.otel`` for
:func:`get_tracer`/:func:`get_meter` -- optional OpenTelemetry
instrumentation, see ``docs/adr/0021-opentelemetry-tracing-and-metrics.md``.
"""

from requisite.telemetry.logging import JSONFormatter, configure_logging
from requisite.telemetry.otel import get_meter, get_tracer

__all__ = ["JSONFormatter", "configure_logging", "get_meter", "get_tracer"]
