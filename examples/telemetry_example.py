"""
OpenTelemetry tracing + metrics example.

Requisite itself only requires ``opentelemetry-api`` (see
``pip install requisite-ai[otel]``) -- tracing/metrics stay no-ops until
*this application* configures a real provider, exactly like every other
OpenTelemetry-instrumented library. That's what this script does: build a
console-exporting ``TracerProvider``/``MeterProvider`` (requires
``opentelemetry-sdk``, installed separately -- the SDK + exporter choice
is always the application's call, never the library's), then run a real
``Agent`` and watch the spans it emits.

Run with:
    pip install requisite-ai[otel] opentelemetry-sdk
    GEMINI_API_KEY=... python examples/telemetry_example.py
"""

from requisite import Agent
from requisite.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 22C in {city}."


def configure_console_telemetry() -> None:
    """Wire up console-printing tracing + metrics. Application code, not
    something Requisite ever does on your behalf -- see
    docs/adr/0021-opentelemetry-tracing-and-metrics.md.
    """
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=2_000)
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))


def main() -> None:
    try:
        configure_console_telemetry()
    except ImportError:
        print(
            "opentelemetry-sdk isn't installed -- tracing/metrics will silently "
            "no-op (that's the intended zero-config behavior). Install it with: "
            "pip install opentelemetry-sdk"
        )

    agent = Agent(
        name="Weather Agent",
        provider="gemini",
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant. Use the get_weather tool when needed.",
    )

    # Every span this produces prints to the console as it completes:
    #   requisite.agent.run
    #     requisite.ai.chat_response   (the tool-calling round-trip)
    #     requisite.agent.tool_call    (get_weather)
    #     requisite.ai.chat_response   (the final answer)
    result = agent.run("What's the weather like in Tokyo? Answer in one sentence.")
    print("\n--- result ---")
    print(result.content)
    print(f"Tools used: {result.tool_calls_executed}, iterations: {result.iterations}")


if __name__ == "__main__":
    main()
