
# 0021. OpenTelemetry tracing and metrics

Status: Accepted
Date: 2026-08-19

## Context

`ROADMAP.md`'s Telemetry section has two remaining 📋 lines: *"Tracing
(e.g. OpenTelemetry spans around provider calls)"* and *"Metrics
(request counts, latency, token usage aggregation)."* Chosen together
(Keyan, via `AskUserQuestion`) since they share one optional dependency
and the same instrumentation points -- `ROADMAP.md` itself groups them
under one "Telemetry" heading, and OpenTelemetry ships trace and metrics
under one umbrella spec.

Structured (JSON) logging (ADR-0003) already established this project's
"opt-in, never automatic" convention for observability: `configure_logging(...)`
is a separate call an *application* makes, never invoked by framework
code itself. This feature extends that convention to tracing/metrics
rather than inventing a new one -- though, as below, OpenTelemetry's own
API/SDK split turns out to deliver "opt-in, never automatic" even more
directly than the logging module's `configure_logging()` pattern did.

`AI`'s six facade methods (`chat_response`, `achat_response`, `stream`,
`astream`, `stream_response`, `astream_response`) are the confirmed
single choke point every provider call funnels through -- already proven
by `RateLimiter` (ADR-0008), which hooks into these exact six call sites
with zero changes needed in any provider file. `Agent.run`/`arun` call
these same `AI` methods, so they're covered for free; the tool-calling
loop itself and tool execution are separate call sites needing their own
instrumentation for a coherent trace tree.

## Decision

### `get_tracer`/`get_meter` never raise -- unlike every other optional integration

Every other optional-dependency pattern in this codebase (`RedisMemory`,
`PineconeVectorStore`: lazy import, `ImportError` -> a helpful
`ConfigurationException`) is appropriate because the user explicitly
opted into that specific backend (`memory="redis"`,
`vector_store="pinecone"`). Tracing instrumentation is different: it
lives directly in the call path of `AI.chat_response(...)` and
`Agent.run(...)`, which every user of the library calls constantly
whether or not they want tracing. Raising there would break basic usage
for anyone without `opentelemetry-api` installed. So `requisite/telemetry/otel.py`'s
`get_tracer`/`get_meter` **never raise** -- without the package
installed, they return small internal no-op stand-ins
(`_NoOpTracer`/`_NoOpSpan`/`_NoOpMeter`/`_NoOpCounter`/`_NoOpHistogram`,
a few lines each) with the same method surface
(`start_as_current_span`, `set_attribute`, `record_exception`,
`create_counter().add(...)`, `create_histogram().record(...)`).

### No constructor injection -- deliberately *not* the `RateLimiter` precedent

The obvious move, given `RateLimiter`'s explicit-injection-only
precedent (ADR-0008, "no globals, explicit injection"), would be
`AI(..., tracer=my_tracer)` / `Agent(..., tracer=my_tracer)`. That
precedent doesn't transfer here, and applying it anyway would actively
work against OpenTelemetry's own design: `RateLimiter` solves a
different problem -- sharing *stateful quota* across specific instances
that draw on one API key. OpenTelemetry's API/SDK split exists
*specifically* so many independent libraries in one process can all
emit coherent traces without each accepting a `tracer=` constructor
argument: the application configures one global `TracerProvider`/
`MeterProvider` once (`opentelemetry.trace.set_tracer_provider(...)`),
and every instrumented library -- Requisite included -- calls
`trace.get_tracer(__name__)`, which is a `ProxyTracer` that
transparently delegates to whatever provider is *currently* globally
configured at call time (verified directly: a tracer obtained before
any provider is configured still correctly emits to a provider
configured afterward -- this is what makes "call `get_tracer()` at
module import time" safe at all). Copying `RateLimiter`'s style would
make Requisite's tracing incompatible with how every other
OTel-instrumented library in a user's stack behaves. So:
module-level `_tracer = get_tracer("requisite.ai")` / `_meter =
get_meter("requisite.ai")` in `ai.py`, and the `"requisite.agent"`
equivalents in `agent.py` -- no constructor parameters added anywhere.
Requisite itself never calls `set_tracer_provider`/the metrics
equivalent -- that stays the application's decision, matching ADR-0003's
"the framework reads/stores preferences, applications decide when to
act on them" rule exactly.

### Manual `span.record_exception()` is redundant and was removed

An early draft wrapped each provider call in `try/except Exception as
exc: span.record_exception(exc); ...; raise`. Verified directly against
the real SDK: `start_as_current_span` already auto-records the
exception *and* sets `StatusCode.ERROR` on the span the moment an
exception propagates out of its `with` block -- calling
`record_exception` manually as well would double the exception event on
the same span. The final code keeps the `try/except` only for what
still needs it (incrementing the `status="error"` counter before
re-raising), and lets the span's own automatic exception handling do
the rest. The same reasoning applies to `Agent.run`/`arun`'s outer span:
a `try/finally` records `requisite.agent.runs`/`requisite.agent.run.duration`
on every exit path (early return, `max_iterations` exhaustion, or any
deeper exception bubbling up) without manually touching span exception
state at all.

### Instrumentation points

**`requisite/ai.py`** -- all six methods, same six call sites
`RateLimiter.acquire()`/`aacquire()` already hooks into: wrap the
provider call in `with _tracer.start_as_current_span("requisite.ai.<method>",
attributes={"requisite.provider": ..., "requisite.model": ...})`, record
`requisite.ai.requests` (counter, `requisite.status=success|error`) and
`requisite.ai.request.duration` (histogram, seconds) on every call.
`chat_response`/`achat_response` additionally record `requisite.ai.tokens`
(counter, `requisite.token_type=prompt|completion`) from `response.usage`
(always populated, defaulting to `0` -- see `Usage`'s own docstring --
so no `None` check needed). The four streaming methods don't get token
metrics: `StreamChunk` has no usage field, and inventing one wasn't
needed by anything driving this ADR. `with`-block context managers work
identically whether the enclosing function is sync, async, or a
generator -- `stream`/`astream`/`stream_response`/`astream_response`
just wrap their existing `for`/`async for`/`yield from` body in the same
`with` block; the span stays open across each `yield` since generator
suspension doesn't trigger `__exit__`.

**`requisite/agents/agent.py`** -- `run`/`arun`: one outer
`"requisite.agent.run"` span wraps the entire tool-calling loop
(attributes: `requisite.agent_name`), recording `requisite.agent.runs`
and `requisite.agent.run.duration` in a `finally` block. Because this
span is entered via `start_as_current_span` (which sets OTel's
current-context), every `AI`-level span opened inside the loop
automatically nests under it -- verified directly with a real
`InMemorySpanExporter`: a tool-calling round trip produces one
`requisite.agent.run` root span with two `requisite.ai.chat_response`
children plus a `requisite.agent.tool_call` child, correctly parented.

Tool execution gets its own child span + `requisite.agent.tool_calls`
counter: in `run`, wrapped inline around each
`tool_instance.execute(...)` call. In `arun`, tool calls run
concurrently via `asyncio.gather` -- a small inline async helper
(`_traced_aexecute`) wraps each call so the span is opened *inside* the
coroutine that becomes its own `asyncio.Task`. Verified directly: each
concurrently-gathered coroutine gets an independent copy of the current
`contextvars` context when wrapped as a Task, so three concurrent tool
calls in one turn each produced a correctly-nested
`requisite.agent.tool_call` span under the same parent run span, with no
cross-task leakage.

### Explicitly deferred (not attempted here)

- **Workflow/orchestrator-level spans** (one span per `Workflow.run()`
  strategy, spanning multiple agents) -- a separate, larger surface (11
  strategies) than "provider calls" + single-agent tool loop, which is
  what `ROADMAP.md`'s Telemetry lines actually describe.
- **Adopting OpenTelemetry's `gen_ai.*` semantic conventions.** Still
  explicitly experimental/unstable upstream as of this writing. This
  feature uses a stable, custom `requisite.*` span/metric/attribute
  namespace instead, to avoid coupling to a spec that's still changing;
  adding (or switching to) `gen_ai.*` later is a compatible follow-up,
  not a breaking one.

### Dependency: `opentelemetry-api` only, not `-sdk`

Standard convention for instrumented libraries: depend on the thin,
stable API only; the *application* installs `opentelemetry-sdk` plus
whichever exporter it wants (console, OTLP, Jaeger, ...) and configures
the provider. This is also what makes "opt-in, never automatic" work for
free -- without an app-configured SDK, OpenTelemetry's own API returns a
safe no-op provider; Requisite doesn't need to detect "is tracing
configured" itself.

`pyproject.toml`: new `otel = ["opentelemetry-api>=1.20"]` extra, added
to `all =`. `dev` extra gains `opentelemetry-sdk>=1.20` -- test-only, to
assert real spans/metrics via `InMemorySpanExporter`/`InMemoryMetricReader`
in `tests/test_telemetry_otel.py`, which itself is gated with
`pytest.importorskip("opentelemetry")` so the suite still passes without
the `dev` extra installed. `[[tool.mypy.overrides]]` gains
`"opentelemetry.*"`, matching every other optional SDK.

## Alternatives considered

- **Constructor injection** (`tracer=`/`meter=` on `AI`/`Agent`). Rejected
  -- see "No constructor injection" above; would make Requisite
  incompatible with how every other OTel-instrumented library in a
  user's dependency stack behaves.
- **`ConfigurationException` on missing `opentelemetry-api`**, matching
  `RedisMemory`/`PineconeVectorStore`. Rejected -- this instrumentation
  sits in the call path of methods every user calls regardless of
  whether they want tracing; raising there would break basic usage.
- **Manually recording exceptions on every span** (`span.record_exception(exc)`
  in each `except` block). Rejected once verified redundant --
  `start_as_current_span`'s own `__exit__` already does this
  automatically for any exception propagating out of its `with` block.
- **`gen_ai.*` OpenTelemetry semantic conventions** for span/attribute
  naming. Rejected for now as still experimental/unstable upstream; see
  "Explicitly deferred" above.

## Consequences

### Positive

- Closes both remaining 📋 lines in `ROADMAP.md`'s Telemetry section.
- Zero-config, zero-behavior-change for anyone not using it: without
  `opentelemetry-api` installed, every instrumentation call is a few
  no-op method calls; with it installed but no provider configured,
  OpenTelemetry's own API already no-ops.
- Purely additive: no changes to any provider file, `BaseOrchestrator`,
  `Workflow`, or any public constructor signature.
- A real trace tree, not just isolated spans: `Agent.run`/`arun`'s outer
  span makes every nested `AI` call and tool call show up under one
  coherent trace, verified directly against a real exporter including
  the concurrent-tool-call case.

### Negative / risks

- Six near-identical instrumentation blocks in `ai.py` (one per facade
  method) rather than one shared helper -- matches `RateLimiter`'s own
  precedent of duplicating its two-line hook at each of the same six
  call sites rather than introducing an abstraction layer; consistent
  with the existing code's style, not a new pattern.
- `requisite.*` span/metric names will need revisiting if OpenTelemetry's
  `gen_ai.*` semantic conventions stabilize and users expect
  interoperability with other GenAI-instrumented tooling out of the box
  -- an accepted, explicit trade-off (see "Explicitly deferred").
- No workflow/orchestrator-level spans yet -- a multi-agent `Workflow.run()`
  produces several independent `requisite.agent.run` trace roots today,
  not one encompassing trace. Follow-up, not attempted here.

### Follow-ups

- Workflow/orchestrator-level spans (one span per `Workflow.run()`
  strategy, parenting the per-agent runs it delegates to) -- not scoped
  here.
- Adopting (or additionally emitting) OpenTelemetry's `gen_ai.*`
  semantic conventions once they stabilize.
- Token-usage metrics on the streaming methods, if a provider ever
  reports usage data on `StreamChunk` -- no such data exists today.
