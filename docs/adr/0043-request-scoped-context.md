# 0043. Request-scoped context for tools and providers

Status: Accepted
Date: 2026-10-04

## Context

A lab that handled support tickets needed the human operator's identity
inside a tool. Requisite tools receive only the model's arguments, so the lab
kept the operator on the tool object, which limited it to one ticket at a time
per platform. It also kept a provider wrapper to carry a tenant id, because
there was no per-request channel to a provider either. The lab's diagnosis was
that "the sync/async bridge loop does not carry `contextvars`".

That diagnosis is only partly right. Checked directly (Python 3.13; these are
stdlib guarantees, not version quirks):

| Path | Context visible? |
|---|---|
| `await` in the same task, `create_task`, `gather` | yes |
| `asyncio.to_thread` | yes (copied) |
| `run_sync` (the sync bridge, `run_coroutine_threadsafe`) | yes (copied at submit) |
| `ThreadPoolExecutor.submit` | **no** (workers start empty) |

So no new transport is needed. What was missing: a supported, typed carrier
with a clear contract, a way for tools to receive it without reading a global,
and the two places Requisite itself used `ThreadPoolExecutor.submit`: five in
the native orchestrator's threaded strategies (parallel, consensus, debate,
map-reduce, tree-of-thoughts) and the bridge's helper-thread fallback.

## Decision

### `RequestContext` and one context variable (`requisite/core/context.py`)

`RequestContext` is a frozen model: `user`, `tenant`, `correlation_id` and a
free-form `attributes` dict. It is immutable because one instance is shared by
every tool and provider call in a run. It holds trusted, application-supplied
data. No ids are generated for the caller.

One module-level `ContextVar` carries it. `current_context()` returns it or
`None`; `require_context()` raises a clear `ConfigurationException` if absent;
`request_context(ctx)` is a nesting-safe `with` block (inner overrides outer;
`None` inherits). `submit_with_context(executor, fn, ...)` runs `fn` in a copy
of the caller's context and replaces every `executor.submit` in Requisite.

### Entry points

`Agent.run/arun(..., context=)` and `Workflow.run/arun(..., context=)`. It is
an explicit keyword, not left in `**kwargs`, because `**kwargs` flows on to the
provider call. At workflow level it is set around the orchestrator, so every
backend and agent inside inherits it. An explicit `context=` on an inner agent
overrides it.

### Tools

A tool function may declare a parameter annotated `RequestContext` (or
`Optional[RequestContext]`, any name). It is excluded from the JSON schema the
model sees and filled in at execution time. Anything the model sent under that
name is discarded first, so a model cannot claim to be another user. With no
context in scope: a required parameter raises a `ToolException` that says to
pass `context=`; an `Optional` parameter receives `None`; a parameter with a
default keeps it. `current_context()` works in any tool regardless.

### Providers

No provider API change. A provider, or a decorator around one, calls
`current_context()` inside `chat`/`achat`. This replaces a tenant-carrying
wrapper.

### Observability

The `requisite.agent.run` span gets `requisite.correlation_id` when set.
User and tenant are deliberately not put on spans (identifier privacy);
metric attributes are untouched, so metric cardinality does not change.

## Verified

- Thread-pool tests fail without `submit_with_context` and pass with it, for
  all five native strategies and the bridge fallback.
- Isolation: 25 concurrent `arun` calls with distinct contexts and 12
  concurrent threads calling `run` each saw only their own context in a tool
  and in the provider. The one-at-a-time limit is gone.
- Every orchestrator backend delivers the context to a tool, sync and async:
  LangGraph, CrewAI, AutoGen, ADK, OpenAI Agents SDK, Strands and Microsoft
  Agent Framework. No adapter needed changes. (The last three are tested in
  separate environments because they are opt-in extras.)
- Live, real Gemini through an ADK workflow: two tickets in flight at once for
  different operators; each `close_ticket` call recorded the right operator,
  tenant and correlation id.

## Alternatives considered

- **Hold the identity on the tool or agent object** (the lab's workaround).
  Rejected: it serialises requests and leaks across them.
- **Pass the context as a model-visible tool argument.** Rejected: identity
  must never come from the model.
- **A new transport (thread-local, explicit parameter threading).** Rejected:
  `contextvars` already flows through every path except the executor, which
  is fixed.
- **Auto-fill user/tenant on spans.** Rejected for now: identifiers are
  sensitive; an opt-in follow-up.
- **Per-call `provider_kwargs` on `Agent.run`.** Rejected: the context covers
  the tenant case without widening the provider surface.

## Consequences

Positive: concurrent requests for different users stay isolated; tools and
providers read identity without bookkeeping; works under every backend.

Negative / risks:

- Context flows only along Python's context chain. A thread started by hand
  (`threading.Thread`, `executor.submit`) starts empty; use
  `submit_with_context` or `contextvars.copy_context().run`. Another process,
  including an MCP server, does not see it.
- The context is not authorization. It tells a tool who the request is for; the
  tool, or the service it calls, must still decide what that user may do, and
  derive it from this verified context, never from message text.

Follow-ups: carry the correlation id to MCP servers through the request's
`_meta`; opt-in user/tenant span attributes; stamping it on log records.
