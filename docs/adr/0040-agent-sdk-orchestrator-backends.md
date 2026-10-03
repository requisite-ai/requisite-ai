# 0040. OpenAI Agents SDK, Strands, and Microsoft Agent Framework orchestrator backends

Status: Accepted
Date: 2026-10-03

## Context

`requisite-ai` has five orchestrator backends registered via
`Workflow.use_*()`: `"native"`, `"langgraph"`, `"crewai"`, `"autogen"`,
`"adk"`. ADR-0027 and ADR-0039 established the design for third-party
*agent frameworks*: the SDK handles coordination only, while every real
model call proxies back to Requisite's own `Agent.run()`/`.arun()` (its
configured provider, rate limiter, tool loop) through a model-adapter
class. The request was to add the equivalent for other vendors' agent
SDKs, so a Requisite workflow can run on OpenAI's, AWS's, or Microsoft's
framework. This ADR follows the same precedent for consistency rather
than re-litigating it, and records which of four candidate SDKs actually
fit it.

Each candidate was installed into its own isolated virtual environment
(keeping the project's dev environment clean) and its real source read,
the same discipline ADR-0027/0039 used, not assumed from documentation.

### The candidates

| SDK (PyPI) | Model-adapter hook | Coordination primitive | Install size |
|---|---|---|---|
| OpenAI Agents SDK (`openai-agents` 0.23) | `agents.models.interface.Model` -- abstract `get_response()`, `stream_response()` | `Runner.run()` chaining | 40 packages |
| AWS Strands Agents (`strands-agents` 1.57) | `strands.models.Model` -- abstract `stream()`, `structured_output()`, `get_config()`, `update_config()` | `GraphBuilder` / `Graph` DAG executor | 50 packages |
| Microsoft Agent Framework (`agent-framework-core` 1.20) | `agent_framework.BaseChatClient` -- abstract `_inner_get_response()` | `WorkflowBuilder` | 12 packages |
| Claude Agent SDK (`claude-agent-sdk` 0.2) | **none** | n/a | 32 packages |

### The Claude Agent SDK does not fit, and is not added

It is not a model-adapter framework. `claude_agent_sdk` launches a bundled
Claude Code CLI binary as a subprocess (`_internal/transport/
subprocess_cli.py` builds `[cli_path, "--output-format", "stream-json",
...]`) and speaks to it over stdio. The model is a `--model <string>`
flag; the agent loop, the tool execution and the model call all happen
inside that CLI. There is no `Model`/`LLM`/`ChatClient` interface to
subclass, so Requisite's providers can't be put underneath it, and
"coordination-only, every model call proxies to the wrapped Requisite
`Agent`" is simply not expressible. Wrapping it anyway would mean either
a backend that always calls Anthropic through Claude Code (abandoning
provider-agnosticism, rate limiting and cost limiting for that backend) or
a fake adapter that doesn't actually coordinate anything. Neither is the
same kind of thing as the other backends, so it is left out rather than
shipped misleadingly named. Anthropic remains fully supported as a
*provider* (`provider="anthropic"`), usable inside every backend here.

### Co-installability, checked

An early assumption that these couldn't share an environment with `adk`
(`google-adk` caps `opentelemetry-api<=1.42.1`, while a fresh install of
the others happened to land on 1.45) was **wrong** and is recorded here
because it was checked, not asserted: a `pip install --dry-run` of
`google-adk`, `openai-agents`, `strands-agents` and `agent-framework-core`
together resolves, with `opentelemetry-api` 1.42.1 inside every range
(`openai-agents` has no requirement on it; `strands-agents` needs
`>=1.30,<2`; `agent-framework-core` `>=1.39,<2`). There is no pin clash
of the `crewai`/`mcp` kind (ADR-0027).

All three extras are still opt-in and excluded from `all`, for footprint
rather than conflict: each pulls a whole vendor SDK's dependency tree (40-50
packages for `openai-agents`/`strands-agents`, the latter including
`boto3`), and nothing in `all` needs any of them.

## Decision

### Shared plumbing: `requisite/orchestrators/_sdk_adapter.py`

Each backend differs only in *how an agent is run through its SDK* and,
for `"sequential"`, how steps are chained. The `"supervisor"` decision
loop is identical, so it is written once. `run_supervisor()` splits the
coordinator from the workers, wraps each worker in an `SdkDelegate`
(`.name` plus async `.arun()` that runs the SDK agent and returns the real
`AgentResult` the proxy model recorded), and calls
`NativeOrchestrator._arun_delegation_loop` -- the same loop the native
backend runs -- then relabels `WorkflowResult.orchestrator` with the
backend's own name. `max_rounds` handling, unknown-worker errors and the
decision prompt are therefore the native implementation, not a per-SDK
re-derivation; a fix there reaches every backend. `check_run_inputs`,
`unsupported_strategy` and `collect_steps` carry the remaining
duplicated boilerplate.

Each SDK's adapter class is a **local class inside a lazy
`_require_sdk()`**, as in ADR-0027/0039, because the ABC it subclasses
doesn't exist until the package is installed. All three are registered in
`factory.py` as `"openai_agents"`, `"strands"`, `"agent_framework"` with
`Workflow.use_openai_agents()` / `.use_strands()` / `.use_agent_framework()`.
Each ships `"sequential"` and `"supervisor"`, the same two strategies
CrewAI/AutoGen/ADK ship; the rest are documented follow-ups.

### Sync `run()` on one long-lived event loop (`core/sync_bridge.py`)

All five async-only backends (these three, AutoGen, ADK) need a synchronous
`run()` that drives their async `arun()`. ADR-0027 and ADR-0039 chose
`asyncio.run()` for that; **this ADR supersedes that part of both**, because
live testing against real Gemini showed it is wrong whenever agents are
reused across calls. Each `asyncio.run()` makes and then closes a new event
loop, but a provider caches its async HTTP client and that client's connection
pool binds to the loop it was first used on, so the second call fails with
`RuntimeError: Event loop is closed`. It reproduces with the bare provider
and no orchestrator at all (an httpx-only `google-genai` 2.28 install: of
three consecutive `asyncio.run(agent.arun(...))` calls the second fails), and
was invisible in the project's own dev environment because there `aiohttp` is
installed and `google-genai` takes a different path. The earlier ADK live
check passed for the same reason -- an environment accident, not a design
guarantee.

`run_sync()` runs every coroutine on a single daemon-thread event loop that
lives for the process, so cached async clients always see the loop they were
created on. All five backends use it. It also works from a thread that
already has a running loop (a notebook), where `asyncio.run()` raises, and a
`run_sync` made from code already running on the bridge loop (a sync tool that
itself runs a workflow) is detected and run on a throwaway loop instead of
deadlocking. Exceptions propagate with their original type. The failure and
the fix are both pinned by tests that use a stand-in client which, like a real
one, works only on the loop it first ran on.

### OpenAI Agents SDK (`openai_agents_orchestrator.py`)

`_RequisiteModel(Model).get_response()` extracts the task from the SDK's
Responses-API `input` items, awaits the wrapped `Agent.arun()`, and returns
a `ModelResponse` carrying one `ResponseOutputMessage`, with token usage
mapped from the real provider response. `stream_response()` raises -- only
`Runner.run_streamed()` calls it and the orchestrator never does.
`"sequential"` chains `Runner.run()` calls, each step's `final_output`
becoming the next step's input -- the SDK's own documented pattern for
deterministic chaining.

**Tracing is explicitly disabled** (`RunConfig(tracing_disabled=True)`).
The SDK uploads traces -- prompts and outputs included -- to OpenAI's
platform by default whenever `OPENAI_API_KEY` is set. For a Requisite user
whose agents run on Ollama or Anthropic that would be an unexpected data
flow to a third party. Verified rather than assumed: with a spy on
`BackendSpanExporter.export`, the backend exports zero spans, while a
control `Runner.run()` over the same proxy model exports four. It is also
a permanent test.

### Strands (`strands_orchestrator.py`)

`_RequisiteModel(Model).stream()` awaits the wrapped `Agent.arun()` and
yields the Bedrock-Converse-shaped event sequence Strands consumes
(`messageStart`, `contentBlockStart`/`Delta`/`Stop`, `messageStop`,
`metadata` with usage). `structured_output()` raises (only
`Agent.structured_output()` calls it). `"sequential"` is a **real
`GraphBuilder` graph** -- one node per step, a linear chain of edges, node
ids taken from the agent names (suffixed only when two steps share one)
so Strands' own `From <name>:` input labels read correctly, and
`set_max_node_executions(len(steps))`, which both states the truth for a
linear chain and silences Strands' "no execution limits" warning on every
run. Every Strands `Agent` gets `callback_handler=None`: the default
handler prints each streamed response to stdout, which would duplicate
Requisite's own output.

A `Graph` reports node failures on a result object instead of
necessarily raising, which is the same trap that made AutoGen relabel
exceptions as `RuntimeError` (ADR-0027). Checked directly: a failing
step surfaces as the real `ProviderException`. Pinned by a test.

### Microsoft Agent Framework (`agent_framework_orchestrator.py`)

The extra depends on **`agent-framework-core`, not the `agent-framework`
umbrella package**: the umbrella installs every connector (Azure
Functions, Redis, Mistral, Bedrock, Foundry, ...), 120+ packages against
`-core`'s 12, none of which a coordination-only backend uses. `-core`
alone provides `Agent`, `BaseChatClient` and `WorkflowBuilder`.

`_RequisiteChatClient(FunctionInvocationLayer, BaseChatClient)` implements
`_inner_get_response()`. `FunctionInvocationLayer` is mixed in the way the
framework's own public clients compose it; without it the framework logs
"chat client does not support function invoking" for every agent built.
It is inert here -- no tools are ever attached.

**A real semantic difference found by experiment.** In a workflow the
agents share one conversation: the second agent's client receives the
original *user* message followed by the first agent's reply as an
*assistant* message (carrying `author_name`) -- not a rewritten "last user
message". The other backends can forward "the last user message" and get
the handoff; here that would silently drop it, and the writer would see
only the original task. `_extract_task_text` therefore rebuilds the task
(last user message) plus prior-agent context (assistant messages after it,
tagged by author) through `NativeOrchestrator._task_prompt_with_context`,
giving a worker the same "task + context from previous steps" shape as on
the native backend. Pinned by a test that asserts the first agent's reply
is present in the second agent's prompt.

## Alternatives considered

- **Wrap the Claude Agent SDK too.** Rejected -- see above; there is no
  model hook, so it could only be a different, non-provider-agnostic
  thing under a misleading shared name.
- **Depend on the `agent-framework` umbrella package.** Rejected -- 120+
  packages for connectors this backend never touches; `-core` has
  everything needed.
- **Re-implement the supervisor loop in each backend** (as the ADK backend
  does inline inside its `BaseAgent`). Rejected for these three -- it
  would be three more copies of the same ~25 lines, each a place for
  `max_rounds`/unknown-worker behaviour to drift. ADK's loop has to live
  inside its `BaseAgent._run_async_impl` to emit ADK events, which is why
  it is the exception.
- **Use each SDK's own LLM-driven handoff/supervisor primitive**
  (OpenAI handoffs, Strands `Swarm`, Microsoft handoff/group-chat
  builders). Rejected -- those depend on the model emitting the SDK's own
  tool calls, which the proxy adapter deliberately bypasses (the wrapped
  Requisite `Agent` runs its own tool loop). Same reason CrewAI's
  `"hierarchical"` was deferred in ADR-0027. The shared delegation loop
  gives `"supervisor"` identical semantics on every backend instead.
- **Keep `asyncio.run()` and instead rebuild each provider's async client
  whenever the running loop changes.** Rejected -- it would have to be done
  in every provider (Gemini, OpenAI and everything wire-compatible with it,
  Anthropic, Ollama), each with its own client type and teardown, and still
  wouldn't help the SDKs' own cached state. One bridge fixes every provider
  at the single place the loop is created.
- **Run each sync call on a fresh thread with its own loop.** Rejected --
  that is `asyncio.run()` again, same closed-loop problem.
- **Leave SDK tracing/printing at defaults.** Rejected -- OpenAI's default
  trace upload and Strands' default stdout handler are both surprising
  side effects of using the SDK as a coordinator.
- **Add the three extras to `all`.** Rejected -- not because of a conflict
  (there is none, see above) but because it would make every
  `pip install requisite-ai[all]` pull three vendor SDK trees, including
  `boto3`, that nothing else in `all` uses.

## Consequences

### Positive

- Three more vendor frameworks can coordinate a Requisite workflow with
  one line (`workflow.use_strands()` etc.), with every agent still on its
  own provider, rate limiter and cost limiter.
- Real SDK code runs in the tests, not mocks: each backend was verified
  against its real installed package -- an adversarial script (validation,
  real handoff, supervisor delegation, `max_rounds`, unknown worker,
  single/three-step chains, odd and duplicate agent names, failure
  propagation) before the permanent suite, then the suite itself.
- The SDK-independent half (`_sdk_adapter`, and each backend's pure
  text-extraction helper) is tested in `tests/test_sdk_adapter.py`, which
  runs in CI where the opt-in SDKs aren't installed -- the per-backend
  files `importorskip` their SDK and so cover nothing there.
- Real exception types propagate on all three, with no AutoGen-style
  `RuntimeError` relabelling.
- AutoGen's and ADK's sync `run()` get the same fix, closing a latent bug
  they shared.

### Negative / risks

- Each is a heavier optional install than `native`/`langgraph`. They
  also cannot be exercised by CI's `[dev,all]` install; correctness there
  rests on the local per-SDK runs and the SDK-independent tests.
- Only `"sequential"` and `"supervisor"` -- none of each SDK's richer
  patterns (OpenAI handoffs/agents-as-tools, Strands `Swarm`, Microsoft
  concurrent/group-chat/Magentic builders).
- On Microsoft's framework, sequential context travels through the shared
  transcript, so a worker's prompt is rebuilt from it; if the framework
  changes how a workflow populates that transcript the extraction must
  follow. The test pinning the handoff is what would catch it.
- The bridge leaves one daemon thread (`requisite-sync-bridge`) running for the
  life of the process once any sync backend call is made, and a sync `run()`
  now blocks its caller while the work happens on that thread. Direct
  `asyncio.run(agent.arun(...))` by user code is unchanged and still
  subject to the closed-loop problem with reused agents; only the sync
  entry points Requisite owns are fixed.
- These SDKs move fast (0.x and young 1.x). Lower bounds in `pyproject.toml`
  are the versions verified here, not a compatibility guarantee.

### Follow-ups

- `"parallel"` is a natural fit on all three (`asyncio.gather` over
  `Runner.run`; a fan-out `Graph`; a `ConcurrentBuilder`).
- Revisit the Claude Agent SDK only if it ever exposes a pluggable model
  or transport hook.
- A demo in `requisite-demo` for each, once released and installable from
  PyPI there.
