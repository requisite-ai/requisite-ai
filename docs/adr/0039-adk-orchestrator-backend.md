# 0039. ADK orchestrator backend

Status: Accepted
Date: 2026-10-02

## Context

`requisite-ai` already has four orchestrator backends registered via
`Workflow.use_*()`: `"native"`, `"langgraph"`, `"crewai"`, `"autogen"`.
ADR-0027 established the precedent for third-party *agent frameworks*
(as opposed to graph libraries like LangGraph): the third-party package
handles coordination/control-flow only, while every real model call
proxies back to Requisite's own `Agent.run()`/`.arun()` (its configured
provider, rate limiter, tool loop). This ADR adds Google's Agent
Development Kit (`google-adk`) as a fifth backend, following that same
precedent for consistency rather than re-litigating the core design
question ADR-0027 already resolved.

Verified live by installing `google-adk` (2.11.0) into this project's
venv and reading its actual current source
(`.venv/Lib/site-packages/google/adk/`), the same discipline ADR-0027
used for `crewai`/`autogen-agentchat` -- not assumed from docs.

### Extension points mirror the existing precedent

`google.adk.models.base_llm.BaseLlm` is a pydantic `BaseModel` with one
abstract method:

```python
async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]
```

-- the same shape of extension point as CrewAI's `BaseLLM.call()` and
AutoGen's `ChatCompletionClient.create()`. `LlmAgent(model=<BaseLlm
instance>)` accepts the adapter instance directly, confirmed from the
docs and from how `model: Union[str, BaseLlm]` resolves -- **no
`LLMRegistry` needed**, unlike what a model-string-based integration
would require.

### The request/response shape needs real translation

`LlmRequest.contents: list[google.genai.types.Content]` -- Gemini-SDK-shaped
(`role` + `parts`, each part possibly `.text`/`.function_call`/
`.function_response`), not a generic message list. `_extract_task_text`
(`adk_orchestrator.py`) walks `contents` in reverse for the last
`role == "user"` entry and joins its text parts -- the same "forward the
last assembled message" shape CrewAI's/AutoGen's own `_extract_task_text`
helpers already use, just translated out of `genai.types` first.
`LlmResponse` has `model_config = ConfigDict(extra='forbid', ...)`, so
the adapter constructs it with exactly `content=genai_types.Content(role="model", parts=[...])`.

### `BaseAgent`, not the deprecated convenience classes

`google.adk.agents.sequential_agent.SequentialAgent`,
`parallel_agent.ParallelAgent`, and `loop_agent.LoopAgent` are each
explicitly decorated `@deprecated`, confirmed directly in source:

> "SequentialAgent is deprecated in favor of Workflow and will be
> removed in a future version"

(identical wording for `ParallelAgent`/`LoopAgent`). The replacement,
`google.adk.workflow.Workflow` (`google/adk/workflow/_workflow.py`),
lives in underscore-prefixed internal-feeling modules
(`_workflow.py`/`_errors.py`) with far less documentation than the
stable, public `google.adk.agents.base_agent.BaseAgent`. Decision: build
directly on `BaseAgent._run_async_impl` for both strategies in this
backend -- the same judgment this project already made for LangGraph
(built on the stable public `StateGraph`, not internal APIs).

Note: `google.adk.workflow.Workflow` is a real class-name collision
with Requisite's own `requisite.workflows.workflow.Workflow` -- a
different fully-qualified path, so no actual import conflict, but worth
disambiguating in one sentence wherever both are mentioned (done in
`Workflow.use_adk()`'s docstring and the `adk_orchestrator.py` module
docstring).

### ADK requires a `Runner` -- a real structural difference

Unlike CrewAI's `Crew(...).kickoff()` or AutoGen's `Team.run()`, both
self-contained one-shot calls, ADK requires constructing a `Runner` and
a session before invoking anything. `InMemoryRunner(agent=root_agent)`
is the lightweight, self-contained option (in-memory session/artifact/
memory services, per its own docstring "for testing and development").

For the sync `.run()` path: `Runner.run()`'s own sync wrapper (confirmed
in `runners.py`) spins up its own background thread with an internal
`asyncio.run()` and correctly re-raises the original exception type on
the calling thread. It would have been usable directly -- but session
creation (`InMemorySessionService.create_session`) is `async def` with
no non-deprecated sync counterpart: `create_session_sync` logs
`'Deprecated. Please migrate to the async method.'` on every single
call, confirmed directly in `in_memory_session_service.py`. Rather than
mix two different sync strategies (a thread-based one for invocation, a
deprecated-logging one for session setup), `AdkOrchestrator.run()`
wraps `asyncio.run(self.arun(...))` end to end, matching
`AutoGenOrchestrator.run()`'s own reasoning for the same shape.

### Dependency weight is real, not hypothetical

`google-adk`'s *base* install (no extras) pulls in `fastapi`, `uvicorn`,
`starlette`, `watchdog`, `graphviz`, `authlib`, `google-auth[pyopenssl]`
-- because `adk web`/`adk eval`/deploy CLI tooling ships in the core
package, not behind extras. This is categorically heavier than
`langgraph` (a pure graph library) or `crewai`/`autogen-agentchat`
(plain Python libraries, no web server).

Confirmed via a real `pip install google-adk` into this project's venv:
it force-upgraded `google-genai` 2.11.0 → 2.27.0 and `google-auth`
2.55.2 → 2.59.1 (our own `gemini` extra pins no upper bound on
`google-genai`, so this is not a hard resolution conflict -- just real
upgrade pressure). Re-ran `pytest tests/ -k gemini` after the forced
upgrade: still 10/10 passing.

## Decision

### `requisite/orchestrators/adk_orchestrator.py`

`_RequisiteLlm(BaseLlm)` (built inside `_require_adk()`, same lazy
local-class pattern as `_RequisiteLLM`/`_RequisiteChatCompletionClient`):
`generate_content_async()` extracts task text via `_extract_task_text`,
proxies to `self._requisite_agent.arun()`, collects the real
`AgentResult` for `WorkflowResult.steps`, and yields one `LlmResponse`.
`stream` is accepted but never branched on -- no participant in this
integration is ever constructed with streaming enabled.

`"sequential"`: `_RequisiteSequentialAgent(BaseAgent)` -- `_run_async_impl`
loops `self.sub_agents` (each an `LlmAgent` wrapping its own
`_RequisiteLlm` instance) and re-yields their events. This is the exact
same one-line loop `SequentialAgent._run_async_impl` contains
internally; using the deprecated class would buy nothing a four-line
subclass doesn't, while keeping one consistent extension point across
both strategies.

`"supervisor"`: `_RequisiteSupervisorAgent(BaseAgent)` -- `_run_async_impl`
reuses `NativeOrchestrator._split_coordinator_and_workers`,
`_SupervisorDecision`, `_supervisor_prompt`, and
`NativeOrchestrator._resolve_delegate` directly, exactly like
`AutoGenOrchestrator`'s `selector_func` does. The coordinator's decision
calls and each delegate's run go through `coordinator.ai.achat(...)`/
`delegate.arun(...)` directly -- no `_RequisiteLlm`/`LlmAgent` involved,
since there's no ADK LLM-call machinery worth round-tripping through
for a strategy that's already a plain decision loop. On `max_rounds`
exhaustion, `AgentException` is raised directly inside the async
generator.

`AdkOrchestrator.run`/`.arun`: build the root agent per strategy,
construct `InMemoryRunner(agent=root_agent)`, create a session, send one
`new_message`, and take the last event with non-empty text content as
the final answer -- this generic extraction works for both strategies
without a strategy-specific branch at the orchestrator level.

### `requisite/orchestrators/factory.py`, `requisite/workflows/workflow.py`

`"adk"` registered via the same lazy-import closure pattern as
`"crewai"`/`"autogen"`. `Workflow.use_adk()` added, modeled on
`use_autogen()`.

### `pyproject.toml`

`adk = ["google-adk>=2.11"]`, a new optional extra, **not** added to
`all`. No new mypy override needed -- the existing `module = [...,
"google.*", ...]` wildcard already covers `google.adk.*`. Version
bumped `0.37.0` → `0.38.0` (minor, new backend, same size as
`CostLimiter`'s bump).

## Alternatives considered

- **`SequentialAgent`/`ParallelAgent`/`LoopAgent` for `"sequential"`.**
  Rejected -- each is `@deprecated` with no removal version given, and
  wrapping one buys nothing a four-line `BaseAgent` subclass doesn't
  already give directly.
- **ADK's newer `google.adk.workflow.Workflow` system** for either
  strategy. Rejected -- lives in underscore-prefixed, internal-feeling
  modules, far less documented than the stable public `BaseAgent`; the
  same judgment already made for LangGraph (built on public
  `StateGraph`, not internal APIs).
- **`Runner.run()`'s own sync generator directly**, instead of
  `asyncio.run(self.arun(...))`. Rejected -- no non-deprecated sync
  session-creation path exists, so a fully sync strategy would still
  need to call the deprecated, warning-logging `create_session_sync`;
  one `asyncio.run()` wrapper covering the whole flow is simpler and
  matches the AutoGen precedent.
- **`runner.run_debug(...)`** for the "just run it" path. Rejected --
  its own docstring states it is for debugging/experimentation only
  ("For production use, please use run_async()"), and it is itself
  `async def`, so it would not even simplify the sync path.
- **Including `adk` in the `all` extra**, for install-target parity
  with every other backend. Rejected on dependency-weight grounds --
  not a hard version conflict like `crewai`/`mcp` (ADR-0027), but a
  real, heavier install footprint worth keeping opt-in. Stated here
  honestly as a softer rationale than ADR-0027's hard conflict, not
  inflated to match it.

## Consequences

### Positive

- A fifth orchestrator backend following the established
  coordination-only precedent, closing the gap between "every strategy
  has native/langgraph parity" and "every coordination-framework
  backend follows the same proxy-adapter design."
- Exceptions raised inside `_RequisiteSupervisorAgent._run_async_impl`
  propagate to the caller unchanged, with zero extra plumbing -- an
  actual structural advantage over `AutoGenOrchestrator`, whose
  `selector_func` needed a found-and-fixed bug workaround (its runtime
  relabels any exception as a generic `RuntimeError`) for exactly this.
- Both strategies verified against the real installed package (an
  adversarial scratch script exercising all 7 real scenarios, then the
  permanent pytest suite, both run against real `google-adk`
  coordination code via `pytest.importorskip`), not mocked end-to-end.

### Negative / risks

- `google-adk`'s base install is categorically heavier than every other
  optional backend (`fastapi`/`uvicorn`/`starlette`/`watchdog`/
  `graphviz`/`authlib`) -- a real cost for anyone who opts into
  `requisite-ai[adk]`.
- Only two of ADK's many possible agent-composition patterns are
  covered -- no `"hierarchical"`/`"parallel"` mapping.
- `google.adk.workflow.Workflow` name-collides with Requisite's own
  `Workflow` -- cosmetic (different fully-qualified paths), but a real
  source of confusion in docs/examples if not called out.

### Follow-ups

- ADK `"hierarchical"`/`"parallel"` support, if a concrete use case
  needs it.
- Revisit `google.adk.workflow.Workflow` once/if it stabilizes out of
  its underscore-prefixed internal modules -- it may eventually be the
  more idiomatic ADK-side extension point.
