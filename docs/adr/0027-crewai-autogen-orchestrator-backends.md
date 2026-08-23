# 0027. CrewAI and AutoGen orchestrator backends

Status: Accepted
Date: 2026-08-23

## Context

`ROADMAP.md`'s orchestration section had two placeholders:
`"crewai"`/`"autogen"` were already registered in
`requisite/orchestrators/factory.py` via a `_not_yet_implemented(...)`
helper, and `Workflow.use_crewai()`/`.use_autogen()` already existed as
stubs setting `self._orchestrator_name` with no real backend behind
them. This ADR implements both for real, against `BaseOrchestrator` --
the same interface `NativeOrchestrator` and `LangGraphOrchestrator`
already implement.

Verified live by installing both packages into this project's venv and
reading their actual current source (`crewai` 1.15.17, `autogen-agentchat`/
`autogen-core` 0.7.5) rather than assumed from training knowledge -- a
real risk here since both libraries have had significant API churn
(AutoGen in particular has a legacy `pyautogen` line and a current
`autogen-agentchat`/`autogen-core` line with an unrelated API shape;
this integration targets the current one).

### Core design question, resolved by precedent

Does a "CrewAI/AutoGen backend" mean (a) the third-party package handles
*coordination only*, while every actual model call still goes through
Requisite's own `Agent.run()`/`.arun()` (reusing its configured
provider, rate limiter, tool loop), or (b) each translated agent gets
the third-party package's *own* native LLM config (e.g. a LiteLLM model
string), calling out independently? `LangGraphOrchestrator` already
answers this: its node functions call `agent.run(state["input"],
**kwargs)` directly (`requisite/orchestrators/langgraph_orchestrator.py`)
-- langgraph is used purely as a state/control-flow engine, never as an
LLM caller. This ADR follows the same precedent: a custom LLM-adapter
class in each backend proxies every actual model call back to the
underlying Requisite `Agent`, so switching to `workflow.use_crewai()`
never changes *which* provider/model/tools an agent uses -- only who's
coordinating.

Both frameworks expose exactly the extension point this needs, verified
directly from source:

- **CrewAI**: `crewai.llms.base_llm.BaseLLM` -- an ABC with **one**
  abstract method, `call(messages, tools=, available_functions=, ...) ->
  str | Any`, plus a non-abstract `acall(...)` that raises
  `NotImplementedError` unless overridden (implementing it is what makes
  `Crew.akickoff()` real async, not thread-wrapped). `crewai.Agent`
  takes `llm=<BaseLLM instance>` directly.
- **AutoGen**: `autogen_core.models.ChatCompletionClient` -- an ABC with
  8 abstract members: `create`, `create_stream`, `close`,
  `actual_usage`, `total_usage`, `count_tokens`, `remaining_tokens`,
  `model_info` (property). More surface than CrewAI's, but every method
  is mechanical once `create`/`create_stream` proxy to
  `agent.run()`/`.arun()`. `autogen_agentchat.agents.AssistantAgent`
  takes `model_client=<ChatCompletionClient instance>` directly.

Both ABCs are only importable once their package is installed, so both
adapter classes are defined as **local classes inside a lazy-import
factory method** (`CrewAIOrchestrator._require_crewai`,
`AutoGenOrchestrator._require_autogen`) rather than at module scope --
`class _RequisiteLLM(BaseLLM):` can't exist at import time if `BaseLLM`
itself is behind a `try/except ImportError`.

### CrewAI's `"hierarchical"` strategy: a real constraint, not implemented

CrewAI's `Process.hierarchical` depends on its own internal
delegation-tool protocol. Read `crewai.crew.Crew._create_manager_agent`
directly: when a `manager_agent=` is supplied explicitly (our case,
since we build our own manager `crewai.Agent` wrapping a Requisite
Agent), CrewAI does **not** auto-attach its
`AgentTools(agents=...).tools()` delegation tools the way it does for a
bare `manager_llm=` string -- the caller is on the hook for wiring those
up, and the manager's LLM then needs to correctly invoke them through
CrewAI's own `tools=`/`available_functions=` tool-calling protocol,
which this proxy-to-`agent.run()` adapter deliberately bypasses
(Requisite's own `Agent` already runs its own tool loop). Re-exposing
CrewAI's internal delegation tools *through* that would need real
tool-bridging work, not designed or verified here. **CrewAI's backend
ships `"sequential"` only** -- `"hierarchical"` is a documented follow-up.

### AutoGen's `"supervisor"`: reuse, not reinvent

AutoGen's `SelectorGroupChat(participants, model_client=,
selector_func=)` accepts a plain Python `selector_func` callable that
AutoGen calls *directly* to pick the next speaker -- it never goes
through the `ChatCompletionClient` adapter at all, avoiding CrewAI's
tool-bridging problem entirely. So `selector_func` reuses
`NativeOrchestrator._split_coordinator_and_workers`,
`_SupervisorDecision`, `_supervisor_prompt`, and `_resolve_delegate`
directly -- the exact same decision protocol
`LangGraphOrchestrator._build_supervisor_graph` already reuses from
`native.py`, applied to a third execution engine instead of reinvented.
**AutoGen's backend ships `"sequential"` and `"supervisor"`.**

### A real `crewai`/`mcp` dependency conflict, found via `pip install`

`crewai` 1.15.17 hard-pins `mcp~=1.28.1` (confirmed directly:
`importlib.metadata.requires("crewai")` lists it, and `pip install
crewai` into a venv that already had `mcp==2.0.0` installed silently
downgraded it back to `1.28.1`, breaking every MCP test in this same
repo until reinstalled -- a real regression this ADR's own verification
process caught and fixed, not a hypothetical). There is no version range
today that satisfies both `requisite-ai[crewai]` and `requisite-ai[mcp]`
in the same environment. `pyproject.toml`'s `crewai` extra is therefore
**deliberately not included in the `all` extra** -- adding it there
would make `pip install requisite-ai[all]` unresolvable, since `all`
already includes `mcp>=2.0,<3.0`. `autogen-agentchat`/`autogen-core`
have no such conflict and are included in `all` normally.

## Decision

### `requisite/orchestrators/crewai_orchestrator.py`

`_RequisiteLLM(BaseLLM)` (built inside `_require_crewai()`): `call()`/
`acall()` extract the last message's text (whatever CrewAI just
assembled for this call, including any `context=[...]`-injected prior
task output) and proxy it to `self._requisite_agent.run()`/`.arun()`,
collecting each real `AgentResult` for `WorkflowResult.steps`.

`CrewAIOrchestrator.run`/`.arun` (`"sequential"` only): one
`crewai.Agent(role=agent.name, ..., llm=_RequisiteLLM(agent))` per step;
one `crewai.Task` per step, each `context=[previous_task]` -- CrewAI's
own native chaining mechanism (verified via `Task.model_fields`
including `context`), not manual state threading.
`Crew(agents=..., tasks=..., process=Process.sequential,
tracing=False).kickoff()`/`.akickoff()`. `tracing=False` is explicit,
not left at CrewAI's own environment/user-settings-dependent default --
verified live that leaving it unset prints an interactive-looking
"Tracing Preference Saved" banner on every run in a non-tty environment.
Final content from `CrewOutput.raw`.

### `requisite/orchestrators/autogen_orchestrator.py`

`_RequisiteChatCompletionClient(ChatCompletionClient)` (built inside
`_require_autogen()`): `create()` proxies to `agent.arun()`;
`create_stream()` is implemented correctly (yields the content, then the
final `CreateResult`) but never actually exercised in this integration,
since `AssistantAgent` only calls it when constructed with
`model_client_stream=True`, which this orchestrator never sets.
`count_tokens`/`remaining_tokens` use a character-count/4 heuristic --
Requisite's `AI` facade doesn't expose a real tokenizer -- used only for
AutoGen's own internal context-window bookkeeping, never for
billing/limits (Requisite's own `RateLimiter`/provider handle those
against the real API response). `model_info` reports
`function_calling=False` deliberately, matching CrewAI's own
tools-bypassed design.

`agent=None` makes `_RequisiteChatCompletionClient` a **no-op sentinel**
instead of a real proxy -- used for the `"supervisor"` strategy's
finishing turn. `SelectorGroupChat` structurally requires
`selector_func` to name a next speaker before its termination condition
is observed, so finishing still costs one turn even after the
coordinator decides `"finish"`; rather than spend a real Requisite Agent
call (and pollute `WorkflowResult.steps`) on a turn whose content is
discarded, a dedicated `__supervisor_finish__` participant backed by
this no-op client takes that turn for free.

`_DecisionTermination(TerminationCondition)`: terminates once
`mark_finished()` is called externally (by `selector_func`, when the
coordinator's `_SupervisorDecision.action == "finish"`) -- not derived
from AutoGen's own text-mention conventions (`TextMentionTermination`),
since Requisite's supervisor protocol already has its own explicit
finish signal.

`AutoGenOrchestrator.run` wraps `asyncio.run(self.arun(...))` --
confirmed directly that `autogen-agentchat`'s `Team` classes
(`RoundRobinGroupChat`/`SelectorGroupChat`) expose only async
`run()`/`run_stream()`, no sync-native equivalent, the same reason
`MCPClient.discover_tools()`/`Tool.execute()` wrap async internally.

**A found-and-fixed bug, not a design choice**: `autogen-agentchat`'s
own agent runtime catches *any* exception raised from a `selector_func`
call and re-raises it at the `team.run()` call site as a generic
`RuntimeError(str(original_error))` -- losing the original
`AgentException`/`ConfigurationException` type Native and LangGraph's
supervisor both raise for the exact same conditions (max_rounds
exceeded, unknown delegate). Verified live: a plain `raise
AgentException(...)` inside `selector_func` surfaced to the caller as
`RuntimeError: AgentException: ...`, not `AgentException` -- a real
cross-backend exception-contract inconsistency a mocked-only test would
not have caught. Fixed by catching the exception *inside*
`selector_func`, stashing it in shared `state`, terminating the team
cleanly via the same sentinel-routing path used for a normal finish, and
re-raising the *original* exception from `arun()` itself once
`team.run()` returns -- restoring the same exception contract every
backend already gives callers.

**A second found-and-fixed bug**: the transcript passed to
`_supervisor_prompt` was never actually populated in the first draft
(`selector_func` only decides who speaks next; it doesn't automatically
see what a worker said after their turn the way `AI.chat`'s own
tool-loop does). Fixed by having `selector_func` inspect the message it
was just handed (`messages[-1]`) at the start of each call -- if it came
from a real worker, fold `(worker_name, state["pending_task"],
worker_output)` into the transcript before asking the coordinator for
its next decision, so the coordinator's prompt sees delegation history
exactly like Native/LangGraph's supervisor does.

## Alternatives considered

- **Each translated agent gets the third-party package's own native LLM
  config** (e.g. `llm="gemini/gemini-2.0-flash"` via CrewAI's built-in
  LiteLLM support). Rejected -- see "Core design question" above;
  breaks provider-agnosticism for several of Requisite's 8 providers
  whose configuration doesn't map cleanly onto a bare model string
  (Azure OpenAI's endpoint, Groq/OpenRouter/Together's base URLs, local
  Ollama) and abandons Requisite's own rate limiter/tool loop entirely.
- **CrewAI `"hierarchical"` support via a tool-bridging layer** that
  surfaces CrewAI's `AgentTools` delegation tools through Requisite's
  own tool-calling loop. Rejected for v1 -- real, non-trivial design
  work (translating CrewAI's delegation-tool schema into a Requisite
  `Tool`, routing its invocation back through CrewAI's
  `available_functions=` protocol) that wasn't designed or verified
  here; a genuine follow-up, not scope creep avoided for its own sake.
- **AutoGen's own `TextMentionTermination`/free-form group-conversation
  idiom** for `"supervisor"`, instead of an explicit decision protocol.
  Rejected -- would abandon the "supervisor explicitly delegates one
  subtask, worker responds, supervisor decides again" semantics
  Native/LangGraph's supervisor already establishes, replacing it with
  AutoGen's different "agents converse until someone says TERMINATE"
  idiom -- a real behavior change across backends, not just an
  implementation detail.
- **Adding `crewai` to the `all` extra** for install-target parity with
  every other optional backend. Rejected -- see the real `mcp~=1.28.1`
  conflict above; would make `pip install requisite-ai[all]`
  unresolvable.

## Consequences

### Positive

- Closes both remaining orchestration-backend 📋 lines in
  `ROADMAP.md`.
- Both backends verified with real coordination code exercised in
  tests (`pytest.importorskip`, only the wrapped `Agent`'s provider
  faked) and with real Gemini calls during development -- not mocked
  end-to-end.
- Two real bugs (exception-type loss through AutoGen's runtime, missing
  transcript accumulation) were found specifically because real
  coordination code ran against real scripted scenarios, not because
  they were anticipated in the design -- the same pattern this
  project's other recent ADRs (0022, 0025, 0026) have each independently
  hit.

### Negative / risks

- CrewAI's `"hierarchical"` and AutoGen's group-conversation idioms
  aren't available -- an application wanting CrewAI's manager-delegation
  pattern specifically still needs `"sequential"` via this backend, or
  `"supervisor"`/`"hierarchical"` via `"native"`/`"langgraph"`.
- AutoGen's `count_tokens`/`remaining_tokens` approximation (chars/4) is
  not a real tokenizer -- fine for AutoGen's own internal bookkeeping,
  would be wrong if anything else ever depended on it for real token
  accounting.
- `crewai` and `mcp>=2.0` cannot coexist in one environment today,
  purely due to `crewai`'s own dependency pin -- outside Requisite's
  control, but a real friction point for any application wanting both
  `requisite-ai[crewai]` and `requisite-ai[mcp]`.

### Follow-ups

- CrewAI `"hierarchical"` strategy, if a concrete use case needs it --
  requires designing the tool-bridging layer noted above.
- AutoGen streaming mode (`model_client_stream=True`), if a concrete use
  case needs token-level streaming -- `create_stream()` is already
  implemented correctly but unexercised.
- Revisit the `crewai`/`mcp` conflict if a future `crewai` release loosens
  its `mcp` pin.
