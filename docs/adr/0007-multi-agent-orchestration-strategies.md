
# 0007. Multi-agent orchestration strategies: reflection, planner, supervisor

Status: Accepted
Date: 2026-08-07

## Context

`ROADMAP.md` and `FEATURES.md` both listed multi-agent orchestration as
the largest gap against the original project vision: `Workflow` and
`NativeOrchestrator` only implemented `sequential` and `parallel`.
`FEATURES.md`'s Agentic Mode table was explicit that "model decides
which agent to delegate to" is blocked on exactly this: "requires a
Supervisor/Planner strategy."

Both `NativeOrchestrator` and `Workflow` already documented the intended
extension point before any of these three strategies existed: add a
`_run_<strategy>` / `_arun_<strategy>` pair to `NativeOrchestrator` and
register the name; nothing else in the framework needs to change. This
ADR follows that pre-existing convention rather than introducing a new
one, and in turn becomes the reference for the *next* strategy to be
added (debate, critic, consensus, hierarchical, map-reduce,
tree-of-thoughts, general graph execution — all still 📋).

Three strategies were in scope for this pass: `reflection` (single agent
critiques and revises its own output), `planner` (one agent decomposes a
task into a plan for others to execute), and `supervisor` (one agent
delegates subtasks to others one at a time, deciding when to stop).
Each needed answers to the same two questions: how does a coordinating
agent address other agents, and how does it make a structured decision
(a plan, a delegation, a stop condition) rather than free-text the
framework would have to parse unreliably.

## Decision

### Coordinator + workers, addressed by `agent.name`

For `planner` and `supervisor`, `steps[0]` (the first agent added to the
`Workflow`) is the coordinator; `steps[1:]` are workers, addressed by
their `Agent.name`. This requires `len(steps) >= 2` and unique worker
names, both enforced with an actionable `ConfigurationException` before
any model call is made.

No new field was added to `Agent` for this (e.g. a `description`) —
routing/planning prompts list workers by name only, the same signal
`_run_parallel`'s output labeling (`f"[{r.agent_name}]"`) already relies
on. Keeping `Agent`'s constructor unchanged matches the stated design
goal ("nothing else needs to change") and avoids adding public API
surface before there's a demonstrated need for richer per-agent routing
metadata.

`reflection` has no coordinator/worker split — it requires exactly one
agent (`len(steps) == 1`), which critiques and revises its own output.
This matches `ROADMAP.md`'s literal wording ("agent critiques and
revises **its own** output") rather than a two-agent generator/critic
design, which is closer to the still-unbuilt `debate`/`critic`
strategies and was deliberately left for those.

### Structured decisions via `AI.chat(response_model=...)`, not a tool-loop

A coordinator's decision (a plan, a delegate/finish choice) is a single
`coordinator.ai.chat(prompt, response_model=X)` call, which already
returns the validated Pydantic instance directly
(`requisite/ai.py`'s `chat()`, existing behavior, unchanged). It is
**not** run through `Agent.run()`'s tool-calling loop.

This is a deliberate scope cut: coordinators route/plan, they don't use
tools while deciding. Workers are invoked normally via `worker.run(...)`
(the full tool-calling loop, with whatever tools/skills/capabilities
that worker was configured with) — only the coordinator's own decision
step bypasses it. If a real use case needs a coordinator that also calls
tools while deciding, that's new scope for a future ADR, not something
this one tries to anticipate.

The three structured-decision models (`_PlanStep`, `_Plan`,
`_SupervisorDecision`) are private to `requisite/orchestrators/native.py`
— no new public module, no framework-wide "decision model" concept.
Response validation (Pydantic) plus an explicit worker-name check
against the known worker dict is what catches a coordinator naming a
worker that doesn't exist; the error lists the available workers,
matching the actionable-error convention already used elsewhere (e.g.
`MCPClient.register_as_capability`'s "Available: [...]" message).

### `reflection`'s early-stop signal

The worker is asked to respond with exactly `NO_CHANGES_NEEDED` (checked
via `.strip() == "NO_CHANGES_NEEDED"` on the critique turn) when it has
nothing left to fix, stopping the loop before `max_rounds` is reached.
This is a plain string sentinel, not structured output — critique text
is naturally free-form, and forcing it through a Pydantic model just to
carry one boolean would add a schema for no real benefit over checking
the one sentinel value the framework itself asks for.

### `max_rounds`, not `max_iterations`

`reflection` (default `max_rounds=3`) and `supervisor` (default
`max_rounds=6`) both take a `max_rounds` keyword-only parameter, popped
by the method signature before `**kwargs` is forwarded to `agent.run(...)`
— this keeps it from leaking into the provider's `chat()` call the way
an unexpected kwarg would. A different name from `Agent.max_iterations`
was chosen deliberately: `max_iterations` is a per-agent tool-calling
budget; `max_rounds` is a per-workflow coordination budget spanning
possibly many underlying `Agent.run()` calls, each with its own
`max_iterations`. Conflating the names would suggest they're the same
knob when they aren't.

`supervisor` exhausting `max_rounds` without reaching `"finish"` raises
the existing `AgentException` (message scoped to "Workflow supervisor",
not a new exception type) — it's the same shape as `Agent`'s own
`max_iterations` exhaustion, just one level up, and doesn't warrant a
dedicated exception class for a single narrow case.

### `WorkflowResult.steps` holds worker results only

For `planner` and `supervisor`, `WorkflowResult.steps` contains one
`AgentResult` per executed *worker* call, not the coordinator's own
decision calls (those return a parsed Pydantic model via `AI.chat`, not
an `AgentResult` — wrapping them as one would strain
`WorkflowResult.steps`'s existing docstring, "the per-agent `AgentResult`
objects produced along the way"). `reflection`'s `steps` includes every
call the single worker made (draft, critiques, revisions) — all genuine
`AgentResult`s, so no such mismatch exists there.

## Alternatives considered

- **A dedicated `Agent(description=...)` field for routing context.**
  Rejected for this pass: `agent.name` alone was sufficient for every
  test and example built against this, and adding constructor surface to
  the most-used class in the framework is exactly the kind of change
  that should wait for a demonstrated need, not be added speculatively
  ahead of one.
- **Coordinator decisions routed through the normal tool-calling loop
  (`Agent.run()`) instead of a direct structured `AI.chat()` call.**
  Rejected: it would let a coordinator use tools while deciding, but
  adds real complexity (the loop's tool-result bookkeeping, `max_iterations`
  interacting with `max_rounds`) for a capability nothing in scope
  needed. Revisit if a real use case needs it.
- **A `steps[0]` vs. a dedicated `coordinator=` constructor argument on
  a hypothetical `Workflow.planner(coordinator=...)`.** Rejected in favor
  of reusing the existing `steps[0]`/`steps[1:]` ordering that `.add()`
  already establishes — no new argument shape, consistent with how
  `sequential`/`parallel` already just use step order.
- **A two-agent generator/critic design for `reflection`.** Rejected —
  see "Coordinator + workers" above; that shape is closer to the
  still-unbuilt `debate`/`critic` strategies and was left for those.

## Consequences

### Positive

- All three strategies are exercised end-to-end against a live Gemini
  model via `examples/workflow_example.py`, not just scripted fakes.
- No changes were needed to `Agent`, `BaseOrchestrator`, or
  `LangGraphOrchestrator` — confirms the "nothing else needs to change"
  design goal these strategies were documented against actually holds
  once real strategies were built against it.
- The coordinator/worker convention and the "structured decision via
  `AI.chat`, not a tool-loop" pattern are now concrete precedent for
  `debate`, `critic`, `consensus`, and the other still-unbuilt
  strategies to follow or explicitly deviate from.

### Negative / risks

- `reflection`, `planner`, and `supervisor` are implemented only on the
  `native` orchestrator. `LangGraphOrchestrator` already raises a clear
  `ConfigurationException` for any non-`"sequential"` strategy, so
  nothing breaks, but a user who reaches for `workflow.use_langgraph()`
  after building on one of these three will hit that error and need to
  switch back to `native` (or wait for the LangGraph backend to grow
  matching graph shapes).
- A coordinator's structured decision depends on the underlying
  provider's structured-output support being reliable for the chosen
  schema; a provider that struggles with `Literal["delegate", "finish"]`
  or nested list schemas will produce worse routing/planning than one
  that doesn't. This is an existing, general structured-output
  limitation (not new to this ADR), but it's more exposed here since
  the whole strategy depends on it working well every round.
- `supervisor`'s transcript (delegation history fed back into the
  routing prompt each round) grows unbounded within a single run — for
  `max_rounds` values much larger than the default 6, prompt size could
  become a real cost/latency concern. Not addressed here; a future
  change could summarize or truncate the transcript the same way
  `SummarizingPolicy`/`MessageCountPolicy` already do for conversation
  memory.

### Follow-ups

- If a real use case needs a coordinator that can call tools while
  deciding, revisit the "structured decision via `AI.chat`, not a
  tool-loop" decision above.
- If per-agent routing quality becomes a problem with name-only context,
  revisit adding an optional `Agent(description=...)` field — deliberately
  not added speculatively in this pass.
- `debate`, `critic`, `consensus`, `hierarchical`, `map-reduce`,
  `tree-of-thoughts`, and general graph execution remain 📋 on
  `ROADMAP.md` — each should decide explicitly whether it fits the
  coordinator/worker convention established here or needs its own shape.
