# 0035. `planner` strategy on the langgraph backend

Status: Accepted
Date: 2026-08-25

## Context

This closes the LangGraph-parity thread that began with ADR-0032: with
`planner`, all twelve native strategies (`sequential`, `supervisor`,
`hierarchical`, `reflection`, `graph`, `parallel`, `consensus`,
`map_reduce`, `critic`, `debate`, `tree_of_thoughts`, `planner`) now run
on both the `native` and `langgraph` backends. No strategy remains
native-only. This was the last item; there is no further follow-up to
name.

Unlike the four strategies shipped across ADR-0032/0033/0034, `planner`
turns out to be genuinely **simpler**, not harder -- it is neither a
fan-out shape nor a delegation loop. Read directly,
`_run_planner`/`_arun_planner` (`native.py:644-696`) is a two-phase
algorithm: **one** upfront structured-output call
(`planner.ai.chat(_planner_prompt(input, list(workers)),
response_model=_Plan)`) produces an entire ordered plan
(`_Plan.steps: list[_PlanStep]`, each `{agent: str, task: str}`); then
`_validate_plan` (non-empty, every step's `agent` is a known worker)
raises before anything executes if the plan is bad; then the plan's
steps run **sequentially, in order**, not concurrently and not
round-based -- for each step, resolve the named worker, build a prompt
via `_task_prompt_with_context(task, context_notes)` (prior steps'
results folded in as context; an empty list leaves the task unchanged),
run it, accumulate a context note. There is no `max_rounds`, no
coordinator deciding one delegation at a time the way `supervisor`
does. The two static helpers are already `@staticmethod` and directly
reusable, same cross-module pattern as every prior ADR:
`_validate_plan(plan, *, planner_name, workers)` and
`_task_prompt_with_context(task, context_notes)` (`native.py:698-715`).
`_planner_prompt` is a plain module-level function; `_Plan`/`_PlanStep`
(`native.py:66-76`) are the structured-output schema -- all reused
directly, none reimplemented.

## Decision

### A plan node, then a bounded loop-back cycle over the plan's own length

The one genuine wrinkle, and the reason this isn't just a copy of
`sequential`'s linear-chain shape: the plan's step *count* and *which
worker each step uses* are not known until the planner's real LLM call
returns. Unlike every strategy shipped since ADR-0032, this width is
**not** build-time-computable -- there is no formula in terms of
`kwargs` the way `map_reduce`'s item count or `tree_of_thoughts`'s beam
width were (both provably determined by integers passed to `run()`
itself). This doesn't call for `Send` though, and doesn't need any new
mechanism at all: it is the same bounded loop-back cycle shape
`supervisor`/`hierarchical`/`reflection`/`critic` already established,
just with the loop bound (`len(plan)`) read from graph state instead of
closed over as a `max_rounds` constant. One `_PLAN_NODE` (`"__plan__"`)
runs the upfront call and writes the plan + `step_index=0` into state;
one `_EXECUTE_NODE` (`"__execute__"`) runs exactly one plan step per
visit, incrementing `step_index`; a router (`add_conditional_edges`,
2-arg form, same as `reflection`'s) loops back to `_EXECUTE_NODE` while
`step_index < len(plan)`, else `END`.

Both node names are fixed constants, never derived from agent names --
same argument already used for `map_reduce`'s/`tree_of_thoughts`'s
synthetic names -- so no `_reject_reserved_node_names` call is needed
here either.

### State: no reducer field at all

```python
class _PlannerGraphState(TypedDict):
    task: str
    plan: list[Any]  # list[_PlanStep], written once by the plan node
    step_index: int
    context_notes: list[str]
    steps: list[Any]
    output: str
```

Unlike every strategy since ADR-0032, nothing here needs
`Annotated[..., operator.add]` -- exactly one node runs per superstep in
this graph (a plain two-node cycle), the same "no reducer needed"
situation ADR-0016's original delegation graph already relied on before
any fan-out strategy existed in this module.

### Node bodies

```python
def _plan_node(state):
    plan = planner.ai.chat(_planner_prompt(state["task"], list(workers)), response_model=_Plan)
    NativeOrchestrator._validate_plan(plan, planner_name=planner.name, workers=workers)
    return {"plan": list(plan.steps), "step_index": 0}

def _execute_node(state):
    plan_step = state["plan"][state["step_index"]]
    worker = workers[plan_step.agent]
    task_prompt = NativeOrchestrator._task_prompt_with_context(plan_step.task, state["context_notes"])
    result = worker.run(task_prompt, **kwargs)
    return {
        "steps": [*state["steps"], result],
        "context_notes": [*state["context_notes"], f"[{worker.name}] {result.content}"],
        "step_index": state["step_index"] + 1,
        "output": result.content,
    }
```

`_validate_plan` raising happens *inside* `_plan_node`'s body at invoke
time, since the plan is only known then -- propagates through
`.invoke()`/`.ainvoke()` the same clean way every node-body exception in
this module already does (verified again here with an ad-hoc
exception-propagation check, mid-plan, not skipped just because the
shape is simpler than the last four). `START -> _PLAN_NODE ->
_EXECUTE_NODE` are plain edges; no routing is needed after planning,
since an invalid/empty plan already raised inside `_plan_node` before
reaching `_execute_node` at all. `_execute_node`'s own conditional edge
is the only loop-back in this graph.

## Alternatives considered

- **Treating this as a fan-out**, executing every plan step
  concurrently. Rejected -- plan steps execute sequentially with each
  step's prompt folding in every prior step's result as context; running
  them concurrently would change real behavior (later steps would lose
  access to earlier steps' output), not just its graph representation.
- **A static unroll**, the same technique `debate`/`tree_of_thoughts`
  used. Rejected -- unlike those, the plan's length genuinely isn't
  computable from `kwargs` at build time, only from a real LLM call
  result, so a bounded loop reading its own bound from state is the
  correct shape here, not a stylistic alternative to unrolling.

## Consequences

### Positive

- Closes ROADMAP's LangGraph-parity line entirely -- all twelve
  strategies now run on both backends, the same way ADR-0026 and
  ADR-0030 each closed their own respective ROADMAP lines.
- The cheapest strategy addition across this whole line of work: zero
  new reducer channels, zero offset-indexing, zero new fan-out/fan-in
  machinery -- the loop-back cycle shape ADR-0016 established for
  `supervisor` turned out to already cover this case too, just with the
  bound sourced from state instead of a constant.
- New test coverage mirrors all five existing native `planner` tests
  (plan execution, empty plan, unknown worker, agent-count, duplicate
  names) plus a native-vs-langgraph parity test. An adversarial pass
  (mid-plan worker failure, and a 5-step plan reusing one worker
  repeatedly to exercise the loop-back cycle itself) surfaced no bugs in
  the implementation; one check's output looked suspicious at first (a
  deeply nested, growing string) until cross-checked byte-for-byte
  against native with an identical scripted setup and confirmed to be an
  `EchoProvider` artifact (it echoes its entire prompt back, so
  accumulating context recursively re-embeds itself) present
  identically on both backends, not a langgraph-specific bug.

### Negative / risks

- No explicit bound on plan length exists here, matching native exactly
  (native has no `max_rounds`-style guard for `planner` either) --
  LangGraph's own default recursion limit (`10007`, confirmed in
  ADR-0016) is the only backstop against a pathologically long plan, the
  same implicit safety margin native's own call stack would eventually
  hit too. Not a new risk introduced by this ADR, just inherited
  unchanged from the behavior being mirrored.

### Follow-ups

None. This is the last native strategy to gain a `langgraph`
counterpart.
