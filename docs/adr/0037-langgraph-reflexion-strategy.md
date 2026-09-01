
# 0037. LangGraph backend: reflexion strategy

Status: Accepted
Date: 2026-09-01

## Context

ADR-0036 shipped `reflexion` (attempt, evaluate via a pluggable
`Evaluator`, reflect on failure, retry -- up to `max_trials` trials)
on the `native` backend only, deferring langgraph parity as an explicit
follow-up. Every other strategy in this repo already runs on both
`native` and `langgraph`; this closes that one remaining gap the same
way every prior parity round did -- its own dedicated ADR, not a
retroactive edit of ADR-0036, which stays an accurate point-in-time
record of what was decided when reflexion was introduced.

## Decision

### The same bounded loop-back cycle shape, not a new mechanism

`reflexion`'s native loop (`_run_reflexion`, `native.py`) is
structurally the same shape `reflection`/`critic` already have on
langgraph (ADR-0028) and `planner` most recently added (ADR-0035): a
node that produces something, a node that decides whether to continue,
and a conditional edge looping back. `max_trials` is a caller-supplied
bound known at graph-build time -- unlike planner's plan length, which
only exists after a real LLM call returns -- so this is closed over
exactly like `reflection`'s `max_rounds`, not read from state like
planner's loop bound.

### State (`_ReflexionGraphState`, no reducer field)

```python
class _ReflexionGraphState(TypedDict):
    task: str
    attempt: str
    feedback: str
    reflections: list[str]
    trial: int
    steps: list[Any]
    succeeded: bool
```

No reducer needed -- exactly one node runs per superstep, the same
situation `_ReflectionGraphState`/`_PlannerGraphState` already rely on.

### Three nodes, one always-taken edge, one conditional edge

`_ATTEMPT_NODE` (`worker.run(...)` on the task, reflections folded in
via `NativeOrchestrator._task_prompt_with_context`, reused verbatim --
no new prompt-assembly logic) `->` `_EVALUATE_NODE` (the caller's
`evaluator(task, attempt)`, or, when none is given,
`worker.ai.chat(..., response_model=EvaluationResult)`, the same
default-evaluator prompt native already defines) `->` conditionally
either `END` (success, or `max_trials` exhausted) or `_REFLECT_NODE`
(writes a lesson, appends to `reflections`) `->` back to
`_ATTEMPT_NODE`.

The one deliberate difference from `_build_reflection_graph`: its
`_draft_node -> _critique_node` edge is *conditional*, skipping
critique entirely when `max_rounds <= 1` (reflection doesn't need to
know a formal "succeeded" for that case). Reflexion's
`_attempt_node -> _evaluate_node` edge is a **plain, unconditional**
edge -- every attempt must be evaluated to populate `succeeded`, even
for a single-trial run, since `WorkflowResult.succeeded` is exactly the
signal reflexion exists to report.

`trial` is incremented inside `_attempt_node` (a 1-indexed count of
attempts made so far, not a 0-indexed loop variable like native's
`for trial in range(max_trials)`). The routing check
`state["trial"] < max_trials` after a failed evaluation reproduces
native's `trial < max_trials - 1` exactly under this different
indexing -- verified directly: `max_trials=3`, attempt 1 fails
(`1 < 3`, reflects), attempt 2 fails (`2 < 3`, reflects), attempt 3
fails (`3 < 3` is false, ends) -- no reflection after the final
attempt, matching native's own "don't waste a call on an answer
nothing will read" behavior byte-for-byte.

### `run()`/`arun()` dispatch and `WorkflowResult.succeeded`

A new `if strategy == "reflexion":` block, not folded into the
`("reflection", "critic")` tuple -- different state shape, and
`succeeded=` is genuinely new here (confirmed: no other
`WorkflowResult(...)` call anywhere in `langgraph_orchestrator.py` sets
it). `max_trials` and `evaluator` are popped from `kwargs` before
building the graph, the same way `max_rounds` is popped for
reflection/critic -- `evaluator` specifically cannot be left in
`**kwargs`, since that dict is passed straight through to
`worker.run(...)`/`agent.ai.chat(...)` inside the graph's nodes, and a
Python callable isn't a valid provider-call kwarg.

### No `_reject_reserved_node_names` call

Confirmed against the file's own stated rule (already applied
identically to `tree_of_thoughts`'s synthetic `__tot_L{level}_{i}__`
names): the check exists only for strategies that call
`graph.add_node()` with a name taken from user input (supervisor/
hierarchical's delegates, consensus's named participants). Reflexion,
like reflection/critic/planner, has exactly one anonymous worker
(`steps[0]`, validated by count, never by name) and always registers
fixed synthetic constants (`_ATTEMPT_NODE`/`_EVALUATE_NODE`/
`_REFLECT_NODE`) -- no user-supplied string ever reaches
`graph.add_node()`, so there is nothing for the collision check to
guard against.

## Alternatives considered

- **Folding reflexion into `_build_reflection_graph`'s existing
  `role` parameter** (a third `role="reflexion"` branch alongside
  `"reflection"`/`"critic"`). Rejected -- the state shapes genuinely
  differ (`draft`/`critique`/`rounds` vs. `attempt`/`feedback`/
  `reflections`/`trial`/`succeeded`), the loop-back condition differs
  (a caller-pluggable evaluator's boolean vs. a fixed `NO_CHANGES_NEEDED`
  string sentinel), and the "always evaluate" vs. "conditionally skip"
  first edge differs -- sharing one method would need enough
  strategy-specific branching inside every node to not actually save
  the complexity a separate, focused method avoids.
- **Reading `max_trials` from state instead of closing over it**,
  matching planner's pattern. Rejected -- `max_trials`, unlike a plan's
  length, is known before the first call (it's a caller-supplied
  argument, not something only discoverable via a real LLM response),
  so there's no reason to route it through state at all; `reflection`'s
  own `max_rounds`-as-closure precedent is the structurally correct
  match, not planner's runtime-discovered-bound pattern.

## Consequences

### Positive

- Closes ADR-0036's only follow-up -- all thirteen multi-agent
  strategies now run on both the `native` and `langgraph` backends.
- No changes outside `orchestrators/langgraph_orchestrator.py` -- every
  native-side prompt/model (`_reflexion_default_evaluation_prompt`,
  `_reflexion_reflect_prompt`, `EvaluationResult`, `Evaluator`,
  `NativeOrchestrator._task_prompt_with_context`) is reused verbatim,
  not reimplemented.
- Adversarially tested before the permanent suite was written (7
  checks): an exception inside a custom evaluator propagates cleanly
  through the compiled graph rather than being swallowed; an exception
  raised by the worker's own provider *during the reflect node
  specifically* also propagates cleanly; native and langgraph produce
  byte-identical content, `succeeded`, and per-step agent-name sequences
  against the same scripted input, both for a mid-run success and for
  exhausting `max_trials` with no dangling reflection after the last
  attempt; the one-agent requirement raises cleanly; the default
  structured-output evaluator path works; the async path works.

### Negative / risks

- Same token-budget caveat ADR-0036 already accepted for the native
  side: no automatic truncation of accumulated `reflections` for a
  large `max_trials`. Unchanged by this ADR, since the prompt-assembly
  logic (`_task_prompt_with_context`) is reused verbatim from native,
  not reimplemented with different limits.

### Follow-ups

None. This was reflexion's only remaining gap.
