
# 0016. LangGraph backend: branching/conditional graphs

Status: Accepted
Date: 2026-08-11

## Context

`ROADMAP.md`'s multi-agent orchestration section had one remaining line
directly under the shipped linear `langgraph` backend: *"`langgraph`
backend: branching / conditional graphs -- 📋."* ADR-0007 flagged the
concrete gap this left: `LangGraphOrchestrator` hard-rejects any
`strategy != "sequential"` with `ConfigurationException`, so *"a user
who reaches for `workflow.use_langgraph()` after building on
[reflection/planner/supervisor] will hit that error and need to switch
back to `native`."*

Of the native-only strategies, `supervisor` is the one whose *shape* is
genuinely branching: each round, a coordinator makes a structured
decision (`_SupervisorDecision`: delegate to a named worker, or finish)
that determines what runs next, looping until it finishes. That's
exactly what `langgraph`'s `add_conditional_edges` plus a cycle
expresses. `planner` (one static plan, decided once, executed in fixed
order) isn't branching in this sense. `reflection` (single-agent
critique loop) and `hierarchical` (supervisor + `Workflow`-as-delegate)
are real candidates for the same treatment but are left for a later
pass -- matching this project's established one-or-two-strategies-per-ADR
pace (ADR-0007 did 3, ADR-0011/0012 did 2 each, ADR-0013 did 1).

Everything below was verified directly against the installed SDK, not
assumed: `langgraph==1.2.9` (`pip show langgraph`), read straight out of
`.venv/Lib/site-packages/langgraph/`: `StateGraph.add_conditional_edges`
(`graph/state.py`), `START`/`END` (`constants.py`, re-exported from
`graph/__init__.py`), and `DEFAULT_RECURSION_LIMIT` (`_internal/_config.py`,
`= 10007`).

## Decision

### Reuse native's decision logic, don't duplicate it

`requisite/orchestrators/langgraph_orchestrator.py` imports
`NativeOrchestrator`, `_SupervisorDecision`, and `_supervisor_prompt`
from `requisite/orchestrators/native.py` (both already in the same
`orchestrators` subpackage; `native.py` never imports the langgraph
module, so no cycle). `NativeOrchestrator._split_coordinator_and_workers`
and `._resolve_delegate` are called as static methods through the
class. This is a deliberate reuse decision: the same structured-decision
schema and prompt text mean `workflow.supervisor()` makes *identical*
decisions regardless of backend -- only how the decision drives
execution differs (a Python `for` loop vs. a real graph cycle).
Duplicating the prompt/schema would risk the two backends silently
behaving differently under the same strategy name.

`_split_coordinator_and_workers` was an instance method that never
touched `self` -- converted to `@staticmethod` specifically to make this
reuse possible without constructing a throwaway `NativeOrchestrator()`
just to call it. Every existing internal call site
(`self._split_coordinator_and_workers(...)`) is unaffected: a
staticmethod is still callable through an instance. `_resolve_delegate`
was already a `@staticmethod`, needing no change.

### A real conditional graph with a loop-back cycle, not a disguised loop

```python
class _SupervisorGraphState(TypedDict):
    task: str
    transcript: list[tuple[str, str, str]]
    steps: list[Any]
    route: str
    pending_task: str
    output: str
    rounds: int
```

One node per worker, named after `worker.name` directly (`_split_coordinator_and_workers`
already guarantees uniqueness -- no index-suffix disambiguation needed,
unlike the anonymous linear-chain nodes in `_build_graph`). One
`"__supervisor__"` node (dunder-wrapped, matching `START`/`END`'s own
`__start__`/`__end__` convention, so it can never collide with a real
agent name) runs the structured decision each round and writes
`state["route"]` -- either a worker's name, or the sentinel
`"__finish__"`. `add_conditional_edges("__supervisor__", _route,
path_map)` maps every worker name to itself and `"__finish__"` to
`END`. Every worker node gets one unconditional `add_edge(worker_name,
"__supervisor__")` back -- the loop. Entry: `add_edge(START,
"__supervisor__")`.

No `Annotated`/reducer fields were needed on the state: exactly one node
executes per superstep in this graph shape (the router always picks
exactly one next node), so there's never a concurrent write to
reconcile -- unlike a `Send`-based fan-out, which this feature doesn't
use.

**Verified as real branching, not a fixed chain**: a test
(`test_workflow_use_langgraph_supervisor_routes_to_both_workers_then_finishes`)
scripts the coordinator to delegate to two *different* workers across
rounds and asserts both actually ran, in that order. This is only
possible if `add_conditional_edges`'s routing function is genuinely
re-evaluated each time control returns to the supervisor node -- a
graph with a fixed edge wired once at build time could not produce this.

### `max_rounds` enforcement matches native's exactly, and langgraph's own recursion limit isn't a factor

The supervisor node function checks `state["rounds"] >= max_rounds` and
raises `AgentException` with the identical message shape
`_run_delegation_loop` already raises, before making the round's
decision call -- same semantics (`max_rounds` decision attempts total,
not `max_rounds` delegations). Checked, not assumed: `langgraph`
1.2.9's `DEFAULT_RECURSION_LIMIT` is `10007` (each supervisor round is 2
graph steps -- supervisor node, then worker node -- so even an unusually
large `max_rounds` stays far below the point where langgraph's own
generic `GraphRecursionError` could fire before this explicit check
does). No custom `recursion_limit` config was needed.

### `langgraph>=0.2` → `>=1.0`

`pyproject.toml`'s floor predated langgraph's 1.0 API stabilization by
a wide margin and gave no real guarantee about `add_conditional_edges`'s
current `Runnable`-typed `path` parameter or `path_map` behavior at that
floor -- the same category of unbounded-constraint risk
`DEVELOPMENT.md`'s dependency policy already flags (the `mcp` 2.0.0
incident). Unlike that incident this is a floor bump, not a cap:
backward-compatible for anyone already on a modern `langgraph`, and it
makes the constraint honestly reflect what's actually verified (`1.2.9`,
installed and exercised directly for everything in this ADR).

## Alternatives considered

- **`reflection` or `hierarchical` instead of/alongside `supervisor`.**
  Rejected for this pass -- see Context. Both are real, well-scoped
  follow-ups, not dismissed.
- **Index-suffixed worker node names** (`f"{worker.name}_{index}"`,
  matching `_build_graph`'s linear-chain convention). Rejected: worker
  names are already unique (enforced by `_split_coordinator_and_workers`),
  and using the name directly as the node name makes `path_map`'s
  `{name: name for name in workers}` construction trivial -- an index
  suffix would need a second lookup table for no benefit.
- **A custom `recursion_limit` passed to `.invoke()`/`.ainvoke()`** as a
  defensive measure against langgraph's own generic recursion error
  masking the framework's clean `AgentException`. Rejected once the
  installed default (`10007`) was read directly from source -- there's
  no realistic `max_rounds` value that gets anywhere near it.

## Consequences

### Positive

- Closes the langgraph-branching line in `ROADMAP.md`.
- `workflow.supervisor()` now works unmodified across both backends --
  `.use_langgraph()` after building on `supervisor` no longer needs the
  ADR-0007-flagged switch back to `.use_native()`.
- `_split_coordinator_and_workers` becoming a `@staticmethod` is a small,
  free correctness improvement (it never needed `self`) that also
  unblocked the reuse this ADR depends on.
- No changes to `Workflow`, `BaseOrchestrator`, or `OrchestratorRegistry`
  -- confirms the langgraph module's own docstring promise ("build a
  richer graph in `_build_graph`... this class's public surface does
  not need to change") held up in practice, not just in theory.

### Negative / risks

- The `langgraph>=1.0` floor bump could, in principle, break an
  application pinned to an older `langgraph` that was relying on the
  loose `>=0.2` constraint -- judged acceptable since `0.2`-era langgraph
  almost certainly couldn't run this codebase's *existing* linear
  `_build_graph` reliably either (langgraph's own breaking 1.0
  stabilization), so the practical risk is low.
- `reflection` and `hierarchical` remain native-only on langgraph;
  nothing breaks (both still raise the existing, now-slightly-more-specific
  `ConfigurationException`), but the gap ADR-0007 originally flagged
  isn't fully closed, just narrowed to exactly `supervisor`.
- The supervisor graph's state (`_SupervisorGraphState`) is a second,
  separate `TypedDict` from the sequential graph's (`_GraphState`) --
  two state shapes to keep in mind when extending this module further,
  rather than one unified shape all strategies share.

### Follow-ups

- `reflection` on langgraph: single-agent critique loop, naturally a
  2-node cycle (draft/critique-revise) with a conditional exit on the
  `NO_CHANGES_NEEDED` sentinel -- a good next candidate, not scoped here.
- `hierarchical` on langgraph: same shape as `supervisor` but a delegate
  may be a `Workflow` (duck-typed via `hasattr`/`getattr`, per ADR-0013)
  -- would need the graph's worker-node closures to handle both `Agent`
  and `Workflow` delegates, not scoped here.
- `Send`-based fan-out (e.g. a langgraph-native `map_reduce`) is a
  distinct, not-yet-needed shape -- `_SupervisorGraphState`'s lack of
  `Annotated` reducer fields would need revisiting if a future strategy
  actually needs concurrent writes to the same state field.
