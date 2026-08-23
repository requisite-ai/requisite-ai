# 0029. `hierarchical` and `graph` strategies on the langgraph backend

Status: Accepted
Date: 2026-08-23

## Context

ADR-0016 and ADR-0028 each explicitly left the same two follow-ups
open: `hierarchical` and `graph` on the langgraph backend. Doing both
together here, in one ADR, is deliberately justified rather than scope
creep -- both turn out to need far less new design work than either
`supervisor` (ADR-0016) or `reflection` (ADR-0028) did, once their
native reference implementations are read closely.

**`hierarchical` reuses `supervisor`'s graph almost entirely.** Read
directly in `requisite/orchestrators/native.py:716-836`:
`NativeOrchestrator._run_hierarchical`/`_arun_hierarchical` call the
*exact same* `_run_delegation_loop`/`_arun_delegation_loop` that
`_run_supervisor`/`_arun_supervisor` already call. The **only**
difference between the two strategies is which split-helper validates
`steps`: `_split_coordinator_and_workers` (Agent-only, strict) for
`supervisor`, vs. `_split_coordinator_and_delegates` (Agent-*or*-named-
`Workflow`, duck-typed via `hasattr`/`getattr` --
`native.py:849-891`) for `hierarchical`. Since a delegate's actual
execution is just `delegate.run(subtask, **kwargs)` /
`await delegate.arun(subtask, **kwargs)`, and both `Agent` and
`Workflow` expose that same call shape, the existing
`_build_supervisor_graph`'s delegate-node closure already works
unchanged for either type -- nothing in it assumed `Agent` specifically.

**`graph` is a mechanical translation, not new decision logic.** Read
directly, `native.py:1319-1451`:
`_index_graph_nodes`/`_validate_graph_edges`/`_resolve_next_graph_node`
are all `@staticmethod`s with **zero LLM/prompt content** -- pure
name-indexing and condition-matching logic (`Workflow.add_edge(from_,
to, condition=...)`'s `condition` is a plain Python callable the
*developer* supplies, evaluated against the previous node's output
text; routing here was never LLM-decided, per ADR-0019). All three are
directly reusable as-is inside langgraph routing functions -- no
reimplementation, the same "reuse native's logic, don't duplicate it"
decision this module has made for every strategy so far
(`_supervisor_prompt` in ADR-0016, `_reflection_*_prompt` in ADR-0028,
now these three static helpers in this ADR).

## Decision

### `hierarchical`: generalize `_build_supervisor_graph`, don't duplicate it

Renamed `_build_supervisor_graph` to `_build_delegation_graph(steps, *,
split_fn, role, max_rounds, **kwargs)`, parameterized by which
split-helper to call and what strategy name to report in error
messages -- mirroring Native's own `_run_delegation_loop` being shared
by both `_run_hierarchical` and `_run_supervisor` rather than
duplicated. Node-building, routing, and the `path_map` shape are
unchanged (they were never actually supervisor-specific beyond
naming -- a coordinator node + finish sentinel is the same shape either
strategy needs); only the module constant was renamed
`_SUPERVISOR_NODE` -> `_COORDINATOR_NODE` for accuracy now that it
serves two strategies, and `_SupervisorGraphState` -> `_DelegationGraphState`
for the same reason (the `Agent`-typed `_make_worker_node` parameter
was also widened to `Any`, since a delegate can now be a `Workflow`).

`run`/`arun` gained a combined `strategy in ("supervisor",
"hierarchical")` branch:

```python
split_fn = (
    NativeOrchestrator._split_coordinator_and_workers
    if strategy == "supervisor"
    else NativeOrchestrator._split_coordinator_and_delegates
)
compiled_graph = self._build_delegation_graph(
    steps, split_fn=split_fn, role=strategy, max_rounds=max_rounds, **kwargs
)
```

not two near-identical branches -- the two strategies differ in
exactly one argument to one shared builder, so writing them as two
copy-pasted `if` blocks would have reintroduced the duplication this
whole ADR exists to avoid.

### `graph`: `_build_arbitrary_graph`, reusing Native's validation/routing helpers directly

New `_ArbitraryGraphState` TypedDict (`input`, `output`, `steps`,
`step_count` -- a fourth distinct state shape in this module, alongside
`_GraphState`/`_DelegationGraphState`/`_ReflectionGraphState`):

```python
def _build_arbitrary_graph(self, steps, *, edges, max_steps, **kwargs):
    StateGraph, START, END = self._require_langgraph()
    if max_steps < 1:
        raise ConfigurationException(f"The 'graph' strategy requires max_steps >= 1. Got {max_steps}.")
    nodes = NativeOrchestrator._index_graph_nodes(steps, role="graph")
    edges_by_source = NativeOrchestrator._validate_graph_edges(edges, nodes)

    def _make_node(name, node):
        def _node_fn(state):
            if state["step_count"] >= max_steps:
                raise AgentException(f"Workflow graph exceeded max_steps={max_steps} without reaching an end.")
            result = node.run(state["input"], **kwargs)
            return {"input": result.content, "output": result.content,
                    "steps": [*state["steps"], result], "step_count": state["step_count"] + 1}
        return _node_fn

    def _make_router(name):
        def _router(state):
            next_name = NativeOrchestrator._resolve_next_graph_node(name, state["output"], edges_by_source)
            return next_name if next_name is not None else END
        return _router

    graph = StateGraph(_ArbitraryGraphState)
    for name, node in nodes.items():
        graph.add_node(name, _make_node(name, node))
        graph.add_conditional_edges(name, _make_router(name))
    graph.add_edge(START, steps[0].name)
    return graph.compile()
```

Every validation error (`unknown source node`, `unknown target node`,
`unique node names`, unnamed-`Workflow`-node) and every routing
decision (`condition is None or condition(output)`, no-match ->
`AgentException`, no-outgoing-edges -> implicit termination) comes
from these three reused static methods -- this module adds zero new
graph-shape logic, only the langgraph node/router glue around it.

### `max_steps` is checked inside the node function, before running it -- not inside the router

Deliberately mirrors `_build_delegation_graph`'s `_coordinator_node`
check-then-raise pattern (an already-shipped, already-verified
approach in this same module) rather than raising from inside an
`add_conditional_edges` routing callback, whose exception-propagation
behavior under langgraph hasn't been exercised anywhere in this
codebase. Verified by hand that this reproduces Native's own `for _ in
range(max_steps)` loop semantics exactly: up to `max_steps` node
executions are always permitted; the raise fires only when a
`(max_steps + 1)`-th execution is attempted because a prior step's
routing still needed to continue.

`run`/`arun` gained a `strategy == "graph"` branch: `edges =
kwargs.pop("edges", ())`, `max_steps = kwargs.pop("max_steps", 25)`,
build, invoke with `{"input": input, "output": "", "steps": [],
"step_count": 0}`, wrap `final_state["output"]`/`final_state["steps"]`
into `WorkflowResult` -- the same shape `Workflow.graph()`'s native
path already produces.

## Alternatives considered

- **Two separate builder methods for `supervisor` and `hierarchical`**
  (`_build_supervisor_graph` kept as-is, a near-identical
  `_build_hierarchical_graph` added alongside it). Rejected -- the two
  methods would differ by exactly one line (which split-helper is
  called), which is precisely the duplication `NativeOrchestrator`
  itself already avoided by sharing `_run_delegation_loop`; keeping
  langgraph's implementation split would silently diverge from that
  established precedent for no reason.
- **A single dispatch `if strategy == "hierarchical": ... elif strategy
  == "supervisor": ...` pair of near-identical blocks in `run`/`arun`**,
  instead of one combined `strategy in (...)` branch computing
  `split_fn`. Rejected for the same duplication reason, on the dispatch
  side this time rather than the builder side.
- **`graph` reimplementing its own node-indexing/edge-validation/
  routing logic**, independent of Native's. Rejected -- per Context,
  that logic is pure Python with zero LLM/backend-specific content;
  reimplementing it would risk the two backends silently drifting on
  edge cases (e.g. what exactly counts as "no outgoing edges" vs. "no
  matching condition") that `native.py`'s existing tests already pin
  down.

## Consequences

### Positive

- Closes `ROADMAP.md`'s last open langgraph-branching line entirely --
  `sequential`/`supervisor`/`reflection`/`hierarchical`/`graph` are all
  now shipped on both `native` and `langgraph` backends.
- `workflow.hierarchical()` and `workflow.graph()` now work unmodified
  across both backends, the same win ADR-0016/ADR-0028 already
  delivered for `supervisor`/`reflection`.
- `hierarchical` verified against a real Gemini call with a genuinely
  nested `Workflow` delegate (not just an `Agent`) on the langgraph
  backend -- coordinator delegated to a named sub-`Workflow`, which ran
  its own real sequential pipeline and returned a `WorkflowResult` back
  into the parent graph's state, then the coordinator delegated again
  to a plain `Agent` before finishing. Confirms the duck-typed
  delegate-node closure genuinely handles both step types on langgraph,
  not just in the scripted unit tests.
- `graph`'s real-branch behavior (a router's own live model output
  driving which of two downstream nodes actually runs) was verified
  against a real Gemini call up through the routing decision itself
  before a free-tier rate limit interrupted the run; the full round
  trip is additionally covered by the scripted test suite
  (`test_workflow_use_langgraph_graph_conditional_branch_takes_yes_edge`/
  `..._takes_no_edge`), which exercises the identical
  `add_conditional_edges` code path deterministically.

### Negative / risks

- A fourth distinct `TypedDict` state shape in this module
  (`_ArbitraryGraphState`, alongside `_GraphState`/
  `_DelegationGraphState`/`_ReflectionGraphState`) -- same trade-off
  ADR-0016/ADR-0028 already accepted for their own additional shapes:
  more state shapes to keep in mind when extending this module
  further, rather than one unified shape every strategy shares.
- `_build_delegation_graph`'s `split_fn` parameter is a bare
  `Callable[..., tuple["Agent", dict[str, Any]]]` -- callers must pass
  one of exactly two known-compatible static methods; nothing enforces
  that a third, incompatible split-helper couldn't be passed by
  mistake in a future edit. Accepted since both current call sites are
  in this same module and neither is part of the public API.

### Follow-ups

- None scoped by this ADR. `planner`/`critic`/`consensus`/`debate`/
  `map_reduce`/`tree_of_thoughts` remain native-only by design --
  genuinely different shapes from a coordinator/worker or
  developer-declared graph, not a gap this line of work was meant to
  close.
