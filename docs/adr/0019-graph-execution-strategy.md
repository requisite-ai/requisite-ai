
# 0019. General graph execution strategy

Status: Accepted
Date: 2026-08-19

## Context

With tree-of-thoughts shipped (ADR-0018), *"General graph execution
(arbitrary DAGs, not just linear/parallel)"* is the one remaining 📋 line
in `ROADMAP.md`'s "Agents & multi-agent orchestration" section — ADR-0018's
own follow-ups named it explicitly as "a separate, broader feature, not a
variant of this one."

Every strategy shipped so far is either a fixed pipeline
(`sequential`/`parallel`) or a flat coordinator/worker shape where *an LLM*
decides routing at run time (`supervisor`, `hierarchical`, `planner`,
`debate`, `consensus`, `map_reduce`, `tree_of_thoughts` all call
`agent.ai.chat(response_model=...)` to pick the next step). ADR-0007's own
follow-up section is the clearest signal that this next strategy doesn't
fit that mold:

> "`debate`, `critic`, `consensus`, `hierarchical`, `map-reduce`,
> `tree-of-thoughts`, and general graph execution remain 📋 ... each
> should decide explicitly whether it fits the coordinator/worker
> convention established here or needs its own shape."

A DAG is nodes + edges, not one coordinator routing to flat named workers
— it needs its own shape. `FEATURES.md` already tracks this as 🚧 rather
than 📋, because the underlying machinery partially exists:
`LangGraphOrchestrator._build_supervisor_graph` (ADR-0016) proved
`StateGraph`/`add_conditional_edges`/loop-back cycles work in this
codebase, but only as one hardcoded shape. Nothing before this let a
caller declare arbitrary nodes/edges themselves.

## Decision

### Edges are explicit and developer-declared, not LLM-decided

This is the core design call, and the direct answer to ADR-0007's
open question. Every prior strategy's routing is a *run-time* decision an
agent makes (`_SupervisorDecision`, `_Plan`, etc.). A `"graph"` strategy
instead treats the graph's shape as fixed at *build* time:
`Workflow.add_edge(from_, to, *, condition=None)` registers an edge
between two node names ahead of `.run()`; `condition` is a plain Python
callable over the source node's output content, not a second LLM call.
This is what makes it "arbitrary" in the DAG sense — the developer wires
the control flow, the same way they'd wire a state machine, rather than
delegating that decision to a coordinator agent every strategy above
already covers well.

### Nodes are peers, addressed by name — no coordinator/worker split

Unlike `_split_coordinator_and_workers`/`_split_coordinator_and_delegates`,
there's no `steps[0]` given a different role than the rest. The new
`NativeOrchestrator._index_graph_nodes(steps, role="graph")` just
name-addresses every step uniformly (`dict[str, Any]`), reusing the same
duck-typed `getattr(node, "name", None)` validation
`_split_coordinator_and_delegates` established for letting a node be
either an `Agent` or a named `Workflow` ("team") — same reasoning: a
graph node calling into a nested team is a natural recursive case, and
`native.py` can't import either concrete type at runtime without
reintroducing the `native.py` → `workflow.py` → `factory.py` → `native.py`
cycle hierarchical's helper already had to dodge.

### Entry point = `steps[0]`; termination = `END` or no outgoing edges

The first node added is where the walk starts, matching
`LangGraphOrchestrator._build_graph`'s existing convention for
`"sequential"`. A node reaches `END` (a sentinel now defined in
`requisite/orchestrators/base.py` — see below) via an edge whose
condition matched, **or** terminates implicitly if it has no outgoing
edges at all — a leaf node doesn't need to be wired to `END` explicitly,
matching sequential's "last agent's output is the result" ergonomic.
Multiple edges from one node are evaluated in `add_edge` call order;
first match wins, so an unconditional edge (`condition=None`) added last
acts as a fallback/default. A node with outgoing edges but no match (no
unconditional fallback, no condition satisfied) raises `AgentException`
— the developer wired an incomplete graph, and failing loudly beats
silently stopping, consistent with every other strategy's "invalid state"
handling (`_resolve_delegate`'s unknown-worker error,
`_validate_plan`'s unknown-worker error).

### Cycles are allowed, bounded by `max_steps` (default 25)

"Arbitrary DAG" in the ROADMAP sense means "not just linear/parallel,"
not a strict mathematical-DAG-only constraint — the same way `supervisor`
already loops (a cycle) bounded by `max_rounds`. A self-correcting node
(loop back to itself until some condition holds) is a real, useful
pattern this strategy should support, so cycles are permitted and guarded
by a new `max_steps` kwarg, raising `AgentException` on exhaustion —
same shape as `_run_delegation_loop`'s `max_rounds` guard, renamed
because "step" (one node execution) is the more accurate unit here than
"round" (one coordinator decision).

### `END` lives in `orchestrators/base.py`, not `workflows/workflow.py`

`END` needs to be importable from both `Workflow` (the public
`add_edge(..., to=END)` call site) and `NativeOrchestrator` (to recognize
a terminating edge target during validation/execution). Defining it in
`workflow.py` and importing it into `native.py` would reintroduce exactly
the circular import every duck-typed helper in this file already avoids.
`orchestrators/base.py` is the one module both already import from
(`WorkflowResult`), so `END = "__end__"` lives there and both
`Workflow` and `NativeOrchestrator` import it from that single source of
truth; `Workflow` re-exports it (`workflow.py`'s new `__all__ = ["END",
"Workflow"]`) and it's re-exported again from top-level `requisite`
alongside `Workflow`/`WorkflowResult`.

### `edges=` reaches the orchestrator only for `strategy == "graph"`

`Workflow.run()`/`.arun()` forward `**kwargs` straight through to
`orchestrator_instance.run(self._steps, input, strategy=..., **kwargs)`,
and every native strategy's `**kwargs` catch-all flows onward into
`agent.run(current_input, **kwargs)` calls (this is how `max_rounds=`,
`map_items=`, etc. stay strategy-scoped without `Workflow` needing to
know which strategy consumes what). Always forwarding
`self._edges` unconditionally would leak an unexpected `edges=[]` kwarg
into every other strategy's `agent.run(...)` calls. Instead, `Workflow`
gates it: `if self._strategy == "graph": kwargs.setdefault("edges",
self._edges)` — a small, explicit special case rather than a generic
mechanism, since no other strategy has needed structural (non-per-call)
data threaded through before.

### Native-only; langgraph deferred

Same "native first" pattern every strategy has followed (supervisor
shipped native in ADR-0007, langgraph nine ADRs later in ADR-0016;
tree-of-thoughts native-only per ADR-0018). A generic langgraph
graph-builder is a materially bigger change than this feature's scope:
ADR-0016's own follow-ups already flagged that two parallel branches
writing to the same state key would need `Annotated` reducer fields on
the `TypedDict` state, a gap nothing in this codebase has needed to close
yet (`_build_graph`/`_build_supervisor_graph` each have their own
non-reducer state shape). Not attempted here.

## Alternatives considered

- **LLM-decided routing, like `supervisor`** (a coordinator agent picks
  the next node each step via `response_model=`). Rejected: that's
  exactly what `supervisor`/`hierarchical` already do well; a `"graph"`
  strategy that just re-implements supervisor with a different name
  wouldn't close the actual gap ADR-0007 identified — a way to declare
  *fixed, deterministic* control flow ahead of time.
- **Nodes passed as objects to `add_edge`** (`add_edge(researcher,
  writer)`) instead of by name string. Rejected for consistency: every
  other strategy addresses workers/delegates by `.name` string
  (`_SupervisorDecision.worker`, `_Plan`'s `agent` field) — introducing
  a second addressing convention just for this strategy would be
  surprising, not simpler.
- **True parallel fan-out/join** (multiple edges converging on one node,
  merging several predecessors' outputs). Rejected for this pass: a real
  join needs explicit state-merge semantics this codebase has never
  needed (`"parallel"` already covers simple same-input fan-out without a
  merge step), and it's the same `Annotated`-reducer-shaped problem noted
  above for the langgraph deferral, just on the native side too. Noted as
  a follow-up.
- **Raising `ConfigurationException` instead of `AgentException` when a
  node's output matches no outgoing edge.** Rejected: this failure
  happens *during* execution, dependent on live agent output, not at
  workflow-construction/validation time — matching the
  `ConfigurationException` (bad wiring, known before running) vs.
  `AgentException` (bad outcome, discovered while running) split already
  used throughout `native.py` (e.g. unknown edge endpoints raise
  `ConfigurationException`; `max_rounds`/now `max_steps` exhaustion raises
  `AgentException`).

## Consequences

### Positive

- Closes the last 📋 line in `ROADMAP.md`'s "Agents & multi-agent
  orchestration" section.
- Genuinely additive: `_KNOWN_STRATEGIES`, `_STRATEGIES`, and one new
  `if strategy == "graph"` branch each in `run()`/`arun()` — no existing
  strategy changed, no change to `BaseOrchestrator`, `OrchestratorRegistry`,
  or `WorkflowResult`.
- Reuses established conventions throughout: name-based addressing,
  duck-typed `Agent`/`Workflow` nodes (`hierarchical`'s pattern), the
  `**kwargs`-popped-by-signature mechanism (`max_rounds`/`map_items`'s
  pattern), and the `ConfigurationException` vs. `AgentException` split.
- `Workflow.add_edge` is the first genuinely reusable "declare structural
  data ahead of `.run()`" API on `Workflow` beyond `.add()` — but it's a
  small, additive method, not a change to any existing one.

### Negative / risks

- `Workflow.add_edge` is a materially bigger public-API surface addition
  than any prior strategy needed (every strategy before this reused
  `.add()` unchanged) — a deliberate, documented deviation, not an
  oversight.
- No parallel fan-out/join within a graph (see Alternatives) — a
  same-input branch that needs to converge and merge isn't expressible
  yet; only one node executes at a time.
- A condition callable only sees the source node's output content
  (`str`), not the full accumulated history — sufficient for every case
  in this ADR's own example (sentinel-prefixed routing, matching the
  `NO_CHANGES_NEEDED` precedent), but a condition needing earlier context
  would have to be threaded through the node's own output text.
- Cycles plus arbitrary Python conditions mean a badly wired graph can
  loop right up to `max_steps` before failing loudly — same accepted
  trade-off `supervisor`/`hierarchical` already made with `max_rounds`.

### Follow-ups

- `LangGraphOrchestrator` support for `"graph"` — needs `Annotated`
  reducer fields on graph state first (see Decision above); not scoped
  here.
- Parallel fan-out/join within a graph (concurrent multi-node execution
  with a merge step) — a separate, harder problem than branching/cycles;
  not scoped here.
- Condition callables that see more than the immediately-preceding node's
  output (e.g. the full `results` list so far) — not needed by any case
  motivating this ADR; add if a real use case needs it.
