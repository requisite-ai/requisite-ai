
# 0013. Hierarchical multi-agent strategy

Status: Accepted
Date: 2026-08-11

## Context

ADR-0011/ADR-0012 flagged hierarchical as needing "materially bigger
structural changes" than critic/consensus/debate/map-reduce -- nested
sub-workflows, unlike anything else shipped. On renewed inspection,
prompted by a direct request to actually design it rather than defer
it again, that assessment was too pessimistic: `Workflow` already has
`.run()`/`.arun()` returning something with `.content`, exactly like
`Agent`. The only real gap was that `Workflow` had no `.name` for the
existing coordinator/worker addressing convention
(`_split_coordinator_and_workers`) to use it as a delegate.

## Decision

### `Workflow` gains an optional `name`

`Workflow.__init__` gained `name: Optional[str] = None`
(`requisite/workflows/workflow.py`) -- backward compatible, every
existing `Workflow()` call is unaffected. Only required when a
`Workflow` is used as a hierarchical delegate; a standalone `Workflow`
never needs one.

### A new `_split_coordinator_and_delegates`, not a change to `_split_coordinator_and_workers`

`requisite/orchestrators/native.py` gained a second split-helper,
duck-typed to accept a mix of `Agent` and `Workflow` in `steps[1:]`,
kept deliberately separate from the existing `_split_coordinator_and_workers`
(still `Agent`-only) so planner/consensus/debate/map-reduce's validation
stays exactly as strict as before -- widening the shared helper would
have let a `Workflow` slip into e.g. `consensus`'s participant list
without ever being intentionally supported there.

Coordinator/delegate validation uses `hasattr`/`getattr` duck-typing,
not `isinstance`: `native.py` cannot import `Workflow` at runtime
without a real circular import (`native.py` -> `workflow.py` ->
`orchestrators/factory.py` -> `native.py`, since `factory.py` registers
`NativeOrchestrator` at module load time). `hasattr(coordinator, "ai")`
distinguishes "has the `.ai` a coordinator needs for
`response_model=`-based routing decisions" without needing to know the
concrete type; `Agent.name` is always a non-`None` str already, so
checking `getattr(delegate, "name", None) is None` for "did someone
forget to name their Workflow delegate" is effectively `Workflow`-specific
without an `isinstance` check either.

### Type-safety at the boundary: `Sequence[Any]`, not `Union[Agent, Workflow]`

Making `Workflow.add()`/`Workflow._steps` accept a `Workflow` too meant
the `steps` type needed to loosen somewhere in the
`Workflow.add()` -> `Workflow.run()` -> `BaseOrchestrator.run()` ->
`NativeOrchestrator.run()`/`LangGraphOrchestrator.run()` chain, or an
example using a `Workflow` delegate wouldn't type-check under
`mypy --strict examples`. Chose `Any` at every boundary point in that
chain (`Workflow.add`, `Workflow._steps`, `Workflow.agents`,
`BaseOrchestrator.run`/`.arun`, `NativeOrchestrator.run`/`.arun`,
`LangGraphOrchestrator.run`/`.arun`) rather than a proper
`Union["Agent", "Workflow"]` threaded through all of them --
`WorkflowResult.steps: list[Any]` already established this exact
"give up type precision at the orchestration boundary for flexibility"
precedent for results; this does the same for inputs. Every internal
`_run_<strategy>`/`_arun_<strategy>` method keeps its existing
`Sequence["Agent"]` typing completely unchanged -- passing an
`Any`-typed value into a more specific parameter type is always valid
for mypy, so none of the other six strategies' signatures needed to
move at all.

### Hierarchical reuses supervisor's round loop via a new shared `_run_delegation_loop` -- a refactor, unlike critic/reflection/supervisor

ADR-0011 explicitly rejected a shared round-loop helper across
reflection/critic/supervisor, since those three genuinely differ in
what they pass to `response_model=`, what terminates them, and what
they accumulate. Hierarchical and supervisor don't differ in any of
those -- same `_SupervisorDecision` model, same "finish" termination,
same `transcript` accumulation. The *only* difference is which
split-helper built the delegate dict. Extracting
`_run_delegation_loop`/`_arun_delegation_loop` (parametrized by
`strategy_name` for the result/error-message label) and having both
`_run_supervisor` and `_run_hierarchical` call it is exactly the
"revisit if a strategy makes the duplication clearly worse than the
abstraction would be" case ADR-0011's own Follow-ups anticipated.

**Regression safety**: this changes already-shipped, already-tested
code. Verified by re-running the existing supervisor tests in
`tests/test_workflows.py` completely unmodified -- they pass, proving
`_run_supervisor`'s public behavior (exception message text,
`WorkflowResult` shape) is byte-identical after the refactor.

### Delegating to a `Workflow` runs whatever strategy it's configured with

`delegate.run(subtask, **kwargs)` / `await delegate.arun(subtask, **kwargs)`
is identical whether `delegate` is an `Agent` or a `Workflow` -- a
`Workflow` delegate runs its own configured strategy (`sequential`,
`supervisor`, even another `hierarchical`), so nesting composes for
free without `_run_delegation_loop` needing to know or care. Reuses
`_supervisor_prompt` unchanged -- from the coordinator's view,
delegating to a sub-team looks identical to delegating to a worker.

## Alternatives considered

- **A proper `Union["Agent", "Workflow"]` threaded through every
  signature in the chain**, instead of `Any`. Rejected -- see above;
  would have touched every one of the other six strategies' method
  signatures for no behavioral benefit, since none of them support
  `Workflow` delegates.
- **`isinstance` checks against real `Agent`/`Workflow` imports**,
  accepting a local/deferred import inside the method body (safe at
  call time, since by then all modules have finished loading) rather
  than `hasattr`/`getattr` duck-typing. Rejected as unnecessary
  complexity -- the duck-typed checks are simpler, need no import at
  all, and give equally clear error messages.
- **Keep `_run_supervisor` and `_run_hierarchical` fully independent**
  (accept the ~30 lines of duplication), matching how critic/reflection/
  supervisor were kept independent in ADR-0011. Rejected specifically
  because -- unlike those three -- hierarchical and supervisor have
  *zero* structural difference beyond the split-helper; the duplication
  here really would be pure copy-paste, not "looks similar but differs
  in ways that would need real parameters to share."

## Consequences

### Positive

- Real recursive hierarchy (a coordinator delegating to a team, which
  itself delegates further) works with zero new execution logic --
  entirely a consequence of `Workflow` already being "delegate-shaped."
- The supervisor/hierarchical duplication that existed as two
  near-identical strategy implementations is gone; future changes to
  the delegation round-loop only need to happen once.

### Negative / risks

- **`**kwargs` forwards flatly to every delegate.** A nested `Workflow`
  delegate can't receive a `max_rounds` different from what the outer
  hierarchical call consumed for itself (the key is already popped out
  of `**kwargs` by the outer strategy's own keyword-only parameter
  before forwarding) -- it falls back to its own strategy's default
  (e.g. 6 for a nested supervisor, 3 for a nested critic/debate). No
  per-delegate kwargs scoping exists. Workaround: bake the desired
  behavior into how the nested `Workflow` itself is constructed, rather
  than trying to parametrize it through the outer call.
- Using `Any` at the orchestration-input boundary means a caller could
  pass a completely unrelated object into `Workflow.add()` and nothing
  would catch it statically -- it would only fail at runtime, inside
  whichever strategy actually tries to call `.run()`/`.name` on it.
  Accepted as consistent with `WorkflowResult.steps: list[Any]`'s
  existing precedent on the output side.

### Follow-ups

- Tree-of-thoughts remains deferred, unscoped -- it's a genuinely
  different shape (branching search with evaluation/pruning, not a
  coordinator/delegate round loop) and needs its own design pass, not
  bundled into this one.
- If `LangGraphOrchestrator` ever wants to support `Workflow` delegates
  too, its `_build_graph`/node-building logic (currently `Agent`-only)
  would need real changes -- not just the signature widening this ADR
  already did to satisfy the `BaseOrchestrator` interface.
