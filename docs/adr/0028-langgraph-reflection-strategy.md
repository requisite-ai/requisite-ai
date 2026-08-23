# 0028. `reflection` strategy on the langgraph backend

Status: Accepted
Date: 2026-08-23

## Context

`ROADMAP.md`'s orchestration section's last remaining line was
`` `langgraph` backend: branching / conditional graphs -- `supervisor`
strategy only; `reflection`/`hierarchical`/`graph` on langgraph remain
📋``. ADR-0016 (which shipped `supervisor` on langgraph) explicitly
pre-scoped this exact follow-up in its own Follow-ups section:
*"`reflection` on langgraph: single-agent critique loop, naturally a
2-node cycle (draft/critique-revise) with a conditional exit on the
`NO_CHANGES_NEEDED` sentinel -- a good next candidate, not scoped here."*
This ADR is that follow-up. `hierarchical` (needs `Agent`-or-`Workflow`
duck-typed delegate handling, per ADR-0013) and `graph` (translating an
already-graph-shaped `Workflow` into langgraph's own graph) remain
separately-scoped follow-ups of their own, deliberately not attempted in
the same pass -- matching this project's established
one-or-two-strategies-per-ADR pace (ADR-0016 itself did exactly one, for
the same reason).

`NativeOrchestrator._run_reflection`/`_arun_reflection`
(`requisite/orchestrators/native.py:547-611`) is the exact reference
semantics reproduced here: one agent, an initial `worker.run(input)`
draft, then up to `max_rounds - 1` iterations of critique-then-maybe-revise:

```python
draft = worker.run(input)
for _ in range(max_rounds - 1):
    critique = worker.run(_reflection_critique_prompt(input, draft.content))
    if critique.content.strip() == "NO_CHANGES_NEEDED":
        break
    draft = worker.run(_reflection_revise_prompt(input, draft.content, critique.content))
return draft.content
```

The one subtlety worth naming precisely: revise always follows a
non-sentinel critique, regardless of remaining round budget -- the `for`
loop's own bound is what stops further *critique* rounds from starting,
not a check gating whether to revise. Translating this loosely (e.g.
gating revise on remaining budget too) would silently change behavior
for the last permitted round.

## Decision

### Reuse native's prompts, don't duplicate them

Same decision ADR-0016 already made for `_supervisor_prompt`, restated
here for `_reflection_critique_prompt`/`_reflection_revise_prompt`:
`requisite/orchestrators/langgraph_orchestrator.py` imports both
directly from `native.py` rather than reimplementing them, so
`workflow.reflection()` produces *identical* prompts regardless of
backend -- only how the critique/revise cycle is driven differs (a
Python `for` loop vs. a real graph cycle).

### A real 3-node conditional cycle, not a disguised loop

```python
class _ReflectionGraphState(TypedDict):
    input: str
    draft: str
    critique: str
    steps: list[Any]
    rounds: int
```

Three dunder-wrapped nodes (`"__draft__"`, `"__critique__"`,
`"__revise__"`, matching `_SUPERVISOR_NODE = "__supervisor__"`'s existing
convention -- collision with a real agent name isn't even a concern
here since `reflection` always has exactly one agent, but the
convention is kept for consistency across this module's node names).
`rounds` counts completed *critique* calls only (incremented in the
critique node, not the revise node) -- the same quantity Native's own
`for _ in range(max_rounds - 1)` bounds.

Routing, translating the subtlety above precisely rather than
approximately:

- After `"__draft__"`: `max_rounds > 1 -> "__critique__"`, else `-> END`
  -- handles `max_rounds <= 1`, where Native's own `range(max_rounds-1)`
  is empty and no critique ever runs. `max_rounds` is a closure
  variable baked in at graph-build time (matching
  `_build_supervisor_graph(steps, *, max_rounds, ...)`'s existing
  pattern), not stored in graph state.
- After `"__critique__"`: `critique.strip() == "NO_CHANGES_NEEDED" ->
  END`, else unconditionally `-> "__revise__"` -- revise is never
  budget-gated here, matching Native exactly (see Context).
- After `"__revise__"`: `rounds < max_rounds - 1 -> "__critique__"`
  (loop back for another round), else `-> END` -- this is the *only*
  place round budget is checked, matching where Native's own `for`
  loop's bound actually takes effect (deciding whether to iterate
  again, not whether to revise the current iteration).

### `add_conditional_edges`'s 2-argument form, not the 3-argument `path_map` form

`_build_supervisor_graph` uses the 3-argument form (`path_map` mapping
a route string to a target); this feature uses the simpler 2-argument
form instead -- each routing function returns the target node name or
`END` directly (verified via `inspect.signature(StateGraph.add_conditional_edges)`
that `path_map` is `Optional[...] = None`, a genuinely supported
usage, not an assumption). No `path_map` dict is needed here since
there's no indirection between a "route" value and a node name the way
supervisor's worker-name routing has -- the routing functions already
compute the exact target.

### Error messages match Native's exactly, not a paraphrase

`_build_reflection_graph`'s `len(steps) != 1` check raises the
byte-for-byte same `ConfigurationException` message
`NativeOrchestrator._run_reflection` already raises -- an application
switching `.use_langgraph()` after building on `.reflection()` sees the
identical error for the identical mistake, not a differently-worded one
that could look like a different bug.

## Alternatives considered

- **The 3-argument `path_map` form**, matching `_build_supervisor_graph`
  for stylistic consistency. Rejected -- would need a redundant
  identity-mapping dict (`{"__critique__": "__critique__", "__revise__":
  "__revise__", END: END}`) for no behavioral benefit, since the routing
  functions already return valid targets directly; the 2-argument form
  is equally valid per langgraph's own signature and simpler here.
- **Gating revise on remaining round budget** (checking `rounds <
  max_rounds - 1` before routing to `"__revise__"`, not just before
  routing back to `"__critique__"`). Rejected -- this is precisely the
  subtlety Context calls out: Native's own loop always revises after a
  non-sentinel critique, and diverging here would make
  `workflow.reflection()` behave differently by backend for the exact
  same `max_rounds` value on the final permitted round.
- **`hierarchical` or `graph` in the same pass.** Rejected -- see
  Context; both are real, separately-scoped follow-ups, not dismissed.

## Consequences

### Positive

- Closes the last remaining line in `ROADMAP.md`'s orchestration
  section (`hierarchical`/`graph` on langgraph are still open, but no
  longer bundled under a single ambiguous "branching" line -- see
  Follow-ups).
- `workflow.reflection()` now works unmodified across both `native` and
  `langgraph` backends, the same win ADR-0016 already delivered for
  `supervisor`.
- Verified against a real Gemini call, not just the scripted fake: a
  full 5-step draft/critique/revise/critique/revise cycle at
  `max_rounds=3`, correctly using the full budget and returning the
  final revision -- confirms the graph's conditional routing genuinely
  drives multiple real rounds, not just one scripted pass.

### Negative / risks

- `reflection` remains native-and-langgraph only; `hierarchical` and
  `graph` still raise the (now three-strategy-aware) `ConfigurationException`
  on langgraph, same as before this ADR just with an updated message.
- A third distinct `TypedDict` state shape in this module
  (`_ReflectionGraphState`, alongside `_GraphState` and
  `_SupervisorGraphState`) -- same trade-off ADR-0016 already accepted
  for its own second shape: more state shapes to keep in mind when
  extending this module further, rather than one unified shape every
  strategy shares.

### Follow-ups

- `hierarchical` on langgraph: same shape as `supervisor` but a
  delegate may be a `Workflow` (duck-typed, per ADR-0013) -- still not
  scoped here.
- `graph` on langgraph: translating an already-graph-shaped `Workflow`
  (`.add_edge(...)`-declared) into langgraph's own graph-building calls
  -- a different kind of translation task (mechanical, from an
  already-fully-specified structure) than `reflection`/`supervisor`'s
  "reuse native's decision logic" pattern; still not scoped here.
