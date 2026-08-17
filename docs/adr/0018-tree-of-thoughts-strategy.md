
# 0018. Tree-of-thoughts multi-agent strategy

Status: Accepted
Date: 2026-08-17

## Context

`ROADMAP.md`'s last remaining line in "Agents & multi-agent orchestration"
is *"Tree-of-thoughts strategy — 📋."* Unlike every other strategy shipped
this stretch, this one has **zero prior design sketch anywhere in the
repo** — confirmed by grepping all 17 prior ADRs, `ROADMAP.md`, and
`FEATURES.md`. Every hit is a bare status-table row or a one-line
deferral repeating the same two facts, most substantively in ADR-0011:

> "tree-of-thoughts (branching, scoring, and pruning a search tree of
> partial solutions) do[es] not fit the flat coordinator/worker list
> without materially bigger structural changes... a whole
> branching-search loop."

And ADR-0013's follow-up: *"a genuinely different shape (branching
search with evaluation/pruning, not a coordinator/delegate round loop)
and needs its own design pass."* This is that pass.

The design below is the classic ToT-BFS (beam search) algorithm from
Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large
Language Models," mapped onto this codebase's existing conventions —
not a new invention, and not a simplified version that fails to deliver
what "tree-of-thoughts" actually promises (the exact trap ADR-0011
warned against rushing into for this feature).

## Decision

### Shape: evaluator (`steps[0]`) + thinker pool (`steps[1:]`)

Reuses `_split_coordinator_and_workers(steps, role="tree-of-thoughts
evaluator")` unchanged — the same ≥2-agents-unique-names validation
every coordinator/worker strategy gets for free, no new helper.

### Breadth is a kwarg decoupled from thinker count, not `len(thinkers)`

This mirrors `map_reduce`'s established "core decision" (ADR-0012):
new breadth-shaped data flows through a keyword-only kwarg, round-robin
assigned across the worker pool, rather than forcing the caller to
provision one agent per unit of breadth:

```python
breadth: int = 3       # candidate thoughts generated per surviving path, per level
beam_width: int = 1    # top-k surviving paths kept after each level's evaluation
max_depth: int = 3     # levels of search before finalizing
```

A user gets `breadth=5` candidates per level from 1-2 thinker agents,
not 5 near-duplicate `Agent` instances. Candidates at level 0: `breadth`
(from the single root path). At level `L > 0`: up to
`beam_width * breadth` (each surviving path forks `breadth` ways),
evaluated together, pruned back to `beam_width`. This bounds growth to
`beam_width`, not exponential — the entire point of beam search over
naive full-tree BFS. All three kwargs are keyword-only, popped by the
method signature before `**kwargs` forwarding, the same mechanism
`max_rounds`/`map_items` already use; each is validated (`>= 1`) with an
actionable `ConfigurationException`, matching `map_reduce`'s
missing-`map_items` precedent.

### Structured evaluation, not free text — a deliberate departure from critic's precedent

Every existing free-form strategy (critic, reflection, consensus,
debate) uses plain text or a sentinel for its "decision," per ADR-0011's
explicit reasoning: "nothing to parse, just prose." Tree-of-thoughts'
evaluator instead uses `response_model=`, matching
`_SupervisorDecision`/`_Plan`'s precedent for routing-shaped decisions
code must parse reliably — because pruning requires *comparing numeric
scores across multiple candidates simultaneously*, not checking one
sentinel. The shape mirrors `LLMReranker`'s listwise
`_ChunkRelevance`/`_RerankScores` (`requisite/rag/reranker.py`) closely,
including the same defensive by-index score lookup that never crashes
on a missing/malformed entry (`score_by_index.get(i, 0.0)`):

```python
class _ThoughtScore(BaseModel):
    index: int
    score: float = Field(ge=0.0, le=10.0)
    finished: bool = False

class _ThoughtEvaluation(BaseModel):
    scores: list[_ThoughtScore] = Field(default_factory=list)
```

One evaluator call per *level* (not per candidate) scores every
candidate generated that level at once — the same reasoning
`LLMReranker` gives for being listwise rather than pointwise: fewer
calls, and the evaluator can weigh candidates relative to each other
directly. If any candidate is marked `finished`, the loop returns the
highest-scoring finished candidate immediately rather than continuing
to `max_depth` — the closest analog to `_SupervisorDecision.action ==
"finish"`, adapted from "one routing choice" to "best of several
simultaneously-evaluated candidates."

### No shared round-loop with critic/reflection/supervisor

ADR-0011 already explicitly rejected a shared loop helper across
critic/reflection/supervisor because they differ in what they pass to
`response_model=`, what terminates them, and what they accumulate.
Tree-of-thoughts differs from all three in exactly those same ways
(structured multi-candidate scoring instead of a sentinel or a single
routing decision; early-finish-on-any-candidate instead of a single
coordinator decision; path-branching accumulation instead of a flat
transcript), so it gets its own clean
`_run_tree_of_thoughts`/`_arun_tree_of_thoughts` — consistent with that
precedent, not contradicting it.

### `steps` records every candidate generated, kept or pruned

No existing strategy discards a worker-produced `AgentResult` from
`WorkflowResult.steps` — debate keeps every debater's every round
(ADR-0007 established "steps holds worker results"; only *coordinator*
decision calls, which aren't `AgentResult`s at all, get excluded).
Tree-of-thoughts follows the same rule: every candidate thought's
`AgentResult`, across every level, lands in `steps`, pruned ones
included. Inventing a "silently drop pruned candidates" precedent would
need its own justification this feature has no reason to invent.

### No exception on `max_depth` exhaustion

Matches critic/reflection/debate/consensus (return the best-effort
result after the loop) rather than supervisor/hierarchical (raise —
those model an explicit "did the coordinator ever decide to finish"
completion state tree-of-thoughts doesn't have). It's a bounded search
that always produces *a* best-scoring path; running out of depth
without an early `finished=True` isn't a failure state.

### Native-only; langgraph deferred

ADR-0016's own follow-ups already named `reflection`/`hierarchical` as
the next langgraph-backend candidates, not tree-of-thoughts — matches
this stretch's established "native first, langgraph as its own later
pass" pattern for every strategy so far (supervisor shipped native in
ADR-0007, langgraph nine ADRs later in ADR-0016). Noted as a follow-up,
not attempted here.

## Alternatives considered

- **Breadth tied to `len(thinkers)`** (one candidate per thinker agent
  added, no separate kwarg). Rejected: would force callers to
  instantiate N near-duplicate `Agent` objects just to widen the
  search, and repeated calls to the *same* agent/provider at a
  reasonable temperature already produce varied candidates — the
  `map_reduce` precedent (item count decoupled from, and round-robin
  over, the worker pool) is the better fit and was already validated in
  this codebase.
- **One evaluator call per candidate** (pointwise) instead of one call
  per level scoring every candidate at once (listwise). Rejected for
  the same reason `LLMReranker` chose listwise (ADR-0010): fewer calls,
  and relative judgment across candidates is often more reliable than
  independent absolute scores.
- **DFS with backtracking** instead of BFS with beam pruning. Rejected
  as the more complex of the two standard ToT search modes for a first
  pass — BFS-with-beam is simpler to reason about, bounds cost
  predictably via `beam_width`, and is the mode most tree-of-thoughts
  implementations default to. DFS is a plausible future addition, not
  scoped here.
- **A separate "synthesize the final answer" step** (like consensus's
  synthesizer or debate's moderator verdict) instead of using the
  winning path's last thought directly as the answer. Rejected: ToT's
  own literature treats the final thought (especially one marked
  `finished`) as the answer already; an extra synthesis call would be
  pure added cost for no clear benefit, and no existing precedent
  (reflection, critic) adds a synthesis step beyond the loop's own
  output either.

## Consequences

### Positive

- Closes the last 📋 line in `ROADMAP.md`'s "Agents & multi-agent
  orchestration" section.
- Reuses every available building block (`_split_coordinator_and_workers`,
  `ThreadPoolExecutor`/`asyncio.gather` concurrency, the `map_reduce`
  kwarg-popping mechanism, the `LLMReranker` listwise-scoring shape) —
  no new concurrency primitive, no new kwarg-threading mechanism.
- `Workflow.tree_of_thoughts()`'s public shape is exactly as
  predictable as every other strategy: add agents, call `.run()`/`.arun()`
  with strategy-specific kwargs — no change to `Workflow`,
  `BaseOrchestrator`, or `OrchestratorRegistry`.

### Negative / risks

- Cost grows with `beam_width * breadth * max_depth` evaluator-visible
  candidates plus `beam_width * breadth * max_depth` thinker calls in
  the worst case (no early finish) — same accepted "caller keeps the
  fan-out reasonably small" limitation already noted for consensus's
  synthesizer and `LLMReranker` (no token-budget guard).
  Defaults (`breadth=3, beam_width=1, max_depth=3`) keep this cheap (9
  thinker calls + 3 evaluator calls) by default.
- The evaluator's single listwise call per level means a very wide
  level (`beam_width * breadth` large) puts many full candidate paths
  in one prompt — same class of context-window risk already accepted
  for consensus/`LLMReranker`.
- No DFS/backtracking mode — a path pruned at level `L` can never be
  revisited even if later levels reveal the whole beam was on a bad
  branch. This is an inherent limitation of beam search, not a bug;
  larger `beam_width` is the mitigation available today.

### Follow-ups

- Tree-of-thoughts on the `langgraph` backend, as a real branching graph
  (naturally graph-shaped, arguably more naturally than `supervisor`
  was) — not scoped here; `reflection`/`hierarchical` are ahead of it
  per ADR-0016's own follow-ups.
- DFS-with-backtracking as an alternative search mode, if beam search's
  "no revisiting a pruned branch" limitation proves to matter in
  practice — not speculatively added now.
- General graph execution (arbitrary DAGs) is the one remaining item on
  `ROADMAP.md`'s orchestration section not covered by any shipped
  strategy — a separate, broader feature, not a variant of this one.
