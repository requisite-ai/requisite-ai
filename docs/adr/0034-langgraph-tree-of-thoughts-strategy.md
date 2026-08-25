# 0034. `tree_of_thoughts` strategy on the langgraph backend

Status: Accepted
Date: 2026-08-25

## Context

`tree_of_thoughts` (`native.py:1185-1242`/`1244-1299`,
`_run_tree_of_thoughts`/`_arun_tree_of_thoughts`) is the most involved
strategy this framework ships, and the last of the four strategies named
in ADR-0032/ADR-0033's follow-up lists to get a langgraph implementation
at Keyan's request (`planner` remains native-only, see Follow-ups). It
combines three things no single langgraph strategy so far has combined
in one shape: a **beam search** (the working set of candidate paths
shrinks via pruning, level to level, not a fixed per-round width like
`debate`), **structured-output evaluation** (`evaluator.ai.chat(...,
response_model=_ThoughtEvaluation)`, one listwise call scoring every
candidate at a level, not a per-agent free-text call), and
**data-dependent early termination at any level** (not just after all
rounds, like `debate`; not a fixed 2-outcome sentinel, like
`reflection`/`critic`).

Read directly: start with `paths = [[]]` (one empty root path). For up
to `max_depth` levels, fan every surviving path out into `breadth` new
candidates (`tasks = [path for path in paths for _ in range(breadth)]`,
thinkers assigned round-robin via `thinker_list[i % len(thinker_list)]`
for `i` **local to that level** -- confirmed via `enumerate(tasks)`
starting fresh each level, not a running total); run all of them
concurrently; extend each into a `candidate_paths` entry; the evaluator
scores every candidate in one listwise structured-output call;
`_select_finished_tot_candidate` returns the best-scored `finished=True`
candidate immediately if any exist; else `_prune_tot_candidates` keeps
the top `beam_width` candidates as the next level's `paths`. If the loop
exhausts `max_depth` levels without a finished candidate, returns
`paths[0]` (the top-ranked survivor). `results`/`steps` accumulates
every level's candidate thinker results unconditionally, including the
terminating level's; the evaluator's structured-output calls are never
added.

Three static helpers, already `@staticmethod` and directly reusable
(the same cross-module-reuse pattern as `_split_coordinator_and_workers`,
confirmed by reading them): `_validate_tot_params(*, breadth,
beam_width, max_depth)` (each `>=1`), `_select_finished_tot_candidate`
(bounds-checks `index`, silently drops out-of-range entries),
`_prune_tot_candidates` (missing-index entries default to score `0.0`,
never raises, provably returns a non-empty list -- see Decision).
`_tot_thinker_prompt`/`_tot_evaluation_prompt` are plain module-level
functions; `_ThoughtScore`/`_ThoughtEvaluation` (`native.py:88-102`) are
the structured-output schema -- all reused directly, none reimplemented.

## Decision

### The beam-search shape is fully determined at graph-build time -- proof

`breadth`, `beam_width`, and `max_depth` are known before
`StateGraph(...)` is constructed, same as every prior ADR in this
module. What makes this ADR's static unroll valid is a stronger claim
than `map_reduce`'s or `debate`'s: **the number of surviving paths
entering each level is also fully determined by these three integers
alone, never by the evaluator's actual scores** -- pruning always keeps
exactly `min(beam_width, candidates_available)` paths, and how many
candidates are available is itself just `paths_count * breadth`. By
induction: `paths_count[0] = 1`; `level_widths[L] = paths_count[L] *
breadth`; `paths_count[L+1] = min(beam_width, level_widths[L])`. Since
`paths_count[L] >= 1 => level_widths[L] >= 1 => paths_count[L+1] >= 1`,
every level has at least one candidate, starting from `paths_count[0] =
1`. This is computed in a plain Python loop at graph-build time
(`level_widths`, and cumulative `level_offsets` into one shared results
channel) -- no agent calls, no LLM involvement, purely the same
arithmetic `_prune_tot_candidates` and the native loop already perform
implicitly.

What genuinely IS data-dependent, and must live in graph state: the
actual path *content* and whether a level's evaluation found a
`finished=True` candidate. These two drive routing; everything else is
structure.

### One shared, offset-indexed reducer channel spanning every level

```python
class _TreeOfThoughtsGraphState(TypedDict):
    task: str
    paths: list[list[str]]
    candidates: Annotated[list[tuple[int, Any]], operator.add]
    steps: list[Any]
    output: str
    finished: bool
```

Same rationale as `_DebateGraphState.results` (ADR-0033): `TypedDict`
schemas are static, so one shared, globally-offset-indexed channel is
the established pattern, not one field per level. `paths` is a plain
field -- exactly one evaluation node (the current level's) writes it
per superstep, never concurrent. `finished` is written by every level's
evaluation node and read immediately after by that same level's router.

### Per level: candidate fan-out -> one eval/prune node -> conditional route to the next level's *expand* node, or `END`

`level_widths[L]` candidate nodes, synthetic names
(`f"__tot_L{level}_{i}__"` -- **never derived from thinker names**, so
unlike `consensus` this needs no `_reject_reserved_node_names` call at
all, the same argument already used for `map_reduce`'s synthetic
`__mapper_{i}__` names). Each reads `state["paths"][i // breadth]` (its
parent, `i` local to the level), picks `thinker_list[i %
len(thinker_list)]`, writes `{"candidates": [(level_offset + i,
result)]}`.

One eval/prune node per level, fed by the already-verified list-form
join edge. It filters+sorts its own offset slice, rebuilds
`candidate_paths` identically to native (`[*state["paths"][i //
breadth], result.content]`, `i` local), calls `evaluator.ai.chat(...,
response_model=_ThoughtEvaluation)` (the established pattern from
`_build_delegation_graph`'s coordinator node), then does exactly what
native's per-level block does: `_select_finished_tot_candidate` first
(write `output`+`finished=True` if a match); else
`_prune_tot_candidates` (write `paths`+`finished=False`) -- and, since a
graph has no "after the loop" step the way a Python `for` does, **if
this is the last level**, also writes `output=pruned[0][-1] if
pruned[0] else ""` right there (closure-known via `is_last_level`), so
exhaustion-termination needs no separate node. `steps` is always
extended with the level's candidate results.

Routing (`add_conditional_edges`, 2-arg form, same as `reflection`'s):
`END if state["finished"] or level == max_depth - 1 else
f"__tot_expand_L{level + 1}__"`. `add_conditional_edges` can only return
one target, not fan out to N -- so a trivial passthrough node per level
transition (`__tot_expand_L{level}__`, body `{}`) is the single named
target the router points at, which then statically fans out (repeated
`add_edge`, the same mechanism `START` already uses for level 0) to all
of that next level's real candidate nodes. Level 0 needs no expand node
-- `START` fans out to its candidates directly. This passthrough is the
one genuinely new piece of machinery this ADR adds; everything else
(offset-indexed reducer, list-form join, repeated-source fan-out,
2-arg conditional routing, `.ai.chat(response_model=...)`) is direct
reuse of mechanisms ADR-0016/0028/0029/0032/0033 already verified.

## Alternatives considered

- **A true dynamic beam search using `Send`.** Rejected, consistent
  with ADR-0032/0033's reasoning: the beam width is build-time-computable
  here too (via the induction above, a less trivial formula than
  `map_reduce`'s but still pure arithmetic), so `Send`'s value
  proposition (width unknown until invoke time) still doesn't apply.
- **Per-level-named state fields instead of one shared offset-indexed
  channel.** Rejected -- `TypedDict` schemas are static; would need a
  schema generated per `max_depth`, unprecedented in this module and
  strictly worse than the established offset-indexing approach.
- **Routing directly to one of the next level's candidate nodes**
  (skipping the expand passthrough, picking an arbitrary "first" node as
  the conditional target and statically fanning the rest out from
  elsewhere). Rejected -- there is no principled choice of "first" node
  once the router has already committed to returning exactly one name;
  a dedicated passthrough keeps the fan-out mechanism uniform and
  independent of any specific candidate node's identity.

## Consequences

### Positive

- Only `planner` remains native-only after this -- ten of eleven
  strategies now run on both `native` and `langgraph`.
- The induction proof and offset-indexing scheme generalize cleanly:
  any future strategy with a build-time-computable, data-independent
  branching factor can reuse this exact pattern rather than reaching for
  `Send`.
- New test coverage mirrors every existing native `tree_of_thoughts`
  test (pruning, early termination, round-robin assignment, all three
  invalid-parameter cases) plus, matching ADR-0032/0033's
  stronger-than-mirroring pattern: an 11-candidate ordering test (same
  lexicographic-write-order risk already found and mitigated in
  ADR-0032, now confirmed at this strategy's own scale) and a
  native-vs-langgraph parity test.
- Verified live: an adversarial pass surfaced no real bugs in the
  implementation, but did catch a bug in a hand-written verification
  script itself (an under-scored evaluation fixture for a
  `beam_width=2` case) -- caught precisely *because* the same scenario
  was cross-checked against native's own real output with identical
  parameters, not assumed correct from the langgraph run alone.

### Negative / risks

- A seventh distinct `TypedDict` state shape in this module. Consistent
  with the trend since ADR-0032: state-shape count grows roughly
  linearly with genuinely-distinct-shape strategies, not with total
  strategies (`critic` in ADR-0033 added zero).
- Graph size scales with the total candidate count across all levels
  (`sum(level_widths)`), which itself can grow multiplicatively with
  `breadth` if `beam_width` is large relative to `breadth` -- a static
  DAG, not native's O(1)-memory loop. Not a concern for realistic
  parameters (this is a reasoning search, not meant to run at `breadth`
  in the hundreds), but a real, worth-naming difference in memory
  profile for pathological inputs, the same class of risk ADR-0033
  named for `debate`'s `max_rounds`.
- The `is_last_level` special-case inside the eval node (writing
  `output` there instead of via a separate terminal node) is a small
  asymmetry against the cleaner "termination is always a router
  decision" story the rest of the design tells. Accepted since
  introducing a dedicated post-last-level node would only move this
  one `if` from a node body to an edge, not remove it.

### Follow-ups

- `planner` remains native-only -- the only strategy from
  ADR-0032/0033's combined follow-up lists not yet picked up. A future
  ADR if wanted; structurally closer to `supervisor` (a coordinator
  decomposing a task into named-worker subtasks) than to anything in
  this ADR.
