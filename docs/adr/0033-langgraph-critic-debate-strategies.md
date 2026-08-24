# 0033. `critic` and `debate` strategies on the langgraph backend

Status: Accepted
Date: 2026-08-25

## Context

ADR-0032 closed by reaffirming that `debate`, `critic`, `tree_of_thoughts`,
and `planner` remain native-only. This ADR picks up two of those four --
`critic` and `debate` -- at Keyan's explicit request. `tree_of_thoughts`
and `planner` remain out of scope, reaffirmed below in Follow-ups;
`tree_of_thoughts` in particular (beam search with per-level pruning and
early termination) is a meaningfully different, harder shape than either
strategy shipped here.

**`critic`** (`native.py:920-952`/`954-986`, `_run_critic`/`_arun_critic`)
turns out to be a near-direct generalization of `reflection`, already on
this backend since ADR-0028, not a new shape: exactly 2 fixed positional
agents (`generator`, `critic`, no name addressing, unlike consensus/
map_reduce), the identical bounded critique/revise loop (revise never
budget-gated, only whether another critique round starts is), the same
`NO_CHANGES_NEEDED` early exit. The only real differences from
`reflection`: two agents instead of one, and `_critic_prompt(task,
draft)` instead of `_reflection_critique_prompt(task, draft)` for the
critique step. `_reflection_revise_prompt(task, draft, critique)` is
**already shared** between the two strategies in `native.py` itself
(confirmed: `_run_critic`'s revise call uses `_reflection_revise_prompt`
directly, not a critic-specific variant) -- so the revise step needs no
per-strategy branching at all. `_ReflectionGraphState`'s existing fields
(`input`, `draft`, `critique`, `steps`, `rounds`) already fit `critic`'s
shape exactly.

**`debate`** (`native.py:1037-1076`/`1078-1117`, `_run_debate`/`_arun_debate`)
is genuinely new territory. `_split_coordinator_and_workers` splits
`steps[0]` (moderator) from `steps[1:]` (debaters, name-addressed). For
`max_rounds` full rounds -- always all of them, no early exit, unlike
`critic` -- every debater runs concurrently, each seeing the full
transcript of every debater's arguments from all prior rounds
(`_debate_prompt(task, debater_names, transcript, agent_name=name,
round_num=round_num)`); each round's results extend a `dict[str,
list[str]]` transcript keyed by debater name. After all rounds, the
moderator issues one final verdict via `_debate_verdict_prompt(task,
debater_names, transcript)`. This is "a loop of fan-outs" -- neither the
existing loop-back-cycle shape (`supervisor`/`hierarchical`/`reflection`,
one node per superstep) nor ADR-0032's single-round fan-out shape
(`parallel`/`consensus`/`map_reduce`) covers it directly.

### Design choice: unroll the rounds at graph-build time, don't build a true cycle

Every `_build_*_graph` method in this module is built fresh inside
`run()`/`arun()`, with `max_rounds` known before `StateGraph(...)` is
even constructed -- the same fact ADR-0032 used to justify static
per-item node generation for `map_reduce` over LangGraph's `Send` API.
The same reasoning applies here: since `max_rounds` is a known integer
at build time, `_build_debate_graph` generates `max_rounds` distinct
fan-out/join node blocks (one per round), each round's join node
connecting to the next round's fan-out nodes via the same "call
`add_edge` multiple times from one source" mechanism ADR-0032 already
verified works for `StateGraph` (the "already found path" guard that
would normally block a repeated `start_key` is skipped for `StateGraph`
specifically -- confirmed there, reused here without re-verifying). This
avoids needing a genuinely cyclic graph with dynamic multi-target
routing (unverified LangGraph territory) entirely -- the graph is a
longer, fully static DAG instead.

**Verified empirically** (ran native's own `_run_debate` with
`max_rounds=0`): zero debater calls happen, the moderator still runs
once against an empty transcript, `result.steps == [moderator_result]`.
`_build_debate_graph` reproduces this exactly: when no round blocks are
built at all, `START` connects directly to the verdict node instead of
the (nonexistent) last round's join node.

**Verified structurally, matching ADR-0032's own reserved-node-name
reasoning**: debater node names (`f"{name}_r{round_num}"`) always end in
a bare digit (from `round_num`), while this module's reserved node names
all end in a double underscore (`__aggregator__`, `__verdict__`,
`__debate_join_r{n}__`) -- textually impossible to collide, the same
guarantee that let `parallel` skip an explicit reserved-name check in
ADR-0032. No `_reject_reserved_node_names` call was added for debate's
debaters, by construction -- confirmed with an adversarial test (11
debaters, ordering preserved) rather than left as an unverified
assertion.

## Decision

### `critic`: generalize `_build_reflection_graph`, don't duplicate it

The exact ADR-0029 precedent (`_build_supervisor_graph` ->
`_build_delegation_graph`, parameterized by `split_fn`/`role`) applied to
this builder. `_build_reflection_graph(self, steps, *, max_rounds,
**kwargs)` gained a `role: str` parameter (`"reflection"` or `"critic"`),
branching once at the top:

```python
if role == "reflection":
    if len(steps) != 1:
        raise ConfigurationException(...)
    generator = critic_agent = steps[0]
    critique_prompt_fn = _reflection_critique_prompt
else:
    if len(steps) != 2:
        raise ConfigurationException(
            f"The 'critic' strategy requires exactly two agents: a generator "
            f"(steps[0]) and a critic (steps[1]). Got {len(steps)}.",
        )
    generator, critic_agent = steps[0], steps[1]
    critique_prompt_fn = _critic_prompt
```

Node bodies changed from the hardcoded `worker` to `generator`/
`critic_agent`; `_critique_node` calls `critique_prompt_fn(...)` instead
of a hardcoded prompt function. `_draft_node` and `_revise_node` both
always use `generator` (for `reflection`, `generator is critic_agent is
steps[0]`, so behavior is unchanged -- confirmed: all 9 existing
`reflection` tests pass without modification). Routing functions
(`_route_after_draft`/`_route_after_critique`/`_route_after_revise`)
needed no changes at all -- they only ever read graph state, never the
agent objects. `run()`/`arun()`'s `strategy == "reflection"` branch
became `strategy in ("reflection", "critic")`, passing `role=strategy`
through -- mirroring the `supervisor`/`hierarchical` combined-branch
precedent exactly. No new `TypedDict`, no new reserved node name.

### `debate`: new `_DebateGraphState` and `_build_debate_graph`

```python
class _DebateGraphState(TypedDict):
    task: str
    # Global-indexed across ALL rounds (index = round_num * len(debaters) + i),
    # not per-round -- one shared reducer channel for the whole debate,
    # since TypedDict schemas are static. Each round's join node filters
    # its own slice by index range.
    results: Annotated[list[tuple[int, Any]], operator.add]
    transcript: dict[str, list[str]]  # plain field -- one join node writes it per superstep, never concurrent
    steps: list[Any]
    output: str
```

New reserved constant: `_VERDICT_NODE = "__verdict__"`.

`_build_debate_graph(self, steps, *, max_rounds, **kwargs)` returns
`(compiled_graph, debater_names)`, not just the compiled graph -- the
caller needs `debater_names` to build the initial `transcript` dict
(`{name: [] for name in debater_names}`) before invoking, and computing
the coordinator/worker split twice (once in the builder, once in
`run`/`arun`) would risk the two falling out of sync. Per round: one
node per debater (`f"{name}_r{round_num}"`, body computes
`_debate_prompt(state["task"], debater_names, state["transcript"],
agent_name=name, round_num=round_num)`, runs `debater.run(...)`, writes
`{"results": [(index, result)]}`); edges in from `START` (round 0) or
the previous round's join node. One join node per round
(`f"__debate_join_r{round_num}__"`): filters+sorts `state["results"]`
for that round's index range, zips against `debater_names` (`strict=True`,
matching native's own alignment discipline), builds a new `transcript`
dict, appends the round's results to `steps`. Fan-in edge:
`graph.add_edge(round_node_names, join_name)` (the list-form waiting
edge ADR-0032 verified). After the last round (or immediately, if
`max_rounds == 0`), one `_VERDICT_NODE` calls `moderator.run(_debate_verdict_prompt(...))`
and writes `output`.

`run()`/`arun()` gained a `strategy == "debate"` branch: `max_rounds =
kwargs.pop("max_rounds", 3)` (matching native's default), build, invoke
with `{"task": input, "results": [], "transcript": {name: [] for name in
debater_names}, "steps": [], "output": ""}`, return
`WorkflowResult(content=final_state["output"], steps=final_state["steps"],
orchestrator=self.name, strategy="debate")`. The trailing
`ConfigurationException` in both methods was extended to list all ten
supported strategies.

## Alternatives considered

- **A true cyclic graph for debate, with dynamic multi-target
  conditional routing back to all debater nodes.** Rejected -- this
  would require a LangGraph capability (a router returning multiple
  target nodes to re-enter concurrently) that this module has never
  used and that wasn't verified to exist cleanly; the static-unroll
  approach achieves the identical result using only mechanisms ADR-0032
  already verified (repeated `add_edge` from one source, list-form join
  edges), with `max_rounds` being build-time-known making the unroll
  free of the downsides a *truly* unbounded loop would have.
- **Three separate node-body functions for `critic`** (duplicating
  `_draft_node`/`_critique_node`/`_revise_node` rather than
  parameterizing the existing reflection builder). Rejected -- exactly
  the duplication ADR-0029 already rejected for `supervisor`/
  `hierarchical`, now avoided for the same reason a second time.
- **A per-round `TypedDict`** (dynamically generating a new state shape
  for each debate's specific `max_rounds`). Rejected -- `TypedDict`
  schemas are static in this module's existing design (confirmed
  nowhere else does this); one shared, globally-indexed `results`
  channel with per-round filtering is simpler and needs no schema
  generation step.

## Consequences

### Positive

- `sequential`/`supervisor`/`hierarchical`/`reflection`/`graph`/
  `parallel`/`consensus`/`map_reduce`/`critic`/`debate` are all now
  shipped on both `native` and `langgraph` backends -- only
  `tree_of_thoughts` and `planner` remain native-only.
- `critic`'s implementation added zero new state shapes and zero new
  reserved node names -- the cheapest strategy addition to this module
  so far, precisely because the reuse-the-existing-builder approach paid
  off as expected from reading `native.py` closely first.
- `debate`'s "loop of fan-outs" shape is now a proven, reusable pattern
  in this module (round-unrolled fan-out/join blocks) should a future
  strategy need something structurally similar.
- New test coverage continues ADR-0032's stronger-than-mirroring pattern:
  a `max_rounds=0` degenerate-case test, an 11-debater ordering test
  (proving the index+sort design against the same lexicographic-write-order
  risk ADR-0032 found, not assuming it away a second time), and
  native-vs-langgraph parity tests for both strategies.

### Negative / risks

- A sixth distinct `TypedDict` state shape in this module
  (`_DebateGraphState`). `critic` did not add a shape (reused
  `_ReflectionGraphState`), keeping the net increase to one shape for
  two new strategy/backend combinations.
- Debate's graph size scales with `max_rounds * len(debaters)` at
  build time (a static DAG, not a loop) -- unlike native's O(1)-memory
  round loop. Not a concern for realistic `max_rounds` (a debate is not
  meant to run hundreds of rounds), but a real, worth-naming difference
  from the native backend's memory profile for pathologically large
  `max_rounds`.
- `_build_debate_graph` returning a tuple (`compiled_graph,
  debater_names`) instead of just the compiled graph is a small
  asymmetry against every other `_build_*_graph` method in this module,
  which return only the compiled graph. Accepted since the alternative
  (computing the coordinator/worker split twice) risks the two call
  sites silently diverging.

### Follow-ups

- `tree_of_thoughts` and `planner` remain native-only, reaffirming
  ADR-0032's own follow-up list, narrowed by two.
