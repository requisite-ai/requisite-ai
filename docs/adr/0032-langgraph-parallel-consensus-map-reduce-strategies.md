# 0032. `parallel`, `consensus`, and `map_reduce` strategies on the langgraph backend

Status: Accepted
Date: 2026-08-24

## Context

ADR-0029 closed with an explicit note: *"`planner`/`critic`/`consensus`/
`debate`/`map_reduce`/`tree_of_thoughts` remain native-only by design --
genuinely different shapes from a coordinator/worker or
developer-declared graph, not a gap this line of work was meant to
close."* This ADR picks three of those six back up -- `parallel`,
`consensus`, `map_reduce` -- because, once their native reference
implementations are read closely, they turn out to share one graph
shape that is genuinely simpler than anything `langgraph_orchestrator.py`
has done so far: fan out N agents concurrently, then run exactly one
aggregator/reducer node. No loop-back cycle, no round-based termination
logic. This is the same "far less new design work, once read closely"
reasoning ADR-0029 itself used to bundle `hierarchical`+`graph`
together, applied to a different bundle here. `debate`, `critic`,
`tree_of_thoughts`, and `planner` remain explicitly out of scope --
each is a genuinely different shape (bounded round loops, beam search
with pruning) -- reaffirming ADR-0029's closing note for those four,
while superseding it for `parallel`/`consensus`/`map_reduce` specifically.

**`parallel`** (`native.py:529-541`/`562-572`, `_run_parallel`/
`_arun_parallel`): no coordinator/worker split -- every step is a peer
agent, run concurrently against the same input. Combined via pure
string formatting (`"\n\n".join(f"[{r.agent_name}]\n{r.content}" for r
in results)`) -- no aggregator agent call at all.

**`consensus`** (`native.py:988-1012`/`1014-1035`, `_run_consensus`/
`_arun_consensus`): `_split_coordinator_and_workers` splits `steps[0]`
(synthesizer) from `steps[1:]` (participants). Participants run
concurrently against the same original input (not each other's
outputs); the synthesizer then combines every independent answer via
`_consensus_prompt`.

**`map_reduce`** (`native.py:1119-1151`/`1153-1183`, `_run_map_reduce`/
`_arun_map_reduce`): requires `map_items=[...]` (a kwarg to
`workflow.run()`/`arun()`, not part of `steps`). `_split_coordinator_and_workers`
splits `steps[0]` (reducer) from `steps[1:]` (mappers). Items are
assigned to mappers round-robin (`mapper_list[i % len(mapper_list)]`)
and run concurrently; the reducer then combines every mapped result via
`_reduce_prompt`, which does `zip(map_items, map_results, strict=True)`
-- alignment between items and results must be exact.

All three already reuse cross-module-ready helpers:
`_split_coordinator_and_workers` is a `@staticmethod` on
`NativeOrchestrator` specifically so `LangGraphOrchestrator` can call it
directly (already does, for `supervisor`/`hierarchical`, per ADR-0016);
`_consensus_prompt`/`_map_prompt`/`_reduce_prompt` are plain
module-level functions in `native.py`, already import-ready the same
way `_supervisor_prompt`/`_reflection_*_prompt` are reused today.

### Verified against the installed `langgraph` package, not assumed

`langgraph 1.2.9` is installed, satisfying the existing `pyproject.toml`
`langgraph>=1.0` floor -- no dependency bump needed.

`StateGraph.add_edge(start_key, end_key)`
(`.venv/Lib/site-packages/langgraph/graph/state.py:915`) accepts
`start_key: str | list[str]`. A `str` start key can be passed multiple
times for a `StateGraph` (its "already found path" guard is skipped for
this class, per the method's own source) -- this is the **fan-out**
mechanism used below. A `list[str]` start key makes the graph "wait for
ALL of the start nodes to complete before executing the end node" (the
method's own docstring) -- this is the **fan-in/join** mechanism an
aggregator node needs. The same docstring says: *"For multiple edges,
use StateGraph with an Annotated state key"* -- confirming a reducer is
required wherever concurrent nodes write to the same state key;
`InvalidUpdateError` (`langgraph/errors.py:90`) is real and fires today
without one.

A real correctness risk was found reading `langgraph/pregel/_algo.py:253-256`:
`apply_writes` does `tasks = sorted(tasks, key=lambda t:
task_path_str(t.path[:3]))` before applying a superstep's writes --
**write order for a reducer channel is node-name lexicographic order,
not declaration order or arrival order.** Naive node names like
`mapper_0..mapper_10` would silently misorder past 10 fan-out nodes
(`mapper_0, mapper_1, mapper_10, mapper_2, ...`). This is an internal
implementation detail, not a documented public-API guarantee, so the
design below sidesteps depending on it entirely (see Decision) rather
than working around it.

Separately, `langgraph/pregel/_runner.py` confirms async node execution
already properly cancels sibling in-flight tasks on a failure -- the
specific "leaves them running unobserved" bug ADR-0031's
`_gather_waiting_for_all` fixed in `native.py`'s raw `asyncio.gather`
usage does not have a langgraph-side counterpart to port; nothing to do
here. And `langgraph/pregel/_runner.py`'s use of `concurrent.futures`
(submitting same-superstep tasks and waiting via
`concurrent.futures.wait(..., return_when=FIRST_COMPLETED)`) confirms
same-superstep fan-out nodes genuinely execute concurrently even under
the synchronous `.invoke()` path, not just `.ainvoke()` -- the design
below gets real concurrency "for free" from langgraph's own runtime, the
same way `native.py`'s `ThreadPoolExecutor`/`_gather_waiting_for_all`
deliver it explicitly.

## Decision

### One shared `_FanOutGraphState`, not three

A fifth `TypedDict` in `langgraph_orchestrator.py`, shared across all
three strategies -- a deliberate deviation from the existing "one shape
per strategy/strategy-pair" pattern (`_GraphState`/
`_DelegationGraphState`/`_ReflectionGraphState`/`_ArbitraryGraphState`),
since these three genuinely share one graph shape:

```python
class _FanOutGraphState(TypedDict):
    task: str
    results: Annotated[list[tuple[int, Any]], operator.add]
    steps: list[Any]
    output: str
```

Every fan-out node writes exactly one `(index, AgentResult)` tuple to
`results` -- its own build-time-assigned index, not derived from node
name. The aggregator node re-sorts by that index before use, which
fully removes any dependency on langgraph's internal (undocumented)
write-ordering described in Context.

New reserved node name: `_AGGREGATOR_NODE = "__aggregator__"`, following
the existing `_COORDINATOR_NODE`/`_FINISH_ROUTE` precedent, reused for
all three strategies' single post-fan-out node.

### Static multi-edge fan-out, not `Send`

`langgraph`'s `Send` API (`langgraph/types.py`) exists in 1.2.9 and is
built for *dynamic, run-time-determined* fan-out against a graph
compiled once and reused across calls. This module never does that --
every `_build_*_graph` method is built fresh inside `run()`/`arun()`,
with all kwargs (including `map_items`) already available before
`StateGraph(...)` is even constructed. So `map_reduce`'s fan-out width
(`len(map_items)`) is exactly as build-time-known as `parallel`/
`consensus`'s (`len(steps)`) -- static per-item node generation in a
plain Python loop is simpler and introduces zero new machinery into a
module that has never used `Send`.

### Three new graph-builder methods

Each follows the existing `_build_*_graph(...) -> Any` shape
(`self._require_langgraph()`, build `StateGraph(_FanOutGraphState)`,
one node per fan-out agent/item + one aggregator, `.compile()`):

- **`_build_parallel_graph(self, steps, **kwargs)`** -- no role split.
  Node names index-suffixed (`f"{agent.name}_{i}"`, since `parallel` has
  no name-uniqueness requirement, unlike `consensus`/`map_reduce`).
  Aggregator node sorts by index and joins with the same
  `"\n\n".join(...)` format `_run_parallel` uses -- no LLM call.
- **`_build_consensus_graph(self, steps, **kwargs)`** -- reuses
  `NativeOrchestrator._split_coordinator_and_workers` (not
  reimplemented) for `synthesizer, participants`, and
  `_reject_reserved_node_names(participants, role=..., reserved=(_AGGREGATOR_NODE,))`
  (already existed for the coordinator/finish names; extended here for
  the aggregator name too). Aggregator node calls
  `synthesizer.run(_consensus_prompt(state["task"], ordered), **kwargs)`.
- **`_build_map_reduce_graph(self, steps, *, map_items, **kwargs)`** --
  raises the identical `ConfigurationException` message
  `_run_map_reduce` raises if `not map_items`. Reuses
  `_split_coordinator_and_workers` for `reducer, mappers`; round-robin
  assignment (`mapper_list[i % len(mapper_list)]`) per item; synthetic
  node names (`f"__mapper_{i}__"`, since the same mapper can appear at
  multiple indices and node names must be unique). Aggregator node
  calls `reducer.run(_reduce_prompt(state["task"], map_items, ordered), **kwargs)`.

### `run()`/`arun()` dispatch

A combined `strategy in ("parallel", "consensus", "map_reduce")` branch
in both methods (not three separate near-identical branches, mirroring
ADR-0029's own reasoning for `supervisor`/`hierarchical`), invoking with
`{"task": input, "results": [], "steps": [], "output": ""}` and
returning `WorkflowResult(content=final_state["output"],
steps=final_state["steps"], orchestrator=self.name, strategy=strategy)`
-- the same shape every other branch already produces. `map_reduce`
pops `map_items` from `kwargs` before building its graph, mirroring how
the `graph` branch already pops `edges`/`max_steps`. The trailing
`raise ConfigurationException(...)` in both methods was extended to
list all eight supported strategies.

## Alternatives considered

- **`Send`-based dynamic fan-out.** Rejected -- see Context/Decision;
  this module never compiles a graph once and reuses it, so `Send`'s
  actual value proposition (fan-out width unknown until invoke time,
  against an already-compiled graph) doesn't apply here.
- **Trusting langgraph's internal reducer-application order directly**
  (append-only `Annotated[list[Any], operator.add]` of bare `AgentResult`s,
  no index). Rejected -- the `_algo.py:253-256` lexicographic-sort
  finding means this would silently misorder past 10 fan-out nodes/items,
  a real, verified bug risk, not a hypothetical one.
- **Per-node dynamically-keyed state fields instead of one shared
  reducer list** (e.g. `result_0`, `result_1`, ... generated per graph
  build). Rejected -- `TypedDict` schemas are static; a truly
  dynamic-width key set would require generating a new `TypedDict` per
  graph-build call, unprecedented in this module and strictly worse
  than the tuple+sort approach.
- **Three separate state shapes** (one per strategy), matching the
  existing "one shape per strategy" pattern strictly. Rejected --
  `parallel`/`consensus`/`map_reduce` are genuinely the same shape (fan
  out, join, aggregate); three near-identical `TypedDict`s would be the
  same kind of duplication ADR-0029 already rejected for
  `_build_supervisor_graph`/`_build_hierarchical_graph`.

## Consequences

### Positive

- `sequential`/`supervisor`/`reflection`/`hierarchical`/`graph`/
  `parallel`/`consensus`/`map_reduce` are all now shipped on both
  `native` and `langgraph` backends -- `debate`/`critic`/
  `tree_of_thoughts`/`planner` are the only strategies remaining
  native-only.
- `workflow.parallel()`/`.consensus()`/`.map_reduce()` now work
  unmodified across both backends via `.use_langgraph()`, the same win
  prior langgraph ADRs delivered for other strategies.
- Genuine concurrency confirmed at the langgraph runtime level (not
  just correctness) -- same-superstep fan-out nodes execute via
  `concurrent.futures` under the hood, even for the synchronous
  `.invoke()` path.
- New test coverage goes beyond straight native-test mirroring: a
  10-plus-fan-out-node ordering test per strategy (proving the
  index+sort design against the verified lexicographic-write-order
  risk, not assuming it away) and a native-vs-langgraph parity test per
  strategy (running the identical `Workflow` through both backends and
  asserting matching content/step order) -- a stronger cross-backend
  proof than `supervisor`/`reflection`/`hierarchical`/`graph`'s existing
  tests currently provide for themselves.

### Negative / risks

- A fifth distinct `TypedDict` state shape in this module, though this
  one is shared by three strategies rather than one -- net one new
  shape for three new strategy/backend combinations, a better ratio
  than any prior langgraph ADR in this file.
- `_FanOutGraphState.results` (the reducer channel) is never explicitly
  cleared after the aggregator reads it. Harmless since every compiled
  graph in this module is single-use, built fresh per `run()`/`arun()`
  call, but worth noting so a future reader doesn't assume otherwise.
- The design's correctness for output ordering rests on reading
  `langgraph/pregel/_algo.py`, an internal module not covered by
  langgraph's public API guarantees. The `(index, result)` + explicit
  sort mitigation is specifically chosen to be robust to that internal
  behavior changing in a future langgraph release (correctness no
  longer depends on it at all), but the underlying finding itself was
  reverse-engineered from source, not from documentation.

### Follow-ups

- `debate`, `critic`, `tree_of_thoughts`, and `planner` remain
  native-only, reaffirming ADR-0029's closing note for these four
  specifically. Each is a genuinely different shape (bounded round
  loops with accumulating transcript state; a fixed 2-agent alternating
  loop; beam search with per-level pruning and early termination) and
  will be separate follow-up ADRs, at this repo's established
  one-or-two-strategies-per-ADR pace.
