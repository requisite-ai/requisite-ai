
# 0012. Debate and map-reduce multi-agent strategies

Status: Accepted
Date: 2026-08-11

## Context

Second of the two passes over the strategies ADR-0007 deferred and
ADR-0011 split into "fits the existing convention now" vs. "needs its
own design" (`ROADMAP.md`). Debate fits the flat `steps: Sequence[Agent]`
model directly, the same way critic/consensus did. Map-reduce fits too,
but raises a real design question every prior strategy sidestepped:
every strategy so far shares one `input: str` across all agents, while
map-reduce needs *multiple* work items to map over.

## Decision

### Debate: coordinator/worker split + per-round concurrency, reusing the consensus/parallel pattern

`_run_debate`/`_arun_debate` (`requisite/orchestrators/native.py`) reuse
`_split_coordinator_and_workers` unchanged: `steps[0]` = moderator
(never debates, only delivers the final verdict), `steps[1:]` = debaters.
No extra "≥2 debaters" validation beyond what `_split_coordinator_and_workers`
already enforces (≥1 worker) -- consistent with consensus not requiring
≥2 participants either (ADR-0011); a one-debater "debate" degenerates
harmlessly into something like critic-via-moderator.

Each round, every debater sees every debater's arguments from the
*previous* round only, then responds -- this is the standard multi-agent
debate pattern (an agent reacting to peers' prior positions, not a
strict turn order), and it's what makes each round safely runnable
concurrently: nothing written during the current round is read until
the *next* round, so there's no read/write race within a round.
Concurrency reuses the exact `ThreadPoolExecutor`/`asyncio.gather`
pattern `_run_parallel`/`_run_consensus` already use. After
`max_rounds` (default 3, matching critic/reflection), the moderator
reviews the full transcript and delivers one final verdict via plain
text (`_debate_verdict_prompt`) -- no `response_model=`, same reasoning
as consensus's synthesis: there's nothing to parse, just prose to
return.

No stance assignment (no built-in "argue for X" / "argue against X").
That composes naturally through each debater `Agent`'s own
`system_prompt` -- consistent with ADR-0007's decision not to add new
per-strategy `Agent` fields, and avoids inventing a stance-configuration
API that would only serve this one strategy.

No exception on `max_rounds` exhaustion -- same as critic/reflection/
consensus, there's no "did we finish" terminal state to fail to reach;
the verdict is always produced after the loop.

### Map-reduce: work items via `map_items=` kwarg, `input` unchanged

**The core decision.** Every strategy's `input: Optional[str]` stayed
exactly as-is -- `Workflow.run`/`.arun`, `NativeOrchestrator.run`/`.arun`
signatures are unchanged. `input` keeps meaning "the overall task/goal"
(used to frame both the map and reduce prompts). The list of items to
map over is a new keyword-only parameter, `map_items: Sequence[str]`,
threaded through the same `**kwargs` channel `max_rounds` already uses.
`_run_map_reduce` declares `*, map_items: Optional[Sequence[str]] = None,
**kwargs` so Python's own keyword-argument binding excludes `map_items`
from the `**kwargs` forwarded to each agent's `run()`/`arun()` call --
the exact mechanism `max_rounds` already relies on, no special-casing
needed. Missing or empty `map_items` raises `ConfigurationException`
with a clear message rather than surfacing a bare `TypeError`.

`steps[0]` = reducer, `steps[1:]` = mappers (reused
`_split_coordinator_and_workers`, role="map-reduce reducer"). Items are
assigned to mappers **round-robin** (`mappers[i % len(mappers)]`), so
`len(map_items)` doesn't need to equal `len(mappers)` -- the realistic
case is more items than workers. All items run concurrently regardless
of how many distinct mapper agents there are.

**Concurrent calls can land on the same mapper `Agent` instance.**
Accepted as safe: each `Agent.run()`/`.arun()` call builds its own
message list fresh with no mutable state shared across calls, beyond a
provider's lazily-constructed HTTP client (`BaseProvider._get_client()`
and friends across all 8 providers) -- every integrated SDK (openai,
anthropic, google-genai, ollama) documents its client as safe for
concurrent request use once constructed. The only race is on the very
first concurrent calls to a not-yet-constructed client, where two
threads could redundantly construct one each (Python's GIL prevents the
attribute assignment itself from tearing, so this can't corrupt state,
only waste one construction) -- a pre-existing, framework-wide
characteristic of every provider's lazy-init pattern, not something new
introduced here, and not worth adding a lock for on this feature's
account alone.

## Alternatives considered

- **Loosen `input`'s type to `Union[str, Sequence[str]]` across every
  strategy** so map-reduce could receive its items via `input` directly.
  Rejected -- would force every *other* strategy's `run()`/`arun()` to
  explicitly reject a list input it never wanted to accept in the first
  place, purely to accommodate one strategy's different shape. The
  `map_items=` kwarg keeps every existing strategy's contract completely
  unchanged.
- **Strict turn-taking within a debate round** (debater 2 sees debater
  1's response from the *same* round before responding) instead of
  round-based concurrency. Rejected -- meaningfully slower (fully
  sequential instead of concurrent per round) for a debate-quality
  benefit that isn't clearly worth it; the "see the previous round"
  model is the established pattern in multi-agent-debate literature and
  reuses the framework's existing concurrency primitives directly.
- **Require `len(map_items) == len(mappers)`** (one item per mapper,
  no round-robin). Rejected -- too restrictive for the realistic case
  of more work items than worker agents, which is the whole point of
  "map" in map-reduce.
- **A lock around each provider's lazy client construction**, to close
  the harmless first-call race described above. Rejected as
  out-of-scope for this feature -- it's a pre-existing characteristic of
  every provider, not something map-reduce's round-robin reuse
  introduces; fixing it (if ever needed) belongs in the provider layer,
  touching all 8 providers, not bundled into an orchestration-strategy PR.

## Consequences

### Positive

- Both strategies took no changes to any existing method signature
  (`Workflow.run`/`.arun`, `NativeOrchestrator.run`/`.arun` all
  unchanged) -- purely additive via the established `**kwargs`
  extension point and the `_run_<strategy>`/`_arun_<strategy>` pattern.
- Both reuse `_split_coordinator_and_workers` and the
  `ThreadPoolExecutor`/`asyncio.gather` concurrency pattern already
  proven by `parallel`/`consensus` -- no new concurrency primitive
  introduced.

### Negative / risks

- Debate's per-round prompt includes the full transcript so far, which
  grows linearly with `max_rounds * len(debaters)` -- for many debaters
  or many rounds this could get long enough to matter for context-window
  budget. No truncation/summarization exists; same accepted limitation
  as consensus's and `LLMReranker`'s un-bounded prompt size (ADR-0010,
  ADR-0011).
- Map-reduce's round-robin assignment means a mapper agent's own
  `system_prompt`/configuration must make sense for *any* item it might
  receive, since which specific items land on which agent isn't
  controllable by the caller beyond ordering `map_items`.

### Follow-ups

- Hierarchical and tree-of-thoughts remain deferred, unscoped -- neither
  fits the flat coordinator/worker list without materially bigger
  structural changes (nested sub-workflows; a branching/scoring/pruning
  search loop).
- If a provider's lazy-client-construction race ever becomes a real
  problem (not just theoretical), add locking in `BaseProvider`/each
  provider's `_get_client()`, not per-strategy.
