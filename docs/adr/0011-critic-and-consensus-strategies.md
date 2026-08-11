
# 0011. Critic and consensus multi-agent strategies

Status: Accepted
Date: 2026-08-11

## Context

`ROADMAP.md` listed "Debate / critic / consensus strategies",
"Hierarchical strategy", "Map-reduce strategy", and "Tree-of-thoughts
strategy" as 📋. ADR-0007 (reflection/planner/supervisor) explicitly
deferred all six without designing any of them, saying only that each
"should decide explicitly whether it fits the coordinator/worker
convention established here or needs its own shape."

This ADR covers the first pass of two strategies; the remaining four are
being deliberately split across further passes rather than attempted
together (see Decision).

## Decision

### Scope: critic + consensus this pass; debate + map-reduce next; hierarchical + tree-of-thoughts deferred indefinitely

Evaluated all six against the existing flat `steps: Sequence[Agent]`
model (`NativeOrchestrator.run`/`.arun`, `requisite/orchestrators/native.py`):

- **Critic** and **consensus** are direct generalizations of strategies
  already shipped -- see below. Shipped this pass.
- **Debate** (agents responding to each other over rounds) and
  **map-reduce** (parallel workers over multiple work items, then a
  reduce step) also fit the flat model, but map-reduce specifically
  needs `Workflow.run()` to accept multiple work items instead of one
  shared `input: str` -- a real, separate design question best not
  rushed alongside this pass. Next pass.
- **Hierarchical** (supervisors delegating to other supervisors, not
  just leaf workers) and **tree-of-thoughts** (branching, scoring, and
  pruning a search tree of partial solutions) do not fit the flat
  coordinator/worker list without materially bigger structural changes
  -- nested sub-workflows for the former, a whole branching-search loop
  for the latter. Deferred with no design commitment yet; forcing a
  simplified version of either into this pass would produce something
  that doesn't actually deliver what "hierarchical" or "tree-of-thoughts"
  promise.

### Critic: reflection generalized to two agents

`_run_critic`/`_arun_critic` (`native.py`) requires exactly two agents:
`steps[0]` (generator), `steps[1]` (critic). Structurally identical to
`_run_reflection`'s draft → critique → revise loop (`max_rounds`,
default 3; the `NO_CHANGES_NEEDED` plain-string sentinel for early stop;
no exception on round exhaustion, just returns the last draft) --
**the only change is that the critique call goes to a separate agent
instead of the same worker critiquing its own output.** `_critic_prompt`
is a new, critic-specific prompt (worded for an external reviewer rather
than self-critique); the revision step reuses `_reflection_revise_prompt`
unchanged, since incorporating a critique doesn't care who wrote it.

### Consensus: parallel + the existing coordinator/worker split

`_run_consensus`/`_arun_consensus` (`native.py`) reuses
`_split_coordinator_and_workers` (already shared by planner/supervisor)
unchanged: `steps[0]` = synthesizer, `steps[1:]` = participants (≥2
agents, unique names -- the same validation planner/supervisor already
enforce, inherited for free). Participants run concurrently on the same
input using the exact `ThreadPoolExecutor`/`asyncio.gather` pattern
`_run_parallel`/`_arun_parallel` already use -- consensus *is* parallel
execution plus one more step. The synthesizer then combines every
participant's answer via `_consensus_prompt`, a new prompt listing each
participant's name and answer and asking for one final, reconciled
answer.

**No `response_model=` for the synthesis step.** Planner/supervisor use
structured output because their coordinator is making a *routing*
decision (which worker, what action) that code needs to parse
reliably. Consensus's synthesizer is producing free-form prose (the
actual final answer), exactly like reflection/critic's revision step --
neither of those uses structured output either, for the same reason:
there's nothing to parse, just text to return directly as
`WorkflowResult.content`.

**No `max_rounds` parameter.** Consensus is single-pass by design:
participants answer once, the synthesizer combines once. There's no
round loop to bound.

## Alternatives considered

- **Consensus resolved by voting/similarity matching instead of an LLM
  synthesizer.** Rejected for v1 -- would need a similarity metric or
  exact-match voting scheme that degrades badly for free-form text
  answers (two correct answers phrased differently "disagree" under
  exact match). An LLM synthesizer handles paraphrase-level agreement
  and can weigh reasoning quality, which a voting scheme can't.
- **Critic strategy using `response_model=` for the critique** (e.g. a
  structured `{needs_changes: bool, issues: list[str]}`) instead of the
  plain-text `NO_CHANGES_NEEDED` sentinel. Rejected to match
  `_run_reflection`'s own established precedent exactly (ADR-0007
  already decided this wasn't worth a Pydantic model for one boolean) --
  introducing a different convention for the same kind of decision in a
  closely related strategy would be inconsistent for no real benefit.
- **A shared `_run_multi_agent_round_loop` helper** extracted from
  reflection/critic/supervisor's near-identical round-loop shapes.
  Rejected -- the three loops differ enough in what they pass to
  `response_model=`, what terminates them, and what they accumulate
  (`transcript` tuples vs. a flat `results` list) that a shared
  abstraction would need enough parameters/hooks to end up more complex
  than the ~15 lines of duplication it would remove. Revisit if a
  fourth round-based strategy makes the duplication clearly worse than
  the abstraction would be.

## Consequences

### Positive

- Both strategies took no changes outside `native.py` (strategy logic)
  and `workflow.py` (`_KNOWN_STRATEGIES` + convenience methods) -- same
  "nothing else needs to change" property ADR-0007 established.
- Both reuse existing, already-tested building blocks
  (`_split_coordinator_and_workers`, the parallel-execution pattern, the
  reflection round-loop shape) rather than inventing new ones -- lower
  risk, less code to review.

### Negative / risks

- Consensus's synthesizer receives every participant's full answer in
  one prompt -- for many participants or long answers this could exceed
  a model's context window. No token-budget guard exists (same
  limitation already accepted for `LLMReranker` in ADR-0010, for the
  same reason: callers are expected to keep the fan-out reasonably
  small).

### Follow-ups

- Debate + map-reduce as the next pass (map-reduce specifically needs a
  `Workflow.run()` API decision for multiple work items).
- Hierarchical and tree-of-thoughts each need their own dedicated design
  pass -- not scoped here.
