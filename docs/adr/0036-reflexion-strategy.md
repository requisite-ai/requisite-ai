
# 0036. Reflexion multi-agent strategy

Status: Accepted
Date: 2026-09-01

## Context

Keyan asked for an assessment of the published Reflexion technique
(Shinn et al.; see https://www.promptingguide.ai/techniques/reflexion)
against what this framework already ships. Reflexion composes an
Actor (attempts the task), an Evaluator (scores the attempt, often via
an external/deterministic signal like test execution rather than an
LLM's own opinion), and a Self-Reflection model (turns that score into
a specific, actionable verbal lesson), with the lesson persisted and
folded into the *next*, independent attempt at the *whole task*.

Checked against `reflection` (ADR-0007) and `critic` (ADR-0011), the
two existing self-improvement strategies, and confirmed genuinely
distinct: both of those revise the *same* draft, in a loop, within one
`.run()` call, and both use an LLM's own free-text judgment (the
`NO_CHANGES_NEEDED` sentinel) as their only stopping signal. Neither
supports re-attempting a task from scratch across independent trials,
and neither has any concept of a pluggable, non-LLM evaluator. This gap
was confirmed absent from `ROADMAP.md`, `FEATURES.md`, and every prior
ADR before this one was written.

Scope for this ADR, per Keyan's explicit choice: **native backend
only**. Every other net-new strategy in this repo (`reflection`
ADR-0007; `critic`/`consensus` ADR-0011; `debate`/`map-reduce`
ADR-0012; `hierarchical` ADR-0013; `tree_of_thoughts` ADR-0018; `graph`
ADR-0019) shipped natively first, with langgraph parity following later
as its own dedicated ADR -- this follows that same pattern rather than
the langgraph-parity-for-an-existing-strategy pattern the immediately
preceding several ADRs (0032-0035) were doing. Also per Keyan's
explicit choice, accumulated reflections are scoped to **one `.run()`
call only** -- no persistent cross-call memory backend in this pass.

## Decision

### Strategy shape: single agent, like `reflection`

`_run_reflexion`/`_arun_reflexion` (`native.py`) require exactly one
agent (`len(steps) != 1` raises `ConfigurationException`), the same
validation shape `_run_reflection`/`_run_critic` already use. The one
agent plays Actor, and -- when no custom evaluator is supplied --
Evaluator and Self-Reflection too, via different prompts against its
own `.ai`, matching how `reflection`'s single agent already plays both
drafter and critic roles.

### The new public contract: `EvaluationResult` + `Evaluator`

Added to `requisite/orchestrators/base.py`, next to `WorkflowResult`:

```python
class EvaluationResult(BaseModel):
    success: bool
    feedback: str

Evaluator = Callable[[str, str], EvaluationResult]  # (task, attempt_content) -> EvaluationResult
```

This lives in `base.py`, not `native.py`, and is exported publicly from
`requisite/__init__.py` -- unlike `_Plan`/`_SupervisorDecision`/
`_ThoughtEvaluation` (private, `native.py`-internal, because the
*framework* constructs them from a structured-output call the caller
never sees), `EvaluationResult` is constructed *by the caller's own
evaluator function* and handed back to the framework, so it has to be
a public, stable contract. `(task, attempt_content)` mirrors the
existing `(task, draft)` convention every prompt-builder in `native.py`
already uses (`_reflection_critique_prompt`, `_critic_prompt`), so a
custom evaluator's signature reads consistently with the rest of the
framework's own functions.

### Default evaluator: the same agent, via structured output

When `evaluator=` is omitted, `_run_reflexion` falls back to
`worker.ai.chat(_reflexion_default_evaluation_prompt(...),
response_model=EvaluationResult)` -- the exact same
`agent.ai.chat(prompt, response_model=X)` pattern `planner`,
`supervisor`, and `tree_of_thoughts` already use for their own
structured decisions. This means `.reflexion()` works out of the box
with zero setup, the same "sensible built-in default, swap in something
real" shape `agent.requires("weather")` already establishes for
capabilities -- and the technique's actual published strength (running
real tests as the reward signal) is exactly what swapping in a custom
`evaluator=` unlocks.

### The loop

```python
for trial in range(max_trials):
    attempt = worker.run(self._task_prompt_with_context(input, reflections), **kwargs)
    evaluation = evaluator(input, attempt.content) if evaluator else worker.ai.chat(..., response_model=EvaluationResult)
    if evaluation.success:
        succeeded = True
        break
    if trial < max_trials - 1:
        reflection = worker.run(_reflexion_reflect_prompt(input, attempt.content, evaluation.feedback), **kwargs)
        reflections.append(reflection.content)
```

Reuses `NativeOrchestrator._task_prompt_with_context(task,
context_notes)` directly for folding accumulated reflections into the
next attempt's prompt -- already exactly "task + prior notes, joined",
the same helper `planner` uses for its own per-step context, no new
prompt-builder needed for that part. Two new prompts were added,
`_reflexion_default_evaluation_prompt` (only used by the default
evaluator path) and `_reflexion_reflect_prompt` (always used, whether
the evaluator was custom or default -- turning raw evaluator feedback
into a specific, actionable lesson is the Self-Reflection model's job
in the source technique, distinct from evaluation itself).

No reflection is generated after the final trial (`trial < max_trials -
1`), matching `_run_reflection`'s existing "don't waste a call
critiquing an answer nothing will read" behavior for its own last
round.

### `WorkflowResult` gains one new optional field

```python
succeeded: Optional[bool] = None
```

Additive: `WorkflowResult` is a plain, non-frozen `BaseModel` with no
`extra="forbid"`, and every existing strategy's `WorkflowResult(...)`
call only passes the pre-existing four fields by keyword, so this
cannot break any of them. `reflexion` is the only strategy that sets
it; every other strategy leaves it `None`. This resolves what every
other strategy has to make the caller do themselves (parse `.content`
or count `.steps` to infer success) -- reflexion has an explicit
evaluator-produced signal worth surfacing directly rather than making
callers re-derive it.

### `Workflow` wiring

`"reflexion"` added to `_KNOWN_STRATEGIES`; a `.reflexion()` chainable
method added next to `.tree_of_thoughts()`, same
docstring-plus-`self._strategy=...`-plus-`return self` shape every
other strategy method uses. The class docstring's strategy list and its
backend-support summary were updated -- and, in the same edit, a real
staleness bug was fixed: that summary still said `planner`/`critic`/
`consensus`/`debate`/`map_reduce`/`tree_of_thoughts` were native-only,
which stopped being true across ADR-0032 through ADR-0035 but was never
corrected in `workflow.py`'s own docstring (only `native.py`'s and
`langgraph_orchestrator.py`'s docstrings were kept current during those
rounds). Now reads accurately: every strategy except `reflexion` runs
on both backends.

## Alternatives considered

- **Extending `critic` to support "re-attempt from scratch" instead of
  a new strategy.** Rejected -- `critic` revises the same draft by
  design (ADR-0011 was explicit that this mirrors `reflection`
  intentionally); forcing a "start over" mode into the same strategy
  name would mean two materially different behaviors sharing one
  string, which is worse for callers than two distinctly named
  strategies with distinct, honest semantics.
- **A single-argument evaluator, `Callable[[str], EvaluationResult]`
  (just the attempt content, no task).** Rejected -- every other
  prompt-builder/evaluator-shaped function in `native.py` takes `task`
  first, and a real evaluator (e.g. "run the right test file for this
  spec") plausibly needs the task to know what it's even checking,
  rather than relying on a closure to smuggle that in.
- **Persistent cross-call reflection memory (a `memory=` kwarg using
  `BaseMemory`) from day one.** Deferred, per Keyan's explicit scope
  decision -- the paper's own usage is bounded within solving one task
  instance across several trials, not a training regime spanning
  separate processes; the in-call `reflections: list[str]` (same
  `context_notes` pattern `planner` already uses) covers that directly
  with no new abstraction. A `BaseMemory`-backed version is a natural,
  separate follow-up if a real need for it shows up.
- **LangGraph parity in this same pass.** Deferred, per Keyan's
  explicit scope decision and matching every other net-new strategy's
  own precedent (native first, langgraph as its own later ADR). The
  shape is already recognizable as another instance of the bounded
  loop-back cycle `reflection`/`critic`/`planner` each already
  established on langgraph, so it should be a cheap follow-up whenever
  it happens, not a hard one.

## Consequences

### Positive

- No changes outside `orchestrators/base.py` (the new public contract
  + one additive `WorkflowResult` field), `native.py` (strategy logic),
  `workflow.py` (`_KNOWN_STRATEGIES` + convenience method), and
  `requisite/__init__.py` (exports) -- the same "nothing else needs to
  change" property every strategy addition in this repo has had since
  ADR-0007.
- Reuses `_task_prompt_with_context`, the `agent.ai.chat(...,
  response_model=...)` structured-output pattern, and the
  `ConfigurationException`-on-wrong-agent-count validation shape
  directly -- no new cross-cutting machinery, all adversarially tested
  (six checks: never-succeeds exhausts `max_trials` cleanly with no
  dangling reflection after the last trial; succeeds mid-way and folds
  the prior reflection into the next attempt's prompt correctly; a
  raising evaluator propagates cleanly rather than being swallowed; the
  one-agent requirement raises cleanly; the default structured-output
  evaluator path works; the async path works) before the permanent
  pytest suite was written.
- Calling `.reflexion().use_langgraph()` today raises the langgraph
  orchestrator's own existing "supports the '...' strategies (got
  'reflexion')" `ConfigurationException` with no extra code needed --
  the same clean-rejection behavior every strategy has had before its
  own langgraph counterpart shipped.

### Negative / risks

- No token-budget guard on accumulated reflections -- for a large
  `max_trials`, the prompt folding every prior lesson into each new
  attempt could grow large enough to matter for a model's context
  window. Same accepted-limitation category as `LLMReranker`'s
  unbounded fan-out prompt (ADR-0010) and consensus's unbounded
  participant-answer prompt (ADR-0011): callers are expected to keep
  `max_trials` reasonably small, no automatic truncation exists.
- The default (no custom `evaluator=`) path is exactly the LLM-judging-
  itself pattern the Reflexion paper's own real gains come from moving
  *away* from -- it exists purely so the strategy works with zero setup,
  not because it's expected to be the common real-world usage. This is
  the same trade-off `capabilities`' default resolvers already accept
  (a keyless built-in that works immediately, a real integration is
  expected to replace it for anything that matters).

### Follow-ups

- LangGraph parity for `reflexion`, matching the pattern every other
  strategy's langgraph counterpart has followed (a bounded loop-back
  cycle: attempt -> evaluate -> conditional edge on success/trials-
  exhausted -> reflect -> back to attempt).
- Optional persistent cross-call reflection memory via `BaseMemory`
  (most naturally `VectorMemory`, for semantic recall across many
  stored reflections), if a real need for reflections surviving beyond
  one `.run()` call shows up.
