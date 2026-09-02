
# 0038. Cost-based spend limiting: `CostLimiter`

Status: Accepted
Date: 2026-09-01

## Context

`RateLimiter` (ADR-0008) paces *request rate* against a provider's own
"N requests per minute" quota. It says nothing about real dollar
*spend* -- a caller running plenty of well-paced calls can still burn
through a meaningful budget. ADR-0008 itself named the exact trigger
for revisiting its own "no `BaseRateLimiter` interface" call: "revisit
if a real need for a different algorithm ... shows up." A cost-based
limiter is precisely that -- a materially different algorithm (reactive
spend accounting, not a proactive sliding-window gate), confirming a
new, separate `CostLimiter` class is the right shape rather than
retrofitting `RateLimiter` itself. Nothing in ADR-0008 discussed or
deferred cost-based limiting as an alternative -- this is genuinely new
scope, not a previously-rejected idea being revived.

Verified directly against current source before designing this: every
non-streaming provider response already carries real token counts --
`ChatResponse.usage: Usage` (`prompt_tokens`/`completion_tokens`/
`total_tokens`) is populated by all 5 first-party providers (OpenAI,
Gemini, Anthropic, Ollama), inherited verbatim by every
OpenAI-wire-compatible subclass (Groq, Azure OpenAI, OpenRouter,
Together). `StreamChunk` carries no usage data on any provider today.
No pricing table or cost concept existed anywhere in the package before
this ADR.

## Decision

### Reactive, not proactive -- the one real asymmetry vs. `RateLimiter`

`RateLimiter.acquire()` gates a call *before* it happens because
request count is fully knowable in advance. Dollar cost cannot be:
prompt-token cost is knowable before a call, but completion-token cost
only exists once the provider responds. `CostLimiter.check()` therefore
only raises before a call once *already-recorded* spend has reached the
budget; `CostLimiter.record()` updates cumulative spend after each call
completes. One call can still push spend over budget -- every call
after that raises immediately. Stated plainly rather than implying a
stronger guarantee: this design cannot strictly prevent ever exceeding
`budget_usd`, only stop further calls once it's gone.

One consequence: `CostLimiter` never blocks or waits (unlike
`RateLimiter.acquire()`, which can sleep), so it needs only one plain
`threading.Lock`, not `RateLimiter`'s separate `asyncio.Lock` for the
async path -- holding a lock across a lock-and-release with nothing
awaited inside it is safe from either sync or async code.

### Caller-supplied `cost_fn`, not a maintained price table

```python
CostFn = Callable[["Usage", str], float]  # (usage, model) -> dollars

def cost_per_token(*, prompt_rate_per_1k: float, completion_rate_per_1k: float) -> CostFn: ...

class CostLimiter:
    def __init__(self, *, budget_usd: float, cost_fn: CostFn) -> None: ...
    def check(self) -> None: ...          # raises CostLimitException if already exhausted
    def record(self, usage: Usage, model: str) -> float: ...  # updates spend, returns $ recorded
    def reset(self) -> None: ...          # zeroes spend for a new budget period
    spent_usd: float                       # property
    remaining_usd: float                   # property, never negative
```

Pricing is a caller-supplied `cost_fn`, mirroring the `evaluator=`
callable pattern the `reflexion` strategy already established
(ADR-0036) -- no per-model dollar rates are shipped or maintained by
the framework. `cost_per_token(...)` covers the common flat-rate case
as a one-liner; a caller with tiered or negotiated pricing writes their
own `cost_fn` instead.

### Fixed total budget, manual `reset()` -- no built-in calendar period

`CostLimiter(budget_usd=10.0)` tracks cumulative spend until code calls
`.reset()` explicitly. No day/month period logic lives in the
framework -- a caller who wants a recurring budget calls `.reset()` on
their own schedule (a cron job, a scheduled task).

### Wiring: `AI`/`Agent`, non-streaming methods only

`AI.__init__` gains `cost_limiter: Optional[CostLimiter] = None`
(no `Settings`-based auto-build the way `RATE_LIMIT_RPM` builds a
default `RateLimiter` -- a pricing function can't come from a scalar
env var). `chat_response`/`achat_response` call `cost_limiter.check()`
before `rate_limiter.acquire()` (fail fast on an already-exhausted
budget without waiting on rate capacity first), and
`cost_limiter.record(response.usage, response.model)` after a
successful call. `stream`/`astream`/`stream_response`/
`astream_response` deliberately do not check or record -- `StreamChunk`
carries no usage data, so there is nothing to enforce or record yet;
each streaming method's docstring says so explicitly rather than
leaving this as a silent gap. `Agent.__init__` gains the same
`cost_limiter=` parameter, forwarded into its internal `AI(...)`
exactly the way `rate_limiter` already is -- `Agent` itself never calls
`.check()`/`.record()` directly.

### Telemetry

A `requisite.ai.cost` OTel counter (unit `"usd"`) joins the existing
`requisite.ai.requests`/`requisite.ai.request.duration`/
`requisite.ai.tokens` instruments on the same `requisite.ai` meter,
recorded in `chat_response`/`achat_response` right after
`cost_limiter.record(...)` returns its dollar amount, tagged with the
same `requisite.provider`/`requisite.model` attributes the token
counter already uses.

## Alternatives considered

- **Pre-call cost estimation via a tokenizer**, to bound spend
  proactively like `RateLimiter` bounds rate. Rejected: this would need
  a real tokenizer per provider (a new dependency, one per SDK), and
  even then could only estimate the *prompt* side -- completion cost is
  fundamentally unknowable before the provider responds regardless. It
  would add real complexity without actually delivering a stronger
  guarantee than the reactive design already gives.
- **A framework-maintained default price table** for common models.
  Rejected: prices change, and new models ship, faster than this
  project could realistically keep a table accurate, with no runtime
  way to detect that it had drifted -- a caller trusting a stale table
  is worse than a caller who must supply their own known-current rate.
- **Built-in rolling calendar period** (`period="daily"`/`"monthly"`).
  Rejected for v1: real timezone and period-boundary edge cases for the
  framework to get right, for no benefit a caller can't trivially get
  themselves by calling `.reset()` on their own schedule.
- **Folding this into `RateLimiter` itself** (e.g. an optional
  `budget_usd=` parameter). Rejected: the enforcement shape is
  genuinely different (reactive check-then-record vs. proactive
  sliding-window gate), and `RateLimiter`'s own design (ADR-0008) is
  already fully specified around one concrete algorithm; bolting a
  second, structurally different one onto the same class would blur
  both rather than clarify either.

## Consequences

### Positive

- Composes independently alongside `RateLimiter` -- a caller can want
  both a rate cap and a spend cap on the same `Agent`/`AI` at once,
  and each is a separate, optional constructor argument.
- Reuses `Usage`/`ChatResponse` plumbing verbatim -- zero changes to
  any of the 5 provider files, the same "this is correctly an
  `AI`-facade-level concern" finding ADR-0008 already reached for rate
  limiting.
- Adversarially verified before the permanent test suite (7 checks):
  basic accrual/exhaustion/reset; a shared instance across two agents
  tracks *combined* spend correctly (the cost-budget analogue of
  ADR-0008's own "several agents sharing one real quota" concern); a
  broken `cost_fn`'s exception propagates cleanly, not swallowed;
  `budget_usd` validation; thread-safety under concurrent `record()`
  calls (no lost updates); the async path raises cleanly once
  exhausted; streaming calls are unaffected by an already-exhausted
  `cost_limiter` (true no-op, not a silent block).

### Negative / risks

- Cannot strictly guarantee `budget_usd` is never exceeded -- the call
  that crosses the threshold always completes first; only calls after
  it are blocked. An honest limitation of dollar-cost accounting given
  completion cost isn't knowable pre-call, not an implementation gap.
- No cross-process sharing, the same limitation `RateLimiter` already
  has -- one `CostLimiter` instance only sees calls made through
  `AI`/`Agent` instances that share that exact object in the same
  process.
- Streaming calls are not covered at all yet -- `StreamChunk` needs a
  `usage` field first, a separate cross-cutting change touching all 5
  providers' streaming paths.

### Follow-ups

- Streaming usage capture (`StreamChunk.usage`) across all 5 providers,
  then wiring `CostLimiter` into `stream`/`astream`/`stream_response`/
  `astream_response` on top of it.
- A possible opt-in, clearly-dated convenience price table as a
  separate, explicitly-non-load-bearing module, only if real demand for
  an out-of-the-box option shows up -- not shipped now, per the
  "Alternatives considered" reasoning above.
