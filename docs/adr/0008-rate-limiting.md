
# 0008. Proactive rate limiting for provider calls

Status: Accepted
Date: 2026-08-07

## Context

Running `examples/workflow_example.py` against a free-tier Gemini key
(15 requests/minute) hit `429 RESOURCE_EXHAUSTED`. The immediate ask was
"make only N API calls per minute; add a config value; make the app
wait" -- but the actual failure mode was more specific than "one agent
calls too fast": `workflow_example.py` builds four independent `Agent`
instances (Researcher, Writer, Planner, Supervisor), all backed by the
same Gemini API key and therefore the same real quota. A rate limiter
scoped to a single `Agent`/`AI` instance would not have fixed this --
each instance would independently believe it had its own 15/min budget
while the real, shared quota is 15/min total across all of them. Any
design that didn't make *sharing* a first-class, easy thing to do would
reproduce the same bug under a different name.

## Decision

### One concrete `RateLimiter` class, not an interface + registry

`requisite/core/rate_limiter.py` ships a single concrete `RateLimiter`
class with `acquire()` (sync) / `aacquire()` (async). This deliberately
does not follow the `Base<Noun>` + registry pattern used for
`BaseProvider`/`BaseOrchestrator`/etc. Per ADR-0001's own precedent
(`Tool` and `Agent` are concrete classes, not ABCs, "since there's only
one reasonable shape for each"), a rate limiter has one well-understood
shape for v1; a registry of pluggable rate-limiting *backends* would be
premature abstraction with no second implementation to justify it.

### Sliding-window log, not a token bucket

`RateLimiter` tracks a `deque[float]` of monotonic timestamps for calls
in the trailing 60 seconds. `acquire()`/`aacquire()` evict expired
entries and either claim a slot immediately or compute exactly how long
until the oldest entry ages out of the window. This was chosen over a
token-bucket algorithm because it matches how the actual failure mode
is specified and enforced -- Gemini's error literally reports "N
requests per minute" against a rolling window -- so the limiter models
the real constraint directly rather than approximating it with a
bucket-refill rate that would need its own tuning to match.

### Thread-safe and async-safe, not cross-safe between the two

`acquire()` is guarded by a `threading.Lock` (required: `NativeOrchestrator`'s
`parallel` strategy runs agents via `ThreadPoolExecutor`, so several
threads can call `acquire()` on a shared limiter concurrently).
`aacquire()` is guarded by a separate `asyncio.Lock` for the same reason
on the async/`asyncio.gather` path. A single `RateLimiter` instance used
concurrently from *both* the sync and async path at once is not
cross-safe between those two locks -- a known, documented limitation,
not silently unsupported. Nothing in the codebase mixes sync and async
calls against the same object within a single run, so this wasn't worth
the real complexity (a unified lock strategy spanning both worlds) that
supporting it properly would require.

### Sharing is explicit constructor injection, not implicit global state

`AI.__init__` gains `rate_limiter: Optional[RateLimiter] = None`, forwarded
through every provider-call path (`chat_response`, `achat_response`,
`stream`, `astream` -- confirmed by reading `requisite/ai.py` that these
four methods are the only points where every public `AI` method
eventually reaches `self._provider`, so instrumenting these four covers
`chat`/`achat` too, which already delegate to `chat_response`/`achat_response`).
`Agent.__init__` gains the same parameter, forwarded into its internal
`AI(...)` construction.

There is deliberately **no automatic global/process-wide limiter** and
no auto-sharing based on, e.g., API key or provider name. That would
violate the "no global state, no singletons" rule already applied
throughout the framework (`DEVELOPMENT.md`) and would make sharing an
implicit side effect of unrelated configuration rather than something a
developer can see at the call site. Instead: construct one `RateLimiter`
and pass the *same instance* to every `Agent`/`AI` that draws on the
same underlying quota:

```python
shared = RateLimiter(requests_per_minute=15)
research = Agent(name="Researcher", provider="gemini", rate_limiter=shared)
writer = Agent(name="Writer", provider="gemini", rate_limiter=shared)
```

### `Settings.rate_limit_rpm` covers the single-instance case for free

`Settings` gains two optional fields, both `None`/off by default,
matching the "never automatic, always opt-in" convention already
established for structured logging (ADR-0003): `rate_limit_rpm` and
`rate_limit_max_wait_seconds`. If `AI` is constructed without an
explicit `rate_limiter=` and `settings.rate_limit_rpm` is set, `AI`
builds its own private `RateLimiter` from that value. This means a
single `Agent`/`AI` use case is solved by setting one env var
(`RATE_LIMIT_RPM=15`) with zero code changes -- but it does **not**
solve the multi-agent sharing case (each instance still builds its own
*private* limiter from `Settings`), which is why the explicit
`rate_limiter=` injection path exists as the primary mechanism, not an
afterthought.

### Exceeding `max_wait_seconds` raises, doesn't wait forever by default choice

`RateLimiter(max_wait_seconds=...)` is optional and `None` by default,
which waits as long as necessary -- matching the literal ask ("so the
app waits"). When set, exceeding it raises a new
`RateLimitException(AIException)` (`requisite/core/exceptions.py`),
consistent with the existing exception hierarchy's "never swallow,
always raise with context" convention, as a safety valve for callers
who'd rather fail fast than block indefinitely.

## Alternatives considered

- **Token bucket algorithm.** Rejected -- see "Sliding-window log" above;
  a sliding-window log matches the actual quota semantics being hit
  without needing a separate refill-rate parameter to tune.
- **Rate limiting inside each provider file** (`OpenAIProvider`,
  `GeminiProvider`, etc.) instead of the `AI` facade. Rejected: call
  frequency is not a wire-protocol concern, and instrumenting 5 provider
  files (plus every future provider) for a cross-cutting concern that
  `AI` already funnels through 4 methods is unnecessary duplication.
- **Implicit sharing keyed by provider name or API key** (e.g. a
  process-wide registry of limiters, auto-selected by `Settings`).
  Rejected: violates the framework's no-global-state/no-singleton rule
  and hides sharing behind configuration rather than an explicit,
  readable constructor argument.
- **A `BaseRateLimiter` interface + `RateLimiterRegistry`,** mirroring
  `BaseProvider`/`ProviderRegistry`. Rejected for v1 -- no second
  implementation exists to justify the abstraction; revisit if a real
  need for a different algorithm (e.g. a distributed/Redis-backed
  limiter for multi-process deployments) shows up.

## Consequences

### Positive

- The exact bug reported (`workflow_example.py`'s four agents
  collectively exceeding a shared 15/min quota) is fixed by constructing
  one `RateLimiter` and passing it to all four agents -- not just
  "possible in theory" but the example itself was updated to do this.
- Setting `RATE_LIMIT_RPM` in `.env` is enough for the common
  single-agent case, with no other code changes.
- No changes were needed to any of the 5 provider files, `BaseProvider`,
  or any orchestrator -- confirms rate limiting is correctly scoped as
  an `AI`-facade-level concern, not a provider-level or orchestrator-level
  one.

### Negative / risks

- Sharing is opt-in and explicit, which means it's also easy to forget:
  building several `Agent`s against the same free-tier key *without*
  passing a shared `rate_limiter=` still reproduces the original bug.
  This is a deliberate trade-off (explicit DI over implicit magic) but
  is worth flagging prominently in `README.md`/`.env.example`, not just
  the API docstrings.
- `RateLimiter` only smooths *this process's* call rate. Multiple
  separate processes (e.g. two terminals running the same script, or a
  multi-worker server deployment) sharing one API key each get their own
  independent `RateLimiter` and can still collectively exceed the real
  quota. A distributed limiter is out of scope here.
- This is proactive-only: it prevents exceeding the configured rate, but
  does not catch or retry an actual 429 that occurs for other reasons
  (e.g. the configured `requests_per_minute` doesn't match the real
  quota). See Follow-ups.

### Follow-ups

- Reactive retry/backoff on an actual 429 response (using the
  provider's own reported `retryDelay` where available) is a natural,
  separate follow-up if proactive limiting alone proves insufficient --
  not built here since it wasn't what was asked for and is a distinct
  concern (recovering from a failure vs. preventing one).
- If a real need for a distributed/shared-across-processes limiter shows
  up, revisit the "no `BaseRateLimiter` interface" decision above.
