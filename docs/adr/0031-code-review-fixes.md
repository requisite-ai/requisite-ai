# 0031. Code review, adversarial testing, and fixes across the whole codebase

Status: Accepted
Date: 2026-08-23

## Context

At 0.29.0 the codebase (85 files, ~14.7k lines) had never had a full,
deliberate adversarial pass across every subsystem -- each feature had
been individually reviewed and real-verified at ship time (per the
established pattern across ADR-0001 through ADR-0030), but no single
pass had gone back across the whole thing hunting specifically for
correctness bugs, races, and edge/negative cases the shipping-time
verification for each individual feature wouldn't have been looking for.

The review combined two methods deliberately, not just one:

1. **Static review**, split into four parallel passes by subsystem (core
   execution path: agents/tools/providers/core; orchestration:
   orchestrators/workflows; memory+RAG; MCP+capabilities+CLI+config+
   telemetry+skills+prompts), each instructed to report only
   high-confidence, concrete findings with a reproduction scenario -- not
   style nits.
2. **Live adversarial testing** against the actual installed package --
   real scripts constructing real `Agent`/`Workflow`/`MCPClient`
   instances with scripted fake providers (no live model calls needed for
   most of this) and deliberately malformed/adversarial/boundary inputs,
   run to see what actually happens, not just what the code implies
   should happen. This caught things static review alone would have
   guessed at or missed -- e.g. the self-referential `Workflow` recursion
   bug was only fully understood by watching a real stack trace via
   `faulthandler.dump_traceback_later`, and the `VectorMemory`
   append-vs-clear race required a real multi-threaded repro with a
   controllable-delay fake embedding provider to reproduce
   deterministically.

One static-review finding was flagged as CRITICAL (`MCPClient.http(...)`
supposedly broken by an `import httpx2` "typo") and turned out to be a
**false positive**: `httpx2` is a real, correctly-required package (a
separate next-gen HTTP client by httpx's own author, a genuine
transitive dependency of the `mcp` SDK) -- confirmed via `pip show
httpx2` and cross-checked against a live HTTP round-trip benchmark
already run earlier the same session. Recorded here as a reminder that
even a structured, multi-angle review process can produce a
high-confidence wrong finding, and that live verification of a finding
before acting on it is not optional. No other finding in this ADR had a
similar false-positive risk -- each of the 18 below was independently
reproduced (either by the reviewing agent, or by direct live testing) or
is source-verified against a stable Python/library behavior (e.g. the
`asyncio.gather` semantics finding).

## Decision

Fixed all 18 confirmed findings, each with a regression test. Grouped
here by theme rather than by severity, since several share one root
cause or one fix mechanism.

### 1. Workflow delegation cycles escape `max_rounds`/`max_steps` entirely (CRITICAL)

A `Workflow` that delegates to itself -- directly, or via a cycle of
other `Workflow`s -- under `hierarchical` or `graph`, on both `native`
and `langgraph` backends, is not bounded by `max_rounds`/`max_steps` at
all: each nested `delegate.run(subtask, **kwargs)` / `node.run(...)`
call is a completely fresh `Workflow.run()` invocation with its own
independent budget, not decremented/threaded through the recursive
chain. It recurses through the real Python call stack and crashes with
an uncatchable `RecursionError` instead of a clean, documented error.
Confirmed live: 122 real coordinator decisions before the crash in a
direct self-reference case, 162 in a `graph`-node self-reference case,
61-per-side in a mutual A&harr;B cycle, all with `max_rounds`/`max_steps`
set to 1,000,000 -- never remotely approached.

**Fix:** `Workflow.run()`/`.arun()` (`requisite/workflows/workflow.py`)
thread a private `_delegation_chain` kwarg -- a tuple of
`(id(workflow), workflow.name)` pairs already "in progress" up the call
chain -- through `**kwargs` on every nested Workflow-to-Workflow
delegation. Before proceeding, each call checks whether `id(self)` is
already in the chain and raises a clean `ConfigurationException` naming
the exact cycle (e.g. `"Team -> Team"` or `"A -> B -> A"`) if so.
`Agent.run()`/`.arun()` (`requisite/agents/agent.py`) pop-and-discard the
same kwarg defensively at entry, so it never leaks into a provider SDK
call when a delegate happens to be an `Agent` rather than a `Workflow`
in a given round. No orchestrator backend needed any change -- every
backend already forwards `**kwargs` unmodified to delegate/node `.run()`
calls, so the chain flows through transparently regardless of which
backend built the graph. Verified fixed on native (`hierarchical`,
`graph`) and langgraph (`hierarchical`) backends.

### 2. A single hallucinated tool call aborts the entire agent run (HIGH)

Neither `ToolRegistry.get(call.name)` (raises for any tool name the
model didn't get exactly right -- the single most common LLM
tool-calling failure mode) nor a tool's own execution failure was caught
inside `Agent.run()`/`.arun()`'s tool-calling loop -- either exception
unwound straight out of the loop on iteration 1, regardless of
`max_iterations`, defeating its documented purpose as a retry budget.

**Fix:** both the sync loop and the async per-call helper
(`_traced_aexecute`) in `requisite/agents/agent.py` now catch
`ToolException` (the type both an unknown-tool lookup and a tool's own
execution failure already raise) and feed the error back to the model as
a normal `tool_result` message (`f"Error: {exc}"`) instead of raising,
so the model can see the failure and retry on a later iteration within
`max_iterations`. A model that never self-corrects still bounds cleanly
at `max_iterations` (verified: exhausts the full budget, then raises the
documented `AgentException`, rather than crashing on the first bad
call).

### 3. Concurrent `asyncio.gather` calls leak orphaned background work on partial failure (HIGH)

Six call sites -- `Agent.arun()`'s concurrent tool execution, and five
native-orchestrator strategies (`parallel`, `consensus`, one round of
`debate`, `map_reduce`, `tree_of_thoughts`) -- used plain
`asyncio.gather(...)` (`return_exceptions=False`, the default). When one
gathered coroutine fails, `gather` raises immediately and returns
control to the caller *without cancelling or awaiting* the others,
leaving real concurrently-running `Agent`/tool calls executing
unattended in the background with no handle and no way to observe their
eventual failure. Confirmed live (consensus, 3 participants, one fails
instantly, one sleeps 3s): the caller's `except` fired after 10ms, but
the slow participant kept running for the full 3 seconds afterward.

**Fix:** `Agent.arun()`'s tool-execution gather no longer needs
`return_exceptions=True` at all, since fix #2 already means
`_traced_aexecute` never raises `ToolException` (the dominant failure
mode) -- it's caught and converted to a result inline. For the five
native-orchestrator strategies, a new shared helper,
`_gather_waiting_for_all` (`requisite/orchestrators/native.py`), wraps
`asyncio.gather(*coroutines, return_exceptions=True)` and re-raises the
first exception (in submission order) only *after* every coroutine has
actually completed -- preserving the existing "any failure fails the
whole call" behavior while closing the background-leak. Verified live:
the same 3-participant repro now waits the full 3.05s (for the slow
sibling to finish) before raising, with the sibling's completion flag
already `True` at that point.

### 4. `@tool` crashes with a raw `NameError` on an unresolvable type hint (HIGH)

`function_to_parameters_schema` (`requisite/tools/schema.py`) called
`typing.get_type_hints(func)` for the whole function's annotations at
once, which raises if even one fails to resolve (a `TYPE_CHECKING`-only
import, a typo, any unresolvable forward reference) -- crashing tool
registration entirely and directly contradicting the module's own
documented "unrecognized annotations fall back to a permissive
`'string'` schema rather than raising" guarantee, since that per-
parameter fallback (`_json_type_for`) never gets a chance to run when
the bulk resolution call itself fails first. Confirmed live and
independently by the core-execution-path review.

**Fix:** wraps `typing.get_type_hints(func)` in a `try/except`, falling
back to the function's raw (possibly still-unresolved-string)
`__annotations__` dict on any failure -- each parameter is then handled
independently as before, and an annotation `_json_type_for` doesn't
recognize (including a bare unresolved string) already degrades to the
documented permissive `"string"` schema. Verified: the exact reproducing
function from the original finding now registers cleanly.

### 5. Provider response-parsing gaps (HIGH + MEDIUM)

- **Empty `choices` list crashes with a raw `IndexError`**, not
  `ProviderException`, in `OpenAIProvider._to_chat_response` -- and,
  since `GroqProvider`/`AzureOpenAIProvider`/`OpenRouterProvider`/
  `TogetherProvider` all subclass `OpenAIProvider` and reuse this
  method unchanged, this affects 5 of the 8 shipped providers. A real,
  documented possibility (a content-filtered/safety-blocked completion,
  or a broken/adversarial OpenAI-compatible proxy), not a hypothetical.
  **Fix:** an explicit `if not completion.choices: raise
  ProviderException(...)` check at the top of `_to_chat_response`.
- **Malformed tool-call-argument JSON silently falls back to `{}`** with
  no trace, in three places (OpenAI non-streaming and streaming,
  Anthropic streaming). If the tool's parameters are all optional, this
  means the tool silently runs with default values instead of whatever
  the model actually intended, and nothing downstream ever learns the
  arguments were corrupted. **Fix:** each `except json.JSONDecodeError`
  now logs a `logger.warning(...)` naming the tool and the raw malformed
  string before falling back -- diagnosable, not silent. `{}` is kept as
  the fallback deliberately (a single malformed call shouldn't fail an
  otherwise-usable response), and fix #2 means a tool with *required*
  parameters already surfaces this as a recoverable tool-result error to
  the model regardless.

### 6. Cross-backend reserved-node-name collisions (HIGH)

A worker/delegate literally named `"__coordinator__"` (langgraph's own
internal coordinator node name) or `"__supervisor_finish__"` (autogen's
internal no-op finishing participant) reaches the third-party library's
own `add_node`/`SelectorGroupChat` construction and raises a raw,
backend-specific `ValueError` -- while the identical `Workflow` succeeds
unchanged on `native`, breaking the cross-backend behavioral parity
`supervisor`/`hierarchical` otherwise guarantee (per ADR-0016).

**Fix:** `_reject_reserved_node_names` (new helper in
`requisite/orchestrators/langgraph_orchestrator.py`) checks delegate/
worker names against langgraph's reserved set right after
`split_fn(...)`, before any graph-building call; the autogen orchestrator
gets an equivalent inline check against `"__supervisor_finish__"`. Both
raise `ConfigurationException` naming the collision. Verified live on
both backends; confirmed the identical `Workflow` still succeeds on
`native` unaffected.

### 7. A `graph` node named `"__end__"` is silently unreachable (HIGH)

`_resolve_next_graph_node`'s routing check (`to == END`) always
short-circuits before checking whether a real node has that name, so a
node literally named `END` (`"__end__"`) can never actually run -- any
edge routed to it just terminates the graph early, **with no error at
all**. Worse than a crash: silently wrong behavior.

**Fix:** `NativeOrchestrator._index_graph_nodes` (shared verbatim by the
langgraph backend's `_build_arbitrary_graph`) now rejects a node named
`END` at build time with a clear `ConfigurationException`, before any
edges are even validated. Fixes both backends via the one shared helper.

### 8. `top_k=0` silently replaced by the instance default (MEDIUM-HIGH)

`Retriever`, `BM25Retriever`, `HybridRetriever` (3 call sites), and
`LLMReranker` (2 call sites) all resolved an optional `top_k` override
via Python's `or` operator (`top_k or self.top_k` /
`top_k or len(results)`), which treats an explicitly-passed `0`
identically to "not given" -- an explicit "give me nothing" request
silently returns the default count (or, for the reranker, *every*
candidate) instead. This made the vector-store-level `top_k<=0 -> []`
guard (ADR-0022) unreachable through the retriever layer, since `0`
never survived to reach it. Separately, `PineconeVectorStore.search` had
no `top_k<=0` guard at all, unlike `InMemoryVectorStore`/
`WeaviateVectorStore`, contradicting ADR-0022's own claim that all three
stores short-circuit uniformly.

**Fix:** every `... or ...` site changed to `... if ... is not None else
...`. `BM25Index.search` and `PineconeVectorStore.search` both gained an
explicit `if top_k <= 0: return []` guard, matching the other two
stores' existing pattern (and additionally covering negative values,
which Python's `list[:top_k]` slicing alone doesn't handle uniformly).
The Pinecone regression test specifically asserts the underlying
`index.query(...)` is never even called for a non-positive `top_k`, not
just that the fake index happens to slice to empty.

### 9. `VectorMemory` reopens the append-vs-clear race ADR-0022 claimed to close (HIGH)

`append()`/`aappend()` release their lock after allocating a message's
`turn_index` but *before* the slow embedding call and the eventual
write -- exactly matching ADR-0022's own stated design (don't hold a
lock across a slow network call). But this means a concurrent `clear()`
can run to completion (acquire the lock, see the counter already
incremented by the in-flight append, delete/reset based on that count)
*while* the append's embed call is still in flight; when that embed
finally returns, its write lands *after* the clear, resurrecting content
into a session the caller believed was wiped, plus an orphaned
vector-store chunk no future `clear()` will ever revisit (the counter
was already reset to 0). Reproduced deterministically with a
controllable-delay fake embedding provider.

**Fix:** a new per-session `_generations` counter, bumped by
`clear()`/`aclear()`. `append()`/`aappend()` capture the current
generation under the *first* lock (alongside the `turn_index`
allocation), do the slow embed unlocked as before, then re-acquire the
lock to check whether the generation changed before committing the
write -- if a `clear()` ran in the interim, the append's effect
(vector-store write and history commit) is discarded entirely rather
than resurrecting the cleared session. This narrows the lock-free window
to just the embed call (the original, still-honored performance intent)
while closing the race. Verified with the exact repro from the review.

### 10. `PromptTemplate` validation bypass and injection risk (MEDIUM-HIGH)

- `_extract_variables` only recognized a field as an input variable if
  the *entire* field name was a bare identifier, silently excluding
  dotted/indexed fields (`{cfg.api_key}`, `{items[0]}`) from
  `input_variables` -- so `.format()` never knew it needed `cfg`/`items`
  supplied, and a missing one surfaced as a raw `KeyError`/
  `AttributeError` instead of `PromptException`.
- `.partial()` substituted values via `format_map` without escaping
  literal `{`/`}` characters in those values, so a value containing
  `{secret}` could inject a *new* placeholder into the returned
  template's text, which a later `.format(secret=...)` call would then
  unexpectedly fill -- `template.partial(name="{secret}")` followed by
  `.format(secret="TOP-SECRET-VALUE")` interpolated the secret into both
  `name`'s position and its own, not just its own.

**Fix:** `_extract_variables` now captures the *root* identifier of a
dotted/indexed field (everything before the first `.`/`[`) instead of
requiring the whole field name to be bare -- this is the actual kwarg
name `str.format` needs supplied, so validation now correctly requires
it. `.partial()` escapes literal braces (`{` -> `{{`, `}` -> `}}`) in
*string* values before substitution (non-string values pass through
unescaped, preserving numeric format-spec compatibility like
`{amount:.2f}`), so a value can no longer introduce a new placeholder --
verified both original PoCs are closed, and that format specs on
non-string `partial()` values still work.

### 11. `InProcessMemory.load()` leaks storage for probed-but-never-appended sessions (MEDIUM)

`self._sessions[session_id]` on a `defaultdict(list)` auto-vivifies an
empty-list entry for any session id merely *read*, not just written --
an application pattern that speculatively probes `load(session_id)` with
unique/ephemeral ids leaks one dict entry per call for the process's
lifetime.

**Fix:** `load()` uses `.get(session_id, [])` instead of indexing.
`append()` is unchanged (it legitimately needs the auto-vivifying
behavior on first real write).

### 12. stdio MCP transport has no timeout anywhere (MEDIUM)

Unlike the HTTP transport (bounded by `httpx.AsyncClient(timeout=...)`),
the stdio transport's `ClientSession` was constructed with no
`read_timeout_seconds`, and `MCPClient.stdio(...)` had no `timeout=`
parameter to set one -- a hung or misbehaving subprocess (wrong command,
a server that never completes the `initialize` handshake) blocks any
call forever, with nothing in the public API able to bound it. Verified
against the real `mcp` SDK source that `read_timeout_seconds` is the
per-request default applied to every `send_request(...)` call
(including `initialize`), confirming it's the correct mechanism, not
just plausibly named.

**Fix:** `MCPClient.stdio(...)` gained a `timeout=` parameter (default
30s, matching `.http()`'s existing default -- the shared constant was
renamed `_DEFAULT_HTTP_TIMEOUT` -> `_DEFAULT_TIMEOUT` accordingly).
Both transports' `ClientSession(...)` construction in `_connect()` now
pass `read_timeout_seconds=self._timeout`. Verified live against a real
hung subprocess (a script that just sleeps): the call now raises a
clean, bounded `MCPException` instead of hanging indefinitely.

### 13. Documentation gaps (MEDIUM/LOW)

- `Workflow`'s class-level docstring claimed `reflection`/`hierarchical`/
  `graph` were native-only, contradicted by the langgraph implementation
  and its own passing tests (stale since those strategies shipped on
  langgraph in ADR-0028/ADR-0029). **Fixed**: rewritten to list the
  actual current per-strategy backend support.
- `Settings` had no *documented* way to opt out of `.env` filesystem
  discovery for a caller constructing it with fully explicit kwargs and
  no intent to touch the filesystem at all (a real, standard
  `pydantic-settings` escape hatch, `Settings(_env_file=None, ...)`,
  existed but wasn't mentioned anywhere in `requisite`'s own docs).
  **Fixed**: documented in the class docstring.
- The MCP persistent-session mode's docstring didn't say what happens if
  the underlying connection dies mid-use outside of `aclose()` (no
  auto-detection/reconnection by design -- calls keep failing with the
  same `MCPException` until the caller notices) or that `aclose()`
  always clears internal state *before* attempting teardown, so it's a
  reliable recovery path even when that teardown itself raises (verified
  live: a connection whose teardown always raises still leaves the
  client cleanly reconnectable afterward). **Fixed**: documented in
  `MCPClient`'s class docstring.

### 14. CLI: `requisite chat --agent X --provider Y` silently ignored `--provider`/`--model` (LOW)

`cmd_chat` branched purely on `if args.agent:` and never read
`args.provider`/`args.model` in that branch -- a user passing both gets
no error and no indication the flags did nothing.

**Fix:** `cmd_chat` now raises `ConfigurationException` (caught by the
CLI's existing top-level `AIException` handler, exit code 1) when
`--agent` is combined with `--provider`/`--model`, naming the conflict
explicitly, rather than silently discarding either flag.

## Alternatives considered

- **Depth-limit counter instead of cycle detection** for finding #1 (a
  simple `_delegation_depth` incremented per call, capped at some N).
  Rejected -- a raw depth limit either fires too late (close to Python's
  own real stack limit, risking a crash before the framework's own check
  fires) or too early (rejecting a legitimately deep, non-cyclic
  delegation chain); cycle detection via chain membership catches the
  actual problem (a repeat) precisely and immediately, regardless of
  depth.
- **`return_exceptions=True` alone**, with the caller responsible for
  checking each result for an exception, for finding #3. Rejected --
  would silently change every existing call site's contract (results
  list could now contain exception objects instead of the promised
  result type) for callers outside this codebase too; `_gather_waiting_for_all`
  preserves the exact prior contract (raises on any failure) while
  fixing only the actual defect (the background leak).
- **Hard-raising instead of logging** on malformed tool-call JSON
  (finding #5). Rejected -- would fail an entire multi-tool-call response
  over one provider-side parsing hiccup in one call; logging makes the
  failure diagnosable without that blast radius, and fix #2 already
  gives required-parameter tools a recoverable path.
- **Auto-reconnecting `MCPClient` on a dead persistent session**
  (finding #13's MCP item). Rejected -- contradicts this project's
  established "explicit over implicit" connection-lifecycle philosophy
  (ADR-0004, ADR-0030); reliably *distinguishing* a dead-connection
  failure from an ordinary protocol-level error across the SDK's
  possible exception shapes wasn't something that could be verified with
  the same confidence as the rest of this pass, so documenting the
  already-correct manual-recovery guarantee was preferred over a guess.

## Consequences

### Positive

- Closes a real, uncontrolled-crash-class bug (#1) and the single most
  common recoverable LLM failure mode (#2) that could hit any production
  agent.
- 40 new regression tests added across `tests/test_workflows.py`,
  `tests/test_agents.py`, `tests/test_providers.py`,
  `tests/test_autogen_orchestrator.py`, `tests/test_tools.py`,
  `tests/test_rag.py`, `tests/test_memory.py`, `tests/test_prompts.py`,
  `tests/test_mcp.py`, and `tests/test_cli.py` -- 509 total, up from 469.
  Every fix was verified live against the actual reproducing scenario
  before being considered done, not just reasoned about.
- Zero behavior changes to any already-correct code path -- every fix is
  additive/corrective at the exact site of the defect.

### Negative / risks

- `Agent.run()`/`.arun()`'s tool loop now silently retries a failing
  tool call up to `max_iterations` times instead of failing fast on the
  first error -- intentional (see finding #2), but an application that
  was relying on the old fail-fast behavior (e.g. to short-circuit an
  expensive multi-tool turn on the first sign of trouble) now burns more
  of its iteration budget before giving up.
- The `_delegation_chain` mechanism (finding #1) is a new, internal-only
  reserved kwarg name (`"_delegation_chain"`) -- if a real tool or
  provider call ever legitimately needs a kwarg with that exact name,
  it would collide. Judged acceptable given the leading-underscore,
  framework-internal naming convention already used elsewhere in this
  codebase for exactly this purpose.

### Follow-ups

- None of the 18 fixes left an open design question -- each is either
  fully closed or explicitly deferred with reasoning in Alternatives
  considered above.
