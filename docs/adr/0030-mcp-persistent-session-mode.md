# 0030. MCP client persistent-session mode

Status: Accepted
Date: 2026-08-23

## Context

ADR-0004 deliberately shipped `MCPClient` with per-call reconnect --
every `discover_tools()` call and every tool call opens a fresh
connection, performs one request, and disconnects -- instead of a
persistent session, and named an explicit trigger to revisit: *"if
reconnect latency becomes a measured problem for a real application
(frequent tool calls to the same server in a tight loop), add an opt-in
persistent-session mode."* As of 0.28.0 this was the last remaining
open line in `ROADMAP.md`'s MCP section (the only other 📋 line
anywhere in the file is an official plugin listing/directory in the
docs -- a documentation/community effort, not a coding task, and out of
scope here).

That precondition has now been measured live, not assumed. 20 sequential
tool calls against a real local Requisite-hosted MCP server
(`examples/mcp_server_example.py`'s server, run standalone), current
per-call-reconnect behavior vs. a session held open across all 20 calls:

| Transport | Current (reconnect/call) | Persistent | Slowdown |
|---|---|---|---|
| stdio | 1381.6ms mean | 1.4ms mean | ~1000x |
| HTTP (local loopback, no TLS) | 37.8ms mean | 2.6ms mean | ~15x |

stdio's cost is dominated by spawning a fresh Python subprocess +
interpreter startup on every single call; a real remote HTTPS server
would widen the HTTP gap further still (TLS handshake + real network
RTT on top of the loopback numbers above). This clears ADR-0004's bar
decisively.

ADR-0004's own Context section named the two costly implementation
options for a persistent session: an async-only `MCPClient`, or a
background-thread bridge (`asyncio.run_coroutine_threadsafe`). Before
picking between them, the actual risk in the "async-only" option was
verified live rather than assumed:

- **Pattern A (broken):** connect a session inside one `asyncio.run()`
  call, store the raw `ClientSession`, reuse it inside a second,
  separate `asyncio.run()` call. This does not raise a clean error -- it
  **deadlocks** at event-loop shutdown (`asyncio.run()`'s internal
  task-cancellation cleanup hangs on the orphaned anyio-managed
  subprocess-communication tasks from `stdio_client`). Verified live:
  both the Python process and the orphaned MCP-server subprocess had to
  be force-killed.
- **Pattern B (works):** connect, make several calls, explicitly close
  -- all within one continuous event loop / one `asyncio.run()` call.
  Verified live: 10 calls at ~1.9ms mean, clean shutdown, no hang.

Conclusion: persistent-session mode is safe only when connect+use+close
happen within one continuous async context. Crossing an `asyncio.run()`
boundary with an open session is a real, silent-deadlock risk, not a
documentation footnote -- the design below exists specifically to
convert that into an immediate, clean exception rather than merely
warn about it.

A third probe confirmed the specific mechanism this ADR's implementation
uses: wrapping the *existing* `_session()` context manager (a single
opaque async-generator-based unit) in an `AsyncExitStack`, rather than
decomposing into the raw SDK-level `stdio_client`/`ClientSession`
managers separately via two `enter_async_context` calls (which is what
produced Pattern A's deadlock), works cleanly -- connect once, make
several real calls, close, all within one loop, no hang.

## Decision

### Explicit `aconnect()`/`aclose()` + `async with client:`, not a `persistent=True` flag

Deviates from ADR-0004's own original sketch
(`MCPClient.stdio(..., persistent=True)`). A boolean flag doesn't
structurally prevent the verified cross-loop deadlock -- it only changes
*when* the first connect happens; a caller can still trivially cross an
`asyncio.run()` boundary by accident (call one async method inside one
`asyncio.run()`, another inside a second). Explicit `aconnect()`/
`aclose()` plus `async with client:` makes "one continuous event loop"
the only ergonomic way to use persistent mode -- `async with` is
definitionally one continuous scope inside whatever loop is currently
running, mirroring established Python async idiom for exactly this kind
of resource (`httpx.AsyncClient`, `aiohttp.ClientSession`). It also
requires zero constructor changes (`MCPClient.__init__`, `.stdio(...)`,
`.http(...)` are unchanged) and makes persistent mode's async-only
nature visible in the API shape itself rather than hidden behind a flag
on an otherwise sync-first-looking class.

New public surface on `MCPClient` (`requisite/mcp/client.py`):

```python
async def aconnect(self) -> None: ...
async def aclose(self) -> None: ...
async def __aenter__(self) -> "MCPClient": ...
async def __aexit__(self, exc_type, exc, tb) -> None: ...
```

New instance state (`__init__`): `self._persistent_session`,
`self._persistent_loop`, `self._exit_stack`.

### `_session()` reuse with a loop-identity guard

The original `_session()` body (the full stdio/http
connect-yield-disconnect cycle) is renamed, unchanged, to a new private
`_connect()`. `_session()` becomes a thin wrapper: if a persistent
session is open, it's yielded directly (after a loop-identity check);
otherwise execution falls through to `_connect()` exactly as before.
Because all 6 existing async methods (`adiscover_tools`, `_call_tool`,
`adiscover_resources`, `aread_resource`, `adiscover_prompts`,
`aget_prompt`) already go through `async with self._session() as
session:` and touch nothing else directly, **none of them needed any
code change** -- this was the natural extension point the existing code
already had.

The loop-identity check --
`asyncio.get_running_loop() is not self._persistent_loop` -- catches a
cross-loop reuse attempt deterministically and cheaply, *before* any SDK
method is ever invoked, converting the verified Pattern-A deadlock into
an immediate `ConfigurationException`. `aclose()` carries the identical
check, since closing also touches the loop-bound transport cleanup; if
tripped, the exception explicitly notes the session may now be stranded
on its original (dead) loop, since a guard can prevent misuse reaching
the SDK but cannot retroactively un-strand a session already left open
on an exited loop.

`aconnect()` uses an `AsyncExitStack` wrapping the single `_connect()`
context manager (`stack.enter_async_context(self._connect())`) -- the
verified-safe mechanism from Context, not the raw two-level SDK
managers.

### Sync methods and `Tool.execute()` reject persistent mode proactively

All 5 existing sync one-liners (`discover_tools`, `discover_resources`,
`read_resource`, `discover_prompts`, `get_prompt`) gained a
`self._reject_sync_when_connected(method_name)` pre-check before their
existing `asyncio.run(...)` call, so misuse never even spins up a second
event loop -- it fails fast with a `ConfigurationException` naming the
correct `a`-prefixed method to use instead.

`requisite/tools/base.py` (`Tool.execute()`) was deliberately left
**unmodified** -- it's general-purpose, not MCP-specific. When an
MCP-backed tool's async wrapper (built in `MCPClient._to_tool`) is
invoked via `Tool.execute()` while persistent mode is connected,
`asyncio.run(self.func(**kwargs))` spins a fresh loop, hits the
`_session()` guard above, and raises `ConfigurationException`, which
`Tool.execute()`'s existing exception handling wraps as `ToolException`
(with the `ConfigurationException` preserved as `__cause__`) -- correct,
pre-existing behavior applied to a new case, verified by a dedicated
test rather than left as an implicit side effect.

### No new `BaseMCPClient` abstract methods; no `MCPClientRegistry` changes

`aconnect`/`aclose`/`__aenter__`/`__aexit__` are concrete methods on
`MCPClient` only, not promoted to the `BaseMCPClient` ABC. Unlike
ADR-0026's resource/prompt discovery (genuine capability surface any
correct MCP client implementation should expose), persistent-session
lifecycle is implementation-specific plumbing tied to how `MCPClient`
manages transport-level async context managers (anyio task groups,
subprocess handles) -- there is exactly one concrete implementation
today, and forcing this shape onto the ABC now would obligate a
hypothetical future implementation (e.g. a stateless client with
near-zero per-call cost) to support connection lifecycle it may not
need.

`requisite/mcp/registry.py` needed no changes -- it has no
lifecycle/cleanup logic today (pure dict bookkeeping); each client
manages its own `aconnect`/`aclose` independently of how it's
registered.

## Alternatives considered

- **Constructor `persistent=True` flag** (ADR-0004's own original
  sketch). Rejected -- doesn't structurally prevent the verified
  cross-loop deadlock risk; still needs an explicit connect/close
  boundary to be safe, which a boolean doesn't provide on its own.
- **Background-thread bridge** (`asyncio.run_coroutine_threadsafe`).
  Rejected, same reasoning ADR-0004 already gave (thread lifecycle,
  shutdown ordering, error propagation across the thread boundary) --
  unnecessary now that an opt-in async-only persistent mode is an
  acceptable trade-off alongside the sync-first default.
- **Silently falling back to reconnect** instead of raising on
  cross-loop reuse. Rejected -- would hide the misuse behind extra
  latency instead of surfacing it, and doesn't solve the case where
  `aclose()` itself is called from the wrong loop (the session is
  already stranded on the dead loop by then).
- **Adding `aconnect`/`aclose` to `BaseMCPClient` now.** Rejected --
  single implementation today, YAGNI; a Follow-up if a second
  implementation needs the same semantics.
- **`MCPClientRegistry.aclose_all()`.** Deferred, not implemented -- no
  concrete multi-client cleanup use case yet.

## Consequences

### Positive

- ~1000x (stdio) / ~15x (HTTP) measured latency reduction for callers
  making repeated tool calls to the same server in a tight loop -- the
  exact scenario ADR-0004's trigger named.
- The verified silent-deadlock risk is closed with a clean, immediate
  `ConfigurationException` instead of a hang, on both the call path
  (`_session()`) and the close path (`aclose()`).
- Zero changes to any of the 6 existing async methods, and zero changes
  to `Tool.execute()`/`tools/base.py` -- fully additive.
- Closes the last open line in `ROADMAP.md`'s MCP section (the entire
  MCP section is now fully shipped).

### Negative / risks

- Persistent mode is async-only by design: `discover_tools()`/
  `read_resource()`/`get_prompt()` and a discovered tool's sync
  `execute()` become unusable (raise immediately) on a client while it's
  connected -- a caller mixing sync and async usage on one `MCPClient`
  instance must be aware of this.
- The cross-loop guard converts a hang into an exception, but can't
  retroactively close a session already stranded on a dead loop -- if a
  caller manually pairs `aconnect()`/`aclose()` across two separate
  `asyncio.run()` calls instead of using `async with client:`, the
  subprocess/connection from the first loop may still leak. `async with`
  is the only fully-safe usage pattern; documented as such in the class
  docstring and README.
- New public surface (`aconnect`/`aclose`/`__aenter__`/`__aexit__`)
  only matters to advanced/tight-loop callers -- most users should keep
  using the sync-first default, which is entirely unchanged.

### Follow-ups

- Promote `aconnect`/`aclose`/`__aenter__`/`__aexit__` to
  `BaseMCPClient` abstract methods if a second concrete implementation
  needs the same semantics.
- `MCPClientRegistry.aclose_all()` if a concrete multi-client cleanup
  use case emerges.
