
# 0004. MCP client integration

Status: Accepted
Date: 2026-07-17

## Context

ADR-0001 specified `BaseMCPClient`'s shape but deferred implementation.
Before implementing, three decisions were made explicitly (with the
project owner) rather than assumed:

1. Which MCP transport(s) to support in v1.
2. Whether tool calls should hold a persistent session or reconnect per call.
3. Confirming the capability-bridge design from ADR-0001 still holds once
   there's a real implementation to test it against.

Decision 1 was made collaboratively: **both stdio and Streamable HTTP,
from day one** (not phased). This ADR records that decision plus the
implementation choices it led to, all verified against the real `mcp`
1.28.1 SDK and real MCP servers (both transports) before being written
down here -- not assumed from documentation alone.

## Decision

### Both transports, verified against the current spec

`MCPClient.stdio(...)` and `MCPClient.http(...)` are two factory methods
on one class, not two classes. Confirmed live against the current MCP
spec and the `mcp` SDK: **stdio** (local subprocess) and **Streamable
HTTP** (remote; replaced the deprecated SSE transport in the November
2025 spec revision) are the two transports that matter today. The plain
constructor (`MCPClient(transport=..., ...)`) exists but the factories
are the documented entry point -- each only exposes the parameters
relevant to its transport (`command`/`args`/`env`/`cwd` for stdio;
`url`/`headers`/`timeout` for HTTP), rather than one constructor with a
pile of optional parameters where half are always irrelevant.

Both transports were smoke-tested against real MCP servers before any
test was written: a `FastMCP`-based stdio server (tool discovery + two
tool calls) and the same server run over Streamable HTTP on a local port
(tool discovery + one tool call), both round-tripping correctly,
including automatic JSON-Schema generation from the server's own tool
signatures and `structuredContent` in responses. The unit test suite
(`tests/test_mcp.py`) fakes the session object rather than depending on
these real servers, per `DEVELOPMENT.md`'s no-network-in-tests rule --
the real-server check was a one-time implementation verification, not
something CI re-runs.

### Per-call connections, not a persistent session

**Decision: every `discover_tools()` call and every tool `execute()` call
opens a fresh connection, performs one request, and disconnects.** There
is no session held open between calls.

This was not the only option. A persistent-session design (connect once,
reuse the session for many calls) would avoid reconnect latency but
requires either an async-only `MCPClient` (pushing the sync/async bridge
problem onto every caller) or a background event loop thread with a
`asyncio.run_coroutine_threadsafe` bridge (real complexity: thread
lifecycle, shutdown ordering, error propagation across the thread
boundary). Per-call connections were chosen instead because:

- It's *simple* -- no background thread, no session lifecycle to manage,
  no "did I forget to close this" risk.
- It's not a novel choice -- it mirrors the default behavior of other MCP
  client libraries in the wild (e.g. LangChain's `MultiServerMCPClient`
  reconnects per call by default too), so it's a known, acceptable
  trade-off, not an untested one.
- `Tool.execute`'s sync path already knows how to run an async function
  via `asyncio.run` (see `tools/base.py`) -- so `MCPClient`'s tool wrapper
  is just one `async def _call(...)` function; `Tool.execute` and
  `Tool.aexecute` both work against it for free, with zero MCP-specific
  sync/async bridging code.

**Trigger to revisit:** if reconnect latency becomes a measured problem
for a real application (frequent tool calls to the same server in a tight
loop), add an opt-in persistent-session mode as an *additional*
constructor option (e.g. `MCPClient.stdio(..., persistent=True)`) rather
than changing the default -- the simple default should stay simple.

### The capability bridge holds up as designed

ADR-0001's `register_as_capability(registry, capability=..., priority=...)`
default implementation -- discover the server's tools, find the one whose
name matches `capability`, register it -- was implemented exactly as
specified and verified against a real server: `agent.requires("github")`
and `agent.requires("weather")` genuinely cannot tell, from the agent's
side, whether the resolved tool is native Python or a live MCP round-trip.
No changes to the ADR-0001 design were needed.

One addition beyond ADR-0001's spec: `register_as_capability` raises
`MCPException` (not silently registering nothing) when the server doesn't
expose a tool matching the requested capability name, listing the tools
it *does* expose -- verified this produces an actionable error message
(`"MCP server 'fs' does not expose a tool named 'does_not_exist'.
Available: ['add', 'get_weather']"`) rather than a confusing later failure
when `CapabilityRegistry.resolve` can't find anything.

### Result handling: prefer `structuredContent`, fall back to text

An MCP tool result (`CallToolResult`) carries both unstructured `content`
(text/image/etc. blocks) and optional `structuredContent` (a JSON object,
when the server provides one). `MCPClient` returns `structuredContent`
when present, falling back to concatenated text blocks otherwise. Verified
against a real `FastMCP` server: even a tool returning a plain scalar
(`def add(a, b) -> int`) gets wrapped as `{"result": 7}` in
`structuredContent` by the server SDK -- so preferring it over text is
both more information-preserving and matches what a real server actually
sends by default, not a hypothetical.

`isError=True` responses raise `MCPException` with the tool's own text
content included in the message, rather than returning an error string
as if it were a normal result -- consistent with the framework's
"never swallow, always raise with context" convention.

## Alternatives considered

- **stdio only for v1, HTTP later.** This was the default assumption
  before the roadmap discussion; overridden by an explicit decision to
  do both immediately, since the SDK already supports both behind the
  same `ClientSession` API -- the marginal implementation cost of adding
  HTTP alongside stdio turned out to be one more factory method and one
  more branch in `_session()`, not a second implementation.
- **A persistent session by default.** Rejected for v1 -- see "Per-call
  connections" above.
- **Returning only text, ignoring `structuredContent`.** Rejected once
  real-server testing showed `structuredContent` is populated by default
  and is strictly more useful to a caller (or a model parsing the tool
  result) than a stringified version of the same data.
- **A separate `StdioMCPClient` / `HTTPMCPClient` class pair.** Rejected:
  the two transports differ only in how the byte stream is established
  (`stdio_client(...)` vs `streamablehttp_client(...)`); everything after
  that (`ClientSession`, `list_tools`, `call_tool`) is identical. Two
  classes would duplicate the tool-wrapping and result-handling logic for
  no benefit.

## Consequences

### Positive

- `agent.requires("weather")` written *before* this ADR's implementation
  existed still works unmodified when `"weather"` is now resolved by a
  live MCP server instead of the built-in Open-Meteo capability --
  confirms ADR-0001's design goal was correct, not just plausible.
- Both transports are real, verified capabilities, not "stdio done,
  HTTP untested" -- the smoke tests exercised full discover + call
  round-trips on each.
- Zero new sync/async bridging code was needed in `MCPClient` itself,
  because `Tool.execute`/`Tool.aexecute` already handle wrapping an async
  function -- a concrete payoff from that design choice in the original
  tool-calling implementation.

### Negative / risks

- Per-call reconnection means an agent that calls the same MCP tool
  repeatedly in a loop pays connection setup cost every time. Not a
  correctness problem, but a real latency cost for that usage pattern --
  see the Follow-ups trigger above.
- `mcp` is now a fifth optional SDK dependency
  (`openai`/`google-genai`/`anthropic`/`langgraph`/`mcp`) with its own
  transitive dependency tree (`httpx`, `anyio`, `starlette`, `uvicorn` for
  the HTTP transport's server-side pieces, pulled in even though
  `MCPClient` only needs the *client* side). Acceptable for now since
  it's fully optional (the `mcp` extra), but worth knowing if dependency
  weight becomes a concern.
- Resource and prompt discovery (MCP's other two primitives beyond tools)
  are out of scope for this ADR entirely -- `BaseMCPClient` only exposes
  `discover_tools`. If a real use case needs MCP resources or prompts,
  that's new interface surface, not something this implementation
  silently half-supports.

### Follow-ups

- Add an opt-in persistent-session mode if reconnect latency becomes a
  measured problem (see trigger above) -- as an additional option, not a
  default change.
- Consider `discover_resources()` / `discover_prompts()` if a real use
  case for MCP resources or prompts (not just tools) emerges. Not
  speculatively added now.
- MCP server integration (exposing Requisite tools/agents *as* an MCP
  server -- the reverse direction from everything in this ADR) remains
  fully out of scope here; tracked separately in `ROADMAP.md`.
