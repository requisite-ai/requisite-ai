
# 0015. MCP server integration: exposing Requisite as an MCP server

Status: Accepted
Date: 2026-08-11

## Context

`ROADMAP.md`'s MCP section had one remaining line: *"MCP server
integration (expose Requisite tools/agents as an MCP server)"* -- the
reverse direction of the MCP *client* integration
(`requisite/mcp/client.py`, ADR-0004). ADR-0004 explicitly deferred this:
*"MCP server integration... remains fully out of scope here; tracked
separately in `ROADMAP.md`."* This ADR is that follow-up.

The `mcp` SDK (pinned `>=1.28,<2.0`, same constraint as the client --
server support predates the 2.0 breaking rewrite and is unaffected by
that cap) ships two ways to build a server: high-level `FastMCP`
(decorator-based, introspects Python functions itself) and low-level
`mcp.server.lowlevel.Server` (schema-agnostic, takes pre-built
`mcp.types.Tool` objects). Requisite's own `Tool`
(`requisite/tools/base.py`) already carries a JSON Schema
(`parameters_schema`, from `requisite/tools/schema.py`) and its own
execution path (`execute`/`aexecute`). Reading `FastMCP`'s internal tool
class (`mcp/server/fastmcp/tools/base.py`) directly confirmed it has no
supported way to accept a pre-computed schema: `Tool.from_function()` is
the only construction path, and it always calls `func_metadata()` to
build a pydantic argument model that `Tool.run()` then depends on for
validation. Using `FastMCP` would make Requisite's own schema vestigial
and route every argument through two independent, potentially-disagreeing
schema systems (Requisite's hand-rolled mapper vs. pydantic's).

## Decision

### Built on `mcp.server.lowlevel.Server`, not `FastMCP`

`requisite/mcp/server.py`'s `MCPServer` constructs a low-level `Server`
and registers two handlers directly:

```python
server = Server(self._name)
server.list_tools()(self._handle_list_tools)
server.call_tool()(self._handle_call_tool)
```

`_handle_list_tools` maps each Requisite `Tool` straight onto
`mcp.types.Tool(name=, description=, inputSchema=tool.parameters_schema)`
-- no re-derivation. `_handle_call_tool` is deliberately thin:

```python
async def _handle_call_tool(self, name, arguments) -> dict:
    tool = self._tool_registry.get(name)
    result = await tool.aexecute(**arguments)
    return result if isinstance(result, dict) else {"result": result}
```

Reading `mcp/server/lowlevel/server.py`'s `call_tool()` decorator
implementation directly (not assumed) showed it already does the
protocol work a hand-rolled server would otherwise have to duplicate:
input validation via `jsonschema.validate(instance=arguments,
schema=tool.inputSchema)` against the same tool definition
`list_tools()` returned, `isError=True` `CallToolResult` construction for
*any* exception the registered handler raises, and -- critically --
treating a `dict` return as structured content, populating both
`CallToolResult.structuredContent` **and** an auto-generated JSON text
fallback. That last point is exactly the "prefer structuredContent, fall
back to text" behavior ADR-0004 established on the client side, gotten
for free by returning a `dict`. None of validation, error-wrapping, or
text-fallback generation is duplicated in `MCPServer` itself.

### One concrete class, no `BaseMCPServer` ABC, no registry

Mirrors `MCPClient`'s own precedent: ADR-0004 rejected separate
`StdioMCPClient`/`HTTPMCPClient` classes because everything past
byte-stream setup is identical. The same holds here --
`run_stdio`/`arun_stdio` and `run_http`/`arun_http` are two methods on
one `MCPServer`, not two classes. No `MCPServerRegistry` either: an app
runs *one* server of its own, unlike `MCPClientRegistry` (several named
*external* connections to look up).

### `Agent.as_tool()`

`requisite/agents/agent.py` gained `Agent.as_tool() -> Tool`, mirroring
`BaseSkill.as_tool()` exactly in name and shape -- wraps
`agent.arun(prompt)` as a single-`prompt`-argument `Tool`. This is how
"expose Requisite **agents**" (not just tools) is satisfied:
`MCPServer(agents=[...])`/`.add_agent(...)` registers
`agent.as_tool()` into the same tool registry `_handle_list_tools`/
`_handle_call_tool` already serve, with zero MCP-specific agent-handling
code. `as_tool()` is usable standalone too (agent-as-tool composition is
not MCP-specific).

### Both stdio and Streamable HTTP, verified real (not just implemented)

Mirrors ADR-0004's client-side "both from day one." `arun_stdio` mirrors
`MCPClient`'s `stdio_client` usage closely (`mcp.server.stdio.stdio_server()`
+ `server.run(...)`). `arun_http` builds on
`mcp.server.streamable_http_manager.StreamableHTTPSessionManager`, which
accepts a low-level `Server` directly, wrapped in a small Starlette app
run via `uvicorn` -- both already transitive dependencies of the `mcp`
extra (ADR-0004's own "Negative/risks" section already flagged
`starlette`/`uvicorn` as pulled in for exactly this future use), so no
new dependency.

**One real bug found and fixed only by testing against a real HTTP round
trip, not by reading the SDK source**: the first `arun_http`
implementation registered a plain `async def` function as the Starlette
`Route`'s `endpoint`. Every real request returned `405 Method Not
Allowed`. Starlette treats a plain function endpoint as a `GET`-only
request handler by default; only a *class instance* with `__call__` is
treated as a raw ASGI app with no method filtering. Reading
`mcp.server.fastmcp.server.StreamableHTTPASGIApp` (the class `FastMCP`
itself uses for exactly this) confirmed the pattern; `MCPServer` now
defines an equivalent `_StreamableHTTPASGIApp` class and uses an instance
of it as the route endpoint. This is the second time this stretch a
real-network check caught something code review and static analysis
both missed clean (the first was `RedisMemory`'s Windows
`localhost`-vs-`127.0.0.1` latency bug) -- concrete evidence for why
`DEVELOPMENT.md`'s verification step isn't optional for protocol-level
code.

Verified live end to end for both transports, using Requisite's own
`MCPClient` as the counterparty (no external MCP server or third-party
tooling needed): `examples/mcp_server_example.py` run as a real stdio
subprocess, and `MCPServer.run_http(...)` run as a real local HTTP
server -- in each case: `discover_tools()`, a successful tool call, a
successful agent-backed tool call (a genuine Gemini API call executing
*inside* the server process), and a deliberately invalid call confirmed
to surface as `ToolException` wrapping `MCPException` on the client side
(the `isError=True` path, working exactly as designed).

## Alternatives considered

- **`FastMCP`.** Rejected -- see Context above. Its tool layer owns
  schema derivation and argument validation itself with no supported
  override point; using it would make Requisite's own `Tool.parameters_schema`
  dead weight and introduce a second, independent schema system.
- **A `BaseMCPServer` ABC + registry**, matching the client side's
  interface/implementation/registry shape. Rejected: there's exactly one
  server backend (the `mcp` SDK) with two transports differing only in
  byte-stream setup -- the same reasoning ADR-0004 already used to reject
  splitting `MCPClient` by transport. A registry has no clear job either:
  unlike connecting to several named external MCP servers, an
  application runs one server of its own.
- **Manually constructing `CallToolResult` (content + structuredContent)
  in `_handle_call_tool`.** Rejected once reading `Server.call_tool()`'s
  own implementation showed it already builds exactly that from a plain
  `dict` return, including the JSON text fallback -- duplicating it would
  be more code with more chances to drift from the SDK's own behavior.
- **A plain function as the Starlette route endpoint for
  `arun_http`.** This was the first implementation, and it was wrong --
  see the bug described above. Kept as an explicit alternative here
  because it's exactly the kind of mistake that looks correct until
  tested against a real request.

## Consequences

### Positive

- Closes the last `📋` line in `ROADMAP.md`'s MCP section.
- `Agent.as_tool()` is a small, genuinely reusable addition beyond MCP --
  any code that wants to treat a full tool-calling agent as a single
  callable can use it.
- Zero new dependencies. `mcp`'s existing `>=1.28,<2.0` extra already
  covers everything (`mcp.server.*`, `starlette`, `uvicorn`).
- `_handle_call_tool`'s thinness (defer validation, error-wrapping, and
  result-shaping to the SDK's own decorator) means less Requisite-side
  code to keep correct as the `mcp` SDK evolves.

### Negative / risks

- `_handle_call_tool` always returns a `dict` (wrapping non-dict results
  as `{"result": ...}`), so a tool's raw return type isn't preserved
  exactly across the MCP boundary -- matches what ADR-0004 observed real
  `FastMCP` servers already do, but it's a real, if minor,
  information-shape change for tools that return e.g. a bare list.
- No MCP resources or prompts -- only tools (and agents-as-tools) are
  exposed, matching `BaseMCPClient`'s existing client-side scope
  (`discover_tools` only, per ADR-0004).
- `run_http`'s Starlette/uvicorn wiring is real, non-trivial protocol
  plumbing (`StreamableHTTPSessionManager`, ASGI app, ASGI-callable-class
  requirement) that a contributor extending it (auth, custom routes,
  middleware) needs to understand at the Starlette level, not just at
  Requisite's `MCPServer` API surface.

### Follow-ups

- No persistent-session mode is needed on the server side the way
  ADR-0004 flagged for the client -- a server naturally holds one session
  per connected client for the lifetime of that connection; this isn't a
  gap to revisit.
- If a real use case needs MCP resources/prompts exposed (not just
  tools), that's new interface surface on `MCPServer`, matching how
  ADR-0004 scoped the client's `discover_resources()`/`discover_prompts()`
  as a future, not-yet-justified addition.
- `requisite serve`-style CLI integration (scaffold + run an `MCPServer`
  from the command line, alongside `requisite chat`) is a natural next
  step for `requisite/cli/` (ADR-0014) but is not scoped here.
