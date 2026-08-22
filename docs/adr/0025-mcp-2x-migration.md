# 0025. Migrate to `mcp` 2.x

Status: Accepted
Date: 2026-08-21

## Context

`ROADMAP.md`'s MCP section carried "Migrate to `mcp` 2.x's API" as a
deliberately-deferred 📋 line: `DEVELOPMENT.md`'s dependency policy
records that `mcp` 2.0.0 reached CI unpinned and broke `mypy` with zero
code changes on requisite's side, and the response then was to cap the
constraint (`mcp>=1.28,<2.0`) and "treat the migration as its own
deliberate change" rather than rush it under CI-failure pressure. This
ADR is that deliberate change, picked up explicitly by Keyan (one of
three MCP-section 📋 lines tackled one-by-one, this one first since it's
foundational -- building persistent sessions or resource/prompt
discovery against the 1.x API first would mean redoing that work here
anyway).

This is a **SDK-forced breaking change, not a design change** --
`BaseMCPClient`'s interface (ADR-0001, ADR-0004) and `MCPServer`'s public
shape (ADR-0015) are unaffected; only `mcp`'s own internal shape moved.
Verified directly against the actual `mcp==2.0.0` and `mcp-types==2.0.0`
source (installed into a scratch directory and read, not inferred from a
changelog), which surfaced a materially larger scope than
`DEVELOPMENT.md`'s note predicted ("restructured package layout, renamed
`CallToolResult` fields, removed `streamablehttp_client`") -- the
low-level `Server`'s entire handler-registration API was rewritten too,
not just renamed.

## Decision

### Hard cutover, no dual 1.x/2.x support

`mcp` is an optional extra (`pip install requisite-ai[mcp]`), not a core
dependency -- only that extra's users are affected. Confirmed with Keyan:
`pyproject.toml`'s `mcp` extra and `all` group move straight to
`mcp>=2.0,<3.0`, with no runtime version-switch branching between the two
API shapes. This matches every other SDK-major-version case in this
framework -- `PineconeVectorStore` targets only `pinecone>=9.0`'s
serverless API (not the older `environment=` API), `WeaviateVectorStore`
only `weaviate-client>=4.0`'s v4 API -- and the project is pre-1.0, where
a minor bump may include breaking changes to an optional surface.

### Client side (`requisite/mcp/client.py`)

- `CallToolResult.structuredContent`/`.isError` -> `.structured_content`/
  `.is_error`; `Tool.inputSchema` -> `.input_schema` (both renamed to
  snake_case in `mcp_types`). `ClientSession.call_tool`/`.list_tools`
  call shapes are unchanged, so `_call_tool`/`adiscover_tools`/`_to_tool`
  needed only these field-name updates, not restructuring.
- `mcp.client.streamable_http.streamablehttp_client` no longer exists --
  renamed to `streamable_http_client`, with a fully different signature:
  `(url, *, http_client: httpx2.AsyncClient | None = None,
  terminate_on_close: bool = True)`, yielding a 2-tuple
  (`read_stream, write_stream`) instead of 1.x's 3-tuple with a
  `get_session_id` callable. `headers=`/`timeout=` no longer exist as
  kwargs -- its own docstring: *"To configure headers, authentication, or
  other HTTP settings, create an `httpx2.AsyncClient` and pass it here."*
  `httpx2` (not `httpx`) is a new package `mcp` itself depends on
  (`Requires-Dist: httpx2>=2.5.0`), present transitively with no new
  `pyproject.toml` entry needed, but now imported directly in
  `client.py`.
- Read `streamable_http_client`'s body directly to resolve client
  lifecycle ownership rather than guessing: *"Only manage client
  lifecycle if we created it"* -- since requisite passes its own
  `httpx2.AsyncClient`, `_session()`'s http branch opens and closes it
  itself, nested outside `streamable_http_client`'s own `async with`:

  ```python
  async with httpx2.AsyncClient(headers=self._headers, timeout=self._timeout) as http_client:
      async with streamable_http_client(self._url or "", http_client=http_client) as (read_stream, write_stream):
          async with ClientSession(read_stream, write_stream) as session:
              await session.initialize()
              yield session
  ```

### Server side (`requisite/mcp/server.py`) -- the real breaking change

`mcp.server.lowlevel.Server` no longer accepts post-construction
decorator-registered handlers (`server.list_tools()(handler)`,
`server.call_tool()(handler)` -- confirmed gone entirely, not just
renamed, by reading the whole class body). Handlers are now constructor
keyword arguments with new typed signatures:

```python
Server(
    name,
    on_list_tools: Callable[[ServerRequestContext, PaginatedRequestParams | None], Awaitable[ListToolsResult]] | None = None,
    on_call_tool: Callable[[ServerRequestContext, CallToolRequestParams], Awaitable[CallToolResult | InputRequiredResult]] | None = None,
    ...
)
```

- `_build_server()` now passes `on_list_tools=self._handle_list_tools,
  on_call_tool=self._handle_call_tool` to the constructor instead of
  calling decorator methods afterward.
- `_handle_list_tools(self, ctx, params)` returns a real
  `ListToolsResult(tools=[...])`, not a bare `list[Tool]`. `ctx`/`params`
  are unused by this handler (Requisite doesn't currently support
  pagination), typed `Any` rather than importing the SDK's request-context
  types just to ignore them.
- `_handle_call_tool(self, ctx, params: CallToolRequestParams)` reads
  `params.name`/`params.arguments` (confirmed fields) instead of the old
  `(name: str, arguments: dict)` positional signature, and **must build
  the full `CallToolResult` itself now** -- 1.x's decorator wrapper used
  to auto-convert a plain `dict` return into `structuredContent` (with an
  auto-generated JSON text fallback) and auto-convert a raised exception
  into an `isError=True` result; neither happens automatically anymore.
  Both are replicated manually, preserving the exact wire contract
  `docs/adr/0004-mcp-integration.md` established on the client side
  ("prefer `structured_content`, fall back to text"):

  ```python
  try:
      tool = self._tool_registry.get(params.name)
      result = await tool.aexecute(**(params.arguments or {}))
  except Exception as exc:
      return CallToolResult(content=[TextContent(type="text", text=str(exc))], is_error=True)

  structured = result if isinstance(result, dict) else {"result": result}
  return CallToolResult(
      content=[TextContent(type="text", text=json.dumps(structured, default=str))],
      structured_content=structured,
  )
  ```

  This is a genuine behavior change worth naming plainly: an unknown tool
  name or a failing tool now returns a clean `is_error=True` result from
  `_handle_call_tool` itself, rather than raising and relying on the SDK
  to convert that exception into an error response. Verified live (see
  Consequences) that this doesn't crash the server session either way --
  the explicit `try`/`except` here is the safer, more defensive choice
  regardless of what the SDK's own dispatch loop does with an unhandled
  handler exception.

### `arun_http` simplified via the SDK's new `streamable_http_app()`

`Server` gained a built-in `streamable_http_app(*, streamable_http_path=,
json_response=, host=, ...) -> Starlette` method that builds the full
Starlette app (routes, `StreamableHTTPSessionManager`, lifespan)
natively -- read directly, confirming it internally constructs the exact
same `StreamableHTTPASGIApp`-wrapping-a-session-manager-behind-a-`Route`
shape requisite's own `_StreamableHTTPASGIApp` was hand-built to
replicate in ADR-0015 (that ADR's own 405-on-POST discovery: a plain
function endpoint gets treated as GET-only by Starlette's `Route`, so a
raw ASGI-callable class was needed). The SDK now solves this itself, so
`_StreamableHTTPASGIApp` is deleted entirely and `arun_http` becomes:

```python
server = self._build_server()
app = server.streamable_http_app(streamable_http_path=path, json_response=True, host=host)
config = uvicorn.Config(app, host=host, port=port, log_level="warning")
await uvicorn.Server(config).serve()
```

This is a genuine simplification found *during* the forced migration,
not scope creep -- it removes code, doesn't add a new capability.

## Alternatives considered

- **Dual mcp 1.x/2.x runtime support** (detect installed version, branch
  between two full server-side handler-registration implementations).
  Rejected -- see "Hard cutover" above; no precedent anywhere else in
  this codebase for dual-supporting two major versions of an optional
  SDK, and the server-side API is structurally different (not just
  renamed), so genuine dual-support would roughly double
  `requisite/mcp/server.py`'s complexity and test surface for a
  compatibility window whose value only shrinks over time.
- **Keep the `<2.0` cap indefinitely.** Rejected -- this was always
  scoped as a deferred-not-skipped migration (`DEVELOPMENT.md`,
  `CHANGELOG.md`'s 0.4.1 entry), and an unbounded cap on an actively
  developed SDK accumulates risk (missed fixes/features, eventual forced
  migration under worse conditions) rather than avoiding it.
- **Let handler exceptions propagate from `_handle_call_tool`, trusting
  the SDK's own dispatch loop to convert them to an error response.**
  Rejected -- not fully verified either way from the SDK source alone
  (the dispatch/request-loop code wasn't traced that deeply), and
  catching explicitly is strictly safer regardless: it guarantees a clean
  `is_error=True` result instead of depending on undocumented SDK
  internals for a case this framework has always handled itself when
  practical.

## Consequences

### Positive

- Closes the "Migrate to `mcp` 2.x's API" 📋 line in `ROADMAP.md`'s MCP
  section.
- `arun_http` gets simpler, not just updated -- one less hand-rolled ASGI
  class to maintain, courtesy of the SDK's own new convenience method.
- Verified against real `mcp==2.0.0`, not just field-renamed mocks: a
  real stdio round trip (`MCPClient.stdio` -> a real `MCPServer`
  subprocess -- tool discovery, a successful call, and a failing call
  all confirmed) and a real Streamable HTTP round trip (`MCPClient.http`
  -> `MCPServer.run_http()`'s new `streamable_http_app()`-based server,
  over a real socket) both passed end-to-end during this migration, the
  same bar ADR-0004/ADR-0015 originally set.

### Negative / risks

- Breaking change for any application already using
  `requisite-ai[mcp]` with a pinned `mcp<2.0` elsewhere in its own
  dependency tree -- documented plainly in `CHANGELOG.md`, not silently
  absorbed.
- `_handle_call_tool`'s manual exception-to-`is_error` conversion is new,
  requisite-owned logic where 1.x delegated to the SDK -- any future
  `mcp` change to `CallToolResult`'s shape needs re-verifying here
  specifically, not just at the field-rename level.

### Follow-ups

- MCP client persistent session mode and MCP resource/prompt discovery
  remain separate, still-📋 `ROADMAP.md` lines -- explicitly out of scope
  here, picked up next per Keyan's "one by one" instruction, now
  building on the 2.x API directly rather than needing to be redone.
