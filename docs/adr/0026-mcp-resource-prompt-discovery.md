# 0026. MCP resource / prompt discovery

Status: Accepted
Date: 2026-08-23

## Context

`ROADMAP.md`'s MCP section had one remaining 📋 line: *"MCP resource /
prompt discovery (beyond tools) | 📋 | Out of scope for the initial
client — ADR-0004."* Keyan picked this up next (of the two MCP items
left after ADR-0025's `mcp` 2.x migration), explicitly deferring
persistent session mode over it -- ADR-0004's own reason for deferring
that ("until reconnect latency is a measured problem") still holds, and
implementing it now would override that documented precondition without
new evidence. Keyan also asked for this to cover **both directions**,
not just the client: `MCPServer` (ADR-0015) gains the ability to expose
resources/prompts too, not only tools, so the feature could be verified
with a full self-contained real round trip purely within Requisite,
matching the rigor ADR-0025 just set.

Verified directly against the real, already-installed `mcp==2.0.0` (read
from source and exercised live in this venv, not assumed):

- `ClientSession` already exposes `list_resources()`, `read_resource(uri)`,
  `list_prompts()`, `get_prompt(name, arguments)` -- present and
  unchanged in shape by ADR-0025's migration.
- `mcp.types.Resource(uri, name, description, mime_type, size, ...)`;
  `ReadResourceResult.contents: list[TextResourceContents |
  BlobResourceContents]`, where text items carry `.text` and blob items
  `.blob`, with **no shared discriminator field** -- distinguishing them
  means checking for `.text`'s presence (`getattr(c, "text", None)`),
  not a `type` tag.
- `mcp.types.Prompt(name, description, arguments: list[PromptArgument])`;
  `GetPromptResult.messages: list[PromptMessage]`, where
  `PromptMessage.role` is `Literal["user", "assistant"]` (confirmed
  live via `list(Role)` -- **not** the full sampling/chat role set) and
  `.content` is a single `ContentBlock`, not a list.
- Server-side: `Server.__init__` already accepts `on_list_resources`,
  `on_read_resource`, `on_list_prompts`, `on_get_prompt` constructor
  kwargs -- the same constructor-kwargs shape ADR-0025 already wired
  `on_list_tools`/`on_call_tool` into, so no new registration mechanism
  is needed on the server side, only two more handlers.
- **The key finding that settled the error-handling design**: unlike
  `CallToolResult` (which carries an `is_error` field precisely so a
  tool-execution failure can be reported as a "successful" result with a
  flag), `ReadResourceResult`/`GetPromptResult` have **no equivalent
  field** -- there is no in-result way to signal "not found." Read
  `mcp.server.runner`'s dispatch loop directly rather than guessing:
  *any* exception raised from a handler is already caught centrally and
  converted into a proper JSON-RPC error response
  (`except Exception as exc: ... raise MCPError(...)`, gated only by
  `raise_exceptions=True` on `server.run()`, which Requisite never
  passes). Verified live that raising `mcp.MCPError(code=INVALID_PARAMS,
  message=...)` surfaces that exact message to the client, whereas a
  plain exception is masked to a generic "Internal server error" by
  `modern_error_data`'s ladder (which only preserves `MCPError`/
  `ValidationError` messages).

## Decision

### New Requisite-native types, `requisite/mcp/base.py`

Alongside `BaseMCPClient`, the same placement `Chunk`/`ScoredChunk` use
next to `BaseRetriever` in `requisite/rag/base.py`:

```python
class MCPResource(BaseModel):
    uri: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None

class MCPPromptArgument(BaseModel):
    name: str
    description: str = ""
    required: bool = False

class MCPPrompt(BaseModel):
    name: str
    description: str = ""
    arguments: list[MCPPromptArgument] = Field(default_factory=list)
```

### `BaseMCPClient`: 4 new operations, 8 methods (sync + async each)

Same shape as `discover_tools`/`adiscover_tools` -- both abstract, no
shared default, each concrete client implements both directly:

- `discover_resources()` / `adiscover_resources() -> list[MCPResource]`
- `read_resource(uri)` / `aread_resource(uri) -> str`
- `discover_prompts()` / `adiscover_prompts() -> list[MCPPrompt]`
- `get_prompt(name, arguments=None)` / `aget_prompt(...) -> list[Message]`

`get_prompt` deliberately returns
:class:`~requisite.core.interfaces.Message`, Requisite's own chat type,
not a raw MCP object -- so `agent.run(client.get_prompt(...))`
composes directly with the rest of the framework's chat surface, the
same way `discover_tools()` returning `list[Tool]` lets discovered tools
compose directly with `Agent.requires`/`ToolRegistry`. Each
`PromptMessage.role` ("user"/"assistant") converts to Requisite's
broader `Role` enum via `Role(prompt_message.role)`.

### `MCPClient`: text-only for now, explicit not silent

`read_resource`/`aread_resource` raise `MCPException` if a resource has
no text content at all (binary-only); `get_prompt`/`aget_prompt` raise
`MCPException` if any rendered message has non-text content. Both are a
deliberate v1 scope line, not an oversight -- see Alternatives.

### `MCPServer`: `add_resource`/`add_prompt` + 4 new handlers

New registration methods, mirroring `add_tool`/`add_agent`'s "explicit
method, not a constructor list" shape (resources/prompts need a small
bundle of metadata + callable, which doesn't fit the same
one-positional-arg convenience `tools=`/`agents=` use -- deliberately
not added to `__init__`):

```python
def add_resource(self, uri, *, name, content, description="", mime_type=None) -> None: ...
def add_prompt(self, name, *, render, description="", arguments=None) -> None: ...
```

`content` is either static text or a zero-arg callable producing the
current text each time the resource is read (so a resource can be
dynamic, e.g. re-reading a file); `render` takes the filled-in arguments
dict and returns the `list[Message]` the prompt expands to -- the exact
inverse of what `get_prompt` returns client-side, so an application's
own prompt-rendering logic is the same shape on both ends.

Handlers wired into `_build_server()`'s existing `Server(...)`
constructor call, alongside `on_list_tools`/`on_call_tool`:

- `_handle_list_resources`/`_handle_list_prompts` -- straightforward,
  wrap the registered dict's values into `ListResourcesResult`/
  `ListPromptsResult`.
- `_handle_read_resource`/`_handle_get_prompt` -- an unknown URI/name
  raises `MCPError(code=INVALID_PARAMS, message=...)` directly (per the
  verified dispatch-loop behavior above), not a manually-built "error
  result," since no such field exists on these two result types. A
  rendered `Message` with a `SYSTEM`/`TOOL` role isn't validated up
  front in `_handle_get_prompt` -- it fails naturally at `PromptMessage`
  construction (MCP's role type is `Literal["user", "assistant"]` only),
  surfaced as a generic internal error. Documented as a known v1
  limitation, not silently handled.

## Alternatives considered

- **Silently coerce binary resource content / non-text prompt content**
  (e.g. base64-decode a blob into the returned string, or stringify a
  non-text content block). Rejected -- would produce a "successful"
  result the caller can't actually use correctly without knowing it's
  encoded differently, which is worse than a clear, immediate
  `MCPException`. Revisit if a concrete use case needs binary resources
  (see Follow-ups).
- **Validate a `Message`'s role against MCP's user/assistant-only set
  before calling `prompt.render(...)`'s result into `PromptMessage`.**
  Rejected for v1 -- adds a validation layer for a case that already
  fails cleanly (if unhelpfully) at construction; an application author
  writing their own `render` callable controls this and can be told
  about the constraint directly in the docstring rather than requiring
  a new checked exception type.
- **Client-side only**, matching ADR-0004's literal "out of scope for
  the initial client" wording and skipping `MCPServer` changes.
  Rejected -- explicitly requested by Keyan specifically so this feature
  could be verified with a real, self-contained round trip (register on
  a real `MCPServer`, discover/fetch with a real `MCPClient`) rather than
  depending on some third-party server's incidental resource/prompt
  support.
- **A `resources=`/`prompts=` constructor-list parameter on
  `MCPServer.__init__`**, matching `tools=`/`agents=`. Rejected -- see
  "explicit method, not a constructor list" above; each resource/prompt
  needs more than one positional value (uri/name + content, or
  name + render + arguments), which doesn't fit the same one-item-list
  convenience shape.

## Consequences

### Positive

- Closes the last 📋 line in `ROADMAP.md`'s MCP section.
- `get_prompt()`/`render=` sharing `list[Message]` as their common
  currency, in both directions, is a genuinely useful integration point
  -- an MCP-hosted prompt template can seed a real `agent.run(...)`
  call with zero translation code.
- Verified with a real, self-contained round trip (not just mocks): a
  real `MCPServer` subprocess with a registered resource and prompt, a
  real `MCPClient.stdio(...)` discovering and fetching both, and a real
  unknown-URI/unknown-name error surfacing its actual message rather
  than a masked generic one.

### Negative / risks

- Text-only is a real functional limit, not just an implementation
  detail -- an MCP server whose resources are genuinely binary (images,
  PDFs) or whose prompts embed non-text content (e.g. `EmbeddedResource`
  blocks) can't be used through this API yet.
- `MCPServer.add_resource`'s dynamic `content` callable has no error
  handling of its own -- if it raises, that exception propagates through
  `_handle_read_resource` uncaught (not wrapped in `MCPError`), so it
  gets the generic, message-masked "Internal server error" treatment
  rather than a specific one. Acceptable for v1 (an application's own
  callable failing is arguably an internal error), but worth knowing.

### Follow-ups

- Binary resource support (return raw bytes / base64, or a richer result
  type distinguishing text from blob) if a concrete use case needs it.
- Non-text prompt-message content support (embedded resources, images)
  if a concrete use case needs it.
- Resource templates (`ResourceTemplate`/`resources/templates/list`)
  weren't scoped here -- `ROADMAP.md`'s line named "resource / prompt"
  specifically, not templates; revisit if requested separately.
