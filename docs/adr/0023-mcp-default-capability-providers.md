# 0023. First-party MCP servers as default capability providers

Status: Accepted
Date: 2026-08-21

## Context

`ROADMAP.md`'s MCP section has one remaining 📋 line: *"First-party MCP
servers as default capability providers (GitHub, databases)."* This is a
pre-committed follow-up, not a speculative idea: ADR-0020 (the `github`
REST resolver) explicitly drew the boundary between itself and this line,
stating the future MCP provider *"would simply register `\"github\"` at a
higher priority than `search_github` once built... no new machinery
needed"* in `CapabilityRegistry`, and `.env.example` already reserves
`GITHUB_TOKEN` specifically for it.

The bridging primitive already exists and is tested:
`BaseMCPClient.register_as_capability(registry, *, capability, priority)`
(`requisite/mcp/base.py`) discovers an MCP server's tools and registers
the one matching `capability` into a `CapabilityRegistry`. Nothing today
actually constructs a first-party MCP client and calls it, though --
that's the real gap. Two properties of the existing primitive stand in
the way of a direct default registration:

1. It requires the MCP server's tool name to match the capability name
   **exactly** (`registry.py` raises `MCPException` otherwise). Real MCP
   servers don't expose a tool literally named `"github"` or
   `"database"` -- GitHub's official remote MCP server exposes
   operation-shaped tools (e.g. `search_repositories`).
2. It's **eager**: `discover_tools()` performs a real subprocess/network
   handshake synchronously, at registration time. Unlike
   `requisite/capabilities/resolvers.py`'s plain-function providers
   (cheap enough to register at package import time, per
   `requisite/capabilities/__init__.py`), this can't be triggered
   automatically without either paying a connection cost on every
   `import requisite`, or crashing application startup when the server
   is unreachable or unconfigured.

Separately, `.env.example` only reserves `GITHUB_TOKEN` -- nothing for
"databases." There is no single canonical database MCP server in the
ecosystem to hardcode as a default without fabricating an unverified
dependency, unlike GitHub's own officially-documented remote server.

## Decision

### New module: `requisite/mcp/defaults.py`, two functions

**`register_mcp_capability(registry, client, *, tool_name, capability, priority=0, is_available=None) -> bool`**
is the generic mechanism. Unlike `register_as_capability`, `tool_name`
and `capability` may differ -- the discovered tool is renamed via
`.model_copy(update={"name": capability})` before registration, the same
pattern `Agent.requires()` already uses (`requisite/agents/agent.py:338`)
to expose a resolved tool under its stable capability name. It catches
connection/discovery failures and a missing `tool_name`, logging a
warning and returning `False` in both cases rather than raising:

```python
try:
    tools = client.discover_tools()
except Exception:
    logger.warning(...)
    return False
for discovered in tools:
    if discovered.name == tool_name:
        tool = discovered if discovered.name == capability else discovered.model_copy(update={"name": capability})
        registry.register(capability, tool, provider_name=f"mcp:{client.name}", priority=priority, is_available=is_available)
        return True
logger.warning(...)
return False
```

This is a deliberate divergence from `register_as_capability`'s
raise-on-missing behavior (see "Alternatives considered"). It is also the
function documented for wiring up a **database** MCP server of any
vendor -- there is no first-party `register_database_mcp_capability()`,
by design (see below).

**`register_github_mcp_capability(registry, *, token=None, tool_name="search_repositories", priority=10) -> bool`**
is the concrete first-party GitHub provider. It no-ops -- returns `False`
without attempting any connection -- if neither `token=` nor
`GITHUB_TOKEN` is set, since GitHub's MCP endpoint requires
authentication and there's no unauthenticated fallback to attempt.
Otherwise it builds:

```python
MCPClient.http(
    name="github",
    url="https://api.githubcopilot.com/mcp/",
    headers={"Authorization": f"Bearer {token}"},
)
```

and calls `register_mcp_capability(..., capability="github", priority=10)`.
Priority `10` beats the existing `search_github` REST resolver's
`priority=0` (`requisite/capabilities/resolvers.py:221`), so it takes
over automatically via `CapabilityRegistry.resolve`'s existing
highest-priority-available ordering -- zero changes to the registry
itself, exactly as ADR-0020 anticipated. `tool_name` is a parameter
(default `"search_repositories"`, GitHub's documented repository-search
tool) specifically so a future rename on GitHub's side is a one-line
override, not a code change.

Both functions are **never called automatically** at import time or from
`register_default_capabilities`. The module docstring says so plainly --
calling them is an explicit application-setup step, mirroring
`resolvers.py`'s own "register a better provider at higher priority"
framing.

### Exports

Added to `requisite/mcp/__init__.py`'s `__all__`:
`register_github_mcp_capability`, `register_mcp_capability`. **Not**
re-exported from top-level `requisite/__init__.py`, matching the existing
precedent that `register_default_capabilities` is submodule-only.

### `.env.example`: `DATABASE_URL` added, generic

A `DATABASE_URL` entry is added alongside `GITHUB_TOKEN`, documented as
generic/vendor-agnostic -- read by whatever MCP database server command
the application configures via `register_mcp_capability`, not by
`Settings` or by any first-party default in this module.

### Why no hardcoded database default

Every other default in this codebase (the four `resolvers.py` functions,
every provider in `requisite/providers/`) is backed by a real, verified
integration. There is no equivalent single, canonical, officially
maintained database MCP server to point to the way GitHub maintains its
own official remote MCP server -- hardcoding a specific npm package or
vendor here would be an unverified guess baked into the framework's
default behavior. Shipping the generic `register_mcp_capability()`
mechanism, demonstrated against a Postgres-shaped example in
`examples/mcp_default_capabilities.py`, closes the *mechanism* gap
honestly without asserting a fact about the external ecosystem that
can't be verified the way GitHub's documented endpoint can.

## Alternatives considered

- **Reuse `register_as_capability` directly**, accepting the exact
  tool-name-match requirement. Rejected -- would require GitHub's MCP
  server to expose a tool literally named `"github"`, which it doesn't;
  forcing a rename means either subclassing `MCPClient` per server or
  writing the discover-and-rename logic somewhere, which is exactly what
  `register_mcp_capability` now centralizes once instead of per-caller.
- **Raise instead of returning `False` on failure**, matching
  `register_as_capability`'s existing contract. Rejected: that method is
  called by application code that already knows it wants an MCP-backed
  capability and should fail loudly if misconfigured. This module is
  explicitly for *optional default* providers meant to layer on top of
  an existing fallback (the REST resolver) -- a raised exception here
  would mean an unreachable optional GitHub MCP server crashes
  application startup instead of silently falling back to
  `search_github`, defeating the entire point of registering it as a
  higher-priority *addition* rather than a replacement.
- **Auto-register at `requisite` import time**, like
  `register_default_capabilities`. Rejected -- that function is cheap
  (registers plain Python functions, zero I/O); this one requires a real
  network/subprocess handshake per ADR-0004's per-call-connection
  design, and would either add real latency to every `import requisite`
  or attempt a connection using credentials the importing application
  may not want used yet.
- **A local stdio-based first-party GitHub server** instead of the
  hosted HTTP endpoint. Rejected as the default: GitHub's officially
  documented integration path is the hosted remote MCP server, requiring
  no local subprocess/npm dependency, matching this framework's existing
  preference (`.env.example`, `ADR-0020`) for zero-extra-dependency
  defaults. Nothing prevents an application from constructing its own
  `MCPClient.stdio(...)` against a self-hosted server and calling
  `register_mcp_capability` directly instead.
- **A hardcoded `register_database_mcp_capability()` for a specific
  vendor** (e.g. Postgres via a specific npm package). Rejected -- see
  "Why no hardcoded database default" above.

## Consequences

### Positive

- Closes the last 📋 line in `ROADMAP.md`'s MCP section.
- Purely additive: one new module, two new functions, no changes to
  `CapabilityRegistry`, `CapabilityProvider`, `BaseMCPClient`, or
  `Agent.requires(...)`.
- `agent.requires("github")` now transparently gets authenticated,
  richer GitHub access when `GITHUB_TOKEN` is configured and
  `register_github_mcp_capability()` has been called, with automatic,
  zero-code fallback to the unauthenticated REST resolver otherwise --
  the exact behavior ADR-0020 designed for.
- `register_mcp_capability()` is immediately reusable for any other MCP
  server an application wants to register as a default provider, not
  just GitHub or databases.

### Negative / risks

- `register_github_mcp_capability()`'s exact `tool_name` default
  (`"search_repositories"`) is asserted from GitHub's public MCP
  documentation, not verified against a live connection in this
  repository's test suite (which never makes real network calls, per
  the no-network-in-tests rule already documented in `tests/test_mcp.py`).
  If GitHub renames or restructures this tool, callers relying on the
  default will see a logged warning and a silent fallback to the REST
  resolver rather than a hard failure -- acceptable given this function
  registers an *additive* higher-priority provider, but worth flagging:
  the `tool_name=` override exists precisely so this is a one-line fix.
- Calling `register_github_mcp_capability()` still pays a real network
  handshake cost at the call site (not hidden, since it's never
  automatic) -- callers should call it once at application setup, not
  per-request.
- "Databases" remains mechanism-only, not a concrete default -- an
  application still has to write a few lines wiring up whichever
  database MCP server it uses. This is treated as the correct scope
  (see "Why no hardcoded database default"), not a gap to close later.

### Follow-ups

- If the equal-priority conflict-handling question (`ROADMAP.md`, "Conflict
  handling when two plugins register the same capability at equal
  priority") is ever resolved, revisit whether `register_github_mcp_capability`'s
  default `priority=10` should change -- today it's a magic number chosen
  specifically to beat `search_github`'s `priority=0` unambiguously
  rather than relying on registration order.
