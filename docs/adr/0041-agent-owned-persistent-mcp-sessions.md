# 0041. Agent-owned persistent MCP sessions, usable from sync code

Status: Accepted
Date: 2026-10-04

## Context

Real traces from a multi-agent lab (stdio MCP tools behind an ADK workflow)
showed `initialize` taking 5.9 s of a 6.4 s tool call, plus an extra
`tools/list` per call. Three tool calls cost 7 to 19 s of a 20 s run.

Causes, each checked in source rather than assumed:

- `MCPClient`'s default path spawns a server and runs `initialize()` on
  every call (ADR-0004). ADR-0030 added a persistent session, but only for
  async code: the sync methods raise while connected, and `Tool.execute()`
  ran a coroutine tool with `asyncio.run`, a fresh loop per call, which
  trips ADR-0030's cross-loop guard. So `Agent.run()` and anything sync
  (ADK's sync `run()`) could not use it.
- The extra `tools/list` is the MCP SDK's own: `ClientSession.call_tool`
  calls `list_tools()` when a tool's output schema is not cached
  (`mcp/client/session.py`). A fresh session per call always starts with an
  empty cache. A persistent session pays it once. No separate fix is needed.
- ADR-0030 rejected a background-thread bridge. 0.38.0 then added
  `requisite.core.sync_bridge.run_sync`, one long-lived daemon-thread loop.
  A session opened on that loop satisfies ADR-0030's "one continuous loop"
  requirement for any number of sync calls. **This supersedes ADR-0030's
  rejection of the thread bridge.**
- Nothing in `Agent` knew about MCP: users passed `tools=client.discover_tools()`
  and owned the lifecycle themselves.

## Decision

### Ownership rule

A persistent session belongs to the event loop it was opened on. There are
two kinds of owner loop:

- the shared sync-bridge loop, reached by `run()`, by `MCPClient.connect()`,
  and by orchestrators/ADK (their sync `run()` drives `arun()` on that loop);
- the caller's own loop, reached by `arun()` from application async code.

Using a session from the other kind of loop keeps raising a clear
`ConfigurationException`. Calls are never silently proxied across loops.

### MCPClient (`requisite/mcp/client.py`)

- New `connect()` / `close()` / `with client:` open and close a session on
  the bridge loop. The sync methods (`discover_tools`, `discover_resources`,
  `read_resource`, `discover_prompts`, `get_prompt`) run on that loop when
  connected that way; unconnected they behave as before; connected on a user
  loop (`aconnect`) they still fail fast.
- `aconnect()` now holds the connection in one dedicated owner task.
  Found live, not predicted: anyio cancel scopes must exit in the task that
  entered them, so `connect()` and `close()` (and an Agent opening in one
  task and closing in another) failed with "Attempted to exit cancel scope in
  a different task". The owner task enters and exits the connection itself;
  `aclose()` signals it and awaits it.

### Tool (`requisite/tools/base.py`)

`Tool.execute()` runs coroutine tools through `run_sync` instead of
`asyncio.run`. Same bug class as 0.38.0's "Event loop is closed": loop-bound
state must see one loop across calls. It is also what lets a sync tool call
reach a session living on the bridge loop.

### Agent (`requisite/agents/agent.py`, `_mcp_sessions.py`)

`Agent(mcp_clients=[...])`. The agent opens each client's persistent session
on first `run()`/`arun()` (or on `with agent:` / `async with agent:`),
registers the discovered tools, and reuses the connections. `close()` /
`aclose()` and the context managers close them; both are idempotent.
Sessions opened on the bridge loop also get a `weakref.finalize` safety net
(collected or at interpreter exit); user-loop sessions are not closed
implicitly. The mode is keyed on the owning loop, not on which API was
called: an ADK workflow's sync `run()` opens through `arun()` on the bridge
loop, and `agent.close()` from sync code must work afterwards. (A first
version keyed on the API and refused that; caught by running the real ADK
path.)

Tool names: an explicit `tools=` entry wins over an MCP tool of the same
name (logged); two MCP clients exposing the same name raise. If opening any
client fails, those already opened are closed and the agent stays reusable.

## Measured

Real stdio server subprocess, fake provider (one tool call per run, so only
MCP cost is measured), Windows, Python 3.13:

| Runs | Default (reconnect per call) | Agent-owned, sync | Agent-owned, async |
|---|---|---|---|
| 3 | 3.6 s, 4 spawns | 1.0 s, 1 spawn | 1.0 s, 1 spawn |
| 20 | 23.5 s, 21 spawns | 1.0 s, 1 spawn | 1.0 s, 1 spawn |

Protocol traffic for 3 tool calls: default 3 `initialize` + 3 `tools/list`;
agent-owned 1 + 1 in total (the discovery). Live check: real Gemini through
an ADK workflow with an MCP-backed agent, two runs, one server spawn.

## Alternatives considered

- **Client-level lifecycle only.** Rejected: leaves every Agent user to
  remember the wrapper, and does not match the ask.
- **Proxy calls across loops** (marshal onto the owner loop). Rejected: hides
  the stranded-loop hazard ADR-0030 documents; a session left on an exited
  user loop would still be unrecoverable.
- **Pre-seed the SDK's private `_tool_output_schemas` cache** to drop the
  extra `tools/list` without persistence. Rejected: private SDK state, and
  persistence already removes it along with the `initialize` cost.
- **Persistent by default.** Rejected: breaking, and it would leak
  subprocesses for code that never closes.
- **Promote `connect`/`aconnect`/`close` to `BaseMCPClient`.** Rejected, same
  YAGNI as ADR-0030: one concrete client.

## Consequences

Positive: about 23x faster at 20 tool calls on this machine; the same
mechanism works from sync code, async code and every orchestrator backend
with no per-backend change; ADR-0030's deadlock guard is kept.

Negative / risks:

- A bridge-loop session lives until `close()`, GC or exit.
- Mixing `run()` and `arun()` on one agent with open sessions raises.
- A `Tool.execute()` called from a thread that already runs the bridge loop
  falls back to a throwaway loop and so cannot reach a bridge-loop session.
- `Tool.execute()` now blocks the calling thread on the bridge loop where
  `asyncio.run` used to raise inside a running loop.

Follow-ups: `MCPClientRegistry.aclose_all()` if a multi-client cleanup case
appears.
