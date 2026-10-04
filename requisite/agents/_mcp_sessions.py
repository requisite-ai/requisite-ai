"""
Agent-owned persistent MCP sessions.

``Agent(mcp_clients=[...])`` hands this class its clients. On the first run
it opens each one's persistent session, discovers its tools and registers
them on the agent, so every later tool call reuses the open connection
instead of spawning a server and running ``initialize`` per call. See
``docs/adr/0041-agent-owned-persistent-mcp-sessions.md``.

Ownership rule (extends ADR-0030): a session belongs to the event loop it was
opened on. ``run()`` opens sessions on the shared sync-bridge loop
(:func:`~requisite.core.sync_bridge.run_sync`); ``arun()`` opens them on the
caller's loop. Using the other API on an open agent raises a clear
:class:`~requisite.core.exceptions.ConfigurationException` instead of risking
the cross-loop deadlock ADR-0030 documents.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.core.sync_bridge import is_bridge_loop
from requisite.tools.base import Tool
from requisite.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from requisite.mcp.client import MCPClient

logger = logging.getLogger("requisite.agents")

# Which event loop owns the open sessions. "bridge": the shared sync-bridge
# loop -- reached by run()/connect(), and by orchestrators/ADK, whose sync run()
# drives arun() on that same loop. "user": the caller's own loop (arun() from
# application async code).
_BRIDGE = "bridge"
_USER = "user"


def _close_quietly(clients: Sequence["MCPClient"]) -> None:
    """Finalizer: best-effort close of sync-opened sessions at GC / exit."""
    for client in reversed(clients):
        try:
            client.close()
        except Exception:  # noqa: BLE001 - a finalizer must never raise
            logger.debug("Failed to close MCP client '%s' at exit", client.name, exc_info=True)


class McpSessions:
    """Opens, tracks and closes the persistent MCP sessions of one agent."""

    def __init__(
        self, agent_name: str, clients: Sequence["MCPClient"], registry: ToolRegistry
    ) -> None:
        self._agent_name = agent_name
        self._clients = list(clients)
        self._registry = registry
        self._mode: Optional[str] = None
        self._owned_tools: set[str] = set()
        self._lock = threading.Lock()  # guards sync opens across threads
        self._alock: Optional[asyncio.Lock] = None  # created on the owning loop
        self._finalizer: Optional[weakref.finalize] = None  # type: ignore[type-arg]

    def __bool__(self) -> bool:
        return bool(self._clients)

    def _wrong_mode(self) -> ConfigurationException:
        if self._mode == _BRIDGE:
            opened, use = "run() (the shared sync-bridge loop)", "close()"
        else:
            opened, use = "arun() on your own event loop", "'await agent.aclose()'"
        return ConfigurationException(
            f"Agent '{self._agent_name}' opened its MCP sessions via {opened} and cannot "
            "reuse them from the other API: a persistent MCP session is bound to the event "
            f"loop it was opened on. Call {use} first, or stick to one of run()/arun() "
            "on this agent.",
        )

    def _register(self, discovered: list[tuple["MCPClient", list[Tool]]]) -> None:
        origin: dict[str, str] = {}
        for client, tools in discovered:
            for tool in tools:
                if tool.name in origin:
                    raise ConfigurationException(
                        f"MCP clients '{origin[tool.name]}' and '{client.name}' both expose "
                        f"a tool named '{tool.name}' to Agent '{self._agent_name}'. Tool "
                        "names must be unique across an agent's MCP clients.",
                    )
                origin[tool.name] = client.name
        for name in self._owned_tools:  # re-open after close(): replace our own
            self._registry.unregister(name)
        self._owned_tools = set()
        for client, tools in discovered:
            for tool in tools:
                if tool.name in self._registry._tools:
                    logger.warning(
                        "Agent '%s': MCP tool '%s' from '%s' skipped; an explicitly "
                        "passed tool already has that name.",
                        self._agent_name,
                        tool.name,
                        client.name,
                    )
                    continue
                self._registry.register(tool)
                self._owned_tools.add(tool.name)

    # -- sync ---------------------------------------------------------------

    def open_sync(self) -> None:
        if not self._clients:
            return
        with self._lock:
            if self._mode == _BRIDGE:
                return
            if self._mode == _USER:
                raise self._wrong_mode()
            opened: list[MCPClient] = []
            try:
                discovered = []
                for client in self._clients:
                    client.connect()
                    opened.append(client)
                    discovered.append((client, client.discover_tools()))
                self._register(discovered)
            except BaseException:
                _close_quietly(opened)
                raise
            self._mode = _BRIDGE
            self._finalizer = weakref.finalize(self, _close_quietly, list(self._clients))

    def close(self) -> None:
        with self._lock:
            if self._mode is None:
                return
            if self._mode == _USER:
                raise ConfigurationException(
                    f"Agent '{self._agent_name}' opened its MCP sessions via arun() on your "
                    "own event loop; close them with 'await agent.aclose()' on that loop.",
                )
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            self._mode = None
            first_error: Optional[BaseException] = None
            for client in reversed(self._clients):
                try:
                    client.close()
                except Exception as exc:  # noqa: BLE001 - close the rest, then report
                    first_error = first_error or exc
            if first_error is not None:
                raise first_error

    # -- async --------------------------------------------------------------

    async def aopen(self) -> None:
        if not self._clients:
            return
        if self._alock is None:
            self._alock = asyncio.Lock()
        domain = _BRIDGE if is_bridge_loop(asyncio.get_running_loop()) else _USER
        async with self._alock:
            if self._mode == domain:
                return
            if self._mode is not None:
                raise self._wrong_mode()
            opened: list[MCPClient] = []
            try:
                discovered = []
                for client in self._clients:
                    await client.aconnect()
                    opened.append(client)
                    discovered.append((client, await client.adiscover_tools()))
                self._register(discovered)
            except BaseException:
                for client in reversed(opened):
                    try:
                        await client.aclose()
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to close '%s' after open error", client.name)
                raise
            self._mode = domain
            if domain == _BRIDGE:
                self._finalizer = weakref.finalize(self, _close_quietly, list(self._clients))

    async def aclose(self) -> None:
        if self._mode is None:
            return
        if self._mode == _BRIDGE:
            await asyncio.to_thread(self.close)
            return
        self._mode = None
        first_error: Optional[BaseException] = None
        for client in reversed(self._clients):
            try:
                await client.aclose()
            except Exception as exc:  # noqa: BLE001
                first_error = first_error or exc
        if first_error is not None:
            raise first_error


def build(agent_name: str, clients: Optional[Sequence[Any]], registry: ToolRegistry) -> McpSessions:
    return McpSessions(agent_name, clients or [], registry)
