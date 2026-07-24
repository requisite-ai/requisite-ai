"""
MCP client registry.

Keyed by **server name**, not capability name -- distinct from
:class:`~requisite.capabilities.registry.CapabilityRegistry`, since one
MCP server commonly exposes several tools mapping to several different
capabilities. Mirrors the shape of every other registry in the framework:
a plain, instantiable class, not a singleton.
"""

from __future__ import annotations

import logging

from requisite.core.exceptions import MCPException
from requisite.mcp.base import BaseMCPClient

logger = logging.getLogger("requisite.mcp.registry")


class MCPClientRegistry:
    """A registry of configured MCP server connections, keyed by server name.

    Examples
    --------
    >>> from requisite.mcp import MCPClient, MCPClientRegistry
    >>> registry = MCPClientRegistry()
    >>> registry.register(MCPClient.stdio(name="filesystem", command="npx", args=[...]))  # doctest: +SKIP
    >>> registry.get("filesystem").discover_tools()  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._clients: dict[str, BaseMCPClient] = {}

    def register(self, client: BaseMCPClient) -> BaseMCPClient:
        """Register a client under its ``.name``. Returns the client for chaining."""
        if client.name in self._clients:
            logger.debug(
                "Overwriting existing MCP client registration: '%s'",
                client.name,
                extra={"mcp_server": client.name},
            )
        self._clients[client.name] = client
        return client

    def unregister(self, name: str) -> None:
        """Remove an MCP client registration, if present. No-op if absent."""
        self._clients.pop(name, None)

    def get(self, name: str) -> BaseMCPClient:
        """Return the registered MCP client named ``name``.

        Raises
        ------
        requisite.core.exceptions.MCPException
            If no client is registered under that name.
        """
        try:
            return self._clients[name]
        except KeyError as exc:
            raise MCPException(
                f"No MCP server registered under '{name}'. Available: {self.list_names()}",
            ) from exc

    def list_names(self) -> list[str]:
        """Return the sorted list of registered server names."""
        return sorted(self._clients)

    def all(self) -> list[BaseMCPClient]:
        """Return all registered MCP clients."""
        return list(self._clients.values())

    def __contains__(self, name: str) -> bool:
        return name in self._clients

    def __len__(self) -> int:
        return len(self._clients)


default_registry = MCPClientRegistry()
