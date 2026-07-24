"""
MCP client interface.

Per ADR-0001, the design goal is that an MCP-backed tool is *just
another* :class:`~requisite.capabilities.registry.CapabilityProvider` --
``agent.requires("github")`` shouldn't need to know or care whether
``"github"`` is resolved by a native tool or an MCP server. That
constrains :class:`BaseMCPClient` to one job: discover an MCP server's
tools and hand back :class:`~requisite.tools.base.Tool` objects, so the
rest of the framework never has to special-case "MCP-ness."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from requisite.core.exceptions import MCPException

if TYPE_CHECKING:
    from requisite.capabilities.registry import CapabilityRegistry
    from requisite.tools.base import Tool


class BaseMCPClient(ABC):
    """Abstract interface for a connection to one MCP server.

    A "server" here is one configured MCP endpoint (one stdio subprocess
    command, or one Streamable HTTP URL) -- not the whole MCP ecosystem.
    An application typically constructs one :class:`BaseMCPClient` per
    MCP server it wants to use, and registers each with an
    :class:`~requisite.mcp.registry.MCPClientRegistry` under a short name.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique identifier for this server connection (e.g. ``"github"``).

        This is a *connection* name, not a capability name -- a single
        server commonly exposes several tools mapping to several
        different capabilities, which is why
        :class:`~requisite.mcp.registry.MCPClientRegistry` (keyed by this
        name) is a separate registry from
        :class:`~requisite.capabilities.registry.CapabilityRegistry`
        (keyed by capability name).
        """

    @abstractmethod
    def discover_tools(self) -> list["Tool"]:
        """Connect to the MCP server and return its exposed tools.

        Each returned :class:`~requisite.tools.base.Tool`'s ``execute``
        proxies the call over MCP -- calling it re-connects to the server,
        invokes the tool, and disconnects. See
        ``docs/adr/0004-mcp-integration.md`` for why per-call connections
        were chosen over a persistent session for this first
        implementation.
        """

    @abstractmethod
    async def adiscover_tools(self) -> list["Tool"]:
        """Async counterpart to :meth:`discover_tools`."""

    def register_as_capability(
        self, registry: "CapabilityRegistry", *, capability: str, priority: int = 0
    ) -> None:
        """Discover this server's tools and register the one named
        ``capability`` into ``registry`` under that same name.

        Parameters
        ----------
        registry:
            The :class:`~requisite.capabilities.registry.CapabilityRegistry`
            to register into.
        capability:
            The capability name to register -- must match the MCP tool's
            own name exactly. Override this method for a server whose
            tool naming doesn't line up with your capability names.
        priority:
            Passed through to ``CapabilityRegistry.register``.

        Raises
        ------
        requisite.core.exceptions.MCPException
            If the server exposes no tool named ``capability``.
        """
        tools = self.discover_tools()
        for discovered_tool in tools:
            if discovered_tool.name == capability:
                registry.register(
                    capability,
                    discovered_tool,
                    provider_name=f"mcp:{self.name}",
                    priority=priority,
                )
                return
        raise MCPException(
            f"MCP server '{self.name}' does not expose a tool named '{capability}'. "
            f"Available: {[t.name for t in tools]}",
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(name={self.name!r})"
