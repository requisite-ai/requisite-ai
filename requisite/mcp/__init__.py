"""
MCP (Model Context Protocol) client integration.

Connect to an MCP server (local via stdio, or remote via Streamable HTTP)
and expose its tools to Requisite in two ways:

- Directly, via :meth:`~requisite.mcp.base.BaseMCPClient.discover_tools`,
  returning plain :class:`~requisite.tools.base.Tool` objects usable
  anywhere a tool is (``Agent(tools=...)``, ``ToolRegistry``, ...).
- As capabilities, via
  :meth:`~requisite.mcp.base.BaseMCPClient.register_as_capability`, so
  ``agent.requires("github")`` can resolve to an MCP server exactly like
  it resolves to a native tool -- see ADR-0001 and ADR-0004.

``default_registry`` is empty by default -- unlike provider/orchestrator
registries, there's no universal default MCP server; register the ones
your application actually uses.
"""

from requisite.mcp.base import BaseMCPClient
from requisite.mcp.client import MCPClient
from requisite.mcp.registry import MCPClientRegistry
from requisite.mcp.registry import default_registry as default_mcp_registry

__all__ = ["BaseMCPClient", "MCPClient", "MCPClientRegistry", "default_mcp_registry"]
