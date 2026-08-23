"""
MCP (Model Context Protocol) integration -- both directions.

**Client**: connect to an MCP server (local via stdio, or remote via
Streamable HTTP) and expose its tools to Requisite in two ways:

- Directly, via :meth:`~requisite.mcp.base.BaseMCPClient.discover_tools`,
  returning plain :class:`~requisite.tools.base.Tool` objects usable
  anywhere a tool is (``Agent(tools=...)``, ``ToolRegistry``, ...).
- As capabilities, via
  :meth:`~requisite.mcp.base.BaseMCPClient.register_as_capability`, so
  ``agent.requires("github")`` can resolve to an MCP server exactly like
  it resolves to a native tool -- see ADR-0001 and ADR-0004.

Resources and prompts are discoverable too, beyond tools:
:meth:`~requisite.mcp.base.BaseMCPClient.discover_resources`/
:meth:`~requisite.mcp.base.BaseMCPClient.read_resource` and
:meth:`~requisite.mcp.base.BaseMCPClient.discover_prompts`/
:meth:`~requisite.mcp.base.BaseMCPClient.get_prompt` -- the latter
returns :class:`~requisite.core.interfaces.Message` objects, so
``agent.run(client.get_prompt(...))`` composes directly with
the chat surface. See ADR-0026.

``default_registry`` is empty by default -- unlike provider/orchestrator
registries, there's no universal default MCP server; register the ones
your application actually uses.

**Server**: expose Requisite's own tools/agents/resources/prompts *as*
an MCP server via :class:`~requisite.mcp.server.MCPServer` -- the
reverse direction. See ADR-0015 (tools) and ADR-0026 (resources/prompts).

**First-party default capability providers**:
:func:`~requisite.mcp.defaults.register_github_mcp_capability` wires
GitHub's official remote MCP server in as the ``"github"`` capability
(gated on ``GITHUB_TOKEN``), and
:func:`~requisite.mcp.defaults.register_mcp_capability` is the generic
mechanism for wiring up any other first-party or third-party MCP server
(e.g. a database) the same way. Neither is registered automatically --
see ``docs/adr/0023-mcp-default-capability-providers.md``.
"""

from requisite.mcp.base import BaseMCPClient, MCPPrompt, MCPPromptArgument, MCPResource
from requisite.mcp.client import MCPClient
from requisite.mcp.defaults import register_github_mcp_capability, register_mcp_capability
from requisite.mcp.registry import MCPClientRegistry
from requisite.mcp.registry import default_registry as default_mcp_registry
from requisite.mcp.server import MCPServer

__all__ = [
    "BaseMCPClient",
    "MCPClient",
    "MCPClientRegistry",
    "MCPPrompt",
    "MCPPromptArgument",
    "MCPResource",
    "MCPServer",
    "default_mcp_registry",
    "register_github_mcp_capability",
    "register_mcp_capability",
]
