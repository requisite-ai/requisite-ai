"""
MCP server: expose Requisite tools/agents *as* an MCP server -- the
reverse direction of :class:`~requisite.mcp.client.MCPClient`.

Built on the ``mcp`` SDK's low-level ``mcp.server.lowlevel.Server``, not
the high-level ``FastMCP``/``mcpserver``. Its tool layer only ever builds
a tool from a raw Python function, deriving its own pydantic-based
argument schema and validation every time -- there is no supported way
to hand it a pre-computed schema. Requisite's own
:class:`~requisite.tools.base.Tool` already carries a JSON Schema
(``parameters_schema``, from :mod:`requisite.tools.schema`) and its own
execution path (``execute``/``aexecute``); the low-level ``Server`` is
schema-agnostic -- it validates incoming calls with plain
``jsonschema.validate()`` against whatever ``input_schema`` dict you give
it -- so it maps onto an existing ``Tool`` directly, with no second,
independent schema-derivation system to keep in sync. See
``docs/adr/0015-mcp-server-integration.md`` (original design) and
``docs/adr/0025-mcp-2x-migration.md`` (the ``mcp`` 2.x rewrite of how
handlers are registered).

Install with: ``pip install mcp``
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from requisite.agents.agent import Agent
from requisite.core.exceptions import ConfigurationException
from requisite.tools.base import Tool
from requisite.tools.registry import ToolLike, ToolRegistry

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server as MCPLowLevelServer
    from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult

logger = logging.getLogger("requisite.mcp.server")

_DEFAULT_HTTP_HOST = "127.0.0.1"
_DEFAULT_HTTP_PORT = 8000
_DEFAULT_HTTP_PATH = "/mcp"


class MCPServer:
    """Exposes a set of Requisite tools and/or agents as an MCP server.

    Parameters
    ----------
    name:
        Server name, reported to connecting MCP clients during
        initialization.
    tools:
        Tools to expose, in any form :class:`~requisite.tools.base.Tool`,
        a plain function, or an ``@tool``-decorated function.
    agents:
        Agents to expose. Each is added via :meth:`add_agent` -- exposed
        as one MCP tool taking a ``prompt`` argument (see
        :meth:`~requisite.agents.agent.Agent.as_tool`).

    Notes
    -----
    One class, no transport subclasses: stdio and Streamable HTTP only
    differ in how the byte stream is established, exactly like
    :class:`~requisite.mcp.client.MCPClient`'s own ``stdio``/``http``
    factories -- everything downstream (tool listing, call dispatch) is
    identical either way.

    Examples
    --------
    >>> from requisite.mcp import MCPServer
    >>> from requisite.tools import tool
    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>> server = MCPServer(name="my-tools", tools=[add])
    >>> server.run_stdio()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        name: str,
        tools: Optional[list[ToolLike]] = None,
        agents: Optional[list[Agent]] = None,
    ) -> None:
        self._name = name
        self._tool_registry = ToolRegistry()
        for tool_like in tools or []:
            self._tool_registry.register(tool_like)
        for agent in agents or []:
            self.add_agent(agent)

    @property
    def name(self) -> str:
        return self._name

    def add_tool(self, tool_or_func: ToolLike) -> Tool:
        """Register an additional tool. Returns the registered :class:`Tool`."""
        return self._tool_registry.register(tool_or_func)

    def add_agent(self, agent: Agent) -> Tool:
        """Register an agent, exposed as one MCP tool (see :meth:`Agent.as_tool`).

        Returns the registered :class:`~requisite.tools.base.Tool`.
        """
        return self._tool_registry.register(agent.as_tool())

    def _build_server(self) -> "MCPLowLevelServer":
        try:
            from mcp.server.lowlevel import Server
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'mcp' package is required for MCPServer. Install it with: pip install mcp",
            ) from exc

        # mcp 2.x: handlers are constructor kwargs, not post-construction
        # decorators (server.list_tools()(handler) no longer exists) -- see
        # docs/adr/0025-mcp-2x-migration.md.
        return Server(
            self._name,
            on_list_tools=self._handle_list_tools,
            on_call_tool=self._handle_call_tool,
        )

    async def _handle_list_tools(self, ctx: Any, params: Any) -> "ListToolsResult":
        from mcp.types import ListToolsResult
        from mcp.types import Tool as MCPTool

        return ListToolsResult(
            tools=[
                MCPTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.parameters_schema or {"type": "object", "properties": {}},
                )
                for tool in self._tool_registry.all()
            ]
        )

    async def _handle_call_tool(
        self, ctx: Any, params: "CallToolRequestParams"
    ) -> "CallToolResult":
        """Run one tool call and return a full ``CallToolResult``.

        mcp 2.x's ``on_call_tool`` handler must build the result itself --
        unlike 1.x's decorator-based registration, there is no longer any
        automatic dict-return-becomes-``structured_content`` wrapping, and
        no automatic exception-to-``is_error`` conversion (both used to
        happen inside the SDK's own decorator wrapper; see
        ``docs/adr/0025-mcp-2x-migration.md``). Replicated manually here to
        preserve the "prefer structured_content, fall back to text" contract
        ``docs/adr/0004-mcp-integration.md`` established on the client side,
        and to keep an unknown tool name or a failing tool from crashing the
        server session instead of returning a clean error result.
        """
        from mcp.types import CallToolResult, TextContent

        try:
            tool = self._tool_registry.get(params.name)
            result = await tool.aexecute(**(params.arguments or {}))
        except Exception as exc:  # noqa: BLE001
            return CallToolResult(content=[TextContent(type="text", text=str(exc))], is_error=True)

        structured = result if isinstance(result, dict) else {"result": result}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(structured, default=str))],
            structured_content=structured,
        )

    async def arun_stdio(self) -> None:
        """Serve over stdio (local subprocess), asynchronously."""
        from mcp.server.stdio import stdio_server

        server = self._build_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    def run_stdio(self) -> None:
        """Serve over stdio (local subprocess), blocking until the client disconnects."""
        asyncio.run(self.arun_stdio())

    async def arun_http(
        self,
        *,
        host: str = _DEFAULT_HTTP_HOST,
        port: int = _DEFAULT_HTTP_PORT,
        path: str = _DEFAULT_HTTP_PATH,
    ) -> None:
        """Serve over Streamable HTTP, asynchronously, until cancelled."""
        import uvicorn

        # mcp 2.x's Server.streamable_http_app(...) builds the full
        # Starlette app (routes, session manager, lifespan) natively --
        # see docs/adr/0025-mcp-2x-migration.md. This replaces the
        # hand-rolled StreamableHTTPSessionManager/_StreamableHTTPASGIApp/
        # Starlette(routes=...) construction 1.x required (ADR-0015's
        # 405-on-POST workaround is now solved by the SDK itself).
        server = self._build_server()
        app = server.streamable_http_app(streamable_http_path=path, json_response=True, host=host)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        await uvicorn.Server(config).serve()

    def run_http(
        self,
        *,
        host: str = _DEFAULT_HTTP_HOST,
        port: int = _DEFAULT_HTTP_PORT,
        path: str = _DEFAULT_HTTP_PATH,
    ) -> None:
        """Serve over Streamable HTTP, blocking until interrupted."""
        asyncio.run(self.arun_http(host=host, port=port, path=path))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"MCPServer(name={self._name!r})"
