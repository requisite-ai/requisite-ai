"""
MCP server: expose Requisite tools/agents *as* an MCP server -- the
reverse direction of :class:`~requisite.mcp.client.MCPClient`.

Built on the ``mcp`` SDK's low-level ``mcp.server.lowlevel.Server``, not
the high-level ``FastMCP``. ``FastMCP``'s tool layer only ever builds a
tool from a raw Python function (``Tool.from_function``), deriving its
own pydantic-based argument schema and validation every time -- there is
no supported way to hand it a pre-computed schema. Requisite's own
:class:`~requisite.tools.base.Tool` already carries a JSON Schema
(``parameters_schema``, from :mod:`requisite.tools.schema`) and its own
execution path (``execute``/``aexecute``); the low-level ``Server`` is
schema-agnostic -- it validates incoming calls with plain
``jsonschema.validate()`` against whatever ``inputSchema`` dict you give
it -- so it maps onto an existing ``Tool`` directly, with no second,
independent schema-derivation system to keep in sync. See
``docs/adr/0015-mcp-server-integration.md``.

Install with: ``pip install mcp``
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from requisite.agents.agent import Agent
from requisite.core.exceptions import ConfigurationException
from requisite.tools.base import Tool
from requisite.tools.registry import ToolLike, ToolRegistry

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server as MCPLowLevelServer
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.types import Tool as MCPTool
    from starlette.types import Receive, Scope, Send

logger = logging.getLogger("requisite.mcp.server")

_DEFAULT_HTTP_HOST = "127.0.0.1"
_DEFAULT_HTTP_PORT = 8000
_DEFAULT_HTTP_PATH = "/mcp"


class _StreamableHTTPASGIApp:
    """A raw ASGI callable wrapping a ``StreamableHTTPSessionManager``.

    Starlette's ``Route`` restricts a plain function endpoint to ``GET``
    by default -- Streamable HTTP needs ``POST``/``DELETE`` too. A
    callable *class instance* is treated as a raw ASGI app instead,
    bypassing that method filtering entirely; this mirrors
    ``mcp.server.fastmcp.server.StreamableHTTPASGIApp`` exactly (verified
    against it directly: a plain function here returns 405 on every
    real POST request).
    """

    def __init__(self, session_manager: "StreamableHTTPSessionManager") -> None:
        self._session_manager = session_manager

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        await self._session_manager.handle_request(scope, receive, send)


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

        server = Server(self._name)
        # mcp's own Server.list_tools() decorator factory carries no
        # return-type annotation in the installed SDK -- mypy strict flags
        # calling it as an untyped call even though the module itself
        # resolves fine (mcp.* is already exempted from missing-stub errors).
        server.list_tools()(self._handle_list_tools)  # type: ignore[no-untyped-call]
        server.call_tool()(self._handle_call_tool)
        return server

    async def _handle_list_tools(self) -> list["MCPTool"]:
        from mcp.types import Tool as MCPTool

        return [
            MCPTool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.parameters_schema or {"type": "object", "properties": {}},
            )
            for tool in self._tool_registry.all()
        ]

    async def _handle_call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call and return a JSON-object result.

        Returning a plain ``dict`` here is deliberate: the SDK's own
        ``call_tool()`` decorator treats a ``dict`` return as structured
        content, populating ``CallToolResult.structuredContent`` *and*
        auto-generating a JSON text fallback -- matching the
        "prefer structuredContent, fall back to text" behavior
        ``docs/adr/0004-mcp-integration.md`` established on the client
        side, with no manual result-building needed here. Likewise, input
        validation (against the tool's own ``inputSchema``) and turning
        an exception raised here into an ``isError=True`` result both
        happen inside the SDK's decorator wrapper -- not duplicated here.
        """
        tool = self._tool_registry.get(name)
        result = await tool.aexecute(**arguments)
        return result if isinstance(result, dict) else {"result": result}

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
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Route

        server = self._build_server()
        session_manager = StreamableHTTPSessionManager(app=server, json_response=True)

        app = Starlette(
            routes=[Route(path, endpoint=_StreamableHTTPASGIApp(session_manager))],
            lifespan=lambda _app: session_manager.run(),
        )
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
