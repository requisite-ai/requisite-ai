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
``docs/adr/0015-mcp-server-integration.md`` (original design),
``docs/adr/0025-mcp-2x-migration.md`` (the ``mcp`` 2.x rewrite of how
handlers are registered), and
``docs/adr/0026-mcp-resource-prompt-discovery.md`` (resources/prompts,
alongside tools).

Install with: ``pip install mcp``
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from requisite.agents.agent import Agent
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import Message
from requisite.mcp.base import MCPPromptArgument
from requisite.tools.base import Tool
from requisite.tools.registry import ToolLike, ToolRegistry

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server as MCPLowLevelServer
    from mcp.types import (
        CallToolRequestParams,
        CallToolResult,
        GetPromptRequestParams,
        GetPromptResult,
        ListPromptsResult,
        ListResourcesResult,
        ListToolsResult,
        ReadResourceRequestParams,
        ReadResourceResult,
    )

logger = logging.getLogger("requisite.mcp.server")

_DEFAULT_HTTP_HOST = "127.0.0.1"
_DEFAULT_HTTP_PORT = 8000
_DEFAULT_HTTP_PATH = "/mcp"


@dataclass
class _RegisteredResource:
    uri: str
    name: str
    description: str
    mime_type: Optional[str]
    content: Union[str, Callable[[], str]]


@dataclass
class _RegisteredPrompt:
    name: str
    description: str
    arguments: list[MCPPromptArgument]
    render: Callable[[dict[str, str]], list[Message]]


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
        self._resources: dict[str, _RegisteredResource] = {}
        self._prompts: dict[str, _RegisteredPrompt] = {}
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

    def add_resource(
        self,
        uri: str,
        *,
        name: str,
        content: Union[str, Callable[[], str]],
        description: str = "",
        mime_type: Optional[str] = None,
    ) -> None:
        """Register a resource under ``uri``.

        Parameters
        ----------
        content:
            Either static text, or a zero-argument callable producing the
            current text each time the resource is read (e.g. re-reading
            a file) -- so a resource's content can be dynamic.
        """
        self._resources[uri] = _RegisteredResource(
            uri=uri, name=name, description=description, mime_type=mime_type, content=content
        )

    def add_prompt(
        self,
        name: str,
        *,
        render: Callable[[dict[str, str]], list[Message]],
        description: str = "",
        arguments: Optional[list[MCPPromptArgument]] = None,
    ) -> None:
        """Register a prompt under ``name``.

        Parameters
        ----------
        render:
            Takes the filled-in arguments dict and returns the list of
            :class:`~requisite.core.interfaces.Message` this prompt
            expands to.
        """
        self._prompts[name] = _RegisteredPrompt(
            name=name, description=description, arguments=arguments or [], render=render
        )

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
            on_list_resources=self._handle_list_resources,
            on_read_resource=self._handle_read_resource,
            on_list_prompts=self._handle_list_prompts,
            on_get_prompt=self._handle_get_prompt,
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

    async def _handle_list_resources(self, ctx: Any, params: Any) -> "ListResourcesResult":
        from mcp.types import ListResourcesResult
        from mcp.types import Resource as MCPResourceType

        return ListResourcesResult(
            resources=[
                MCPResourceType(
                    uri=resource.uri,
                    name=resource.name,
                    description=resource.description,
                    mime_type=resource.mime_type,
                )
                for resource in self._resources.values()
            ]
        )

    async def _handle_read_resource(
        self, ctx: Any, params: "ReadResourceRequestParams"
    ) -> "ReadResourceResult":
        """Fetch a registered resource's content by URI.

        Unlike ``on_call_tool``, ``ReadResourceResult`` has no ``is_error``
        -like field to report a failure "successfully" -- there is no
        in-result error channel for this operation. Read
        ``mcp.server.runner``'s dispatch loop directly to confirm: any
        exception raised from a handler is already caught centrally and
        converted into a proper JSON-RPC error response, so an unknown
        URI is signaled by simply raising ``MCPError``, not by building a
        result -- see ``docs/adr/0026-mcp-resource-prompt-discovery.md``.
        """
        from mcp import MCPError
        from mcp.types import INVALID_PARAMS, ReadResourceResult, TextResourceContents

        resource = self._resources.get(params.uri)
        if resource is None:
            raise MCPError(code=INVALID_PARAMS, message=f"Unknown resource URI: {params.uri!r}")

        text = resource.content() if callable(resource.content) else resource.content
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=resource.uri, mime_type=resource.mime_type, text=text)
            ]
        )

    async def _handle_list_prompts(self, ctx: Any, params: Any) -> "ListPromptsResult":
        from mcp.types import ListPromptsResult
        from mcp.types import Prompt as MCPPromptType
        from mcp.types import PromptArgument as MCPPromptArgumentType

        return ListPromptsResult(
            prompts=[
                MCPPromptType(
                    name=prompt.name,
                    description=prompt.description,
                    arguments=[
                        MCPPromptArgumentType(
                            name=arg.name, description=arg.description, required=arg.required
                        )
                        for arg in prompt.arguments
                    ]
                    or None,
                )
                for prompt in self._prompts.values()
            ]
        )

    async def _handle_get_prompt(
        self, ctx: Any, params: "GetPromptRequestParams"
    ) -> "GetPromptResult":
        """Render a registered prompt into its messages.

        Same error-handling contract as :meth:`_handle_read_resource`:
        ``GetPromptResult`` has no ``is_error``-like field, so an unknown
        prompt name is signaled by raising ``MCPError``. A rendered
        :class:`~requisite.core.interfaces.Message` with a ``SYSTEM``/
        ``TOOL`` role isn't validated up front -- it fails naturally at
        ``PromptMessage`` construction below, since MCP's own role type is
        ``Literal["user", "assistant"]`` only (verified live). This is a
        known v1 limitation, not silently handled -- see
        ``docs/adr/0026-mcp-resource-prompt-discovery.md``.
        """
        from mcp import MCPError
        from mcp.types import INVALID_PARAMS, GetPromptResult, PromptMessage, TextContent

        prompt = self._prompts.get(params.name)
        if prompt is None:
            raise MCPError(code=INVALID_PARAMS, message=f"Unknown prompt: {params.name!r}")

        messages = prompt.render(params.arguments or {})
        return GetPromptResult(
            description=prompt.description,
            messages=[
                # Requisite's Role is broader (system/tool too) than MCP's
                # user/assistant-only role -- a SYSTEM/TOOL message fails
                # here at pydantic validation, not statically provable.
                PromptMessage(
                    role=message.role.value,  # type: ignore[arg-type]
                    content=TextContent(type="text", text=message.content),
                )
                for message in messages
            ],
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
