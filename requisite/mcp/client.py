"""
MCP client implementation, wrapping the official ``mcp`` Python SDK.

Supports both current MCP transports (verified against ``mcp`` 2.0.0):
**stdio** (local subprocess, the default for local dev tools) and
**Streamable HTTP** (remote servers; replaced the now-deprecated SSE
transport in the November 2025 spec). Construct via :meth:`MCPClient.stdio`
or :meth:`MCPClient.http` rather than the plain constructor -- each takes
only the parameters relevant to that transport.

Install with: ``pip install mcp`` -- see
``docs/adr/0025-mcp-2x-migration.md`` for the 1.x -> 2.x migration this
module went through (a hard cutover, no dual-version support).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Optional

from requisite.core.exceptions import ConfigurationException, MCPException
from requisite.mcp.base import BaseMCPClient
from requisite.tools.base import Tool

if TYPE_CHECKING:
    from mcp import ClientSession as MCPClientSession

logger = logging.getLogger("requisite.mcp.client")

_DEFAULT_HTTP_TIMEOUT = 30.0


class MCPClient(BaseMCPClient):
    """Connects to one MCP server over stdio or Streamable HTTP.

    Prefer the :meth:`stdio` / :meth:`http` factory methods over calling
    the constructor directly.

    Notes
    -----
    Each tool call (and each :meth:`discover_tools` call) opens a fresh
    connection, performs one request, and disconnects -- there is no
    persistent session held open between calls. This mirrors the default
    behavior of other MCP client libraries (e.g. LangChain's
    ``MultiServerMCPClient``) and keeps this first implementation simple
    and easy to reason about, at the cost of reconnect latency on every
    call. See ``docs/adr/0004-mcp-integration.md`` for the reasoning and
    the plan for an optional persistent-session mode later.

    Examples
    --------
    >>> client = MCPClient.stdio(name="filesystem", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])  # doctest: +SKIP
    >>> tools = client.discover_tools()  # doctest: +SKIP

    >>> client = MCPClient.http(name="github", url="https://api.example.com/mcp", headers={"Authorization": "Bearer ..."})  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        name: str,
        transport: str,
        command: Optional[str] = None,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: float = _DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        if transport not in ("stdio", "http"):
            raise ConfigurationException(
                f"Unknown MCP transport '{transport}'. Use 'stdio' or 'http'."
            )
        if transport == "stdio" and not command:
            raise ConfigurationException("MCPClient(transport='stdio') requires 'command'.")
        if transport == "http" and not url:
            raise ConfigurationException("MCPClient(transport='http') requires 'url'.")

        self._name = name
        self._transport = transport
        self._command = command
        self._args = args or []
        self._env = env
        self._cwd = cwd
        self._url = url
        self._headers = headers
        self._timeout = timeout

    @classmethod
    def stdio(
        cls,
        *,
        name: str,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> "MCPClient":
        """Connect to a local MCP server run as a subprocess over stdio.

        Parameters
        ----------
        name:
            Short identifier for this connection (e.g. ``"filesystem"``).
        command:
            The executable to run (e.g. ``"npx"``, ``"python"``).
        args:
            Arguments to the command.
        env:
            Extra environment variables for the subprocess.
        cwd:
            Working directory for the subprocess.
        """
        return cls(name=name, transport="stdio", command=command, args=args, env=env, cwd=cwd)

    @classmethod
    def http(
        cls,
        *,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: float = _DEFAULT_HTTP_TIMEOUT,
    ) -> "MCPClient":
        """Connect to a remote MCP server over Streamable HTTP.

        Parameters
        ----------
        name:
            Short identifier for this connection (e.g. ``"github"``).
        url:
            The server's MCP endpoint URL.
        headers:
            Extra HTTP headers (e.g. ``{"Authorization": "Bearer ..."}``).
        timeout:
            Per-request timeout, in seconds.
        """
        return cls(name=name, transport="http", url=url, headers=headers, timeout=timeout)

    @property
    def name(self) -> str:
        return self._name

    @asynccontextmanager
    async def _session(self) -> AsyncIterator["MCPClientSession"]:
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'mcp' package is required for MCPClient. Install it with: pip install mcp",
            ) from exc

        if self._transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self._command or "",
                args=self._args,
                env=self._env,
                cwd=self._cwd,
            )
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
        else:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            # streamable_http_client no longer takes headers=/timeout= directly
            # (mcp 2.x) -- its own docstring says to build an httpx2.AsyncClient
            # and pass it in. It only manages that client's lifecycle itself
            # when it constructs one internally ("only manage client lifecycle
            # if we created it") -- since we're passing our own, we open and
            # close it ourselves here.
            async with httpx2.AsyncClient(
                headers=self._headers, timeout=self._timeout
            ) as http_client:
                async with streamable_http_client(self._url or "", http_client=http_client) as (
                    read_stream,
                    write_stream,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session

    async def adiscover_tools(self) -> list[Tool]:
        try:
            async with self._session() as session:
                result = await session.list_tools()
        except ConfigurationException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MCPException(
                f"Failed to discover tools from MCP server '{self.name}': {exc}",
                details={"server": self.name},
            ) from exc

        return [self._to_tool(mcp_tool) for mcp_tool in result.tools]

    def discover_tools(self) -> list[Tool]:
        return asyncio.run(self.adiscover_tools())

    def _to_tool(self, mcp_tool: Any) -> Tool:
        """Wrap one MCP-discovered tool as a :class:`Tool` whose ``execute``
        proxies the call back to this server.

        The wrapper function is defined ``async`` deliberately: ``Tool.execute``
        already knows how to run an async function via ``asyncio.run`` when
        called synchronously, and ``Tool.aexecute`` awaits it directly --
        writing one async implementation covers both call styles for free.
        """
        tool_name = mcp_tool.name

        async def _call(**kwargs: Any) -> Any:
            return await self._call_tool(tool_name, kwargs)

        _call.__name__ = tool_name

        return Tool(
            name=tool_name,
            description=mcp_tool.description or "",
            parameters_schema=mcp_tool.input_schema or {"type": "object", "properties": {}},
            func=_call,
        )

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            async with self._session() as session:
                result = await session.call_tool(tool_name, arguments)
        except ConfigurationException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MCPException(
                f"Failed to call MCP tool '{tool_name}' on server '{self.name}': {exc}",
                details={"server": self.name, "tool": tool_name},
            ) from exc

        if result.is_error:
            raise MCPException(
                f"MCP tool '{tool_name}' on server '{self.name}' returned an error: "
                f"{self._extract_text(result)}",
                details={"server": self.name, "tool": tool_name},
            )

        if result.structured_content is not None:
            return result.structured_content
        return self._extract_text(result)

    @staticmethod
    def _extract_text(result: Any) -> str:
        parts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        return "\n".join(parts)
