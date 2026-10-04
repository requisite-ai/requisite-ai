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

By default, every call reconnects (see :class:`MCPClient`'s docstring).
For repeated calls to the same server in a tight loop, an opt-in
persistent-session mode is available via :meth:`MCPClient.aconnect` /
:meth:`MCPClient.aclose`, or ``async with client:`` -- see
``docs/adr/0030-mcp-persistent-session-mode.md``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from requisite.core.exceptions import ConfigurationException, MCPException
from requisite.core.interfaces import Message, Role
from requisite.core.sync_bridge import is_bridge_loop, run_sync
from requisite.mcp.base import BaseMCPClient, MCPPrompt, MCPPromptArgument, MCPResource
from requisite.tools.base import Tool

if TYPE_CHECKING:
    from mcp import ClientSession as MCPClientSession

logger = logging.getLogger("requisite.mcp.client")

_T = TypeVar("_T")

# Shared by both transports -- bounds a single request/response round
# trip (including the initial `session.initialize()` handshake). Without
# it, a hung/misbehaving subprocess or server leaves a call blocked
# forever, with nothing in MCPClient's public API able to bound or
# cancel it. See docs/adr/0031-code-review-fixes.md.
_DEFAULT_TIMEOUT = 30.0


def _unwrap_exception(exc: BaseException) -> BaseException:
    """Unwrap nested (Base)ExceptionGroups down to the first real
    underlying exception.

    anyio's task groups (used internally by ``stdio_client``/
    ``ClientSession``) wrap a request-level exception -- e.g. a real
    protocol-level ``MCPError`` the server raised, like "unknown resource
    URI" -- in a ``BaseExceptionGroup`` during connection cleanup. Without
    this, ``str(exc)`` on the outer group is a useless generic
    "unhandled errors in a TaskGroup" instead of the actual error message
    -- caught live via a real round-trip test, not theoretical. Duck-typed
    on ``.exceptions`` rather than importing ``BaseExceptionGroup``
    directly, so this works whether it's the Python 3.11+ builtin or a
    backport.
    """
    current = exc
    while True:
        sub_exceptions = getattr(current, "exceptions", None)
        if not sub_exceptions:
            return current
        current = sub_exceptions[0]


class MCPClient(BaseMCPClient):
    """Connects to one MCP server over stdio or Streamable HTTP.

    Prefer the :meth:`stdio` / :meth:`http` factory methods over calling
    the constructor directly.

    Notes
    -----
    By default, each tool call (and each :meth:`discover_tools` call)
    opens a fresh connection, performs one request, and disconnects --
    there is no persistent session held open between calls. This mirrors
    the default behavior of other MCP client libraries (e.g. LangChain's
    ``MultiServerMCPClient``) and keeps the default path simple and easy
    to reason about, at the cost of reconnect latency on every call. See
    ``docs/adr/0004-mcp-integration.md`` for the reasoning.

    For repeated calls to the same server in a tight loop, where that
    per-call reconnect latency is measured to matter (real numbers: as
    much as ~1000x for stdio, ~15x for HTTP -- see
    ``docs/adr/0030-mcp-persistent-session-mode.md``), call
    :meth:`aconnect` once to open a session held open until
    :meth:`aclose`, or use ``async with client:``. Every async method
    (``adiscover_tools``, tool calls, ``adiscover_resources``,
    ``aread_resource``, ``adiscover_prompts``, ``aget_prompt``)
    transparently reuses the open session instead of reconnecting.
    From synchronous code, use :meth:`connect` / :meth:`close` or
    ``with client:`` instead: the session is opened on the shared sync-bridge
    loop, and the sync methods (:meth:`discover_tools`,
    :meth:`read_resource`, :meth:`get_prompt`, ...) and a discovered tool's
    :meth:`~requisite.tools.base.Tool.execute` reuse it. A session is bound
    to the event loop it was opened on: a sync method called while connected
    via :meth:`aconnect` (a user loop) raises
    :class:`~requisite.core.exceptions.ConfigurationException` immediately
    rather than risk a hang -- use the ``a``-prefixed methods there. See
    ``docs/adr/0041-agent-owned-persistent-mcp-sessions.md``; an
    :class:`~requisite.agents.agent.Agent` can also own the lifecycle via
    ``Agent(mcp_clients=[...])``.

    If the underlying connection dies on its own while persistent
    (subprocess crash, network drop -- anything other than an explicit
    :meth:`aclose`), this client does **not** auto-detect or
    auto-reconnect: every call through the dead session keeps failing
    with the same :class:`~requisite.core.exceptions.MCPException` until
    you notice and recover manually. To recover, call :meth:`aclose`
    (even though it may itself raise, surfacing the underlying
    connection's own teardown failure -- this is expected, not a bug in
    ``aclose`` itself) followed by a fresh :meth:`aconnect`:
    :meth:`aclose` always clears this client's internal connected-state
    *before* attempting the underlying teardown, so it leaves the client
    cleanly reconnectable even when that teardown fails. See
    ``docs/adr/0031-code-review-fixes.md``.

    Examples
    --------
    >>> client = MCPClient.stdio(name="filesystem", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])  # doctest: +SKIP
    >>> tools = client.discover_tools()  # doctest: +SKIP

    >>> client = MCPClient.http(name="github", url="https://api.example.com/mcp", headers={"Authorization": "Bearer ..."})  # doctest: +SKIP

    >>> async with MCPClient.stdio(name="fs", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as client:  # doctest: +SKIP
    ...     tools = await client.adiscover_tools()
    ...     result = await tools[0].aexecute(path="/tmp")
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
        timeout: float = _DEFAULT_TIMEOUT,
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

        self._persistent_session: Optional["MCPClientSession"] = None
        self._persistent_loop: Optional[asyncio.AbstractEventLoop] = None
        # One dedicated task owns the open connection for its whole life:
        # anyio cancel scopes (inside stdio_client/ClientSession) must be
        # entered and exited in the *same* task, and connect()/close() -- or
        # an Agent opening and later closing a client -- run in different ones.
        self._owner_task: Optional["asyncio.Task[None]"] = None
        self._stop_event: Optional[asyncio.Event] = None

    @classmethod
    def stdio(
        cls,
        *,
        name: str,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
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
        timeout:
            Per-request timeout, in seconds (including the initial
            connection handshake) -- bounds how long a call waits on a
            hung/misbehaving subprocess before raising, rather than
            blocking forever.
        """
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            timeout=timeout,
        )

    @classmethod
    def http(
        cls,
        *,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: float = _DEFAULT_TIMEOUT,
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
    async def _connect(self) -> AsyncIterator["MCPClientSession"]:
        """Open a fresh connection, yield a ready (``initialize()``d) session,
        then tear it down on exit. The per-call-reconnect default path
        (see :meth:`_session`) and :meth:`aconnect` both drive this."""
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
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=self._timeout
                ) as session:
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
                    async with ClientSession(
                        read_stream, write_stream, read_timeout_seconds=self._timeout
                    ) as session:
                        await session.initialize()
                        yield session

    @asynccontextmanager
    async def _session(self) -> AsyncIterator["MCPClientSession"]:
        """Yield a ready session for one call -- the open persistent
        session if :meth:`aconnect` has been called, otherwise a fresh
        per-call connection via :meth:`_connect` (the default path).

        Every async method in this class goes through this method rather
        than :meth:`_connect` directly, so persistent-session reuse is
        transparent to all of them.
        """
        if self._persistent_session is not None:
            if asyncio.get_running_loop() is not self._persistent_loop:
                raise ConfigurationException(
                    f"MCPClient '{self.name}' has a persistent session open on a "
                    "different event loop than the one currently running. Reusing "
                    "an MCP session across asyncio.run() boundaries silently "
                    "deadlocks the underlying SDK's transport cleanup -- this is "
                    "raised instead of hanging. Keep all use of a connected "
                    "MCPClient inside one continuous 'async with client:' block "
                    "(or one asyncio.run() call), and call aclose() before using "
                    "it from a different event loop.",
                )
            yield self._persistent_session
            return
        async with self._connect() as session:
            yield session

    async def aconnect(self) -> None:
        """Open a persistent session, held open until :meth:`aclose`.

        Every async method on this client transparently reuses it
        instead of reconnecting per call -- see the class docstring for
        the measured latency win. Prefer ``async with client:`` over
        calling this directly, since it guarantees a paired
        :meth:`aclose` even on an exception.

        Must be called from, and only used from, one continuous event
        loop until :meth:`aclose` -- crossing an ``asyncio.run()``
        boundary while connected is a silent-deadlock risk in the
        underlying SDK (verified live; see
        ``docs/adr/0030-mcp-persistent-session-mode.md``). Doing so is
        guarded against (see :meth:`_session`), turning the hang into a
        clean exception -- but the guard can only do that once execution
        reaches this client again on the wrong loop; it cannot un-strand
        a session left open on a loop that has already exited.

        Raises
        ------
        requisite.core.exceptions.ConfigurationException
            If already connected -- call :meth:`aclose` first.
        """
        if self._persistent_session is not None:
            raise ConfigurationException(
                f"MCPClient '{self.name}' is already connected. Call aclose() "
                "first, or avoid nested aconnect() calls.",
            )
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[MCPClientSession] = loop.create_future()
        stop = asyncio.Event()

        async def _own_connection() -> None:
            try:
                async with self._connect() as session:
                    ready.set_result(session)
                    await stop.wait()
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                    return
                raise

        task = loop.create_task(_own_connection())
        try:
            session = await ready
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        self._owner_task = task
        self._stop_event = stop
        self._persistent_session = session
        self._persistent_loop = loop

    async def aclose(self) -> None:
        """Close a persistent session opened by :meth:`aconnect`. No-op if
        not connected.

        Must be called from the same event loop :meth:`aconnect` was
        called from -- see :meth:`_session`'s cross-loop guard; the same
        constraint applies here, since closing also touches the
        loop-bound transport cleanup.

        Raises
        ------
        requisite.core.exceptions.ConfigurationException
            If called from a different event loop than the one this
            client connected on.
        """
        if self._persistent_session is None:
            return
        if asyncio.get_running_loop() is not self._persistent_loop:
            raise ConfigurationException(
                f"MCPClient '{self.name}' cannot be closed from a different "
                "event loop than the one it was connected on -- closing also "
                "touches the loop-bound transport cleanup that hangs across "
                "asyncio.run() boundaries (see aconnect()). The session may "
                "now be stranded on its original (dead) loop; if that loop "
                "already exited, its subprocess/connection cannot be closed "
                "cleanly from here.",
            )
        task, stop = self._owner_task, self._stop_event
        self._owner_task = self._stop_event = None
        self._persistent_session = None
        self._persistent_loop = None
        if task is not None and stop is not None:
            stop.set()
            await task

    async def __aenter__(self) -> "MCPClient":
        await self.aconnect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    def connect(self) -> None:
        """Open a persistent session usable from **synchronous** code.

        The session is opened on the shared sync-bridge loop
        (:func:`~requisite.core.sync_bridge.run_sync`), a single long-lived
        background loop, so it stays valid across any number of sync calls:
        :meth:`discover_tools`, the other sync methods, and a discovered
        tool's :meth:`~requisite.tools.base.Tool.execute` all reuse it
        instead of reconnecting. Pair with :meth:`close`, or use
        ``with client:``. A session opened this way is **not** usable from
        the async methods running on another loop -- use :meth:`aconnect`
        there. See ``docs/adr/0041-agent-owned-persistent-mcp-sessions.md``.
        """
        run_sync(self.aconnect())

    def close(self) -> None:
        """Close a session opened by :meth:`connect`. No-op if not connected."""
        if self._persistent_session is None:
            return
        if not is_bridge_loop(self._persistent_loop):
            raise ConfigurationException(
                f"MCPClient '{self.name}' was connected with aconnect() on its own "
                "event loop; close it with 'await client.aclose()' from that loop, "
                "not close().",
            )
        run_sync(self.aclose())

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _run_sync(
        self, method_name: str, make_coroutine: Callable[[], Coroutine[Any, Any, _T]]
    ) -> _T:
        """Run one async method from sync code.

        Not connected: a fresh loop per call (``asyncio.run``), as before.
        Connected via :meth:`connect` (session lives on the sync-bridge
        loop): run on that same loop so the session is reused. Connected via
        :meth:`aconnect` on some other loop: fail fast, before any new event
        loop is created, rather than risk a hang.
        """
        if self._persistent_session is None:
            return asyncio.run(make_coroutine())
        if is_bridge_loop(self._persistent_loop):
            return run_sync(make_coroutine())
        raise ConfigurationException(
            f"MCPClient '{self.name}' has a persistent session open on its own event "
            f"loop -- use 'a{method_name}' instead of '{method_name}' while connected "
            "via aconnect(), or open the session with connect() to use the sync "
            "methods. Synchronous methods cannot safely reuse a session opened on a "
            "different loop.",
        )

    async def adiscover_tools(self) -> list[Tool]:
        try:
            async with self._session() as session:
                result = await session.list_tools()
        except ConfigurationException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MCPException(
                f"Failed to discover tools from MCP server '{self.name}': {_unwrap_exception(exc)}",
                details={"server": self.name},
            ) from exc

        return [self._to_tool(mcp_tool) for mcp_tool in result.tools]

    def discover_tools(self) -> list[Tool]:
        return self._run_sync("discover_tools", self.adiscover_tools)

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
                f"Failed to call MCP tool '{tool_name}' on server '{self.name}': "
                f"{_unwrap_exception(exc)}",
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

    async def adiscover_resources(self) -> list[MCPResource]:
        try:
            async with self._session() as session:
                result = await session.list_resources()
        except ConfigurationException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MCPException(
                f"Failed to discover resources from MCP server '{self.name}': "
                f"{_unwrap_exception(exc)}",
                details={"server": self.name},
            ) from exc

        return [
            MCPResource(
                uri=resource.uri,
                name=resource.name,
                description=resource.description or "",
                mime_type=resource.mime_type,
            )
            for resource in result.resources
        ]

    def discover_resources(self) -> list[MCPResource]:
        return self._run_sync("discover_resources", self.adiscover_resources)

    async def aread_resource(self, uri: str) -> str:
        try:
            async with self._session() as session:
                result = await session.read_resource(uri)
        except ConfigurationException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MCPException(
                f"Failed to read resource '{uri}' from MCP server '{self.name}': "
                f"{_unwrap_exception(exc)}",
                details={"server": self.name, "uri": uri},
            ) from exc

        parts = [text for c in result.contents if (text := getattr(c, "text", None)) is not None]
        if not parts:
            raise MCPException(
                f"Resource '{uri}' on server '{self.name}' returned no text content "
                "(binary-only resources aren't supported yet).",
                details={"server": self.name, "uri": uri},
            )
        return "\n".join(parts)

    def read_resource(self, uri: str) -> str:
        return self._run_sync("read_resource", lambda: self.aread_resource(uri))

    async def adiscover_prompts(self) -> list[MCPPrompt]:
        try:
            async with self._session() as session:
                result = await session.list_prompts()
        except ConfigurationException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MCPException(
                f"Failed to discover prompts from MCP server '{self.name}': "
                f"{_unwrap_exception(exc)}",
                details={"server": self.name},
            ) from exc

        return [
            MCPPrompt(
                name=prompt.name,
                description=prompt.description or "",
                arguments=[
                    MCPPromptArgument(
                        name=arg.name,
                        description=arg.description or "",
                        required=bool(arg.required),
                    )
                    for arg in (prompt.arguments or [])
                ],
            )
            for prompt in result.prompts
        ]

    def discover_prompts(self) -> list[MCPPrompt]:
        return self._run_sync("discover_prompts", self.adiscover_prompts)

    async def aget_prompt(
        self, name: str, arguments: Optional[dict[str, str]] = None
    ) -> list[Message]:
        try:
            async with self._session() as session:
                result = await session.get_prompt(name, arguments)
        except ConfigurationException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MCPException(
                f"Failed to get prompt '{name}' from MCP server '{self.name}': "
                f"{_unwrap_exception(exc)}",
                details={"server": self.name, "prompt": name},
            ) from exc

        messages: list[Message] = []
        for prompt_message in result.messages:
            text = getattr(prompt_message.content, "text", None)
            if text is None:
                raise MCPException(
                    f"Prompt '{name}' on server '{self.name}' returned a non-text message "
                    "(not supported yet).",
                    details={"server": self.name, "prompt": name},
                )
            messages.append(Message(role=Role(prompt_message.role), content=text))
        return messages

    def get_prompt(self, name: str, arguments: Optional[dict[str, str]] = None) -> list[Message]:
        return self._run_sync("get_prompt", lambda: self.aget_prompt(name, arguments))
