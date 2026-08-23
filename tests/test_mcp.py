"""Unit tests for requisite.mcp.

The ``mcp`` SDK's session/transport objects are faked so these tests
never spawn a real subprocess or make a real network call -- consistent
with the framework's no-network-in-tests rule. (A real end-to-end check
against actual ``mcp`` stdio and Streamable HTTP servers was done
manually during development -- see ADR-0004 (client) and ADR-0015
(server) -- but that's not something a fast, deterministic CI suite
should depend on.)

``MCPServer`` tests take a different, simpler approach than the client
tests below: rather than faking the SDK, they call
``MCPServer._handle_list_tools``/``_handle_call_tool`` directly -- these
are plain async methods, passed as constructor kwargs to the real
``mcp.server.lowlevel.Server`` in ``_build_server`` (mcp 2.x's handler
registration -- see ``docs/adr/0025-mcp-2x-migration.md``), but never
dependent on it themselves. ``ctx`` is unused by either handler, so
tests pass ``None``; ``params`` only needs a duck-typed stand-in
(``SimpleNamespace``) carrying the couple of attributes each handler
actually reads -- no real SDK request object needs faking.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from requisite.agents.agent import Agent
from requisite.capabilities.registry import CapabilityRegistry
from requisite.capabilities.resolvers import register_default_capabilities
from requisite.core.exceptions import ConfigurationException, MCPException, ToolException
from requisite.core.interfaces import ChatResponse, Message, Role, Usage
from requisite.mcp.base import MCPPromptArgument
from requisite.mcp.client import MCPClient
from requisite.mcp.defaults import register_github_mcp_capability, register_mcp_capability
from requisite.mcp.registry import MCPClientRegistry
from requisite.mcp.server import MCPServer
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry
from requisite.tools.base import Tool
from requisite.tools.decorator import tool


class _FakeMCPTool:
    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema


class _FakeListToolsResult:
    def __init__(self, tools: list[_FakeMCPTool]) -> None:
        self.tools = tools


class _FakeTextContent:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeCallToolResult:
    def __init__(self, *, text: str = "", structured: Any = None, is_error: bool = False) -> None:
        self.content = [_FakeTextContent(text)] if text else []
        self.structured_content = structured
        self.is_error = is_error


class _FakeResource:
    def __init__(
        self, uri: str, name: str, description: str = "", mime_type: str | None = None
    ) -> None:
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type


class _FakeListResourcesResult:
    def __init__(self, resources: list[_FakeResource]) -> None:
        self.resources = resources


class _FakeResourceContents:
    """A fake TextResourceContents/BlobResourceContents -- only carries
    whichever of .text/.blob is given, matching how the real SDK's union
    has no shared discriminator field."""

    def __init__(
        self,
        *,
        text: str | None = None,
        blob: str | None = None,
        uri: str = "",
        mime_type: str | None = None,
    ) -> None:
        self.uri = uri
        self.mime_type = mime_type
        if text is not None:
            self.text = text
        if blob is not None:
            self.blob = blob


class _FakeReadResourceResult:
    def __init__(self, contents: list[_FakeResourceContents]) -> None:
        self.contents = contents


class _FakePromptArgument:
    def __init__(self, name: str, description: str = "", required: bool = False) -> None:
        self.name = name
        self.description = description
        self.required = required


class _FakePrompt:
    def __init__(
        self, name: str, description: str = "", arguments: list[_FakePromptArgument] | None = None
    ) -> None:
        self.name = name
        self.description = description
        self.arguments = arguments or []


class _FakeListPromptsResult:
    def __init__(self, prompts: list[_FakePrompt]) -> None:
        self.prompts = prompts


class _FakePromptMessageContent:
    """A fake ContentBlock -- only carries .text when it represents text
    content, matching how a real non-text block has no .text attribute."""

    def __init__(self, *, text: str | None = None) -> None:
        if text is not None:
            self.text = text


class _FakePromptMessage:
    def __init__(self, role: str, content: _FakePromptMessageContent) -> None:
        self.role = role
        self.content = content


class _FakeGetPromptResult:
    def __init__(self, messages: list[_FakePromptMessage], description: str = "") -> None:
        self.messages = messages
        self.description = description


class _FakeSession:
    """Fakes mcp.ClientSession's async API for the calls MCPClient makes."""

    def __init__(
        self,
        tools: list[_FakeMCPTool],
        call_results: dict[str, _FakeCallToolResult],
        *,
        resources: list[_FakeResource] | None = None,
        resource_contents: dict[str, _FakeReadResourceResult] | None = None,
        prompts: list[_FakePrompt] | None = None,
        prompt_results: dict[str, _FakeGetPromptResult] | None = None,
    ) -> None:
        self._tools = tools
        self._call_results = call_results
        self._resources = resources or []
        self._resource_contents = resource_contents or {}
        self._prompts = prompts or []
        self._prompt_results = prompt_results or {}
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> _FakeListToolsResult:
        return _FakeListToolsResult(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeCallToolResult:
        if name not in self._call_results:
            raise KeyError(f"no fake result configured for tool '{name}'")
        return self._call_results[name]

    async def list_resources(self) -> _FakeListResourcesResult:
        return _FakeListResourcesResult(self._resources)

    async def read_resource(self, uri: str) -> _FakeReadResourceResult:
        if uri not in self._resource_contents:
            raise KeyError(f"no fake resource contents configured for '{uri}'")
        return self._resource_contents[uri]

    async def list_prompts(self) -> _FakeListPromptsResult:
        return _FakeListPromptsResult(self._prompts)

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> _FakeGetPromptResult:
        if name not in self._prompt_results:
            raise KeyError(f"no fake prompt result configured for '{name}'")
        return self._prompt_results[name]


def _patch_session(
    monkeypatch: pytest.MonkeyPatch, client: MCPClient, fake_session: _FakeSession
) -> None:
    """Replace MCPClient._session with a fake async context manager yielding fake_session."""

    @asynccontextmanager
    async def _fake_session_cm():
        yield fake_session

    monkeypatch.setattr(client, "_session", _fake_session_cm)


def _patch_connect(
    monkeypatch: pytest.MonkeyPatch,
    client: MCPClient,
    fake_session: _FakeSession,
    *,
    track_calls: list[int] | None = None,
) -> None:
    """Replace MCPClient._connect with a fake async context manager yielding
    fake_session, for testing the real _session()/aconnect()/aclose() reuse
    and cross-loop guard logic -- unlike _patch_session, which replaces
    _session directly and bypasses that logic entirely."""

    @asynccontextmanager
    async def _fake_connect():
        if track_calls is not None:
            track_calls.append(1)
        yield fake_session

    monkeypatch.setattr(client, "_connect", _fake_connect)


# ---------------------------------------------------------------------------
# MCPClient construction
# ---------------------------------------------------------------------------


def test_stdio_factory_builds_client() -> None:
    client = MCPClient.stdio(name="fs", command="npx", args=["-y", "server"])
    assert client.name == "fs"


def test_http_factory_builds_client() -> None:
    client = MCPClient.http(name="gh", url="https://example.com/mcp")
    assert client.name == "gh"


def test_stdio_requires_command() -> None:
    with pytest.raises(ConfigurationException):
        MCPClient(name="fs", transport="stdio")


def test_http_requires_url() -> None:
    with pytest.raises(ConfigurationException):
        MCPClient(name="gh", transport="http")


def test_unknown_transport_raises() -> None:
    with pytest.raises(ConfigurationException):
        MCPClient(name="x", transport="carrier-pigeon")


# ---------------------------------------------------------------------------
# discover_tools / adiscover_tools
# ---------------------------------------------------------------------------


def test_discover_tools_returns_requisite_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [
        _FakeMCPTool(
            "add", "Add two numbers.", {"type": "object", "properties": {"a": {"type": "integer"}}}
        ),
    ]
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, {}))

    tools = client.discover_tools()
    assert len(tools) == 1
    assert tools[0].name == "add"
    assert tools[0].description == "Add two numbers."
    assert tools[0].parameters_schema == fake_tools[0].input_schema


@pytest.mark.asyncio
async def test_adiscover_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [_FakeMCPTool("noop", "", {"type": "object", "properties": {}})]
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, {}))

    tools = await client.adiscover_tools()
    assert [t.name for t in tools] == ["noop"]


# ---------------------------------------------------------------------------
# Tool execution proxies back over MCP
# ---------------------------------------------------------------------------


def test_discovered_tool_execute_returns_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [_FakeMCPTool("add", "", {"type": "object", "properties": {}})]
    fake_results = {"add": _FakeCallToolResult(structured={"result": 7})}
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, fake_results))

    tools = client.discover_tools()
    result = tools[0].execute(a=3, b=4)
    assert result == {"result": 7}


def test_discovered_tool_execute_falls_back_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [_FakeMCPTool("greet", "", {"type": "object", "properties": {}})]
    fake_results = {"greet": _FakeCallToolResult(text="hello there")}
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, fake_results))

    tools = client.discover_tools()
    result = tools[0].execute()
    assert result == "hello there"


def test_discovered_tool_execute_raises_on_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [_FakeMCPTool("boom", "", {"type": "object", "properties": {}})]
    fake_results = {"boom": _FakeCallToolResult(text="something broke", is_error=True)}
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, fake_results))

    tools = client.discover_tools()
    with pytest.raises(Exception) as excinfo:  # ToolException wraps the MCPException raised inside
        tools[0].execute()
    assert "something broke" in str(excinfo.value)


@pytest.mark.asyncio
async def test_discovered_tool_aexecute(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [_FakeMCPTool("add", "", {"type": "object", "properties": {}})]
    fake_results = {"add": _FakeCallToolResult(structured={"result": 9})}
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, fake_results))

    tools = await client.adiscover_tools()
    result = await tools[0].aexecute(a=4, b=5)
    assert result == {"result": 9}


def test_missing_sdk_raises_configuration_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "mcp", None)
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    with pytest.raises(ConfigurationException):
        client.discover_tools()


# ---------------------------------------------------------------------------
# resource / prompt discovery (client side)
# ---------------------------------------------------------------------------


def test_discover_resources_returns_mcp_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_resources = [_FakeResource("file:///a.txt", "a.txt", "First file", "text/plain")]
    _patch_session(monkeypatch, client, _FakeSession([], {}, resources=fake_resources))

    resources = client.discover_resources()
    assert len(resources) == 1
    assert resources[0].uri == "file:///a.txt"
    assert resources[0].name == "a.txt"
    assert resources[0].description == "First file"
    assert resources[0].mime_type == "text/plain"


def test_read_resource_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    contents = {
        "file:///a.txt": _FakeReadResourceResult([_FakeResourceContents(text="hello world")])
    }
    _patch_session(monkeypatch, client, _FakeSession([], {}, resource_contents=contents))

    assert client.read_resource("file:///a.txt") == "hello world"


def test_read_resource_binary_only_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    contents = {"file:///a.bin": _FakeReadResourceResult([_FakeResourceContents(blob="ZmFrZQ==")])}
    _patch_session(monkeypatch, client, _FakeSession([], {}, resource_contents=contents))

    with pytest.raises(MCPException, match="no text content"):
        client.read_resource("file:///a.bin")


def test_discover_prompts_returns_mcp_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_prompts = [
        _FakePrompt(
            "greet", "Greet someone", [_FakePromptArgument("name", "Who to greet", required=True)]
        )
    ]
    _patch_session(monkeypatch, client, _FakeSession([], {}, prompts=fake_prompts))

    prompts = client.discover_prompts()
    assert len(prompts) == 1
    assert prompts[0].name == "greet"
    assert prompts[0].description == "Greet someone"
    assert len(prompts[0].arguments) == 1
    assert prompts[0].arguments[0].name == "name"
    assert prompts[0].arguments[0].required is True


def test_get_prompt_returns_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    prompt_results = {
        "greet": _FakeGetPromptResult(
            [_FakePromptMessage("user", _FakePromptMessageContent(text="Hello, Ada!"))]
        )
    }
    _patch_session(monkeypatch, client, _FakeSession([], {}, prompt_results=prompt_results))

    messages = client.get_prompt("greet", {"name": "Ada"})
    assert messages == [Message(role=Role.USER, content="Hello, Ada!")]


def test_get_prompt_non_text_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    prompt_results = {
        "greet": _FakeGetPromptResult([_FakePromptMessage("user", _FakePromptMessageContent())])
    }
    _patch_session(monkeypatch, client, _FakeSession([], {}, prompt_results=prompt_results))

    with pytest.raises(MCPException, match="non-text message"):
        client.get_prompt("greet")


# ---------------------------------------------------------------------------
# Persistent session mode (aconnect/aclose/async with)
# ---------------------------------------------------------------------------


def test_aconnect_reuses_session_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """_connect is invoked only once even though two calls go through
    _session() -- proof the persistent session is genuinely reused, not
    reconnected per call."""
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    call_results = {"add": _FakeCallToolResult(structured={"result": 3})}
    fake_session = _FakeSession([_FakeMCPTool("add", "Add", {})], call_results)
    track_calls: list[int] = []
    _patch_connect(monkeypatch, client, fake_session, track_calls=track_calls)

    async def scenario() -> None:
        await client.aconnect()
        tools1 = await client.adiscover_tools()
        tools2 = await client.adiscover_tools()
        assert tools1[0].name == tools2[0].name == "add"
        await client.aclose()

    asyncio.run(scenario())
    assert track_calls == [1]


def test_aconnect_twice_raises_configuration_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    _patch_connect(monkeypatch, client, _FakeSession([], {}))

    async def scenario() -> None:
        await client.aconnect()
        with pytest.raises(ConfigurationException, match="already connected"):
            await client.aconnect()
        await client.aclose()

    asyncio.run(scenario())


def test_aclose_when_not_connected_is_noop() -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])

    async def scenario() -> None:
        await client.aclose()

    asyncio.run(scenario())


def test_async_context_manager_connects_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    _patch_connect(monkeypatch, client, _FakeSession([], {}))

    async def scenario() -> None:
        assert client._persistent_session is None
        async with client as ctx:
            assert ctx is client
            assert client._persistent_session is not None
        assert client._persistent_session is None

    asyncio.run(scenario())


def test_async_context_manager_closes_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    _patch_connect(monkeypatch, client, _FakeSession([], {}))

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with client:
                assert client._persistent_session is not None
                raise RuntimeError("boom")
        assert client._persistent_session is None

    asyncio.run(scenario())


def test_cross_loop_reuse_raises_configuration_exception_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key regression test for the verified deadlock: reusing a
    persistent session from a different event loop than the one it was
    opened on must raise cleanly, not hang. _connect is faked (no real
    subprocess/anyio task groups), so this cannot actually deadlock --
    it specifically tests the guard logic, not the underlying SDK."""
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    _patch_connect(monkeypatch, client, _FakeSession([], {}))

    asyncio.run(client.aconnect())

    with pytest.raises(ConfigurationException, match="different event loop"):
        asyncio.run(client.adiscover_tools())

    with pytest.raises(ConfigurationException, match="different event loop"):
        asyncio.run(client.aclose())


def test_sync_methods_raise_when_persistent_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync methods must fail fast -- before ever calling asyncio.run --
    when a persistent session is open, rather than risk a hang."""
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    _patch_connect(monkeypatch, client, _FakeSession([], {}))
    asyncio.run(client.aconnect())

    def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("asyncio.run should not be called by a rejected sync method")

    monkeypatch.setattr(asyncio, "run", _fail_if_called)

    with pytest.raises(ConfigurationException, match="adiscover_tools"):
        client.discover_tools()
    with pytest.raises(ConfigurationException, match="adiscover_resources"):
        client.discover_resources()
    with pytest.raises(ConfigurationException, match="aread_resource"):
        client.read_resource("memo://x")
    with pytest.raises(ConfigurationException, match="adiscover_prompts"):
        client.discover_prompts()
    with pytest.raises(ConfigurationException, match="aget_prompt"):
        client.get_prompt("greet")


def test_discovered_tool_execute_wraps_configuration_exception_when_persistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discovered tool's sync execute() (not just the 5 discover_*/
    read_*/get_* wrappers) also hits the _session() cross-loop guard when
    persistent mode is connected -- Tool.execute()'s existing, unmodified
    exception handling wraps it as ToolException, the same as any other
    exception from a tool's underlying function."""
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    mcp_tool = _FakeMCPTool("add", "Add two numbers", {"type": "object", "properties": {}})
    fake_session = _FakeSession([mcp_tool], {"add": _FakeCallToolResult(structured={"result": 3})})
    _patch_connect(monkeypatch, client, fake_session)
    requisite_tool = client._to_tool(mcp_tool)

    asyncio.run(client.aconnect())

    with pytest.raises(ToolException) as exc_info:
        requisite_tool.execute(a=1, b=2)

    assert isinstance(exc_info.value.__cause__, ConfigurationException)


# ---------------------------------------------------------------------------
# register_as_capability bridge
# ---------------------------------------------------------------------------


def test_register_as_capability_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [
        _FakeMCPTool("get_weather", "Weather lookup.", {"type": "object", "properties": {}})
    ]
    fake_results = {"get_weather": _FakeCallToolResult(structured={"result": "sunny"})}
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, fake_results))

    registry = CapabilityRegistry()
    client.register_as_capability(registry, capability="get_weather", priority=5)

    resolved = registry.resolve("get_weather")
    assert resolved.execute() == {"result": "sunny"}
    providers = registry.providers_for("get_weather")
    assert providers[0].provider_name == "mcp:fs"
    assert providers[0].priority == 5


def test_register_as_capability_missing_tool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    _patch_session(monkeypatch, client, _FakeSession([], {}))

    registry = CapabilityRegistry()
    with pytest.raises(MCPException, match="does not expose a tool"):
        client.register_as_capability(registry, capability="does_not_exist")


# ---------------------------------------------------------------------------
# requisite.mcp.defaults -- register_mcp_capability
# ---------------------------------------------------------------------------


def test_register_mcp_capability_renames_tool_to_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [
        _FakeMCPTool("search_repositories", "Search repos.", {"type": "object", "properties": {}})
    ]
    fake_results = {"search_repositories": _FakeCallToolResult(structured={"result": "ok"})}
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, fake_results))

    registry = CapabilityRegistry()
    result = register_mcp_capability(
        registry, client, tool_name="search_repositories", capability="github", priority=10
    )

    assert result is True
    resolved = registry.resolve("github")
    assert resolved.name == "github"
    assert resolved.execute() == {"result": "ok"}
    providers = registry.providers_for("github")
    assert providers[0].provider_name == "mcp:fs"
    assert providers[0].priority == 10


def test_register_mcp_capability_no_rename_when_names_already_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    fake_tools = [_FakeMCPTool("database", "", {"type": "object", "properties": {}})]
    _patch_session(monkeypatch, client, _FakeSession(fake_tools, {}))

    registry = CapabilityRegistry()
    assert (
        register_mcp_capability(registry, client, tool_name="database", capability="database")
        is True
    )
    assert registry.resolve("database").name == "database"


def test_register_mcp_capability_missing_tool_returns_false_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    _patch_session(monkeypatch, client, _FakeSession([], {}))

    registry = CapabilityRegistry()
    result = register_mcp_capability(registry, client, tool_name="nope", capability="github")

    assert result is False
    assert "github" not in registry


def test_register_mcp_capability_discovery_failure_returns_false_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])

    def _boom() -> list[Tool]:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(client, "discover_tools", _boom)

    registry = CapabilityRegistry()
    result = register_mcp_capability(registry, client, tool_name="x", capability="github")

    assert result is False
    assert "github" not in registry


# ---------------------------------------------------------------------------
# requisite.mcp.defaults -- register_github_mcp_capability
# ---------------------------------------------------------------------------


def test_register_github_mcp_capability_noop_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("should not construct an MCP client without a token")

    monkeypatch.setattr(MCPClient, "http", classmethod(_fail))

    registry = CapabilityRegistry()
    result = register_github_mcp_capability(registry)

    assert result is False
    assert "github" not in registry


def test_register_github_mcp_capability_success_outranks_rest_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    def _fake_discover_tools(self: MCPClient) -> list[Tool]:
        return [
            Tool(
                name="search_repositories",
                description="Search GitHub repositories.",
                parameters_schema={"type": "object", "properties": {}},
                func=lambda **kwargs: "mcp result",
            )
        ]

    monkeypatch.setattr(MCPClient, "discover_tools", _fake_discover_tools)

    registry = CapabilityRegistry()
    register_default_capabilities(registry)  # seeds the priority-0 REST resolver

    assert register_github_mcp_capability(registry) is True

    resolved = registry.resolve("github")
    assert resolved.name == "github"
    assert resolved.execute() == "mcp result"

    providers = registry.providers_for("github")
    assert [p.provider_name for p in providers] == ["mcp:github", "github-rest-api"]
    assert providers[0].priority == 10
    assert providers[1].priority == 0


# ---------------------------------------------------------------------------
# MCPClientRegistry
# ---------------------------------------------------------------------------


def test_mcp_client_registry_register_and_get() -> None:
    registry = MCPClientRegistry()
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    registry.register(client)
    assert registry.get("fs") is client
    assert "fs" in registry
    assert len(registry) == 1
    assert registry.list_names() == ["fs"]
    assert registry.all() == [client]


def test_mcp_client_registry_get_missing_raises() -> None:
    registry = MCPClientRegistry()
    with pytest.raises(MCPException):
        registry.get("nope")


def test_mcp_client_registry_unregister_is_idempotent() -> None:
    registry = MCPClientRegistry()
    client = MCPClient.stdio(name="fs", command="python", args=["server.py"])
    registry.register(client)
    registry.unregister("fs")
    registry.unregister("fs")  # no error
    assert "fs" not in registry


# ---------------------------------------------------------------------------
# MCPServer -- the reverse direction (exposing Requisite as an MCP server)
# ---------------------------------------------------------------------------


class _FixedAnswerProvider(BaseProvider):
    """A fake provider that always returns one fixed final answer, no tool calls."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))

    @property
    def name(self) -> str:
        return "fixed"

    def chat(self, messages, *, model=None, temperature=None, tools=None, **kwargs) -> ChatResponse:
        last = messages[-1].content if messages else ""
        return ChatResponse(
            content=f"agent answer: {last}",
            model=self._model,
            provider=self.name,
            usage=Usage(total_tokens=1),
        )

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, model=model, temperature=temperature, tools=tools, **kwargs)

    def stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


def _fixed_answer_agent(name: str = "assistant") -> Agent:
    registry = ProviderRegistry()
    registry.register("fixed", _FixedAnswerProvider)
    return Agent(name=name, provider="fixed", registry=registry)


def test_mcp_server_construction_registers_tools_and_agents() -> None:
    server = MCPServer(name="demo", tools=[add], agents=[_fixed_answer_agent()])
    assert server.name == "demo"
    assert {t.name for t in server._tool_registry.all()} == {"add", "assistant"}


def _call_tool_params(name: str, arguments: dict[str, Any]) -> Any:
    """Minimal stand-in for mcp.types.CallToolRequestParams -- _handle_call_tool
    only reads .name/.arguments, so a real SDK object isn't needed."""
    return SimpleNamespace(name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_handle_list_tools_returns_mcp_tools() -> None:
    server = MCPServer(name="demo", tools=[add])

    result = await server._handle_list_tools(None, None)
    assert len(result.tools) == 1
    assert result.tools[0].name == "add"
    assert result.tools[0].description == "Add two numbers."
    assert result.tools[0].input_schema["required"] == ["a", "b"]


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_non_dict_result() -> None:
    server = MCPServer(name="demo", tools=[add])

    result = await server._handle_call_tool(None, _call_tool_params("add", {"a": 2, "b": 3}))
    assert result.structured_content == {"result": 5}
    assert result.is_error is False


@pytest.mark.asyncio
async def test_handle_call_tool_passes_dict_result_through() -> None:
    @tool
    def get_status() -> dict:
        """Return a structured status."""
        return {"ok": True, "count": 3}

    server = MCPServer(name="demo", tools=[get_status])

    result = await server._handle_call_tool(None, _call_tool_params("get_status", {}))
    assert result.structured_content == {"ok": True, "count": 3}


@pytest.mark.asyncio
async def test_handle_call_tool_unknown_tool_returns_error_result() -> None:
    """mcp 2.x's on_call_tool handler must build the result itself -- unlike
    1.x's decorator wrapper, there's no automatic exception-to-is_error
    conversion, so MCPServer now catches this itself (see the docstring
    on _handle_call_tool) rather than letting it propagate."""
    server = MCPServer(name="demo", tools=[add])

    result = await server._handle_call_tool(None, _call_tool_params("does_not_exist", {}))
    assert result.is_error is True


@pytest.mark.asyncio
async def test_handle_call_tool_failure_returns_error_result() -> None:
    server = MCPServer(name="demo", tools=[boom])

    result = await server._handle_call_tool(None, _call_tool_params("boom", {}))
    assert result.is_error is True
    assert "kaboom" in result.content[0].text


@pytest.mark.asyncio
async def test_handle_call_tool_agent_round_trip() -> None:
    agent = _fixed_answer_agent(name="assistant")
    server = MCPServer(name="demo", agents=[agent])

    result = await server._handle_call_tool(
        None, _call_tool_params("assistant", {"prompt": "hello"})
    )
    assert result.structured_content == {"result": "agent answer: hello"}


# ---------------------------------------------------------------------------
# resource / prompt discovery (server side)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_list_resources_returns_registered_resources() -> None:
    server = MCPServer(name="demo")
    server.add_resource(
        "file:///a.txt", name="a.txt", content="hello", description="A file", mime_type="text/plain"
    )

    result = await server._handle_list_resources(None, None)
    assert len(result.resources) == 1
    assert result.resources[0].uri == "file:///a.txt"
    assert result.resources[0].name == "a.txt"
    assert result.resources[0].mime_type == "text/plain"


@pytest.mark.asyncio
async def test_handle_read_resource_returns_static_content() -> None:
    server = MCPServer(name="demo")
    server.add_resource("file:///a.txt", name="a.txt", content="hello")

    result = await server._handle_read_resource(None, SimpleNamespace(uri="file:///a.txt"))
    assert result.contents[0].text == "hello"


@pytest.mark.asyncio
async def test_handle_read_resource_calls_dynamic_content() -> None:
    server = MCPServer(name="demo")
    server.add_resource("file:///counter", name="counter", content=lambda: "dynamic value")

    result = await server._handle_read_resource(None, SimpleNamespace(uri="file:///counter"))
    assert result.contents[0].text == "dynamic value"


@pytest.mark.asyncio
async def test_handle_read_resource_unknown_uri_raises_mcp_error() -> None:
    from mcp import MCPError

    server = MCPServer(name="demo")

    with pytest.raises(MCPError, match="Unknown resource URI"):
        await server._handle_read_resource(None, SimpleNamespace(uri="file:///missing"))


@pytest.mark.asyncio
async def test_handle_list_prompts_returns_registered_prompts() -> None:
    server = MCPServer(name="demo")
    server.add_prompt(
        "greet",
        render=lambda args: [Message(role=Role.USER, content=f"Hello, {args['name']}!")],
        description="Greet someone",
        arguments=[MCPPromptArgument(name="name", description="Who to greet", required=True)],
    )

    result = await server._handle_list_prompts(None, None)
    assert len(result.prompts) == 1
    assert result.prompts[0].name == "greet"
    assert result.prompts[0].description == "Greet someone"
    assert result.prompts[0].arguments[0].name == "name"


@pytest.mark.asyncio
async def test_handle_get_prompt_renders_messages() -> None:
    server = MCPServer(name="demo")
    server.add_prompt(
        "greet", render=lambda args: [Message(role=Role.USER, content=f"Hello, {args['name']}!")]
    )

    result = await server._handle_get_prompt(
        None, SimpleNamespace(name="greet", arguments={"name": "Ada"})
    )
    assert len(result.messages) == 1
    assert result.messages[0].role == "user"
    assert result.messages[0].content.text == "Hello, Ada!"


@pytest.mark.asyncio
async def test_handle_get_prompt_unknown_name_raises_mcp_error() -> None:
    from mcp import MCPError

    server = MCPServer(name="demo")

    with pytest.raises(MCPError, match="Unknown prompt"):
        await server._handle_get_prompt(None, SimpleNamespace(name="missing", arguments=None))


def test_add_tool_and_add_agent_return_registered_tool() -> None:
    server = MCPServer(name="demo")

    registered = server.add_tool(add)
    assert registered.name == "add"

    registered_agent_tool = server.add_agent(_fixed_answer_agent())
    assert registered_agent_tool.name == "assistant"
    assert {t.name for t in server._tool_registry.all()} == {"add", "assistant"}


def test_mcp_server_missing_sdk_raises_configuration_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    # Earlier tests in this module already import mcp.server.lowlevel for
    # real, caching it in sys.modules -- setting sys.modules["mcp"] = None
    # alone wouldn't stop `from mcp.server.lowlevel import Server` from
    # resolving straight from that cache, so also drop every already-cached
    # mcp submodule to genuinely simulate the package being uninstalled.
    for module_name in list(sys.modules):
        if module_name == "mcp" or module_name.startswith("mcp."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setitem(sys.modules, "mcp", None)

    server = MCPServer(name="demo", tools=[add])
    with pytest.raises(ConfigurationException):
        server._build_server()
