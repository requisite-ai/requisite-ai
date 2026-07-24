"""Unit tests for requisite.mcp.

The ``mcp`` SDK's session/transport objects are faked so these tests
never spawn a real subprocess or make a real network call -- consistent
with the framework's no-network-in-tests rule. (A real end-to-end check
against actual ``mcp`` stdio and Streamable HTTP servers was done
manually during development -- see ADR-0004 -- but that's not something
a fast, deterministic CI suite should depend on.)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from requisite.capabilities.registry import CapabilityRegistry
from requisite.core.exceptions import ConfigurationException, MCPException
from requisite.mcp.client import MCPClient
from requisite.mcp.registry import MCPClientRegistry


class _FakeMCPTool:
    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema


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
        self.structuredContent = structured
        self.isError = is_error


class _FakeSession:
    """Fakes mcp.ClientSession's async API for the calls MCPClient makes."""

    def __init__(
        self, tools: list[_FakeMCPTool], call_results: dict[str, _FakeCallToolResult]
    ) -> None:
        self._tools = tools
        self._call_results = call_results
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> _FakeListToolsResult:
        return _FakeListToolsResult(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeCallToolResult:
        if name not in self._call_results:
            raise KeyError(f"no fake result configured for tool '{name}'")
        return self._call_results[name]


def _patch_session(
    monkeypatch: pytest.MonkeyPatch, client: MCPClient, fake_session: _FakeSession
) -> None:
    """Replace MCPClient._session with a fake async context manager yielding fake_session."""

    @asynccontextmanager
    async def _fake_session_cm():
        yield fake_session

    monkeypatch.setattr(client, "_session", _fake_session_cm)


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
    assert tools[0].parameters_schema == fake_tools[0].inputSchema


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
