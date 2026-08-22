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

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from requisite.agents.agent import Agent
from requisite.capabilities.registry import CapabilityRegistry
from requisite.capabilities.resolvers import register_default_capabilities
from requisite.core.exceptions import ConfigurationException, MCPException
from requisite.core.interfaces import ChatResponse, Usage
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
