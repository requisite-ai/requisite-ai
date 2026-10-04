"""Agent-owned persistent MCP sessions (``Agent(mcp_clients=[...])``).

Most tests run against a real stdio MCP subprocess (``tests/mcp_counting_server.py``)
and count how many server processes were spawned -- the property the feature exists for.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from requisite.agents.agent import Agent
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse, Role, ToolCall
from requisite.mcp import MCPClient
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry

pytest.importorskip("mcp")

_SERVER = str(Path(__file__).parent / "mcp_counting_server.py")


class _AddThenAnswerProvider(BaseProvider):
    """Calls the MCP ``add`` tool once per run, then answers with its result."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="k", model="m")

    @property
    def name(self) -> str:
        return "addthenanswer"

    def chat(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> ChatResponse:
        last = messages[-1]
        if last.role == Role.TOOL:
            return ChatResponse(content=f"sum={last.content}", model="m", provider=self.name)
        assert tools and "add" in {t.name for t in tools}
        call = ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})
        return ChatResponse(content="", model="m", provider=self.name, tool_calls=[call])

    async def achat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def astream(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
        yield


def _client(log: Path, name: str = "counting") -> MCPClient:
    return MCPClient.stdio(
        name=name,
        command=sys.executable,
        args=[_SERVER],
        env={**os.environ, "MCP_SPAWN_LOG": str(log)},
    )


def _agent(**kwargs: Any) -> Agent:
    registry = ProviderRegistry()
    registry.register("p", _AddThenAnswerProvider)
    return Agent(
        name="A",
        provider="p",
        settings=Settings(default_provider="p", model="m"),
        registry=registry,
        **kwargs,
    )


def _spawns(log: Path) -> int:
    return len(log.read_text(encoding="utf-8").splitlines())


@pytest.fixture
def log(tmp_path: Path) -> Path:
    path = tmp_path / "spawns.txt"
    path.write_text("")
    return path


def test_sync_runs_share_one_server_process(log: Path) -> None:
    agent = _agent(mcp_clients=[_client(log)])
    for _ in range(3):
        assert "sum=" in agent.run("go").content
    agent.close()

    assert _spawns(log) == 1  # discover + 3 tool calls, one process


@pytest.mark.asyncio
async def test_async_runs_share_one_server_process(log: Path) -> None:
    agent = _agent(mcp_clients=[_client(log)])
    for _ in range(3):
        assert "sum=" in (await agent.arun("go")).content
    await agent.aclose()

    assert _spawns(log) == 1


def test_context_manager_opens_eagerly_and_closes(log: Path) -> None:
    client = _client(log)
    with _agent(mcp_clients=[client]) as agent:
        assert _spawns(log) == 1  # opened on entry, before any run
        agent.run("go")
    assert client._persistent_session is None


@pytest.mark.asyncio
async def test_async_context_manager_closes(log: Path) -> None:
    client = _client(log)
    async with _agent(mcp_clients=[client]) as agent:
        await agent.arun("go")
    assert client._persistent_session is None


def test_close_is_idempotent_and_safe_before_open(log: Path) -> None:
    agent = _agent(mcp_clients=[_client(log)])
    agent.close()
    agent.run("go")
    agent.close()
    agent.close()


def test_reopens_after_close(log: Path) -> None:
    agent = _agent(mcp_clients=[_client(log)])
    agent.run("go")
    agent.close()
    agent.run("go")
    agent.close()

    assert _spawns(log) == 2


@pytest.mark.asyncio
async def test_using_sync_run_after_async_open_raises_clearly(log: Path) -> None:
    agent = _agent(mcp_clients=[_client(log)])
    await agent.arun("go")
    try:
        with pytest.raises(ConfigurationException, match="bound to the event loop"):
            agent.run("go")
    finally:
        await agent.aclose()


def test_using_async_after_sync_open_raises_clearly(log: Path) -> None:
    import asyncio

    agent = _agent(mcp_clients=[_client(log)])
    agent.run("go")
    try:
        with pytest.raises(ConfigurationException, match="bound to the event loop"):
            asyncio.run(agent.arun("go"))
    finally:
        agent.close()


def test_two_mcp_clients_exposing_same_tool_name_raise(log: Path) -> None:
    agent = _agent(mcp_clients=[_client(log, "one"), _client(log, "two")])
    with pytest.raises(ConfigurationException, match="both expose"):
        agent.run("go")
    # Nothing leaked: both sessions closed after the failed open.
    assert all(c._persistent_session is None for c in agent._mcp._clients)


def test_explicit_tool_wins_over_mcp_tool_of_same_name(log: Path) -> None:
    from requisite.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Local add."""
        return 100

    agent = _agent(tools=[add], mcp_clients=[_client(log)])
    assert agent.run("go").content == "sum=100"
    agent.close()


def test_failed_connect_leaves_agent_reusable() -> None:
    bad = MCPClient.stdio(name="bad", command=sys.executable, args=["-c", "raise SystemExit(1)"])
    agent = _agent(mcp_clients=[bad])
    with pytest.raises(Exception):
        agent.run("go")
    assert agent._mcp._mode is None


def test_agent_without_mcp_clients_is_unchanged() -> None:
    from requisite.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Local add."""
        return a + b

    agent = _agent(tools=[add])
    assert agent.run("go").content == "sum=5"
    agent.close()  # no-op


def test_adk_sync_workflow_reuses_one_session_and_closes_from_sync_code(log: Path) -> None:
    """ADK's sync run() drives arun() on the shared bridge loop, so the agent's
    sessions live on that loop and a plain close() must work afterwards."""
    pytest.importorskip("google.adk")
    from requisite.workflows import Workflow

    agent = _agent(mcp_clients=[_client(log)])
    workflow = Workflow().add(agent).sequential().use_adk()
    for _ in range(3):
        assert "sum=" in workflow.run("go").content
    agent.close()

    assert _spawns(log) == 1
