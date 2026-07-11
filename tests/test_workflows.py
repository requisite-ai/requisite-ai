"""Unit tests for :class:`requisite.workflows.workflow.Workflow` and the
native orchestrator's sequential/parallel execution strategies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import pytest

from requisite.agents.agent import Agent
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse, Message, StreamChunk
from requisite.orchestrators.factory import (
    OrchestratorRegistry,
    default_registry as default_orchestrator_registry,
)
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry
from requisite.workflows.workflow import Workflow


class EchoProvider(BaseProvider):
    """A fake provider that echoes back a transformed version of the last user message."""

    def __init__(self, *, prefix: str = "echo", model: str = "fake-model", **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=model)
        self.prefix = prefix

    @property
    def name(self) -> str:
        return "echo"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model=None,
        temperature=None,
        tools=None,
        response_model=None,
        **kwargs,
    ) -> ChatResponse:
        last_user = next((m.content for m in reversed(messages) if m.role.value == "user"), "")
        return ChatResponse(
            content=f"{self.prefix}:{last_user}", model=self._model, provider=self.name
        )

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> Iterator[StreamChunk]:  # pragma: no cover
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


def make_agent(name: str, prefix: str) -> Agent:
    provider_registry = ProviderRegistry()
    provider_registry.register("echo", lambda **kwargs: EchoProvider(prefix=prefix))
    settings = Settings(default_provider="echo", model="fake-model")
    return Agent(name=name, provider="echo", settings=settings, registry=provider_registry)


def test_workflow_add_returns_self_for_chaining() -> None:
    workflow = Workflow()
    a = make_agent("A", "a")
    b = make_agent("B", "b")
    result = workflow.add(a).add(b)
    assert result is workflow
    assert [agent.name for agent in workflow.agents] == ["A", "B"]


def test_workflow_sequential_pipes_output_between_agents() -> None:
    workflow = Workflow()
    workflow.add(make_agent("Researcher", "research")).add(make_agent("Writer", "write"))
    result = workflow.run("AI trends")
    assert result.content == "write:research:AI trends"
    assert result.orchestrator == "native"
    assert result.strategy == "sequential"
    assert len(result.steps) == 2


def test_workflow_parallel_combines_all_outputs() -> None:
    workflow = Workflow().parallel()
    workflow.add(make_agent("Researcher", "research")).add(make_agent("Writer", "write"))
    result = workflow.run("AI trends")
    assert "[Researcher]" in result.content
    assert "[Writer]" in result.content
    assert result.strategy == "parallel"
    assert len(result.steps) == 2


def test_workflow_run_without_input_raises() -> None:
    workflow = Workflow()
    workflow.add(make_agent("A", "a"))
    with pytest.raises(ConfigurationException):
        workflow.run()


def test_workflow_run_without_agents_raises() -> None:
    workflow = Workflow()
    with pytest.raises(ConfigurationException):
        workflow.run("hello")


def test_workflow_use_langgraph_without_dependency_raises_helpful_error() -> None:
    workflow = Workflow()
    workflow.add(make_agent("A", "a"))
    workflow.use_langgraph()
    with pytest.raises(ConfigurationException, match="langgraph"):
        workflow.run("hello")


def test_workflow_use_crewai_raises_roadmap_error() -> None:
    workflow = Workflow()
    workflow.add(make_agent("A", "a"))
    workflow.use_crewai()
    with pytest.raises(ConfigurationException, match="not yet implemented"):
        workflow.run("hello")


def test_workflow_unknown_strategy_raises() -> None:
    workflow = Workflow()
    workflow.add(make_agent("A", "a"))
    workflow._strategy = "tree_of_thoughts"  # not yet a supported strategy value
    with pytest.raises(ConfigurationException):
        workflow.run("hello")


@pytest.mark.asyncio
async def test_workflow_arun_sequential() -> None:
    workflow = Workflow()
    workflow.add(make_agent("Researcher", "research")).add(make_agent("Writer", "write"))
    result = await workflow.arun("AI trends")
    assert result.content == "write:research:AI trends"


def test_default_orchestrator_registry_has_native_and_langgraph() -> None:
    assert "native" in default_orchestrator_registry.available()
    assert "langgraph" in default_orchestrator_registry.available()
    assert "crewai" in default_orchestrator_registry.available()
    assert "autogen" in default_orchestrator_registry.available()


def test_orchestrator_registry_unknown_backend_raises() -> None:
    registry = OrchestratorRegistry()
    with pytest.raises(ConfigurationException):
        registry.create("does-not-exist")
