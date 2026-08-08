"""Unit tests for :class:`requisite.workflows.workflow.Workflow` and the
native orchestrator's sequential/parallel execution strategies.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import pytest

from requisite.agents.agent import Agent
from requisite.config.settings import Settings
from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.core.interfaces import ChatResponse, Message, StreamChunk
from requisite.orchestrators.factory import (
    OrchestratorRegistry,
    default_registry as default_orchestrator_registry,
)
from requisite.orchestrators.native import _Plan, _PlanStep, _SupervisorDecision
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


def make_agent_with_provider(name: str, provider: BaseProvider) -> Agent:
    """Build an Agent backed by an already-constructed fake provider instance."""
    provider_registry = ProviderRegistry()
    provider_registry.register("scripted", lambda **kwargs: provider)
    settings = Settings(default_provider="scripted", model="fake-model")
    return Agent(name=name, provider="scripted", settings=settings, registry=provider_registry)


class ScriptedPlannerProvider(BaseProvider):
    """A fake provider whose chat() returns a fixed `_Plan` as `.parsed`."""

    def __init__(self, *, plan: _Plan, model: str = "fake-model", **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=model)
        self._plan = plan

    @property
    def name(self) -> str:
        return "scripted-planner"

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
        return ChatResponse(content="", model=self._model, provider=self.name, parsed=self._plan)

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


class ScriptedSupervisorProvider(BaseProvider):
    """A fake provider whose chat() returns successive `_SupervisorDecision`s as `.parsed`."""

    def __init__(
        self, *, decisions: list[_SupervisorDecision], model: str = "fake-model", **kwargs: Any
    ) -> None:
        super().__init__(api_key="fake-key", model=model)
        self._decisions = decisions
        self._call_count = 0

    @property
    def name(self) -> str:
        return "scripted-supervisor"

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
        decision = self._decisions[self._call_count]
        self._call_count += 1
        return ChatResponse(content="", model=self._model, provider=self.name, parsed=decision)

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


class ScriptedReflectionProvider(BaseProvider):
    """A fake provider that returns pre-set text responses in sequence, by call count."""

    def __init__(self, *, responses: list[str], model: str = "fake-model", **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=model)
        self._responses = responses
        self._call_count = 0

    @property
    def name(self) -> str:
        return "scripted-reflection"

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
        content = self._responses[self._call_count]
        self._call_count += 1
        return ChatResponse(content=content, model=self._model, provider=self.name)

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


def test_workflow_use_langgraph_without_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the lazy `from langgraph.graph import ...` import inside
    # LangGraphOrchestrator to fail, regardless of whether langgraph is
    # actually installed in the environment running this test (CI installs
    # it via the `all` extra, so this can't rely on it being absent).
    monkeypatch.setitem(sys.modules, "langgraph", None)
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)

    workflow = Workflow()
    workflow.add(make_agent("A", "a"))
    workflow.use_langgraph()
    with pytest.raises(ConfigurationException, match="langgraph"):
        workflow.run("hello")


def test_workflow_use_langgraph_runs_a_real_sequential_pipeline() -> None:
    pytest.importorskip("langgraph")

    workflow = Workflow()
    workflow.add(make_agent("Researcher", "research")).add(make_agent("Writer", "write"))
    workflow.use_langgraph()

    result = workflow.run("AI trends")
    assert result.content == "write:research:AI trends"
    assert result.orchestrator == "langgraph"
    assert result.strategy == "sequential"
    assert len(result.steps) == 2


@pytest.mark.asyncio
async def test_workflow_use_langgraph_arun_real_sequential_pipeline() -> None:
    pytest.importorskip("langgraph")

    workflow = Workflow()
    workflow.add(make_agent("Researcher", "research")).add(make_agent("Writer", "write"))
    workflow.use_langgraph()

    result = await workflow.arun("AI trends")
    assert result.content == "write:research:AI trends"


def test_workflow_use_langgraph_rejects_parallel_strategy() -> None:
    pytest.importorskip("langgraph")

    workflow = Workflow().parallel()
    workflow.add(make_agent("A", "a"))
    workflow.use_langgraph()
    with pytest.raises(ConfigurationException, match="sequential"):
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


# ---------------------------------------------------------------------------
# Reflection strategy
# ---------------------------------------------------------------------------


def test_workflow_reflection_revises_output() -> None:
    provider = ScriptedReflectionProvider(
        responses=["draft v1", "fix the tone", "draft v2, better tone"]
    )
    worker = make_agent_with_provider("Writer", provider)
    workflow = Workflow().reflection()
    workflow.add(worker)

    result = workflow.run("Write a haiku", max_rounds=2)

    assert result.content == "draft v2, better tone"
    assert result.strategy == "reflection"
    assert result.orchestrator == "native"
    assert len(result.steps) == 3


def test_workflow_reflection_stops_early_on_no_changes_needed() -> None:
    provider = ScriptedReflectionProvider(responses=["draft v1", "NO_CHANGES_NEEDED"])
    worker = make_agent_with_provider("Writer", provider)
    workflow = Workflow().reflection()
    workflow.add(worker)

    result = workflow.run("Write a haiku", max_rounds=5)

    assert result.content == "draft v1"
    assert len(result.steps) == 2


def test_workflow_reflection_requires_exactly_one_agent() -> None:
    workflow = Workflow().reflection()
    workflow.add(make_agent("A", "a")).add(make_agent("B", "b"))
    with pytest.raises(ConfigurationException, match="reflection"):
        workflow.run("task")


@pytest.mark.asyncio
async def test_workflow_arun_reflection() -> None:
    provider = ScriptedReflectionProvider(responses=["draft", "NO_CHANGES_NEEDED"])
    worker = make_agent_with_provider("Writer", provider)
    workflow = Workflow().reflection()
    workflow.add(worker)

    result = await workflow.arun("task", max_rounds=5)

    assert result.content == "draft"


# ---------------------------------------------------------------------------
# Planner strategy
# ---------------------------------------------------------------------------


def test_workflow_planner_executes_plan_across_workers() -> None:
    plan = _Plan(
        steps=[
            _PlanStep(agent="Researcher", task="Find 3 facts about RAG"),
            _PlanStep(agent="Writer", task="Summarize the facts"),
        ]
    )
    planner_agent = make_agent_with_provider("Planner", ScriptedPlannerProvider(plan=plan))
    researcher = make_agent("Researcher", "research")
    writer = make_agent("Writer", "write")

    workflow = Workflow().planner()
    workflow.add(planner_agent).add(researcher).add(writer)
    result = workflow.run("Explain RAG")

    assert result.strategy == "planner"
    assert result.orchestrator == "native"
    assert len(result.steps) == 2
    assert result.steps[0].agent_name == "Researcher"
    assert result.steps[1].agent_name == "Writer"
    assert result.content.startswith("write:Summarize the facts")
    assert "[Researcher] research:Find 3 facts about RAG" in result.content


def test_workflow_planner_empty_plan_raises() -> None:
    planner_agent = make_agent_with_provider(
        "Planner", ScriptedPlannerProvider(plan=_Plan(steps=[]))
    )
    workflow = Workflow().planner()
    workflow.add(planner_agent).add(make_agent("Researcher", "research"))
    with pytest.raises(ConfigurationException, match="empty plan"):
        workflow.run("task")


def test_workflow_planner_unknown_worker_raises() -> None:
    plan = _Plan(steps=[_PlanStep(agent="Nonexistent", task="do X")])
    planner_agent = make_agent_with_provider("Planner", ScriptedPlannerProvider(plan=plan))
    workflow = Workflow().planner()
    workflow.add(planner_agent).add(make_agent("Researcher", "research"))
    with pytest.raises(ConfigurationException, match="unknown worker"):
        workflow.run("task")


def test_workflow_planner_requires_at_least_two_agents() -> None:
    workflow = Workflow().planner()
    workflow.add(make_agent("Solo", "solo"))
    with pytest.raises(ConfigurationException, match="planner"):
        workflow.run("task")


def test_workflow_planner_duplicate_worker_names_raises() -> None:
    workflow = Workflow().planner()
    workflow.add(make_agent("Planner", "plan"))
    workflow.add(make_agent("Dup", "a")).add(make_agent("Dup", "b"))
    with pytest.raises(ConfigurationException, match="unique worker names"):
        workflow.run("task")


@pytest.mark.asyncio
async def test_workflow_arun_planner() -> None:
    plan = _Plan(steps=[_PlanStep(agent="Researcher", task="do X")])
    planner_agent = make_agent_with_provider("Planner", ScriptedPlannerProvider(plan=plan))
    researcher = make_agent("Researcher", "research")
    workflow = Workflow().planner()
    workflow.add(planner_agent).add(researcher)

    result = await workflow.arun("task")

    assert result.content == "research:do X"


# ---------------------------------------------------------------------------
# Supervisor strategy
# ---------------------------------------------------------------------------


def test_workflow_supervisor_delegates_then_finishes() -> None:
    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts about RAG"),
        _SupervisorDecision(
            action="finish", final_answer="RAG combines retrieval with generation."
        ),
    ]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )
    researcher = make_agent("Researcher", "research")

    workflow = Workflow().supervisor()
    workflow.add(supervisor_agent).add(researcher)
    result = workflow.run("Explain RAG")

    assert result.strategy == "supervisor"
    assert result.orchestrator == "native"
    assert result.content == "RAG combines retrieval with generation."
    assert len(result.steps) == 1
    assert result.steps[0].agent_name == "Researcher"


def test_workflow_supervisor_exceeds_max_rounds_raises() -> None:
    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts")
        for _ in range(3)
    ]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )
    researcher = make_agent("Researcher", "research")

    workflow = Workflow().supervisor()
    workflow.add(supervisor_agent).add(researcher)
    with pytest.raises(AgentException, match="max_rounds"):
        workflow.run("Explain RAG", max_rounds=3)


def test_workflow_supervisor_unknown_worker_raises() -> None:
    decisions = [_SupervisorDecision(action="delegate", worker="Nonexistent", task="do X")]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )
    workflow = Workflow().supervisor()
    workflow.add(supervisor_agent).add(make_agent("Researcher", "research"))
    with pytest.raises(ConfigurationException, match="unknown"):
        workflow.run("task")


def test_workflow_supervisor_requires_at_least_two_agents() -> None:
    workflow = Workflow().supervisor()
    workflow.add(make_agent("Solo", "solo"))
    with pytest.raises(ConfigurationException, match="supervisor"):
        workflow.run("task")


@pytest.mark.asyncio
async def test_workflow_arun_supervisor() -> None:
    decisions = [_SupervisorDecision(action="finish", final_answer="done")]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )
    workflow = Workflow().supervisor()
    workflow.add(supervisor_agent).add(make_agent("Worker", "w"))

    result = await workflow.arun("task")

    assert result.content == "done"


def test_default_orchestrator_registry_has_native_and_langgraph() -> None:
    assert "native" in default_orchestrator_registry.available()
    assert "langgraph" in default_orchestrator_registry.available()
    assert "crewai" in default_orchestrator_registry.available()
    assert "autogen" in default_orchestrator_registry.available()


def test_orchestrator_registry_unknown_backend_raises() -> None:
    registry = OrchestratorRegistry()
    with pytest.raises(ConfigurationException):
        registry.create("does-not-exist")
