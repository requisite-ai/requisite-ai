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
from requisite.orchestrators.base import WorkflowResult
from requisite.orchestrators.factory import (
    OrchestratorRegistry,
    default_registry as default_orchestrator_registry,
)
from requisite.orchestrators.native import (
    _Plan,
    _PlanStep,
    _SupervisorDecision,
    _ThoughtEvaluation,
    _ThoughtScore,
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


class ScriptedThoughtEvaluatorProvider(BaseProvider):
    """A fake provider whose chat() returns successive `_ThoughtEvaluation`s as `.parsed`."""

    def __init__(
        self, *, evaluations: list[_ThoughtEvaluation], model: str = "fake-model", **kwargs: Any
    ) -> None:
        super().__init__(api_key="fake-key", model=model)
        self._evaluations = evaluations
        self._call_count = 0

    @property
    def name(self) -> str:
        return "scripted-tot-evaluator"

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
        evaluation = self._evaluations[self._call_count]
        self._call_count += 1
        return ChatResponse(content="", model=self._model, provider=self.name, parsed=evaluation)

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
        self.last_messages: list[Message] = []

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
        self.last_messages = list(messages)
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


def test_workflow_use_langgraph_supervisor_routes_to_both_workers_then_finishes() -> None:
    """Proof of real conditional routing, not a disguised chain: the
    supervisor delegates to two *different* workers across rounds --
    only possible if add_conditional_edges is genuinely re-evaluated
    each time the graph returns to the supervisor node, not a fixed
    edge wired once at build time."""
    pytest.importorskip("langgraph")

    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts about RAG"),
        _SupervisorDecision(action="delegate", worker="Writer", task="Draft a summary"),
        _SupervisorDecision(
            action="finish", final_answer="RAG combines retrieval with generation."
        ),
    ]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )
    researcher = make_agent("Researcher", "research")
    writer = make_agent("Writer", "write")

    workflow = Workflow().supervisor()
    workflow.add(supervisor_agent).add(researcher).add(writer)
    workflow.use_langgraph()
    result = workflow.run("Explain RAG")

    assert result.strategy == "supervisor"
    assert result.orchestrator == "langgraph"
    assert result.content == "RAG combines retrieval with generation."
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]


def test_workflow_use_langgraph_supervisor_exceeds_max_rounds_raises() -> None:
    pytest.importorskip("langgraph")

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
    workflow.use_langgraph()
    with pytest.raises(AgentException, match="max_rounds"):
        workflow.run("Explain RAG", max_rounds=3)


def test_workflow_use_langgraph_supervisor_unknown_worker_raises() -> None:
    pytest.importorskip("langgraph")

    decisions = [_SupervisorDecision(action="delegate", worker="Nonexistent", task="do X")]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )
    workflow = Workflow().supervisor()
    workflow.add(supervisor_agent).add(make_agent("Researcher", "research"))
    workflow.use_langgraph()
    with pytest.raises(ConfigurationException, match="unknown"):
        workflow.run("task")


@pytest.mark.asyncio
async def test_workflow_use_langgraph_arun_supervisor() -> None:
    pytest.importorskip("langgraph")

    decisions = [_SupervisorDecision(action="finish", final_answer="done")]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )
    workflow = Workflow().supervisor()
    workflow.add(supervisor_agent).add(make_agent("Worker", "w"))
    workflow.use_langgraph()
    result = await workflow.arun("task")

    assert result.content == "done"
    assert result.orchestrator == "langgraph"
    assert result.strategy == "supervisor"


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


# ---------------------------------------------------------------------------
# Critic strategy
# ---------------------------------------------------------------------------


def test_workflow_critic_revises_output() -> None:
    generator = make_agent_with_provider(
        "Generator", ScriptedReflectionProvider(responses=["draft v1", "draft v2, better tone"])
    )
    critic = make_agent_with_provider(
        "Critic", ScriptedReflectionProvider(responses=["fix the tone"])
    )
    workflow = Workflow().critic()
    workflow.add(generator).add(critic)

    result = workflow.run("Write a haiku", max_rounds=2)

    assert result.content == "draft v2, better tone"
    assert result.strategy == "critic"
    assert result.orchestrator == "native"
    assert [s.agent_name for s in result.steps] == ["Generator", "Critic", "Generator"]


def test_workflow_critic_stops_early_on_no_changes_needed() -> None:
    generator = make_agent_with_provider(
        "Generator", ScriptedReflectionProvider(responses=["draft v1"])
    )
    critic = make_agent_with_provider(
        "Critic", ScriptedReflectionProvider(responses=["NO_CHANGES_NEEDED"])
    )
    workflow = Workflow().critic()
    workflow.add(generator).add(critic)

    result = workflow.run("Write a haiku", max_rounds=5)

    assert result.content == "draft v1"
    assert len(result.steps) == 2


def test_workflow_critic_requires_exactly_two_agents() -> None:
    workflow = Workflow().critic()
    workflow.add(make_agent("A", "a"))
    with pytest.raises(ConfigurationException, match="critic"):
        workflow.run("task")

    workflow2 = Workflow().critic()
    workflow2.add(make_agent("A", "a")).add(make_agent("B", "b")).add(make_agent("C", "c"))
    with pytest.raises(ConfigurationException, match="critic"):
        workflow2.run("task")


@pytest.mark.asyncio
async def test_workflow_arun_critic() -> None:
    generator = make_agent_with_provider(
        "Generator", ScriptedReflectionProvider(responses=["draft"])
    )
    critic = make_agent_with_provider(
        "Critic", ScriptedReflectionProvider(responses=["NO_CHANGES_NEEDED"])
    )
    workflow = Workflow().critic()
    workflow.add(generator).add(critic)

    result = await workflow.arun("task", max_rounds=5)

    assert result.content == "draft"


# ---------------------------------------------------------------------------
# Consensus strategy
# ---------------------------------------------------------------------------


def test_workflow_consensus_synthesizes_participant_answers() -> None:
    synthesizer_provider = ScriptedReflectionProvider(responses=["synthesized answer"])
    synthesizer = make_agent_with_provider("Synthesizer", synthesizer_provider)
    a = make_agent("A", "alpha")
    b = make_agent("B", "beta")

    workflow = Workflow().consensus()
    workflow.add(synthesizer).add(a).add(b)
    result = workflow.run("What is RAG?")

    assert result.strategy == "consensus"
    assert result.orchestrator == "native"
    assert result.content == "synthesized answer"
    assert len(result.steps) == 3  # 2 participants + the synthesis
    participant_names = {result.steps[0].agent_name, result.steps[1].agent_name}
    assert participant_names == {"A", "B"}
    assert result.steps[2].agent_name == "Synthesizer"

    # The synthesizer actually saw both participants' independent answers.
    synthesis_prompt = synthesizer_provider.last_messages[-1].content
    assert "alpha:What is RAG?" in synthesis_prompt
    assert "beta:What is RAG?" in synthesis_prompt


def test_workflow_consensus_requires_at_least_two_agents() -> None:
    workflow = Workflow().consensus()
    workflow.add(make_agent("Solo", "solo"))
    with pytest.raises(ConfigurationException, match="consensus"):
        workflow.run("task")


def test_workflow_consensus_duplicate_participant_names_raises() -> None:
    workflow = Workflow().consensus()
    workflow.add(make_agent("Synth", "synth"))
    workflow.add(make_agent("Dup", "a")).add(make_agent("Dup", "b"))
    with pytest.raises(ConfigurationException, match="unique worker names"):
        workflow.run("task")


@pytest.mark.asyncio
async def test_workflow_arun_consensus() -> None:
    synthesizer = make_agent_with_provider(
        "Synthesizer", ScriptedReflectionProvider(responses=["final"])
    )
    a = make_agent("A", "alpha")
    b = make_agent("B", "beta")
    workflow = Workflow().consensus()
    workflow.add(synthesizer).add(a).add(b)

    result = await workflow.arun("What is RAG?")

    assert result.content == "final"
    assert len(result.steps) == 3


# ---------------------------------------------------------------------------
# Debate strategy
# ---------------------------------------------------------------------------


def test_workflow_debate_runs_rounds_and_delivers_verdict() -> None:
    moderator_provider = ScriptedReflectionProvider(responses=["final verdict"])
    moderator = make_agent_with_provider("Moderator", moderator_provider)
    debater_a = make_agent_with_provider(
        "A", ScriptedReflectionProvider(responses=["A round 1", "A round 2"])
    )
    debater_b = make_agent_with_provider(
        "B", ScriptedReflectionProvider(responses=["B round 1", "B round 2"])
    )
    workflow = Workflow().debate()
    workflow.add(moderator).add(debater_a).add(debater_b)

    result = workflow.run("Is X better than Y?", max_rounds=2)

    assert result.content == "final verdict"
    assert result.strategy == "debate"
    assert result.orchestrator == "native"
    # 2 rounds * 2 debaters + 1 verdict
    assert len(result.steps) == 5
    assert result.steps[-1].agent_name == "Moderator"
    assert {s.agent_name for s in result.steps[:-1]} == {"A", "B"}

    # The moderator's verdict prompt saw the full debate transcript.
    verdict_prompt = moderator_provider.last_messages[-1].content
    assert "A round 1" in verdict_prompt
    assert "A round 2" in verdict_prompt
    assert "B round 1" in verdict_prompt
    assert "B round 2" in verdict_prompt


def test_workflow_debate_requires_at_least_two_agents() -> None:
    workflow = Workflow().debate()
    workflow.add(make_agent("Solo", "solo"))
    with pytest.raises(ConfigurationException, match="debate"):
        workflow.run("task")


@pytest.mark.asyncio
async def test_workflow_arun_debate() -> None:
    moderator = make_agent_with_provider(
        "Moderator", ScriptedReflectionProvider(responses=["verdict"])
    )
    debater = make_agent_with_provider("A", ScriptedReflectionProvider(responses=["round 1"]))
    workflow = Workflow().debate()
    workflow.add(moderator).add(debater)

    result = await workflow.arun("task", max_rounds=1)

    assert result.content == "verdict"
    assert len(result.steps) == 2


# ---------------------------------------------------------------------------
# Map-reduce strategy
# ---------------------------------------------------------------------------


def test_workflow_map_reduce_distributes_items_round_robin() -> None:
    reducer = make_agent_with_provider(
        "Reducer", ScriptedReflectionProvider(responses=["combined result"])
    )
    mapper_a = make_agent("MapperA", "a")
    mapper_b = make_agent("MapperB", "b")
    workflow = Workflow().map_reduce()
    workflow.add(reducer).add(mapper_a).add(mapper_b)

    result = workflow.run("Summarize these items", map_items=["item1", "item2", "item3"])

    assert result.content == "combined result"
    assert result.strategy == "map_reduce"
    assert result.orchestrator == "native"
    assert len(result.steps) == 4  # 3 mapped items + 1 reduce
    assert result.steps[-1].agent_name == "Reducer"
    # Round-robin over 2 mappers: item1->MapperA, item2->MapperB, item3->MapperA
    assert [s.agent_name for s in result.steps[:-1]] == ["MapperA", "MapperB", "MapperA"]


def test_workflow_map_reduce_missing_map_items_raises() -> None:
    workflow = Workflow().map_reduce()
    workflow.add(make_agent("Reducer", "r")).add(make_agent("Mapper", "m"))
    with pytest.raises(ConfigurationException, match="map_items"):
        workflow.run("task")


def test_workflow_map_reduce_empty_map_items_raises() -> None:
    workflow = Workflow().map_reduce()
    workflow.add(make_agent("Reducer", "r")).add(make_agent("Mapper", "m"))
    with pytest.raises(ConfigurationException, match="map_items"):
        workflow.run("task", map_items=[])


def test_workflow_map_reduce_requires_at_least_two_agents() -> None:
    workflow = Workflow().map_reduce()
    workflow.add(make_agent("Solo", "solo"))
    with pytest.raises(ConfigurationException, match="map-reduce"):
        workflow.run("task", map_items=["a"])


@pytest.mark.asyncio
async def test_workflow_arun_map_reduce() -> None:
    reducer = make_agent_with_provider("Reducer", ScriptedReflectionProvider(responses=["final"]))
    mapper = make_agent("Mapper", "m")
    workflow = Workflow().map_reduce()
    workflow.add(reducer).add(mapper)

    result = await workflow.arun("task", map_items=["x", "y"])

    assert result.content == "final"
    assert len(result.steps) == 3


# ---------------------------------------------------------------------------
# Tree-of-thoughts strategy
# ---------------------------------------------------------------------------


def test_workflow_tree_of_thoughts_prunes_to_best_scoring_path() -> None:
    evaluations = [
        # Level 1: candidate 1 (Beta) scores higher -> survives.
        _ThoughtEvaluation(
            scores=[_ThoughtScore(index=0, score=3.0), _ThoughtScore(index=1, score=9.0)]
        ),
        # Level 2: candidate 0 (Alpha, continuing Beta's path) scores higher -> final.
        _ThoughtEvaluation(
            scores=[_ThoughtScore(index=0, score=9.0), _ThoughtScore(index=1, score=2.0)]
        ),
    ]
    evaluator = make_agent_with_provider(
        "Evaluator", ScriptedThoughtEvaluatorProvider(evaluations=evaluations)
    )
    alpha = make_agent("Alpha", "alpha")
    beta = make_agent("Beta", "beta")

    workflow = Workflow().tree_of_thoughts()
    workflow.add(evaluator).add(alpha).add(beta)
    result = workflow.run("Solve the task", breadth=2, beam_width=1, max_depth=2)

    assert result.strategy == "tree_of_thoughts"
    assert result.orchestrator == "native"
    assert len(result.steps) == 4  # 2 candidates/level * 2 levels, pruned ones included
    # Level 1 candidates: Alpha then Beta (round-robin); level 2 candidates continue
    # only Beta's surviving path, again Alpha then Beta.
    assert [s.agent_name for s in result.steps] == ["Alpha", "Beta", "Alpha", "Beta"]
    # The winning final thought came from level 2's Alpha candidate.
    assert result.content == result.steps[2].content


def test_workflow_tree_of_thoughts_stops_early_on_finished_candidate() -> None:
    evaluations = [
        _ThoughtEvaluation(
            scores=[
                _ThoughtScore(index=0, score=5.0),
                _ThoughtScore(index=1, score=9.0, finished=True),
            ]
        ),
    ]
    evaluator = make_agent_with_provider(
        "Evaluator", ScriptedThoughtEvaluatorProvider(evaluations=evaluations)
    )
    workflow = Workflow().tree_of_thoughts()
    workflow.add(evaluator).add(make_agent("Alpha", "alpha")).add(make_agent("Beta", "beta"))

    # max_depth=5 but should stop after level 1 since a candidate is finished.
    result = workflow.run("Solve the task", breadth=2, beam_width=1, max_depth=5)

    assert len(result.steps) == 2
    assert result.content == result.steps[1].content


def test_workflow_tree_of_thoughts_round_robins_across_thinkers() -> None:
    evaluations = [
        _ThoughtEvaluation(
            scores=[
                _ThoughtScore(index=0, score=1.0),
                _ThoughtScore(index=1, score=2.0),
                _ThoughtScore(index=2, score=3.0),
            ]
        ),
    ]
    evaluator = make_agent_with_provider(
        "Evaluator", ScriptedThoughtEvaluatorProvider(evaluations=evaluations)
    )
    workflow = Workflow().tree_of_thoughts()
    workflow.add(evaluator).add(make_agent("Alpha", "alpha")).add(make_agent("Beta", "beta"))

    result = workflow.run("Solve the task", breadth=3, beam_width=1, max_depth=1)

    assert [s.agent_name for s in result.steps] == ["Alpha", "Beta", "Alpha"]


def test_workflow_tree_of_thoughts_requires_at_least_two_agents() -> None:
    workflow = Workflow().tree_of_thoughts()
    workflow.add(make_agent("Solo", "solo"))
    with pytest.raises(ConfigurationException, match="tree-of-thoughts"):
        workflow.run("task")


def test_workflow_tree_of_thoughts_invalid_breadth_raises() -> None:
    workflow = Workflow().tree_of_thoughts()
    workflow.add(make_agent("Evaluator", "e")).add(make_agent("Thinker", "t"))
    with pytest.raises(ConfigurationException, match="breadth"):
        workflow.run("task", breadth=0)


def test_workflow_tree_of_thoughts_invalid_beam_width_raises() -> None:
    workflow = Workflow().tree_of_thoughts()
    workflow.add(make_agent("Evaluator", "e")).add(make_agent("Thinker", "t"))
    with pytest.raises(ConfigurationException, match="beam_width"):
        workflow.run("task", beam_width=0)


def test_workflow_tree_of_thoughts_invalid_max_depth_raises() -> None:
    workflow = Workflow().tree_of_thoughts()
    workflow.add(make_agent("Evaluator", "e")).add(make_agent("Thinker", "t"))
    with pytest.raises(ConfigurationException, match="max_depth"):
        workflow.run("task", max_depth=0)


@pytest.mark.asyncio
async def test_workflow_arun_tree_of_thoughts() -> None:
    evaluations = [
        _ThoughtEvaluation(
            scores=[_ThoughtScore(index=0, score=5.0, finished=True)],
        ),
    ]
    evaluator = make_agent_with_provider(
        "Evaluator", ScriptedThoughtEvaluatorProvider(evaluations=evaluations)
    )
    workflow = Workflow().tree_of_thoughts()
    workflow.add(evaluator).add(make_agent("Alpha", "alpha"))

    result = await workflow.arun("task", breadth=1, beam_width=1, max_depth=3)

    assert result.strategy == "tree_of_thoughts"
    assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# Hierarchical strategy
# ---------------------------------------------------------------------------


def test_workflow_hierarchical_delegates_to_agent_then_finishes() -> None:
    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts about RAG"),
        _SupervisorDecision(
            action="finish", final_answer="RAG combines retrieval with generation."
        ),
    ]
    coordinator = make_agent_with_provider(
        "Coordinator", ScriptedSupervisorProvider(decisions=decisions)
    )
    researcher = make_agent("Researcher", "research")

    workflow = Workflow().hierarchical()
    workflow.add(coordinator).add(researcher)
    result = workflow.run("Explain RAG")

    assert result.strategy == "hierarchical"
    assert result.orchestrator == "native"
    assert result.content == "RAG combines retrieval with generation."
    assert len(result.steps) == 1
    assert result.steps[0].agent_name == "Researcher"


def test_workflow_hierarchical_delegates_to_nested_workflow() -> None:
    decisions = [
        _SupervisorDecision(action="delegate", worker="ResearchTeam", task="Find facts about RAG"),
        _SupervisorDecision(action="finish", final_answer="done"),
    ]
    coordinator = make_agent_with_provider(
        "Coordinator", ScriptedSupervisorProvider(decisions=decisions)
    )
    research_team = Workflow(name="ResearchTeam")
    research_team.add(make_agent("SubResearcher", "research"))

    workflow = Workflow().hierarchical()
    workflow.add(coordinator).add(research_team)
    result = workflow.run("Explain RAG")

    assert result.content == "done"
    assert len(result.steps) == 1
    # The delegate result is a WorkflowResult (from the nested team), not an AgentResult.
    assert isinstance(result.steps[0], WorkflowResult)
    assert result.steps[0].content == "research:Find facts about RAG"
    assert result.steps[0].strategy == "sequential"


def test_workflow_hierarchical_requires_at_least_two_agents() -> None:
    workflow = Workflow().hierarchical()
    workflow.add(make_agent("Solo", "solo"))
    with pytest.raises(ConfigurationException, match="hierarchical"):
        workflow.run("task")


def test_workflow_hierarchical_coordinator_must_be_agent() -> None:
    sub_team = Workflow(name="Team")
    sub_team.add(make_agent("Leaf", "leaf"))
    not_a_coordinator = Workflow(name="NotACoordinator")

    workflow = Workflow().hierarchical()
    workflow.add(not_a_coordinator).add(sub_team)
    with pytest.raises(ConfigurationException, match="Agent"):
        workflow.run("task")


def test_workflow_hierarchical_unnamed_workflow_delegate_raises() -> None:
    coordinator = make_agent_with_provider("Coordinator", ScriptedSupervisorProvider(decisions=[]))
    unnamed_team = Workflow()  # no name=
    unnamed_team.add(make_agent("Leaf", "leaf"))

    workflow = Workflow().hierarchical()
    workflow.add(coordinator).add(unnamed_team)
    with pytest.raises(ConfigurationException, match="name"):
        workflow.run("task")


def test_workflow_hierarchical_duplicate_delegate_names_raises() -> None:
    workflow = Workflow().hierarchical()
    workflow.add(make_agent("Coordinator", "c"))
    workflow.add(make_agent("Dup", "a")).add(make_agent("Dup", "b"))
    with pytest.raises(ConfigurationException, match="unique delegate names"):
        workflow.run("task")


def test_workflow_hierarchical_exceeds_max_rounds_raises() -> None:
    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts")
        for _ in range(3)
    ]
    coordinator = make_agent_with_provider(
        "Coordinator", ScriptedSupervisorProvider(decisions=decisions)
    )
    researcher = make_agent("Researcher", "research")

    workflow = Workflow().hierarchical()
    workflow.add(coordinator).add(researcher)
    with pytest.raises(AgentException, match="max_rounds"):
        workflow.run("task", max_rounds=3)


@pytest.mark.asyncio
async def test_workflow_arun_hierarchical() -> None:
    decisions = [_SupervisorDecision(action="finish", final_answer="done")]
    coordinator = make_agent_with_provider(
        "Coordinator", ScriptedSupervisorProvider(decisions=decisions)
    )
    workflow = Workflow().hierarchical()
    workflow.add(coordinator).add(make_agent("Worker", "w"))

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
