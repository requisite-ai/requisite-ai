"""Unit tests for :class:`requisite.orchestrators.openai_agents_orchestrator.OpenAIAgentsOrchestrator`.

Real ``openai-agents`` coordination code runs for real when installed
(``pytest.importorskip``) -- only the wrapped Requisite ``Agent``'s
provider is faked, so no real network/LLM call happens.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from requisite.core.exceptions import AgentException, ConfigurationException, ProviderException
from requisite.orchestrators.native import _SupervisorDecision
from requisite.orchestrators.openai_agents_orchestrator import OpenAIAgentsOrchestrator
from tests.test_workflows import (
    EchoProvider,
    ScriptedSupervisorProvider,
    make_agent,
    make_agent_with_provider,
)


class _FailingProvider(EchoProvider):
    def chat(self, messages: Any, **kwargs: Any) -> Any:
        raise ProviderException("upstream exploded", provider="failing")

    async def achat(self, messages: Any, **kwargs: Any) -> Any:
        raise ProviderException("upstream exploded", provider="failing")


def test_openai_agents_without_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in list(sys.modules):
        if module_name == "agents" or module_name.startswith("agents."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setitem(sys.modules, "agents", None)

    with pytest.raises(ConfigurationException, match="openai-agents"):
        OpenAIAgentsOrchestrator().run([make_agent("A", "a")], "task")


@pytest.mark.asyncio
async def test_openai_agents_sequential_real_pipeline() -> None:
    pytest.importorskip("agents.models.interface")

    result = await OpenAIAgentsOrchestrator().arun(
        [make_agent("Researcher", "research"), make_agent("Writer", "write")], "Explain RAG"
    )

    assert result.orchestrator == "openai_agents"
    assert result.strategy == "sequential"
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    # The writer saw the researcher's real output, via Runner.run chaining.
    assert result.steps[1].content == "write:research:Explain RAG"
    assert result.content == result.steps[1].content


def test_openai_agents_run_sync_wraps_arun() -> None:
    pytest.importorskip("agents.models.interface")

    result = OpenAIAgentsOrchestrator().run(
        [make_agent("Researcher", "research"), make_agent("Writer", "write")], "Explain RAG"
    )

    assert len(result.steps) == 2


@pytest.mark.asyncio
async def test_openai_agents_supervisor_routes_to_both_workers_then_finishes() -> None:
    pytest.importorskip("agents.models.interface")

    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts about RAG"),
        _SupervisorDecision(action="delegate", worker="Writer", task="Draft a summary"),
        _SupervisorDecision(action="finish", final_answer="RAG combines retrieval and generation."),
    ]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    result = await OpenAIAgentsOrchestrator().arun(
        [supervisor, make_agent("Researcher", "research"), make_agent("Writer", "write")],
        "Explain RAG",
        strategy="supervisor",
    )

    assert result.orchestrator == "openai_agents"
    assert result.strategy == "supervisor"
    assert result.content == "RAG combines retrieval and generation."
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    assert result.steps[0].content == "research:Find facts about RAG"


@pytest.mark.asyncio
async def test_openai_agents_supervisor_exceeding_max_rounds_raises_agent_exception() -> None:
    pytest.importorskip("agents.models.interface")

    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="x") for _ in range(3)
    ]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    with pytest.raises(AgentException, match="max_rounds"):
        await OpenAIAgentsOrchestrator().arun(
            [supervisor, make_agent("Researcher", "research")],
            "task",
            strategy="supervisor",
            max_rounds=3,
        )


@pytest.mark.asyncio
async def test_openai_agents_step_failure_surfaces_the_real_provider_exception() -> None:
    pytest.importorskip("agents.models.interface")

    with pytest.raises(ProviderException, match="upstream exploded"):
        await OpenAIAgentsOrchestrator().arun(
            [make_agent("A", "a"), make_agent_with_provider("B", _FailingProvider())], "go"
        )


@pytest.mark.asyncio
async def test_openai_agents_never_uploads_traces_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK uploads traces (prompts and outputs) to OpenAI's platform by
    default -- a surprising side effect for a user whose agents run on
    another provider. Control run proves the spy would catch it."""
    pytest.importorskip("agents.models.interface")
    from agents import Agent, Runner
    from agents.tracing import get_trace_provider, processors

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-tracing-check")
    exported: list[int] = []
    monkeypatch.setattr(
        processors.BackendSpanExporter,
        "export",
        lambda self, items: exported.append(len(items)),
    )

    orchestrator = OpenAIAgentsOrchestrator()
    await orchestrator.arun([make_agent("A", "a"), make_agent("B", "b")], "hello")
    get_trace_provider().force_flush()
    assert exported == []

    # Control: the same proxy model under a bare Runner.run (tracing on) does export.
    model = orchestrator._require_sdk()["RequisiteModel"](make_agent("C", "c"))
    await Runner.run(Agent(name="C", model=model), "hello")
    get_trace_provider().force_flush()
    assert exported != []


def test_openai_agents_rejects_unknown_strategy() -> None:
    pytest.importorskip("agents.models.interface")

    with pytest.raises(ConfigurationException, match="sequential.*supervisor"):
        OpenAIAgentsOrchestrator().run(
            [make_agent("A", "a"), make_agent("B", "b")], "task", strategy="hierarchical"
        )


def test_openai_agents_requires_agents() -> None:
    with pytest.raises(ConfigurationException, match="no agents"):
        OpenAIAgentsOrchestrator().run([], "task")


def test_openai_agents_requires_input() -> None:
    with pytest.raises(ConfigurationException, match="input"):
        OpenAIAgentsOrchestrator().run([make_agent("A", "a")], None)
