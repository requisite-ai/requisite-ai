"""Unit tests for :class:`requisite.orchestrators.autogen_orchestrator.AutoGenOrchestrator`.

Real ``autogen-agentchat``/``autogen-core`` coordination code runs for
real when installed (``pytest.importorskip``) -- only the wrapped
Requisite ``Agent``'s provider is faked (``EchoProvider``/
``ScriptedSupervisorProvider``), so no real network/LLM call happens,
consistent with the framework's no-network-in-tests rule.
"""

from __future__ import annotations

import sys

import pytest

from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.orchestrators.autogen_orchestrator import AutoGenOrchestrator
from requisite.orchestrators.native import _SupervisorDecision
from tests.test_workflows import (  # noqa: F401
    EchoProvider,
    ScriptedSupervisorProvider,
    make_agent,
    make_agent_with_provider,
)


def test_autogen_without_dependency_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name.startswith("autogen_agentchat") or module_name.startswith("autogen_core"):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setitem(sys.modules, "autogen_agentchat", None)

    orchestrator = AutoGenOrchestrator()
    with pytest.raises(ConfigurationException, match="autogen"):
        orchestrator.run([make_agent("A", "a")], "task")


@pytest.mark.asyncio
async def test_autogen_orchestrator_sequential_real_pipeline() -> None:
    pytest.importorskip("autogen_agentchat")

    researcher = make_agent("Researcher", "research")
    writer = make_agent("Writer", "write")

    orchestrator = AutoGenOrchestrator()
    result = await orchestrator.arun([researcher, writer], "Explain RAG", strategy="sequential")

    assert result.orchestrator == "autogen"
    assert result.strategy == "sequential"
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    # The writer's turn saw the researcher's real output via RoundRobinGroupChat's
    # own running-conversation turn-taking, not manual state passing.
    assert "research:Explain RAG" in result.steps[1].content


def test_autogen_orchestrator_run_sync_wraps_arun() -> None:
    pytest.importorskip("autogen_agentchat")

    researcher = make_agent("Researcher", "research")
    writer = make_agent("Writer", "write")

    orchestrator = AutoGenOrchestrator()
    result = orchestrator.run([researcher, writer], "Explain RAG", strategy="sequential")

    assert len(result.steps) == 2


@pytest.mark.asyncio
async def test_autogen_orchestrator_supervisor_routes_to_both_workers_then_finishes() -> None:
    """Proof of real conditional routing: the supervisor delegates to two
    *different* workers across rounds -- only possible if selector_func is
    genuinely re-evaluated each round, not a fixed order."""
    pytest.importorskip("autogen_agentchat")

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

    orchestrator = AutoGenOrchestrator()
    result = await orchestrator.arun(
        [supervisor_agent, researcher, writer], "Explain RAG", strategy="supervisor"
    )

    assert result.orchestrator == "autogen"
    assert result.strategy == "supervisor"
    assert result.content == "RAG combines retrieval with generation."
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]


@pytest.mark.asyncio
async def test_autogen_orchestrator_supervisor_exceeds_max_rounds_raises() -> None:
    pytest.importorskip("autogen_agentchat")

    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts")
        for _ in range(3)
    ]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )
    researcher = make_agent("Researcher", "research")

    orchestrator = AutoGenOrchestrator()
    with pytest.raises(AgentException, match="max_rounds"):
        await orchestrator.arun(
            [supervisor_agent, researcher], "Explain RAG", strategy="supervisor", max_rounds=3
        )


@pytest.mark.asyncio
async def test_autogen_orchestrator_supervisor_unknown_worker_raises() -> None:
    pytest.importorskip("autogen_agentchat")

    decisions = [_SupervisorDecision(action="delegate", worker="Nonexistent", task="do X")]
    supervisor_agent = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    orchestrator = AutoGenOrchestrator()
    with pytest.raises(ConfigurationException, match="unknown"):
        await orchestrator.arun(
            [supervisor_agent, make_agent("Researcher", "research")], "task", strategy="supervisor"
        )


def test_autogen_orchestrator_rejects_unknown_strategy() -> None:
    pytest.importorskip("autogen_agentchat")

    orchestrator = AutoGenOrchestrator()
    with pytest.raises(ConfigurationException, match="sequential.*supervisor"):
        orchestrator.run(
            [make_agent("A", "a"), make_agent("B", "b")], "task", strategy="hierarchical"
        )


def test_autogen_orchestrator_requires_agents() -> None:
    orchestrator = AutoGenOrchestrator()
    with pytest.raises(ConfigurationException, match="no agents"):
        orchestrator.run([], "task")


def test_autogen_orchestrator_requires_input() -> None:
    orchestrator = AutoGenOrchestrator()
    with pytest.raises(ConfigurationException, match="input"):
        orchestrator.run([make_agent("A", "a")], None)
