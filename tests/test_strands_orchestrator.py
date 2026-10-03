"""Unit tests for :class:`requisite.orchestrators.strands_orchestrator.StrandsOrchestrator`.

Real ``strands-agents`` coordination code (its actual ``Graph`` executor)
runs for real when installed (``pytest.importorskip``) -- only the wrapped
Requisite ``Agent``'s provider is faked, so no real network/LLM call
happens.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from requisite.core.exceptions import AgentException, ConfigurationException, ProviderException
from requisite.orchestrators.native import _SupervisorDecision
from requisite.orchestrators.strands_orchestrator import StrandsOrchestrator
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


def test_strands_without_dependency_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "strands" or module_name.startswith("strands."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setitem(sys.modules, "strands", None)

    with pytest.raises(ConfigurationException, match="strands-agents"):
        StrandsOrchestrator().run([make_agent("A", "a")], "task")


@pytest.mark.asyncio
async def test_strands_sequential_runs_as_a_real_graph() -> None:
    pytest.importorskip("strands.multiagent")

    result = await StrandsOrchestrator().arun(
        [make_agent("Researcher", "research"), make_agent("Writer", "write")], "Explain RAG"
    )

    assert result.orchestrator == "strands"
    assert result.strategy == "sequential"
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    # Strands' own graph executor assembled the writer's input from the
    # researcher's real output, labelled with the researcher's real name.
    assert "From Researcher" in result.steps[1].content
    assert "research:Explain RAG" in result.steps[1].content
    assert result.content == result.steps[1].content


def test_strands_run_sync_wraps_arun() -> None:
    pytest.importorskip("strands.multiagent")

    result = StrandsOrchestrator().run(
        [make_agent("Researcher", "research"), make_agent("Writer", "write")], "Explain RAG"
    )

    assert len(result.steps) == 2


@pytest.mark.asyncio
async def test_strands_sequential_allows_steps_that_share_a_name() -> None:
    pytest.importorskip("strands.multiagent")

    result = await StrandsOrchestrator().arun(
        [make_agent("Same", "one"), make_agent("Same", "two")], "go"
    )

    assert [step.agent_name for step in result.steps] == ["Same", "Same"]


@pytest.mark.asyncio
async def test_strands_does_not_print_agent_responses_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Strands' default callback handler prints every streamed response; the
    orchestrator opts out so it doesn't duplicate Requisite's own output."""
    pytest.importorskip("strands.multiagent")

    await StrandsOrchestrator().arun([make_agent("A", "a"), make_agent("B", "b")], "go")

    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_strands_supervisor_routes_to_both_workers_then_finishes() -> None:
    pytest.importorskip("strands.multiagent")

    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts about RAG"),
        _SupervisorDecision(action="delegate", worker="Writer", task="Draft a summary"),
        _SupervisorDecision(action="finish", final_answer="RAG combines retrieval and generation."),
    ]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    result = await StrandsOrchestrator().arun(
        [supervisor, make_agent("Researcher", "research"), make_agent("Writer", "write")],
        "Explain RAG",
        strategy="supervisor",
    )

    assert result.orchestrator == "strands"
    assert result.strategy == "supervisor"
    assert result.content == "RAG combines retrieval and generation."
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    assert result.steps[0].content == "research:Find facts about RAG"


@pytest.mark.asyncio
async def test_strands_supervisor_exceeding_max_rounds_raises_agent_exception() -> None:
    pytest.importorskip("strands.multiagent")

    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="x") for _ in range(3)
    ]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    with pytest.raises(AgentException, match="max_rounds"):
        await StrandsOrchestrator().arun(
            [supervisor, make_agent("Researcher", "research")],
            "task",
            strategy="supervisor",
            max_rounds=3,
        )


@pytest.mark.asyncio
async def test_strands_graph_step_failure_surfaces_the_real_provider_exception() -> None:
    """A Graph reports node failures on a result object rather than
    necessarily raising -- pin that the real exception type reaches the caller."""
    pytest.importorskip("strands.multiagent")

    with pytest.raises(ProviderException, match="upstream exploded"):
        await StrandsOrchestrator().arun(
            [make_agent("A", "a"), make_agent_with_provider("B", _FailingProvider())], "go"
        )


def test_strands_rejects_unknown_strategy() -> None:
    pytest.importorskip("strands.multiagent")

    with pytest.raises(ConfigurationException, match="sequential.*supervisor"):
        StrandsOrchestrator().run(
            [make_agent("A", "a"), make_agent("B", "b")], "task", strategy="hierarchical"
        )


def test_strands_requires_agents() -> None:
    with pytest.raises(ConfigurationException, match="no agents"):
        StrandsOrchestrator().run([], "task")


def test_strands_requires_input() -> None:
    with pytest.raises(ConfigurationException, match="input"):
        StrandsOrchestrator().run([make_agent("A", "a")], None)
