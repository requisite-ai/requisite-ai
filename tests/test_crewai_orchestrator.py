"""Unit tests for :class:`requisite.orchestrators.crewai_orchestrator.CrewAIOrchestrator`.

Real ``crewai`` coordination code runs for real when installed
(``pytest.importorskip``) -- only the wrapped Requisite ``Agent``'s
provider is faked (``EchoProvider``), so no real network/LLM call
happens, consistent with the framework's no-network-in-tests rule.
"""

from __future__ import annotations

import sys

import pytest

from requisite.core.exceptions import ConfigurationException
from requisite.orchestrators.crewai_orchestrator import CrewAIOrchestrator
from tests.test_workflows import EchoProvider, make_agent, make_agent_with_provider  # noqa: F401


def test_crewai_without_dependency_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "crewai" or module_name.startswith("crewai."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setitem(sys.modules, "crewai", None)

    orchestrator = CrewAIOrchestrator()
    with pytest.raises(ConfigurationException, match="crewai"):
        orchestrator.run([make_agent("A", "a")], "task")


def test_crewai_orchestrator_runs_a_real_sequential_pipeline() -> None:
    pytest.importorskip("crewai")

    researcher = make_agent("Researcher", "research")
    writer = make_agent("Writer", "write")

    orchestrator = CrewAIOrchestrator()
    result = orchestrator.run([researcher, writer], "Explain RAG")

    assert result.orchestrator == "crewai"
    assert result.strategy == "sequential"
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    # The writer's step ran with the researcher's real output threaded in via
    # CrewAI's own context=[previous_task] chaining, not manual state passing.
    assert "research:" in result.steps[1].content
    assert "Explain RAG" in result.steps[1].content


@pytest.mark.asyncio
async def test_crewai_orchestrator_arun_real_sequential_pipeline() -> None:
    pytest.importorskip("crewai")

    researcher = make_agent("Researcher", "research")
    writer = make_agent("Writer", "write")

    orchestrator = CrewAIOrchestrator()
    result = await orchestrator.arun([researcher, writer], "Explain RAG")

    assert result.orchestrator == "crewai"
    assert len(result.steps) == 2


def test_crewai_orchestrator_rejects_hierarchical_strategy() -> None:
    pytest.importorskip("crewai")

    orchestrator = CrewAIOrchestrator()
    with pytest.raises(ConfigurationException, match="sequential"):
        orchestrator.run(
            [make_agent("A", "a"), make_agent("B", "b")], "task", strategy="hierarchical"
        )


def test_crewai_orchestrator_requires_agents() -> None:
    orchestrator = CrewAIOrchestrator()
    with pytest.raises(ConfigurationException, match="no agents"):
        orchestrator.run([], "task")


def test_crewai_orchestrator_requires_input() -> None:
    orchestrator = CrewAIOrchestrator()
    with pytest.raises(ConfigurationException, match="input"):
        orchestrator.run([make_agent("A", "a")], None)
