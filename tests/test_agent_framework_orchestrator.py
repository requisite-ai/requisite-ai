"""Unit tests for :class:`requisite.orchestrators.agent_framework_orchestrator.AgentFrameworkOrchestrator`.

Real ``agent-framework-core`` coordination code (its actual
``WorkflowBuilder`` engine) runs for real when installed
(``pytest.importorskip``) -- only the wrapped Requisite ``Agent``'s
provider is faked, so no real network/LLM call happens.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import pytest

from requisite.core.exceptions import AgentException, ConfigurationException, ProviderException
from requisite.orchestrators.agent_framework_orchestrator import AgentFrameworkOrchestrator
from requisite.orchestrators.native import _SupervisorDecision
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


def test_agent_framework_without_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in list(sys.modules):
        if module_name == "agent_framework" or module_name.startswith("agent_framework."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setitem(sys.modules, "agent_framework", None)

    with pytest.raises(ConfigurationException, match="agent-framework-core"):
        AgentFrameworkOrchestrator().run([make_agent("A", "a")], "task")


@pytest.mark.asyncio
async def test_agent_framework_sequential_runs_as_a_real_workflow() -> None:
    pytest.importorskip("agent_framework")

    result = await AgentFrameworkOrchestrator().arun(
        [make_agent("Researcher", "research"), make_agent("Writer", "write")], "Explain RAG"
    )

    assert result.orchestrator == "agent_framework"
    assert result.strategy == "sequential"
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    # The framework hands the writer the researcher's reply as an *assistant*
    # message, not a new user message -- the adapter must rebuild that into the
    # writer's task, or the handoff would silently vanish.
    assert "Explain RAG" in result.steps[1].content
    assert "[Researcher] research:Explain RAG" in result.steps[1].content
    assert result.content == result.steps[1].content


def test_agent_framework_run_sync_wraps_arun() -> None:
    pytest.importorskip("agent_framework")

    result = AgentFrameworkOrchestrator().run(
        [make_agent("Researcher", "research"), make_agent("Writer", "write")], "Explain RAG"
    )

    assert len(result.steps) == 2


@pytest.mark.asyncio
async def test_agent_framework_third_step_sees_every_earlier_agent() -> None:
    pytest.importorskip("agent_framework")

    result = await AgentFrameworkOrchestrator().arun(
        [make_agent("A", "a"), make_agent("B", "b"), make_agent("C", "c")], "go"
    )

    assert [step.agent_name for step in result.steps] == ["A", "B", "C"]
    assert "[A]" in result.steps[2].content and "[B]" in result.steps[2].content


@pytest.mark.asyncio
async def test_agent_framework_sequential_allows_steps_that_share_a_name() -> None:
    pytest.importorskip("agent_framework")

    result = await AgentFrameworkOrchestrator().arun(
        [make_agent("Same", "one"), make_agent("Same", "two")], "go"
    )

    assert [step.agent_name for step in result.steps] == ["Same", "Same"]


@pytest.mark.asyncio
async def test_agent_framework_does_not_warn_about_function_invoking(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The proxy client mixes in FunctionInvocationLayer so the framework
    doesn't log 'chat client does not support function invoking' per agent."""
    pytest.importorskip("agent_framework")

    with caplog.at_level(logging.WARNING):
        await AgentFrameworkOrchestrator().arun([make_agent("A", "a")], "go")

    assert "function invoking" not in caplog.text


@pytest.mark.asyncio
async def test_agent_framework_supervisor_routes_to_both_workers_then_finishes() -> None:
    pytest.importorskip("agent_framework")

    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts about RAG"),
        _SupervisorDecision(action="delegate", worker="Writer", task="Draft a summary"),
        _SupervisorDecision(action="finish", final_answer="RAG combines retrieval and generation."),
    ]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    result = await AgentFrameworkOrchestrator().arun(
        [supervisor, make_agent("Researcher", "research"), make_agent("Writer", "write")],
        "Explain RAG",
        strategy="supervisor",
    )

    assert result.orchestrator == "agent_framework"
    assert result.strategy == "supervisor"
    assert result.content == "RAG combines retrieval and generation."
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    assert result.steps[0].content == "research:Find facts about RAG"


@pytest.mark.asyncio
async def test_agent_framework_supervisor_exceeding_max_rounds_raises_agent_exception() -> None:
    pytest.importorskip("agent_framework")

    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="x") for _ in range(3)
    ]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    with pytest.raises(AgentException, match="max_rounds"):
        await AgentFrameworkOrchestrator().arun(
            [supervisor, make_agent("Researcher", "research")],
            "task",
            strategy="supervisor",
            max_rounds=3,
        )


@pytest.mark.asyncio
async def test_agent_framework_step_failure_surfaces_the_real_provider_exception() -> None:
    pytest.importorskip("agent_framework")

    with pytest.raises(ProviderException, match="upstream exploded"):
        await AgentFrameworkOrchestrator().arun(
            [make_agent("A", "a"), make_agent_with_provider("B", _FailingProvider())], "go"
        )


def test_agent_framework_rejects_unknown_strategy() -> None:
    pytest.importorskip("agent_framework")

    with pytest.raises(ConfigurationException, match="sequential.*supervisor"):
        AgentFrameworkOrchestrator().run(
            [make_agent("A", "a"), make_agent("B", "b")], "task", strategy="hierarchical"
        )


def test_agent_framework_requires_agents() -> None:
    with pytest.raises(ConfigurationException, match="no agents"):
        AgentFrameworkOrchestrator().run([], "task")


def test_agent_framework_requires_input() -> None:
    with pytest.raises(ConfigurationException, match="input"):
        AgentFrameworkOrchestrator().run([make_agent("A", "a")], None)
