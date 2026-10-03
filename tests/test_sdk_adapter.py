"""Unit tests for the SDK-independent half of the agent-SDK orchestrator backends.

``requisite.orchestrators._sdk_adapter`` (the shared supervisor loop and
step plumbing) and each backend's pure task-text extraction helper need no
third-party SDK, so -- unlike the per-backend test files, which
``pytest.importorskip`` their SDK -- these always run, including in CI,
where the opt-in SDKs aren't installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.orchestrators._sdk_adapter import (
    SdkDelegate,
    check_run_inputs,
    collect_steps,
    run_supervisor,
    unsupported_strategy,
)
from requisite.orchestrators.native import _SupervisorDecision
from tests.test_workflows import (
    ScriptedSupervisorProvider,
    make_agent,
    make_agent_with_provider,
)


def _delegate_for(agent: Any) -> SdkDelegate:
    """A delegate that 'runs through an SDK' by calling the real Requisite agent."""
    results: list[Any] = []

    async def _run(task: str) -> None:
        results.append(await agent.arun(task))

    return SdkDelegate(agent.name, _run, results)


def test_check_run_inputs_requires_agents_and_input() -> None:
    with pytest.raises(ConfigurationException, match="no agents"):
        check_run_inputs([], "task")
    with pytest.raises(ConfigurationException, match="input"):
        check_run_inputs([make_agent("A", "a")], None)
    check_run_inputs([make_agent("A", "a")], "task")  # no error


def test_unsupported_strategy_names_the_backend_and_both_supported_strategies() -> None:
    error = unsupported_strategy("strands", "debate")

    assert isinstance(error, ConfigurationException)
    assert "strands" in str(error) and "debate" in str(error)
    assert "sequential" in str(error) and "supervisor" in str(error)


def test_collect_steps_flattens_in_order() -> None:
    first = SimpleNamespace(_collected_results=["a1", "a2"])
    second = SimpleNamespace(_collected_results=["b1"])

    assert collect_steps([first, second]) == ["a1", "a2", "b1"]


@pytest.mark.asyncio
async def test_sdk_delegate_returns_the_result_recorded_during_the_run() -> None:
    delegate = _delegate_for(make_agent("Researcher", "research"))

    result = await delegate.arun("Find facts")

    assert delegate.name == "Researcher"
    assert result.content == "research:Find facts"


@pytest.mark.asyncio
async def test_sdk_delegate_raises_if_the_sdk_never_called_the_wrapped_agent() -> None:
    async def _run_without_calling_the_agent(task: str) -> None:
        return None

    delegate = SdkDelegate("Ghost", _run_without_calling_the_agent, [])

    with pytest.raises(AgentException, match="Ghost"):
        await delegate.arun("task")


@pytest.mark.asyncio
async def test_run_supervisor_delegates_then_finishes_and_labels_the_backend() -> None:
    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="Find facts about RAG"),
        _SupervisorDecision(action="delegate", worker="Writer", task="Draft a summary"),
        _SupervisorDecision(action="finish", final_answer="RAG combines retrieval and generation."),
    ]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    result = await run_supervisor(
        backend="some_sdk",
        steps=[supervisor, make_agent("Researcher", "research"), make_agent("Writer", "write")],
        input="Explain RAG",
        max_rounds=6,
        build_delegate=_delegate_for,
    )

    assert result.orchestrator == "some_sdk"  # not "native", whose loop it reuses
    assert result.strategy == "supervisor"
    assert result.content == "RAG combines retrieval and generation."
    assert [step.agent_name for step in result.steps] == ["Researcher", "Writer"]
    assert result.steps[0].content == "research:Find facts about RAG"


@pytest.mark.asyncio
async def test_run_supervisor_exceeding_max_rounds_raises_the_real_exception_type() -> None:
    decisions = [
        _SupervisorDecision(action="delegate", worker="Researcher", task="x") for _ in range(3)
    ]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    with pytest.raises(AgentException, match="max_rounds"):
        await run_supervisor(
            backend="some_sdk",
            steps=[supervisor, make_agent("Researcher", "research")],
            input="task",
            max_rounds=3,
            build_delegate=_delegate_for,
        )


@pytest.mark.asyncio
async def test_run_supervisor_unknown_worker_raises_the_real_exception_type() -> None:
    decisions = [_SupervisorDecision(action="delegate", worker="Nobody", task="x")]
    supervisor = make_agent_with_provider(
        "Supervisor", ScriptedSupervisorProvider(decisions=decisions)
    )

    with pytest.raises(ConfigurationException, match="unknown"):
        await run_supervisor(
            backend="some_sdk",
            steps=[supervisor, make_agent("Researcher", "research")],
            input="task",
            max_rounds=6,
            build_delegate=_delegate_for,
        )


@pytest.mark.asyncio
async def test_run_supervisor_needs_a_coordinator_and_at_least_one_worker() -> None:
    with pytest.raises(ConfigurationException, match="at least 2 agents"):
        await run_supervisor(
            backend="some_sdk",
            steps=[make_agent("Only", "o")],
            input="task",
            max_rounds=6,
            build_delegate=_delegate_for,
        )


# --- pure task-text extraction, one per backend ---------------------------------


def test_openai_agents_extract_task_text_handles_every_input_shape() -> None:
    from requisite.orchestrators.openai_agents_orchestrator import _extract_task_text

    assert _extract_task_text("plain") == "plain"
    assert _extract_task_text([{"role": "user", "content": "from a dict"}]) == "from a dict"
    assert (
        _extract_task_text(
            [{"role": "user", "content": [{"type": "input_text", "text": "part one, "}]}]
        )
        == "part one, "
    )
    obj = SimpleNamespace(role="user", content=[SimpleNamespace(text="from objects")])
    assert _extract_task_text([obj]) == "from objects"
    # The last *user* item wins; an assistant item after it is ignored.
    items = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ignored"},
    ]
    assert _extract_task_text(items) == "second"
    assert _extract_task_text([{"role": "assistant", "content": "only assistant"}]) == ""


def test_strands_extract_task_text_joins_text_blocks_of_the_last_user_message() -> None:
    from requisite.orchestrators.strands_orchestrator import _extract_task_text

    messages = [
        {"role": "user", "content": [{"text": "old"}]},
        {"role": "assistant", "content": [{"text": "reply"}]},
        {
            "role": "user",
            "content": [{"text": "Original Task: go. "}, {"toolUse": {}}, {"text": "More"}],
        },
    ]

    assert _extract_task_text(messages) == "Original Task: go. More"
    assert _extract_task_text([{"role": "assistant", "content": [{"text": "x"}]}]) == ""


def test_agent_framework_extract_task_text_folds_in_earlier_agents_as_context() -> None:
    from requisite.orchestrators.agent_framework_orchestrator import _extract_task_text

    def message(role: str, text: str, author: str | None = None) -> Any:
        return SimpleNamespace(role=role, text=text, author_name=author)

    # What the framework hands the second agent: the original task, then the
    # first agent's reply as an *assistant* message -- not a new user message.
    transcript = [
        message("user", "Explain RAG"),
        message("assistant", "bullet points", "Researcher"),
    ]

    prompt = _extract_task_text(transcript)

    assert prompt.startswith("Explain RAG")
    assert "[Researcher] bullet points" in prompt
    # Same "task + context from previous steps" shape the native planner produces.
    assert "Context from previous steps" in prompt

    # With no earlier agent, the task passes through untouched.
    assert _extract_task_text([message("user", "Explain RAG")]) == "Explain RAG"
    assert _extract_task_text([message("assistant", "stray", "X")]) == ""
