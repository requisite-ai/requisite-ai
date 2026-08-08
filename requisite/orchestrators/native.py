"""
Native orchestrator: sequential, parallel, reflection, planner, and
supervisor execution -- no external orchestration framework required.

This is the default backend for :class:`~requisite.workflows.workflow.Workflow`.
Further strategies (debate, critic, consensus, hierarchical, map-reduce,
tree-of-thoughts, general graph execution) are natural extensions of this
class -- add a ``_run_<strategy>`` / ``_arun_<strategy>`` pair and a
branch in :meth:`NativeOrchestrator.run` / :meth:`NativeOrchestrator.arun`;
nothing else needs to change.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field

from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult

if TYPE_CHECKING:
    from requisite.agents.agent import Agent, AgentResult

logger = logging.getLogger("requisite.orchestrators.native")


class _PlanStep(BaseModel):
    """One step of a `planner`-strategy plan: a worker name + its subtask."""

    agent: str
    task: str


class _Plan(BaseModel):
    """Structured output requested from the coordinating agent in the `planner` strategy."""

    steps: list[_PlanStep] = Field(default_factory=list)


class _SupervisorDecision(BaseModel):
    """Structured output requested from the coordinating agent in the `supervisor` strategy."""

    action: Literal["delegate", "finish"]
    worker: Optional[str] = None
    task: Optional[str] = None
    final_answer: Optional[str] = None


def _planner_prompt(task: str, worker_names: Sequence[str]) -> str:
    return (
        "You are a planning agent. Break the following task into an ordered "
        "list of subtasks, each assigned to one of the available workers.\n\n"
        f"Task: {task}\n\n"
        f"Available workers: {', '.join(worker_names)}\n\n"
        "Respond with a plan: an ordered list of steps, each naming one worker "
        "(by its exact name above) and the subtask to give it."
    )


def _supervisor_prompt(
    task: str, worker_names: Sequence[str], transcript: Sequence[tuple[str, str, str]]
) -> str:
    lines = [
        "You are a supervising agent coordinating a team of workers to complete "
        "a task. Each round, either delegate a subtask to one worker or finish "
        "with a final answer once the task is complete.",
        "",
        f"Task: {task}",
        f"Available workers: {', '.join(worker_names)}",
    ]
    if transcript:
        lines.append("")
        lines.append("Delegations so far:")
        for worker, subtask, output in transcript:
            lines.append(f"- {worker} was asked: {subtask}")
            lines.append(f"  {worker} responded: {output}")
    lines.append("")
    lines.append(
        "Decide the next action: delegate (choose a worker and a task for them) "
        "or finish (provide the final answer)."
    )
    return "\n".join(lines)


def _reflection_critique_prompt(task: str, draft: str) -> str:
    return (
        f"Original task: {task}\n\n"
        f"Your previous answer:\n{draft}\n\n"
        "Critique your own answer above for accuracy, completeness, and clarity. "
        "If it is already excellent and needs no changes, respond with exactly: "
        "NO_CHANGES_NEEDED. Otherwise, list the specific issues to fix."
    )


def _reflection_revise_prompt(task: str, draft: str, critique: str) -> str:
    return (
        f"Original task: {task}\n\n"
        f"Your previous answer:\n{draft}\n\n"
        f"Critique of that answer:\n{critique}\n\n"
        "Provide an improved answer that addresses the critique. Respond with "
        "only the improved answer, not commentary about the revision."
    )


class NativeOrchestrator(BaseOrchestrator):
    """Runs agents sequentially or in parallel using plain Python -- no
    external orchestration framework required.

    Sequential strategy
        Each agent's output becomes the next agent's input, forming a
        pipeline (e.g. a "research" agent feeding a "writer" agent).

    Parallel strategy
        Every agent runs against the *same* input concurrently; their
        outputs are collected and combined.

    Reflection strategy
        A single agent (``steps[0]``) produces a draft, then critiques
        and revises its own output for up to ``max_rounds`` rounds,
        stopping early if it decides no changes are needed.

    Planner strategy
        The first agent (``steps[0]``) breaks the task into an ordered
        plan of subtasks assigned to the remaining agents (``steps[1:]``,
        addressed by name), which are then executed in order.

    Supervisor strategy
        The first agent (``steps[0]``) coordinates the remaining agents
        (``steps[1:]``, addressed by name), delegating one subtask at a
        time and deciding when the task is complete, for up to
        ``max_rounds`` rounds.
    """

    _STRATEGIES = ("sequential", "parallel", "reflection", "planner", "supervisor")

    @property
    def name(self) -> str:
        return "native"

    def run(
        self,
        steps: Sequence["Agent"],
        input: Optional[str],  # noqa: A002
        *,
        strategy: str = "sequential",
        **kwargs: Any,
    ) -> WorkflowResult:
        if not steps:
            raise ConfigurationException("Workflow has no agents; call workflow.add(agent) first.")
        if input is None:
            raise ConfigurationException("Workflow.run(...) requires an initial input/task.")

        if strategy == "sequential":
            return self._run_sequential(steps, input, **kwargs)
        if strategy == "parallel":
            return self._run_parallel(steps, input, **kwargs)
        if strategy == "reflection":
            return self._run_reflection(steps, input, **kwargs)
        if strategy == "planner":
            return self._run_planner(steps, input, **kwargs)
        if strategy == "supervisor":
            return self._run_supervisor(steps, input, **kwargs)
        raise ConfigurationException(
            f"Unknown execution strategy '{strategy}' for the native orchestrator. "
            f"Supported: {', '.join(self._STRATEGIES)}.",
        )

    async def arun(
        self,
        steps: Sequence["Agent"],
        input: Optional[str],  # noqa: A002
        *,
        strategy: str = "sequential",
        **kwargs: Any,
    ) -> WorkflowResult:
        if not steps:
            raise ConfigurationException("Workflow has no agents; call workflow.add(agent) first.")
        if input is None:
            raise ConfigurationException("Workflow.run(...) requires an initial input/task.")

        if strategy == "sequential":
            return await self._arun_sequential(steps, input, **kwargs)
        if strategy == "parallel":
            return await self._arun_parallel(steps, input, **kwargs)
        if strategy == "reflection":
            return await self._arun_reflection(steps, input, **kwargs)
        if strategy == "planner":
            return await self._arun_planner(steps, input, **kwargs)
        if strategy == "supervisor":
            return await self._arun_supervisor(steps, input, **kwargs)
        raise ConfigurationException(
            f"Unknown execution strategy '{strategy}' for the native orchestrator. "
            f"Supported: {', '.join(self._STRATEGIES)}.",
        )

    def _split_coordinator_and_workers(
        self, steps: Sequence["Agent"], *, role: str
    ) -> tuple["Agent", dict[str, "Agent"]]:
        """Split ``steps`` into a coordinating agent + name-addressed workers.

        Shared by the ``planner`` and ``supervisor`` strategies: ``steps[0]``
        is the coordinator, ``steps[1:]`` are workers it addresses by
        ``agent.name``.
        """
        if len(steps) < 2:
            raise ConfigurationException(
                f"The '{role}' strategy requires at least 2 agents: one {role} "
                f"(steps[0]) and one or more workers (steps[1:]). Got {len(steps)}.",
            )
        coordinator, workers = steps[0], list(steps[1:])
        worker_names = [worker.name for worker in workers]
        if len(set(worker_names)) != len(worker_names):
            raise ConfigurationException(
                f"The '{role}' strategy requires unique worker names; got {worker_names}.",
            )
        return coordinator, {worker.name: worker for worker in workers}

    def _run_sequential(
        self,
        steps: Sequence["Agent"],
        input: str,
        **kwargs: Any,  # noqa: A002
    ) -> WorkflowResult:
        results: list["AgentResult"] = []
        current_input = input
        for agent in steps:
            result = agent.run(current_input, **kwargs)
            results.append(result)
            current_input = result.content
        return WorkflowResult(
            content=results[-1].content if results else "",
            steps=results,
            orchestrator=self.name,
            strategy="sequential",
        )

    def _run_parallel(
        self,
        steps: Sequence["Agent"],
        input: str,
        **kwargs: Any,  # noqa: A002
    ) -> WorkflowResult:
        with ThreadPoolExecutor(max_workers=max(len(steps), 1)) as executor:
            futures = [executor.submit(agent.run, input, **kwargs) for agent in steps]
            results = [future.result() for future in futures]
        combined = "\n\n".join(f"[{r.agent_name}]\n{r.content}" for r in results)
        return WorkflowResult(
            content=combined, steps=results, orchestrator=self.name, strategy="parallel"
        )

    async def _arun_sequential(
        self,
        steps: Sequence["Agent"],
        input: str,
        **kwargs: Any,  # noqa: A002
    ) -> WorkflowResult:
        results: list["AgentResult"] = []
        current_input = input
        for agent in steps:
            result = await agent.arun(current_input, **kwargs)
            results.append(result)
            current_input = result.content
        return WorkflowResult(
            content=results[-1].content if results else "",
            steps=results,
            orchestrator=self.name,
            strategy="sequential",
        )

    async def _arun_parallel(
        self,
        steps: Sequence["Agent"],
        input: str,
        **kwargs: Any,  # noqa: A002
    ) -> WorkflowResult:
        results = list(await asyncio.gather(*(agent.arun(input, **kwargs) for agent in steps)))
        combined = "\n\n".join(f"[{r.agent_name}]\n{r.content}" for r in results)
        return WorkflowResult(
            content=combined, steps=results, orchestrator=self.name, strategy="parallel"
        )

    def _run_reflection(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 3,
        **kwargs: Any,
    ) -> WorkflowResult:
        if len(steps) != 1:
            raise ConfigurationException(
                f"The 'reflection' strategy requires exactly one agent (it critiques "
                f"and revises its own output). Got {len(steps)}.",
            )
        worker = steps[0]
        results: list["AgentResult"] = []

        draft = worker.run(input, **kwargs)
        results.append(draft)

        for _ in range(max_rounds - 1):
            critique = worker.run(_reflection_critique_prompt(input, draft.content), **kwargs)
            results.append(critique)
            if critique.content.strip() == "NO_CHANGES_NEEDED":
                break
            revised = worker.run(
                _reflection_revise_prompt(input, draft.content, critique.content), **kwargs
            )
            results.append(revised)
            draft = revised

        return WorkflowResult(
            content=draft.content, steps=results, orchestrator=self.name, strategy="reflection"
        )

    async def _arun_reflection(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 3,
        **kwargs: Any,
    ) -> WorkflowResult:
        if len(steps) != 1:
            raise ConfigurationException(
                f"The 'reflection' strategy requires exactly one agent (it critiques "
                f"and revises its own output). Got {len(steps)}.",
            )
        worker = steps[0]
        results: list["AgentResult"] = []

        draft = await worker.arun(input, **kwargs)
        results.append(draft)

        for _ in range(max_rounds - 1):
            critique = await worker.arun(
                _reflection_critique_prompt(input, draft.content), **kwargs
            )
            results.append(critique)
            if critique.content.strip() == "NO_CHANGES_NEEDED":
                break
            revised = await worker.arun(
                _reflection_revise_prompt(input, draft.content, critique.content), **kwargs
            )
            results.append(revised)
            draft = revised

        return WorkflowResult(
            content=draft.content, steps=results, orchestrator=self.name, strategy="reflection"
        )

    def _run_planner(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        **kwargs: Any,
    ) -> WorkflowResult:
        planner, workers = self._split_coordinator_and_workers(steps, role="planner")

        plan = planner.ai.chat(_planner_prompt(input, list(workers)), response_model=_Plan)
        self._validate_plan(plan, planner_name=planner.name, workers=workers)

        results: list["AgentResult"] = []
        context_notes: list[str] = []
        for plan_step in plan.steps:
            worker = workers[plan_step.agent]
            task_prompt = self._task_prompt_with_context(plan_step.task, context_notes)
            result = worker.run(task_prompt, **kwargs)
            results.append(result)
            context_notes.append(f"[{worker.name}] {result.content}")

        return WorkflowResult(
            content=results[-1].content if results else "",
            steps=results,
            orchestrator=self.name,
            strategy="planner",
        )

    async def _arun_planner(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        **kwargs: Any,
    ) -> WorkflowResult:
        planner, workers = self._split_coordinator_and_workers(steps, role="planner")

        plan = await planner.ai.achat(_planner_prompt(input, list(workers)), response_model=_Plan)
        self._validate_plan(plan, planner_name=planner.name, workers=workers)

        results: list["AgentResult"] = []
        context_notes: list[str] = []
        for plan_step in plan.steps:
            worker = workers[plan_step.agent]
            task_prompt = self._task_prompt_with_context(plan_step.task, context_notes)
            result = await worker.arun(task_prompt, **kwargs)
            results.append(result)
            context_notes.append(f"[{worker.name}] {result.content}")

        return WorkflowResult(
            content=results[-1].content if results else "",
            steps=results,
            orchestrator=self.name,
            strategy="planner",
        )

    @staticmethod
    def _validate_plan(plan: _Plan, *, planner_name: str, workers: dict[str, "Agent"]) -> None:
        if not plan.steps:
            raise ConfigurationException(
                f"Planner agent '{planner_name}' produced an empty plan.",
            )
        for plan_step in plan.steps:
            if plan_step.agent not in workers:
                raise ConfigurationException(
                    f"Planner agent '{planner_name}' assigned a step to unknown worker "
                    f"'{plan_step.agent}'. Available workers: {sorted(workers)}.",
                )

    @staticmethod
    def _task_prompt_with_context(task: str, context_notes: Sequence[str]) -> str:
        if not context_notes:
            return task
        return f"{task}\n\nContext from previous steps:\n" + "\n".join(context_notes)

    def _run_supervisor(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 6,
        **kwargs: Any,
    ) -> WorkflowResult:
        supervisor, workers = self._split_coordinator_and_workers(steps, role="supervisor")

        results: list["AgentResult"] = []
        transcript: list[tuple[str, str, str]] = []

        for _ in range(max_rounds):
            decision = supervisor.ai.chat(
                _supervisor_prompt(input, list(workers), transcript),
                response_model=_SupervisorDecision,
            )
            if decision.action == "finish":
                return WorkflowResult(
                    content=decision.final_answer or "",
                    steps=results,
                    orchestrator=self.name,
                    strategy="supervisor",
                )

            worker = self._resolve_delegate(
                decision, supervisor_name=supervisor.name, workers=workers
            )
            subtask = decision.task or input
            result = worker.run(subtask, **kwargs)
            results.append(result)
            transcript.append((worker.name, subtask, result.content))

        raise AgentException(
            f"Workflow supervisor '{supervisor.name}' exceeded max_rounds={max_rounds} "
            f"without reaching a final answer.",
        )

    async def _arun_supervisor(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 6,
        **kwargs: Any,
    ) -> WorkflowResult:
        supervisor, workers = self._split_coordinator_and_workers(steps, role="supervisor")

        results: list["AgentResult"] = []
        transcript: list[tuple[str, str, str]] = []

        for _ in range(max_rounds):
            decision = await supervisor.ai.achat(
                _supervisor_prompt(input, list(workers), transcript),
                response_model=_SupervisorDecision,
            )
            if decision.action == "finish":
                return WorkflowResult(
                    content=decision.final_answer or "",
                    steps=results,
                    orchestrator=self.name,
                    strategy="supervisor",
                )

            worker = self._resolve_delegate(
                decision, supervisor_name=supervisor.name, workers=workers
            )
            subtask = decision.task or input
            result = await worker.arun(subtask, **kwargs)
            results.append(result)
            transcript.append((worker.name, subtask, result.content))

        raise AgentException(
            f"Workflow supervisor '{supervisor.name}' exceeded max_rounds={max_rounds} "
            f"without reaching a final answer.",
        )

    @staticmethod
    def _resolve_delegate(
        decision: _SupervisorDecision, *, supervisor_name: str, workers: dict[str, "Agent"]
    ) -> "Agent":
        if decision.worker not in workers:
            raise ConfigurationException(
                f"Supervisor agent '{supervisor_name}' tried to delegate to unknown "
                f"worker '{decision.worker}'. Available workers: {sorted(workers)}.",
            )
        return workers[decision.worker]
