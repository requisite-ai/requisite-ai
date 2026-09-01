"""
Native orchestrator: sequential, parallel, reflection, planner,
supervisor, critic, consensus, debate, map-reduce, hierarchical,
tree-of-thoughts, reflexion, and graph execution -- no external
orchestration framework required.

This is the default backend for :class:`~requisite.workflows.workflow.Workflow`.
Further strategies are natural extensions of this class -- add a
``_run_<strategy>`` / ``_arun_<strategy>`` pair and a branch in
:meth:`NativeOrchestrator.run` / :meth:`NativeOrchestrator.arun`; nothing
else needs to change. See
``docs/adr/0011-critic-and-consensus-strategies.md``,
``docs/adr/0012-debate-and-map-reduce-strategies.md``,
``docs/adr/0013-hierarchical-strategy.md``,
``docs/adr/0018-tree-of-thoughts-strategy.md``, and
``docs/adr/0019-graph-execution-strategy.md`` for which of those fit this
flat coordinator/worker shape directly and which don't.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field

from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.orchestrators.base import (
    END,
    BaseOrchestrator,
    EvaluationResult,
    Evaluator,
    WorkflowResult,
)

if TYPE_CHECKING:
    from requisite.agents.agent import Agent, AgentResult

logger = logging.getLogger("requisite.orchestrators.native")


async def _gather_waiting_for_all(*coroutines: Any) -> list[Any]:
    """Like ``asyncio.gather(*coroutines)``, but guarantees every
    coroutine actually finishes (success or failure) before this
    returns or raises.

    Plain ``asyncio.gather(...)`` (the default, ``return_exceptions=False``)
    raises as soon as the first coroutine fails, leaving the others
    running as orphaned, unattended background tasks -- real
    concurrently-running agent/tool calls the caller has no handle on
    and can't observe if they also fail. Used by every concurrent
    strategy (``parallel``, ``consensus``, one round of ``debate``,
    ``map_reduce``, ``tree_of_thoughts``) in place of a bare
    ``asyncio.gather`` call. See
    ``docs/adr/0031-code-review-fixes.md``.

    Re-raises the first exception, in submission order (not completion
    order) -- once every coroutine has completed -- preserving the
    pre-existing "any single failure fails the whole call" behavior,
    just without the background leak.
    """
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


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


class _ThoughtScore(BaseModel):
    """One candidate's score in the `tree_of_thoughts` strategy's listwise evaluation."""

    index: int
    score: float = Field(ge=0.0, le=10.0)
    finished: bool = False


class _ThoughtEvaluation(BaseModel):
    """Structured output requested from the evaluating agent in the `tree_of_thoughts`
    strategy -- scores every candidate thought generated this level in one call,
    mirroring :class:`~requisite.rag.reranker.LLMReranker`'s listwise re-ranking shape.
    """

    scores: list[_ThoughtScore] = Field(default_factory=list)


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


def _critic_prompt(task: str, draft: str) -> str:
    return (
        f"Task: {task}\n\n"
        f"Proposed answer:\n{draft}\n\n"
        "Critique the answer above for accuracy, completeness, and clarity. "
        "If it is already excellent and needs no changes, respond with exactly: "
        "NO_CHANGES_NEEDED. Otherwise, list the specific issues to fix."
    )


def _reflexion_default_evaluation_prompt(task: str, attempt: str) -> str:
    return (
        f"Task: {task}\n\nAttempt:\n{attempt}\n\n"
        "Judge whether this attempt fully and correctly solves the task. "
        "Set success=True only if it does; otherwise set success=False and "
        "explain specifically what is wrong or missing in feedback."
    )


def _reflexion_reflect_prompt(task: str, attempt: str, feedback: str) -> str:
    return (
        f"Task: {task}\n\nYour attempt:\n{attempt}\n\n"
        f"Evaluator feedback:\n{feedback}\n\n"
        "Write a short, specific, actionable lesson for your next attempt at "
        "this same task -- what to do differently, not a restatement of the "
        "feedback."
    )


def _debate_prompt(
    task: str,
    debater_names: Sequence[str],
    transcript: dict[str, list[str]],
    *,
    agent_name: str,
    round_num: int,
) -> str:
    lines = [
        f"You are '{agent_name}', one of several agents debating the task below. "
        "Consider the other participants' arguments and respond -- agree, disagree, "
        "or refine your own position, giving your reasoning.",
        "",
        f"Task: {task}",
    ]
    if round_num == 0:
        lines.append("")
        lines.append("This is the first round. State your initial position.")
    else:
        lines.append("")
        lines.append("Arguments so far:")
        for name in debater_names:
            for i, argument in enumerate(transcript[name], start=1):
                lines.append(f"- {name} (round {i}): {argument}")
        lines.append("")
        lines.append("Respond with your updated position for this round.")
    return "\n".join(lines)


def _debate_verdict_prompt(
    task: str, debater_names: Sequence[str], transcript: dict[str, list[str]]
) -> str:
    lines = [
        "The following agents debated the task below over several rounds. Review "
        "the full debate and provide a final, well-reasoned answer -- resolve "
        "disagreements by weighing the strongest arguments, not just picking a side.",
        "",
        f"Task: {task}",
        "",
        "Debate transcript:",
    ]
    for name in debater_names:
        for i, argument in enumerate(transcript[name], start=1):
            lines.append(f"- {name} (round {i}): {argument}")
    lines.append("")
    lines.append("Provide the final answer only.")
    return "\n".join(lines)


def _map_prompt(task: str, item: str) -> str:
    return (
        f"Overall task: {task}\n\n"
        f"Process the following individual item as part of that task:\n{item}\n\n"
        "Respond with your result for this item only."
    )


def _reduce_prompt(task: str, items: Sequence[str], results: Sequence["AgentResult"]) -> str:
    lines = [f"Overall task: {task}", "", "Individual results from processing each item:"]
    for item, result in zip(items, results, strict=True):
        lines.append(f"- Input: {item}")
        lines.append(f"  Result: {result.content}")
    lines.append("")
    lines.append(
        "Combine these individual results into a single final answer for the overall task."
    )
    return "\n".join(lines)


def _consensus_prompt(task: str, results: Sequence["AgentResult"]) -> str:
    lines = [
        "Several independent responses to the same task are shown below. "
        "Synthesize them into a single, well-reasoned final answer -- resolve "
        "any disagreements by weighing the strongest reasoning, not just "
        "picking one response.",
        "",
        f"Task: {task}",
        "",
        "Independent responses:",
    ]
    for result in results:
        lines.append(f"- {result.agent_name}: {result.content}")
    lines.append("")
    lines.append("Provide the final, synthesized answer only.")
    return "\n".join(lines)


def _tot_thinker_prompt(task: str, path: Sequence[str]) -> str:
    lines = [f"Task: {task}", ""]
    if path:
        lines.append("Reasoning steps so far, in order:")
        for i, thought in enumerate(path, start=1):
            lines.append(f"{i}. {thought}")
        lines.append("")
        lines.append(
            "Propose exactly one next reasoning step that continues toward solving "
            "the task. If the steps so far already fully solve the task, state the "
            "complete final answer instead."
        )
    else:
        lines.append("Propose exactly one first reasoning step toward solving the task.")
    lines.append("Respond with that single step only -- no commentary about the process.")
    return "\n".join(lines)


def _tot_evaluation_prompt(task: str, candidates: Sequence[Sequence[str]]) -> str:
    lines = [
        "Several candidate next reasoning steps for the task below were generated "
        "independently. Score each candidate from 0 (a dead end or incorrect) to "
        "10 (clearly correct and promising), based on its full path so far. Mark "
        "`finished=true` only for a candidate whose path already constitutes a "
        "complete, correct final answer to the task.",
        "",
        f"Task: {task}",
        "",
    ]
    for index, path in enumerate(candidates):
        lines.append(f"Candidate {index}:")
        for i, thought in enumerate(path, start=1):
            lines.append(f"  {i}. {thought}")
        lines.append("")
    lines.append("Score every candidate listed above, identified by its index.")
    return "\n".join(lines)


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

    Critic strategy
        Two agents: a generator (``steps[0]``) produces a draft, a
        separate critic (``steps[1]``) critiques it, and the generator
        revises -- for up to ``max_rounds`` rounds, stopping early if the
        critic decides no changes are needed. Same shape as reflection,
        generalized to two distinct agents instead of one agent
        critiquing itself.

    Consensus strategy
        The first agent (``steps[0]``) is a synthesizer; the remaining
        agents (``steps[1:]``) independently answer the same input
        concurrently, then the synthesizer combines their answers into
        one final response.

    Debate strategy
        The first agent (``steps[0]``) is a moderator; the remaining
        agents (``steps[1:]``) debate the input over ``max_rounds``
        rounds, each round seeing every debater's arguments from the
        *previous* round, after which the moderator delivers a final
        verdict.

    Map-reduce strategy
        The first agent (``steps[0]``) is a reducer; the remaining
        agents (``steps[1:]``) each process one item from ``map_items``
        (assigned round-robin, run concurrently), after which the
        reducer combines every item's result into one final answer.

    Hierarchical strategy
        Same shape as ``supervisor``, except a delegate (``steps[1:]``)
        may be either an ``Agent`` or a named
        :class:`~requisite.workflows.workflow.Workflow` ("team").
        Delegating to a ``Workflow`` runs whatever strategy it's
        configured with -- including another ``supervisor`` or
        ``hierarchical`` -- giving real recursive delegation.

    Tree-of-thoughts strategy
        The first agent (``steps[0]``) evaluates candidate reasoning
        steps; the remaining agents (``steps[1:]``) generate them,
        assigned round-robin. Each level, ``breadth`` candidates are
        generated per surviving path, all scored together in one
        structured call, and pruned to the top ``beam_width`` -- for up
        to ``max_depth`` levels, stopping early if any candidate is
        scored as a complete final answer. See
        ``docs/adr/0018-tree-of-thoughts-strategy.md``.

    Reflexion strategy
        A single agent (``steps[0]``) attempts the task from scratch, is
        scored by an ``evaluator`` callback (``(task, attempt) ->
        EvaluationResult``; falls back to the same agent judging itself
        via structured output if none is given), and, on failure, writes
        a self-reflection lesson that's folded into the *next* attempt's
        prompt -- for up to ``max_trials`` independent trials, stopping
        as soon as the evaluator reports success. Unlike ``reflection``,
        each trial re-attempts the whole task rather than revising the
        previous draft. See ``docs/adr/0036-reflexion-strategy.md``.

    Graph strategy
        An arbitrary graph of nodes (``steps``, each an ``Agent`` or a
        named ``Workflow`` "team") wired together with edges declared
        ahead of time via ``Workflow.add_edge(...)``. Unlike every
        strategy above, routing isn't decided by an LLM at run time --
        the first node (``steps[0]``) is the entry point, then each
        node's output content is checked against its outgoing edges (in
        the order they were declared) to pick the next node
        deterministically. Cycles are allowed, bounded by ``max_steps``
        (default 25). See ``docs/adr/0019-graph-execution-strategy.md``.
    """

    _STRATEGIES = (
        "sequential",
        "parallel",
        "reflection",
        "planner",
        "supervisor",
        "critic",
        "consensus",
        "debate",
        "map_reduce",
        "tree_of_thoughts",
        "hierarchical",
        "reflexion",
        "graph",
    )

    @property
    def name(self) -> str:
        return "native"

    def run(
        self,
        steps: Sequence[Any],
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
        if strategy == "critic":
            return self._run_critic(steps, input, **kwargs)
        if strategy == "consensus":
            return self._run_consensus(steps, input, **kwargs)
        if strategy == "debate":
            return self._run_debate(steps, input, **kwargs)
        if strategy == "map_reduce":
            return self._run_map_reduce(steps, input, **kwargs)
        if strategy == "tree_of_thoughts":
            return self._run_tree_of_thoughts(steps, input, **kwargs)
        if strategy == "hierarchical":
            return self._run_hierarchical(steps, input, **kwargs)
        if strategy == "reflexion":
            return self._run_reflexion(steps, input, **kwargs)
        if strategy == "graph":
            return self._run_graph(steps, input, **kwargs)
        raise ConfigurationException(
            f"Unknown execution strategy '{strategy}' for the native orchestrator. "
            f"Supported: {', '.join(self._STRATEGIES)}.",
        )

    async def arun(
        self,
        steps: Sequence[Any],
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
        if strategy == "critic":
            return await self._arun_critic(steps, input, **kwargs)
        if strategy == "consensus":
            return await self._arun_consensus(steps, input, **kwargs)
        if strategy == "debate":
            return await self._arun_debate(steps, input, **kwargs)
        if strategy == "map_reduce":
            return await self._arun_map_reduce(steps, input, **kwargs)
        if strategy == "tree_of_thoughts":
            return await self._arun_tree_of_thoughts(steps, input, **kwargs)
        if strategy == "hierarchical":
            return await self._arun_hierarchical(steps, input, **kwargs)
        if strategy == "reflexion":
            return await self._arun_reflexion(steps, input, **kwargs)
        if strategy == "graph":
            return await self._arun_graph(steps, input, **kwargs)
        raise ConfigurationException(
            f"Unknown execution strategy '{strategy}' for the native orchestrator. "
            f"Supported: {', '.join(self._STRATEGIES)}.",
        )

    @staticmethod
    def _split_coordinator_and_workers(
        steps: Sequence["Agent"], *, role: str
    ) -> tuple["Agent", dict[str, "Agent"]]:
        """Split ``steps`` into a coordinating agent + name-addressed workers.

        Shared by the ``planner`` and ``supervisor`` strategies: ``steps[0]``
        is the coordinator, ``steps[1:]`` are workers it addresses by
        ``agent.name``. A ``staticmethod`` (it never touches ``self``)
        specifically so :class:`~requisite.orchestrators.langgraph_orchestrator.LangGraphOrchestrator`
        can reuse it too, via ``NativeOrchestrator._split_coordinator_and_workers(...)`` --
        see ``docs/adr/0016-langgraph-branching.md``.
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
        results = await _gather_waiting_for_all(*(agent.arun(input, **kwargs) for agent in steps))
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

    def _run_reflexion(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_trials: int = 3,
        evaluator: Optional[Evaluator] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        if len(steps) != 1:
            raise ConfigurationException(
                f"The 'reflexion' strategy requires exactly one agent (it attempts, "
                f"evaluates, and reflects on its own output). Got {len(steps)}.",
            )
        worker = steps[0]
        results: list["AgentResult"] = []
        reflections: list[str] = []
        attempt: Optional["AgentResult"] = None
        succeeded = False

        for trial in range(max_trials):
            attempt = worker.run(self._task_prompt_with_context(input, reflections), **kwargs)
            results.append(attempt)

            if evaluator is not None:
                evaluation = evaluator(input, attempt.content)
            else:
                evaluation = worker.ai.chat(
                    _reflexion_default_evaluation_prompt(input, attempt.content),
                    response_model=EvaluationResult,
                )

            if evaluation.success:
                succeeded = True
                break
            if trial < max_trials - 1:
                reflection = worker.run(
                    _reflexion_reflect_prompt(input, attempt.content, evaluation.feedback),
                    **kwargs,
                )
                results.append(reflection)
                reflections.append(reflection.content)

        assert attempt is not None  # max_trials >= 1 is enforced by callers via Workflow
        return WorkflowResult(
            content=attempt.content,
            steps=results,
            orchestrator=self.name,
            strategy="reflexion",
            succeeded=succeeded,
        )

    async def _arun_reflexion(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_trials: int = 3,
        evaluator: Optional[Evaluator] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        if len(steps) != 1:
            raise ConfigurationException(
                f"The 'reflexion' strategy requires exactly one agent (it attempts, "
                f"evaluates, and reflects on its own output). Got {len(steps)}.",
            )
        worker = steps[0]
        results: list["AgentResult"] = []
        reflections: list[str] = []
        attempt: Optional["AgentResult"] = None
        succeeded = False

        for trial in range(max_trials):
            attempt = await worker.arun(
                self._task_prompt_with_context(input, reflections), **kwargs
            )
            results.append(attempt)

            if evaluator is not None:
                evaluation = evaluator(input, attempt.content)
            else:
                evaluation = await worker.ai.achat(
                    _reflexion_default_evaluation_prompt(input, attempt.content),
                    response_model=EvaluationResult,
                )

            if evaluation.success:
                succeeded = True
                break
            if trial < max_trials - 1:
                reflection = await worker.arun(
                    _reflexion_reflect_prompt(input, attempt.content, evaluation.feedback),
                    **kwargs,
                )
                results.append(reflection)
                reflections.append(reflection.content)

        assert attempt is not None  # max_trials >= 1 is enforced by callers via Workflow
        return WorkflowResult(
            content=attempt.content,
            steps=results,
            orchestrator=self.name,
            strategy="reflexion",
            succeeded=succeeded,
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
        return self._run_delegation_loop(
            supervisor, workers, input, max_rounds=max_rounds, strategy_name="supervisor", **kwargs
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
        return await self._arun_delegation_loop(
            supervisor, workers, input, max_rounds=max_rounds, strategy_name="supervisor", **kwargs
        )

    def _run_hierarchical(
        self,
        steps: Sequence[Any],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 6,
        **kwargs: Any,
    ) -> WorkflowResult:
        coordinator, delegates = self._split_coordinator_and_delegates(steps, role="hierarchical")
        return self._run_delegation_loop(
            coordinator,
            delegates,
            input,
            max_rounds=max_rounds,
            strategy_name="hierarchical",
            **kwargs,
        )

    async def _arun_hierarchical(
        self,
        steps: Sequence[Any],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 6,
        **kwargs: Any,
    ) -> WorkflowResult:
        coordinator, delegates = self._split_coordinator_and_delegates(steps, role="hierarchical")
        return await self._arun_delegation_loop(
            coordinator,
            delegates,
            input,
            max_rounds=max_rounds,
            strategy_name="hierarchical",
            **kwargs,
        )

    def _run_delegation_loop(
        self,
        coordinator: "Agent",
        delegates: dict[str, Any],
        input: str,  # noqa: A002
        *,
        max_rounds: int,
        strategy_name: str,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Shared by ``supervisor`` and ``hierarchical`` -- structurally
        identical round loops (same decision model, same "finish"
        termination, same transcript shape); the only difference between
        the two strategies is which split-helper validated/built
        ``delegates``, so that's the only part not shared. See
        ``docs/adr/0013-hierarchical-strategy.md``.
        """
        results: list[Any] = []
        transcript: list[tuple[str, str, str]] = []

        for _ in range(max_rounds):
            decision = coordinator.ai.chat(
                _supervisor_prompt(input, list(delegates), transcript),
                response_model=_SupervisorDecision,
            )
            if decision.action == "finish":
                return WorkflowResult(
                    content=decision.final_answer or "",
                    steps=results,
                    orchestrator=self.name,
                    strategy=strategy_name,
                )

            delegate = self._resolve_delegate(
                decision, supervisor_name=coordinator.name, workers=delegates
            )
            subtask = decision.task or input
            result = delegate.run(subtask, **kwargs)
            results.append(result)
            transcript.append((delegate.name, subtask, result.content))

        raise AgentException(
            f"Workflow {strategy_name} '{coordinator.name}' exceeded max_rounds={max_rounds} "
            f"without reaching a final answer.",
        )

    async def _arun_delegation_loop(
        self,
        coordinator: "Agent",
        delegates: dict[str, Any],
        input: str,  # noqa: A002
        *,
        max_rounds: int,
        strategy_name: str,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Async counterpart to :meth:`_run_delegation_loop`."""
        results: list[Any] = []
        transcript: list[tuple[str, str, str]] = []

        for _ in range(max_rounds):
            decision = await coordinator.ai.achat(
                _supervisor_prompt(input, list(delegates), transcript),
                response_model=_SupervisorDecision,
            )
            if decision.action == "finish":
                return WorkflowResult(
                    content=decision.final_answer or "",
                    steps=results,
                    orchestrator=self.name,
                    strategy=strategy_name,
                )

            delegate = self._resolve_delegate(
                decision, supervisor_name=coordinator.name, workers=delegates
            )
            subtask = decision.task or input
            result = await delegate.arun(subtask, **kwargs)
            results.append(result)
            transcript.append((delegate.name, subtask, result.content))

        raise AgentException(
            f"Workflow {strategy_name} '{coordinator.name}' exceeded max_rounds={max_rounds} "
            f"without reaching a final answer.",
        )

    @staticmethod
    def _resolve_delegate(
        decision: _SupervisorDecision, *, supervisor_name: str, workers: dict[str, Any]
    ) -> Any:
        if decision.worker not in workers:
            raise ConfigurationException(
                f"Supervisor agent '{supervisor_name}' tried to delegate to unknown "
                f"worker '{decision.worker}'. Available workers: {sorted(workers)}.",
            )
        return workers[decision.worker]

    @staticmethod
    def _split_coordinator_and_delegates(
        steps: Sequence[Any], *, role: str
    ) -> tuple["Agent", dict[str, Any]]:
        """Split ``steps`` into a coordinating agent + name-addressed delegates.

        Used by the ``hierarchical`` strategy, which -- unlike every
        other coordinator/worker strategy -- allows each delegate
        (``steps[1:]``) to be either an :class:`~requisite.agents.agent.Agent`
        or a named :class:`~requisite.workflows.workflow.Workflow` (a
        nested "team"). Kept separate from
        :meth:`_split_coordinator_and_workers` rather than loosening
        that one, so every other strategy's Agent-only validation stays
        exactly as strict as before. Duck-typed (``hasattr``, not
        ``isinstance``) to avoid a real circular import -- ``native.py``
        cannot import ``Workflow`` at runtime (``native.py`` ->
        ``workflow.py`` -> ``orchestrators/factory.py`` -> ``native.py``).
        """
        if len(steps) < 2:
            raise ConfigurationException(
                f"The '{role}' strategy requires at least 2 agents: one {role} "
                f"(steps[0]) and one or more delegates (steps[1:]). Got {len(steps)}.",
            )
        coordinator = steps[0]
        if not hasattr(coordinator, "ai"):
            raise ConfigurationException(
                f"The '{role}' strategy requires steps[0] to be an Agent (it makes "
                f"the routing decisions) -- got {type(coordinator).__name__}.",
            )
        delegates = list(steps[1:])
        for delegate in delegates:
            if getattr(delegate, "name", None) is None:
                raise ConfigurationException(
                    f"The '{role}' strategy requires every delegate to have a name -- "
                    f"a Workflow used as a delegate must be constructed with name=... "
                    f"(got an unnamed {type(delegate).__name__}).",
                )
        delegate_names = [delegate.name for delegate in delegates]
        if len(set(delegate_names)) != len(delegate_names):
            raise ConfigurationException(
                f"The '{role}' strategy requires unique delegate names; got {delegate_names}.",
            )
        return coordinator, {delegate.name: delegate for delegate in delegates}

    def _run_critic(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 3,
        **kwargs: Any,
    ) -> WorkflowResult:
        if len(steps) != 2:
            raise ConfigurationException(
                f"The 'critic' strategy requires exactly two agents: a generator "
                f"(steps[0]) and a critic (steps[1]). Got {len(steps)}.",
            )
        generator, critic = steps[0], steps[1]
        results: list["AgentResult"] = []

        draft = generator.run(input, **kwargs)
        results.append(draft)

        for _ in range(max_rounds - 1):
            critique = critic.run(_critic_prompt(input, draft.content), **kwargs)
            results.append(critique)
            if critique.content.strip() == "NO_CHANGES_NEEDED":
                break
            revised = generator.run(
                _reflection_revise_prompt(input, draft.content, critique.content), **kwargs
            )
            results.append(revised)
            draft = revised

        return WorkflowResult(
            content=draft.content, steps=results, orchestrator=self.name, strategy="critic"
        )

    async def _arun_critic(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 3,
        **kwargs: Any,
    ) -> WorkflowResult:
        if len(steps) != 2:
            raise ConfigurationException(
                f"The 'critic' strategy requires exactly two agents: a generator "
                f"(steps[0]) and a critic (steps[1]). Got {len(steps)}.",
            )
        generator, critic = steps[0], steps[1]
        results: list["AgentResult"] = []

        draft = await generator.arun(input, **kwargs)
        results.append(draft)

        for _ in range(max_rounds - 1):
            critique = await critic.arun(_critic_prompt(input, draft.content), **kwargs)
            results.append(critique)
            if critique.content.strip() == "NO_CHANGES_NEEDED":
                break
            revised = await generator.arun(
                _reflection_revise_prompt(input, draft.content, critique.content), **kwargs
            )
            results.append(revised)
            draft = revised

        return WorkflowResult(
            content=draft.content, steps=results, orchestrator=self.name, strategy="critic"
        )

    def _run_consensus(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        **kwargs: Any,
    ) -> WorkflowResult:
        synthesizer, participants = self._split_coordinator_and_workers(
            steps, role="consensus synthesizer"
        )

        with ThreadPoolExecutor(max_workers=max(len(participants), 1)) as executor:
            futures = [
                executor.submit(participant.run, input, **kwargs)
                for participant in participants.values()
            ]
            results = [future.result() for future in futures]

        synthesis = synthesizer.run(_consensus_prompt(input, results), **kwargs)

        return WorkflowResult(
            content=synthesis.content,
            steps=[*results, synthesis],
            orchestrator=self.name,
            strategy="consensus",
        )

    async def _arun_consensus(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        **kwargs: Any,
    ) -> WorkflowResult:
        synthesizer, participants = self._split_coordinator_and_workers(
            steps, role="consensus synthesizer"
        )

        results = await _gather_waiting_for_all(
            *(participant.arun(input, **kwargs) for participant in participants.values())
        )

        synthesis = await synthesizer.arun(_consensus_prompt(input, results), **kwargs)

        return WorkflowResult(
            content=synthesis.content,
            steps=[*results, synthesis],
            orchestrator=self.name,
            strategy="consensus",
        )

    def _run_debate(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 3,
        **kwargs: Any,
    ) -> WorkflowResult:
        moderator, debaters = self._split_coordinator_and_workers(steps, role="debate moderator")
        debater_names = list(debaters)

        transcript: dict[str, list[str]] = {name: [] for name in debater_names}
        results: list["AgentResult"] = []

        for round_num in range(max_rounds):
            with ThreadPoolExecutor(max_workers=max(len(debaters), 1)) as executor:
                futures = {
                    name: executor.submit(
                        debater.run,
                        _debate_prompt(
                            input, debater_names, transcript, agent_name=name, round_num=round_num
                        ),
                        **kwargs,
                    )
                    for name, debater in debaters.items()
                }
                round_results = {name: future.result() for name, future in futures.items()}
            for name in debater_names:
                result = round_results[name]
                transcript[name].append(result.content)
                results.append(result)

        verdict = moderator.run(_debate_verdict_prompt(input, debater_names, transcript), **kwargs)

        return WorkflowResult(
            content=verdict.content,
            steps=[*results, verdict],
            orchestrator=self.name,
            strategy="debate",
        )

    async def _arun_debate(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        max_rounds: int = 3,
        **kwargs: Any,
    ) -> WorkflowResult:
        moderator, debaters = self._split_coordinator_and_workers(steps, role="debate moderator")
        debater_names = list(debaters)

        transcript: dict[str, list[str]] = {name: [] for name in debater_names}
        results: list["AgentResult"] = []

        for round_num in range(max_rounds):
            round_results_list = await _gather_waiting_for_all(
                *(
                    debaters[name].arun(
                        _debate_prompt(
                            input, debater_names, transcript, agent_name=name, round_num=round_num
                        ),
                        **kwargs,
                    )
                    for name in debater_names
                )
            )
            for name, result in zip(debater_names, round_results_list, strict=True):
                transcript[name].append(result.content)
                results.append(result)

        verdict = await moderator.arun(
            _debate_verdict_prompt(input, debater_names, transcript), **kwargs
        )

        return WorkflowResult(
            content=verdict.content,
            steps=[*results, verdict],
            orchestrator=self.name,
            strategy="debate",
        )

    def _run_map_reduce(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        map_items: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        if not map_items:
            raise ConfigurationException(
                "The 'map_reduce' strategy requires map_items=[...] -- the list of "
                "individual items to process, passed to workflow.run()/arun().",
            )
        reducer, mappers = self._split_coordinator_and_workers(steps, role="map-reduce reducer")
        mapper_list = list(mappers.values())

        with ThreadPoolExecutor(max_workers=max(len(map_items), 1)) as executor:
            futures = [
                executor.submit(
                    mapper_list[i % len(mapper_list)].run, _map_prompt(input, item), **kwargs
                )
                for i, item in enumerate(map_items)
            ]
            map_results = [future.result() for future in futures]

        reduced = reducer.run(_reduce_prompt(input, map_items, map_results), **kwargs)

        return WorkflowResult(
            content=reduced.content,
            steps=[*map_results, reduced],
            orchestrator=self.name,
            strategy="map_reduce",
        )

    async def _arun_map_reduce(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        map_items: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        if not map_items:
            raise ConfigurationException(
                "The 'map_reduce' strategy requires map_items=[...] -- the list of "
                "individual items to process, passed to workflow.run()/arun().",
            )
        reducer, mappers = self._split_coordinator_and_workers(steps, role="map-reduce reducer")
        mapper_list = list(mappers.values())

        map_results = await _gather_waiting_for_all(
            *(
                mapper_list[i % len(mapper_list)].arun(_map_prompt(input, item), **kwargs)
                for i, item in enumerate(map_items)
            )
        )

        reduced = await reducer.arun(_reduce_prompt(input, map_items, map_results), **kwargs)

        return WorkflowResult(
            content=reduced.content,
            steps=[*map_results, reduced],
            orchestrator=self.name,
            strategy="map_reduce",
        )

    def _run_tree_of_thoughts(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        breadth: int = 3,
        beam_width: int = 1,
        max_depth: int = 3,
        **kwargs: Any,
    ) -> WorkflowResult:
        self._validate_tot_params(breadth=breadth, beam_width=beam_width, max_depth=max_depth)
        evaluator, thinkers = self._split_coordinator_and_workers(
            steps, role="tree-of-thoughts evaluator"
        )
        thinker_list = list(thinkers.values())

        paths: list[list[str]] = [[]]
        results: list["AgentResult"] = []

        for _ in range(max_depth):
            tasks = [path for path in paths for _ in range(breadth)]
            with ThreadPoolExecutor(max_workers=max(len(tasks), 1)) as executor:
                futures = [
                    executor.submit(
                        thinker_list[i % len(thinker_list)].run,
                        _tot_thinker_prompt(input, path),
                        **kwargs,
                    )
                    for i, path in enumerate(tasks)
                ]
                candidate_results = [future.result() for future in futures]
            results.extend(candidate_results)
            candidate_paths = [
                [*path, result.content]
                for path, result in zip(tasks, candidate_results, strict=True)
            ]

            evaluation = evaluator.ai.chat(
                _tot_evaluation_prompt(input, candidate_paths), response_model=_ThoughtEvaluation
            )
            final = self._select_finished_tot_candidate(evaluation, candidate_paths)
            if final is not None:
                return WorkflowResult(
                    content=final[-1],
                    steps=results,
                    orchestrator=self.name,
                    strategy="tree_of_thoughts",
                )

            paths = self._prune_tot_candidates(evaluation, candidate_paths, beam_width=beam_width)

        best_path = paths[0]
        return WorkflowResult(
            content=best_path[-1] if best_path else "",
            steps=results,
            orchestrator=self.name,
            strategy="tree_of_thoughts",
        )

    async def _arun_tree_of_thoughts(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
        *,
        breadth: int = 3,
        beam_width: int = 1,
        max_depth: int = 3,
        **kwargs: Any,
    ) -> WorkflowResult:
        self._validate_tot_params(breadth=breadth, beam_width=beam_width, max_depth=max_depth)
        evaluator, thinkers = self._split_coordinator_and_workers(
            steps, role="tree-of-thoughts evaluator"
        )
        thinker_list = list(thinkers.values())

        paths: list[list[str]] = [[]]
        results: list["AgentResult"] = []

        for _ in range(max_depth):
            tasks = [path for path in paths for _ in range(breadth)]
            candidate_results = await _gather_waiting_for_all(
                *(
                    thinker_list[i % len(thinker_list)].arun(
                        _tot_thinker_prompt(input, path), **kwargs
                    )
                    for i, path in enumerate(tasks)
                )
            )
            results.extend(candidate_results)
            candidate_paths = [
                [*path, result.content]
                for path, result in zip(tasks, candidate_results, strict=True)
            ]

            evaluation = await evaluator.ai.achat(
                _tot_evaluation_prompt(input, candidate_paths), response_model=_ThoughtEvaluation
            )
            final = self._select_finished_tot_candidate(evaluation, candidate_paths)
            if final is not None:
                return WorkflowResult(
                    content=final[-1],
                    steps=results,
                    orchestrator=self.name,
                    strategy="tree_of_thoughts",
                )

            paths = self._prune_tot_candidates(evaluation, candidate_paths, beam_width=beam_width)

        best_path = paths[0]
        return WorkflowResult(
            content=best_path[-1] if best_path else "",
            steps=results,
            orchestrator=self.name,
            strategy="tree_of_thoughts",
        )

    @staticmethod
    def _validate_tot_params(*, breadth: int, beam_width: int, max_depth: int) -> None:
        if breadth < 1:
            raise ConfigurationException(
                f"The 'tree_of_thoughts' strategy requires breadth >= 1. Got {breadth}.",
            )
        if beam_width < 1:
            raise ConfigurationException(
                f"The 'tree_of_thoughts' strategy requires beam_width >= 1. Got {beam_width}.",
            )
        if max_depth < 1:
            raise ConfigurationException(
                f"The 'tree_of_thoughts' strategy requires max_depth >= 1. Got {max_depth}.",
            )

    @staticmethod
    def _select_finished_tot_candidate(
        evaluation: _ThoughtEvaluation, candidate_paths: Sequence[list[str]]
    ) -> Optional[list[str]]:
        finished = [
            s for s in evaluation.scores if s.finished and 0 <= s.index < len(candidate_paths)
        ]
        if not finished:
            return None
        best = max(finished, key=lambda s: s.score)
        return candidate_paths[best.index]

    @staticmethod
    def _prune_tot_candidates(
        evaluation: _ThoughtEvaluation, candidate_paths: Sequence[list[str]], *, beam_width: int
    ) -> list[list[str]]:
        score_by_index = {s.index: s.score for s in evaluation.scores}
        ranked = sorted(
            range(len(candidate_paths)),
            key=lambda i: score_by_index.get(i, 0.0),
            reverse=True,
        )
        return [candidate_paths[i] for i in ranked[:beam_width]]

    def _run_graph(
        self,
        steps: Sequence[Any],
        input: str,  # noqa: A002
        *,
        edges: Sequence[tuple[str, str, Optional[Callable[[str], bool]]]] = (),
        max_steps: int = 25,
        **kwargs: Any,
    ) -> WorkflowResult:
        if max_steps < 1:
            raise ConfigurationException(
                f"The 'graph' strategy requires max_steps >= 1. Got {max_steps}."
            )
        nodes = self._index_graph_nodes(steps, role="graph")
        edges_by_source = self._validate_graph_edges(edges, nodes)

        results: list[Any] = []
        current_name = steps[0].name
        current_input = input
        for _ in range(max_steps):
            result = nodes[current_name].run(current_input, **kwargs)
            results.append(result)
            current_input = result.content
            next_name = self._resolve_next_graph_node(current_name, result.content, edges_by_source)
            if next_name is None:
                return WorkflowResult(
                    content=result.content, steps=results, orchestrator=self.name, strategy="graph"
                )
            current_name = next_name

        raise AgentException(
            f"Workflow graph exceeded max_steps={max_steps} without reaching an end.",
        )

    async def _arun_graph(
        self,
        steps: Sequence[Any],
        input: str,  # noqa: A002
        *,
        edges: Sequence[tuple[str, str, Optional[Callable[[str], bool]]]] = (),
        max_steps: int = 25,
        **kwargs: Any,
    ) -> WorkflowResult:
        if max_steps < 1:
            raise ConfigurationException(
                f"The 'graph' strategy requires max_steps >= 1. Got {max_steps}."
            )
        nodes = self._index_graph_nodes(steps, role="graph")
        edges_by_source = self._validate_graph_edges(edges, nodes)

        results: list[Any] = []
        current_name = steps[0].name
        current_input = input
        for _ in range(max_steps):
            result = await nodes[current_name].arun(current_input, **kwargs)
            results.append(result)
            current_input = result.content
            next_name = self._resolve_next_graph_node(current_name, result.content, edges_by_source)
            if next_name is None:
                return WorkflowResult(
                    content=result.content, steps=results, orchestrator=self.name, strategy="graph"
                )
            current_name = next_name

        raise AgentException(
            f"Workflow graph exceeded max_steps={max_steps} without reaching an end.",
        )

    @staticmethod
    def _index_graph_nodes(steps: Sequence[Any], *, role: str) -> dict[str, Any]:
        """Name-address the ``graph`` strategy's nodes.

        Unlike :meth:`_split_coordinator_and_delegates`, there's no
        coordinator/worker split here -- every step is a peer node,
        including ``steps[0]`` (the entry point). Duck-typed
        (``getattr(node, "name", None)``, not ``isinstance``) for the same
        reason ``_split_coordinator_and_delegates`` is: a node may be
        either an ``Agent`` or a named ``Workflow`` "team", and
        ``native.py`` can't import either at runtime without a circular
        import.
        """
        for node in steps:
            if getattr(node, "name", None) is None:
                raise ConfigurationException(
                    f"The '{role}' strategy requires every node to have a name -- a "
                    f"Workflow used as a node must be constructed with name=... (got "
                    f"an unnamed {type(node).__name__}).",
                )
            if node.name == END:
                raise ConfigurationException(
                    f"The '{role}' strategy reserves the name {END!r} for the "
                    f"terminating-edge sentinel -- a real node cannot be named {END!r}, "
                    f"since _resolve_next_graph_node would treat any edge routed to it as "
                    f"'terminate' rather than ever actually running it. Rename the node.",
                )
        node_names = [node.name for node in steps]
        if len(set(node_names)) != len(node_names):
            raise ConfigurationException(
                f"The '{role}' strategy requires unique node names; got {node_names}.",
            )
        return {node.name: node for node in steps}

    @staticmethod
    def _validate_graph_edges(
        edges: Sequence[tuple[str, str, Optional[Callable[[str], bool]]]],
        nodes: dict[str, Any],
    ) -> dict[str, list[tuple[str, Optional[Callable[[str], bool]]]]]:
        edges_by_source: dict[str, list[tuple[str, Optional[Callable[[str], bool]]]]] = {}
        for from_, to, condition in edges:
            if from_ not in nodes:
                raise ConfigurationException(
                    f"Graph edge references unknown source node '{from_}'. "
                    f"Available nodes: {sorted(nodes)}.",
                )
            if to != END and to not in nodes:
                raise ConfigurationException(
                    f"Graph edge from '{from_}' references unknown target node '{to}'. "
                    f"Available nodes: {sorted(nodes)}, or END.",
                )
            edges_by_source.setdefault(from_, []).append((to, condition))
        return edges_by_source

    @staticmethod
    def _resolve_next_graph_node(
        current_name: str,
        output: str,
        edges_by_source: dict[str, list[tuple[str, Optional[Callable[[str], bool]]]]],
    ) -> Optional[str]:
        outgoing = edges_by_source.get(current_name, [])
        if not outgoing:
            return None
        for to, condition in outgoing:
            if condition is None or condition(output):
                return None if to == END else to
        raise AgentException(
            f"Workflow graph node '{current_name}' produced output that matched none "
            f"of its outgoing edges' conditions, and none is an unconditional "
            f"fallback edge. Add a fallback edge (condition=None) or a condition "
            f"that always matches.",
        )
