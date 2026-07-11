"""
Native orchestrator: sequential and parallel execution, no external
orchestration framework required.

This is the default backend for :class:`~requisite.workflows.workflow.Workflow`.
Additional strategies (supervisor, planner, reflection, debate, critic,
consensus, hierarchical, map-reduce, tree-of-thoughts, graph execution)
are natural extensions of this class -- add a ``_run_<strategy>`` /
``_arun_<strategy>`` pair and register the name in ``_SYNC_STRATEGIES`` /
``_ASYNC_STRATEGIES`` below; nothing else needs to change.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult

if TYPE_CHECKING:
    from requisite.agents.agent import Agent, AgentResult

logger = logging.getLogger("requisite.orchestrators.native")


class NativeOrchestrator(BaseOrchestrator):
    """Runs agents sequentially or in parallel using plain Python -- no
    external orchestration framework required.

    Sequential strategy
        Each agent's output becomes the next agent's input, forming a
        pipeline (e.g. a "research" agent feeding a "writer" agent).

    Parallel strategy
        Every agent runs against the *same* input concurrently; their
        outputs are collected and combined.
    """

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
        raise ConfigurationException(
            f"Unknown execution strategy '{strategy}' for the native orchestrator. "
            f"Supported: sequential, parallel.",
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
        raise ConfigurationException(
            f"Unknown execution strategy '{strategy}' for the native orchestrator. "
            f"Supported: sequential, parallel.",
        )

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
