"""
Shared plumbing for the coordination-only agent-SDK orchestrator backends
(``openai_agents``, ``strands``, ``agent_framework``).

Each of those wraps a Requisite :class:`~requisite.agents.agent.Agent` in a
third-party SDK's own agent type, with a model adapter that proxies every
real model call back to ``Agent.arun()``. What differs per SDK is only
*how an agent is run* (and, for ``"sequential"``, how steps are chained);
the ``"supervisor"`` decision loop is identical, so it lives here and
delegates to :meth:`NativeOrchestrator._arun_delegation_loop` -- one
implementation of delegation, ``max_rounds`` and unknown-worker handling
shared by every backend rather than re-derived per SDK.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Optional

from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.orchestrators.base import WorkflowResult
from requisite.orchestrators.native import NativeOrchestrator

if TYPE_CHECKING:
    from requisite.agents.agent import Agent, AgentResult


def check_run_inputs(steps: Sequence[Any], input: Optional[str]) -> None:  # noqa: A002
    if not steps:
        raise ConfigurationException("Workflow has no agents; call workflow.add(agent) first.")
    if input is None:
        raise ConfigurationException("Workflow.run(...) requires an initial input/task.")


def unsupported_strategy(backend: str, strategy: str) -> ConfigurationException:
    return ConfigurationException(
        f"The {backend} orchestrator supports the 'sequential' and 'supervisor' "
        f"strategies (got '{strategy}').",
    )


def collect_steps(models: Sequence[Any]) -> list["AgentResult"]:
    """Flatten the ``AgentResult``s each proxy model recorded, in step order."""
    steps: list["AgentResult"] = []
    for model in models:
        steps.extend(model._collected_results)
    return steps


class SdkDelegate:
    """A supervisor worker that runs through a third-party SDK.

    Exposes the ``.name`` / async ``.arun()`` shape the native delegation
    loop expects. ``run_through_sdk`` executes the SDK agent for one
    subtask; the real :class:`~requisite.agents.agent.AgentResult` is
    whatever the proxy model recorded in ``results`` while it ran.
    """

    def __init__(
        self,
        name: str,
        run_through_sdk: Callable[[str], Awaitable[Any]],
        results: list["AgentResult"],
    ) -> None:
        self.name = name
        self._run_through_sdk = run_through_sdk
        self._results = results

    async def arun(self, task: str, **kwargs: Any) -> "AgentResult":
        before = len(self._results)
        await self._run_through_sdk(task)
        if len(self._results) == before:
            raise AgentException(
                f"Worker '{self.name}' finished its SDK run without ever calling the wrapped "
                f"Requisite agent -- no result to report.",
            )
        return self._results[-1]


async def run_supervisor(
    *,
    backend: str,
    steps: Sequence["Agent"],
    input: str,  # noqa: A002
    max_rounds: int,
    build_delegate: Callable[["Agent"], SdkDelegate],
) -> WorkflowResult:
    coordinator, workers = NativeOrchestrator._split_coordinator_and_workers(
        steps, role="supervisor"
    )
    delegates = {name: build_delegate(worker) for name, worker in workers.items()}
    result = await NativeOrchestrator()._arun_delegation_loop(
        coordinator, delegates, input, max_rounds=max_rounds, strategy_name="supervisor"
    )
    return result.model_copy(update={"orchestrator": backend})
