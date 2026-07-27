"""
Orchestrator interface.

An orchestrator takes an ordered list of agents ("steps") plus an
initial input and executes them according to some strategy (sequential,
parallel, ...), producing a normalized :class:`WorkflowResult`.

:class:`~requisite.workflows.workflow.Workflow` is the small, ergonomic
facade end users interact with; it delegates actual execution to
whichever orchestrator backend is configured (``"native"``,
``"langgraph"``, and -- in future -- ``"crewai"``, ``"autogen"``), so
switching backends is a configuration change, exactly like switching
providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from requisite.agents.agent import Agent


class WorkflowResult(BaseModel):
    """The outcome of running a :class:`~requisite.workflows.workflow.Workflow`.

    Parameters
    ----------
    content:
        The workflow's final output text.
    steps:
        The per-agent :class:`~requisite.agents.agent.AgentResult`
        objects produced along the way, in execution order.
    orchestrator:
        Name of the orchestrator backend that executed this workflow
        (e.g. ``"native"``, ``"langgraph"``).
    strategy:
        Execution strategy used (e.g. ``"sequential"``, ``"parallel"``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    steps: list[Any] = Field(default_factory=list)
    orchestrator: str
    strategy: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.content


class BaseOrchestrator(ABC):
    """Abstract interface every orchestration backend implements.

    Adding a new orchestration backend (e.g. a real CrewAI or AutoGen
    integration) requires implementing this interface only, and
    registering it with
    :class:`~requisite.orchestrators.factory.OrchestratorRegistry` --
    no changes to :class:`~requisite.workflows.workflow.Workflow` or
    any other framework code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique, lowercase identifier for this orchestrator (e.g. ``"native"``)."""

    @abstractmethod
    def run(
        self,
        steps: Sequence[Agent],
        input: str | None,
        *,
        strategy: str = "sequential",
        **kwargs: Any,
    ) -> WorkflowResult:
        """Synchronously execute ``steps`` according to ``strategy``."""

    @abstractmethod
    async def arun(
        self,
        steps: Sequence[Agent],
        input: str | None,
        *,
        strategy: str = "sequential",
        **kwargs: Any,
    ) -> WorkflowResult:
        """Asynchronous counterpart to :meth:`run`."""
