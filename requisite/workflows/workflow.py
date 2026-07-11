"""
``Workflow``: compose agents into a multi-agent pipeline.

Examples
--------
>>> from requisite import Agent, Workflow
>>>
>>> research = Agent(name="Researcher", provider="openai")  # doctest: +SKIP
>>> writer = Agent(name="Writer", provider="openai")  # doctest: +SKIP
>>>
>>> workflow = Workflow()
>>> workflow.add(research)  # doctest: +SKIP
>>> workflow.add(writer)  # doctest: +SKIP
>>> result = workflow.run("Research and summarize the latest AI trends.")  # doctest: +SKIP

Switching the execution engine is a configuration change, not a rewrite:

>>> workflow.use_langgraph()  # doctest: +SKIP
>>> result = workflow.run("Research and summarize the latest AI trends.")  # doctest: +SKIP

Running agents in parallel instead of as a pipeline:

>>> workflow.parallel()  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.orchestrators.base import WorkflowResult
from requisite.orchestrators.factory import OrchestratorRegistry
from requisite.orchestrators.factory import default_registry as default_orchestrator_registry

if TYPE_CHECKING:
    from requisite.agents.agent import Agent

logger = logging.getLogger("requisite.workflows")

_KNOWN_STRATEGIES = {"sequential", "parallel"}


class Workflow:
    """Compose :class:`~requisite.agents.agent.Agent` instances into a pipeline.

    Parameters
    ----------
    strategy:
        How agents are executed relative to one another: ``"sequential"``
        (each agent's output feeds the next -- the default) or
        ``"parallel"`` (every agent runs against the same input
        concurrently). See the module docstring for further multi-agent
        strategies planned on the roadmap (supervisor, planner,
        reflection, debate, critic, consensus, hierarchical, map-reduce,
        tree-of-thoughts, graph execution) -- each becomes a new
        ``strategy`` value handled by the underlying orchestrator, with
        no change to this class's public API.
    orchestrator:
        Execution backend: ``"native"`` (default, pure Python, no extra
        dependency) or ``"langgraph"``. Change via :meth:`use_langgraph` /
        :meth:`use_native` instead of passing this directly, for
        readability at the call site.
    registry:
        The :class:`~requisite.orchestrators.factory.OrchestratorRegistry`
        used to resolve ``orchestrator``. Defaults to the framework's
        built-in registry.

    Examples
    --------
    >>> workflow = Workflow()
    >>> workflow.add(research).add(writer)  # doctest: +SKIP
    >>> workflow.run("Research AI trends.")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        strategy: str = "sequential",
        orchestrator: str = "native",
        registry: Optional[OrchestratorRegistry] = None,
    ) -> None:
        self._steps: list["Agent"] = []
        self._strategy = strategy
        self._orchestrator_name = orchestrator
        self._registry = registry or default_orchestrator_registry

    def add(self, agent: "Agent") -> "Workflow":
        """Append an agent to the pipeline. Returns ``self`` for chaining."""
        self._steps.append(agent)
        return self

    @property
    def agents(self) -> list["Agent"]:
        """The agents added so far, in order."""
        return list(self._steps)

    def sequential(self) -> "Workflow":
        """Switch to the sequential execution strategy. Returns ``self`` for chaining."""
        self._strategy = "sequential"
        return self

    def parallel(self) -> "Workflow":
        """Switch to the parallel execution strategy. Returns ``self`` for chaining."""
        self._strategy = "parallel"
        return self

    def use_native(self) -> "Workflow":
        """Use the built-in, dependency-free execution engine. Returns ``self`` for chaining."""
        self._orchestrator_name = "native"
        return self

    def use_langgraph(self) -> "Workflow":
        """Delegate execution to `langgraph <https://github.com/langchain-ai/langgraph>`_.

        Requires ``pip install langgraph``. Returns ``self`` for chaining.
        """
        self._orchestrator_name = "langgraph"
        return self

    def use_crewai(self) -> "Workflow":
        """Delegate execution to CrewAI. On the roadmap; not yet implemented."""
        self._orchestrator_name = "crewai"
        return self

    def use_autogen(self) -> "Workflow":
        """Delegate execution to AutoGen. On the roadmap; not yet implemented."""
        self._orchestrator_name = "autogen"
        return self

    def run(self, input: Optional[str] = None, **kwargs: Any) -> WorkflowResult:  # noqa: A002
        """Execute the workflow.

        Parameters
        ----------
        input:
            The initial task/prompt handed to the first agent (or to
            every agent, under the ``"parallel"`` strategy).
        **kwargs:
            Passed through to each agent's ``run``/``arun`` call.

        Returns
        -------
        WorkflowResult
        """
        if self._strategy not in _KNOWN_STRATEGIES:
            raise ConfigurationException(
                f"Unknown strategy '{self._strategy}'. Supported: {sorted(_KNOWN_STRATEGIES)}",
            )
        orchestrator_instance = self._registry.create(self._orchestrator_name)
        return orchestrator_instance.run(self._steps, input, strategy=self._strategy, **kwargs)

    async def arun(self, input: Optional[str] = None, **kwargs: Any) -> WorkflowResult:  # noqa: A002
        """Async counterpart to :meth:`run`."""
        if self._strategy not in _KNOWN_STRATEGIES:
            raise ConfigurationException(
                f"Unknown strategy '{self._strategy}'. Supported: {sorted(_KNOWN_STRATEGIES)}",
            )
        orchestrator_instance = self._registry.create(self._orchestrator_name)
        return await orchestrator_instance.arun(
            self._steps, input, strategy=self._strategy, **kwargs
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        agent_names = [a.name for a in self._steps]
        return (
            f"Workflow(agents={agent_names!r}, strategy={self._strategy!r}, "
            f"orchestrator={self._orchestrator_name!r})"
        )
