"""
CrewAI orchestrator backend.

Executes the same list of agents as :class:`~requisite.orchestrators.native.NativeOrchestrator`,
but delegates coordination to `CrewAI <https://github.com/crewAIInc/crewAI>`_.
The public :class:`~requisite.workflows.workflow.Workflow` API is
identical either way -- switching backends via ``workflow.use_crewai()``
never changes how you call ``.add()`` or ``.run()``.

CrewAI is used for coordination only -- every actual model call still
goes through the wrapped :class:`~requisite.agents.agent.Agent`'s own
``run()``/``arun()``, via a ``crewai.llms.base_llm.BaseLLM`` subclass
(built lazily -- see :meth:`CrewAIOrchestrator._require_crewai` -- since
``BaseLLM`` only exists once ``crewai`` is actually installed) that
proxies ``call()``/``acall()`` back to it. This mirrors
:class:`~requisite.orchestrators.langgraph_orchestrator.LangGraphOrchestrator`'s
own node functions, which call ``agent.run(...)`` directly rather than
handing execution to langgraph's own machinery. Switching to
``workflow.use_crewai()`` therefore never changes *which*
provider/model/tools an agent uses -- only who's coordinating.

Install with: ``pip install crewai``

Notes
-----
Only the ``"sequential"`` strategy is supported (CrewAI's
``Process.sequential``, one ``Task`` per agent chained via
``context=[previous_task]`` -- CrewAI's own native mechanism for
feeding a prior task's output into the next, not manual state
threading). ``"hierarchical"`` (CrewAI's ``Process.hierarchical``) is
deliberately not implemented yet: it depends on CrewAI's own internal
delegation-tool protocol (``AgentTools``), which a text-in-text-out
proxy adapter can't transparently support without real tool-bridging
work -- see ``docs/adr/0027-crewai-autogen-orchestrator-backends.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult

if TYPE_CHECKING:
    from requisite.agents.agent import Agent, AgentResult

logger = logging.getLogger("requisite.orchestrators.crewai")


def _extract_task_text(messages: Any) -> str:
    """Extract the text CrewAI wants answered from its assembled message list.

    ``messages`` is ``str | list[LLMMessage]`` (a ``TypedDict`` with
    ``role``/``content``) -- the last message is whatever CrewAI just
    assembled for this call, including any ``context=[...]``-injected
    prior task output, so forwarding it whole to the wrapped Agent
    preserves CrewAI's own native sequential-chaining behavior.
    """
    if isinstance(messages, str):
        return messages
    if not messages:
        return ""
    return str(messages[-1].get("content", ""))


class CrewAIOrchestrator(BaseOrchestrator):
    """Runs agents as a CrewAI ``Crew`` (``Process.sequential`` only). See module docstring."""

    @property
    def name(self) -> str:
        return "crewai"

    def _require_crewai(self) -> tuple[Any, Any, Any, Any, type]:
        try:
            from crewai import Agent as CrewAgent
            from crewai import Crew, Process, Task
            from crewai.llms.base_llm import BaseLLM
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'crewai' package is required to use the crewai orchestrator. "
                "Install it with: pip install crewai",
            ) from exc

        from pydantic import PrivateAttr

        class _RequisiteLLM(BaseLLM):  # type: ignore[misc]
            """Proxies every CrewAI LLM call back to one wrapped Requisite ``Agent``.

            ``tools=``/``available_functions=`` (CrewAI's own tool-calling
            protocol) are deliberately ignored -- the wrapped ``Agent``
            already runs its own tool loop via ``run()``/``arun()``.
            """

            _requisite_agent: Any = PrivateAttr()
            _collected_results: list = PrivateAttr(default_factory=list)  # type: ignore[type-arg]

            def __init__(self, agent: "Agent", **data: Any) -> None:
                provider = agent.ai.provider
                super().__init__(model=f"requisite/{provider.name}/{provider.model}", **data)
                self._requisite_agent = agent
                self._collected_results = []

            def call(
                self,
                messages: Any,
                tools: Optional[list[Any]] = None,
                callbacks: Optional[list[Any]] = None,
                available_functions: Optional[dict[str, Any]] = None,
                from_task: Optional[Any] = None,
                from_agent: Optional[Any] = None,
                response_model: Optional[type[Any]] = None,
            ) -> str:
                result = self._requisite_agent.run(_extract_task_text(messages))
                self._collected_results.append(result)
                return str(result.content)

            async def acall(
                self,
                messages: Any,
                tools: Optional[list[Any]] = None,
                callbacks: Optional[list[Any]] = None,
                available_functions: Optional[dict[str, Any]] = None,
                from_task: Optional[Any] = None,
                from_agent: Optional[Any] = None,
                response_model: Optional[type[Any]] = None,
            ) -> str:
                result = await self._requisite_agent.arun(_extract_task_text(messages))
                self._collected_results.append(result)
                return str(result.content)

        return CrewAgent, Task, Crew, Process, _RequisiteLLM

    def _build_crew(
        self,
        steps: Sequence["Agent"],
        input: str,  # noqa: A002
    ) -> tuple[Any, list[Any]]:
        CrewAgent, Task, Crew, Process, RequisiteLLM = self._require_crewai()  # noqa: N806

        llms = [RequisiteLLM(agent) for agent in steps]
        crew_agents = [
            CrewAgent(
                role=agent.name,
                goal=f"Complete tasks assigned to '{agent.name}' as part of a larger workflow.",
                backstory=f"'{agent.name}' is a specialized agent in a Requisite Workflow.",
                llm=llm,
            )
            for agent, llm in zip(steps, llms)
        ]

        tasks = []
        previous_task = None
        for index, crew_agent in enumerate(crew_agents):
            task = Task(
                description=input if index == 0 else "Continue the task using the context above.",
                expected_output="A complete, direct answer.",
                agent=crew_agent,
                context=[previous_task] if previous_task is not None else None,
            )
            tasks.append(task)
            previous_task = task

        # tracing=False, explicit rather than left at CrewAI's own
        # environment/user-settings-dependent default: a Requisite-embedded
        # Crew shouldn't prompt for (or silently depend on) CrewAI's own
        # opt-in telemetry decision -- verified live that leaving this
        # unset prints an interactive-looking "Tracing Preference Saved"
        # banner on every run in a non-tty environment.
        crew = Crew(agents=crew_agents, tasks=tasks, process=Process.sequential, tracing=False)
        return crew, llms

    @staticmethod
    def _collect_steps(llms: Sequence[Any]) -> list["AgentResult"]:
        steps: list["AgentResult"] = []
        for llm in llms:
            steps.extend(llm._collected_results)
        return steps

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
        if strategy != "sequential":
            raise ConfigurationException(
                f"The crewai orchestrator supports the 'sequential' strategy only "
                f"(got '{strategy}').",
            )

        crew, llms = self._build_crew(steps, input)
        crew_output = crew.kickoff()
        return WorkflowResult(
            content=crew_output.raw,
            steps=self._collect_steps(llms),
            orchestrator=self.name,
            strategy=strategy,
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
        if strategy != "sequential":
            raise ConfigurationException(
                f"The crewai orchestrator supports the 'sequential' strategy only "
                f"(got '{strategy}').",
            )

        crew, llms = self._build_crew(steps, input)
        crew_output = await crew.akickoff()
        return WorkflowResult(
            content=crew_output.raw,
            steps=self._collect_steps(llms),
            orchestrator=self.name,
            strategy=strategy,
        )
