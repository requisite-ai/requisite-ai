"""
ADK orchestrator backend.

Executes the same list of agents as :class:`~requisite.orchestrators.native.NativeOrchestrator`,
but delegates coordination to Google's `Agent Development Kit
<https://github.com/google/adk-python>`_ (``google-adk``). The public
:class:`~requisite.workflows.workflow.Workflow` API is identical either
way -- switching backends via ``workflow.use_adk()`` never changes how
you call ``.add()`` or ``.run()``.

ADK is used for coordination only -- every actual model call still goes
through the wrapped :class:`~requisite.agents.agent.Agent`'s own
``run()``/``arun()``, via a ``google.adk.models.base_llm.BaseLlm``
subclass (built lazily -- see :meth:`AdkOrchestrator._require_adk` --
since that class only exists once ``google-adk`` is installed) that
proxies ``generate_content_async()`` back to it. This mirrors
:class:`~requisite.orchestrators.langgraph_orchestrator.LangGraphOrchestrator`'s
own node functions, which call ``agent.run(...)`` directly rather than
handing execution to langgraph's own machinery -- the same
coordination-only precedent
:class:`~requisite.orchestrators.crewai_orchestrator.CrewAIOrchestrator`
and :class:`~requisite.orchestrators.autogen_orchestrator.AutoGenOrchestrator`
already follow. Switching to ``workflow.use_adk()`` therefore never
changes *which* provider/model/tools an agent uses -- only who's
coordinating.

Install with: ``pip install google-adk``

Note: ``google.adk.workflow.Workflow`` is an internal ADK class with the
same name as :class:`~requisite.workflows.workflow.Workflow` -- a real
class-name collision, but a different fully-qualified path, so there is
no actual import conflict.

Notes
-----
Two strategies are supported: ``"sequential"`` (a custom
``google.adk.agents.BaseAgent`` that runs each step's ``LlmAgent`` in
order -- not ADK's own ``SequentialAgent``, which is deprecated in favor
of a newer, internal-feeling ``Workflow`` system) and ``"supervisor"`` (a
custom ``BaseAgent`` that reuses
:meth:`~requisite.orchestrators.native.NativeOrchestrator._split_coordinator_and_workers`,
``_SupervisorDecision``, ``_supervisor_prompt``, and ``_resolve_delegate``
directly -- the exact same decision protocol
:class:`LangGraphOrchestrator` and :class:`AutoGenOrchestrator` already
reuse, applied to a fourth execution engine instead of reinvented). The
coordinator and each delegate call Requisite's own ``Agent.run()``/
``.arun()`` directly inside the custom agent's ``_run_async_impl`` --
unlike ``"sequential"``, no ``BaseLlm``/``LlmAgent`` wrapping is used for
``"supervisor"``, since there's no ADK LLM-call machinery worth
round-tripping through for a strategy that's already a plain decision
loop. See ``docs/adr/0039-adk-orchestrator-backend.md``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult
from requisite.orchestrators.native import (
    NativeOrchestrator,
    _SupervisorDecision,
    _supervisor_prompt,
)

if TYPE_CHECKING:
    from requisite.agents.agent import Agent, AgentResult

logger = logging.getLogger("requisite.orchestrators.adk")


def _extract_task_text(contents: Any) -> str:
    """Extract the text ADK wants answered from its assembled ``LlmRequest.contents``.

    Unlike CrewAI's/AutoGen's own message formats, ``contents`` is a list
    of ``google.genai.types.Content`` (Gemini-SDK-shaped: ``role`` +
    ``parts``, where each part may carry ``.text``, ``.function_call``,
    etc.). The last ``role == "user"`` entry is whatever ADK just
    assembled for this turn, so its text parts are forwarded whole to the
    wrapped Agent -- the same "forward the last assembled message" shape
    CrewAI's/AutoGen's own adapters already use, just translated out of
    ``genai.types`` first.
    """
    for content in reversed(contents):
        if content.role == "user":
            return "".join(part.text or "" for part in (content.parts or []) if part.text)
    return ""


class AdkOrchestrator(BaseOrchestrator):
    """Runs agents as an ADK agent tree (``"sequential"`` and ``"supervisor"``
    only). See module docstring.
    """

    @property
    def name(self) -> str:
        return "adk"

    def _require_adk(self) -> dict[str, Any]:
        try:
            from google.adk.agents import BaseAgent, LlmAgent
            from google.adk.events import Event
            from google.adk.models.base_llm import BaseLlm
            from google.adk.runners import InMemoryRunner
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'google-adk' package is required to use the adk orchestrator. "
                "Install it with: pip install google-adk",
            ) from exc

        from pydantic import PrivateAttr

        class _RequisiteLlm(BaseLlm):  # type: ignore[misc]
            """Proxies every ADK model call back to one wrapped Requisite ``Agent``.

            ADK's own ``tools``/function-calling protocol is deliberately
            never exercised -- the wrapped ``Agent`` already runs its own
            tool loop via ``run()``/``arun()``, and no tools are attached
            to the ``LlmAgent`` wrapping this model. ``stream`` is never
            ``True`` in this integration (no participant is ever
            constructed with streaming enabled), so it's accepted but
            never branched on.
            """

            _requisite_agent: Any = PrivateAttr()
            _collected_results: list = PrivateAttr(default_factory=list)  # type: ignore[type-arg]

            def __init__(self, agent: "Agent", **data: Any) -> None:
                super().__init__(model=f"requisite/{agent.name}", **data)
                self._requisite_agent = agent
                self._collected_results = []

            async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
                from google.adk.models.llm_response import LlmResponse

                task_text = _extract_task_text(llm_request.contents)
                result = await self._requisite_agent.arun(task_text)
                self._collected_results.append(result)
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model", parts=[genai_types.Part(text=str(result.content))]
                    ),
                )

        class _RequisiteSequentialAgent(BaseAgent):  # type: ignore[misc]
            """Runs ``sub_agents`` (each an ``LlmAgent`` wrapping a
            ``_RequisiteLlm``) in order.

            Not ADK's own ``SequentialAgent`` -- see module docstring.
            This is the exact same one-line loop ``SequentialAgent``
            contains internally; using the deprecated class would buy
            nothing a direct ``BaseAgent`` subclass doesn't already give,
            and keeps one consistent extension point across both
            strategies in this backend.
            """

            async def _run_async_impl(self, ctx: Any) -> Any:
                for sub_agent in self.sub_agents:
                    async for event in sub_agent.run_async(ctx):
                        yield event

        class _RequisiteSupervisorAgent(BaseAgent):  # type: ignore[misc]
            """Reuses Native's supervisor decision protocol directly inside
            ``_run_async_impl`` -- no ``_RequisiteLlm``/``LlmAgent``
            involved. See module docstring.
            """

            _coordinator: Any = PrivateAttr()
            _workers: Any = PrivateAttr()
            _task: Any = PrivateAttr()
            _max_rounds: int = PrivateAttr()
            _collected_results: list = PrivateAttr(default_factory=list)  # type: ignore[type-arg]

            def __init__(
                self,
                *,
                name: str,
                coordinator: "Agent",
                workers: dict[str, "Agent"],
                task: str,
                max_rounds: int,
                **data: Any,
            ) -> None:
                super().__init__(name=name, **data)
                self._coordinator = coordinator
                self._workers = workers
                self._task = task
                self._max_rounds = max_rounds
                self._collected_results = []

            async def _run_async_impl(self, ctx: Any) -> Any:
                transcript: list[tuple[str, str, str]] = []
                for _round in range(self._max_rounds):
                    decision = await self._coordinator.ai.achat(
                        _supervisor_prompt(self._task, list(self._workers), transcript),
                        response_model=_SupervisorDecision,
                    )
                    if decision.action == "finish":
                        yield Event(
                            invocation_id=ctx.invocation_id,
                            author=self.name,
                            content=genai_types.Content(
                                role="model",
                                parts=[genai_types.Part(text=decision.final_answer or "")],
                            ),
                        )
                        return

                    delegate = NativeOrchestrator._resolve_delegate(
                        decision, supervisor_name=self._coordinator.name, workers=self._workers
                    )
                    subtask = decision.task or self._task
                    result = await delegate.arun(subtask)
                    self._collected_results.append(result)
                    transcript.append((delegate.name, subtask, result.content))

                raise AgentException(
                    f"Workflow supervisor '{self._coordinator.name}' exceeded "
                    f"max_rounds={self._max_rounds} without reaching a final answer.",
                )

        return {
            "BaseAgent": BaseAgent,
            "LlmAgent": LlmAgent,
            "InMemoryRunner": InMemoryRunner,
            "genai_types": genai_types,
            "RequisiteLlm": _RequisiteLlm,
            "RequisiteSequentialAgent": _RequisiteSequentialAgent,
            "RequisiteSupervisorAgent": _RequisiteSupervisorAgent,
        }

    @staticmethod
    def _collect_steps(llms: Sequence[Any]) -> list["AgentResult"]:
        steps: list["AgentResult"] = []
        for llm in llms:
            steps.extend(llm._collected_results)
        return steps

    def _build_sequential_agent(
        self, adk: dict[str, Any], steps: Sequence["Agent"]
    ) -> tuple[Any, list[Any]]:
        llms = [adk["RequisiteLlm"](agent) for agent in steps]
        sub_agents = [
            adk["LlmAgent"](name=agent.name, model=llm) for agent, llm in zip(steps, llms)
        ]
        root = adk["RequisiteSequentialAgent"](
            name="__requisite_sequential__", sub_agents=sub_agents
        )
        return root, llms

    def _build_supervisor_agent(
        self,
        adk: dict[str, Any],
        steps: Sequence["Agent"],
        *,
        input: str,
        max_rounds: int,  # noqa: A002
    ) -> Any:
        coordinator, workers = NativeOrchestrator._split_coordinator_and_workers(
            steps, role="supervisor"
        )
        return adk["RequisiteSupervisorAgent"](
            name="__requisite_supervisor__",
            coordinator=coordinator,
            workers=workers,
            task=input,
            max_rounds=max_rounds,
        )

    def run(
        self,
        steps: Sequence[Any],
        input: Optional[str],  # noqa: A002
        *,
        strategy: str = "sequential",
        **kwargs: Any,
    ) -> WorkflowResult:
        # ADK's Runner requires an async session-creation call with no
        # non-deprecated sync counterpart (InMemorySessionService's own
        # create_session_sync logs a deprecation warning on every call,
        # confirmed directly in its source) -- asyncio.run() here matches
        # AutoGenOrchestrator.run()'s own reasoning for the same shape.
        return asyncio.run(self.arun(steps, input, strategy=strategy, **kwargs))

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

        adk = self._require_adk()

        if strategy == "sequential":
            root_agent, llms = self._build_sequential_agent(adk, steps)
        elif strategy == "supervisor":
            max_rounds = kwargs.pop("max_rounds", 6)
            root_agent = self._build_supervisor_agent(
                adk, steps, input=input, max_rounds=max_rounds
            )
            llms = [root_agent]
        else:
            raise ConfigurationException(
                f"The adk orchestrator supports the 'sequential' and 'supervisor' "
                f"strategies (got '{strategy}').",
            )

        runner = adk["InMemoryRunner"](agent=root_agent)
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="requisite"
        )
        final_text = ""
        async for event in runner.run_async(
            user_id="requisite",
            session_id=session.id,
            new_message=adk["genai_types"].Content(
                role="user", parts=[adk["genai_types"].Part(text=input)]
            ),
        ):
            if event.content and event.content.parts:
                text = "".join(part.text or "" for part in event.content.parts if part.text)
                if text:
                    final_text = text

        return WorkflowResult(
            content=final_text,
            steps=self._collect_steps(llms),
            orchestrator=self.name,
            strategy=strategy,
        )
