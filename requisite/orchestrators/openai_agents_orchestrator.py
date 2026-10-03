"""
OpenAI Agents SDK orchestrator backend.

Executes the same list of agents as :class:`~requisite.orchestrators.native.NativeOrchestrator`,
but runs each one through the `OpenAI Agents SDK
<https://github.com/openai/openai-agents-python>`_ (``openai-agents``).
The public :class:`~requisite.workflows.workflow.Workflow` API is
identical either way -- switching backends via
``workflow.use_openai_agents()`` never changes how you call ``.add()`` or
``.run()``.

The SDK is used for coordination only -- every actual model call still
goes through the wrapped :class:`~requisite.agents.agent.Agent`'s own
``run()``/``arun()``, via an ``agents.models.interface.Model`` subclass
(built lazily -- see :meth:`OpenAIAgentsOrchestrator._require_sdk` --
since that ABC only exists once ``openai-agents`` is installed) that
proxies ``get_response()`` back to it. Despite the name, this never calls
OpenAI: any provider an agent is configured with keeps working, exactly
as with the other coordination-only backends (see
``docs/adr/0027-crewai-autogen-orchestrator-backends.md``).

Install with: ``pip install openai-agents``

Notes
-----
Two strategies are supported. ``"sequential"`` chains ``Runner.run(...)``
calls, feeding each step's final output in as the next step's input --
the SDK's own documented pattern for deterministic agent chaining.
``"supervisor"`` reuses the native backend's delegation loop (see
:mod:`requisite.orchestrators._sdk_adapter`), with each worker executed
through ``Runner.run(...)``. SDK tracing is switched off
(``RunConfig(tracing_disabled=True)``) in both: the SDK otherwise uploads
traces, including prompts and outputs, to OpenAI's platform by default --
an unexpected side effect for a Requisite user running, say, Ollama. See
``docs/adr/0040-agent-sdk-orchestrator-backends.md``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.core.sync_bridge import run_sync
from requisite.orchestrators._sdk_adapter import (
    SdkDelegate,
    check_run_inputs,
    collect_steps,
    run_supervisor,
    unsupported_strategy,
)
from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult

if TYPE_CHECKING:
    from requisite.agents.agent import Agent

logger = logging.getLogger("requisite.orchestrators.openai_agents")


def _part_text(part: Any) -> str:
    text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
    return text if isinstance(text, str) else ""


def _extract_task_text(items: Any) -> str:
    """Extract the text the SDK wants answered from ``Model.get_response``'s ``input``.

    ``input`` is a plain string, or a list of Responses-API input items
    (dicts or objects with ``role``/``content``). The last ``user`` item
    is whatever the SDK just assembled for this turn.
    """
    if isinstance(items, str):
        return items
    for item in reversed(items):
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        if role != "user":
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if isinstance(content, str):
            return content
        return "".join(_part_text(part) for part in content or [])
    return ""


class OpenAIAgentsOrchestrator(BaseOrchestrator):
    """Runs agents through the OpenAI Agents SDK (``"sequential"`` and
    ``"supervisor"`` only). See module docstring.
    """

    @property
    def name(self) -> str:
        return "openai_agents"

    def _require_sdk(self) -> dict[str, Any]:
        try:
            from agents import Agent as SdkAgent
            from agents import RunConfig, Runner
            from agents.items import ModelResponse
            from agents.models.interface import Model
            from agents.usage import Usage
            from openai.types.responses import ResponseOutputMessage, ResponseOutputText
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'openai-agents' package is required to use the openai_agents "
                "orchestrator. Install it with: pip install openai-agents",
            ) from exc

        class _RequisiteModel(Model):  # type: ignore[misc]
            """Proxies every SDK model call back to one wrapped Requisite ``Agent``.

            The SDK's own tools/handoffs/output-schema arguments are
            deliberately ignored -- the wrapped ``Agent`` already runs its
            own tool loop via ``run()``/``arun()``, and none are attached
            to the SDK agent wrapping this model.
            """

            def __init__(self, agent: "Agent") -> None:
                self._requisite_agent = agent
                self._collected_results: list[Any] = []

            async def get_response(
                self,
                system_instructions: Any,
                input: Any,  # noqa: A002
                model_settings: Any,
                tools: Any,
                output_schema: Any,
                handoffs: Any,
                tracing: Any,
                *,
                previous_response_id: Any = None,
                conversation_id: Any = None,
                prompt: Any = None,
            ) -> Any:
                result = await self._requisite_agent.arun(_extract_task_text(input))
                self._collected_results.append(result)

                raw = result.raw_response
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=raw.usage.prompt_tokens,
                        output_tokens=raw.usage.completion_tokens,
                        total_tokens=raw.usage.total_tokens,
                    )
                    if raw is not None
                    else Usage()
                )
                message = ResponseOutputMessage(
                    id=f"msg_{uuid.uuid4().hex}",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            text=str(result.content), type="output_text", annotations=[]
                        )
                    ],
                )
                return ModelResponse(output=[message], usage=usage, response_id=None)

            def stream_response(self, *args: Any, **kwargs: Any) -> Any:
                # Never exercised: only Runner.run_streamed() calls this, and
                # this orchestrator only ever uses Runner.run().
                raise NotImplementedError(
                    "The openai_agents orchestrator never streams; it only uses Runner.run()."
                )

        return {
            "Agent": SdkAgent,
            "Runner": Runner,
            "RunConfig": RunConfig,
            "RequisiteModel": _RequisiteModel,
        }

    @staticmethod
    def _run_config(sdk: dict[str, Any]) -> Any:
        # tracing_disabled=True, explicit: the SDK's default uploads traces
        # (prompts and outputs) to OpenAI's platform -- see module docstring.
        return sdk["RunConfig"](tracing_disabled=True)

    def _build_delegate(self, sdk: dict[str, Any], agent: "Agent") -> SdkDelegate:
        model = sdk["RequisiteModel"](agent)
        sdk_agent = sdk["Agent"](name=agent.name, model=model)
        run_config = self._run_config(sdk)

        async def _run(task: str) -> Any:
            return await sdk["Runner"].run(sdk_agent, task, run_config=run_config)

        return SdkDelegate(agent.name, _run, model._collected_results)

    def run(
        self,
        steps: Sequence[Any],
        input: Optional[str],  # noqa: A002
        *,
        strategy: str = "sequential",
        **kwargs: Any,
    ) -> WorkflowResult:
        return run_sync(self.arun(steps, input, strategy=strategy, **kwargs))

    async def arun(
        self,
        steps: Sequence[Any],
        input: Optional[str],  # noqa: A002
        *,
        strategy: str = "sequential",
        **kwargs: Any,
    ) -> WorkflowResult:
        check_run_inputs(steps, input)
        assert input is not None
        sdk = self._require_sdk()

        if strategy == "supervisor":
            return await run_supervisor(
                backend=self.name,
                steps=steps,
                input=input,
                max_rounds=kwargs.pop("max_rounds", 6),
                build_delegate=lambda agent: self._build_delegate(sdk, agent),
            )
        if strategy != "sequential":
            raise unsupported_strategy("openai_agents", strategy)

        models = [sdk["RequisiteModel"](agent) for agent in steps]
        run_config = self._run_config(sdk)
        text = input
        for agent, model in zip(steps, models):
            sdk_agent = sdk["Agent"](name=agent.name, model=model)
            run = await sdk["Runner"].run(sdk_agent, text, run_config=run_config)
            text = str(run.final_output)

        return WorkflowResult(
            content=text,
            steps=collect_steps(models),
            orchestrator=self.name,
            strategy=strategy,
        )
