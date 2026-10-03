"""
Strands Agents orchestrator backend.

Executes the same list of agents as :class:`~requisite.orchestrators.native.NativeOrchestrator`,
but runs them through `Strands Agents <https://github.com/strands-agents/sdk-python>`_
(``strands-agents``, AWS's open-source agent SDK). The public
:class:`~requisite.workflows.workflow.Workflow` API is identical either
way -- switching backends via ``workflow.use_strands()`` never changes how
you call ``.add()`` or ``.run()``.

Strands is used for coordination only -- every actual model call still
goes through the wrapped :class:`~requisite.agents.agent.Agent`'s own
``run()``/``arun()``, via a ``strands.models.Model`` subclass (built
lazily -- see :meth:`StrandsOrchestrator._require_sdk` -- since that ABC
only exists once ``strands-agents`` is installed) that proxies
``stream()`` back to it. Despite its AWS origin this never calls Bedrock:
any provider an agent is configured with keeps working, exactly as with
the other coordination-only backends (see
``docs/adr/0027-crewai-autogen-orchestrator-backends.md``).

Install with: ``pip install strands-agents``

Notes
-----
Two strategies are supported. ``"sequential"`` is a real Strands
``GraphBuilder`` graph -- one node per step, a linear chain of edges --
so Strands' own graph executor decides when each step runs and what it
sees (the upstream node's output, assembled by Strands). ``"supervisor"``
reuses the native backend's delegation loop (see
:mod:`requisite.orchestrators._sdk_adapter`), with each worker executed
as a Strands ``Agent``. Every Strands ``Agent`` is built with
``callback_handler=None``: its default handler prints each streamed
response to stdout, which would duplicate Requisite's own output. See
``docs/adr/0040-agent-sdk-orchestrator-backends.md``.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger("requisite.orchestrators.strands")


def _extract_task_text(messages: Any) -> str:
    """Extract the text Strands wants answered from its assembled ``Messages``.

    Each message is ``{"role": ..., "content": [{"text": ...}, ...]}``; the
    last ``user`` message is whatever Strands just built for this turn
    (including, for a graph node, the upstream nodes' outputs).
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        return "".join(block["text"] for block in message.get("content", []) if "text" in block)
    return ""


class StrandsOrchestrator(BaseOrchestrator):
    """Runs agents through Strands Agents (``"sequential"`` as a ``Graph``,
    ``"supervisor"`` via the shared delegation loop). See module docstring.
    """

    @property
    def name(self) -> str:
        return "strands"

    def _require_sdk(self) -> dict[str, Any]:
        try:
            from strands import Agent as SdkAgent
            from strands.models import Model
            from strands.multiagent import GraphBuilder
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'strands-agents' package is required to use the strands orchestrator. "
                "Install it with: pip install strands-agents",
            ) from exc

        class _RequisiteModel(Model):  # type: ignore[misc]
            """Proxies every Strands model call back to one wrapped Requisite ``Agent``.

            Strands' own ``tool_specs`` are deliberately ignored -- the
            wrapped ``Agent`` already runs its own tool loop via
            ``run()``/``arun()``, and no tools are attached to the Strands
            agent wrapping this model.
            """

            def __init__(self, agent: "Agent") -> None:
                self._requisite_agent = agent
                self._collected_results: list[Any] = []
                self._config: dict[str, Any] = {}

            def update_config(self, **model_config: Any) -> None:
                self._config.update(model_config)

            def get_config(self) -> Any:
                return dict(self._config)

            async def structured_output(
                self,
                output_model: Any,
                prompt: Any,
                system_prompt: Optional[str] = None,
                **kwargs: Any,
            ) -> Any:
                # Never exercised: only Agent.structured_output() calls this,
                # and the orchestrator never uses it.
                raise NotImplementedError(
                    "The strands orchestrator doesn't support structured_output(); "
                    "use the wrapped Requisite Agent's own response_model support."
                )
                yield  # pragma: no cover - makes this an async generator, like the ABC's

            async def stream(
                self,
                messages: Any,
                tool_specs: Any = None,
                system_prompt: Optional[str] = None,
                **kwargs: Any,
            ) -> Any:
                result = await self._requisite_agent.arun(_extract_task_text(messages))
                self._collected_results.append(result)

                raw = result.raw_response
                usage = {
                    "inputTokens": raw.usage.prompt_tokens if raw is not None else 0,
                    "outputTokens": raw.usage.completion_tokens if raw is not None else 0,
                    "totalTokens": raw.usage.total_tokens if raw is not None else 0,
                }
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": str(result.content)}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "end_turn"}}
                yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 0}}}

        return {
            "Agent": SdkAgent,
            "GraphBuilder": GraphBuilder,
            "RequisiteModel": _RequisiteModel,
        }

    @staticmethod
    def _sdk_agent(sdk: dict[str, Any], agent: "Agent", model: Any) -> Any:
        # callback_handler=None: Strands' default handler prints every
        # streamed response to stdout -- see module docstring.
        return sdk["Agent"](model=model, name=agent.name, callback_handler=None)

    def _build_delegate(self, sdk: dict[str, Any], agent: "Agent") -> SdkDelegate:
        model = sdk["RequisiteModel"](agent)

        async def _run(task: str) -> Any:
            # A fresh Strands agent per call: it accumulates conversation
            # history across invocations, which the proxy ignores anyway.
            return await self._sdk_agent(sdk, agent, model).invoke_async(task)

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
            raise unsupported_strategy("strands", strategy)

        models = [sdk["RequisiteModel"](agent) for agent in steps]
        # Strands labels a node's upstream input with the node id, so use the
        # agent's own name (suffixed only if two steps share one -- node ids
        # must be unique).
        names = [agent.name for agent in steps]
        node_ids = [
            name if names.count(name) == 1 else f"{name}_{index}"
            for index, name in enumerate(names)
        ]
        builder = sdk["GraphBuilder"]()
        for node_id, agent, model in zip(node_ids, steps, models):
            builder.add_node(self._sdk_agent(sdk, agent, model), node_id)
        for upstream, downstream in zip(node_ids, node_ids[1:]):
            builder.add_edge(upstream, downstream)
        builder.set_entry_point(node_ids[0])
        # A linear chain runs each node exactly once; stating that also
        # silences Strands' "no execution limits" warning on every run.
        builder.set_max_node_executions(len(steps))

        await builder.build().invoke_async(input)

        results = collect_steps(models)
        return WorkflowResult(
            content=str(results[-1].content) if results else "",
            steps=results,
            orchestrator=self.name,
            strategy=strategy,
        )
