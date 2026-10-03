"""
Microsoft Agent Framework orchestrator backend.

Executes the same list of agents as :class:`~requisite.orchestrators.native.NativeOrchestrator`,
but runs them through Microsoft's `Agent Framework
<https://github.com/microsoft/agent-framework>`_ (``agent-framework-core``,
the successor to Semantic Kernel's agents and AutoGen). The public
:class:`~requisite.workflows.workflow.Workflow` API is identical either
way -- switching backends via ``workflow.use_agent_framework()`` never
changes how you call ``.add()`` or ``.run()``.

The framework is used for coordination only -- every actual model call
still goes through the wrapped :class:`~requisite.agents.agent.Agent`'s
own ``run()``/``arun()``, via an ``agent_framework.BaseChatClient``
subclass (built lazily -- see
:meth:`AgentFrameworkOrchestrator._require_sdk` -- since that ABC only
exists once the package is installed) that proxies
``_inner_get_response()`` back to it. Despite its Azure/OpenAI lineage
this never calls either: any provider an agent is configured with keeps
working, exactly as with the other coordination-only backends (see
``docs/adr/0027-crewai-autogen-orchestrator-backends.md``).

Install with: ``pip install agent-framework-core`` -- deliberately the
``-core`` package, not the ``agent-framework`` umbrella, which installs
every connector (Azure, Redis, Mistral, Bedrock, ...) for 120+ packages
versus ``-core``'s 12.

Notes
-----
Two strategies are supported. ``"sequential"`` is a real
``WorkflowBuilder`` workflow -- one agent per step, a linear chain of
edges -- so the framework's own workflow engine decides when each step
runs. Its agents share one conversation: a later agent receives the
original user task followed by earlier agents' replies as *assistant*
messages, not as a rewritten "last user message", so the adapter rebuilds
the task and its prior-agent context from the whole transcript (see
:func:`_extract_task_text`) rather than just forwarding the last user
message the way the other backends can. ``"supervisor"`` reuses the
native backend's delegation loop (see
:mod:`requisite.orchestrators._sdk_adapter`), with each worker executed as
an ``agent_framework.Agent``. See
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
from requisite.orchestrators.native import NativeOrchestrator

if TYPE_CHECKING:
    from requisite.agents.agent import Agent

logger = logging.getLogger("requisite.orchestrators.agent_framework")


def _extract_task_text(messages: Any) -> str:
    """Rebuild what an agent should answer from the framework's shared transcript.

    The task is the last ``user`` message; any ``assistant`` replies after
    it are earlier agents' output in the same workflow (each carrying its
    author's name), folded in through the same
    ``NativeOrchestrator._task_prompt_with_context`` helper the native
    ``planner`` strategy uses -- so a worker sees the same
    "task + context from previous steps" shape as on the native backend.
    """
    transcript = list(messages)
    user_index = next(
        (i for i in range(len(transcript) - 1, -1, -1) if transcript[i].role == "user"), None
    )
    if user_index is None:
        return ""
    notes = [
        f"[{message.author_name or 'assistant'}] {message.text}"
        for message in transcript[user_index + 1 :]
        if message.role == "assistant" and message.text
    ]
    return NativeOrchestrator._task_prompt_with_context(transcript[user_index].text, notes)


class AgentFrameworkOrchestrator(BaseOrchestrator):
    """Runs agents through Microsoft Agent Framework (``"sequential"`` as a
    ``Workflow``, ``"supervisor"`` via the shared delegation loop). See
    module docstring.
    """

    @property
    def name(self) -> str:
        return "agent_framework"

    def _require_sdk(self) -> dict[str, Any]:
        try:
            from agent_framework import (
                Agent as SdkAgent,
            )
            from agent_framework import (
                BaseChatClient,
                ChatResponse,
                FunctionInvocationLayer,
                Message,
                WorkflowBuilder,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'agent-framework-core' package is required to use the agent_framework "
                "orchestrator. Install it with: pip install agent-framework-core",
            ) from exc

        # FunctionInvocationLayer is mixed in the way the framework's own
        # public clients compose it -- without it the framework logs "chat
        # client does not support function invoking" on every Agent built.
        # It's inert here: no tools are ever attached to these agents.
        class _RequisiteChatClient(FunctionInvocationLayer, BaseChatClient):  # type: ignore[misc]
            """Proxies every framework model call back to one wrapped Requisite ``Agent``.

            The framework's own tool-calling protocol is deliberately never
            exercised -- the wrapped ``Agent`` already runs its own tool
            loop via ``run()``/``arun()``. ``stream=True`` is never
            requested by this orchestrator (it only ever awaits ``run()``).
            """

            def __init__(self, agent: "Agent") -> None:
                super().__init__()
                self._requisite_agent = agent
                self._collected_results: list[Any] = []

            def _inner_get_response(
                self, *, messages: Any, stream: bool, options: Any, **kwargs: Any
            ) -> Any:
                if stream:
                    raise NotImplementedError(
                        "The agent_framework orchestrator never streams; it only awaits run()."
                    )

                async def _respond() -> Any:
                    result = await self._requisite_agent.arun(_extract_task_text(messages))
                    self._collected_results.append(result)
                    raw = result.raw_response
                    usage = (
                        {
                            "input_token_count": raw.usage.prompt_tokens,
                            "output_token_count": raw.usage.completion_tokens,
                            "total_token_count": raw.usage.total_tokens,
                        }
                        if raw is not None
                        else None
                    )
                    return ChatResponse(
                        messages=[Message(role="assistant", contents=[str(result.content)])],
                        usage_details=usage,
                    )

                return _respond()

        return {
            "Agent": SdkAgent,
            "WorkflowBuilder": WorkflowBuilder,
            "RequisiteChatClient": _RequisiteChatClient,
        }

    def _build_delegate(self, sdk: dict[str, Any], agent: "Agent") -> SdkDelegate:
        client = sdk["RequisiteChatClient"](agent)

        async def _run(task: str) -> Any:
            # A fresh framework agent per call: it keeps conversation state
            # across runs, which the proxy rebuilds from scratch anyway.
            return await sdk["Agent"](client=client, name=agent.name).run(task)

        return SdkDelegate(agent.name, _run, client._collected_results)

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
            raise unsupported_strategy("agent_framework", strategy)

        clients = [sdk["RequisiteChatClient"](agent) for agent in steps]
        # Executor ids must be unique within a workflow: suffix only on a clash.
        names = [agent.name for agent in steps]
        sdk_agents = [
            sdk["Agent"](client=client, name=name if names.count(name) == 1 else f"{name}_{index}")
            for index, (name, client) in enumerate(zip(names, clients))
        ]
        builder = sdk["WorkflowBuilder"](start_executor=sdk_agents[0])
        for upstream, downstream in zip(sdk_agents, sdk_agents[1:]):
            builder.add_edge(upstream, downstream)

        await builder.build().run(input)

        results = collect_steps(clients)
        return WorkflowResult(
            content=str(results[-1].content) if results else "",
            steps=results,
            orchestrator=self.name,
            strategy=strategy,
        )
