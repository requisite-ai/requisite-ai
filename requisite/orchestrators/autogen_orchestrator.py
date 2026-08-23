"""
AutoGen orchestrator backend.

Executes the same list of agents as :class:`~requisite.orchestrators.native.NativeOrchestrator`,
but delegates coordination to `AutoGen <https://github.com/microsoft/autogen>`_
(the ``autogen-agentchat``/``autogen-core`` packages). The public
:class:`~requisite.workflows.workflow.Workflow` API is identical either
way -- switching backends via ``workflow.use_autogen()`` never changes
how you call ``.add()`` or ``.run()``.

AutoGen is used for coordination only -- every actual model call still
goes through the wrapped :class:`~requisite.agents.agent.Agent`'s own
``run()``/``arun()``, via an ``autogen_core.models.ChatCompletionClient``
subclass (built lazily -- see :meth:`AutoGenOrchestrator._require_autogen`
-- since that ABC only exists once ``autogen-core`` is installed) that
proxies ``create()``/``create_stream()`` back to it. This mirrors
:class:`~requisite.orchestrators.langgraph_orchestrator.LangGraphOrchestrator`'s
own node functions, which call ``agent.run(...)`` directly rather than
handing execution to langgraph's own machinery. Switching to
``workflow.use_autogen()`` therefore never changes *which*
provider/model/tools an agent uses -- only who's coordinating.

Install with: ``pip install autogen-agentchat autogen-core``

Notes
-----
Two strategies are supported: ``"sequential"`` (AutoGen's
``RoundRobinGroupChat``, one turn per agent in order) and
``"supervisor"`` (AutoGen's ``SelectorGroupChat`` with a
``selector_func`` that reuses
:meth:`~requisite.orchestrators.native.NativeOrchestrator._split_coordinator_and_workers`,
``_SupervisorDecision``, ``_supervisor_prompt``, and ``_resolve_delegate``
directly -- the exact same decision protocol
:class:`LangGraphOrchestrator` already reuses for its own supervisor
graph, applied to a third execution engine instead of reinvented). The
selected worker's turn still flows through the proxy adapter, receiving
the coordinator's explicitly delegated subtask text -- not AutoGen's own
free-form group-conversation transcript -- matching Native/LangGraph's
"supervisor delegates a subtask" semantics rather than AutoGen's usual
group-chat idiom. See ``docs/adr/0027-crewai-autogen-orchestrator-backends.md``.
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

logger = logging.getLogger("requisite.orchestrators.autogen")

_CHARS_PER_TOKEN_ESTIMATE = 4
_ASSUMED_CONTEXT_WINDOW = 128_000


def _extract_task_text(messages: Any) -> str:
    """Extract the text AutoGen wants answered from its assembled message list.

    Each message is a pydantic object (``SystemMessage``/``UserMessage``/
    ``AssistantMessage``/...) with a ``.content`` attribute -- the last
    message is whatever AutoGen just assembled for this turn (the running
    conversation/delegated task), so forwarding it whole to the wrapped
    Agent preserves AutoGen's own turn-taking behavior.
    """
    if not messages:
        return ""
    content = messages[-1].content
    return content if isinstance(content, str) else str(content)


class AutoGenOrchestrator(BaseOrchestrator):
    """Runs agents as an AutoGen team (``RoundRobinGroupChat`` for
    ``"sequential"``, ``SelectorGroupChat`` for ``"supervisor"``). See
    module docstring.
    """

    @property
    def name(self) -> str:
        return "autogen"

    def _require_autogen(self) -> dict[str, Any]:
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.base import TerminationCondition
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.messages import StopMessage
            from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
            from autogen_core.models import (
                ChatCompletionClient,
                CreateResult,
                ModelInfo,
                RequestUsage,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'autogen-agentchat'/'autogen-core' packages are required to use the "
                "autogen orchestrator. Install them with: pip install autogen-agentchat autogen-core",
            ) from exc

        class _RequisiteChatCompletionClient(ChatCompletionClient):
            """Proxies every AutoGen model call back to one wrapped Requisite ``Agent``.

            AutoGen's own ``tools=``/tool-calling protocol is deliberately
            never exercised -- the wrapped ``Agent`` already runs its own
            tool loop via ``run()``/``arun()``, and no ``tools=`` are
            passed to the ``AssistantAgent`` wrapping this client.

            ``agent=None`` makes this a no-op sentinel client instead: used
            for the ``"supervisor"`` strategy's finishing turn (see
            :meth:`AutoGenOrchestrator._build_supervisor_team`), where
            ``SelectorGroupChat`` structurally requires *some* participant
            to take one more turn before it observes a termination
            condition -- rather than spend a real Requisite Agent call
            (and pollute ``WorkflowResult.steps``) on a turn whose content
            is discarded, this returns a fixed response with no real call.
            """

            def __init__(self, agent: Optional["Agent"]) -> None:
                self._requisite_agent = agent
                self._collected_results: list["AgentResult"] = []
                self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

            async def create(
                self,
                messages: Any,
                *,
                tools: Any = (),
                tool_choice: Any = "auto",
                json_output: Any = None,
                extra_create_args: Any = None,
                cancellation_token: Any = None,
            ) -> Any:
                if self._requisite_agent is None:
                    return CreateResult(
                        finish_reason="stop",
                        content="",
                        usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                        cached=False,
                    )
                result = await self._requisite_agent.arun(_extract_task_text(messages))
                self._collected_results.append(result)
                usage = self._result_usage(result)
                self._usage = RequestUsage(
                    prompt_tokens=self._usage.prompt_tokens + usage.prompt_tokens,
                    completion_tokens=self._usage.completion_tokens + usage.completion_tokens,
                )
                return CreateResult(
                    finish_reason="stop", content=result.content, usage=usage, cached=False
                )

            async def create_stream(self, messages: Any, **kwargs: Any) -> Any:
                # Never actually exercised: AssistantAgent only calls this
                # when constructed with model_client_stream=True, which the
                # orchestrator below never sets. Implemented anyway for a
                # correct ChatCompletionClient, not left raising.
                result = await self.create(messages, **kwargs)
                yield result.content
                yield result

            async def close(self) -> None:
                return None

            def actual_usage(self) -> Any:
                return self._usage

            def total_usage(self) -> Any:
                return self._usage

            def count_tokens(self, messages: Any, *, tools: Any = ()) -> int:
                # No real tokenizer available from Requisite's AI facade --
                # a documented approximation (see ADR-0027), used only for
                # AutoGen's own context-window bookkeeping, never for
                # billing/limits (Requisite's own RateLimiter/provider
                # handle those against the real API response).
                total_chars = sum(len(_extract_task_text([m])) for m in messages)
                return total_chars // _CHARS_PER_TOKEN_ESTIMATE

            def remaining_tokens(self, messages: Any, *, tools: Any = ()) -> int:
                return max(0, _ASSUMED_CONTEXT_WINDOW - self.count_tokens(messages, tools=tools))

            @property
            def model_info(self) -> Any:
                return ModelInfo(
                    vision=False,
                    function_calling=False,
                    json_output=False,
                    family="unknown",
                    structured_output=False,
                )

            @property
            def capabilities(self) -> Any:  # deprecated on the ABC; mirrors model_info
                return self.model_info

            @staticmethod
            def _result_usage(result: "AgentResult") -> Any:
                if result.raw_response is None:
                    return RequestUsage(prompt_tokens=0, completion_tokens=0)
                usage = result.raw_response.usage
                return RequestUsage(
                    prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
                )

        class _DecisionTermination(TerminationCondition):
            """Terminates once the coordinator's ``_SupervisorDecision`` is
            ``"finish"`` -- set externally by ``selector_func`` (see
            :meth:`AutoGenOrchestrator._run_supervisor`), not derived from
            AutoGen's own text-mention conventions.
            """

            def __init__(self) -> None:
                self._terminated = False

            @property
            def terminated(self) -> bool:
                return self._terminated

            def mark_finished(self) -> None:
                self._terminated = True

            async def __call__(self, messages: Any) -> Any:
                if self._terminated:
                    return StopMessage(content="Supervisor decided to finish.", source="supervisor")
                return None

            async def reset(self) -> None:
                self._terminated = False

        return {
            "AssistantAgent": AssistantAgent,
            "RoundRobinGroupChat": RoundRobinGroupChat,
            "SelectorGroupChat": SelectorGroupChat,
            "MaxMessageTermination": MaxMessageTermination,
            "RequisiteChatCompletionClient": _RequisiteChatCompletionClient,
            "DecisionTermination": _DecisionTermination,
        }

    @staticmethod
    def _collect_steps(clients: Sequence[Any]) -> list["AgentResult"]:
        steps: list["AgentResult"] = []
        for client in clients:
            steps.extend(client._collected_results)
        return steps

    def _build_sequential_team(self, autogen: dict[str, Any], steps: Sequence["Agent"]) -> Any:
        clients = [autogen["RequisiteChatCompletionClient"](agent) for agent in steps]
        participants = [
            autogen["AssistantAgent"](name=agent.name, model_client=client)
            for agent, client in zip(steps, clients)
        ]
        team = autogen["RoundRobinGroupChat"](participants, max_turns=len(steps))
        return team, clients

    def _build_supervisor_team(
        self, autogen: dict[str, Any], steps: Sequence["Agent"], *, max_rounds: int
    ) -> tuple[Any, list[Any], Any]:
        coordinator, workers = NativeOrchestrator._split_coordinator_and_workers(
            steps, role="supervisor"
        )
        if "__supervisor_finish__" in workers:
            raise ConfigurationException(
                "The 'supervisor' strategy on the autogen backend reserves "
                "'__supervisor_finish__' for its own internal finishing participant -- "
                "rename the worker '__supervisor_finish__' to something else.",
            )

        worker_clients = {
            name: autogen["RequisiteChatCompletionClient"](worker)
            for name, worker in workers.items()
        }
        worker_participants = {
            name: autogen["AssistantAgent"](name=name, model_client=client)
            for name, client in worker_clients.items()
        }
        coordinator_client = autogen["RequisiteChatCompletionClient"](coordinator)

        # A no-op sentinel participant + client (agent=None -- never calls a
        # real Requisite Agent) for the finishing turn: SelectorGroupChat
        # structurally requires selector_func to name a next speaker before
        # the termination condition is observed, so finishing still costs
        # one turn -- this makes that turn free instead of wasting a real
        # worker call. See _RequisiteChatCompletionClient's docstring.
        finisher_name = "__supervisor_finish__"
        finisher = autogen["AssistantAgent"](
            name=finisher_name, model_client=autogen["RequisiteChatCompletionClient"](None)
        )

        termination = autogen["DecisionTermination"]()
        transcript: list[tuple[str, str, str]] = []
        state: dict[str, Any] = {"pending_task": "", "rounds": 0}

        def _selector_func(messages: Any) -> Optional[str]:
            # The message just produced (if any) is the previously-selected
            # worker's real response -- fold it into the transcript before
            # deciding the next action, so the coordinator's prompt sees
            # delegation history exactly like Native/LangGraph's supervisor.
            if messages and getattr(messages[-1], "source", None) in workers:
                transcript.append(
                    (messages[-1].source, state["pending_task"], _extract_task_text([messages[-1]]))
                )

            # autogen-agentchat's own agent runtime catches any exception
            # raised from a selector_func call and re-raises it at the
            # team.run() call site as a generic RuntimeError -- losing the
            # original AgentException/ConfigurationException type Native
            # and LangGraph's supervisor both raise for the exact same
            # conditions (max_rounds exceeded, unknown delegate). Verified
            # live: a plain `raise AgentException(...)` here surfaces to
            # the caller as `RuntimeError: AgentException: ...`, not
            # `AgentException`. Caught and stashed here instead, so arun()
            # can re-raise the real exception itself after team.run()
            # returns, restoring the same exception contract every backend
            # already gives callers.
            try:
                if state["rounds"] >= max_rounds:
                    raise AgentException(
                        f"Workflow supervisor '{coordinator.name}' exceeded "
                        f"max_rounds={max_rounds} without reaching a final answer.",
                    )
                decision = coordinator.ai.chat(
                    _supervisor_prompt(state["task"], list(workers), transcript),
                    response_model=_SupervisorDecision,
                )
                state["rounds"] += 1
                if decision.action == "finish":
                    state["final_answer"] = decision.final_answer or ""
                    termination.mark_finished()
                    return finisher_name

                delegate = NativeOrchestrator._resolve_delegate(
                    decision, supervisor_name=coordinator.name, workers=workers
                )
                state["pending_task"] = decision.task or state["task"]
                return str(delegate.name)
            except (AgentException, ConfigurationException) as exc:
                state["error"] = exc
                termination.mark_finished()
                return finisher_name

        team = autogen["SelectorGroupChat"](
            [*worker_participants.values(), finisher],
            model_client=coordinator_client,
            selector_func=_selector_func,
            termination_condition=termination,
        )
        return team, [coordinator_client, *worker_clients.values()], state

    def run(
        self,
        steps: Sequence[Any],
        input: Optional[str],  # noqa: A002
        *,
        strategy: str = "sequential",
        **kwargs: Any,
    ) -> WorkflowResult:
        # autogen-agentchat's teams (Team.run()/RoundRobinGroupChat/
        # SelectorGroupChat) are async-only -- no sync-native equivalent
        # exists on the SDK side (confirmed directly: only run()/run_stream(),
        # both async). asyncio.run() here matches every other orchestrator's
        # sync entry point (MCPClient.discover_tools(), Tool.execute(), ...).
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

        autogen = self._require_autogen()

        if strategy == "sequential":
            team, clients = self._build_sequential_team(autogen, steps)
            task_result = await team.run(task=input)
            last_text = next(
                (
                    m.content
                    for m in reversed(task_result.messages)
                    if isinstance(getattr(m, "content", None), str)
                ),
                "",
            )
            return WorkflowResult(
                content=last_text,
                steps=self._collect_steps(clients),
                orchestrator=self.name,
                strategy=strategy,
            )

        if strategy == "supervisor":
            max_rounds = kwargs.pop("max_rounds", 6)
            team, clients, state = self._build_supervisor_team(
                autogen, steps, max_rounds=max_rounds
            )
            state["task"] = input
            await team.run(task=input)
            if "error" in state:
                raise state["error"]
            return WorkflowResult(
                content=state.get("final_answer", ""),
                steps=self._collect_steps(clients),
                orchestrator=self.name,
                strategy=strategy,
            )

        raise ConfigurationException(
            f"The autogen orchestrator supports the 'sequential' and 'supervisor' "
            f"strategies (got '{strategy}').",
        )
