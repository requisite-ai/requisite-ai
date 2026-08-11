"""
LangGraph orchestrator backend.

Executes the same list of agents as :class:`~requisite.orchestrators.native.NativeOrchestrator`,
but builds and runs a `langgraph <https://github.com/langchain-ai/langgraph>`_
``StateGraph`` under the hood. The public
:class:`~requisite.workflows.workflow.Workflow` API is identical either
way -- switching backends via ``workflow.use_langgraph()`` never changes
how you call ``.add()`` or ``.run()``.

Install with: ``pip install langgraph``

Notes
-----
Two strategies are supported: ``"sequential"`` (a linear chain -- each
agent is a node, wired node-to-node in the order added) and
``"supervisor"`` (a real conditional graph: one coordinator node routes,
via ``add_conditional_edges``, to whichever worker node it delegates to
next, each worker looping back to the coordinator for another round,
until it decides to finish). See ``docs/adr/0016-langgraph-branching.md``
for why ``supervisor`` specifically -- it's the one existing strategy
whose shape is genuinely conditional routing, not a disguised loop.
Further strategies (``reflection``, ``hierarchical``) are natural
extensions: build another graph-building method and add a branch in
:meth:`LangGraphOrchestrator.run` / :meth:`.arun`; this class's public
surface does not need to change.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, TypedDict

from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult
from requisite.orchestrators.native import (
    NativeOrchestrator,
    _SupervisorDecision,
    _supervisor_prompt,
)

if TYPE_CHECKING:
    from requisite.agents.agent import Agent

logger = logging.getLogger("requisite.orchestrators.langgraph")

_SUPERVISOR_NODE = "__supervisor__"
_FINISH_ROUTE = "__finish__"


class _GraphState(TypedDict):
    input: str
    output: str
    steps: list[Any]


class _SupervisorGraphState(TypedDict):
    task: str
    transcript: list[tuple[str, str, str]]
    steps: list[Any]
    route: str
    pending_task: str
    output: str
    rounds: int


class LangGraphOrchestrator(BaseOrchestrator):
    """Runs agents as a ``langgraph`` ``StateGraph`` -- linear for
    ``"sequential"``, a real conditional graph with a loop-back cycle for
    ``"supervisor"``. See module docstring.
    """

    @property
    def name(self) -> str:
        return "langgraph"

    def _require_langgraph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'langgraph' package is required to use the langgraph orchestrator. "
                "Install it with: pip install langgraph",
            ) from exc
        return StateGraph, START, END

    def _build_graph(self, steps: Sequence["Agent"], **kwargs: Any) -> Any:
        StateGraph, _START, END = self._require_langgraph()  # noqa: N806

        graph = StateGraph(_GraphState)

        def _make_node(agent: "Agent") -> Any:
            def _node(state: _GraphState) -> _GraphState:
                result = agent.run(state["input"], **kwargs)
                return {
                    "input": result.content,
                    "output": result.content,
                    "steps": [*state["steps"], result],
                }

            return _node

        previous_name: Optional[str] = None
        for index, agent in enumerate(steps):
            node_name = f"{agent.name}_{index}"
            graph.add_node(node_name, _make_node(agent))
            if previous_name is None:
                graph.set_entry_point(node_name)
            else:
                graph.add_edge(previous_name, node_name)
            previous_name = node_name

        if previous_name is not None:
            graph.add_edge(previous_name, END)

        return graph.compile()

    def _build_supervisor_graph(
        self, steps: Sequence["Agent"], *, max_rounds: int, **kwargs: Any
    ) -> Any:
        """Build a real conditional graph for the ``supervisor`` strategy.

        One node per worker (named after ``worker.name`` -- already
        guaranteed unique by ``_split_coordinator_and_workers``) plus one
        ``_SUPERVISOR_NODE``. The supervisor node makes the same
        structured decision :class:`~requisite.orchestrators.native.NativeOrchestrator`
        makes (reused, not duplicated -- see module docstring) and writes
        it to ``state["route"]``; ``add_conditional_edges`` reads that to
        pick the next worker node or ``END``. Each worker node routes
        back to the supervisor node -- the loop.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        coordinator, workers = NativeOrchestrator._split_coordinator_and_workers(
            steps, role="supervisor"
        )

        def _supervisor_node(state: _SupervisorGraphState) -> dict[str, Any]:
            if state["rounds"] >= max_rounds:
                raise AgentException(
                    f"Workflow supervisor '{coordinator.name}' exceeded max_rounds={max_rounds} "
                    f"without reaching a final answer.",
                )
            decision = coordinator.ai.chat(
                _supervisor_prompt(state["task"], list(workers), state["transcript"]),
                response_model=_SupervisorDecision,
            )
            if decision.action == "finish":
                return {
                    "route": _FINISH_ROUTE,
                    "output": decision.final_answer or "",
                    "rounds": state["rounds"] + 1,
                }

            delegate = NativeOrchestrator._resolve_delegate(
                decision, supervisor_name=coordinator.name, workers=workers
            )
            return {
                "route": delegate.name,
                "pending_task": decision.task or state["task"],
                "rounds": state["rounds"] + 1,
            }

        def _make_worker_node(worker: "Agent") -> Any:
            def _node(state: _SupervisorGraphState) -> dict[str, Any]:
                result = worker.run(state["pending_task"], **kwargs)
                return {
                    "steps": [*state["steps"], result],
                    "transcript": [
                        *state["transcript"],
                        (worker.name, state["pending_task"], result.content),
                    ],
                }

            return _node

        def _route(state: _SupervisorGraphState) -> str:
            return state["route"]

        graph = StateGraph(_SupervisorGraphState)
        graph.add_node(_SUPERVISOR_NODE, _supervisor_node)
        for worker_name, worker in workers.items():
            graph.add_node(worker_name, _make_worker_node(worker))
            graph.add_edge(worker_name, _SUPERVISOR_NODE)

        path_map = {worker_name: worker_name for worker_name in workers}
        path_map[_FINISH_ROUTE] = END
        graph.add_conditional_edges(_SUPERVISOR_NODE, _route, path_map)
        graph.add_edge(START, _SUPERVISOR_NODE)

        return graph.compile()

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

        if strategy == "sequential":
            compiled_graph = self._build_graph(steps, **kwargs)
            final_state = compiled_graph.invoke({"input": input, "output": "", "steps": []})
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        if strategy == "supervisor":
            max_rounds = kwargs.pop("max_rounds", 6)
            compiled_graph = self._build_supervisor_graph(steps, max_rounds=max_rounds, **kwargs)
            final_state = compiled_graph.invoke(
                {
                    "task": input,
                    "transcript": [],
                    "steps": [],
                    "route": "",
                    "pending_task": "",
                    "output": "",
                    "rounds": 0,
                },
            )
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        raise ConfigurationException(
            f"The langgraph orchestrator supports the 'sequential' and 'supervisor' "
            f"strategies (got '{strategy}').",
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

        if strategy == "sequential":
            compiled_graph = self._build_graph(steps, **kwargs)
            final_state = await compiled_graph.ainvoke({"input": input, "output": "", "steps": []})
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        if strategy == "supervisor":
            max_rounds = kwargs.pop("max_rounds", 6)
            compiled_graph = self._build_supervisor_graph(steps, max_rounds=max_rounds, **kwargs)
            final_state = await compiled_graph.ainvoke(
                {
                    "task": input,
                    "transcript": [],
                    "steps": [],
                    "route": "",
                    "pending_task": "",
                    "output": "",
                    "rounds": 0,
                },
            )
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        raise ConfigurationException(
            f"The langgraph orchestrator supports the 'sequential' and 'supervisor' "
            f"strategies (got '{strategy}').",
        )
