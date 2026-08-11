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
This ships a linear (sequential) graph -- each agent is a node, wired
node-to-node in the order added -- as the initial integration. Fan-out/
fan-in graphs (parallel branches, conditional routing, cycles for
reflection loops, etc.) are natural extensions: build a richer graph in
:meth:`_build_graph` and this class's public surface does not need to
change.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, TypedDict

from requisite.core.exceptions import ConfigurationException
from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult

if TYPE_CHECKING:
    from requisite.agents.agent import Agent

logger = logging.getLogger("requisite.orchestrators.langgraph")


class _GraphState(TypedDict):
    input: str
    output: str
    steps: list[Any]


class LangGraphOrchestrator(BaseOrchestrator):
    """Runs agents as a linear ``langgraph`` ``StateGraph``.

    Only the ``"sequential"`` strategy is currently supported through
    langgraph; parallel branching is on the roadmap (see module docstring).
    """

    @property
    def name(self) -> str:
        return "langgraph"

    def _require_langgraph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'langgraph' package is required to use the langgraph orchestrator. "
                "Install it with: pip install langgraph",
            ) from exc
        return StateGraph, END

    def _build_graph(self, steps: Sequence["Agent"], **kwargs: Any) -> Any:
        StateGraph, END = self._require_langgraph()  # noqa: N806

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
                f"The langgraph orchestrator currently only supports the 'sequential' "
                f"strategy (got '{strategy}').",
            )

        compiled_graph = self._build_graph(steps, **kwargs)
        final_state = compiled_graph.invoke({"input": input, "output": "", "steps": []})
        return WorkflowResult(
            content=final_state["output"],
            steps=final_state["steps"],
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
                f"The langgraph orchestrator currently only supports the 'sequential' "
                f"strategy (got '{strategy}').",
            )

        compiled_graph = self._build_graph(steps, **kwargs)
        final_state = await compiled_graph.ainvoke({"input": input, "output": "", "steps": []})
        return WorkflowResult(
            content=final_state["output"],
            steps=final_state["steps"],
            orchestrator=self.name,
            strategy=strategy,
        )
