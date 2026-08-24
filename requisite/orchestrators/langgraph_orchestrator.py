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
Eight strategies are supported: ``"sequential"`` (a linear chain -- each
agent is a node, wired node-to-node in the order added); ``"supervisor"``
and ``"hierarchical"`` (both a real conditional graph built by the same
:meth:`LangGraphOrchestrator._build_delegation_graph` -- one coordinator
node routes, via ``add_conditional_edges``, to whichever delegate node
it delegates to next, each delegate looping back to the coordinator for
another round, until it decides to finish; the only difference between
the two strategies is which split-helper validates ``steps`` --
Agent-only for ``supervisor``, Agent-or-named-``Workflow`` for
``hierarchical`` -- mirroring how
:class:`~requisite.orchestrators.native.NativeOrchestrator` itself
shares one ``_run_delegation_loop`` between both); ``"reflection"`` (a
3-node cycle -- draft, critique, revise -- with a conditional exit on
the ``NO_CHANGES_NEEDED`` sentinel); ``"graph"`` (an arbitrary,
developer-declared graph -- one node per step, routed by
``Workflow.add_edge(...)``'s own conditions, reusing
:class:`NativeOrchestrator`'s validation/routing helpers directly rather
than reimplementing them); and ``"parallel"``, ``"consensus"``, and
``"map_reduce"`` (all three share one fan-out/fan-in shape -- N agents
run concurrently as separate nodes in the same superstep, writing
``(index, result)`` tuples into a reducer channel, then one aggregator
node sorts by index and combines -- no loop-back cycle needed, unlike
every strategy above). See ``docs/adr/0016-langgraph-branching.md``
(``supervisor``), ``docs/adr/0028-langgraph-reflection-strategy.md``
(``reflection``), ``docs/adr/0029-langgraph-hierarchical-graph-strategies.md``
(``hierarchical``, ``graph``), and
``docs/adr/0032-langgraph-parallel-consensus-map-reduce-strategies.md``
(``parallel``, ``consensus``, ``map_reduce``) for the full design of each.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional, TypedDict

from requisite.core.exceptions import AgentException, ConfigurationException
from requisite.orchestrators.base import BaseOrchestrator, WorkflowResult
from requisite.orchestrators.native import (
    NativeOrchestrator,
    _SupervisorDecision,
    _consensus_prompt,
    _map_prompt,
    _reduce_prompt,
    _reflection_critique_prompt,
    _reflection_revise_prompt,
    _supervisor_prompt,
)

if TYPE_CHECKING:
    from requisite.agents.agent import Agent

logger = logging.getLogger("requisite.orchestrators.langgraph")

# Shared by both "supervisor" and "hierarchical" -- see
# _build_delegation_graph. Not renamed to something strategy-neutral
# beyond this: a coordinator node + finish sentinel is the same shape
# either strategy needs, and neither name implies Agent-only semantics.
_COORDINATOR_NODE = "__coordinator__"
_FINISH_ROUTE = "__finish__"
_DRAFT_NODE = "__draft__"
_CRITIQUE_NODE = "__critique__"
_REVISE_NODE = "__revise__"
_NO_CHANGES_NEEDED = "NO_CHANGES_NEEDED"
# Shared by "parallel", "consensus", and "map_reduce" -- see
# _FanOutGraphState / _build_parallel_graph / _build_consensus_graph /
# _build_map_reduce_graph.
_AGGREGATOR_NODE = "__aggregator__"


class _GraphState(TypedDict):
    input: str
    output: str
    steps: list[Any]


class _DelegationGraphState(TypedDict):
    """Shared by "supervisor" and "hierarchical" -- see _build_delegation_graph."""

    task: str
    transcript: list[tuple[str, str, str]]
    steps: list[Any]
    route: str
    pending_task: str
    output: str
    rounds: int


class _ReflectionGraphState(TypedDict):
    input: str
    draft: str
    critique: str
    steps: list[Any]
    rounds: int


class _ArbitraryGraphState(TypedDict):
    input: str
    output: str
    steps: list[Any]
    step_count: int


class _FanOutGraphState(TypedDict):
    """Shared by "parallel", "consensus", and "map_reduce" -- all three
    fan out to N agent nodes that run concurrently in the same
    superstep, then join into one aggregator node. See
    _build_parallel_graph / _build_consensus_graph /
    _build_map_reduce_graph and docs/adr/0032.
    """

    task: str
    # Every fan-out node writes exactly one (build-time-assigned index,
    # AgentResult) tuple here. A reducer (Annotated[..., operator.add])
    # is required because multiple nodes write to this key in the same
    # superstep -- langgraph raises InvalidUpdateError on a plain,
    # non-Annotated key with concurrent writers. The index lets the
    # aggregator node restore build-time order regardless of what order
    # langgraph itself applies the concurrent writes in (confirmed via
    # langgraph/pregel/_algo.py: write order is node-name lexicographic,
    # not declaration order -- silently wrong past 10 fan-out nodes if
    # trusted directly). See docs/adr/0032.
    results: Annotated[list[tuple[int, Any]], operator.add]
    steps: list[Any]  # final ordered AgentResult list, written once by the aggregator
    output: str


def _reject_reserved_node_names(
    named: dict[str, Any], *, role: str, reserved: Sequence[str]
) -> None:
    """Raise a clean ``ConfigurationException`` if any key in ``named``
    (a delegate/worker name -> object mapping) collides with one of this
    backend's own internal node names.

    Without this check, a delegate/worker literally named e.g.
    ``"__coordinator__"`` reaches ``StateGraph.add_node`` and collides
    with the coordinator node this method already added under that same
    name, raising a raw, backend-specific ``ValueError`` instead of a
    clean, actionable error -- and the identical ``Workflow`` succeeds
    unchanged on the ``native`` backend, breaking the cross-backend
    parity these strategies otherwise guarantee. See
    ``docs/adr/0031-code-review-fixes.md``.
    """
    collisions = sorted(set(named) & set(reserved))
    if collisions:
        raise ConfigurationException(
            f"The '{role}' strategy on the langgraph backend reserves "
            f"{sorted(reserved)!r} for its own internal graph nodes -- rename the "
            f"delegate/worker(s) {collisions!r} to something else.",
        )


class LangGraphOrchestrator(BaseOrchestrator):
    """Runs agents as a ``langgraph`` ``StateGraph`` -- linear for
    ``"sequential"``, real conditional graphs with loop-back cycles for
    ``"supervisor"``/``"hierarchical"``/``"reflection"``, and an
    arbitrary developer-declared graph for ``"graph"``. See module
    docstring.
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

    def _build_delegation_graph(
        self,
        steps: Sequence[Any],
        *,
        split_fn: Callable[..., tuple["Agent", dict[str, Any]]],
        role: str,
        max_rounds: int,
        **kwargs: Any,
    ) -> Any:
        """Build a real conditional graph shared by ``supervisor`` and ``hierarchical``.

        One node per delegate (named after ``delegate.name`` -- already
        guaranteed unique by ``split_fn``) plus one ``_COORDINATOR_NODE``.
        The coordinator node makes the same structured decision
        :class:`~requisite.orchestrators.native.NativeOrchestrator` makes
        (reused, not duplicated -- see module docstring) and writes it to
        ``state["route"]``; ``add_conditional_edges`` reads that to pick
        the next delegate node or ``END``. Each delegate node routes back
        to the coordinator node -- the loop.

        ``split_fn`` is the only thing that differs between the two
        strategies: ``NativeOrchestrator._split_coordinator_and_workers``
        (Agent-only) for ``supervisor``, or
        ``NativeOrchestrator._split_coordinator_and_delegates``
        (Agent-or-named-``Workflow``) for ``hierarchical`` -- mirroring
        how ``NativeOrchestrator`` itself shares one
        ``_run_delegation_loop`` between both. A delegate node's own body
        only ever calls ``delegate.run(...)``, which both ``Agent`` and
        ``Workflow`` support identically, so nothing else here needs to
        change per strategy.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        coordinator, delegates = split_fn(steps, role=role)
        _reject_reserved_node_names(
            delegates, role=role, reserved=(_COORDINATOR_NODE, _FINISH_ROUTE)
        )

        def _coordinator_node(state: _DelegationGraphState) -> dict[str, Any]:
            if state["rounds"] >= max_rounds:
                raise AgentException(
                    f"Workflow {role} '{coordinator.name}' exceeded max_rounds={max_rounds} "
                    f"without reaching a final answer.",
                )
            decision = coordinator.ai.chat(
                _supervisor_prompt(state["task"], list(delegates), state["transcript"]),
                response_model=_SupervisorDecision,
            )
            if decision.action == "finish":
                return {
                    "route": _FINISH_ROUTE,
                    "output": decision.final_answer or "",
                    "rounds": state["rounds"] + 1,
                }

            delegate = NativeOrchestrator._resolve_delegate(
                decision, supervisor_name=coordinator.name, workers=delegates
            )
            return {
                "route": delegate.name,
                "pending_task": decision.task or state["task"],
                "rounds": state["rounds"] + 1,
            }

        def _make_delegate_node(delegate: Any) -> Any:
            def _node(state: _DelegationGraphState) -> dict[str, Any]:
                result = delegate.run(state["pending_task"], **kwargs)
                return {
                    "steps": [*state["steps"], result],
                    "transcript": [
                        *state["transcript"],
                        (delegate.name, state["pending_task"], result.content),
                    ],
                }

            return _node

        def _route(state: _DelegationGraphState) -> str:
            return state["route"]

        graph = StateGraph(_DelegationGraphState)
        graph.add_node(_COORDINATOR_NODE, _coordinator_node)
        for delegate_name, delegate in delegates.items():
            graph.add_node(delegate_name, _make_delegate_node(delegate))
            graph.add_edge(delegate_name, _COORDINATOR_NODE)

        path_map = {delegate_name: delegate_name for delegate_name in delegates}
        path_map[_FINISH_ROUTE] = END
        graph.add_conditional_edges(_COORDINATOR_NODE, _route, path_map)
        graph.add_edge(START, _COORDINATOR_NODE)

        return graph.compile()

    def _build_arbitrary_graph(
        self, steps: Sequence[Any], *, edges: Sequence[Any], max_steps: int, **kwargs: Any
    ) -> Any:
        """Build a real conditional graph for the ``graph`` strategy.

        One node per step (named after ``step.name``), routed by the
        developer-declared ``edges`` -- exactly the ``(from_, to,
        condition)`` triples :class:`~requisite.orchestrators.native.NativeOrchestrator`
        already validates and resolves for its own ``graph`` strategy.
        Reused verbatim here (``_index_graph_nodes``,
        ``_validate_graph_edges``, ``_resolve_next_graph_node``) since
        none of that logic is LLM-decided or backend-specific -- it's
        plain name-indexing and condition-matching over developer-
        supplied callables (see module docstring, ADR-0019).

        ``max_steps`` is checked inside each node function, before it
        runs -- not inside the router -- mirroring
        :meth:`_build_delegation_graph`'s ``_coordinator_node`` check-
        then-raise pattern, and reproducing Native's own ``for _ in
        range(max_steps)`` loop semantics exactly: up to ``max_steps``
        node executions are allowed, and the raise fires only when a
        ``(max_steps + 1)``-th execution is attempted.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        if max_steps < 1:
            raise ConfigurationException(
                f"The 'graph' strategy requires max_steps >= 1. Got {max_steps}.",
            )
        nodes = NativeOrchestrator._index_graph_nodes(steps, role="graph")
        edges_by_source = NativeOrchestrator._validate_graph_edges(edges, nodes)

        def _make_node(name: str, node: Any) -> Any:
            def _node_fn(state: _ArbitraryGraphState) -> dict[str, Any]:
                if state["step_count"] >= max_steps:
                    raise AgentException(
                        f"Workflow graph exceeded max_steps={max_steps} without reaching an end.",
                    )
                result = node.run(state["input"], **kwargs)
                return {
                    "input": result.content,
                    "output": result.content,
                    "steps": [*state["steps"], result],
                    "step_count": state["step_count"] + 1,
                }

            return _node_fn

        def _make_router(name: str) -> Any:
            def _router(state: _ArbitraryGraphState) -> Any:
                next_name = NativeOrchestrator._resolve_next_graph_node(
                    name, state["output"], edges_by_source
                )
                return next_name if next_name is not None else END

            return _router

        graph = StateGraph(_ArbitraryGraphState)
        for name, node in nodes.items():
            graph.add_node(name, _make_node(name, node))
            graph.add_conditional_edges(name, _make_router(name))
        graph.add_edge(START, steps[0].name)

        return graph.compile()

    def _build_reflection_graph(
        self, steps: Sequence["Agent"], *, max_rounds: int, **kwargs: Any
    ) -> Any:
        """Build a real conditional graph for the ``reflection`` strategy.

        Mirrors :meth:`~requisite.orchestrators.native.NativeOrchestrator._run_reflection`
        exactly: draft, then up to ``max_rounds - 1`` rounds of
        critique-then-maybe-revise. Revise always follows a non-sentinel
        critique regardless of remaining budget -- only whether the loop
        goes around *again* (another critique) is budget-gated -- so
        ``rounds`` (critique count) is checked after revise, not before
        it. ``max_rounds`` is a closure variable baked in at graph-build
        time, the same way :meth:`_build_supervisor_graph` closes over it
        rather than storing it in graph state.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        if len(steps) != 1:
            raise ConfigurationException(
                f"The 'reflection' strategy requires exactly one agent (it critiques "
                f"and revises its own output). Got {len(steps)}.",
            )
        worker = steps[0]

        def _draft_node(state: _ReflectionGraphState) -> dict[str, Any]:
            result = worker.run(state["input"], **kwargs)
            return {"draft": result.content, "steps": [*state["steps"], result], "rounds": 0}

        def _critique_node(state: _ReflectionGraphState) -> dict[str, Any]:
            result = worker.run(
                _reflection_critique_prompt(state["input"], state["draft"]), **kwargs
            )
            return {
                "critique": result.content,
                "steps": [*state["steps"], result],
                "rounds": state["rounds"] + 1,
            }

        def _revise_node(state: _ReflectionGraphState) -> dict[str, Any]:
            result = worker.run(
                _reflection_revise_prompt(state["input"], state["draft"], state["critique"]),
                **kwargs,
            )
            return {"draft": result.content, "steps": [*state["steps"], result]}

        # Return type Any, not str: END is a langgraph sentinel object
        # (loosely typed here since it's obtained via the lazy
        # _require_langgraph() import), not a plain string.
        def _route_after_draft(state: _ReflectionGraphState) -> Any:
            return _CRITIQUE_NODE if max_rounds > 1 else END

        def _route_after_critique(state: _ReflectionGraphState) -> Any:
            if state["critique"].strip() == _NO_CHANGES_NEEDED:
                return END
            return _REVISE_NODE

        def _route_after_revise(state: _ReflectionGraphState) -> Any:
            return _CRITIQUE_NODE if state["rounds"] < max_rounds - 1 else END

        graph = StateGraph(_ReflectionGraphState)
        graph.add_node(_DRAFT_NODE, _draft_node)
        graph.add_node(_CRITIQUE_NODE, _critique_node)
        graph.add_node(_REVISE_NODE, _revise_node)
        graph.add_edge(START, _DRAFT_NODE)
        graph.add_conditional_edges(_DRAFT_NODE, _route_after_draft)
        graph.add_conditional_edges(_CRITIQUE_NODE, _route_after_critique)
        graph.add_conditional_edges(_REVISE_NODE, _route_after_revise)

        return graph.compile()

    def _build_parallel_graph(self, steps: Sequence["Agent"], **kwargs: Any) -> Any:
        """Build a fan-out/fan-in graph for the ``parallel`` strategy.

        No coordinator/worker split -- every step is a peer agent, run
        concurrently against the same input, purely string-combined
        (no aggregator agent call). Mirrors
        :meth:`~requisite.orchestrators.native.NativeOrchestrator._run_parallel`
        exactly. See :class:`_FanOutGraphState` and docs/adr/0032.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        graph = StateGraph(_FanOutGraphState)

        def _make_node(agent: "Agent", index: int) -> Any:
            def _node(state: _FanOutGraphState) -> dict[str, Any]:
                result = agent.run(state["task"], **kwargs)
                return {"results": [(index, result)]}

            return _node

        node_names = []
        for index, agent in enumerate(steps):
            # Index-suffixed: parallel has no name-uniqueness requirement
            # on agents, unlike consensus/map_reduce's name-addressed
            # workers.
            node_name = f"{agent.name}_{index}"
            graph.add_node(node_name, _make_node(agent, index))
            graph.add_edge(START, node_name)
            node_names.append(node_name)

        def _combine_node(state: _FanOutGraphState) -> dict[str, Any]:
            ordered = [r for _, r in sorted(state["results"], key=lambda pair: pair[0])]
            combined = "\n\n".join(f"[{r.agent_name}]\n{r.content}" for r in ordered)
            return {"output": combined, "steps": ordered}

        graph.add_node(_AGGREGATOR_NODE, _combine_node)
        graph.add_edge(node_names, _AGGREGATOR_NODE)
        graph.add_edge(_AGGREGATOR_NODE, END)

        return graph.compile()

    def _build_consensus_graph(self, steps: Sequence["Agent"], **kwargs: Any) -> Any:
        """Build a fan-out/fan-in graph for the ``consensus`` strategy.

        ``steps[0]`` is the synthesizer, ``steps[1:]`` are independent
        participants run concurrently against the same original input
        (not each other's outputs). The synthesizer then combines every
        participant's answer into one final answer. Reuses
        :meth:`NativeOrchestrator._split_coordinator_and_workers` and
        ``_consensus_prompt`` rather than reimplementing them -- mirrors
        :meth:`~requisite.orchestrators.native.NativeOrchestrator._run_consensus`
        exactly. See :class:`_FanOutGraphState` and docs/adr/0032.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        synthesizer, participants = NativeOrchestrator._split_coordinator_and_workers(
            steps, role="consensus synthesizer"
        )
        _reject_reserved_node_names(
            participants, role="consensus synthesizer", reserved=(_AGGREGATOR_NODE,)
        )
        graph = StateGraph(_FanOutGraphState)

        def _make_node(participant: "Agent", index: int) -> Any:
            def _node(state: _FanOutGraphState) -> dict[str, Any]:
                result = participant.run(state["task"], **kwargs)
                return {"results": [(index, result)]}

            return _node

        node_names = []
        for index, (name, participant) in enumerate(participants.items()):
            graph.add_node(name, _make_node(participant, index))
            graph.add_edge(START, name)
            node_names.append(name)

        def _synthesize_node(state: _FanOutGraphState) -> dict[str, Any]:
            ordered = [r for _, r in sorted(state["results"], key=lambda pair: pair[0])]
            synthesis = synthesizer.run(_consensus_prompt(state["task"], ordered), **kwargs)
            return {"output": synthesis.content, "steps": [*ordered, synthesis]}

        graph.add_node(_AGGREGATOR_NODE, _synthesize_node)
        graph.add_edge(node_names, _AGGREGATOR_NODE)
        graph.add_edge(_AGGREGATOR_NODE, END)

        return graph.compile()

    def _build_map_reduce_graph(
        self, steps: Sequence["Agent"], *, map_items: Optional[Sequence[str]], **kwargs: Any
    ) -> Any:
        """Build a fan-out/fan-in graph for the ``map_reduce`` strategy.

        One node per ``map_items`` entry (not per agent -- items are
        assigned to ``steps[1:]`` workers round-robin, so the same
        worker can appear at multiple indices), run concurrently, then
        ``steps[0]`` reduces every mapped result into one final answer.
        Node names are synthetic (``__mapper_<index>__``) rather than
        the worker's own name, since a worker's name isn't unique across
        the items it's assigned. Mirrors
        :meth:`~requisite.orchestrators.native.NativeOrchestrator._run_map_reduce`
        exactly, including its error message. See :class:`_FanOutGraphState`
        and docs/adr/0032.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        if not map_items:
            raise ConfigurationException(
                "The 'map_reduce' strategy requires map_items=[...] -- the list of "
                "individual items to process, passed to workflow.run()/arun().",
            )
        reducer, mappers = NativeOrchestrator._split_coordinator_and_workers(
            steps, role="map-reduce reducer"
        )
        mapper_list = list(mappers.values())
        graph = StateGraph(_FanOutGraphState)

        def _make_node(mapper: "Agent", item: str, index: int) -> Any:
            def _node(state: _FanOutGraphState) -> dict[str, Any]:
                result = mapper.run(_map_prompt(state["task"], item), **kwargs)
                return {"results": [(index, result)]}

            return _node

        node_names = []
        for index, item in enumerate(map_items):
            mapper = mapper_list[index % len(mapper_list)]
            node_name = f"__mapper_{index}__"
            graph.add_node(node_name, _make_node(mapper, item, index))
            graph.add_edge(START, node_name)
            node_names.append(node_name)

        def _reduce_node(state: _FanOutGraphState) -> dict[str, Any]:
            ordered = [r for _, r in sorted(state["results"], key=lambda pair: pair[0])]
            reduced = reducer.run(_reduce_prompt(state["task"], map_items, ordered), **kwargs)
            return {"output": reduced.content, "steps": [*ordered, reduced]}

        graph.add_node(_AGGREGATOR_NODE, _reduce_node)
        graph.add_edge(node_names, _AGGREGATOR_NODE)
        graph.add_edge(_AGGREGATOR_NODE, END)

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
        if strategy in ("supervisor", "hierarchical"):
            split_fn = (
                NativeOrchestrator._split_coordinator_and_workers
                if strategy == "supervisor"
                else NativeOrchestrator._split_coordinator_and_delegates
            )
            max_rounds = kwargs.pop("max_rounds", 6)
            compiled_graph = self._build_delegation_graph(
                steps, split_fn=split_fn, role=strategy, max_rounds=max_rounds, **kwargs
            )
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
        if strategy == "reflection":
            max_rounds = kwargs.pop("max_rounds", 3)
            compiled_graph = self._build_reflection_graph(steps, max_rounds=max_rounds, **kwargs)
            final_state = compiled_graph.invoke(
                {"input": input, "draft": "", "critique": "", "steps": [], "rounds": 0},
            )
            return WorkflowResult(
                content=final_state["draft"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        if strategy == "graph":
            edges = kwargs.pop("edges", ())
            max_steps = kwargs.pop("max_steps", 25)
            compiled_graph = self._build_arbitrary_graph(
                steps, edges=edges, max_steps=max_steps, **kwargs
            )
            final_state = compiled_graph.invoke(
                {"input": input, "output": "", "steps": [], "step_count": 0},
            )
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        if strategy in ("parallel", "consensus", "map_reduce"):
            if strategy == "parallel":
                compiled_graph = self._build_parallel_graph(steps, **kwargs)
            elif strategy == "consensus":
                compiled_graph = self._build_consensus_graph(steps, **kwargs)
            else:
                map_items = kwargs.pop("map_items", None)
                compiled_graph = self._build_map_reduce_graph(steps, map_items=map_items, **kwargs)
            final_state = compiled_graph.invoke(
                {"task": input, "results": [], "steps": [], "output": ""},
            )
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        raise ConfigurationException(
            f"The langgraph orchestrator supports the 'sequential', 'supervisor', "
            f"'hierarchical', 'reflection', 'graph', 'parallel', 'consensus', and "
            f"'map_reduce' strategies (got '{strategy}').",
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
        if strategy in ("supervisor", "hierarchical"):
            split_fn = (
                NativeOrchestrator._split_coordinator_and_workers
                if strategy == "supervisor"
                else NativeOrchestrator._split_coordinator_and_delegates
            )
            max_rounds = kwargs.pop("max_rounds", 6)
            compiled_graph = self._build_delegation_graph(
                steps, split_fn=split_fn, role=strategy, max_rounds=max_rounds, **kwargs
            )
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
        if strategy == "reflection":
            max_rounds = kwargs.pop("max_rounds", 3)
            compiled_graph = self._build_reflection_graph(steps, max_rounds=max_rounds, **kwargs)
            final_state = await compiled_graph.ainvoke(
                {"input": input, "draft": "", "critique": "", "steps": [], "rounds": 0},
            )
            return WorkflowResult(
                content=final_state["draft"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        if strategy == "graph":
            edges = kwargs.pop("edges", ())
            max_steps = kwargs.pop("max_steps", 25)
            compiled_graph = self._build_arbitrary_graph(
                steps, edges=edges, max_steps=max_steps, **kwargs
            )
            final_state = await compiled_graph.ainvoke(
                {"input": input, "output": "", "steps": [], "step_count": 0},
            )
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        if strategy in ("parallel", "consensus", "map_reduce"):
            if strategy == "parallel":
                compiled_graph = self._build_parallel_graph(steps, **kwargs)
            elif strategy == "consensus":
                compiled_graph = self._build_consensus_graph(steps, **kwargs)
            else:
                map_items = kwargs.pop("map_items", None)
                compiled_graph = self._build_map_reduce_graph(steps, map_items=map_items, **kwargs)
            final_state = await compiled_graph.ainvoke(
                {"task": input, "results": [], "steps": [], "output": ""},
            )
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        raise ConfigurationException(
            f"The langgraph orchestrator supports the 'sequential', 'supervisor', "
            f"'hierarchical', 'reflection', 'graph', 'parallel', 'consensus', and "
            f"'map_reduce' strategies (got '{strategy}').",
        )
