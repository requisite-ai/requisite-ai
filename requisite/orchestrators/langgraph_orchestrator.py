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
Eleven strategies are supported: ``"sequential"`` (a linear chain -- each
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
shares one ``_run_delegation_loop`` between both); ``"reflection"`` and
``"critic"`` (both a 3-node cycle -- draft, critique, revise -- with a
conditional exit on the ``NO_CHANGES_NEEDED`` sentinel, built by the
same :meth:`LangGraphOrchestrator._build_reflection_graph`; the only
difference is whether one agent plays both the drafting and critiquing
role, or two separate agents do); ``"graph"`` (an arbitrary,
developer-declared graph -- one node per step, routed by
``Workflow.add_edge(...)``'s own conditions, reusing
:class:`NativeOrchestrator`'s validation/routing helpers directly rather
than reimplementing them); ``"parallel"``, ``"consensus"``, and
``"map_reduce"`` (all three share one fan-out/fan-in shape -- N agents
run concurrently as separate nodes in the same superstep, writing
``(index, result)`` tuples into a reducer channel, then one aggregator
node sorts by index and combines -- no loop-back cycle needed); and
``"debate"`` (``max_rounds`` fan-out/join blocks in a row -- one per
round, each round's debaters seeing the transcript as of the previous
round's join node -- then one verdict node; a static unroll of the
round loop rather than a true cycle, since ``max_rounds`` is known at
graph-build time); and ``"tree_of_thoughts"`` (a beam search unrolled
the same way -- ``breadth``/``beam_width``/``max_depth`` fully determine
every level's fan-out width at graph-build time, even though the paths
themselves are only known at runtime; each level is a fan-out of
candidate-thought nodes joined into one structured-output evaluation/prune
node, which conditionally routes to a trivial "expand" passthrough node
that fans out to the next level, or straight to ``END`` on early
termination or exhausting ``max_depth``). See
``docs/adr/0016-langgraph-branching.md`` (``supervisor``),
``docs/adr/0028-langgraph-reflection-strategy.md`` (``reflection``),
``docs/adr/0029-langgraph-hierarchical-graph-strategies.md``
(``hierarchical``, ``graph``),
``docs/adr/0032-langgraph-parallel-consensus-map-reduce-strategies.md``
(``parallel``, ``consensus``, ``map_reduce``),
``docs/adr/0033-langgraph-critic-debate-strategies.md``
(``critic``, ``debate``), and
``docs/adr/0034-langgraph-tree-of-thoughts-strategy.md``
(``tree_of_thoughts``) for the full design of each.
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
    _ThoughtEvaluation,
    _consensus_prompt,
    _critic_prompt,
    _debate_prompt,
    _debate_verdict_prompt,
    _map_prompt,
    _reduce_prompt,
    _reflection_critique_prompt,
    _reflection_revise_prompt,
    _supervisor_prompt,
    _tot_evaluation_prompt,
    _tot_thinker_prompt,
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
# "debate" only -- see _DebateGraphState / _build_debate_graph.
_VERDICT_NODE = "__verdict__"


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


class _DebateGraphState(TypedDict):
    """ "debate" only -- see _build_debate_graph.

    Unlike _FanOutGraphState, this strategy has multiple rounds, each
    its own fan-out/join block (see _build_debate_graph's module
    docstring reference, docs/adr/0033). ``results`` is still one
    shared reducer channel for the *whole* debate, not one per round --
    TypedDict schemas are static, so per-round-named keys aren't an
    option -- each round's join node writes a globally-unique index
    range (round_num * len(debaters) + i) and reads back only its own
    slice.
    """

    task: str
    results: Annotated[list[tuple[int, Any]], operator.add]
    transcript: dict[str, list[str]]  # plain field -- exactly one join node writes it per superstep
    steps: list[Any]
    output: str


class _TreeOfThoughtsGraphState(TypedDict):
    """ "tree_of_thoughts" only -- see _build_tree_of_thoughts_graph.

    Like _DebateGraphState, one shared reducer channel spans every
    level rather than one field per level -- TypedDict schemas are
    static. Each level's evaluation node knows its own offset range
    (precomputed at graph-build time, see docs/adr/0034) and reads back
    only its own slice. ``paths`` is a plain field: exactly one
    evaluation node (the current level's) writes it per superstep.
    ``finished`` is written by every level's evaluation node and read
    by that same level's routing function immediately after.
    """

    task: str
    paths: list[list[str]]
    candidates: Annotated[list[tuple[int, Any]], operator.add]
    steps: list[Any]
    output: str
    finished: bool


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
    ``"supervisor"``/``"hierarchical"``/``"reflection"``/``"critic"``,
    an arbitrary developer-declared graph for ``"graph"``, a
    fan-out/fan-in graph (no loop-back cycle) for ``"parallel"``/
    ``"consensus"``/``"map_reduce"``, a static per-round unroll of
    fan-out/join blocks for ``"debate"``, and a static per-level unroll
    of fan-out/evaluate/prune blocks for ``"tree_of_thoughts"``. See
    module docstring.
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
        self, steps: Sequence["Agent"], *, role: str, max_rounds: int, **kwargs: Any
    ) -> Any:
        """Build a real conditional graph shared by ``reflection`` and ``critic``.

        Mirrors :meth:`~requisite.orchestrators.native.NativeOrchestrator._run_reflection`/
        :meth:`~requisite.orchestrators.native.NativeOrchestrator._run_critic`
        exactly: draft, then up to ``max_rounds - 1`` rounds of
        critique-then-maybe-revise. Revise always follows a non-sentinel
        critique regardless of remaining budget -- only whether the loop
        goes around *again* (another critique) is budget-gated -- so
        ``rounds`` (critique count) is checked after revise, not before
        it. ``max_rounds`` is a closure variable baked in at graph-build
        time, the same way :meth:`_build_delegation_graph` closes over it
        rather than storing it in graph state.

        ``role`` is the only thing that differs between the two
        strategies, mirroring how :meth:`_build_delegation_graph` is
        shared by ``supervisor``/``hierarchical`` via a ``split_fn``
        parameter: for ``"reflection"``, one agent (``steps[0]``) plays
        both roles -- it drafts, critiques its own draft, and revises.
        For ``"critic"``, ``steps[0]`` only ever drafts/revises and
        ``steps[1]`` only ever critiques -- and critiques via
        ``_critic_prompt`` instead of ``_reflection_critique_prompt``.
        ``_reflection_revise_prompt`` is shared by both in
        ``native.py`` itself, so the revise step needs no per-role
        branching at all.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        if role == "reflection":
            if len(steps) != 1:
                raise ConfigurationException(
                    f"The 'reflection' strategy requires exactly one agent (it critiques "
                    f"and revises its own output). Got {len(steps)}.",
                )
            generator = critic_agent = steps[0]
            critique_prompt_fn = _reflection_critique_prompt
        else:
            if len(steps) != 2:
                raise ConfigurationException(
                    f"The 'critic' strategy requires exactly two agents: a generator "
                    f"(steps[0]) and a critic (steps[1]). Got {len(steps)}.",
                )
            generator, critic_agent = steps[0], steps[1]
            critique_prompt_fn = _critic_prompt

        def _draft_node(state: _ReflectionGraphState) -> dict[str, Any]:
            result = generator.run(state["input"], **kwargs)
            return {"draft": result.content, "steps": [*state["steps"], result], "rounds": 0}

        def _critique_node(state: _ReflectionGraphState) -> dict[str, Any]:
            result = critic_agent.run(critique_prompt_fn(state["input"], state["draft"]), **kwargs)
            return {
                "critique": result.content,
                "steps": [*state["steps"], result],
                "rounds": state["rounds"] + 1,
            }

        def _revise_node(state: _ReflectionGraphState) -> dict[str, Any]:
            result = generator.run(
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

    def _build_debate_graph(
        self, steps: Sequence["Agent"], *, max_rounds: int, **kwargs: Any
    ) -> tuple[Any, list[str]]:
        """Build a graph for the ``debate`` strategy: ``max_rounds``
        fan-out/join blocks in a row (one per round), then one verdict
        node -- not a true cycle. ``max_rounds`` is build-time-known
        (same as every other strategy in this module), so the round
        loop is unrolled into a longer static graph instead of a
        dynamic/cyclic one. Each round's debater nodes read the
        transcript as of the end of the *previous* round's join node
        (guaranteed by the edge between them); each round's join node
        writes a fresh ``transcript`` dict, read by the next round.
        Mirrors :meth:`~requisite.orchestrators.native.NativeOrchestrator._run_debate`
        exactly, including the degenerate ``max_rounds=0`` case (zero
        debater calls, the moderator still runs once against an empty
        transcript). See docs/adr/0033.

        Returns ``(compiled_graph, debater_names)`` -- the caller needs
        ``debater_names`` to build the initial ``transcript`` dict
        before invoking, and computing the split twice (once here, once
        in ``run``/``arun``) would risk the two falling out of sync.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        moderator, debaters = NativeOrchestrator._split_coordinator_and_workers(
            steps, role="debate moderator"
        )
        debater_names = list(debaters)
        n = len(debater_names)

        graph = StateGraph(_DebateGraphState)

        def _make_debater_node(name: str, debater: "Agent", round_num: int, index: int) -> Any:
            def _node(state: _DebateGraphState) -> dict[str, Any]:
                prompt = _debate_prompt(
                    state["task"],
                    debater_names,
                    state["transcript"],
                    agent_name=name,
                    round_num=round_num,
                )
                result = debater.run(prompt, **kwargs)
                return {"results": [(index, result)]}

            return _node

        def _make_join_node(round_num: int) -> Any:
            start_index = round_num * n

            def _node(state: _DebateGraphState) -> dict[str, Any]:
                round_entries = sorted(
                    (idx, r) for idx, r in state["results"] if start_index <= idx < start_index + n
                )
                round_results = [r for _, r in round_entries]
                new_transcript = {name: list(state["transcript"][name]) for name in debater_names}
                for name, result in zip(debater_names, round_results, strict=True):
                    new_transcript[name].append(result.content)
                return {"transcript": new_transcript, "steps": [*state["steps"], *round_results]}

            return _node

        previous_join_name: Optional[str] = None
        for round_num in range(max_rounds):
            round_node_names = []
            for i, (name, debater) in enumerate(debaters.items()):
                index = round_num * n + i
                node_name = f"{name}_r{round_num}"
                graph.add_node(node_name, _make_debater_node(name, debater, round_num, index))
                if previous_join_name is None:
                    graph.add_edge(START, node_name)
                else:
                    graph.add_edge(previous_join_name, node_name)
                round_node_names.append(node_name)

            join_name = f"__debate_join_r{round_num}__"
            graph.add_node(join_name, _make_join_node(round_num))
            graph.add_edge(round_node_names, join_name)
            previous_join_name = join_name

        def _verdict_node(state: _DebateGraphState) -> dict[str, Any]:
            verdict = moderator.run(
                _debate_verdict_prompt(state["task"], debater_names, state["transcript"]),
                **kwargs,
            )
            return {"output": verdict.content, "steps": [*state["steps"], verdict]}

        graph.add_node(_VERDICT_NODE, _verdict_node)
        graph.add_edge(START if previous_join_name is None else previous_join_name, _VERDICT_NODE)
        graph.add_edge(_VERDICT_NODE, END)

        return graph.compile(), debater_names

    def _build_tree_of_thoughts_graph(
        self,
        steps: Sequence["Agent"],
        *,
        breadth: int,
        beam_width: int,
        max_depth: int,
        **kwargs: Any,
    ) -> Any:
        """Build a graph for the ``tree_of_thoughts`` strategy: a beam
        search unrolled into ``max_depth`` levels, each a fan-out of
        candidate-thought nodes joined into one structured-output
        evaluation/prune node.

        ``breadth``/``beam_width``/``max_depth`` fully determine every
        level's fan-out width at graph-build time (see docs/adr/0034 for
        the induction proof) even though the actual thoughts are only
        known at runtime -- the same "unroll since the shape is
        build-time-known" reasoning ADR-0032 used for ``map_reduce`` and
        ADR-0033 used for ``debate``. Mirrors
        :meth:`~requisite.orchestrators.native.NativeOrchestrator._run_tree_of_thoughts`
        exactly, including its per-level bookkeeping (thinkers assigned
        round-robin *within* each level, not globally) and its
        termination rules (an early-finished candidate wins immediately;
        otherwise the top-ranked survivor after ``max_depth`` levels).

        Node names are always synthetic (``__tot_L{level}_{i}__``, never
        derived from thinker names), so unlike ``consensus`` this needs
        no ``_reject_reserved_node_names`` call -- the same argument
        already used for ``map_reduce``'s synthetic ``__mapper_{i}__``
        names.
        """
        StateGraph, START, END = self._require_langgraph()  # noqa: N806
        NativeOrchestrator._validate_tot_params(
            breadth=breadth, beam_width=beam_width, max_depth=max_depth
        )
        evaluator, thinkers = NativeOrchestrator._split_coordinator_and_workers(
            steps, role="tree-of-thoughts evaluator"
        )
        thinker_list = list(thinkers.values())

        # Precompute every level's fan-out width and its offset into the
        # one shared `candidates` reducer channel -- pure arithmetic, no
        # agent calls. paths_count[0] = 1 (the root empty path); each
        # level's candidate count is paths_count[level] * breadth, and
        # the next level's surviving-path count is min(beam_width, that).
        paths_count = [1]
        level_widths: list[int] = []
        for level in range(max_depth):
            width = paths_count[level] * breadth
            level_widths.append(width)
            paths_count.append(min(beam_width, width))
        level_offsets: list[int] = []
        running = 0
        for width in level_widths:
            level_offsets.append(running)
            running += width

        graph = StateGraph(_TreeOfThoughtsGraphState)

        def _make_candidate_node(index_in_level: int, global_index: int) -> Any:
            def _node(state: _TreeOfThoughtsGraphState) -> dict[str, Any]:
                parent_path = state["paths"][index_in_level // breadth]
                thinker = thinker_list[index_in_level % len(thinker_list)]
                result = thinker.run(_tot_thinker_prompt(state["task"], parent_path), **kwargs)
                return {"candidates": [(global_index, result)]}

            return _node

        def _make_eval_node(level: int) -> Any:
            offset = level_offsets[level]
            width = level_widths[level]
            is_last_level = level == max_depth - 1

            def _node(state: _TreeOfThoughtsGraphState) -> dict[str, Any]:
                level_entries = sorted(
                    (idx, r) for idx, r in state["candidates"] if offset <= idx < offset + width
                )
                candidate_results = [r for _, r in level_entries]
                candidate_paths = [
                    [*state["paths"][i // breadth], result.content]
                    for i, result in enumerate(candidate_results)
                ]
                evaluation = evaluator.ai.chat(
                    _tot_evaluation_prompt(state["task"], candidate_paths),
                    response_model=_ThoughtEvaluation,
                )
                final = NativeOrchestrator._select_finished_tot_candidate(
                    evaluation, candidate_paths
                )
                if final is not None:
                    return {
                        "output": final[-1],
                        "steps": [*state["steps"], *candidate_results],
                        "finished": True,
                    }
                pruned = NativeOrchestrator._prune_tot_candidates(
                    evaluation, candidate_paths, beam_width=beam_width
                )
                update: dict[str, Any] = {
                    "paths": pruned,
                    "steps": [*state["steps"], *candidate_results],
                    "finished": False,
                }
                if is_last_level:
                    update["output"] = pruned[0][-1] if pruned[0] else ""
                return update

            return _node

        def _make_router(level: int) -> Any:
            def _router(state: _TreeOfThoughtsGraphState) -> Any:
                if state["finished"] or level == max_depth - 1:
                    return END
                return f"__tot_expand_L{level + 1}__"

            return _router

        def _expand_node(state: _TreeOfThoughtsGraphState) -> dict[str, Any]:
            return {}

        level_candidate_names: list[list[str]] = []
        for level in range(max_depth):
            names = [f"__tot_L{level}_{i}__" for i in range(level_widths[level])]
            for i, name in enumerate(names):
                graph.add_node(name, _make_candidate_node(i, level_offsets[level] + i))
            level_candidate_names.append(names)

            eval_name = f"__tot_eval_L{level}__"
            graph.add_node(eval_name, _make_eval_node(level))
            graph.add_edge(names, eval_name)
            graph.add_conditional_edges(eval_name, _make_router(level))

        for name in level_candidate_names[0]:
            graph.add_edge(START, name)

        for level in range(1, max_depth):
            expand_name = f"__tot_expand_L{level}__"
            graph.add_node(expand_name, _expand_node)
            for name in level_candidate_names[level]:
                graph.add_edge(expand_name, name)

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
        if strategy in ("reflection", "critic"):
            max_rounds = kwargs.pop("max_rounds", 3)
            compiled_graph = self._build_reflection_graph(
                steps, role=strategy, max_rounds=max_rounds, **kwargs
            )
            final_state = compiled_graph.invoke(
                {"input": input, "draft": "", "critique": "", "steps": [], "rounds": 0},
            )
            return WorkflowResult(
                content=final_state["draft"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        if strategy == "debate":
            max_rounds = kwargs.pop("max_rounds", 3)
            compiled_graph, debater_names = self._build_debate_graph(
                steps, max_rounds=max_rounds, **kwargs
            )
            final_state = compiled_graph.invoke(
                {
                    "task": input,
                    "results": [],
                    "transcript": {name: [] for name in debater_names},
                    "steps": [],
                    "output": "",
                },
            )
            return WorkflowResult(
                content=final_state["output"],
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
        if strategy == "tree_of_thoughts":
            breadth = kwargs.pop("breadth", 3)
            beam_width = kwargs.pop("beam_width", 1)
            max_depth = kwargs.pop("max_depth", 3)
            compiled_graph = self._build_tree_of_thoughts_graph(
                steps, breadth=breadth, beam_width=beam_width, max_depth=max_depth, **kwargs
            )
            final_state = compiled_graph.invoke(
                {
                    "task": input,
                    "paths": [[]],
                    "candidates": [],
                    "steps": [],
                    "output": "",
                    "finished": False,
                },
            )
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        raise ConfigurationException(
            f"The langgraph orchestrator supports the 'sequential', 'supervisor', "
            f"'hierarchical', 'reflection', 'graph', 'parallel', 'consensus', "
            f"'map_reduce', 'critic', 'debate', and 'tree_of_thoughts' strategies "
            f"(got '{strategy}').",
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
        if strategy in ("reflection", "critic"):
            max_rounds = kwargs.pop("max_rounds", 3)
            compiled_graph = self._build_reflection_graph(
                steps, role=strategy, max_rounds=max_rounds, **kwargs
            )
            final_state = await compiled_graph.ainvoke(
                {"input": input, "draft": "", "critique": "", "steps": [], "rounds": 0},
            )
            return WorkflowResult(
                content=final_state["draft"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        if strategy == "debate":
            max_rounds = kwargs.pop("max_rounds", 3)
            compiled_graph, debater_names = self._build_debate_graph(
                steps, max_rounds=max_rounds, **kwargs
            )
            final_state = await compiled_graph.ainvoke(
                {
                    "task": input,
                    "results": [],
                    "transcript": {name: [] for name in debater_names},
                    "steps": [],
                    "output": "",
                },
            )
            return WorkflowResult(
                content=final_state["output"],
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
        if strategy == "tree_of_thoughts":
            breadth = kwargs.pop("breadth", 3)
            beam_width = kwargs.pop("beam_width", 1)
            max_depth = kwargs.pop("max_depth", 3)
            compiled_graph = self._build_tree_of_thoughts_graph(
                steps, breadth=breadth, beam_width=beam_width, max_depth=max_depth, **kwargs
            )
            final_state = await compiled_graph.ainvoke(
                {
                    "task": input,
                    "paths": [[]],
                    "candidates": [],
                    "steps": [],
                    "output": "",
                    "finished": False,
                },
            )
            return WorkflowResult(
                content=final_state["output"],
                steps=final_state["steps"],
                orchestrator=self.name,
                strategy=strategy,
            )
        raise ConfigurationException(
            f"The langgraph orchestrator supports the 'sequential', 'supervisor', "
            f"'hierarchical', 'reflection', 'graph', 'parallel', 'consensus', "
            f"'map_reduce', 'critic', 'debate', and 'tree_of_thoughts' strategies "
            f"(got '{strategy}').",
        )
