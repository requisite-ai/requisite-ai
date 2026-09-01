"""
``Workflow``: compose agents into a multi-agent pipeline.

Examples
--------
>>> from requisite import Agent, Workflow
>>>
>>> research = Agent(name="Researcher", provider="openai")  # doctest: +SKIP
>>> writer = Agent(name="Writer", provider="openai")  # doctest: +SKIP
>>>
>>> workflow = Workflow()
>>> workflow.add(research)  # doctest: +SKIP
>>> workflow.add(writer)  # doctest: +SKIP
>>> result = workflow.run("Research and summarize the latest AI trends.")  # doctest: +SKIP

Switching the execution engine is a configuration change, not a rewrite:

>>> workflow.use_langgraph()  # doctest: +SKIP
>>> result = workflow.run("Research and summarize the latest AI trends.")  # doctest: +SKIP

Running agents in parallel instead of as a pipeline:

>>> workflow.parallel()  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.orchestrators.base import END, WorkflowResult
from requisite.orchestrators.factory import OrchestratorRegistry
from requisite.orchestrators.factory import default_registry as default_orchestrator_registry

logger = logging.getLogger("requisite.workflows")

_KNOWN_STRATEGIES = {
    "sequential",
    "parallel",
    "reflection",
    "planner",
    "supervisor",
    "critic",
    "consensus",
    "debate",
    "map_reduce",
    "hierarchical",
    "tree_of_thoughts",
    "reflexion",
    "graph",
}

__all__ = ["END", "Workflow"]


class Workflow:
    """Compose :class:`~requisite.agents.agent.Agent` instances into a pipeline.

    Parameters
    ----------
    strategy:
        How agents are executed relative to one another:

        - ``"sequential"`` (default): each agent's output feeds the next.
        - ``"parallel"``: every agent runs against the same input
          concurrently.
        - ``"reflection"``: a single agent (see :meth:`reflection`)
          critiques and revises its own output over several rounds.
        - ``"planner"``: the first agent breaks the task into a plan of
          subtasks for the remaining agents (see :meth:`planner`).
        - ``"supervisor"``: the first agent delegates subtasks to the
          remaining agents one at a time until the task is done (see
          :meth:`supervisor`).
        - ``"critic"``: two agents -- a generator and a separate critic
          -- iterate on a draft together (see :meth:`critic`).
        - ``"consensus"``: the first agent synthesizes the independent
          answers of the remaining agents into one (see :meth:`consensus`).
        - ``"debate"``: the first agent moderates while the rest debate
          the input over several rounds (see :meth:`debate`).
        - ``"map_reduce"``: the first agent reduces the results the
          remaining agents produce from ``map_items=`` (see
          :meth:`map_reduce`).
        - ``"hierarchical"``: same as ``"supervisor"``, except a
          delegate may also be another named ``Workflow`` ("team"),
          giving real recursive delegation (see :meth:`hierarchical`).
        - ``"tree_of_thoughts"``: the first agent evaluates candidate
          reasoning steps the remaining agents generate, branching and
          pruning a search tree over several levels (see
          :meth:`tree_of_thoughts`).
        - ``"reflexion"``: a single agent attempts the task from
          scratch, is scored by a pluggable evaluator, and reflects on
          failure before retrying, over several independent trials (see
          :meth:`reflexion`).
        - ``"graph"``: an arbitrary graph of nodes wired together with
          explicit, developer-declared edges (see :meth:`graph` and
          :meth:`add_edge`) -- unlike every strategy above, routing isn't
          decided by an LLM at run time; the graph's shape is fixed when
          you build it.

        Every strategy except ``"reflexion"`` runs on both the
        ``"native"`` and ``"langgraph"`` orchestrator backends.
        ``"reflexion"`` is currently ``"native"``-only; langgraph parity
        is a natural follow-up, not yet built. On langgraph,
        ``"supervisor"``/``"hierarchical"`` are a real conditional graph
        (``add_conditional_edges`` plus a loop-back cycle) and
        ``"graph"`` builds the identical developer-declared graph
        natively, not a disguised Python loop; see
        ``docs/adr/0016-langgraph-branching.md``,
        ``docs/adr/0028-langgraph-reflection-strategy.md``,
        ``docs/adr/0029-langgraph-hierarchical-graph-strategies.md``,
        ``docs/adr/0032-langgraph-parallel-consensus-map-reduce-strategies.md``,
        ``docs/adr/0033-langgraph-critic-debate-strategies.md``,
        ``docs/adr/0034-langgraph-tree-of-thoughts-strategy.md``, and
        ``docs/adr/0035-langgraph-planner-strategy.md``.
    orchestrator:
        Execution backend: ``"native"`` (default, pure Python, no extra
        dependency), ``"langgraph"``, ``"crewai"``, or ``"autogen"`` --
        each supports a different strategy subset, see :meth:`use_langgraph`/
        :meth:`use_crewai`/:meth:`use_autogen`. Change via those methods
        (or :meth:`use_native`) instead of passing this directly, for
        readability at the call site.
    registry:
        The :class:`~requisite.orchestrators.factory.OrchestratorRegistry`
        used to resolve ``orchestrator``. Defaults to the framework's
        built-in registry.
    name:
        Optional identifier. Not needed for standalone use -- only
        required when this ``Workflow`` is used as a delegate ("team")
        inside another workflow's :meth:`hierarchical` strategy, the
        same way an :class:`~requisite.agents.agent.Agent` needs a name
        to be addressed by a coordinator.

    Examples
    --------
    >>> workflow = Workflow()
    >>> workflow.add(research).add(writer)  # doctest: +SKIP
    >>> workflow.run("Research AI trends.")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        strategy: str = "sequential",
        orchestrator: str = "native",
        registry: Optional[OrchestratorRegistry] = None,
    ) -> None:
        self.name = name
        self._steps: list[Any] = []
        self._edges: list[tuple[str, str, Optional[Callable[[str], bool]]]] = []
        self._strategy = strategy
        self._orchestrator_name = orchestrator
        self._registry = registry or default_orchestrator_registry

    def add(self, agent: Any) -> "Workflow":
        """Append a step to the pipeline. Returns ``self`` for chaining.

        Normally an :class:`~requisite.agents.agent.Agent`. Under the
        :meth:`hierarchical` or :meth:`graph` strategies, a step may
        instead be another named ``Workflow`` used as a delegate/node
        ("team") -- see :meth:`hierarchical` and :meth:`graph`.
        """
        self._steps.append(agent)
        return self

    def add_edge(
        self, from_: str, to: str, *, condition: Optional[Callable[[str], bool]] = None
    ) -> "Workflow":
        """Declare an edge for the :meth:`graph` strategy. Returns ``self`` for chaining.

        ``from_`` and ``to`` are node names (``agent.name``, matching how
        every other strategy addresses workers/delegates by name), or
        ``to=`` may be :data:`~requisite.workflows.workflow.END` to mark a
        terminating edge. ``condition``, if given, receives the source
        node's output content (a ``str``) and returns ``bool``; ``None``
        means "always matches" -- useful as a fallback/default edge when
        added last. When a node has multiple outgoing edges, they're
        evaluated in the order ``add_edge`` was called and the first match
        wins. A node with no outgoing edges at all terminates the graph
        implicitly, so wiring every leaf to ``END`` is optional.
        """
        self._edges.append((from_, to, condition))
        return self

    @property
    def agents(self) -> list[Any]:
        """The steps added so far, in order (usually ``Agent`` instances --
        see :meth:`add`)."""
        return list(self._steps)

    def sequential(self) -> "Workflow":
        """Switch to the sequential execution strategy. Returns ``self`` for chaining."""
        self._strategy = "sequential"
        return self

    def parallel(self) -> "Workflow":
        """Switch to the parallel execution strategy. Returns ``self`` for chaining."""
        self._strategy = "parallel"
        return self

    def reflection(self) -> "Workflow":
        """Switch to the reflection strategy. Returns ``self`` for chaining.

        Requires exactly one agent (added via :meth:`add`), which
        critiques and revises its own output. Pass ``max_rounds=`` to
        :meth:`run`/:meth:`arun` to control how many rounds it gets
        (default 3); it may also stop earlier on its own. Works on both
        :meth:`use_native` (the default) and :meth:`use_langgraph` -- the
        langgraph backend builds a real conditional graph (a 3-node
        draft/critique/revise cycle) rather than a Python loop, see
        ``docs/adr/0028-langgraph-reflection-strategy.md``.
        """
        self._strategy = "reflection"
        return self

    def planner(self) -> "Workflow":
        """Switch to the planner strategy. Returns ``self`` for chaining.

        The first agent added becomes the planner; every agent added
        after it is a worker the planner can assign subtasks to, by
        name. Requires at least 2 agents.
        """
        self._strategy = "planner"
        return self

    def supervisor(self) -> "Workflow":
        """Switch to the supervisor strategy. Returns ``self`` for chaining.

        The first agent added becomes the supervisor; every agent added
        after it is a worker the supervisor can delegate to, by name.
        Requires at least 2 agents. Pass ``max_rounds=`` to
        :meth:`run`/:meth:`arun` to bound how many delegation rounds it
        gets before giving up (default 6). Works on both
        :meth:`use_native` (the default) and :meth:`use_langgraph` --
        the langgraph backend builds a real conditional graph rather
        than a Python loop, see ``docs/adr/0016-langgraph-branching.md``.
        """
        self._strategy = "supervisor"
        return self

    def critic(self) -> "Workflow":
        """Switch to the critic strategy. Returns ``self`` for chaining.

        Requires exactly two agents: the first added is the generator,
        the second is the critic that reviews the generator's drafts.
        Same shape as :meth:`reflection`, generalized to two distinct
        agents instead of one agent critiquing itself. Pass
        ``max_rounds=`` to :meth:`run`/:meth:`arun` to control how many
        rounds it gets (default 3); it may also stop earlier on its own.
        """
        self._strategy = "critic"
        return self

    def consensus(self) -> "Workflow":
        """Switch to the consensus strategy. Returns ``self`` for chaining.

        The first agent added becomes the synthesizer; every agent added
        after it independently answers the same input, then the
        synthesizer combines their answers into one final response.
        Requires at least 2 agents.
        """
        self._strategy = "consensus"
        return self

    def debate(self) -> "Workflow":
        """Switch to the debate strategy. Returns ``self`` for chaining.

        The first agent added becomes the moderator (never debates,
        only delivers the final verdict); every agent added after it is
        a debater. Each round, every debater sees every debater's
        arguments from the *previous* round and responds. Requires at
        least 2 agents. Pass ``max_rounds=`` to :meth:`run`/:meth:`arun`
        to control how many rounds of debate happen before the
        moderator's verdict (default 3).
        """
        self._strategy = "debate"
        return self

    def hierarchical(self) -> "Workflow":
        """Switch to the hierarchical strategy. Returns ``self`` for chaining.

        Same shape as :meth:`supervisor`: the first agent added becomes
        the coordinator, every agent added after it is a delegate the
        coordinator can route to by name. The difference: a delegate
        may be either an :class:`~requisite.agents.agent.Agent` *or*
        another named ``Workflow`` (a "team") -- delegating to a
        ``Workflow`` runs whatever strategy it's configured with,
        including another ``supervisor`` or ``hierarchical``, giving
        real recursive delegation. A ``Workflow`` used this way must be
        constructed with ``name=...``. Requires at least 2 steps. Pass
        ``max_rounds=`` to :meth:`run`/:meth:`arun` to bound how many
        delegation rounds it gets before giving up (default 6). Works on
        both :meth:`use_native` (the default) and :meth:`use_langgraph`
        -- the langgraph backend builds a real conditional graph rather
        than a Python loop, see
        ``docs/adr/0029-langgraph-hierarchical-graph-strategies.md``.
        """
        self._strategy = "hierarchical"
        return self

    def map_reduce(self) -> "Workflow":
        """Switch to the map-reduce strategy. Returns ``self`` for chaining.

        The first agent added becomes the reducer; every agent added
        after it is a mapper. Pass ``map_items=[...]`` to
        :meth:`run`/:meth:`arun` -- the list of individual items to
        process, assigned to mappers round-robin (so the item count
        doesn't need to match the mapper count) and run concurrently.
        The reducer then combines every item's result into one final
        answer. ``input`` is still required -- it's the overall task/goal
        that frames both the per-item mapping and the final reduction,
        not one of the items itself. Requires at least 2 agents.
        """
        self._strategy = "map_reduce"
        return self

    def tree_of_thoughts(self) -> "Workflow":
        """Switch to the tree-of-thoughts strategy. Returns ``self`` for chaining.

        The first agent added becomes the evaluator; every agent added
        after it is a thinker that proposes candidate next reasoning
        steps, assigned round-robin. Each level, pass ``breadth=`` to
        :meth:`run`/:meth:`arun` to control how many candidates are
        generated per surviving path (default 3); all candidates that
        level are scored together in one call, then pruned to the top
        ``beam_width=`` (default 1) before continuing. Runs for up to
        ``max_depth=`` levels (default 3), stopping early if any
        candidate is scored as a complete final answer. Requires at
        least 2 agents. See
        ``docs/adr/0018-tree-of-thoughts-strategy.md``.
        """
        self._strategy = "tree_of_thoughts"
        return self

    def reflexion(self) -> "Workflow":
        """Switch to the reflexion strategy. Returns ``self`` for chaining.

        Requires exactly one agent. Unlike :meth:`reflection`, which
        revises the *same* draft, each trial here attempts the whole
        task again from scratch. Pass ``evaluator=`` to
        :meth:`run`/:meth:`arun` (a ``(task, attempt_content) ->
        EvaluationResult`` callable -- see
        :class:`~requisite.orchestrators.base.EvaluationResult`) to
        score each attempt; omit it and the same agent judges its own
        attempt via structured output instead. On failure, a
        self-reflection lesson is generated and folded into the next
        attempt's prompt. Runs for up to ``max_trials=`` independent
        trials (default 3), stopping as soon as the evaluator reports
        success. See ``docs/adr/0036-reflexion-strategy.md``.

        Only implemented on :meth:`use_native` (the default) today.
        """
        self._strategy = "reflexion"
        return self

    def graph(self) -> "Workflow":
        """Switch to the graph strategy. Returns ``self`` for chaining.

        Execute an arbitrary graph of nodes (added via :meth:`add`, each
        an :class:`~requisite.agents.agent.Agent` or a named ``Workflow``
        "team") wired together with edges declared via :meth:`add_edge`.
        The first node added is the entry point. Unlike every other
        strategy, routing is deterministic and developer-declared, not
        decided by an LLM at run time: each node runs, then its output
        content is checked against its outgoing edges (in the order they
        were added) to pick the next node. Cycles are allowed -- pass
        ``max_steps=`` to :meth:`run`/:meth:`arun` to bound them (default
        25). See ``docs/adr/0019-graph-execution-strategy.md``. Works on
        both :meth:`use_native` (the default) and :meth:`use_langgraph`
        -- the langgraph backend builds a real conditional graph reusing
        the exact same edge-validation/routing logic, see
        ``docs/adr/0029-langgraph-hierarchical-graph-strategies.md``.
        """
        self._strategy = "graph"
        return self

    def use_native(self) -> "Workflow":
        """Use the built-in, dependency-free execution engine. Returns ``self`` for chaining."""
        self._orchestrator_name = "native"
        return self

    def use_langgraph(self) -> "Workflow":
        """Delegate execution to `langgraph <https://github.com/langchain-ai/langgraph>`_.

        Requires ``pip install langgraph``. Returns ``self`` for chaining.
        """
        self._orchestrator_name = "langgraph"
        return self

    def use_crewai(self) -> "Workflow":
        """Delegate coordination to `CrewAI <https://github.com/crewAIInc/crewAI>`_.

        Requires ``pip install crewai``. Returns ``self`` for chaining.
        Supports the ``"sequential"`` strategy only (CrewAI's
        ``Process.sequential``) -- ``"hierarchical"`` isn't implemented
        yet, see ``docs/adr/0027-crewai-autogen-orchestrator-backends.md``.
        Every actual model call still goes through each agent's own
        configured provider (CrewAI handles coordination only, not LLM
        calls) -- see the ADR for why.
        """
        self._orchestrator_name = "crewai"
        return self

    def use_autogen(self) -> "Workflow":
        """Delegate coordination to `AutoGen <https://github.com/microsoft/autogen>`_
        (``autogen-agentchat``/``autogen-core``).

        Requires ``pip install autogen-agentchat autogen-core``. Returns
        ``self`` for chaining. Supports ``"sequential"``
        (``RoundRobinGroupChat``) and ``"supervisor"``
        (``SelectorGroupChat``, reusing the same decision protocol
        :meth:`use_langgraph`'s supervisor graph already reuses from the
        native backend) -- see
        ``docs/adr/0027-crewai-autogen-orchestrator-backends.md``. Every
        actual model call still goes through each agent's own configured
        provider (AutoGen handles coordination only, not LLM calls).
        """
        self._orchestrator_name = "autogen"
        return self

    def run(self, input: Optional[str] = None, **kwargs: Any) -> WorkflowResult:  # noqa: A002
        """Execute the workflow.

        Parameters
        ----------
        input:
            The initial task/prompt handed to the first agent (or to
            every agent, under the ``"parallel"`` strategy). Still
            required under ``"map_reduce"`` -- see :meth:`map_reduce`.
        **kwargs:
            Passed through to each agent's ``run``/``arun`` call, except
            strategy-specific keywords a strategy consumes itself
            (``max_rounds=``, ``map_items=``).

        Returns
        -------
        WorkflowResult

        Raises
        ------
        requisite.core.exceptions.ConfigurationException
            If this ``Workflow`` already appears in the current
            delegation chain (a ``hierarchical`` delegate or ``graph``
            node that is this same ``Workflow``, directly or through a
            cycle of other ``Workflow``\\ s) -- see
            ``docs/adr/0031-code-review-fixes.md``.
        """
        if self._strategy not in _KNOWN_STRATEGIES:
            raise ConfigurationException(
                f"Unknown strategy '{self._strategy}'. Supported: {sorted(_KNOWN_STRATEGIES)}",
            )
        if self._strategy == "graph":
            kwargs.setdefault("edges", self._edges)
        chain = kwargs.pop("_delegation_chain", ())
        self._check_delegation_cycle(chain)
        kwargs["_delegation_chain"] = (*chain, (id(self), self.name))
        orchestrator_instance = self._registry.create(self._orchestrator_name)
        return orchestrator_instance.run(self._steps, input, strategy=self._strategy, **kwargs)

    async def arun(self, input: Optional[str] = None, **kwargs: Any) -> WorkflowResult:  # noqa: A002
        """Async counterpart to :meth:`run`."""
        if self._strategy not in _KNOWN_STRATEGIES:
            raise ConfigurationException(
                f"Unknown strategy '{self._strategy}'. Supported: {sorted(_KNOWN_STRATEGIES)}",
            )
        if self._strategy == "graph":
            kwargs.setdefault("edges", self._edges)
        chain = kwargs.pop("_delegation_chain", ())
        self._check_delegation_cycle(chain)
        kwargs["_delegation_chain"] = (*chain, (id(self), self.name))
        orchestrator_instance = self._registry.create(self._orchestrator_name)
        return await orchestrator_instance.arun(
            self._steps, input, strategy=self._strategy, **kwargs
        )

    def _check_delegation_cycle(self, chain: tuple[tuple[int, Optional[str]], ...]) -> None:
        """Raise if this ``Workflow`` is already an ancestor in ``chain``.

        ``chain`` is threaded through ``**kwargs`` on every nested
        ``delegate.run(...)``/``node.run(...)`` call under the
        ``hierarchical``/``graph`` strategies (both backends) -- see
        ``docs/adr/0031-code-review-fixes.md``. Without this check, a
        Workflow that (directly or via a cycle of other Workflows)
        delegates back into itself recurses through the real Python
        call stack -- each nested call gets its own independent, fresh
        ``max_rounds``/``max_steps`` budget, so neither bound ever
        catches it -- and eventually crashes with an uncatchable
        ``RecursionError`` instead of this clean, immediate error.
        """
        if not any(workflow_id == id(self) for workflow_id, _ in chain):
            return
        chain_names = " -> ".join(name or "<unnamed>" for _, name in chain)
        this_name = self.name or "<unnamed>"
        raise ConfigurationException(
            f"Workflow delegation cycle detected: {chain_names} -> {this_name}. A Workflow "
            "cannot delegate back into itself, directly or through a chain of other "
            "Workflows, under the 'hierarchical' or 'graph' strategies -- max_rounds/"
            "max_steps cannot bound this, since each nested delegation starts a fresh, "
            "independent budget rather than sharing one across the chain.",
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        agent_names = [a.name for a in self._steps]
        name_part = f"name={self.name!r}, " if self.name is not None else ""
        return (
            f"Workflow({name_part}agents={agent_names!r}, strategy={self._strategy!r}, "
            f"orchestrator={self._orchestrator_name!r})"
        )
