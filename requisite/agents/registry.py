"""Agent registry -- mirrors the shape of the other framework registries."""

from __future__ import annotations

import logging

from requisite.agents.agent import Agent
from requisite.core.exceptions import AgentException

logger = logging.getLogger("requisite.agents.registry")


class AgentRegistry:
    """A registry of agents, keyed by name.

    Examples
    --------
    >>> from requisite.agents import Agent, AgentRegistry
    >>> registry = AgentRegistry()
    >>> agent = Agent(name="research")  # doctest: +SKIP
    >>> registry.register(agent)  # doctest: +SKIP
    >>> registry.get("research")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> Agent:
        """Register an agent under its ``.name``. Returns the agent for chaining."""
        if agent.name in self._agents:
            logger.debug("Overwriting existing agent registration: '%s'", agent.name)
        self._agents[agent.name] = agent
        return agent

    def unregister(self, name: str) -> None:
        """Remove an agent registration, if present. No-op if absent."""
        self._agents.pop(name, None)

    def get(self, name: str) -> Agent:
        """Return the registered agent named ``name``.

        Raises
        ------
        requisite.core.exceptions.AgentException
            If no agent is registered under that name.
        """
        try:
            return self._agents[name]
        except KeyError as exc:
            raise AgentException(
                f"No agent registered under '{name}'. Available: {self.list_names()}",
            ) from exc

    def list_names(self) -> list[str]:
        """Return the sorted list of registered agent names."""
        return sorted(self._agents)

    def all(self) -> list[Agent]:
        """Return all registered agents."""
        return list(self._agents.values())

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)
