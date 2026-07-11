"""
Agent: an :class:`~requisite.ai.AI` instance equipped with tools and
skills, capable of autonomously deciding which to invoke and looping
until it has a final answer.

Examples
--------
>>> from requisite import Agent
>>> from requisite.tools import tool
>>>
>>> @tool
... def search_weather(city: str) -> str:
...     '''Look up the current weather for a city.'''
...     return f"Sunny, 22C in {city}"
>>>
>>> agent = Agent(name="Weather Agent", provider="openai", tools=[search_weather])  # doctest: +SKIP
>>> result = agent.run("What's the weather in Paris?")  # doctest: +SKIP
>>> print(result)  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict

from requisite.ai import AI
from requisite.capabilities.registry import CapabilityRegistry
from requisite.capabilities import default_registry as default_capability_registry
from requisite.config.settings import Settings
from requisite.core.exceptions import AgentException
from requisite.core.interfaces import ChatResponse, Message
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry
from requisite.skills.base import BaseSkill
from requisite.tools.base import Tool
from requisite.tools.registry import ToolRegistry

logger = logging.getLogger("requisite.agents")

ToolLike = Union[Tool, Callable[..., Any]]


class AgentResult(BaseModel):
    """The outcome of an :meth:`Agent.run` (or :meth:`Agent.arun`) call.

    Parameters
    ----------
    content:
        The agent's final answer.
    agent_name:
        Name of the agent that produced this result.
    iterations:
        Number of model round-trips it took to reach a final answer
        (i.e. 1 + the number of tool-calling loops).
    tool_calls_executed:
        Names of the tools that were invoked along the way, in order.
    raw_response:
        The final :class:`~requisite.core.interfaces.ChatResponse`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    agent_name: str
    iterations: int
    tool_calls_executed: list[str] = []
    raw_response: Optional[ChatResponse] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.content


class Agent:
    """An LLM configured with a name, tools, and skills, that can run tasks autonomously.

    Parameters
    ----------
    name:
        Human-readable identifier for the agent (used in
        :class:`~requisite.agents.registry.AgentRegistry` and in
        multi-agent orchestration logs/results).
    provider:
        Provider name (``"openai"``, ``"gemini"``) or an already-built
        :class:`~requisite.providers.base.BaseProvider` instance.
        Defaults to the configured default provider.
    model:
        Overrides the provider's default model.
    tools:
        Plain functions, ``@tool``-decorated functions, or
        :class:`~requisite.tools.base.Tool` instances this agent may call.
    skills:
        :class:`~requisite.skills.base.BaseSkill` instances this agent
        may call; each is automatically exposed to the model as a tool.
    requires:
        Capability names (e.g. ``"weather"``, ``"internet_search"``,
        ``"filesystem"``) this agent needs. Each is resolved, at
        construction time, to whichever registered implementation is
        currently available -- see
        :mod:`requisite.capabilities`. Prefer this over ``tools`` when
        you want the agent's code to stay stable as the underlying
        implementation (native tool, MCP server, paid API, plugin)
        evolves.
    capability_registry:
        The :class:`~requisite.capabilities.registry.CapabilityRegistry`
        used to resolve ``requires``. Defaults to the framework's
        built-in registry, pre-populated with reference implementations
        for ``"filesystem"``, ``"weather"``, and ``"internet_search"``.
    system_prompt:
        System prompt establishing the agent's role/persona.
    settings:
        Configuration source; defaults to a fresh
        :class:`~requisite.config.settings.Settings`.
    registry:
        Provider registry used to resolve ``provider`` by name.
    max_iterations:
        Maximum number of tool-calling round-trips before
        :class:`~requisite.core.exceptions.AgentException` is raised,
        guarding against infinite tool-call loops.

    Examples
    --------
    >>> agent = Agent(name="Research Agent", provider="openai")  # doctest: +SKIP
    >>> agent.run("Research AI trends.")  # doctest: +SKIP

    Declaring capabilities instead of binding to specific tools:

    >>> agent = Agent(name="Assistant", provider="openai")  # doctest: +SKIP
    >>> agent.requires("weather", "internet_search", "filesystem")  # doctest: +SKIP
    >>> agent.run("What's the weather in Tokyo?")  # doctest: +SKIP
    """

    def __init__(
        self,
        name: str,
        *,
        provider: Optional[Union[str, BaseProvider]] = None,
        model: Optional[str] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        skills: Optional[Sequence[BaseSkill]] = None,
        requires: Optional[Sequence[str]] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
        system_prompt: Optional[str] = None,
        settings: Optional[Settings] = None,
        registry: Optional[ProviderRegistry] = None,
        max_iterations: int = 5,
    ) -> None:
        self.name = name
        self.max_iterations = max_iterations
        self._capability_registry = capability_registry or default_capability_registry

        self._tool_registry = ToolRegistry()
        for tool_like in tools or []:
            self._tool_registry.register(tool_like)
        for skill in skills or []:
            self._tool_registry.register(skill.as_tool())

        self._ai = AI(
            provider=provider,
            model=model,
            settings=settings,
            registry=registry,
            system_prompt=system_prompt,
        )

        if requires:
            self.requires(*requires)

    @property
    def ai(self) -> AI:
        """The underlying :class:`~requisite.ai.AI` facade backing this agent."""
        return self._ai

    @property
    def tools(self) -> ToolRegistry:
        """The tool registry backing this agent's tool-calling loop."""
        return self._tool_registry

    def requires(self, *capabilities: str) -> "Agent":
        """Declare capabilities this agent needs, resolved to whichever
        implementation is currently available.

        This is the framework-agnostic alternative to binding an agent to
        a specific tool: ``agent.requires("weather")`` resolves against
        :class:`~requisite.capabilities.registry.CapabilityRegistry`
        (native tools, MCP servers, cloud APIs, or third-party plugins
        can all register the same capability name), so application code
        stays stable as the underlying implementation evolves.

        Parameters
        ----------
        *capabilities:
            One or more capability names (e.g. ``"weather"``,
            ``"internet_search"``, ``"filesystem"``, ``"github"``).

        Returns
        -------
        Agent
            ``self``, for fluent chaining:
            ``agent.requires("weather").requires("internet_search")``.

        Raises
        ------
        requisite.core.exceptions.CapabilityException
            If any requested capability has no registered provider, or
            every registered provider for it is currently unavailable.

        Examples
        --------
        >>> agent = Agent(name="Assistant", provider="openai")  # doctest: +SKIP
        >>> agent.requires("internet_search", "weather", "filesystem")  # doctest: +SKIP
        """
        for capability in capabilities:
            resolved_tool = self._capability_registry.resolve(capability)
            if resolved_tool.name != capability:
                # Expose the tool to the model under the stable capability
                # name, not the implementation's function name -- so the
                # model-facing tool name never changes even if the
                # provider behind "weather" is swapped out later.
                resolved_tool = resolved_tool.model_copy(update={"name": capability})
            self._tool_registry.register(resolved_tool)
        return self

    def run(self, prompt: Union[str, Sequence[Message]], **kwargs: Any) -> AgentResult:
        """Run the agent on a task, looping through tool calls until a final answer.

        Parameters
        ----------
        prompt:
            The task, as a string or an existing conversation history.
        **kwargs:
            Passed through to each underlying ``chat_response`` call
            (e.g. ``temperature``, ``model``).

        Returns
        -------
        AgentResult

        Raises
        ------
        requisite.core.exceptions.AgentException
            If ``max_iterations`` tool-calling round-trips are exhausted
            without the model returning a final answer.
        """
        messages: list[Message] = (
            [Message.user(prompt)] if isinstance(prompt, str) else list(prompt)
        )
        available_tools = self._tool_registry.all() or None
        tools_executed: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            response = self._ai.chat_response(messages, tools=available_tools, **kwargs)

            if not response.has_tool_calls:
                return AgentResult(
                    content=response.content,
                    agent_name=self.name,
                    iterations=iteration,
                    tool_calls_executed=tools_executed,
                    raw_response=response,
                )

            messages.append(
                Message.assistant_tool_calls(response.tool_calls, content=response.content)
            )
            for call in response.tool_calls:
                tool_instance = self._tool_registry.get(call.name)
                result = tool_instance.execute(**call.arguments)
                tools_executed.append(call.name)
                messages.append(
                    Message.tool_result(str(result), tool_call_id=call.id, name=call.name)
                )

        raise AgentException(
            f"Agent '{self.name}' exceeded max_iterations={self.max_iterations} "
            f"without reaching a final answer.",
        )

    async def arun(self, prompt: Union[str, Sequence[Message]], **kwargs: Any) -> AgentResult:
        """Async counterpart to :meth:`run`."""
        messages: list[Message] = (
            [Message.user(prompt)] if isinstance(prompt, str) else list(prompt)
        )
        available_tools = self._tool_registry.all() or None
        tools_executed: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            response = await self._ai.achat_response(messages, tools=available_tools, **kwargs)

            if not response.has_tool_calls:
                return AgentResult(
                    content=response.content,
                    agent_name=self.name,
                    iterations=iteration,
                    tool_calls_executed=tools_executed,
                    raw_response=response,
                )

            messages.append(
                Message.assistant_tool_calls(response.tool_calls, content=response.content)
            )
            for call in response.tool_calls:
                tool_instance = self._tool_registry.get(call.name)
                result = await tool_instance.aexecute(**call.arguments)
                tools_executed.append(call.name)
                messages.append(
                    Message.tool_result(str(result), tool_call_id=call.id, name=call.name)
                )

        raise AgentException(
            f"Agent '{self.name}' exceeded max_iterations={self.max_iterations} "
            f"without reaching a final answer.",
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Agent(name={self.name!r}, provider={self._ai.provider.name!r})"
