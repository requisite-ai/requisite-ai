"""
requisite
=========

Declare what your AI application needs -- not which SDK provides it.

A provider-agnostic, plugin-based framework for building AI-powered
applications and agents.

This top-level package intentionally exposes a *small* public surface.
Everything a typical developer needs to get started lives here; deeper
building blocks (provider base classes, registries, exceptions, etc.)
live in their respective sub-packages and can be imported explicitly
when you need to extend the framework.

Quick start
-----------
>>> from requisite import AI
>>> ai = AI()  # reads configuration from environment / .env
>>> response = ai.chat("Explain LangGraph in one sentence.")
>>> print(response)

Switching providers requires no code changes beyond configuration:
>>> ai = AI(provider="gemini", model="gemini-2.5-flash")

Tool calling:
>>> from requisite.tools import tool
>>> @tool
... def search_weather(city: str) -> str:
...     '''Look up the current weather for a city.'''
...     return f"Sunny in {city}"
>>> ai.chat("What's the weather in Paris?", tools=[search_weather])  # doctest: +SKIP

Agents and multi-agent workflows:
>>> from requisite import Agent, Workflow
>>> research = Agent(name="Researcher", provider="openai")  # doctest: +SKIP
>>> writer = Agent(name="Writer", provider="openai")  # doctest: +SKIP
>>> workflow = Workflow()
>>> workflow.add(research).add(writer)  # doctest: +SKIP
>>> workflow.run("Research AI trends and write a summary.")  # doctest: +SKIP
>>> workflow.use_langgraph()  # doctest: +SKIP

Declaring capabilities instead of binding to specific tool implementations:
>>> assistant = Agent(name="Assistant", provider="openai")  # doctest: +SKIP
>>> assistant.requires("weather", "internet_search", "filesystem")  # doctest: +SKIP
>>> assistant.run("What's the weather in Tokyo?")  # doctest: +SKIP
"""

from requisite.agents.agent import Agent, AgentResult
from requisite.agents.registry import AgentRegistry
from requisite.ai import AI
from requisite.capabilities.registry import CapabilityProvider, CapabilityRegistry
from requisite.capabilities import default_registry as default_capability_registry
from requisite.config.settings import Settings
from requisite.core.exceptions import (
    AgentException,
    AIException,
    CapabilityException,
    ConfigurationException,
    MCPException,
    ProviderException,
    SkillException,
    ToolException,
)
from requisite.core.interfaces import ChatResponse, Message, Role, ToolCall
from requisite.orchestrators.base import WorkflowResult
from requisite.providers.factory import ProviderRegistry, default_registry
from requisite.skills.base import BaseSkill
from requisite.skills.registry import SkillRegistry
from requisite.tools.base import Tool
from requisite.tools.decorator import tool
from requisite.tools.registry import ToolRegistry
from requisite.workflows.workflow import Workflow

__all__ = [
    # The core facade
    "AI",
    "Settings",
    # Messages / responses
    "Message",
    "Role",
    "ChatResponse",
    "ToolCall",
    # Tool calling
    "tool",
    "Tool",
    "ToolRegistry",
    # Skills
    "BaseSkill",
    "SkillRegistry",
    # Agents
    "Agent",
    "AgentResult",
    "AgentRegistry",
    # Capabilities (agent.requires(...))
    "CapabilityRegistry",
    "CapabilityProvider",
    "default_capability_registry",
    # Multi-agent workflows
    "Workflow",
    "WorkflowResult",
    # Providers
    "ProviderRegistry",
    "default_registry",
    # Exceptions
    "AIException",
    "ProviderException",
    "ConfigurationException",
    "ToolException",
    "SkillException",
    "AgentException",
    "MCPException",
    "CapabilityException",
]

__version__ = "0.3.0"
