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
>>> ai = AI(provider="anthropic", model="claude-sonnet-4-6")
>>> ai = AI(provider="groq", model="llama-3.3-70b-versatile")

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

Connecting to an MCP server -- its tools become capabilities like any other:
>>> from requisite.mcp import MCPClient, default_mcp_registry
>>> from requisite.capabilities import default_registry as default_capability_registry
>>> github = MCPClient.http(name="github", url="https://api.example.com/mcp")  # doctest: +SKIP
>>> default_mcp_registry.register(github)  # doctest: +SKIP
>>> github.register_as_capability(default_capability_registry, capability="github")  # doctest: +SKIP
>>> assistant.requires("github")  # doctest: +SKIP

Persisting conversation history across separate run() calls, kept bounded
with a conversation policy:
>>> from requisite.memory import InProcessMemory, MessageCountPolicy
>>> memory = InProcessMemory()
>>> chatty = Agent(
...     name="Assistant", provider="openai", memory=memory, session_id="user-42",
...     conversation_policy=MessageCountPolicy(max_messages=20),
... )  # doctest: +SKIP
>>> chatty.run("My name is Alex.")  # doctest: +SKIP
>>> chatty.run("What's my name?")  # doctest: +SKIP

Reusable, parameterized prompts:
>>> from requisite.prompts import ChatPromptTemplate
>>> chat_template = ChatPromptTemplate.from_messages([
...     ("system", "You are a {persona}."),
...     ("user", "{question}"),
... ])
>>> ai.chat(chat_template.format_messages(persona="pirate", question="Where's the treasure?"))  # doctest: +SKIP

Structured (JSON) logging, opt-in:
>>> from requisite.telemetry import configure_logging
>>> configure_logging(level="DEBUG", json_format=True)  # doctest: +SKIP
"""

from requisite.agents.agent import Agent, AgentResult
from requisite.agents.registry import AgentRegistry
from requisite.ai import AI
from requisite.capabilities import default_registry as default_capability_registry
from requisite.capabilities.registry import CapabilityProvider, CapabilityRegistry
from requisite.config.settings import Settings
from requisite.core.exceptions import (
    AgentException,
    AIException,
    CapabilityException,
    ConfigurationException,
    MCPException,
    PromptException,
    ProviderException,
    SkillException,
    ToolException,
)
from requisite.core.interfaces import ChatResponse, Message, Role, ToolCall
from requisite.mcp import default_mcp_registry
from requisite.mcp.base import BaseMCPClient
from requisite.mcp.client import MCPClient
from requisite.mcp.registry import MCPClientRegistry
from requisite.memory import default_registry as default_memory_registry
from requisite.memory.base import BaseMemory
from requisite.memory.factory import MemoryRegistry
from requisite.memory.in_process import InProcessMemory
from requisite.memory.policies import BaseConversationPolicy, MessageCountPolicy, SummarizingPolicy
from requisite.orchestrators.base import WorkflowResult
from requisite.prompts import default_prompt_registry
from requisite.prompts.registry import PromptTemplateRegistry
from requisite.prompts.template import ChatPromptTemplate, PromptTemplate
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
    # MCP
    "BaseMCPClient",
    "MCPClient",
    "MCPClientRegistry",
    "default_mcp_registry",
    # Memory + conversation management
    "BaseMemory",
    "InProcessMemory",
    "MemoryRegistry",
    "default_memory_registry",
    "BaseConversationPolicy",
    "MessageCountPolicy",
    "SummarizingPolicy",
    # Prompt templates
    "PromptTemplate",
    "ChatPromptTemplate",
    "PromptTemplateRegistry",
    "default_prompt_registry",
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
    "PromptException",
]

__version__ = "0.2.0"
