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

Exposing Requisite's own tools/agents as an MCP server -- the reverse direction:
>>> from requisite.mcp import MCPServer
>>> server = MCPServer(name="my-tools", agents=[assistant])  # doctest: +SKIP
>>> server.run_stdio()  # doctest: +SKIP

Retrieval-augmented generation -- a retriever is a capability too:
>>> from requisite.rag import Retriever
>>> from requisite.rag.embeddings import OpenAIEmbeddingProvider
>>> from requisite.rag.vectorstores import InMemoryVectorStore
>>> retriever = Retriever(
...     embedding_provider=OpenAIEmbeddingProvider(api_key="sk-..."),
...     vector_store=InMemoryVectorStore(),
... )  # doctest: +SKIP
>>> retriever.add_texts(["Paris is the capital of France."])  # doctest: +SKIP
>>> default_capability_registry.register("knowledge_base", retriever.as_tool())  # doctest: +SKIP
>>> assistant.requires("knowledge_base")  # doctest: +SKIP

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

Proactive rate limiting, shared across every agent that calls the same
underlying API key/quota:
>>> from requisite import RateLimiter
>>> shared_limit = RateLimiter(requests_per_minute=15)
>>> research = Agent(name="Researcher", provider="gemini", rate_limiter=shared_limit)  # doctest: +SKIP
>>> writer = Agent(name="Writer", provider="gemini", rate_limiter=shared_limit)  # doctest: +SKIP

Discovering installed plugins (packages that register themselves under
the ``"requisite.plugins"`` entry-point group):
>>> from requisite.plugins import discover
>>> result = discover()  # doctest: +SKIP
>>> result.loaded  # doctest: +SKIP
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
    MemoryException,
    PromptException,
    ProviderException,
    RateLimitException,
    SkillException,
    ToolException,
    VectorStoreException,
)
from requisite.core.interfaces import ChatResponse, Message, Role, ToolCall
from requisite.core.rate_limiter import RateLimiter
from requisite.mcp import default_mcp_registry
from requisite.mcp.base import BaseMCPClient
from requisite.mcp.client import MCPClient
from requisite.mcp.registry import MCPClientRegistry
from requisite.mcp.server import MCPServer
from requisite.memory import default_registry as default_memory_registry
from requisite.memory.base import BaseMemory
from requisite.memory.factory import MemoryRegistry
from requisite.memory.in_process import InProcessMemory
from requisite.memory.policies import BaseConversationPolicy, MessageCountPolicy, SummarizingPolicy
from requisite.memory.sqlite import SQLiteMemory
from requisite.memory.vector import VectorMemory
from requisite.orchestrators.base import END, EvaluationResult, Evaluator, WorkflowResult
from requisite.prompts import default_prompt_registry
from requisite.prompts.registry import PromptTemplateRegistry
from requisite.prompts.template import ChatPromptTemplate, PromptTemplate
from requisite.providers.factory import ProviderRegistry, default_registry
from requisite.rag.base import (
    BaseEmbeddingProvider,
    BaseReranker,
    BaseRetriever,
    BaseVectorStore,
    Chunk,
    ScoredChunk,
)
from requisite.rag.bm25 import BM25Retriever
from requisite.rag.chunking import chunk_text
from requisite.rag.factory import (
    EmbeddingRegistry,
    VectorStoreRegistry,
    default_embedding_registry,
    default_vector_store_registry,
)
from requisite.rag.hybrid_retriever import HybridRetriever
from requisite.rag.reranker import LLMReranker
from requisite.rag.retriever import Retriever
from requisite.skills.base import BaseSkill
from requisite.skills.registry import SkillRegistry
from requisite.tools.base import Tool
from requisite.tools.decorator import tool
from requisite.tools.registry import ToolRegistry
from requisite.workflows.workflow import Workflow

__all__ = [
    # The core facade
    "AI",
    # Exceptions
    "AIException",
    # Agents
    "Agent",
    "AgentException",
    "AgentRegistry",
    "AgentResult",
    "BM25Retriever",
    "BaseConversationPolicy",
    "BaseEmbeddingProvider",
    # MCP
    "BaseMCPClient",
    # Memory + conversation management
    "BaseMemory",
    # RAG
    "BaseReranker",
    "BaseRetriever",
    # Skills
    "BaseSkill",
    "BaseVectorStore",
    "CapabilityException",
    "CapabilityProvider",
    # Capabilities (agent.requires(...))
    "CapabilityRegistry",
    "ChatPromptTemplate",
    "ChatResponse",
    # RAG
    "Chunk",
    "ConfigurationException",
    # Multi-agent workflows: terminating edge sentinel for the "graph" strategy
    "END",
    "EmbeddingRegistry",
    # Multi-agent workflows: the "reflexion" strategy's pluggable evaluator contract
    "EvaluationResult",
    "Evaluator",
    "HybridRetriever",
    "InProcessMemory",
    "LLMReranker",
    "MCPClient",
    "MCPClientRegistry",
    "MCPException",
    "MCPServer",
    "MemoryException",
    "MemoryRegistry",
    # Messages / responses
    "Message",
    "MessageCountPolicy",
    "PromptException",
    # Prompt templates
    "PromptTemplate",
    "PromptTemplateRegistry",
    "ProviderException",
    # Providers
    "ProviderRegistry",
    "RateLimitException",
    # Rate limiting
    "RateLimiter",
    "Retriever",
    "Role",
    "SQLiteMemory",
    "ScoredChunk",
    "Settings",
    "SkillException",
    "SkillRegistry",
    "SummarizingPolicy",
    "Tool",
    "ToolCall",
    "ToolException",
    "ToolRegistry",
    "VectorMemory",
    "VectorStoreException",
    "VectorStoreRegistry",
    # Multi-agent workflows
    "Workflow",
    "WorkflowResult",
    "chunk_text",
    "default_capability_registry",
    "default_embedding_registry",
    "default_mcp_registry",
    "default_memory_registry",
    "default_prompt_registry",
    "default_registry",
    "default_vector_store_registry",
    # Tool calling
    "tool",
]

__version__ = "0.36.0"
