# Requisite

[![CI](https://github.com/requisite-ai/requisite-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/requisite-ai/requisite-ai/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/requisite-ai/requisite-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/requisite-ai/requisite-ai)
[![PyPI](https://img.shields.io/pypi/v/requisite-ai.svg)](https://pypi.org/project/requisite-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Declare what your AI application needs — not which SDK provides it.**

A provider-agnostic, plugin-based Python framework for building AI
applications and agents. Swap the LLM provider, the multi-agent execution
engine, or the implementation behind a capability like `"weather"` or
`"internet_search"` — all via configuration, never a rewrite.

```python
from requisite import AI

ai = AI()  # provider="openai" by default
ai = AI(provider="anthropic", model="claude-sonnet-4-6")  # same API, different provider
ai = AI(provider="gemini", model="gemini-2.5-flash")
ai = AI(provider="groq", model="llama-3.3-70b-versatile")
```

```python
from requisite import Agent

agent = Agent(name="Assistant", provider="openai")
agent.requires("weather", "internet_search", "filesystem")  # not use_tool(specific_impl)
agent.run("What's the weather in Tokyo?")
```

## Install

Requisite is [published on PyPI](https://pypi.org/project/requisite-ai/):

```bash
pip install requisite-ai                  # core only -- no provider SDKs
pip install "requisite-ai[all]"           # every provider + langgraph
pip install "requisite-ai[openai]"        # OpenAI only
pip install "requisite-ai[anthropic]"     # Anthropic (Claude) only
pip install "requisite-ai[gemini]"        # Gemini only
pip install "requisite-ai[groq]"          # Groq only (uses the openai package -- wire-compatible)
pip install "requisite-ai[azure_openai]"  # Azure OpenAI only (uses the openai package)
pip install "requisite-ai[mcp]"           # MCP client + server integration
pip install "requisite-ai[rag]"           # RAG (embedding providers; in-memory vector store needs nothing extra)
pip install "requisite-ai[langgraph]"     # native + langgraph orchestration
pip install "requisite-ai[crewai]"        # native + CrewAI orchestration (sequential only)
pip install "requisite-ai[autogen]"       # native + AutoGen orchestration (sequential + supervisor)
```

> Quoting the package name (`"requisite-ai[all]"`) avoids shell globbing
> issues with `[...]` in zsh; plain `pip install requisite-ai[all]` also
> works in bash.

**Contributing to Requisite itself** (not just using it)? See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for an editable install from source
(`pip install -e ".[dev,all]"`) and running the test suite/linters.

> **Note on the Gemini SDK:** this framework uses the current, unified
> `google-genai` package (`from google import genai`). Do not install the
> deprecated `google-generativeai` package — the two conflict.
>
> **Note on Azure OpenAI:** uses Azure's current **v1 GA API** (the plain
> `openai` client pointed at your endpoint) — no separate SDK, no dated
> `api-version` string. See [ADR-0002](docs/adr/0002-provider-kwargs-and-memory-integration.md).
>
> **Note on `crewai` + MCP:** `crewai` (verified at 1.15.17) hard-pins
> `mcp~=1.28.1`, which conflicts with this project's own `mcp>=2.0,<3.0`
> ([ADR-0025](docs/adr/0025-mcp-2x-migration.md)). Installing both in
> the same environment can let pip's resolver silently downgrade `mcp`
> below 2.0, breaking any MCP client/server code with errors like
> `'Tool' object has no attribute 'input_schema'`. That's why `crewai`
> is deliberately **not** included in `requisite-ai[all]` — if you need
> both the `crewai` orchestrator backend and MCP in the same project,
> install `requisite-ai[crewai]` first, then re-assert
> `pip install "mcp>=2.0,<3.0"` afterward (and again any time you
> reinstall/upgrade `crewai`), or keep them in separate environments.
> See [ADR-0027](docs/adr/0027-crewai-autogen-orchestrator-backends.md).

## Configuration

Copy `.env.example` to `.env` and fill in the key(s) for the provider(s) you use:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
DEFAULT_PROVIDER=openai
MODEL=gpt-4o-mini
TEMPERATURE=0.2
RATE_LIMIT_RPM=          # optional -- see "Rate limiting" below
```

`Settings` (`requisite/config/settings.py`) reads this automatically — you
never call `os.environ.get` yourself. `.env.example` also reserves
placeholders for planned integrations (GitHub, Hugging Face, AWS, Azure
general-purpose credentials, Pinecone, Weaviate) — see `ROADMAP.md`.

## CLI

Get from `pip install` to a running agent without writing any Python:

```bash
requisite init my-app --provider gemini
cd my-app
pip install -r requirements.txt
cp .env.example .env   # then fill in your Gemini API key
python main.py
```

`requisite init` scaffolds a runnable project, including an `agents.py`
with an example `Agent` registered on a module-level `agent_registry` —
the convention the rest of the CLI looks for:

```bash
requisite providers        # every registered provider -- SDK installed? API key set?
requisite capabilities     # every registered capability and its competing providers
requisite agents           # agents registered in this project's agents.py
requisite plugins          # installed packages registered under the "requisite.plugins" entry-point group
requisite chat             # interactive chat REPL (bare AI, or --agent NAME for a project agent)
requisite chat "explain LangGraph in one sentence"   # one-shot
```

Also runnable as `python -m requisite`. See
[ADR-0014](docs/adr/0014-cli.md) for why "list registered agents" needed
a project convention rather than a new global registry, and why the CLI
prints directly to stdout instead of going through `logging` like the
rest of the framework.

## Usage

### Chat

```python
from requisite import AI

ai = AI()
print(ai.chat("Explain LangGraph in one sentence."))
```

### Supported providers

| Provider | `provider=` | Model examples | Notes |
|---|---|---|---|
| OpenAI | `"openai"` | `gpt-4o-mini`, `gpt-4o` | |
| Anthropic | `"anthropic"` | `claude-sonnet-4-6`, `claude-opus-4-8` | Native structured output via `messages.parse` |
| Gemini | `"gemini"` | `gemini-2.5-flash`, `gemini-2.5-pro` | Uses the unified `google-genai` SDK |
| Groq | `"groq"` | `llama-3.3-70b-versatile`, `openai/gpt-oss-20b` | OpenAI-wire-compatible; uses the `openai` package |
| Azure OpenAI | `"azure_openai"` | your deployment name | Requires `azure_endpoint` (or `AZURE_OPENAI_ENDPOINT`); current v1 GA API, no `api-version` needed |
| OpenRouter | `"openrouter"` | `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-6` | OpenAI-wire-compatible; routes to many underlying providers |
| Together AI | `"together"` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | OpenAI-wire-compatible; hosts open-source models |
| Ollama | `"ollama"` | `llama3.2`, `qwen3` | Local (or remote) models; uses the native `ollama` client, not an OpenAI-compat shim |

Switching between any of these is the `provider=`/`AZURE_OPENAI_ENDPOINT`
change shown above — no other code changes. See
[ADR-0002](docs/adr/0002-provider-kwargs-and-memory-integration.md) for
why Groq, Azure OpenAI, OpenRouter, and Together AI are implemented as
thin `OpenAIProvider` subclasses rather than separate SDKs.

### Structured output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = ai.chat("Extract: John is 30 years old.", response_model=Person)
print(person.name, person.age)  # John 30
```

### Tool calling

```python
from requisite.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22C in {city}"

response = ai.chat_response("What's the weather in Paris?", tools=[get_weather])
if response.has_tool_calls:
    call = response.tool_calls[0]
    result = get_weather.tool.execute(**call.arguments)
```

That's the low-level view. For most applications, let an `Agent` run the
full tool-calling loop for you (see below).

### Agents

```python
from requisite import Agent
from requisite.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22C in {city}"

agent = Agent(name="Weather Agent", provider="openai", tools=[get_weather])
result = agent.run("What's the weather in Paris?")
print(result.content)              # "It's sunny and 22C in Paris."
print(result.tool_calls_executed)  # ["get_weather"]
```

`Agent` automatically: offers its tools/skills to the model, executes any
tool calls the model requests, feeds results back, and repeats (up to
`max_iterations`) until it has a final answer.

### Multi-agent workflows

```python
from requisite import Agent, Workflow

research = Agent(name="Researcher", provider="openai")
writer = Agent(name="Writer", provider="openai")

workflow = Workflow()
workflow.add(research)
workflow.add(writer)

result = workflow.run("Research AI trends and write a summary.")
print(result.content)
```

Run agents in parallel against the same input instead of as a pipeline:

```python
workflow.parallel()
result = workflow.run("What is retrieval-augmented generation?")
```

Switch the execution engine — the `.add()` / `.run()` API never changes:

```python
workflow.use_langgraph()   # requires: pip install langgraph
result = workflow.run("Research AI trends and write a summary.")

workflow.use_crewai()      # requires: pip install crewai -- "sequential" only
workflow.use_autogen()     # requires: pip install autogen-agentchat autogen-core -- "sequential" + "supervisor"

workflow.use_native()      # back to the built-in, dependency-free engine
```

`langgraph`/`crewai`/`autogen` are coordination-only backends — every
actual model call still goes through each agent's own configured
provider (rate limiting, tools, everything), never the third-party
package's own LLM client. See
[ADR-0027](docs/adr/0027-crewai-autogen-orchestrator-backends.md) for
which strategy each backend supports and why.

Let a supervisor agent delegate to a team of workers, addressed by name,
deciding when the task is done:

```python
supervisor = Agent(name="Supervisor", provider="openai")
researcher = Agent(name="Researcher", provider="openai")
writer = Agent(name="Writer", provider="openai")

workflow = Workflow().supervisor()
workflow.add(supervisor).add(researcher).add(writer)
result = workflow.run("Research AI trends and write a summary.")
```

The first agent added is the coordinator; every agent added after it is a
worker. `.planner()` works the same way, but the first agent decomposes
the task into an ordered plan up front instead of delegating round by
round. `.reflection()` takes a single agent that critiques and revises
its own output over several rounds.

`supervisor` also runs on the `langgraph` backend (`workflow.use_langgraph()`)
as a real conditional graph — `add_conditional_edges` routing to
whichever worker the coordinator delegates to, looping back for another
round — not a disguised Python loop. Same `.add()`/`.run()` call site
either way; see [ADR-0016](docs/adr/0016-langgraph-branching.md).

```python
workflow = Workflow().reflection()
workflow.add(writer)
result = workflow.run("Write a tagline for an AI framework.", max_rounds=3)
```

`.tree_of_thoughts()` branches and prunes a search tree of candidate
reasoning steps instead of following a single path — the first agent
evaluates, the rest generate candidates:

```python
evaluator = Agent(name="Evaluator", provider="openai")
thinker = Agent(name="Thinker", provider="openai")

workflow = Workflow().tree_of_thoughts()
workflow.add(evaluator).add(thinker)
result = workflow.run(
    "A train travels 60 miles in the first hour and 90 in the second. "
    "What is its average speed?",
    breadth=3, beam_width=2, max_depth=3,
)
```

Each level, `breadth` candidates are generated and scored together, then
pruned to the top `beam_width` before continuing — see
[ADR-0018](docs/adr/0018-tree-of-thoughts-strategy.md).

### Rate limiting

Free-tier and other quota-limited API keys often cap requests per
minute — set `RATE_LIMIT_RPM` (and, optionally, `RATE_LIMIT_MAX_WAIT_SECONDS`)
in `.env` and `AI`/`Agent` wait for capacity instead of letting the
provider reject the call:

```env
RATE_LIMIT_RPM=15
```

That covers a single `Agent`/`AI`. **Several agents that call the same
underlying API key share the same real quota**, so build one
`RateLimiter` and pass it to each of them explicitly — this is what a
multi-agent `Workflow` (research + writer + planner + supervisor, say)
needs to avoid exceeding the quota collectively even if each individual
agent looks fine on its own:

```python
from requisite import Agent, RateLimiter

shared_limit = RateLimiter(requests_per_minute=15)
research = Agent(name="Researcher", provider="gemini", rate_limiter=shared_limit)
writer = Agent(name="Writer", provider="gemini", rate_limiter=shared_limit)
```

### Skills

A skill is a reusable, higher-level capability (vs. a tool, which is
typically a single function). Skills expose themselves to the model as
tools automatically:

```python
from requisite.skills import BaseSkill

class ReadFileSkill(BaseSkill):
    def __init__(self):
        super().__init__(name="read_file", description="Read a text file's contents.")

    def run(self, path: str) -> str:
        with open(path) as f:
            return f.read()

agent = Agent(name="File Agent", provider="openai", skills=[ReadFileSkill()])
```

### Capabilities: declare *what*, not *which*

`agent.use_tool(specific_impl)` binds an agent to one concrete
implementation at write-time. `agent.requires("weather")` binds it to a
*name*, resolved at runtime against whichever implementation is
currently available — a native tool, an MCP server, a cloud API, or a
third-party plugin:

```python
from requisite import Agent

agent = Agent(name="Assistant", provider="openai")
agent.requires("weather", "internet_search", "filesystem")

result = agent.run("What's the weather in Tokyo?")
```

Three capabilities ship out of the box, backed by free/keyless APIs and
the local filesystem (see `requisite/capabilities/resolvers.py`):
`"filesystem"`, `"weather"`, `"internet_search"`.

Register a better provider for the same capability at a higher priority
and it takes over automatically — application code never changes:

```python
from requisite.capabilities import default_registry

default_registry.register(
    "weather",
    my_paid_weather_tool,
    provider_name="acme-weather",
    priority=10,
    is_available=lambda: bool(os.environ.get("ACME_API_KEY")),
)
# agent.requires("weather") now resolves to "acme-weather" when the key
# is set, and quietly falls back to the built-in provider otherwise.
```

This is the same interface + registry pattern used for providers and
orchestrators, one layer up: `CapabilityRegistry.resolve(name)` picks the
highest-priority provider whose `is_available()` currently returns
`True`, raising `CapabilityException` if none are.

### Plugins

Third-party packages register with any of the registries above the same
way first-party code does — no special `Plugin` class. Discover every
installed one (rather than manually importing each) by declaring it
under the `"requisite.plugins"` entry-point group and calling:

```python
from requisite.plugins import discover

result = discover()
print(result.loaded)   # names of plugins that registered successfully
print(result.failed)   # {name: error} for any that didn't -- one broken
                        # plugin never blocks the rest
```

Never automatic — nothing runs from a package you didn't ask to
discover. Also available as `requisite plugins` on the CLI. See
[`CONTRIBUTING.md`](CONTRIBUTING.md#writing-a-plugin) for how to write
one, [ADR-0017](docs/adr/0017-entry-point-plugin-discovery.md) for the
design, and [`PLUGINS.md`](PLUGINS.md) for the directory of published
third-party plugins.

### MCP integration

Connect to any MCP server (local via stdio, or remote via Streamable
HTTP) and its tools become capabilities like any other:

```python
from requisite import Agent
from requisite.mcp import MCPClient
from requisite.capabilities import default_registry as capabilities

# Local, subprocess-based MCP server
filesystem = MCPClient.stdio(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"],
)
filesystem.register_as_capability(capabilities, capability="read_file")

# Remote MCP server
github = MCPClient.http(name="github", url="https://api.example.com/mcp", headers={"Authorization": "Bearer ..."})
github.register_as_capability(capabilities, capability="github")

agent = Agent(name="Assistant", provider="openai")
agent.requires("read_file", "github")  # agent can't tell these apart from native tools
```

You can also use an MCP server's tools directly, without going through
capabilities:

```python
tools = filesystem.discover_tools()
agent = Agent(name="Assistant", provider="openai", tools=tools)
```

Both transports were verified against real MCP servers during
development. By default, each tool call re-connects, calls, and
disconnects rather than holding a persistent session — see
[ADR-0004](docs/adr/0004-mcp-integration.md) for why. For repeated calls
to the same server in a tight loop, an opt-in persistent-session mode
avoids that reconnect cost — measured live: ~1000x faster for stdio,
~15x for HTTP:

```python
async with MCPClient.stdio(name="filesystem", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as filesystem:
    tools = await filesystem.adiscover_tools()
    result = await tools[0].aexecute(path="/tmp")
    # ... as many more calls as you like, all reusing the same session ...
```

Persistent mode is async-only (`aconnect`/`aclose`/`async with`, plus the
existing `a`-prefixed methods) — the sync methods raise immediately if
called while connected, rather than risk a hang crossing an
`asyncio.run()` boundary. See
[ADR-0030](docs/adr/0030-mcp-persistent-session-mode.md) for the full
design and the real deadlock risk it was built to avoid.

### Expose Requisite as an MCP server

The reverse direction: turn Requisite tools and agents *into* an MCP
server, so any MCP client (Claude Desktop, Claude Code, or Requisite's
own `MCPClient`) can use them:

```python
from requisite import Agent, MCPServer
from requisite.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22C in {city}"

researcher = Agent(name="researcher", provider="openai")

server = MCPServer(name="my-tools", tools=[get_weather], agents=[researcher])
server.run_stdio()  # or: server.run_http(host="127.0.0.1", port=8000)
```

An agent exposed this way (`agents=[...]`, or `server.add_agent(...)`)
becomes a single MCP tool taking a `prompt` argument and returning its
final answer — `Agent.as_tool()` under the hood, reusable outside MCP
too. Both transports were verified end to end against Requisite's own
`MCPClient` (real subprocess, real HTTP port, including a real
provider call executing inside the server process) — see
[ADR-0015](docs/adr/0015-mcp-server-integration.md).

### RAG (Retrieval-Augmented Generation)

A retriever is a capability too — `agent.requires("knowledge_base")`
works exactly like `"weather"` or an MCP-backed tool:

```python
from requisite import Agent
from requisite.rag import Retriever
from requisite.rag.embeddings import OpenAIEmbeddingProvider
from requisite.rag.vectorstores import InMemoryVectorStore
from requisite.capabilities import default_registry as capabilities

retriever = Retriever(
    embedding_provider=OpenAIEmbeddingProvider(api_key="sk-..."),
    vector_store=InMemoryVectorStore(),  # zero-dependency default
)
retriever.add_texts([
    "Paris is the capital of France.",
    "The Eiffel Tower was completed in 1889.",
])

capabilities.register("knowledge_base", retriever.as_tool())

agent = Agent(name="Assistant", provider="openai")
agent.requires("knowledge_base")
print(agent.run("What's the capital of France?").content)
```

`add_texts` chunks each document (character-based, with overlap) before
embedding and storing it:

```python
retriever.add_texts(long_document_text, chunk_size=1000, chunk_overlap=200)
```

`InMemoryVectorStore` is the zero-dependency default — pure-Python cosine
similarity, fine for a few thousand chunks. Swap in a real vector
database the same way you'd swap a provider — construct it and pass it
to `Retriever(vector_store=...)`:

```python
from requisite.rag.vectorstores.pinecone import PineconeVectorStore
# pip install requisite-ai[pinecone]
vector_store = PineconeVectorStore(api_key="...", index_name="my-index", dimension=1536)

from requisite.rag.vectorstores.weaviate import WeaviateVectorStore
# pip install requisite-ai[weaviate]
vector_store = WeaviateVectorStore(url="https://my-cluster.weaviate.network", api_key="...")
```

See [ADR-0005](docs/adr/0005-rag-integration.md) for the interface
decomposition (embeddings / vector stores / retrievers are three
independent extension points) and why the in-memory default was chosen
over requiring a real vector DB from day one.

### Memory: conversation history across separate calls

```python
from requisite import Agent
from requisite.memory import InProcessMemory

memory = InProcessMemory()
agent = Agent(name="Assistant", provider="openai", memory=memory, session_id="user-42")

agent.run("My name is Alex.")
result = agent.run("What's my name?")  # remembers "Alex" via `memory`
print(result.content)
```

`session_id` is required whenever `memory` is set — there's no implicit
"current user," so an agent with memory but no session_id fails at
construction time rather than silently sharing one conversation across
callers. When memory is configured, `run()`/`arun()` must be called with
a plain string (the new turn); prior history is loaded from `memory`
automatically. Only the user's turn and the agent's final answer are
persisted, not intermediate tool-call round-trips — see
[ADR-0002](docs/adr/0002-provider-kwargs-and-memory-integration.md) for
the reasoning.

`InProcessMemory` (dict-backed, lost on restart) is the zero-dependency
default. Implement `requisite.memory.base.BaseMemory` for a persistent
backend (Redis, SQLite, ...) — see `ROADMAP.md`.

### Conversation policies: keeping long histories bounded

A conversation that grows unbounded eventually blows past the model's
context window (or just gets expensive). `conversation_policy=` trims or
summarizes history once, before each `run()`/`arun()` call — independent
of whether `memory` is configured:

```python
from requisite import Agent
from requisite.memory import InProcessMemory, MessageCountPolicy

agent = Agent(
    name="Assistant",
    provider="openai",
    memory=InProcessMemory(),
    session_id="user-42",
    conversation_policy=MessageCountPolicy(max_messages=20),  # keep the most recent 20
)
```

For long-running conversations where you'd rather compress old context
than drop it, `SummarizingPolicy` collapses older messages into one
LLM-generated summary, keeping the most recent few verbatim:

```python
from requisite import AI
from requisite.memory import SummarizingPolicy

# A separate, cheaper AI instance for summarization is a reasonable choice --
# summarization quality requirements are usually lower than the agent's own task.
summarizer = AI(provider="groq", model="llama-3.3-70b-versatile")

agent = Agent(
    name="Assistant",
    provider="openai",
    memory=InProcessMemory(),
    session_id="user-42",
    conversation_policy=SummarizingPolicy(summarizer, max_messages=20, keep_recent=6),
)
```

See [ADR-0003](docs/adr/0003-prompt-templates-structured-logging-conversation-policy.md)
for why the policy is applied once per call rather than mid-tool-loop,
and why it doesn't change what gets persisted to `memory`.

### Prompt templates

```python
from requisite.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {persona}."),
    ("user", "{question}"),
])

messages = chat_template.format_messages(persona="pirate", question="Where's the treasure?")
print(ai.chat(messages))
```

`ChatPromptTemplate.format_messages(...)` returns a plain `list[Message]`
— pass it to `ai.chat(...)` or `agent.run(...)` exactly like any other
message sequence. For a single string instead of a full conversation,
use `PromptTemplate`:

```python
from requisite.prompts import PromptTemplate

translate = PromptTemplate.from_template("Translate to {language}: {text}")
print(ai.chat(translate.format(language="French", text="Good morning")))

# Pre-fill some variables now, leave the rest for later:
french_translator = translate.partial(language="French")
print(ai.chat(french_translator.format(text="Good night")))
```

Register named templates for reuse across an application with
`PromptTemplateRegistry`.

### Structured logging

Every framework module logs through the standard library
(`logging.getLogger("requisite.<subpackage>")`). Opt into JSON output
with one call, wherever your application configures logging — this is
never done automatically by the framework:

```python
from requisite.telemetry import configure_logging

configure_logging(level="INFO", json_format=True)
```

```json
{"timestamp": "...", "level": "DEBUG", "logger": "requisite.capabilities", "message": "Resolved capability 'weather' -> 'open-meteo'", "capability": "weather", "provider_name": "open-meteo"}
```

Any `extra={...}` fields passed to a log call are merged into the JSON
payload automatically — the same log call produces a readable line with
the default formatter and a structured payload with `json_format=True`.

### Streaming & async

```python
for token in ai.stream("Write a haiku about distributed systems."):
    print(token, end="")

text = await ai.achat("Hello!")
async for token in ai.astream("Hello, streamed!"):
    print(token, end="")

result = await agent.arun("What's the weather in Paris?")
result = await workflow.arun("Research AI trends.")
```

### Conversation history

```python
from requisite import Message

history = [
    Message.user("My name is Alex."),
    Message.assistant("Nice to meet you, Alex!"),
    Message.user("What's my name?"),
]
print(ai.chat(history))
```

See `examples/` for complete, runnable scripts covering each of the above.

## Architecture

```
requisite/
├── core/           # Provider-agnostic data models (Message, ChatResponse,
│                   # ToolCall, ...) and the AIException hierarchy
├── config/         # Settings (pydantic-settings, reads .env)
├── providers/      # BaseProvider interface + OpenAI, Anthropic, Gemini, Groq,
│                   # Azure OpenAI, OpenRouter, Together AI, Ollama
│                   # + ProviderRegistry (extensible, DI-friendly)
├── tools/          # Tool, @tool decorator, ToolRegistry, JSON Schema derivation
├── skills/         # BaseSkill, SkillRegistry -- reusable higher-level capabilities
├── capabilities/   # CapabilityRegistry -- resolve a named capability (e.g.
│                   # "weather") to whichever implementation is available
├── mcp/            # BaseMCPClient + MCPClient (stdio + Streamable HTTP)
│                   # + MCPClientRegistry -- bridges MCP tools into capabilities
│                   # + MCPServer -- reverse direction, expose Requisite as a server
├── rag/            # BaseEmbeddingProvider, BaseVectorStore, BaseRetriever
│                   # + Retriever (dense) + InMemory/Pinecone/Weaviate vector stores
├── memory/         # BaseMemory + InProcessMemory + MemoryRegistry, plus
│                   # BaseConversationPolicy (MessageCountPolicy, SummarizingPolicy)
├── prompts/        # PromptTemplate, ChatPromptTemplate, PromptTemplateRegistry
├── telemetry/      # Structured (JSON) logging -- opt-in, never automatic
├── agents/         # Agent (tool-calling loop, .requires(), optional memory) + AgentRegistry
├── orchestrators/  # BaseOrchestrator interface + native (sequential, parallel,
│                   # reflection, planner, supervisor) and langgraph backends
│                   # + OrchestratorRegistry
├── workflows/      # Workflow -- the small, ergonomic multi-agent facade
├── cli/            # requisite init/providers/capabilities/agents/plugins/chat --
│                   # see the CLI section above and ADR-0014
├── plugins.py      # discover() -- entry-point plugin discovery, see ADR-0017
└── ai.py           # The `AI` facade -- the class most users touch directly
```

Every layer follows the same pattern: a small abstract interface
(`BaseProvider`, `BaseOrchestrator`, ...), one or more concrete
implementations, and a plain, instantiable registry (not a singleton)
mapping names to constructors. That's what makes each of the following a
*configuration* change rather than a *code* change:

- `AI(provider="openai")` → `AI(provider="gemini")`
- `Workflow(orchestrator="native")` → `workflow.use_langgraph()`
- `agent.use_tool(specific_impl)` → `agent.requires("weather")`

See **[`ARCHITECTURE.md`](ARCHITECTURE.md)** for the full dependency
diagram, request-flow walkthroughs (a chat call, an agent's tool-calling
loop, capability resolution, a multi-agent workflow), and the design
decisions behind them. See **[`CONTRIBUTING.md`](CONTRIBUTING.md)** for
the step-by-step to add a new provider, orchestrator backend, or
capability resolver.

## Error handling

All framework exceptions inherit from `AIException`:

```
AIException
├── ConfigurationException   # missing/invalid config, unknown provider/orchestrator name
├── ProviderException        # provider SDK call failed (wraps the original error)
├── ToolException            # a tool wasn't found, or raised while executing
├── SkillException           # a skill wasn't found, or raised while executing
├── AgentException           # agent execution failed (e.g. max_iterations exceeded)
├── CapabilityException      # a required capability has no available provider
├── PromptException          # a prompt template was rendered without a required variable
└── MCPException              # an MCP server call/discovery failed, or a capability bridge found no matching tool
```

Provider SDK errors are never swallowed — they're wrapped with `provider`
and `original_error` attached, and re-raised via `raise ... from original_error`
so the original traceback is preserved.

## Testing

```bash
pytest
```

Tests never hit the network: the OpenAI and Gemini SDKs are faked via
`sys.modules` injection, and the `AI` / `Agent` / `Workflow` facades are
tested against fully in-memory fake providers.

## Roadmap

Implemented: 8 providers (OpenAI, Anthropic, Gemini, Groq, Azure OpenAI,
OpenRouter, Together AI, Ollama), structured outputs, tool calling,
skills, capability resolution (`agent.requires(...)`), MCP client and
server integration (stdio + Streamable HTTP, both directions, plus an
opt-in persistent-session client mode), RAG (embeddings, in-memory /
Pinecone / Weaviate vector stores, dense/BM25/hybrid retrieval,
re-ranking, context compression), memory + conversation policies
(`Agent(memory=..., conversation_policy=...)`), prompt templates,
structured logging plus OpenTelemetry tracing/metrics, agents +
registry, multi-agent workflows (sequential, parallel, reflection,
planner, supervisor, critic, consensus, debate, map-reduce,
hierarchical, tree-of-thoughts, and graph on the native backend;
sequential, supervisor, hierarchical, reflection, graph, parallel,
consensus, and map-reduce on langgraph; sequential on CrewAI and AutoGen,
supervisor also on AutoGen), entry-point plugin discovery, an official
plugin directory (`PLUGINS.md`).

See [`ROADMAP.md`](ROADMAP.md) for the full, per-layer status table
(providers, orchestration strategies, MCP, memory, RAG, ...) and what's
explicitly out of scope. See [`FEATURES.md`](FEATURES.md) for the same
information organized as a line-by-line checklist against the original
project vision.

## Contributing

Contributions are welcome:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup, running checks, and
  step-by-step guides for adding a provider, orchestrator backend, or
  capability resolver.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the framework fits together
  and why: the interface + registry pattern, request-flow walkthroughs,
  design decisions.
- [`docs/adr/`](docs/adr/) — Architecture Decision Records: the formal
  rationale behind core interfaces, extension points, plugin discovery,
  configuration model, and the `requisite-core` vs. optional-integrations
  boundary. Start with [ADR-0001](docs/adr/0001-core-architecture-and-interfaces.md).
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — coding standards, testing
  philosophy, docstring format, logging/error-handling conventions,
  versioning policy.
- [`ROADMAP.md`](ROADMAP.md) — what's shipped, planned, or out of scope.
- [`PLUGINS.md`](PLUGINS.md) — the directory of published third-party
  plugins, and how to get yours listed.

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md). Security
issues should go through [`SECURITY.md`](SECURITY.md), not a public issue.

See [`CHANGELOG.md`](CHANGELOG.md) for release history.

## License

MIT — see [`LICENSE`](LICENSE).
