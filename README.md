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
ai = AI(provider="gemini", model="gemini-2.5-flash")  # same API, different provider
```

```python
from requisite import Agent

agent = Agent(name="Assistant", provider="openai")
agent.requires("weather", "internet_search", "filesystem")  # not use_tool(specific_impl)
agent.run("What's the weather in Tokyo?")
```

## Install

```bash
pip install -e .[all]       # both providers + langgraph
pip install -e .[openai]    # OpenAI only
pip install -e .[gemini]    # Gemini only
pip install -e .[langgraph] # native + langgraph orchestration
```

Or with the plain requirements file: `pip install -r requirements.txt`

> **Note on the Gemini SDK:** this framework uses the current, unified
> `google-genai` package (`from google import genai`). Do not install the
> deprecated `google-generativeai` package — the two conflict.

## Configuration

Copy `.env.example` to `.env` and fill in the key(s) for the provider(s) you use:

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
DEFAULT_PROVIDER=openai
MODEL=gpt-4o-mini
TEMPERATURE=0.2
```

`Settings` (`requisite/config/settings.py`) reads this automatically — you
never call `os.environ.get` yourself.

## Usage

### Chat

```python
from requisite import AI

ai = AI()
print(ai.chat("Explain LangGraph in one sentence."))
```

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

workflow.use_native()      # back to the built-in, dependency-free engine
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
├── providers/      # BaseProvider interface + OpenAI/Gemini implementations
│                   # + ProviderRegistry (extensible, DI-friendly factory)
├── tools/          # Tool, @tool decorator, ToolRegistry, JSON Schema derivation
├── skills/         # BaseSkill, SkillRegistry -- reusable higher-level capabilities
├── capabilities/   # CapabilityRegistry -- resolve a named capability (e.g.
│                   # "weather") to whichever implementation is available
├── agents/         # Agent (tool-calling loop, .requires()) + AgentRegistry
├── orchestrators/  # BaseOrchestrator interface + native (sequential/parallel)
│                   # and langgraph backends + OrchestratorRegistry
├── workflows/      # Workflow -- the small, ergonomic multi-agent facade
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
└── MCPException              # reserved for the upcoming MCP integration
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

Implemented: provider connectivity (OpenAI, Gemini), structured outputs,
tool calling, skills, capability resolution (`agent.requires(...)`),
agents + registry, multi-agent workflows (sequential/parallel, native +
langgraph backends).

See [`ROADMAP.md`](ROADMAP.md) for the full, per-layer status table
(providers, orchestration strategies, MCP, memory, RAG, ...) and what's
explicitly out of scope.

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

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md). Security
issues should go through [`SECURITY.md`](SECURITY.md), not a public issue.

See [`CHANGELOG.md`](CHANGELOG.md) for release history.

## License

MIT — see [`LICENSE`](LICENSE).
