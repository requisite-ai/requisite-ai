# Architecture

This document explains *how* Requisite is built and *why*, so a new
contributor can extend it without first reverse-engineering the pattern
from source. It complements `ROADMAP.md` (what's planned) and
`CONTRIBUTING.md` (how to submit a change).

## The one idea everything else follows from

> Every major capability is: a small abstract **interface**, one or more
> concrete **implementations**, and a plain, instantiable **registry**
> mapping names to constructors.

Providers, orchestrators, tools, and capabilities all follow this shape:

```
BaseProvider          BaseOrchestrator        Tool (via ToolRegistry)      CapabilityProvider (via CapabilityRegistry)
   ├── OpenAIProvider     ├── NativeOrchestrator                              ├── read_file (filesystem)
   ├── GeminiProvider     ├── LangGraphOrchestrator                           ├── get_weather (weather)
   └── (yours)            └── (yours)                                        └── (yours)

ProviderRegistry       OrchestratorRegistry
   .register(name, ctor)  .register(name, ctor)
   .create(name)          .create(name)
```

Concretely, this is why each of the following is a *configuration* change,
not a *code* change:

```python
AI(provider="openai")   →  AI(provider="gemini")
workflow.use_native()    →  workflow.use_langgraph()
agent.use_tool(specific_impl)  →  agent.requires("weather")  # resolved at runtime
```

### Why a registry and not a singleton

A classic singleton (`Provider.get_instance()`) makes every part of an
application share one global, mutable object — which makes testing (you
can't isolate one test's providers from another's) and multi-tenant use
(different configs per request) needlessly hard.

Instead, every registry (`ProviderRegistry`, `OrchestratorRegistry`,
`ToolRegistry`, `CapabilityRegistry`, `AgentRegistry`) is a **plain class**.
The framework ships one pre-populated `default_registry` per layer purely
for convenience — most applications just use that one — but nothing stops
you from constructing your own `ProviderRegistry()` instance (tests do
exactly this, to guarantee isolation between test cases; see
`tests/test_ai.py`'s `registry_with_fake` fixture).

### Why lazy imports for SDKs

`requisite.providers` imports cleanly even if neither `openai` nor
`google-genai` is installed. Each provider module defers its SDK import to
inside the method that needs it (`OpenAIProvider._build_client`,
`GeminiProvider._get_client`), raising a clear `ConfigurationException`
with an install hint if the SDK is missing. This means:

- Installing `requisite-ai` doesn't force every optional dependency onto
  every user.
- A broken/incompatible optional SDK (e.g. a `langgraph` version bump)
  can't break imports for people who don't use that backend.

Apply the same pattern to any new provider or orchestrator backend you add.

## Layers, top to bottom

```
requisite/
├── core/           # Message, ChatResponse, ToolCall, Usage, Role, StreamChunk
│                   # + the AIException hierarchy. Pure data + errors, no I/O.
├── config/         # Settings: pydantic-settings, reads .env, masks secrets
├── providers/      # BaseProvider + OpenAI/Gemini + ProviderRegistry
├── tools/          # Tool, @tool, ToolRegistry, JSON Schema derivation
├── skills/         # BaseSkill, SkillRegistry (reusable, higher-level capabilities)
├── capabilities/   # CapabilityRegistry -- name -> best available Tool
├── agents/         # Agent (owns an AI + a ToolRegistry + the tool-calling loop)
├── orchestrators/  # BaseOrchestrator + native/langgraph + OrchestratorRegistry
├── workflows/      # Workflow -- the ergonomic facade over orchestrators
└── ai.py           # AI -- the facade most application code touches directly
```

Dependencies point strictly downward: `workflows` depends on
`orchestrators` and `agents`; `agents` depends on `ai`, `tools`, `skills`,
`capabilities`; `ai` depends on `providers`; `providers` and `tools` depend
on `core`. Nothing in `core` imports from any other layer. This is what
keeps, e.g., adding a capability resolver from ever requiring a change to
`ai.py`.

## Request flows

### A plain chat call

```
AI.chat(prompt)
  → AI.chat_response(prompt)       # builds Message list, resolves temperature
    → BaseProvider.chat(messages)  # OpenAIProvider / GeminiProvider
      → provider SDK call
      → wraps SDK errors as ProviderException
      → returns ChatResponse (content, usage, tool_calls, parsed, raw)
  → returns .content (or .parsed, if response_model was given)
```

`chat_response` (not `chat`) is the method to reach for when you need
`usage`, `tool_calls`, or the raw provider payload — `chat` is a thin
convenience that returns just the text (or the parsed model).

### An agent's tool-calling loop

```
Agent.run(prompt)
  messages = [Message.user(prompt)]
  loop (up to max_iterations):
    response = AI.chat_response(messages, tools=agent's ToolRegistry.all())
    if not response.has_tool_calls:
        return AgentResult(content=response.content, ...)
    messages.append(Message.assistant_tool_calls(response.tool_calls))
    for each ToolCall:
        result = Tool.execute(**call.arguments)
        messages.append(Message.tool_result(result, tool_call_id=call.id))
    # loop again with the tool results appended
  raise AgentException("max_iterations exceeded")
```

The `Message.assistant_tool_calls` / `Message.tool_result` constructors
exist specifically so this loop's message history round-trips correctly
through *either* provider's wire format — `OpenAIProvider` encodes them as
OpenAI's `tool_calls` / `role: "tool"` messages; `GeminiProvider` encodes
the same `Message` objects as Gemini's `function_call` / `function_response`
parts. Application code (and `Agent`) never needs to know which.

### Capability resolution

```
Agent.requires("weather")
  → CapabilityRegistry.resolve("weather")
      providers = [ (priority=10, "acme-weather", is_available=...),
                    (priority=0,  "open-meteo",   is_available=lambda: True) ]
      → sorted by priority, descending
      → first provider whose is_available() is True wins
      → CapabilityException if none are available
  → Tool renamed to "weather" (the capability name, not the impl's function
    name) and registered into the agent's ToolRegistry
```

Renaming to the capability name is deliberate: it's what makes the
model-facing tool name stable even when the provider behind `"weather"`
changes. See `Agent.requires` in `agents/agent.py` for the exact line.

### A multi-agent workflow

```
Workflow().add(research).add(writer).run("...")
  → OrchestratorRegistry.create(self._orchestrator_name)   # "native" by default
  → NativeOrchestrator.run(steps=[research, writer], input="...", strategy="sequential")
      sequential: current_input = input
                  for agent in steps: result = agent.run(current_input); current_input = result.content
      parallel:   every agent runs agent.run(input) concurrently (ThreadPoolExecutor for run(),
                  asyncio.gather for arun()); outputs are concatenated, labeled by agent name
  → WorkflowResult(content=..., steps=[AgentResult, ...], orchestrator="native", strategy="sequential")
```

`workflow.use_langgraph()` swaps `self._orchestrator_name` to `"langgraph"`;
`LangGraphOrchestrator` builds an actual `langgraph.graph.StateGraph` with
one node per agent, wired linearly, and compiles/invokes it. The
`Workflow.add()` / `.run()` call site is identical either way.

## Design decisions worth knowing

- **Pydantic everywhere for data, not for business logic.** `Message`,
  `ChatResponse`, `Tool`, `WorkflowResult`, etc. are all Pydantic models —
  validation, `.model_copy()`, and predictable `repr()`/serialization come
  for free. Classes with actual behavior (`AI`, `Agent`, `Workflow`,
  every registry) are plain Python classes, not Pydantic models.
- **`Message` is frozen** (`ConfigDict(frozen=True)`). Conversation history
  is built by appending new `Message` instances, never mutating one in
  place — this avoids a whole class of bugs where a message shared across
  two code paths gets silently mutated.
- **Exceptions are never swallowed.** Every provider/tool/skill call site
  that can fail wraps the underlying exception in the framework's own
  exception type (`ProviderException`, `ToolException`, ...) with
  `raise ... from original_error`, so the original traceback is always
  reachable. See `core/exceptions.py` for the full hierarchy and when to
  use each branch.
- **Sync and async are hand-written pairs, not one derived from the
  other.** `chat`/`achat`, `run`/`arun`, `stream`/`astream` are each
  implemented directly against the provider SDK's sync/async client,
  rather than wrapping the sync path in a thread pool. This avoids
  surprising blocking behavior in async applications, at the cost of a
  small amount of duplication per provider — judged worth it.
- **Tool-calling message encoding lives entirely inside the provider.**
  `Message.assistant_tool_calls` / `Message.tool_result` are
  provider-agnostic; `_to_openai_messages` and
  `GeminiProvider._build_contents_and_system` are where that generic shape
  gets translated to each SDK's specific wire format. If you add a
  provider with a different tool-calling wire format, this is the one
  place that needs provider-specific logic.
- **Structured output (`response_model=`) and tool calling
  (`tools=`) are currently mutually exclusive per call**, mirroring a real
  constraint in both the OpenAI and Gemini APIs today. `AI.chat` accepts
  both parameters but a provider that received both should prefer
  `response_model` and may ignore `tools` for that call — check the
  provider's own docstring/implementation, this isn't unified into a
  single abstraction yet since the underlying APIs aren't unified either.

## Extension points at a glance

| To add... | Implement | Register with |
|---|---|---|
| A new LLM provider | `providers.base.BaseProvider` | `providers.factory.default_registry` |
| A new orchestration backend | `orchestrators.base.BaseOrchestrator` | `orchestrators.factory.default_registry` |
| A new capability implementation | any callable / `Tool` | `capabilities.default_registry.register(name, ...)` |
| A new multi-agent strategy | a `_run_<name>` / `_arun_<name>` pair on an orchestrator | strategy string passed to `Workflow(strategy=...)` |
| A new reusable capability (vs. a one-off tool) | `skills.base.BaseSkill` | pass to `Agent(skills=[...])` |

See `CONTRIBUTING.md` for the step-by-step for each of these.
