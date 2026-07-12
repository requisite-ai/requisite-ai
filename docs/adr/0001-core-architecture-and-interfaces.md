# 0001. Core architecture, interfaces, and extension model

Status: Accepted
Date: 2026-07-11

## Context

Requisite's stated goal is a provider-agnostic AI application framework
where providers, orchestration backends, tools, and capabilities can all
be swapped via configuration rather than code changes, and where the
ecosystem (third-party providers, plugins, MCP servers) can grow without
forking the core. Making that true requires the *first* few interfaces to
be right — retrofitting a plugin/extension model onto a framework already
in wide use is far more expensive than defining it before there's much to
break.

This ADR is written after the first four layers (providers, tools/skills,
capabilities, agents/orchestrators) already exist in code, so it also
serves to make explicit several decisions that were made implicitly while
building them — and to specify two interfaces (`Memory`, `MCPClient`) that
don't have code yet, so their eventual implementation doesn't have to
guess at the shape.

## Decision

### Core interfaces

Every core interface follows one naming and shape convention:
`Base<Noun>` — an `abc.ABC` with a small number of abstract methods, no
constructor logic beyond storing configuration, and a companion
`<Noun>Registry` (a plain class, not a singleton) for name-based
resolution. This is a deliberate choice over the more literal `AIProvider`
/ `Agent`-as-interface naming the framework is sometimes described in —
see [Alternatives](#alternatives-considered).

| Conceptual role | Interface (code) | Registry | Status |
|---|---|---|---|
| AI provider (chat completion) | `providers.base.BaseProvider` | `ProviderRegistry` | Shipped |
| Callable function exposed to a model | `tools.base.Tool` (concrete, not abstract — see note below) | `ToolRegistry` | Shipped |
| Reusable higher-level capability | `skills.base.BaseSkill` | `SkillRegistry` | Shipped |
| Named capability -> best available `Tool` | *(no single interface — see below)* | `CapabilityRegistry` | Shipped |
| Autonomous tool-calling unit | `agents.agent.Agent` (concrete, not abstract — see note below) | `AgentRegistry` | Shipped |
| Multi-agent execution strategy | `orchestrators.base.BaseOrchestrator` | `OrchestratorRegistry` | Shipped |
| Conversation/long-term memory | `memory.base.BaseMemory` | `MemoryRegistry` | **Specified here, not yet implemented** |
| MCP server connection | `mcp.base.BaseMCPClient` | `MCPClientRegistry` | **Specified here, not yet implemented** |

**Why `Tool` and `Agent` are concrete classes, not ABCs:** unlike
providers/orchestrators/skills — where genuinely different *kinds* of
implementation exist (OpenAI vs. Gemini; native vs. langgraph) — there is
only one reasonable shape for "a callable the model can invoke" or "an AI
with tools and a run loop." Introducing `BaseTool` / `BaseAgent`
abstractions with a single concrete subclass each would be exactly the
kind of unnecessary abstraction the project's own design principles warn
against. `CapabilityRegistry` similarly has no `BaseCapability` interface
because a capability *is* a `Tool` (or resolves to one) — it's a naming
layer over `ToolRegistry`, not a new kind of thing.

#### `BaseProvider` (shipped -- `providers/base.py`)

```python
class BaseProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def chat(self, messages: Sequence[Message], *, model=None, temperature=None,
             tools=None, response_model=None, **kwargs) -> ChatResponse: ...
    @abstractmethod
    async def achat(self, ...) -> ChatResponse: ...
    @abstractmethod
    def stream(self, ...) -> Iterator[StreamChunk]: ...
    @abstractmethod
    def astream(self, ...) -> AsyncIterator[StreamChunk]: ...

    def validate_config(self) -> None: ...  # default: require an api_key
```

#### `BaseOrchestrator` (shipped -- `orchestrators/base.py`)

```python
class BaseOrchestrator(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, steps: Sequence[Agent], input: Optional[str], *,
            strategy: str = "sequential", **kwargs) -> WorkflowResult: ...
    @abstractmethod
    async def arun(self, ...) -> WorkflowResult: ...
```

#### `BaseSkill` (shipped -- `skills/base.py`)

```python
class BaseSkill(ABC):
    def __init__(self, *, name: str, description: str = "") -> None: ...

    @abstractmethod
    def run(self, **kwargs) -> Any: ...
    async def arun(self, **kwargs) -> Any: ...  # default: runs `run` in a thread
    def as_tool(self) -> Tool: ...              # default: Tool.from_function(self.run)
```

#### `BaseMemory` (specified now, not yet implemented -- planned `memory/base.py`)

Memory is scoped narrowly on purpose: it is *conversation-shaped storage*,
not retrieval or ranking (that's RAG's job — see
[Extension points](#extension-points)). An `Agent` or `Workflow` reads and
appends `Message` objects through this interface; it does not reach into
storage internals.

```python
class BaseMemory(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, session_id: str) -> list[Message]:
        """Return the stored conversation history for a session, oldest first."""

    @abstractmethod
    def append(self, session_id: str, message: Message) -> None:
        """Persist one new message to a session's history."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete a session's stored history."""

    async def aload(self, session_id: str) -> list[Message]: ...   # default: thread-wrapped
    async def aappend(self, session_id: str, message: Message) -> None: ...
    async def aclear(self, session_id: str) -> None: ...
```

Planned implementations, each an independent optional integration:
in-process (`dict`-backed, zero dependencies, ships in core as the
default), SQLite, Redis, a generic vector-store-backed variant for
similarity-scoped recall. `Agent(memory=..., session_id=...)` is the
anticipated integration point — deferred until `BaseMemory` ships, to
avoid designing `Agent`'s memory parameter against a guess.

#### `BaseMCPClient` (specified now, not yet implemented -- planned `mcp/base.py`)

The design goal (stated in `ROADMAP.md`) is that an MCP-backed tool is
*just another `CapabilityProvider`* — `agent.requires("github")` should
not need to know whether `"github"` resolves to a native tool or an MCP
server. That constrains the interface to one job: discover an MCP
server's tools and hand back `Tool` objects, so the rest of the framework
never has to special-case "MCP-ness."

```python
class BaseMCPClient(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def discover_tools(self) -> list[Tool]:
        """Connect to the MCP server and return its exposed tools, translated
        to Requisite's Tool shape (each Tool.execute proxies the call over MCP)."""

    async def adiscover_tools(self) -> list[Tool]: ...

    def register_as_capability(
        self, registry: CapabilityRegistry, *, capability: str, priority: int = 0
    ) -> None:
        """Convenience: discover this server's tools and register the ones
        matching `capability`'s name into `registry`. Default implementation
        matches by Tool.name == capability; override for servers whose tool
        naming doesn't line up with Requisite's capability names."""
```

`MCPClientRegistry` maps a *server name* (not a capability name) to a
configured `BaseMCPClient`, since a single MCP server commonly exposes
several tools mapping to several different capabilities.

### Dependency flow

```
workflows  ->  orchestrators  ->  agents  ->  ai  ->  providers  ->  core
                                     |
                          tools, skills, capabilities  ->  core
                                     |
                          (planned) memory, mcp  ->  core, tools
```

Rule: **dependencies point strictly downward; `core` depends on nothing
else in the framework.** A layer may depend on anything below it, never
sideways-and-back or upward. This is enforced by convention and code
review today; see [Follow-ups](#follow-ups) for when that might become a
lint rule instead.

### Extension points

Every row in the interface table above is an extension point, resolved
the same way: implement the interface, register an instance/constructor
with the matching registry. There is deliberately **no separate "plugin"
base class or `Plugin` interface** — a plugin *is* one or more of these
registrations, typically made in the plugin package's own `__init__.py`.
This keeps "how do I extend Requisite" to one answer per layer instead of
two (the interface, and a plugin wrapper around the interface).

RAG is the one planned area *not* listed as a single interface above,
because it decomposes into several independent extension points rather
than one: `BaseEmbeddingProvider`, `BaseVectorStore`, `BaseRetriever`,
chunking strategies, re-rankers. That decomposition is deferred to its own
ADR when RAG work starts, rather than guessed at here.

### Plugin discovery

**Decision: explicit, import-time registration for v1. No automatic
entry-point discovery yet.**

A plugin package registers itself by being imported and calling
`default_registry.register(...)` (or by the application constructing its
own registry instance and passing it explicitly). There is no mechanism
today for Requisite to discover and auto-import a plugin package just
because it's `pip install`ed.

This is a narrower decision than it looks: it means today, "installing a
plugin" is `pip install some-requisite-plugin` **and** `import
some_requisite_plugin` (or the plugin's own instructions) — not just the
former. We chose this deliberately over `importlib.metadata.entry_points`
auto-discovery for now, because:

- It keeps import-time behavior fully predictable — nothing runs code
  from a package you didn't explicitly import, which matters for a
  framework that will run model-directed tool execution.
- It avoids committing to an entry-point group naming scheme
  (`requisite.providers`, `requisite.capabilities`, ...) before there's a
  second or third real plugin to validate the scheme against.

**Trigger to revisit:** once there are a handful of real third-party
packages doing this manually, or once the planned `requisite` CLI wants to
answer "what's installed" without every plugin needing to already be
imported — implement entry-point discovery as an *additive* layer (a
`requisite.plugins.discover()` call that imports registered entry points
and lets them self-register exactly as they do today), not a replacement
for explicit registration.

### Configuration model

**Decision: one `Settings` object (`pydantic-settings`), passed down via
constructor injection, never read from the environment implicitly deeper
in the stack.**

- `Settings` is the only place `os.environ` / `.env` gets read.
  `AI`, `Agent`, and every provider receive already-resolved values
  (`api_key`, `model`, `temperature`, ...) through constructor parameters —
  none of them call `os.environ.get` themselves. This is what makes
  constructing an isolated `Settings(...)` in a test sufficient to fully
  control configuration, with no environment leakage.
- Secrets are typed `SecretStr` specifically so `repr(settings)` — and by
  extension, an accidental `logger.debug(settings)` — can never leak a key.
- Provider-specific and cross-cutting config live in the *same* `Settings`
  object today (there's one `default_provider`, one `model`, one
  `temperature`, not per-provider sub-configs). This is intentionally
  simple for v1; see [Follow-ups](#follow-ups) for the scaling question.

### Public API principles

1. **A public API is anything importable from `requisite` directly, or
   from a sub-package without a leading underscore.** Everything else can
   change without a deprecation cycle.
2. **Every configuration surface accepts both a name (string) and an
   already-built instance.** `AI(provider="openai")` and
   `AI(provider=my_provider_instance)` both work; same pattern for
   `Workflow(orchestrator=...)`. This is what lets tests inject fakes
   without needing a registry at all.
3. **Fluent methods return `self`.** `Workflow.add`, `Workflow.use_*`,
   `Agent.requires` all return the instance they were called on, so
   `agent.requires("weather").requires("internet_search")` and
   `workflow.add(a).add(b)` both work. New chainable methods should follow
   this, not return `None`.
4. **A facade method's simple form returns the simple type; the
   `_response` / richer form returns the full object.** `ai.chat(...)`
   returns `str` (or the parsed model); `ai.chat_response(...)` returns the
   full `ChatResponse`. `agent.run(...)` returns `AgentResult` rather than
   following this split, because an agent's result (tool calls executed,
   iteration count) is closer to the primary value than to the diagnostic
   extra.
5. **No method signature changes for a provider-specific capability.**
   `tools=` and `response_model=` are on every provider's `chat`/`achat`,
   even though not every provider supports both — an unsupported
   combination should raise a clear `ProviderException`, not require a
   different method name per provider.

### What belongs in `requisite-core` vs. optional integrations

**Decision: ship as a single PyPI distribution (`requisite-ai`) with
optional extras, not separate `requisite-core` / `requisite-openai` /
`requisite-langgraph` packages — for now.**

The *logical* boundary already exists and is enforced today by the lazy
SDK import pattern described in `ARCHITECTURE.md`; this ADR makes the
boundary explicit so it's the same one used if/when a physical split
happens:

| Belongs in core (zero extra deps, always importable) | Is an integration (needs an extra) |
|---|---|
| `core/`, `config/`, `ai.py` | -- |
| `providers/base.py`, `providers/factory.py` | `providers/openai_provider.py` (`openai`), `providers/gemini_provider.py` (`google-genai`) |
| `tools/`, `skills/`, `agents/` | -- |
| `capabilities/registry.py` | -- |
| `capabilities/resolvers.py` — **counts as core**, despite making network calls, because it depends only on the standard library (`urllib.request`). The core/integration line is drawn at "requires an extra pip dependency," not "makes a network call." | A future paid-search or auth-gated resolver that needs its own SDK |
| `orchestrators/base.py`, `orchestrators/factory.py`, `orchestrators/native.py` | `orchestrators/langgraph_orchestrator.py` (`langgraph`) |
| `workflows/` | -- |
| (planned) `memory/base.py`, in-process memory impl | (planned) Redis/SQLite/vector-store memory impls |
| (planned) `mcp/base.py` | (planned) concrete MCP client implementations |

**Trigger to revisit the single-package decision:** if core's own
dependency footprint (`pydantic`, `pydantic-settings`) ever grows, if a
third-party integration maintainer wants a release cadence independent of
core, or if an integration's transitive dependencies start conflicting
with core's for a meaningful number of users. None of those are true
today, and splitting prematurely would mean solving cross-package
versioning (does `requisite-langgraph` pin an exact `requisite-core`
version range?) before there's a concrete need to.

## Alternatives considered

- **`AIProvider` / `Agent` / `Tool` as the literal interface names**,
  matching this ADR's request almost verbatim. Rejected in favor of
  `Base<Noun>` for the *abstract* interfaces, to stay consistent with
  `BaseSkill`/`BaseOrchestrator` already shipped, and because a bare
  `Provider`/`Agent` name is easily shadowed by an application's own
  domain classes of the same name. `Agent` and `Tool` themselves (the
  concrete classes) do keep the plain names, per the note in
  [Core interfaces](#core-interfaces).
- **`importlib.metadata` entry-point plugin discovery from the start.**
  Rejected for now — see [Plugin discovery](#plugin-discovery) for the
  full reasoning and the trigger to reconsider.
- **Splitting into `requisite-core` + per-integration packages
  immediately.** Rejected for now — see the trigger conditions above.
  The lazy-import + extras pattern already gives most of the practical
  benefit (you only install what you use) without the versioning overhead
  of multiple distributions.
- **A single `Plugin` base class** that a plugin implements once
  (`register(self, providers, tools, capabilities, ...)`), rather than
  registering with each registry directly. Rejected because it would be
  an abstraction with no behavior of its own — every real plugin still
  just calls `registry.register(...)` inside it — and would make partial
  plugins (one that only adds a capability resolver) more awkward, not
  less.

## Consequences

### Positive

- Every extension point (provider, orchestrator, capability, and — once
  built — memory, MCP client) is discoverable by the same mental model:
  find the `Base*` interface, find its `*Registry`, register. A new
  contributor who's read one `CONTRIBUTING.md` walkthrough can generalize
  to the others.
- `Memory` and `MCPClient` can be implemented later by someone who never
  talked to whoever writes them, because the contract (this ADR) already
  exists — including the specific decision that MCP tools surface as
  `CapabilityProvider`s, not as a separate "MCP mode" applications have to
  branch on.
- The core-vs-integration boundary is a checkable rule ("does it need an
  extra pip install?"), not a judgment call per file.

### Negative / risks

- Two interfaces (`BaseMemory`, `BaseMCPClient`) are speculative — written
  before any real implementation exists to stress-test them. There's real
  risk the first concrete `Memory` implementation reveals the interface
  is wrong in some way (e.g. sync-only session IDs not fitting a
  connection-pooled backend well). If that happens, revise this ADR (or
  supersede it) rather than quietly diverging code from the documented
  contract.
- Explicit plugin registration (no auto-discovery) puts more burden on
  plugin authors to document "and then call `.register(...)`" — a rough
  edge until the CLI/discovery follow-up lands.
- The dependency-flow rule ("strictly downward") is enforced by review,
  not tooling, today — see Follow-ups.

### Follow-ups

- Add an import-linter (or similar) config enforcing the dependency-flow
  diagram mechanically, once there are enough layers that a review-only
  policy becomes error-prone.
- Revisit single-`Settings`-object-for-everything once/if a second
  provider needs enough provider-specific configuration (e.g. separate
  timeouts per provider) that stuffing it all into one flat `Settings`
  gets awkward — likely a `ProviderSettings` sub-model at that point,
  documented in a follow-up ADR rather than expanded here speculatively.
- Implement `BaseMemory` (in-process default in core) and wire
  `Agent(memory=...)` — first real-world test of the interface specified
  above.
- Implement `BaseMCPClient` and the capability-provider bridge described
  above — first real-world test of "MCP tools are just capabilities."
