# Feature Tracker

A line-by-line traceability matrix against the **original project
vision** (the initial spec this framework was scoped from), so it's easy
to see at a glance what's implemented, what's in progress, and what's
still pending — independent of how `ROADMAP.md` organizes the same
information by framework layer. If this file and `ROADMAP.md` disagree,
trust `ROADMAP.md` for planning and open an issue to reconcile this one.

Status legend: ✅ Done · 🚧 Partial · 📋 Not started · N/A Deliberately out of scope (see `ROADMAP.md`)

## Core philosophy

| Principle | Status | Notes |
|---|---|---|
| Simplicity for end users | ✅ | `AI()`, `Agent()`, `Workflow()` — see README Quick Start |
| Extensibility | ✅ | Interface + registry pattern throughout — `ARCHITECTURE.md` |
| Loose coupling | ✅ | Dependencies point strictly downward — ADR-0001 |
| Dependency injection | ✅ | Constructor injection everywhere; no globals except convenience `default_registry` instances |
| Clean architecture | ✅ | `core/ → providers/ → tools/skills/capabilities/memory → agents → orchestrators → workflows` |
| SOLID principles | ✅ | Single-responsibility registries; interfaces segregated per concern |
| Pythonic APIs | ✅ | Fluent chaining, dataclass-like Pydantic models, context-manager-free simple calls |
| Strong typing | ✅ | `mypy --strict` clean across the codebase; `py.typed` marker shipped |
| Testability | ✅ | Every provider/registry has a fake/mock-based test; zero network calls in the suite |
| Excellent documentation | ✅ | `README.md`, `ARCHITECTURE.md`, `DEVELOPMENT.md`, `CONTRIBUTING.md`, ADRs, per-class docstrings |

## Primary goals

| Goal | Status | Notes |
|---|---|---|
| AI provider configuration | ✅ | `Settings` + `AI(provider=...)` |
| Chat completions | ✅ | `AI.chat` / `AI.chat_response` (+ async) |
| Structured outputs | ✅ | `response_model=` on all 8 providers |
| Streaming responses | ✅ | `AI.stream` / `AI.astream` on all 8 providers |
| Function / Tool calling | ✅ | `@tool`, `Tool`, `ToolRegistry`; wired into all 8 providers |
| Skills | ✅ | `BaseSkill`, `.as_tool()` |
| Skills registry | ✅ | `SkillRegistry` |
| Agent creation | ✅ | `Agent(...)` |
| Agent execution | ✅ | `Agent.run` / `Agent.arun` (tool-calling loop) |
| Agent registry | ✅ | `AgentRegistry` |
| Multi-agent orchestration | 🚧 | `Workflow` ships sequential, parallel, reflection, planner, and supervisor (native orchestrator only — ADR-0007); see the Multi-Agent System table below for the rest |
| Agentic execution | 🚧 | `Agent`'s own tool-calling loop is agentic (model decides which tool); the supervisor/planner strategies add model-decided agent delegation (ADR-0007); full autonomous planning across agents/skills/MCP beyond that is 📋 — see Agentic Mode below |
| MCP client integration | ✅ | `MCPClient` (stdio + Streamable HTTP) — ADR-0004 |
| MCP server integration | 📋 | |
| Memory | ✅ | `BaseMemory`, `InProcessMemory`, wired into `Agent(memory=..., session_id=...)` |
| Retrieval | ✅ | `Retriever.retrieve()` / `.aretrieve()` |
| RAG | 🚧 | Core interfaces + in-memory/Pinecone/Weaviate vector stores shipped (ADR-0005); hybrid search, re-ranking, context compression still 📋 — see RAG table below |
| Prompt templates | ✅ | `PromptTemplate`, `ChatPromptTemplate`, `PromptTemplateRegistry` — ADR-0003 |
| Conversation management | ✅ | `Message` history + `BaseMemory` (storage) + `BaseConversationPolicy` (retention: `MessageCountPolicy`, `SummarizingPolicy`) — ADR-0003 |
| Workflow execution | ✅ | `Workflow.run` / `Workflow.arun` |

## Supported providers

| Provider | Status | Notes |
|---|---|---|
| OpenAI | ✅ | `OpenAIProvider` |
| Anthropic Claude | ✅ | `AnthropicProvider` |
| Google Gemini | ✅ | `GeminiProvider` |
| Azure OpenAI | ✅ | `AzureOpenAIProvider` (v1 GA API) |
| Ollama | ✅ | `OllamaProvider` — native `ollama` client, not the OpenAI-compat subclass pattern (Ollama's own compat endpoint is documented experimental) |
| OpenRouter | ✅ | `OpenRouterProvider` (`OpenAIProvider` subclass) — ADR-0002 |
| Groq | ✅ | `GroqProvider` (`OpenAIProvider` subclass) |
| Together AI | ✅ | `TogetherProvider` (`OpenAIProvider` subclass) |
| Local models | ✅ | Via Ollama |

## Architecture

| Requirement | Status | Notes |
|---|---|---|
| Clean layered architecture | ✅ | See `ARCHITECTURE.md` |
| Business logic never depends on implementation details | ✅ | `AI`/`Agent`/`Workflow` depend only on `Base*` interfaces |

## Dependency injection

| Requirement | Status | Notes |
|---|---|---|
| Everything injectable | ✅ | Provider, orchestrator, memory, capability registry, settings — all constructor params |
| No direct instantiation of implementations inside business logic | ✅ | Enforced by convention + code review (ADR-0001 follow-up: not yet lint-enforced) |
| Constructor injection preferred | ✅ | |

## Configuration

| Requirement | Status | Notes |
|---|---|---|
| Simple `.env`-based configuration | ✅ | `.env.example`, `pydantic-settings` |
| Configuration classes, not manual `os.environ.get` | ✅ | `Settings` |
| Extensible for provider-specific config | ✅ | `Settings.provider_kwargs(name)` — ADR-0002 |

## AI API

| Requirement | Status | Notes |
|---|---|---|
| `AI().chat("...")` | ✅ | |
| Streaming (`for token in ai.stream(...)`) | ✅ | |
| Structured output (`response_model=Person`) | ✅ | |

## Tool calling

| Requirement | Status | Notes |
|---|---|---|
| `@tool` decorator on plain functions | ✅ | |
| `agent.use_tool(x)` — or its equivalent | ✅ | Via `tools=[...]` at `Agent` construction, or `ToolRegistry.register` |
| No provider-specific code required from the developer | ✅ | JSON Schema auto-derived; each provider translates internally |

## Skills

| Requirement | Status | Notes |
|---|---|---|
| Composable | ✅ | `BaseSkill.as_tool()` |
| Reusable | ✅ | |
| Discoverable | ✅ | `SkillRegistry` |
| Independently testable | ✅ | Plain classes, no framework coupling required to test |

## Agents

| Requirement | Status | Notes |
|---|---|---|
| `Agent(name=..., provider=..., tools=..., skills=...)` | ✅ | |
| `agent.run("...")` | ✅ | Returns `AgentResult` |

## Agent registry

| Requirement | Status | Notes |
|---|---|---|
| `registry.register(agent)` / `registry.get(name)` | ✅ | `AgentRegistry` |

## Multi-agent system

| Strategy | Status | Notes |
|---|---|---|
| Sequential | ✅ | `NativeOrchestrator`, `LangGraphOrchestrator` |
| Parallel | ✅ | `NativeOrchestrator` only so far |
| Supervisor | ✅ | `native` orchestrator only — coordinator (`steps[0]`) delegates to named workers (`steps[1:]`) via structured decisions, up to `max_rounds`; ADR-0007 |
| Planner | ✅ | `native` orchestrator only — coordinator (`steps[0]`) decomposes the task into a plan executed by named workers (`steps[1:]`); ADR-0007 |
| Reflection | ✅ | `native` orchestrator only — single agent critiques and revises its own output, up to `max_rounds`; ADR-0007 |
| Debate | 📋 | |
| Critic | 📋 | |
| Consensus | 📋 | |
| Hierarchical | 📋 | |
| Map-reduce | 📋 | |
| Tree of thoughts | 📋 | |
| Graph execution (arbitrary DAGs) | 🚧 | `LangGraphOrchestrator` currently builds only a linear graph; general DAGs are 📋 |

## Orchestration

| Backend | Status | Notes |
|---|---|---|
| Native (no external framework) | ✅ | `NativeOrchestrator` |
| LangGraph | ✅ | `LangGraphOrchestrator` (sequential only so far) |
| LangChain | 📋 | Not currently planned as a distinct backend — evaluate if a real need emerges |
| CrewAI | 📋 | Registered today as a clear "not yet implemented" placeholder (`workflow.use_crewai()`) |
| AutoGen | 📋 | Same placeholder treatment |
| Public API stays identical across backends | ✅ | `Workflow.add()` / `.run()` unchanged regardless of `.use_*()` |

## Agentic mode

| Requirement | Status | Notes |
|---|---|---|
| Model decides which tool to call | ✅ | `Agent`'s tool-calling loop |
| Model decides which skill to use | ✅ | Skills are exposed as tools, so this falls out of the above |
| Model decides which agent to delegate to (multi-agent) | ✅ | `workflow.supervisor()` / `workflow.planner()` — see Multi-Agent System table, ADR-0007 |
| Model decides whether MCP is needed | 🚧 | MCP tools are available via `agent.requires(...)` like any capability, but there's no dedicated "decide to use MCP" planning step beyond ordinary tool-calling |
| Model decides whether retrieval is needed | 📋 | Depends on RAG |

## MCP (Model Context Protocol)

| Requirement | Status | Notes |
|---|---|---|
| `BaseMCPClient` interface | ✅ | ADR-0001 (spec) / ADR-0004 (implementation) |
| MCP client integration | ✅ | `MCPClient` — both stdio and Streamable HTTP transports, verified against real MCP servers |
| MCP server integration | 📋 | Reverse direction (exposing Requisite as an MCP server) — not started |
| Tool discovery | ✅ | `BaseMCPClient.discover_tools()` / `.adiscover_tools()` |
| Remote tools | ✅ | Streamable HTTP transport |
| Local tools | ✅ | stdio transport |
| Filesystem MCP server | 📋 | A native (non-MCP) `"filesystem"` capability already ships — see Capabilities below; an MCP-based one is a config away (`MCPClient.stdio(...)` pointed at any filesystem MCP server) but not shipped as a default |
| GitHub MCP server | 📋 | `GITHUB_TOKEN` reserved in `.env.example`; works today via `MCPClient` + a GitHub MCP server, just not registered as a default capability yet |
| Databases MCP server | 📋 | Same as above — usable today via `MCPClient`, not a shipped default |
| Any MCP tool surfaces as a `CapabilityProvider` | ✅ | `BaseMCPClient.register_as_capability(...)` — verified `agent.requires(...)` can't tell native tool from MCP server |

## Memory

| Requirement | Status | Notes |
|---|---|---|
| `BaseMemory` interface | ✅ | ADR-0001 (spec) / ADR-0002 (implementation) |
| Conversation memory | ✅ | `InProcessMemory` |
| Redis-backed memory | 📋 | `AWS_*` / general cloud creds reserved in `.env.example` for this class of backend |
| SQLite-backed memory | 📋 | |
| Vector-database-backed memory | 📋 | Depends on RAG's vector store work |
| Postgres-backed memory | 📋 | |
| Knowledge-graph-backed memory | 📋 | |

## RAG

Core interfaces, the in-memory default, and Pinecone/Weaviate are all
shipped (ADR-0005).

| Requirement | Status | Notes |
|---|---|---|
| Embedding providers | ✅ | `OpenAIEmbeddingProvider`, `GeminiEmbeddingProvider` |
| Vector stores | ✅ | `InMemoryVectorStore` (zero deps, matches `InProcessMemory`), `PineconeVectorStore`, `WeaviateVectorStore` |
| Chunking | ✅ | `chunk_text()` — character-based with overlap; token-aware chunking is a follow-up |
| Retrievers | ✅ | `Retriever` (dense); exposed as a `CapabilityProvider` via `.as_tool()` — `agent.requires("knowledge_base")`, not a new `Agent` constructor parameter |
| Hybrid search | 📋 | `BaseRetriever` is independent of embeddings at the interface level specifically to allow this later |
| Re-ranking | 📋 | |
| Context compression | 📋 | |

## Prompt templates & conversation management

| Requirement | Status | Notes |
|---|---|---|
| Prompt templates | ✅ | `PromptTemplate` (single string), `ChatPromptTemplate` (renders to `list[Message]`), both `str.format`-based — ADR-0003 |
| Conversation management | ✅ | Storage via `BaseMemory`; retention via `BaseConversationPolicy` (`MessageCountPolicy`, `SummarizingPolicy`), applied once per `Agent.run()` — ADR-0003 |

## Workflow execution

| Requirement | Status | Notes |
|---|---|---|
| `workflow.add(agent)` | ✅ | |
| `workflow.run()` | ✅ | |
| `workflow.use_langgraph()` (swap orchestration engine) | ✅ | |

## Extensibility

| Replaceable without modifying framework code | Status | Notes |
|---|---|---|
| Provider | ✅ | `ProviderRegistry` |
| Memory | ✅ | `MemoryRegistry` |
| Retriever | ✅ | `EmbeddingRegistry` / `VectorStoreRegistry`; swap `BaseRetriever` implementations directly |
| Tool registry | ✅ | Construct your own `ToolRegistry()` |
| Skill registry | ✅ | Construct your own `SkillRegistry()` |
| Execution engine (orchestrator) | ✅ | `OrchestratorRegistry` |
| Planner | 📋 | The `planner` multi-agent strategy now ships (ADR-0007), but its planning logic lives inside `NativeOrchestrator._run_planner`, not behind a separate, replaceable `BasePlanner`-style interface yet |
| Prompt builder | ✅ | `PromptTemplate` / `ChatPromptTemplate` |
| Configuration | ✅ | Pass an explicit `Settings(...)` instance anywhere one is accepted |

## Plugin architecture

| Plugins may register | Status | Notes |
|---|---|---|
| Providers | ✅ | `default_registry.register(name, ctor)` |
| Skills | ✅ | Just construct and pass a `BaseSkill` — no registration ceremony needed |
| Agents | ✅ | `AgentRegistry.register(agent)` |
| Tools | ✅ | `ToolRegistry.register(...)` / `CapabilityRegistry.register(...)` |
| MCP servers | ✅ | `MCPClientRegistry.register(...)`, then `.register_as_capability(...)` into `CapabilityRegistry` |
| Retrievers | ✅ | `Retriever.as_tool()` -> `CapabilityRegistry.register(...)` |
| Vector stores | ✅ | `VectorStoreRegistry.register(...)` — `in_memory`, `pinecone`, `weaviate` all registered by default |
| Memory providers | ✅ | `MemoryRegistry.register(...)` |
| Prompt templates | ✅ | `PromptTemplateRegistry.register(...)` |
| Discovery mechanism (entry points) | 📋 | Deliberately deferred — explicit import + register for v1; see ADR-0001's Plugin Discovery section for the trigger to add it |

## Coding standards

| Requirement | Status | Notes |
|---|---|---|
| PEP 8 compliant | ✅ | Enforced via `ruff check` + `ruff format` in CI |
| Type hints everywhere | ✅ | `mypy --strict` clean |
| Pydantic models for data | ✅ | `Message`, `ChatResponse`, `Tool`, `WorkflowResult`, ... |
| Meaningful docstrings | ✅ | NumPy-style, documented in `DEVELOPMENT.md` |
| Small functions, high cohesion, low coupling | ✅ | |
| Composition over inheritance | 🚧 | Mostly true; one deliberate, documented exception (`OpenAIProvider` subclassing for wire-compatible providers) — ADR-0002 |
| No global state, no singletons | ✅ | Registries are plain classes; `default_*` instances are convenience, not enforced singletons |
| Dataclasses/Pydantic where appropriate | ✅ | |

## Documentation

| Requirement | Status | Notes |
|---|---|---|
| Purpose, parameters, examples, return values, edge cases per public class | ✅ | Enforced by convention, spot-checked in review — see `DEVELOPMENT.md` |

## Testing

| Requirement | Status | Notes |
|---|---|---|
| Unit tests | ✅ | 184 tests as of this writing |
| Integration tests | 🚧 | Real (installed, not faked) `langgraph` integration tests exist; real-network provider integration tests do not (deliberately — see `DEVELOPMENT.md`'s no-network-in-tests rule) |
| Mock providers | ✅ | `sys.modules`-injected fakes for OpenAI/Gemini/Anthropic SDKs |
| Mock MCP servers | ✅ | `tests/test_mcp.py` fakes the `mcp` session object; a real stdio + Streamable HTTP server round-trip was verified manually during implementation (ADR-0004) but isn't part of CI |
| Mock LLM responses | ✅ | Scripted fake providers throughout (`tests/test_agents.py`, `tests/test_workflows.py`, ...) |
| High coverage | 🚧 | Coverage tracked via Codecov in CI; no enforced minimum threshold yet |

## Error handling

| Requirement | Status | Notes |
|---|---|---|
| Never swallow exceptions | ✅ | `raise ... from original_error` everywhere — `DEVELOPMENT.md` |
| Custom exception hierarchy | ✅ | `AIException` → `ProviderException`, `ToolException`, `SkillException`, `AgentException`, `CapabilityException`, `ConfigurationException`, `MCPException` (reserved) |

## Logging

| Requirement | Status | Notes |
|---|---|---|
| Python `logging`, no `print()` | ✅ | |
| Configurable log levels | ✅ | Standard `logging` configuration applies; per-subsystem loggers namespaced `requisite.<subpackage>` |
| Structured logging | ✅ | `requisite.telemetry.JSONFormatter` + `configure_logging()` — opt-in, never automatic (ADR-0003) |

## Async support

| Requirement | Status | Notes |
|---|---|---|
| Sync | ✅ | |
| Async | ✅ | Hand-written pairs (`chat`/`achat`, `run`/`arun`, ...), not derived via a thread-pool wrapper — ADR-0001 |
| Streaming | ✅ | Both sync and async streaming on every provider |

## Performance

| Requirement | Status | Notes |
|---|---|---|
| Lazy initialization | ✅ | Provider SDK clients built on first use, not at construction; SDK imports deferred |
| Cache expensive objects | ✅ | Provider clients cached per instance after first build |
| Reuse HTTP clients | ✅ | One client instance reused across calls on a given provider instance |
| Support concurrency | ✅ | `Workflow`'s parallel strategy (`ThreadPoolExecutor` for sync, `asyncio.gather` for async) |
| Proactive rate limiting for provider quotas | ✅ | `RateLimiter` (sliding-window log), opt-in via `Settings.rate_limit_rpm` or explicit `rate_limiter=` on `AI`/`Agent`; share one instance across agents that draw on the same API key — ADR-0008 |

## Security

| Requirement | Status | Notes |
|---|---|---|
| Never expose secrets | ✅ | API keys typed `SecretStr`; masked in `repr()` |
| Never log API keys | ✅ | Convention documented in `DEVELOPMENT.md`; no code path logs `.get_secret_value()` |
| Validate inputs | 🚧 | Pydantic validates data shapes; tool-argument-level validation is the tool author's responsibility (documented in `SECURITY.md`) |
| Sanitize tool outputs when appropriate | 📋 | Not currently done automatically — an application-level concern today |

## Open source standards

| Requirement | Status | Notes |
|---|---|---|
| Professional OSS practices | ✅ | `LICENSE`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates, CI |
| Community-contribution-ready code | ✅ | `CONTRIBUTING.md` walkthroughs for each extension point |
| Small modules | ✅ | |
| Avoid breaking public APIs | ✅ | Policy documented in `DEVELOPMENT.md` |
| Document breaking changes | ✅ | `CHANGELOG.md` |
| Semantic versioning | ✅ | |

## Backward compatibility

| Requirement | Status | Notes |
|---|---|---|
| Public APIs remain stable | ✅ | Policy in `DEVELOPMENT.md` / ADR-0001's Public API Principles |
| Prefer deprecation over removal | ✅ | Policy documented; `ToolRegistry._resolve` is the one existing example in code |

---

*Last updated alongside `CHANGELOG.md`'s 0.6.0 entry. When a roadmap item
ships, update its row here **and** the corresponding row in
`ROADMAP.md` — they're allowed to organize the same facts differently,
but not to disagree about what's actually shipped.*
