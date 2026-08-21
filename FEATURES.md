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
| Multi-agent orchestration | 🚧 | `Workflow` ships sequential, parallel, reflection, planner, supervisor (ADR-0007), critic, consensus (ADR-0011), debate, map-reduce (ADR-0012), and hierarchical (ADR-0013) — native orchestrator only; see the Multi-Agent System table below for the rest |
| Agentic execution | 🚧 | `Agent`'s own tool-calling loop is agentic (model decides which tool); the supervisor/planner strategies add model-decided agent delegation (ADR-0007); full autonomous planning across agents/skills/MCP beyond that is 📋 — see Agentic Mode below |
| MCP client integration | ✅ | `MCPClient` (stdio + Streamable HTTP) — ADR-0004 |
| MCP server integration | ✅ | `MCPServer` (stdio + Streamable HTTP) — ADR-0015 |
| Memory | ✅ | `BaseMemory`, `InProcessMemory`/`SQLiteMemory`/`RedisMemory`, wired into `Agent(memory=..., session_id=...)` |
| Retrieval | ✅ | `Retriever.retrieve()` / `.aretrieve()` |
| RAG | 🚧 | Core interfaces, vector stores, hybrid/BM25 retrieval, and re-ranking shipped (ADR-0005, ADR-0010); context compression still 📋 — see RAG table below |
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
| Streaming + tool calls together | ✅ | `AI.stream_response`/`.astream_response` (`StreamChunk.tool_calls`), `AI.stream`/`.astream` accept `tools=` too — see [ADR-0009](docs/adr/0009-streaming-tool-calls.md) |

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
| Supervisor | ✅ | `native` and `langgraph` orchestrators — coordinator (`steps[0]`) delegates to named workers (`steps[1:]`) via structured decisions, up to `max_rounds`; ADR-0007, ADR-0016 (langgraph: a real conditional graph, not a Python loop) |
| Planner | ✅ | `native` orchestrator only — coordinator (`steps[0]`) decomposes the task into a plan executed by named workers (`steps[1:]`); ADR-0007 |
| Reflection | ✅ | `native` orchestrator only — single agent critiques and revises its own output, up to `max_rounds`; ADR-0007 |
| Debate | ✅ | `native` orchestrator only — moderator (`steps[0]`) judges debaters (`steps[1:]`) after `max_rounds` of each seeing the others' prior-round arguments; ADR-0012 |
| Critic | ✅ | `native` orchestrator only — generator (`steps[0]`) and a separate critic (`steps[1]`) iterate on a draft, up to `max_rounds`; ADR-0011 |
| Consensus | ✅ | `native` orchestrator only — synthesizer (`steps[0]`) combines the independent, concurrently-run answers of `steps[1:]`; ADR-0011 |
| Hierarchical | ✅ | `native` orchestrator only — same shape as Supervisor, except a delegate may be an `Agent` or a named `Workflow` (nested "team"); ADR-0013 |
| Map-reduce | ✅ | `native` orchestrator only — reducer (`steps[0]`) combines mapper (`steps[1:]`) results for `map_items=`, assigned round-robin; ADR-0012 |
| Tree of thoughts | ✅ | `native` orchestrator only — evaluator (`steps[0]`) scores candidate reasoning steps thinkers (`steps[1:]`) generate; `breadth`/`beam_width`/`max_depth` control the search; ADR-0018 |
| Graph execution (arbitrary DAGs) | ✅ | `native` orchestrator only — nodes (`Agent` or named `Workflow`) wired with developer-declared edges via `Workflow.add_edge(from_, to, condition=...)`; routing is deterministic (checked against a node's output), not LLM-decided like every strategy above; cycles allowed, bounded by `max_steps`; ADR-0019. `LangGraphOrchestrator` still only builds two specific graph shapes (linear for sequential, conditional+cycle for supervisor — ADR-0016); a generic langgraph graph-builder remains 📋 |

## Orchestration

| Backend | Status | Notes |
|---|---|---|
| Native (no external framework) | ✅ | `NativeOrchestrator` |
| LangGraph | ✅ | `LangGraphOrchestrator` — sequential (linear chain) and supervisor (real conditional graph, ADR-0016) so far |
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
| MCP server integration | ✅ | `MCPServer` — reverse direction (exposing Requisite tools/agents as an MCP server), stdio + Streamable HTTP, verified against Requisite's own `MCPClient` — ADR-0015 |
| Tool discovery | ✅ | `BaseMCPClient.discover_tools()` / `.adiscover_tools()` |
| Remote tools | ✅ | Streamable HTTP transport |
| Local tools | ✅ | stdio transport |
| Filesystem MCP server | 📋 | A native (non-MCP) `"filesystem"` capability already ships — see Capabilities below; an MCP-based one is a config away (`MCPClient.stdio(...)` pointed at any filesystem MCP server) but not shipped as a default |
| GitHub MCP server | ✅ | `register_github_mcp_capability()` — GitHub's official remote MCP server, gated on `GITHUB_TOKEN`, registers `"github"` at priority 10 over the native `search_github` REST resolver (ADR-0020) — ADR-0023 |
| Databases MCP server | 📋 | No single canonical database MCP server to hardcode as a default; the generic `register_mcp_capability()` helper (same one `register_github_mcp_capability()` is built on) is shipped for wiring up any database MCP server — see `examples/mcp_default_capabilities.py`, ADR-0023 |
| Any MCP tool surfaces as a `CapabilityProvider` | ✅ | `BaseMCPClient.register_as_capability(...)` — verified `agent.requires(...)` can't tell native tool from MCP server |

## Memory

| Requirement | Status | Notes |
|---|---|---|
| `BaseMemory` interface | ✅ | ADR-0001 (spec) / ADR-0002 (implementation) |
| Conversation memory | ✅ | `InProcessMemory` |
| Redis-backed memory | ✅ | `RedisMemory` (`requisite.memory.redis`, `pip install requisite-ai[redis]`) |
| SQLite-backed memory | ✅ | `SQLiteMemory` (`requisite.memory.sqlite`) — zero extra dependency, stdlib `sqlite3` |
| Vector-database-backed memory | ✅ | `VectorMemory` (`requisite.memory.vector`) — composes a chronological `BaseMemory` delegate with RAG's `BaseEmbeddingProvider`/`BaseVectorStore` for semantic top-k recall (`load_relevant()`), beyond `BaseMemory`'s own plain-storage contract; ADR-0022 |
| Postgres-backed memory | 📋 | |
| Knowledge-graph-backed memory | 📋 | |

## RAG

Core interfaces, the in-memory default, Pinecone/Weaviate (ADR-0005),
and hybrid/BM25 retrieval + re-ranking (ADR-0010) are all shipped.

| Requirement | Status | Notes |
|---|---|---|
| Embedding providers | ✅ | `OpenAIEmbeddingProvider`, `GeminiEmbeddingProvider` |
| Vector stores | ✅ | `InMemoryVectorStore` (zero deps, matches `InProcessMemory`), `PineconeVectorStore`, `WeaviateVectorStore` |
| Chunking | ✅ | `chunk_text()` — character-based with overlap; token-aware chunking is a follow-up |
| Retrievers | ✅ | `Retriever` (dense), `BM25Retriever` (keyword, zero deps), `HybridRetriever` (dense + BM25 fused via Reciprocal Rank Fusion); each exposed as a `CapabilityProvider` via `.as_tool()` — `agent.requires("knowledge_base")`, not a new `Agent` constructor parameter |
| Hybrid search | ✅ | `HybridRetriever` — see [ADR-0010](docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md) |
| Re-ranking | ✅ | `BaseReranker` + `LLMReranker` (listwise, reuses `AI`/`response_model=` — no new ML dependency) |
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
| Discovery mechanism (entry points) | ✅ | `requisite.plugins.discover()` scans the `"requisite.plugins"` entry-point group and imports/calls each one; explicit `.register(...)` registration (v1's mechanism) is unchanged and still required inside the plugin — discovery only automates the import step. `requisite plugins` on the CLI — see [ADR-0017](docs/adr/0017-entry-point-plugin-discovery.md) |

## CLI

| Item | Status | Notes |
|---|---|---|
| `requisite` CLI | ✅ | `requisite init/providers/capabilities/agents/plugins/chat` — see [ADR-0014](docs/adr/0014-cli.md) |
| Scaffold a new project | ✅ | `requisite init NAME [--provider]` |
| List registered providers | ✅ | `requisite providers` — also reports SDK-installed / API-key-configured status |
| List registered capabilities | ✅ | `requisite capabilities` |
| List registered agents | ✅ | `requisite agents` — reads the scaffolded project's `agents.py` `agent_registry` convention, since (unlike providers/capabilities) there's no framework-wide default `AgentRegistry` to introspect |
| Discover installed plugins | ✅ | `requisite plugins [--group]` — see [ADR-0017](docs/adr/0017-entry-point-plugin-discovery.md) |
| Quick chat | ✅ | `requisite chat [PROMPT]` — one-shot or interactive REPL; `--agent NAME` routes through a scaffolded project's agent |

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
| Python `logging`, no `print()` | ✅ | Framework code only — `requisite/cli/` deliberately prints its user-facing command output directly rather than logging it; see [ADR-0014](docs/adr/0014-cli.md) |
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
