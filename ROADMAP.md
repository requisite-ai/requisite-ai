# Roadmap

Requisite is deliberately scoped: **a provider-agnostic AI application
framework with pluggable execution engines, where every layer is an
interface + implementation(s) + a plain registry.** Everything below is a
module riding that same pattern — nothing on this roadmap requires
changing the core shape of the framework to add.

> See [`FEATURES.md`](FEATURES.md) for the same status information
> organized as a line-by-line checklist against the original project
> vision, rather than by framework layer.

Status legend: ✅ Shipped · 🚧 In progress · 📋 Planned · 💭 Under discussion

## Core

| Item | Status |
|---|---|
| `AI` facade — chat, structured output, streaming, async | ✅ |
| `BaseProvider` interface + `ProviderRegistry` | ✅ |
| `Settings` (pydantic-settings, `.env`) | ✅ |
| `AIException` hierarchy | ✅ |
| `py.typed` / PEP 561 typed distribution | ✅ |
| Proactive rate limiting (`RateLimiter`, shareable across `Agent`/`AI` instances) | ✅ — see [ADR-0008](docs/adr/0008-rate-limiting.md) |

## Providers

| Item | Status |
|---|---|
| OpenAI (`openai>=1.35` client SDK) | ✅ |
| Gemini (`google-genai` unified SDK) | ✅ |
| Anthropic Claude (`anthropic>=0.116`, native structured output + tool calling) | ✅ |
| Azure OpenAI (v1 GA API, `OpenAIProvider` subclass) | ✅ |
| Groq (OpenAI-wire-compatible, `OpenAIProvider` subclass) | ✅ |
| OpenRouter (OpenAI-wire-compatible, `OpenAIProvider` subclass — [ADR-0002](docs/adr/0002-provider-kwargs-and-memory-integration.md)) | ✅ |
| Together AI (OpenAI-wire-compatible, `OpenAIProvider` subclass — same pattern) | ✅ |
| Ollama (local models, native `ollama` client — not the OpenAI-compat subclass pattern, since Ollama's own compat endpoint is documented experimental) | ✅ |
| Local models (general) | ✅ — via Ollama |

Each is an implementation of `BaseProvider` — see `CONTRIBUTING.md` for the
exact steps. **Community-contributed providers are one of the highest-value,
lowest-risk contributions** since they can't break other providers by
construction (each lives in its own module, imported lazily).

## Tool calling, skills, structured output

| Item | Status |
|---|---|
| `@tool` decorator + JSON Schema auto-derivation | ✅ |
| `ToolRegistry` | ✅ |
| Tool calling wired into OpenAI + Gemini (incl. multi-turn round-trip) | ✅ |
| `response_model=` structured output (OpenAI `.parse()`, Gemini `response_schema`) | ✅ |
| `BaseSkill` / `SkillRegistry` | ✅ |
| Streaming + tool calls together (partial tool-call deltas) | ✅ — see [ADR-0009](docs/adr/0009-streaming-tool-calls.md); `StreamChunk.tool_calls` reports complete tool calls once known, never partial JSON |
| Parallel tool calls in a single turn (multiple calls, one round-trip) | ✅ — `Agent.arun()` runs a turn's tool calls concurrently via `asyncio.gather`; `Agent.run()` (sync) executes them sequentially, which is inherent to sync execution rather than a gap |

## Capabilities (`agent.requires(...)`)

| Item | Status |
|---|---|
| `CapabilityRegistry` — priority + availability-based resolution | ✅ |
| Default resolvers: `filesystem`, `weather`, `internet_search` | ✅ |
| `Agent.requires(...)` | ✅ |
| `github` default resolver (public, unauthenticated REST API) | ✅ — `search_github(query)` searches GitHub's free, keyless Search API (`sort=stars`), top 5 results; unauthenticated only (`GITHUB_TOKEN` is reserved separately for the future MCP server, below); see [ADR-0020](docs/adr/0020-github-capability-resolver.md) |
| Cost-based / policy-based resolution (beyond priority + availability) | 💭 |
| Conflict handling when two plugins register the same capability at equal priority | 💭 — currently first-registered wins; needs a real spec, not a default |

## Agents & multi-agent orchestration

| Item | Status |
|---|---|
| `Agent` — tool-calling loop, sync + async, `max_iterations` guard | ✅ |
| `AgentRegistry` | ✅ |
| `Workflow` — `.add()` / `.run()` / `.arun()` | ✅ |
| Native orchestrator: sequential strategy | ✅ |
| Native orchestrator: parallel strategy | ✅ |
| `langgraph` orchestrator backend (linear graph) | ✅ |
| `langgraph` backend: branching / conditional graphs | ✅ — `supervisor` strategy only (real `add_conditional_edges` + a loop-back cycle, not a disguised loop); `reflection`/`hierarchical`/`graph` on langgraph remain 📋 — see [ADR-0016](docs/adr/0016-langgraph-branching.md) |
| Supervisor strategy (a coordinating agent delegates to others) | ✅ — `native` orchestrator only; see [ADR-0007](docs/adr/0007-multi-agent-orchestration-strategies.md) |
| Planner strategy | ✅ — `native` orchestrator only; see [ADR-0007](docs/adr/0007-multi-agent-orchestration-strategies.md) |
| Reflection strategy (agent critiques and revises its own output) | ✅ — `native` orchestrator only; see [ADR-0007](docs/adr/0007-multi-agent-orchestration-strategies.md) |
| Debate / critic / consensus strategies | ✅ — `native` orchestrator only; see [ADR-0011](docs/adr/0011-critic-and-consensus-strategies.md), [ADR-0012](docs/adr/0012-debate-and-map-reduce-strategies.md) |
| Hierarchical strategy | ✅ — `native` orchestrator only; a delegate may be an `Agent` or a named `Workflow` (nested "team"), giving real recursive delegation; see [ADR-0013](docs/adr/0013-hierarchical-strategy.md) |
| Map-reduce strategy | ✅ — `native` orchestrator only; work items via `map_items=`, round-robin across mappers; see [ADR-0012](docs/adr/0012-debate-and-map-reduce-strategies.md) |
| Tree-of-thoughts strategy | ✅ — `native` orchestrator only; evaluator (`steps[0]`) scores candidates thinkers (`steps[1:]`) generate, branching and pruning a search tree via `breadth`/`beam_width`/`max_depth`; see [ADR-0018](docs/adr/0018-tree-of-thoughts-strategy.md) |
| General graph execution (arbitrary DAGs, not just linear/parallel) | ✅ — `native` orchestrator only; nodes (`Agent` or named `Workflow`) wired with developer-declared edges via `Workflow.add_edge(from_, to, condition=...)` — routing is deterministic, not LLM-decided like every other strategy; cycles allowed, bounded by `max_steps`; see [ADR-0019](docs/adr/0019-graph-execution-strategy.md) |
| CrewAI orchestrator backend | 📋 — registered today as a clear "not yet implemented" placeholder |
| AutoGen orchestrator backend | 📋 — same |

Each new strategy is a `_run_<strategy>` / `_arun_<strategy>` pair on
`NativeOrchestrator` (or an equivalent on another backend) — see
`ARCHITECTURE.md` for how strategies plug in without changing `Workflow`.

## MCP (Model Context Protocol)

| Item | Status | Notes |
|---|---|---|
| `BaseMCPClient` interface + registry | ✅ | `requisite/mcp/` — spec'd in [ADR-0001](docs/adr/0001-core-architecture-and-interfaces.md), implemented per [ADR-0004](docs/adr/0004-mcp-integration.md) |
| MCP client: stdio transport | ✅ | `MCPClient.stdio(...)` |
| MCP client: Streamable HTTP transport | ✅ | `MCPClient.http(...)` |
| MCP client: persistent session mode (vs. today's per-call reconnect) | 📋 | Deferred until reconnect latency is a measured problem — ADR-0004 |
| Migrate to `mcp` 2.x's API | 📋 | Currently capped at `mcp>=1.28,<2.0` — 2.0.0 is a breaking rewrite (restructured package layout, renamed `CallToolResult` fields, removed `streamablehttp_client`). Deliberately deferred as its own change rather than rushed under CI-failure pressure — see `CHANGELOG.md`'s 0.4.1 entry and `DEVELOPMENT.md`'s dependency policy |
| Bridge into `CapabilityRegistry` (`agent.requires("github")` -> MCP server) | ✅ | `BaseMCPClient.register_as_capability(...)` |
| MCP resource / prompt discovery (beyond tools) | 📋 | Out of scope for the initial client — ADR-0004 |
| MCP server integration (expose Requisite tools/agents as an MCP server) | ✅ | `MCPServer` (stdio + Streamable HTTP), `Agent.as_tool()` -- see [ADR-0015](docs/adr/0015-mcp-server-integration.md) |
| First-party MCP servers as default capability providers (GitHub, databases) | 📋 | `GITHUB_TOKEN` already reserved in `.env.example` |

Shipped shape: an MCP-backed tool is just another `CapabilityProvider`
in `CapabilityRegistry` — `agent.requires("github")` doesn't need to know
or care whether `"github"` is resolved by a native tool or an MCP server.

## Memory

| Item | Status |
|---|---|
| `BaseMemory` interface + registry — specified in [ADR-0001](docs/adr/0001-core-architecture-and-interfaces.md), implemented per [ADR-0002](docs/adr/0002-provider-kwargs-and-memory-integration.md) | ✅ |
| Conversation memory (in-process, ships as the default in core) | ✅ |
| `Agent(memory=..., session_id=...)` integration | ✅ |
| Redis-backed memory | ✅ — `RedisMemory`, `pip install requisite-ai[redis]` |
| SQLite-backed memory | ✅ — `SQLiteMemory`, zero extra dependency (stdlib `sqlite3`) |
| Vector-database-backed memory | 📋 |
| Knowledge-graph-backed memory | 💭 |

## RAG

Core interfaces, the in-memory default, and both Pinecone/Weaviate are
now shipped (see [ADR-0005](docs/adr/0005-rag-integration.md) for the
original interface design).

| Item | Status | Notes |
|---|---|---|
| `BaseEmbeddingProvider` interface + registry | ✅ | `requisite/rag/base.py`, `requisite/rag/factory.py` |
| `BaseVectorStore` interface + registry | ✅ | |
| OpenAI embedding provider | ✅ | `OpenAIEmbeddingProvider` (`text-embedding-3-small` default) |
| Gemini embedding provider | ✅ | `GeminiEmbeddingProvider` (`gemini-embedding-001` default) |
| In-memory vector store (default, zero dependencies) | ✅ | `InMemoryVectorStore` — pure-Python cosine similarity |
| Pinecone vector store | ✅ | `PineconeVectorStore` — current `pinecone>=9.0` SDK (serverless `cloud`/`region`, not the older `environment=` API) |
| Weaviate vector store | ✅ | `WeaviateVectorStore` — current `weaviate-client>=4.0` v4 API, bring-your-own-vectors |
| Chunking (character-based, with overlap) | ✅ | `chunk_text()` — token-aware chunking is a follow-up, not this phase |
| `BaseRetriever` interface | ✅ | Independent of embeddings/vector store at the interface level, by design |
| Dense retriever (`Retriever`) | ✅ | Composes an embedding provider + vector store |
| Hybrid / BM25 retriever | ✅ | `BM25Retriever` (keyword-only), `HybridRetriever` (dense + BM25, fused via Reciprocal Rank Fusion) — see [ADR-0010](docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md) |
| Retriever exposed as a `CapabilityProvider` (`agent.requires("knowledge_base")`) | ✅ | `Retriever.as_tool()` |
| Re-ranking | ✅ | `BaseReranker` + `LLMReranker` (listwise, via `AI.chat_response(response_model=...)`) — see [ADR-0010](docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md) |
| Context compression | 📋 | |

## Prompt templates & conversation management

| Item | Status | Notes |
|---|---|---|
| `PromptTemplate` (single string, `{named}` variables) | ✅ | `requisite/prompts/template.py` |
| `ChatPromptTemplate` (renders to `list[Message]`) | ✅ | |
| `PromptTemplateRegistry` | ✅ | |
| Conversation history storage | ✅ | `BaseMemory` — see Memory section above |
| Conversation retention policy (truncation) | ✅ | `MessageCountPolicy` — see [ADR-0003](docs/adr/0003-prompt-templates-structured-logging-conversation-policy.md) |
| Conversation retention policy (LLM summarization) | ✅ | `SummarizingPolicy` |
| `Agent(conversation_policy=...)` integration | ✅ | |

## Plugin architecture

| Item | Status |
|---|---|
| Plugins register providers/tools/capabilities via the existing registries | ✅ — already possible today; no plugin-specific API needed since every registry is a plain, importable class |
| A `requisite-plugin-*` naming/discovery convention (entry points) | ✅ — `requisite.plugins.discover()` (entry-point group `"requisite.plugins"`), `requisite plugins` on the CLI — see [ADR-0017](docs/adr/0017-entry-point-plugin-discovery.md) |
| Official plugin listing / directory in the docs | 📋 |

## Telemetry

| Item | Status | Notes |
|---|---|---|
| Structured (JSON) logging | ✅ | `requisite.telemetry.JSONFormatter` + `configure_logging()` — opt-in, never automatic; see [ADR-0003](docs/adr/0003-prompt-templates-structured-logging-conversation-policy.md) |
| Tracing (e.g. OpenTelemetry spans around provider calls) | 📋 | |
| Metrics (request counts, latency, token usage aggregation) | 📋 | |

## CLI

| Item | Status |
|---|---|
| `requisite` CLI (scaffold a project, list registered providers/capabilities/agents, run a quick chat) | ✅ — see [ADR-0014](docs/adr/0014-cli.md); `requisite init`/`providers`/`capabilities`/`agents`/`chat`, installed as a console script (`requisite`) and via `python -m requisite` |

## Explicit non-goals (for now)

To keep scope honest, things we are **not** planning to build as part of
Requisite itself:

- Training or fine-tuning models.
- Hosting/serving infrastructure (Requisite calls provider APIs; it isn't
  an inference server).
- A hosted/SaaS version of anything in this repo.

If your use case needs one of these, Requisite is meant to compose with
tools that do — not replace them.

## How this roadmap changes

Open a [feature request](.github/ISSUE_TEMPLATE/feature_request.yml)
tagged with the relevant category. Anything that's a new implementation
of an existing interface (a provider, an orchestrator backend, a
capability resolver, a multi-agent strategy) can generally go straight to
a PR — see `CONTRIBUTING.md`. Anything that changes a public API shape,
or introduces a new top-level module (`mcp/`, `memory/`, `rag/`), should
be discussed in an issue first.

This roadmap itself is versioned in git — check `git log ROADMAP.md` for
how priorities have shifted over time.
