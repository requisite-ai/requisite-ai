# Roadmap

Requisite is deliberately scoped: **a provider-agnostic AI application
framework with pluggable execution engines, where every layer is an
interface + implementation(s) + a plain registry.** Everything below is a
module riding that same pattern — nothing on this roadmap requires
changing the core shape of the framework to add.

Status legend: ✅ Shipped · 🚧 In progress · 📋 Planned · 💭 Under discussion

## Core

| Item | Status |
|---|---|
| `AI` facade — chat, structured output, streaming, async | ✅ |
| `BaseProvider` interface + `ProviderRegistry` | ✅ |
| `Settings` (pydantic-settings, `.env`) | ✅ |
| `AIException` hierarchy | ✅ |
| `py.typed` / PEP 561 typed distribution | ✅ |

## Providers

| Item | Status |
|---|---|
| OpenAI (`openai>=1.35` client SDK) | ✅ |
| Gemini (`google-genai` unified SDK) | ✅ |
| Anthropic Claude | 📋 |
| Azure OpenAI | 📋 |
| Ollama (local models) | 📋 |
| Groq | 📋 |
| OpenRouter | 📋 |
| Together AI | 📋 |

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
| Streaming + tool calls together (partial tool-call deltas) | 📋 |
| Parallel tool calls in a single turn (multiple calls, one round-trip) | 📋 — partially works today (providers return a list of `ToolCall`s); `Agent`'s loop executes them sequentially, not concurrently |

## Capabilities (`agent.requires(...)`)

| Item | Status |
|---|---|
| `CapabilityRegistry` — priority + availability-based resolution | ✅ |
| Default resolvers: `filesystem`, `weather`, `internet_search` | ✅ |
| `Agent.requires(...)` | ✅ |
| `github` default resolver (public, unauthenticated REST API) | 📋 |
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
| `langgraph` backend: branching / conditional graphs | 📋 |
| Supervisor strategy (a coordinating agent delegates to others) | 📋 |
| Planner strategy | 📋 |
| Reflection strategy (agent critiques and revises its own output) | 📋 |
| Debate / critic / consensus strategies | 📋 |
| Hierarchical strategy | 📋 |
| Map-reduce strategy | 📋 |
| Tree-of-thoughts strategy | 📋 |
| General graph execution (arbitrary DAGs, not just linear/parallel) | 📋 |
| CrewAI orchestrator backend | 📋 — registered today as a clear "not yet implemented" placeholder |
| AutoGen orchestrator backend | 📋 — same |

Each new strategy is a `_run_<strategy>` / `_arun_<strategy>` pair on
`NativeOrchestrator` (or an equivalent on another backend) — see
`ARCHITECTURE.md` for how strategies plug in without changing `Workflow`.

## MCP (Model Context Protocol)

| Item | Status |
|---|---|
| MCP client integration (consume remote/local MCP tool servers as capabilities) | 📋 |
| MCP server integration (expose Requisite tools/agents as an MCP server) | 📋 |
| First-party MCP servers as capability providers (filesystem, GitHub, databases) | 📋 |

Planned shape: an MCP-backed tool becomes just another `CapabilityProvider`
in `CapabilityRegistry` — `agent.requires("github")` shouldn't need to know
or care whether `"github"` is resolved by a native tool or an MCP server.

## Memory

| Item | Status |
|---|---|
| `BaseMemory` interface + registry | 📋 |
| Conversation memory (in-process) | 📋 |
| Redis-backed memory | 📋 |
| SQLite-backed memory | 📋 |
| Vector-database-backed memory | 📋 |
| Knowledge-graph-backed memory | 💭 |

## RAG

| Item | Status |
|---|---|
| `BaseEmbeddingProvider` + `BaseVectorStore` interfaces | 📋 |
| Chunking strategies | 📋 |
| Retrievers (dense, hybrid) | 📋 |
| Re-ranking | 📋 |
| Context compression | 📋 |

## Prompt templates & conversation management

| Item | Status |
|---|---|
| Prompt template objects (beyond raw `system_prompt=` strings) | 📋 |
| Conversation/session objects wrapping `Message` history + persistence | 📋 |

## Plugin architecture

| Item | Status |
|---|---|
| Plugins register providers/tools/capabilities via the existing registries | ✅ — already possible today; no plugin-specific API needed since every registry is a plain, importable class |
| A `requisite-plugin-*` naming/discovery convention (entry points) | 📋 |
| Official plugin listing / directory in the docs | 📋 |

## CLI

| Item | Status |
|---|---|
| `requisite` CLI (scaffold a project, list registered providers/capabilities/agents, run a quick chat) | 📋 |

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
