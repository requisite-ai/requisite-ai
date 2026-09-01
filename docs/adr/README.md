# Architecture Decision Records

An ADR captures a significant architectural decision, the context that
drove it, and its consequences — recorded at the time the decision is
made, not reconstructed later. Requisite uses them for decisions that
would be expensive to reverse or relitigate from scratch (public API
shape, package boundaries, extension mechanisms), as a complement to
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) (which describes the *current*
architecture) and [`ROADMAP.md`](../../ROADMAP.md) (what's planned).

## When to write one

Open a new ADR for anything that:

- Changes or fixes the shape of a public interface (`BaseProvider`,
  `BaseOrchestrator`, a new core interface like `BaseMemory`).
- Adds or removes a package boundary (e.g. splitting `requisite-core`
  from an integration package).
- Establishes a convention that future contributions are expected to
  follow (a plugin discovery mechanism, a configuration model).
- Reverses a previous ADR.

Small implementation choices, bug fixes, and new implementations of an
*existing* interface (a new provider, a new capability resolver) don't
need one — `CONTRIBUTING.md` already covers those.

## Process

1. Copy [`template.md`](template.md) to `NNNN-short-title.md` (next
   sequential number, kebab-case title).
2. Open a PR with status `Proposed`. Discussion happens on the PR.
3. On merge, update the status to `Accepted` (or `Rejected`, with the
   record kept for history).
4. If a later ADR reverses an earlier one, mark the old one
   `Superseded by ADR-NNNN` rather than deleting it — the history of
   *why* a decision changed is often as valuable as the decision itself.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-core-architecture-and-interfaces.md) | Core architecture, interfaces, and extension model | Accepted |
| [0002](0002-provider-kwargs-and-memory-integration.md) | Provider-specific configuration, OpenAI-compatible providers, and Memory integration | Accepted |
| [0003](0003-prompt-templates-structured-logging-conversation-policy.md) | Prompt templates, structured logging, and conversation policies | Accepted |
| [0004](0004-mcp-integration.md) | MCP client integration | Accepted |
| [0005](0005-rag-integration.md) | RAG integration | Accepted |
| [0006](0006-gemini-thought-signature.md) | Gemini thought_signature echoing | Accepted |
| [0007](0007-multi-agent-orchestration-strategies.md) | Multi-agent orchestration strategies: reflection, planner, supervisor | Accepted |
| [0008](0008-rate-limiting.md) | Proactive rate limiting for provider calls | Accepted |
| [0009](0009-streaming-tool-calls.md) | Streaming + tool calls together | Accepted |
| [0010](0010-hybrid-bm25-retrieval-and-reranking.md) | Hybrid/BM25 retrieval and re-ranking | Accepted |
| [0011](0011-critic-and-consensus-strategies.md) | Critic and consensus multi-agent strategies | Accepted |
| [0012](0012-debate-and-map-reduce-strategies.md) | Debate and map-reduce multi-agent strategies | Accepted |
| [0013](0013-hierarchical-strategy.md) | Hierarchical multi-agent strategy | Accepted |
| [0014](0014-cli.md) | `requisite` CLI: scaffolding, registry introspection, and quick chat | Accepted |
| [0015](0015-mcp-server-integration.md) | MCP server integration: exposing Requisite as an MCP server | Accepted |
| [0016](0016-langgraph-branching.md) | LangGraph backend: branching/conditional graphs | Accepted |
| [0017](0017-entry-point-plugin-discovery.md) | Entry-point plugin discovery | Accepted |
| [0018](0018-tree-of-thoughts-strategy.md) | Tree-of-thoughts multi-agent strategy | Accepted |
| [0019](0019-graph-execution-strategy.md) | General graph execution strategy | Accepted |
| [0020](0020-github-capability-resolver.md) | `github` default capability resolver | Accepted |
| [0021](0021-opentelemetry-tracing-and-metrics.md) | OpenTelemetry tracing and metrics | Accepted |
| [0022](0022-vector-memory.md) | Vector-database-backed memory | Accepted |
| [0023](0023-mcp-default-capability-providers.md) | First-party MCP servers as default capability providers | Accepted |
| [0024](0024-rag-context-compression.md) | RAG context compression | Accepted |
| [0025](0025-mcp-2x-migration.md) | Migrate to `mcp` 2.x | Accepted |
| [0026](0026-mcp-resource-prompt-discovery.md) | MCP resource / prompt discovery | Accepted |
| [0027](0027-crewai-autogen-orchestrator-backends.md) | CrewAI and AutoGen orchestrator backends | Accepted |
| [0028](0028-langgraph-reflection-strategy.md) | LangGraph backend: reflection strategy | Accepted |
| [0029](0029-langgraph-hierarchical-graph-strategies.md) | LangGraph backend: hierarchical and graph strategies | Accepted |
| [0030](0030-mcp-persistent-session-mode.md) | MCP persistent session mode | Accepted |
| [0031](0031-code-review-fixes.md) | Code review, adversarial testing, and fixes across the whole codebase | Accepted |
| [0032](0032-langgraph-parallel-consensus-map-reduce-strategies.md) | LangGraph backend: parallel, consensus, and map-reduce strategies | Accepted |
| [0033](0033-langgraph-critic-debate-strategies.md) | LangGraph backend: critic and debate strategies | Accepted |
| [0034](0034-langgraph-tree-of-thoughts-strategy.md) | LangGraph backend: tree-of-thoughts strategy | Accepted |
| [0035](0035-langgraph-planner-strategy.md) | LangGraph backend: planner strategy | Accepted |
| [0036](0036-reflexion-strategy.md) | Reflexion multi-agent strategy | Accepted |
