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
