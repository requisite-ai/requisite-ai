# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-08-08

### Added

- Three new providers: `OpenRouterProvider` and `TogetherProvider`
  (`provider="openrouter"` / `"together"` -- thin `OpenAIProvider`
  subclasses, same pattern as Groq/Azure OpenAI per ADR-0002, confirmed
  OpenAI-wire-compatible against each vendor's current docs) and
  `OllamaProvider` (`provider="ollama"` -- a full translation layer using
  the native `ollama` client, *not* the `OpenAIProvider`-subclass
  pattern, since Ollama's own OpenAI-compatible endpoint is documented
  by Ollama itself as experimental). `Settings` gains
  `openrouter_api_key`, `together_api_key`, `ollama_api_key`, and
  `ollama_host` fields.
- Two new RAG vector stores: `PineconeVectorStore`
  (`requisite.rag.vectorstores.pinecone`, `pip install
  requisite-ai[pinecone]`) and `WeaviateVectorStore`
  (`requisite.rag.vectorstores.weaviate`, `pip install
  requisite-ai[weaviate]`), both implementing `BaseVectorStore` and
  registered in `default_vector_store_registry` as `"pinecone"` /
  `"weaviate"`. Both verified against their current SDKs (`pinecone>=9.0`,
  `weaviate-client>=4.0`) -- Pinecone's index creation uses the current
  serverless `cloud`/`region` spec, not the older, now-removed
  `environment=` API; Weaviate uses the current v4 `WeaviateClient`
  collections API, not the older v3 `weaviate.Client(...)`. New
  `VectorStoreException(AIException)` for vector store operation
  failures, matching `ProviderException`'s shape.
- New optional dependency groups: `openrouter`, `together`, `ollama`,
  `pinecone`, `weaviate` (all included in the `all` extra).

All eight providers and all three vector stores are new implementations
of existing interfaces (`BaseProvider` / `BaseVectorStore`) -- no public
API shape changed for existing code.

## [0.5.1] - 2026-08-08

### Changed

- `.github/workflows/publish.yml` no longer trusts that `main`'s branch
  protection alone kept an unverified commit from reaching PyPI. It now
  invokes `ci.yml`'s full job graph (lint, type check, test matrix,
  build) as a reusable workflow (`workflow_call`) and gates the build +
  publish steps on it succeeding. This closes the gap where a manually
  triggered `workflow_dispatch` could target any branch/commit and skip
  verification entirely, regardless of what protection `main` has.

### Fixed

- Republished under a new version after a `0.5.0` upload attempt was
  deleted from PyPI: PyPI permanently blocks re-uploading a filename
  once used, even after deletion (`400 This filename was previously
  used by a file that has since been deleted`), so `0.5.0` can never be
  published again. No other code changes from `0.5.0` besides the CI
  gate above.

## [0.5.0] - 2026-08-08

### Added

- Proactive rate limiting for provider calls: `requisite.core.rate_limiter.RateLimiter`
  (sliding-window log, thread-safe `acquire()` / async-safe `aacquire()`),
  a new `RateLimitException(AIException)`, and two new opt-in `Settings`
  fields (`rate_limit_rpm`, `rate_limit_max_wait_seconds` / env
  `RATE_LIMIT_RPM`, `RATE_LIMIT_MAX_WAIT_SECONDS`). `AI` and `Agent` both
  gain a `rate_limiter=` constructor parameter -- pass the *same*
  `RateLimiter` instance to several `Agent`/`AI` objects that draw on the
  same underlying API key/quota to share one real budget across them,
  which a single `Settings.rate_limit_rpm` value alone does not do (each
  instance would otherwise build its own private limiter). Fixes the
  free-tier Gemini `429 RESOURCE_EXHAUSTED` errors surfaced when running
  `examples/workflow_example.py` -- that example now constructs one
  shared `RateLimiter` for its four agents. See
  `docs/adr/0008-rate-limiting.md` for the full design rationale.

## [0.4.1] - 2026-08-07

### Fixed

- CI's type-check and test jobs install `mcp` via the unbounded
  `mcp>=1.28` constraint in the `mcp`/`all` extras, so they picked up
  `mcp` 2.0.0 -- a breaking rewrite (restructured package layout,
  `CallToolResult.isError`/`structuredContent` renamed to
  `is_error`/`structured_content`, `streamablehttp_client` removed in
  favor of `streamable_http_client`) -- and failed `mypy` with four
  `attr-defined` errors in `requisite/mcp/client.py`, with zero code
  changes on our side. Same shape as the ruff 0.16.0 incident
  (0.3.2's fix): an unbounded dependency constraint let a breaking
  upstream release reach CI unpinned. Fixed by capping `mcp` to
  `>=1.28,<2.0` in `pyproject.toml` (`mcp` and `all` extras) and
  `requirements.txt`, verified against the real `mcp` 2.0.0 wheel
  (downloaded and inspected, not assumed) to confirm the scope of the
  break before deciding to pin rather than migrate. Migrating
  `requisite/mcp/client.py` to the `mcp` 2.x API is tracked as a
  separate, deliberate change -- see `ROADMAP.md`.

## [0.4.0] - 2026-08-07

### Added

- Three new multi-agent orchestration strategies on the `native`
  orchestrator: `Workflow().reflection()`, `.planner()`, and
  `.supervisor()`. `reflection` takes a single agent that critiques and
  revises its own output over `max_rounds` rounds (default 3),
  optionally stopping early. `planner`/`supervisor` take a coordinating
  agent (`steps[0]`) plus named workers (`steps[1:]`): `planner`
  decomposes the task into an ordered plan up front and executes it;
  `supervisor` delegates one subtask at a time, deciding each round
  whether to delegate again or finish (`max_rounds` default 6, raising
  `AgentException` if exhausted without a final answer). Coordinator
  decisions use `AI.chat(response_model=...)` for structured routing/
  planning rather than free-text parsing. See
  `docs/adr/0007-multi-agent-orchestration-strategies.md` for the full
  design rationale and deliberate scope cuts. These strategies are
  implemented on the `native` orchestrator only; `LangGraphOrchestrator`
  continues to only support `sequential`.

## [0.3.4] - 2026-08-07

### Fixed

- `examples/mcp_example.py` hardcoded `/tmp` as the filesystem MCP
  server's allowed directory. `@modelcontextprotocol/server-filesystem`
  validates that directory at startup and exits before completing the
  MCP handshake if it doesn't exist -- on Windows, where `/tmp` isn't a
  valid path, this surfaced as `mcp.shared.exceptions.McpError:
  Connection closed` during `session.initialize()`, not as an obviously
  path-related error. Fixed by using `tempfile.gettempdir()` instead, and
  by writing a known demo file into it so the agent has something real
  to read rather than guessing a filename that may not exist.

## [0.3.3] - 2026-08-07

### Fixed

- `GeminiProvider` failed multi-turn tool-calling conversations with
  `400 INVALID_ARGUMENT: Function call is missing a thought_signature`,
  because it read responses via the `response.text` /
  `response.function_calls` convenience properties, both of which
  discard the `thought_signature` field Gemini now requires echoed back
  verbatim on `function_call` parts across turns (the same discard also
  caused a noisy but non-fatal "there are non-text parts in the
  response" warning). Fixed by walking
  `response.candidates[0].content.parts` directly in
  `_to_chat_response`, and by echoing a captured signature back onto the
  reconstructed `function_call` part in `_build_contents_and_system`.
  See `docs/adr/0006-gemini-thought-signature.md`.
- `ToolCall` gained an optional `provider_data: Any` field to carry this
  kind of opaque, provider-specific replay data. It's `None` for every
  other provider and ignored by them.

## [0.3.2] - 2026-07-28

### Fixed

- CI's lint job installed `ruff` unpinned (`pip install ruff`), bypassing
  the version pin already set in `pyproject.toml`'s `dev` extra and
  `requirements.txt`. When Ruff 0.16.0 (released July 23, 2026) expanded
  its default lint rule set from 59 to 413 rules, the lint job broke
  overnight with no code change on our side — 295 findings, mostly
  `UP045`/`UP007` (pyupgrade's `Optional`/`Union` -> `X | None`/`X | Y`
  suggestions), plus `RUF100`/`UP037`/`UP035`/`SIM117`. Fixed by having
  the lint job install from `pyproject.toml`'s pinned `dev` extra
  (`pip install -e ".[dev]"`) instead of a bare `pip install ruff`, so
  there's exactly one source of truth for the pinned version across all
  three places it's declared (`pyproject.toml`, `requirements.txt`, CI).
- Documented the pinning policy in `DEVELOPMENT.md`: dev tool versions
  (`ruff`, `mypy`) are pinned exactly, bumped deliberately in their own
  PR, never left to drift via an unpinned install.
- `examples/rag_example.py`'s docstring no longer claims "doesn't
  hardcode a provider" after the example was simplified to use Gemini
  directly for both embeddings and chat.

## [0.3.1] - 2026-07-26

### Fixed

- **`AI.chat`/`chat_response`/`achat`/`achat_response`'s `tools=` parameter
  now accepts `@tool`-decorated functions and plain functions directly**,
  not just `Tool` instances — matching what `Agent(tools=...)` already
  did, and what the README's own tool-calling example showed. Previously,
  `ai.chat(prompt, tools=[my_decorated_function])` raised an
  `AttributeError` at the provider layer (`to_openai_schema` doesn't
  exist on a plain function, only on the `.tool` attached to it) — a real
  runtime bug affecting the framework's documented public API, not just
  an example. Fixed by resolving each item via
  `requisite.tools.registry.resolve_tool_like` before dispatching to the
  provider, same as `ToolRegistry` already did.
- `@tool`'s type signature now returns a proper `Protocol`
  (`ToolFunction[P, R]`) declaring both the original call signature and
  the attached `.tool: Tool` attribute, instead of just returning the
  original function type unchanged. This makes `.tool` access and
  passing a decorated function to `tools=[...]` both type-check
  correctly under `mypy --strict`, rather than only working at runtime
  with no static verification.
- Fixed a test isolation bug in `tests/test_settings.py`:
  `Settings(_env_file=None)` only disables reading the `.env` *file* —
  it does not and should not block real OS environment variables. A test
  only cleared 2 of 13 `Settings`-relevant env vars, so a real
  `DEFAULT_PROVIDER` (or similar) set in the shell -- commonly injected
  by VS Code's Python extension loading `.env` into the integrated
  terminal / debug session -- could make the test fail on some machines
  despite passing in CI. Fixed with an `autouse` fixture clearing every
  `Settings` field's env var before each test.
- `examples/rag_example.py` no longer hardcodes OpenAI for embeddings or
  chat — it now picks the embedding provider based on whichever API key
  is actually configured, and lets the agent's chat provider default to
  `Settings.default_provider` rather than assuming `"openai"`.

## [0.3.0] - 2026-07-23

### Added

- RAG integration: `BaseEmbeddingProvider`, `BaseVectorStore`, and
  `BaseRetriever` interfaces (RAG decomposes into independent extension
  points, per ADR-0001), plus a shipped `Retriever` (dense retrieval)
  composing an embedding provider and a vector store.
- `OpenAIEmbeddingProvider` (`text-embedding-3-small` default) and
  `GeminiEmbeddingProvider` (`gemini-embedding-001` default).
- `InMemoryVectorStore` — zero-dependency default, pure-Python cosine
  similarity, mirroring `InProcessMemory`'s role for conversation memory.
  Pinecone and Weaviate integrations are a deliberate scope cut for this
  release, not yet implemented — `.env.example` already reserves their
  keys.
- `chunk_text()` — character-based chunking with overlap; a token-aware
  chunker is a documented follow-up, not this release.
- `Retriever.as_tool()` — bridges a retriever into `CapabilityRegistry`
  exactly like an MCP server or a native tool:
  `agent.requires("knowledge_base")`. This was an explicit design
  decision (over a new `Agent(retriever=...)` parameter), reusing the
  existing capability mechanism.
- ADR-0005, documenting the interface decomposition, the chunking
  approach, the in-memory-default/Pinecone-Weaviate-deferred decision,
  and the capability-bridge design.
- `ROADMAP.md`: added an Evaluation section (not yet implemented,
  logged following external architecture review feedback).

## [0.2.0] - 2026-07-17

### Added

- MCP (Model Context Protocol) client integration: `BaseMCPClient`
  interface (specified in ADR-0001) implemented as `MCPClient`, wrapping
  the official `mcp` SDK (1.28+). Supports both `MCPClient.stdio(...)`
  (local subprocess) and `MCPClient.http(...)` (remote, Streamable HTTP)
  from day one, verified against real MCP servers on both transports.
- `MCPClientRegistry` — keyed by server name, mirrors every other
  registry's shape.
- `BaseMCPClient.register_as_capability(...)` — bridges an MCP server's
  tools into `CapabilityRegistry`, so `agent.requires("github")` can
  resolve to an MCP server exactly like it resolves to a native tool.
  Verified this holds with a real server: `Agent` cannot tell the
  difference.
- `mcp` added as an optional dependency (`pip install requisite-ai[mcp]`).
- ADR-0004, documenting the transport decisions, the per-call (not
  persistent-session) connection model and why, and result-handling
  (`structuredContent` preferred over text, verified against a real
  server's actual response shape).
- Decided (not yet implemented) the RAG architecture direction: an
  in-memory default vector store plus Pinecone/Weaviate as optional
  integrations, with retrievers exposed to agents as a
  `CapabilityProvider` rather than a new `Agent` constructor parameter --
  tracked in `ROADMAP.md`, full design to land in ADR-0005 alongside
  implementation.

## [0.1.0] - 2026-07-13

### Added

- `PromptTemplate` and `ChatPromptTemplate` — reusable, `{named}`-variable
  prompts; `ChatPromptTemplate.format_messages()` renders directly to
  `list[Message]`. `PromptTemplateRegistry` for naming and reuse.
- `requisite.telemetry.JSONFormatter` + `configure_logging()` — opt-in
  structured (JSON) logging for the `requisite` logger tree, never
  invoked automatically by the framework. A representative set of
  registration/resolution log calls across the registries now pass
  structured `extra=` fields.
- `BaseConversationPolicy`, `MessageCountPolicy`, and `SummarizingPolicy`
  in `requisite.memory.policies` — conversation retention/truncation,
  wired into `Agent(conversation_policy=...)`. Applied once per
  `run()`/`arun()` call, independent of whether `memory` is configured.
- `PromptException` added to the exception hierarchy.
- `Settings.log_format` (``"plain"`` or ``"json"``) — a stored preference
  only; does not itself configure logging (see ADR-0003).
- ADR-0003, documenting the design decisions above.
- Three new providers: `AnthropicProvider` (native structured output via
  `messages.parse`, proper tool-use/tool-result multi-turn round-trip),
  `GroqProvider`, and `AzureOpenAIProvider` (current v1 GA API — no
  dated `api-version` string). Groq and Azure OpenAI are implemented as
  `OpenAIProvider` subclasses, confirmed wire-compatible against both
  vendors' current docs -- see
  [ADR-0002](docs/adr/0002-provider-kwargs-and-memory-integration.md).
- `BaseMemory` interface + `InProcessMemory` default + `MemoryRegistry`,
  matching the shape specified in ADR-0001. Wired into `Agent(memory=...,
  session_id=...)`.
- `Settings.provider_kwargs(name)` — a generic mechanism for
  provider-specific constructor arguments (used today for Azure OpenAI's
  `azure_endpoint`), documented in ADR-0002.
- `.env.example` expanded with keys for the new providers, plus reserved
  placeholders for planned integrations (GitHub, Hugging Face, AWS, Azure
  general-purpose credentials, Pinecone, Weaviate).
- `docs/adr/` — Architecture Decision Records, starting with ADR-0001
  (core interfaces, dependency flow, extension points, plugin discovery,
  configuration model, public API principles, `requisite-core` vs.
  optional-integrations boundary) and ADR-0002 (this release's decisions).
- `CapabilityRegistry` and `Agent.requires(...)`: declare a named
  capability (e.g. `"weather"`, `"internet_search"`, `"filesystem"`)
  instead of binding to one specific tool implementation. Resolution
  picks the highest-priority currently-available provider; ships with
  three keyless default resolvers.
- Project renamed to **Requisite** (PyPI: `requisite-ai`).
- Tool calling: `@tool` decorator, `Tool`, `ToolRegistry`, automatic
  JSON Schema derivation from function signatures. Wired into both the
  OpenAI and Gemini providers, including proper multi-turn tool-call /
  tool-result message round-tripping.
- Structured output: `ai.chat(prompt, response_model=SomeModel)`.
- `Agent` and `AgentRegistry`: an `AI` equipped with tools/skills and an
  autonomous tool-calling loop (sync `run()` and async `arun()`).
- `BaseSkill` and `SkillRegistry`: reusable, higher-level capabilities
  that expose themselves to the model as tools automatically.
- `Workflow`: compose agents into sequential or parallel multi-agent
  pipelines, with a `"native"` (pure Python) and `"langgraph"`
  orchestrator backend, switchable via `.use_langgraph()` / `.use_native()`.
  `"crewai"` / `"autogen"` are registered as clear "not yet implemented"
  placeholders.
- Initial release: provider-agnostic `AI` facade, `BaseProvider`
  interface, `OpenAIProvider` (openai>=1.35 client-based SDK) and
  `GeminiProvider` (google-genai unified SDK), `ProviderRegistry`,
  `pydantic-settings`-based `Settings`, `Message`/`ChatResponse` models,
  and the `AIException` hierarchy.

[Unreleased]: https://github.com/requisite-ai/requisite-ai/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/requisite-ai/requisite-ai/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/requisite-ai/requisite-ai/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/requisite-ai/requisite-ai/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/requisite-ai/requisite-ai/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/requisite-ai/requisite-ai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/requisite-ai/requisite-ai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/requisite-ai/requisite-ai/releases/tag/v0.1.0
