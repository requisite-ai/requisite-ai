# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-07-13

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

## [0.4.0] - 2026-07-12

### Added

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

## [0.3.0] - 2026-07-11

### Added

- `CapabilityRegistry` and `Agent.requires(...)`: declare a named
  capability (e.g. `"weather"`, `"internet_search"`, `"filesystem"`)
  instead of binding to one specific tool implementation. Resolution
  picks the highest-priority currently-available provider; ships with
  three keyless default resolvers.
- Project renamed to **Requisite** (PyPI: `requisite-ai`).

## [0.2.0] - 2026-07-11

### Added

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

## [0.1.0] - 2026-07-09

### Added

- Initial release: provider-agnostic `AI` facade, `BaseProvider`
  interface, `OpenAIProvider` (openai>=1.35 client-based SDK) and
  `GeminiProvider` (google-genai unified SDK), `ProviderRegistry`,
  `pydantic-settings`-based `Settings`, `Message`/`ChatResponse` models,
  and the `AIException` hierarchy.

[Unreleased]: https://github.com/requisite-ai/requisite-ai/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/requisite-ai/requisite-ai/releases/tag/v0.1.0
