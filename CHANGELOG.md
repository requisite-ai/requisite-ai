# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/requisite-ai/requisite-ai/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/requisite-ai/requisite-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/requisite-ai/requisite-ai/releases/tag/v0.1.0
