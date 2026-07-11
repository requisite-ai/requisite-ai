# Contributing to Requisite

Thanks for considering a contribution! Start here, then branch out to the
doc that covers your specific question:

- **This file** — setup, running checks, the extension-point walkthroughs
  (new provider / orchestrator / capability), PR process.
- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — how the framework fits
  together and why (the interface + registry pattern, request flows,
  design decisions). Read this before writing non-trivial code.
- **[`DEVELOPMENT.md`](DEVELOPMENT.md)** — the detailed engineering
  reference: docstring format, testing philosophy, logging/error-handling
  conventions, versioning & deprecation policy.
- **[`ROADMAP.md`](ROADMAP.md)** — what's shipped, planned, or explicitly
  out of scope. Check here before proposing something large.

## Getting set up

```bash
git clone https://github.com/requisite-ai/requisite-ai.git
cd requisite-ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev,all]"
cp .env.example .env  # fill in keys only if you're running the live examples
```

Requires Python 3.10+.

## Running the checks

```bash
pytest                 # tests -- no network access or API keys required
ruff check .           # lint
ruff format --check .  # formatting
mypy requisite         # type-check, strict mode
```

All four run in CI (`.github/workflows/ci.yml`) and must pass before a PR
merges. Run a single test file or test while iterating:

```bash
pytest tests/test_agents.py -v
pytest tests/test_agents.py::test_agent_executes_tool_and_returns_final_answer
```

See `DEVELOPMENT.md` for *why* the test suite is structured this way
(short version: every test fakes the provider SDK or uses an in-memory
fake provider — no real network calls, ever).

## Project layout

```
requisite/
├── core/           # Message, ChatResponse, ToolCall, ... + AIException hierarchy
├── config/         # Settings (pydantic-settings, reads .env)
├── providers/      # BaseProvider + OpenAI/Gemini + ProviderRegistry
├── tools/          # Tool, @tool, ToolRegistry, JSON Schema derivation
├── skills/         # BaseSkill, SkillRegistry
├── capabilities/   # CapabilityRegistry -- agent.requires("weather") resolution
├── agents/         # Agent (tool-calling loop) + AgentRegistry
├── orchestrators/  # BaseOrchestrator + native/langgraph + OrchestratorRegistry
├── workflows/      # Workflow -- the multi-agent facade
└── ai.py           # The `AI` facade
```

Each sub-package has a short module docstring explaining its role — read
that first when orienting yourself in a new area. `ARCHITECTURE.md` has
the full dependency diagram and request-flow walkthroughs.

## Adding a new provider

1. Implement `requisite.providers.base.BaseProvider` (`chat`, `achat`,
   `stream`, `astream`, `name`). Use `requisite/providers/openai_provider.py`
   as a reference — note the lazy SDK import inside `_get_client()`, so
   importing `requisite` never requires every provider's SDK to be installed.
2. Support `tools=` and `response_model=` on `chat`/`achat` if the
   provider's API supports function calling / structured output — translate
   to/from the provider's wire format in that one file only.
3. Register it: `default_registry.register("your_provider", YourProvider)`.
4. Add tests in `tests/test_providers.py` faking the SDK via `sys.modules`
   injection (see the existing `fake_openai_module` / `fake_genai_module`
   fixtures for the pattern) — no real API key or network access.
5. Add it to `requirements.txt` / `pyproject.toml` `[project.optional-dependencies]`,
   a row in `ROADMAP.md`'s Providers table, and a short section in `README.md`.

## Adding a new orchestration backend

1. Implement `requisite.orchestrators.base.BaseOrchestrator` (`run`, `arun`,
   `name`). See `orchestrators/native.py` (pure Python) and
   `orchestrators/langgraph_orchestrator.py` (wraps an external framework)
   for the two flavors of reference implementation.
2. Register it: `default_registry.register("your_backend", YourOrchestrator)`
   in `orchestrators/factory.py`, or from your own plugin package.
3. `Workflow` never needs to change — add a `workflow.use_your_backend()`
   convenience method only if you want the discoverable, chainable style
   used by `use_langgraph()` / `use_native()`.
4. Add tests exercising both the `"sequential"` and `"parallel"` strategies
   if applicable (see `tests/test_workflows.py`).

## Adding a new capability resolver

This is often the easiest, highest-leverage contribution: a new default
resolver in `requisite/capabilities/resolvers.py`, or a third-party package
that registers its own.

```python
from requisite.capabilities import default_registry

def my_search_tool(query: str) -> str:
    """A real web search implementation."""
    ...

default_registry.register(
    "internet_search",
    my_search_tool,
    provider_name="my-search-provider",
    priority=10,  # outranks the built-in DuckDuckGo fallback when available
    is_available=lambda: bool(os.environ.get("MY_SEARCH_API_KEY")),
)
```

If you're adding a *new* capability name (not competing on an existing one
like `"weather"`), open an issue first to discuss the name — capability
names are a shared namespace across the ecosystem, so bikeshedding them in
public is worth the friction.

## Commit messages & PRs

- Keep commits focused; prefer several small, reviewable commits over one
  large one.
- PR description should state: what changed, why, and how you tested it
  (or point at the new/updated tests). The PR template
  (`.github/PULL_REQUEST_TEMPLATE.md`) walks you through this.
- Breaking changes to any public API (anything importable from `requisite`
  directly, or from a sub-package without a leading underscore) must be
  called out explicitly in the PR description and in `CHANGELOG.md` under
  an `## Unreleased` → `### Changed` (or `### Removed`) heading. See
  `DEVELOPMENT.md`'s versioning & deprecation section for the full policy.
- CI (lint, format, type-check, tests) must be green before merge.

## Reporting bugs & requesting features

Please use the issue templates (`.github/ISSUE_TEMPLATE/`) — they ask for
the minimum needed to act on a report (repro steps, environment, expected
vs. actual behavior for bugs; problem statement, proposed shape, and
roadmap category for features). For security issues, see `SECURITY.md`
instead of opening a public issue.

## Maintainer setup (one-time, repo admin only)

A couple of CI jobs need repository secrets/config that a code contributor
doesn't need to touch, but a new maintainer setting up the repo should know
about:

- **`CODECOV_TOKEN`** (repo secret) — required for the coverage upload
  step in `.github/workflows/ci.yml` to work. Get it from
  [codecov.io](https://codecov.io) after linking the repo.
- **PyPI trusted publisher** — `.github/workflows/publish.yml` publishes
  releases via OIDC, not a stored API token. One-time setup on pypi.org:
  add a trusted publisher for this repo with workflow `publish.yml` and
  environment `pypi`. Details in the comment header of that workflow file.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you're expected to uphold it.
