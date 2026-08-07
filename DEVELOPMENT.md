# Development Standards

This document is the detailed reference for *how* code should look and
behave in Requisite. `CONTRIBUTING.md` covers the *process* (setup, PR
flow); this covers the *conventions*. When the two overlap, this file is
the source of truth.

## Tooling

```bash
ruff check .          # lint
ruff format .          # format (run before committing)
mypy requisite         # type-check, strict mode, must be clean
pytest                 # must pass, no network access required
```

All four run in CI (`.github/workflows/ci.yml`) and are required to pass
before merge. Run them locally before pushing — CI failures on formatting
alone are a waste of a review cycle.

**Dev tool versions (`ruff`, `mypy`) are pinned exactly** (`ruff==X.Y.Z`,
not `ruff>=X.Y.Z`) in `pyproject.toml`'s `dev` extra, `requirements.txt`,
and every CI job installs from one of those rather than a bare
`pip install ruff`. This is deliberate, not an oversight: ruff 0.16.0
(July 2026) expanded its *default* lint rule set from 59 to 413 rules in
one release, and CI jobs that did `pip install ruff` unpinned broke
overnight with no code change on our side. Bump a pinned tool version
deliberately, in its own PR, after checking what changed — never let it
drift silently via an unpinned install.

## Typing

- **Type hints on everything**, including private helpers and test
  fixtures. `mypy requisite` runs in `strict = true` mode (see
  `pyproject.toml`) and must report zero errors.
- The package ships a `py.typed` marker (PEP 561) — type hints are part of
  the public contract for downstream users, not just internal
  documentation. Treat a type hint change as an API change.
- Optional third-party SDKs (`openai`, `google-genai`, `langgraph`) are
  exempted from `mypy`'s missing-stub errors via
  `[[tool.mypy.overrides]]` in `pyproject.toml` — because they're imported
  lazily and aren't hard dependencies. Don't broaden that override to
  cover framework code; it should only ever list optional SDK module
  names.
- Prefer `from __future__ import annotations` (present at the top of every
  module) plus `Optional[X]` / `Union[X, Y]` over `X | None` /`X | Y` for
  now, to keep a consistent style across the codebase — this may change
  once the minimum supported Python version rules out the older syntax
  being ambiguous in tooling.

## Docstrings

Every public class, method, and function gets a docstring in this shape
(NumPy-style, matching the existing codebase):

```python
def resolve(self, capability: str) -> Tool:
    """One-line summary in the imperative mood.

    Longer description if needed -- what it does, any non-obvious
    behavior, what "available" means in this context, etc.

    Parameters
    ----------
    capability:
        What this parameter is, including format/units if relevant.

    Returns
    -------
    Tool
        What's returned and what it represents.

    Raises
    ------
    requisite.core.exceptions.CapabilityException
        The specific condition that triggers this.

    Examples
    --------
    >>> registry.resolve("weather")  # doctest: +SKIP
    """
```

- `Examples` blocks should be runnable doctests where possible. Add
  `# doctest: +SKIP` for anything needing a live network call or a real
  API key — the example is still valuable as documentation even if it's
  not executed in CI.
- Private helpers (leading underscore) get at least a one-line docstring
  explaining *why*, not just restating the name.

## Testing philosophy

**No test may require network access or a real API key.** This is
enforced by convention, not tooling, so take it seriously:

- **Provider SDKs are faked via `sys.modules` injection.** See
  `tests/test_providers.py`'s `fake_openai_module` / `fake_genai_module`
  fixtures — they build a minimal fake module tree matching just enough
  of the real SDK's shape (`OpenAI`, `.chat.completions.create`, etc.) to
  exercise the provider's translation logic, then `monkeypatch.setitem`
  it into `sys.modules` before importing the provider module.
- **Higher-level facades (`AI`, `Agent`, `Workflow`) are tested against a
  fully in-memory fake `BaseProvider`.** See `FakeProvider` in
  `tests/test_ai.py`, `ScriptedToolCallingProvider` in
  `tests/test_agents.py`, and `EchoProvider` in `tests/test_workflows.py`
  for three different flavors of "scripted fake" depending on what the
  test needs to exercise (deterministic single response, a tool-call then
  final-answer sequence, or content-transforming echo for pipeline tests).
- **Every new public method needs a test covering at least: the happy
  path, and the primary error path** (what exception is raised, and
  when). Registries in particular should each have a test for "unknown
  name raises the right exception."
- Use `pytest.raises(SpecificException)`, never a bare
  `pytest.raises(Exception)` — the whole point of the exception hierarchy
  is that callers can and should catch specifically.
- Async tests use `@pytest.mark.asyncio` (config: `asyncio_mode = "auto"`
  in `pyproject.toml`, so `pytest-asyncio` picks up async test functions
  automatically — the marker is present in this codebase for clarity but
  isn't strictly required).

## Logging

- Never use `print()`. Use the standard library `logging` module.
- Every module gets its own logger, namespaced under `"requisite."`:
  `logger = logging.getLogger("requisite.providers.openai")`. This lets
  downstream applications configure log levels per-subsystem
  (`logging.getLogger("requisite.agents").setLevel(logging.DEBUG)`).
- Log registration/resolution events at `DEBUG` (e.g. "registered provider
  'openai'", "resolved capability 'weather' -> 'open-meteo'") — useful for
  debugging a misconfigured registry, too noisy for normal operation.
- When a log event has an obvious structured shape (an entity name, a
  count, a decision), pass it via `extra={...}` on the log call, e.g.
  `logger.debug("Resolved capability '%s' -> '%s'", capability, name, extra={"capability": capability, "provider_name": name})`.
  `requisite.telemetry.JSONFormatter` merges `extra` fields into the JSON
  payload automatically; the same call still reads fine with the default
  plain formatter. Not every log call needs this — a genuinely prose
  message doesn't need invented structure — but do it for anything with
  clear entity/count/decision shape. See
  [ADR-0003](docs/adr/0003-prompt-templates-structured-logging-conversation-policy.md).
- Never log secrets. `Settings` already stores API keys as `SecretStr`
  specifically so an accidental `logger.debug(settings)` doesn't leak one
  — don't work around that by logging `.get_secret_value()` anywhere.
- When a log event has an obvious structured shape (an entity name, a
  count, a decision made among alternatives), pass it via `extra={...}`
  rather than only interpolating it into the message string:
  `logger.debug("Resolved capability '%s' -> '%s'", capability, provider_name, extra={"capability": capability, "provider_name": provider_name})`.
  This costs nothing when logs are consumed as plain text, and makes them
  queryable when an application opts into
  `requisite.telemetry.configure_logging(json_format=True)` (see
  `docs/adr/0003-prompt-templates-structured-logging-conversation-policy.md`).
  Not every log call needs this — a purely prose message doesn't need
  `extra=` invented for it.

## Error handling

- Every exception raised by the framework inherits from `AIException`
  (see `core/exceptions.py`). Pick the most specific existing subclass;
  only add a new one if none fits (and if you do, add it to the tree in
  both `core/exceptions.py` and the table in `README.md`).
- Wrap, don't swallow: `except SomeSDKError as exc: raise
  ProviderException(..., original_error=exc) from exc`. The `from exc`
  is not optional — it's what keeps the original traceback attached.
- Include enough context in the message to debug without a repro:
  provider/tool/capability name, and relevant non-secret arguments.

## Composition over inheritance

New behavior should almost always be a new *implementation* of an
existing interface (`BaseProvider`, `BaseOrchestrator`, a capability
resolver function, a `BaseSkill` subclass), not a subclass of a concrete
class like `OpenAIProvider` or `Agent`. If you find yourself subclassing a
concrete class to change its behavior, stop and check whether the
behavior you want should instead be:

- a constructor parameter (most flexible, least code),
- a new registry entry (if it's a genuinely different implementation),
  or
- a small new abstraction (last resort — open an issue to discuss first
  if it touches a public API).

## No global mutable state, no singletons

Every registry is a plain, instantiable class. `default_registry` /
`default_capability_registry` / etc. are convenience instances created at
import time, not enforced singletons — tests routinely construct their
own isolated registry instead (see any `registry_with_fake` fixture).
Don't add a new subsystem that only works through one shared global
instance; give it the same registry shape as everything else.

## Versioning & deprecation

- [Semantic Versioning](https://semver.org/). A public API is anything
  importable from `requisite` directly, or from a sub-package without a
  leading underscore.
- Prefer deprecation over removal. If a public API must change:
  1. Add the new API alongside the old one.
  2. Have the old one emit `DeprecationWarning` (via `warnings.warn`,
     `stacklevel=2`) pointing at the replacement.
  3. Note it in `CHANGELOG.md` under `### Deprecated`.
  4. Remove only in a subsequent major version, documented under
     `### Removed`.
- `ToolRegistry._resolve` is an existing example of this pattern — kept as
  a thin wrapper around the module-level `resolve_tool_like` for backward
  compatibility after the refactor that introduced the latter.

## Dependency policy

- Core dependencies (`pydantic`, `pydantic-settings`) are the only ones
  installed unconditionally. Everything provider- or backend-specific
  (`openai`, `google-genai`, `langgraph`) is an optional extra in
  `pyproject.toml`'s `[project.optional-dependencies]`, imported lazily
  inside the one module that needs it, with a `ConfigurationException`
  and install hint if it's missing.
- Think twice before adding a new hard dependency. Prefer the standard
  library where reasonable (e.g. `capabilities/resolvers.py`'s default
  providers use `urllib.request`, not `requests`, specifically to avoid
  adding a dependency for a reference implementation).

## Commits & branches

- Branch from `main`; keep PRs focused (see `CONTRIBUTING.md` for the PR
  checklist).
- Commit messages: imperative mood, one logical change per commit where
  practical (`Add GitHub capability resolver`, not
  `fix stuff / more changes / wip`).
