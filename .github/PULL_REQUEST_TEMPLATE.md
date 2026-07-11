## What does this PR do?

<!-- One or two sentences. -->

## Category

<!-- Check the one(s) that apply. -->

- [ ] New provider
- [ ] New orchestrator backend
- [ ] New capability resolver
- [ ] New multi-agent strategy
- [ ] Bug fix
- [ ] Core framework change (public API)
- [ ] Documentation
- [ ] Other

## How was this tested?

<!--
Point to new/updated tests. Remember: no real network calls or API keys in
tests -- fake the SDK via sys.modules injection (see tests/test_providers.py)
or use an in-memory fake provider (see tests/test_ai.py).
-->

## Breaking changes?

- [ ] This PR changes a public API (anything importable from `requisite`
      directly, or from a sub-package without a leading underscore).

If checked, describe the change and update `CHANGELOG.md` under
`## Unreleased`.

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy requisite` passes
- [ ] `pytest` passes
- [ ] New/changed public classes and functions have docstrings
      (Parameters, Returns, Examples)
- [ ] `CHANGELOG.md` updated if this is a user-facing change
