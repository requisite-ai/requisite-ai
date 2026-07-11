# Security Policy

## Supported Versions

Requisite is pre-1.0 and evolving quickly. Security fixes are made against
the latest released version on PyPI; there is no long-term support branch
yet.

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| older   | :x:                |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report it privately using
[GitHub's private vulnerability reporting](https://github.com/requisite-ai/requisite-ai/security/advisories/new)
(Security tab → "Report a vulnerability"), or email the maintainers at
**security@example.com** *(replace with a real, monitored address before publishing this repo)*.

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a minimal proof-of-concept.
- The version(s) affected.

We'll acknowledge your report within 5 business days, and aim to provide a
fix or mitigation plan within 30 days for confirmed issues. We'll credit
reporters (unless you prefer to remain anonymous) in the release notes once
a fix ships.

## Scope Notes Specific to Requisite

A few areas worth extra scrutiny when reporting or reviewing:

- **API keys**: `Settings` stores provider API keys as `pydantic.SecretStr`
  specifically so they never appear in `repr()`/logs. If you find a path
  where a key leaks into logs, error messages, or `raw` provider payloads
  passed back to application code, that's a valid report.
- **Tool execution**: `Tool.execute` / `Agent.run`'s tool-calling loop
  executes model-requested function calls with model-supplied arguments.
  Applications that register tools with filesystem, network, or shell
  access are responsible for validating/sandboxing those arguments — but
  if you find a framework-level way to bypass argument validation entirely
  (e.g. schema coercion tricks), please report it.
- **Capability resolution**: if you find a way for a lower-priority or
  unavailable capability provider to be selected over a higher-priority
  available one (a resolution-order bug), that's a valid report even
  though it's not a classic "vulnerability" — it can cause silent
  behavior changes in security-sensitive tool selection.
