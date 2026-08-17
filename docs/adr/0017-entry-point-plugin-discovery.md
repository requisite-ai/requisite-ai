
# 0017. Entry-point plugin discovery

Status: Accepted
Date: 2026-08-17

## Context

`ROADMAP.md`'s Plugin architecture section had one remaining line: *"A
`requisite-plugin-*` naming/discovery convention (entry points) — 📋."*
This isn't a new design problem — ADR-0001 already worked it out in
detail and deliberately deferred it, naming the exact trigger to revisit:

> "once there are a handful of real third-party packages doing this
> manually, or once the planned `requisite` CLI wants to answer "what's
> installed" without every plugin needing to already be imported —
> implement entry-point discovery as an *additive* layer (a
> `requisite.plugins.discover()` call that imports registered entry
> points and lets them self-register exactly as they do today), not a
> replacement for explicit registration."

The `requisite` CLI shipped this session (ADR-0014). That's the trigger.
ADR-0001 also already ruled out the one alternative worth re-litigating
today: a `Plugin` base class, rejected because "every real plugin still
just calls `registry.register(...)` inside it" — discovery only needs to
automate the *import* step, not invent a new registration API.

Confirmed directly from the nine registries' actual `.register(...)`
signatures (`ProviderRegistry`, `CapabilityRegistry`, `OrchestratorRegistry`,
`ToolRegistry`, `MemoryRegistry`, `EmbeddingRegistry`, `VectorStoreRegistry`,
`PromptTemplateRegistry`, `MCPClientRegistry`) that they're genuinely
heterogeneous — some register `(name, builder)` pairs, some register a
single already-built instance keyed off its own `.name`, and
`CapabilityRegistry.register` alone takes two positional plus three
keyword-only arguments and *returns* a record. This rules out any
"generic normalized registration" design and confirms the shape ADR-0001
already sketched is the right one: discovery imports/calls each plugin's
own entry point, and that code does whatever `.register(...)` calls it
needs, unchanged from `CONTRIBUTING.md`'s existing third-party guidance
("a third-party package that registers its own").

## Decision

### One entry-point group, `"requisite.plugins"`, not one per registry

ADR-0001 floated per-registry group names (`requisite.providers`,
`requisite.capabilities`, ...) but explicitly declined to commit to
them, "before there's a second or third real plugin to validate the
scheme against." That validation still hasn't happened, and inventing
nine group names now (one per registry) would force a plugin that wants
to register a provider *and* a capability to declare two entry points
for what is conceptually one release — exactly the "partial plugins...
more awkward, not less" concern ADR-0001 raised about a `Plugin` class,
just relocated to entry-point groups. A single group, matching the
already-sketched `requisite.plugins.discover()` function's own module
name, avoids the problem entirely: one entry point, any number of
registrations inside it.

### `requisite/plugins.py`: one function, one small result type

```python
DEFAULT_GROUP = "requisite.plugins"

class PluginDiscoveryResult(BaseModel):
    loaded: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)

def discover(*, group: str = DEFAULT_GROUP) -> PluginDiscoveryResult:
    ...
```

For each entry point in the group: `entry_point.load()` (this alone
imports the target; a plain-module target's top-level
`registry.register(...)` calls already ran by the time `.load()`
returns — nothing further needed). If the loaded object is also
callable — an entry point pointing at `module:register` rather than a
bare module — it's called with zero arguments. Both are legitimate
plugin-authoring shapes per ADR-0001's own sketch ("imports registered
entry points and lets them self-register"); discovery doesn't force a
plugin author into one over the other.

### A broken plugin doesn't abort discovery of the rest

Each entry point's load/call is wrapped individually; failures land in
`result.failed[name] = str(exc)` (also logged at `ERROR`), successes in
`result.loaded`, in discovery order. This is a deliberate, scoped
departure from `DEVELOPMENT.md`'s usual "wrap and re-raise, never
swallow" convention — that convention is about a *single* call's
failure propagating to its caller with context attached. This is a
*batch* of independent, unrelated third-party packages, where one
author's bug shouldn't prevent every other installed plugin from
loading. Nothing is silently lost: every failure is both logged and
present in the returned result for the caller (an application, or
`requisite plugins` on the CLI) to inspect and act on.

### Never automatic

`discover()` is not called anywhere in `requisite`'s own import chain.
ADR-0001's stated reason for deferring auto-discovery in the first place
still holds verbatim: "keeps import-time behavior fully predictable —
nothing runs code from a package you didn't explicitly import, which
matters for a framework that will run model-directed tool execution."
An application (or the CLI) calls `discover()` explicitly, typically
once, at startup.

### Not re-exported at the top level

`requisite/__init__.py`'s `__all__` already carries six `default_*_registry`
names plus a bare `default_registry`. `requisite.plugins.discover()`
stays namespaced (`from requisite.plugins import discover`), matching
the existing precedent that `requisite.telemetry.configure_logging` —
also a one-time startup call, not a constructor reached for constantly
— isn't flattened into the top-level `__all__` either, despite getting
its own docstring example in `requisite/__init__.py`'s module docstring,
which `discover()` gets too.

### `requisite plugins` CLI subcommand

Directly satisfies the trigger's own wording: "the CLI wants to answer
'what's installed'." `requisite/cli/commands.py` gains `cmd_plugins`,
calling `discover()` and printing loaded/failed; `--group` overrides the
scanned group. Returns exit code `1` if any plugin failed to load (so
`requisite plugins` is script-friendly — a CI step or startup check can
treat a nonzero exit as "a plugin is broken"), `0` otherwise, including
when nothing was found at all.

### Doesn't touch conflict resolution

`CONTRIBUTING.md` already treats capability *names* as a shared,
socially-coordinated namespace ("open an issue first" for a genuinely
new one), and `CapabilityRegistry`'s priority-based resolution /
first-registered-wins ties are unchanged and untouched by this feature.
Entry-point iteration order isn't guaranteed deterministic across
Python/setuptools versions — plugin authors needing precedence already
have `priority=` on `CapabilityRegistry.register(...)`; this feature
doesn't add a second, competing ordering mechanism. Explicitly out of
scope, matching `ROADMAP.md`'s still-💭 "capability conflict-resolution
policy" line.

## Alternatives considered

- **Per-registry entry-point groups** (`requisite.providers`,
  `requisite.capabilities`, ...). Rejected — see Decision above; forces
  multi-registration plugins to split across several entry points for
  no benefit, and ADR-0001 already explicitly declined to commit to
  this scheme without real plugins to validate it against.
- **A `Plugin` base class** plugins implement once. Already rejected by
  ADR-0001 for the same reason restated there: no behavior of its own
  beyond calling `registry.register(...)`, and it makes partial plugins
  more awkward. Nothing about building actual discovery changes that
  reasoning.
- **Raising on the first failed plugin** instead of collecting failures
  in the result. Rejected: one broken third-party package (out of the
  caller's control) shouldn't prevent every other installed plugin from
  loading. The framework's general "never swallow" convention is about
  single-call failures reaching their caller with context — this
  preserves that spirit (every failure is surfaced, not hidden) while
  adapting the mechanism to a batch of independent operations.
- **Automatic discovery on `import requisite`.** Rejected — ADR-0001's
  predictability argument (no code runs from a package you didn't
  explicitly ask to run, relevant for a framework doing model-directed
  tool execution) applies with undiminished force now that there's
  actually a discovery mechanism to make automatic.

## Consequences

### Positive

- Closes the last `📋` line in `ROADMAP.md`'s Plugin architecture
  section.
- A plugin author writes exactly the same code CONTRIBUTING.md already
  documents today (`default_registry.register(...)` in their package's
  `__init__.py`, or a dedicated `register()` function) — discovery adds
  a `[project.entry-points."requisite.plugins"]` declaration to *their*
  `pyproject.toml`, nothing more.
- `requisite plugins` gives real, CI-usable signal (nonzero exit on any
  plugin failure) without requiring an application to write its own
  `discover()` + result-inspection boilerplate.

### Negative / risks

- Entry-point iteration order is not guaranteed — a plugin author
  relying on "my plugin's registrations always win" without setting
  `priority=` explicitly will get non-deterministic behavior across
  environments. This is an existing `CapabilityRegistry` property, not
  new here, but discovery makes it easier to hit by accident (multiple
  plugins now load without the application author writing an explicit,
  ordered sequence of imports).
- `discover()` executes arbitrary third-party code the moment
  `entry_point.load()` runs, same as any Python import — no sandboxing,
  same trust model as `pip install`ing anything else.

### Follow-ups

- Official plugin listing/directory in the docs (`ROADMAP.md`'s other
  remaining Plugin architecture line) is unaffected by this ADR — a
  documentation/community effort, not a code change.
- If real third-party plugins surface a need for per-registry entry-point
  groups after all (e.g. an application wanting to discover *only*
  capability-providing plugins, skipping provider plugins), that's a
  concrete case to design against — not speculated on here.
