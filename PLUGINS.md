# Plugin Directory

A list of third-party packages that extend Requisite via the plugin
mechanism described in [`CONTRIBUTING.md`](CONTRIBUTING.md#writing-a-plugin)
and [ADR-0017](docs/adr/0017-entry-point-plugin-discovery.md). This page
is a discovery aid, not a gate: any package that registers with one of
Requisite's registries (`ProviderRegistry`, `CapabilityRegistry`,
`OrchestratorRegistry`, `ToolRegistry`, `MemoryRegistry`,
`EmbeddingRegistry`, `VectorStoreRegistry`, `PromptTemplateRegistry`,
`MCPClientRegistry`) works today whether or not it is listed here.

## Listed plugins

No third-party plugins have been submitted for listing yet. If you
publish one, this table is where it goes -- see "Getting listed" below.

| Name | What it adds | Registries used | PyPI | Maintainer |
|---|---|---|---|---|
| _none yet_ | | | | |

## Getting listed

To add your plugin to the table above:

1. Publish it on PyPI, named `requisite-plugin-<something>` per
   `CONTRIBUTING.md`'s naming convention.
2. Declare it under the `"requisite.plugins"` entry-point group in your
   own `pyproject.toml`, so `requisite.plugins.discover()` and
   `requisite plugins` on the CLI find it automatically.
3. Give it its own `README` documenting what it registers and why.
4. Open a PR against this repo adding one row to the table above. Keep
   the "What it adds" cell short (one line); link out to your package's
   own README/docs for details rather than duplicating them here.

There is no approval process beyond a normal PR review -- this is a
directory of what exists, not an endorsement or a compatibility
guarantee. If a listed plugin stops working with a current Requisite
release, open an issue (or a PR removing/updating its row) rather than
leaving the table stale.

## See also

- [`CONTRIBUTING.md`](CONTRIBUTING.md#writing-a-plugin) -- how to write
  a plugin from scratch, with a worked example.
- [ADR-0017](docs/adr/0017-entry-point-plugin-discovery.md) -- the
  design behind entry-point discovery, and why there is no `Plugin` base
  class.
