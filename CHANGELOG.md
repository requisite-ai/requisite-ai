# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.19.0] - 2026-08-19

### Added

- `github` default capability resolver -- see
  [ADR-0020](docs/adr/0020-github-capability-resolver.md) for the full
  design. Closes the last remaining line in `ROADMAP.md`'s Capabilities
  section.
- `agent.requires("github")` now resolves to `search_github(query)`,
  backed by GitHub's free, unauthenticated Search API (top 5 repositories
  by stars). No API key required -- `GITHUB_TOKEN` (reserved in
  `.env.example`) remains earmarked for a separate, future first-party
  MCP GitHub server, which would register at a higher priority.
- `examples/capability_example.py` extended with a `"github"`
  demonstration alongside the existing three built-in capabilities.

## [0.18.0] - 2026-08-19

### Added

- General graph execution: a new `"graph"` multi-agent strategy -- see
  [ADR-0019](docs/adr/0019-graph-execution-strategy.md) for the full
  design. Closes the last remaining line in `ROADMAP.md`'s "Agents &
  multi-agent orchestration" section.
- `Workflow.graph()` / `Workflow.add_edge(from_, to, *, condition=None)`:
  nodes (an `Agent` or a named `Workflow` "team", added via `.add()` as
  usual) are wired together with developer-declared edges. Unlike every
  other strategy, routing isn't decided by an LLM at run time -- each
  edge's `condition` is a plain Python callable checked against the
  source node's output content; the first matching edge (in declaration
  order) is taken. The first node added is the entry point; a node
  reaches `requisite.END` via a matching edge, or terminates implicitly
  if it has no outgoing edges. Cycles are allowed, bounded by `max_steps=`
  (default 25).
- New top-level export `requisite.END`, the sentinel edge target marking
  a terminating edge in the `"graph"` strategy.
- `examples/workflow_example.py` extended with `.graph()` demonstrations:
  a conditional-branch triage pipeline (run twice with different inputs
  to show two different paths taken) and a self-revising cycle bounded
  by `max_steps=`.

## [0.17.0] - 2026-08-17

### Added

- Tree-of-thoughts multi-agent strategy -- see
  [ADR-0018](docs/adr/0018-tree-of-thoughts-strategy.md) for the full
  design (the classic ToT-BFS/beam-search algorithm mapped onto this
  codebase's conventions -- there was no prior design sketch anywhere
  in the repo for this one, unlike every other strategy shipped this
  stretch).
- `Workflow.tree_of_thoughts()`: the first agent added becomes the
  evaluator; the remaining agents generate candidate reasoning steps,
  assigned round-robin. Each level, `breadth=` candidates (default 3)
  are generated per surviving path, scored together in one structured
  call, and pruned to the top `beam_width=` (default 1) before
  continuing, for up to `max_depth=` levels (default 3) -- stopping
  early if any candidate is scored as a complete final answer.
- `examples/workflow_example.py` extended with a `.tree_of_thoughts()`
  demonstration alongside the existing strategies.

## [0.16.0] - 2026-08-17

### Added

- Entry-point plugin discovery -- see
  [ADR-0017](docs/adr/0017-entry-point-plugin-discovery.md) for the
  full design, including why a single `"requisite.plugins"` entry-point
  group (not one per registry) and why a broken plugin can't block
  discovery of the rest.
- `requisite.plugins.discover(group="requisite.plugins")`: imports every
  package registered under the group, letting each self-register with
  whichever framework registry it targets exactly as manual registration
  already works (`default_registry.register(...)` in the plugin's own
  `__init__.py`, or a `module:register` callable target). No new
  registration mechanism -- this only automates the import step ADR-0001
  originally deferred.
- `requisite plugins [--group]` on the CLI: discovers and reports
  loaded/failed plugins, exits `1` if any plugin failed to load.
- Never automatic -- `discover()` is not called anywhere in `requisite`'s
  own import chain; an application (or the CLI) calls it explicitly.

## [0.15.1] - 2026-08-11

### Fixed

- `tests/test_rate_limiter.py::test_rate_limiter_shared_across_threads_never_exceeds_limit_in_any_window`
  flaked on GitHub Actions' py3.11 job (`assert 3 <= 2`) on an unrelated,
  already-pushed commit. Root cause: the test measures wall-clock
  timestamps recorded *after* `RateLimiter.acquire()` returns (plus a
  separate `threading.Lock()` + list-append), not the limiter's own
  internal claim times, against a window shrunk to 0.3s for test speed --
  tight enough that ordinary CI thread-scheduling jitter (GIL contention,
  a loaded/noisy runner) could make a correctly-spaced call look bunched
  from the outside even though `RateLimiter.acquire()` itself claimed
  slots correctly (verified by reading `requisite/core/rate_limiter.py`
  directly -- no logic change needed there). Widened the test's window to
  1.0s, giving enough headroom to absorb that jitter while staying fast
  (~2.5s locally, run 5x with no failures). Same shape as the ruff
  0.16.0 and `mcp` 2.0.0 incidents in spirit: a real CI failure traced to
  its actual root cause before touching anything, not silenced.

## [0.15.0] - 2026-08-11

### Added

- LangGraph backend branching/conditional graphs -- see
  [ADR-0016](docs/adr/0016-langgraph-branching.md) for the full design,
  including why `supervisor` specifically (the one existing strategy
  whose shape is genuinely conditional routing, not a disguised loop)
  and how it reuses `NativeOrchestrator`'s decision logic instead of
  duplicating it.
- `Workflow().supervisor().use_langgraph()` now works -- previously
  `LangGraphOrchestrator` raised `ConfigurationException` for any
  non-`"sequential"` strategy. Built on a real `add_conditional_edges`
  graph with a loop-back cycle, verified to genuinely re-route (not a
  fixed chain) by a test that delegates to two different workers across
  rounds and asserts both ran, in order.
- `examples/workflow_example.py` extended with a
  `.supervisor().use_langgraph()` demonstration alongside the existing
  native one.

### Changed

- `NativeOrchestrator._split_coordinator_and_workers` is now a
  `@staticmethod` (it never used `self`) so `LangGraphOrchestrator` can
  reuse it directly. Every existing internal call site is unaffected.
- `langgraph` dependency floor: `>=0.2` -> `>=1.0` (both the `langgraph`
  extra and the `all` extra, plus `requirements.txt`). The old floor
  predated langgraph's 1.0 API stabilization and gave no real guarantee
  about the `add_conditional_edges` API this feature depends on; the new
  floor honestly reflects what's actually verified (`1.2.9`, installed
  and exercised directly). A floor bump, not a cap -- backward-compatible
  for anyone already on a modern `langgraph`.

## [0.14.0] - 2026-08-11

### Added

- MCP server integration -- see [ADR-0015](docs/adr/0015-mcp-server-integration.md)
  for the full design, including why it's built on the low-level
  `mcp.server.lowlevel.Server` rather than `FastMCP` (Requisite's own
  `Tool` already carries a JSON Schema; `FastMCP`'s tool layer has no
  supported way to accept one, and always re-derives it via pydantic),
  and a real bug (`405 Method Not Allowed` on every Streamable HTTP
  request) found only by testing a real round trip, not by reading the
  SDK source.
- `MCPServer(name=, tools=, agents=)`: exposes Requisite tools/agents as
  an MCP server. `.add_tool(...)`/`.add_agent(...)` for incremental
  registration; `.run_stdio()`/`.run_http(host=, port=)` (plus
  `arun_stdio`/`arun_http` async counterparts) to serve. One class, no
  transport subclasses -- mirrors `MCPClient`'s existing
  `stdio`/`http` shape.
- `Agent.as_tool() -> Tool`: exposes an agent as a single tool taking a
  `prompt` argument and returning its final answer, mirroring
  `BaseSkill.as_tool()`. Reusable beyond MCP (agent-as-tool composition).
- `examples/mcp_server_example.py`: a runnable server (two tools, one
  agent) -- also the subprocess target used to verify `MCPServer`
  end-to-end against Requisite's own `MCPClient`, real stdio subprocess
  and real Streamable HTTP port, including a real Gemini call executing
  inside the server process.

## [0.13.0] - 2026-08-11

### Added

- `requisite` CLI -- see [ADR-0014](docs/adr/0014-cli.md) for the full
  design, including why "list registered agents" needed a project
  convention rather than a new global registry, and why the CLI's
  `print()`-based output is a deliberate, scoped exception to the
  framework's "never `print()`" rule.
- `requisite init NAME [--provider] [--force]`: scaffolds a runnable
  project (`.env.example`, `.gitignore`, `requirements.txt`, `agents.py`
  with an `agent_registry` convention, `main.py`, `README.md`).
- `requisite providers`: lists every registered provider, whether its SDK
  is installed, and whether an API key is configured for it.
- `requisite capabilities`: lists every registered capability and its
  competing providers (priority + current availability).
- `requisite agents [--module]`: lists agents registered in the current
  project's `agents.py` (or a `--module` override).
- `requisite chat [PROMPT] [--provider] [--model] [--agent] [--module]`:
  one-shot or interactive chat; `--agent NAME` routes through a
  scaffolded project's `Agent` (so its tools/capabilities work), rather
  than a bare `AI` call.
- Installed as the `requisite` console script
  (`[project.scripts]` in `pyproject.toml`) and runnable via
  `python -m requisite`. No new dependency -- built on stdlib `argparse`,
  per `DEVELOPMENT.md`'s dependency policy.

## [0.12.0] - 2026-08-11

### Added

- Hierarchical multi-agent strategy -- see
  [ADR-0013](docs/adr/0013-hierarchical-strategy.md) for the full
  design, including why this turned out smaller than ADR-0011/ADR-0012
  originally estimated.
- `Workflow.hierarchical()`: same shape as `.supervisor()`, except a
  delegate (`steps[1:]`) may be either an `Agent` or another named
  `Workflow` ("team") -- delegating to a `Workflow` runs whatever
  strategy it's configured with, including another `supervisor` or
  `hierarchical`, giving real recursive delegation.
- `Workflow(name=...)`: new optional constructor parameter, backward
  compatible. Only required when a `Workflow` is used as a hierarchical
  delegate.

### Changed

- `Workflow.add()`/`.agents`, `BaseOrchestrator.run`/`.arun`,
  `NativeOrchestrator.run`/`.arun`, and `LangGraphOrchestrator.run`/
  `.arun` now type their `steps` parameter as `Any` rather than
  `Agent`, so a `Workflow` delegate type-checks under `mypy --strict`
  at the call site -- matching `WorkflowResult.steps`'s existing
  `list[Any]` precedent on the output side. No behavior changed for
  any existing caller; every internal strategy method keeps its
  previous, more specific typing unchanged.
- Internal: `_run_supervisor`/`_arun_supervisor` refactored to share a
  round-loop implementation with the new hierarchical strategy
  (`_run_delegation_loop`/`_arun_delegation_loop`) -- the two are
  structurally identical, differing only in how delegates are
  validated. Verified byte-identical behavior via the existing,
  unmodified supervisor tests.

## [0.11.0] - 2026-08-11

### Added

- Two more native-orchestrator multi-agent strategies -- see
  [ADR-0012](docs/adr/0012-debate-and-map-reduce-strategies.md) for the
  full design. Second of two planned passes over the strategies
  ADR-0007/ADR-0011 deferred; hierarchical and tree-of-thoughts remain
  📋, still not a fit for the existing flat `steps` model.
- `Workflow.debate()`: the first agent added becomes a moderator; every
  agent added after it debates the input over `max_rounds` (default 3),
  each round seeing every debater's arguments from the *previous* round
  only (which is what makes each round safely concurrent), after which
  the moderator delivers a final verdict.
- `Workflow.map_reduce()`: the first agent added becomes a reducer;
  every agent added after it is a mapper. Pass `map_items=[...]` to
  `run()`/`arun()` -- assigned to mappers round-robin (so the item count
  doesn't need to match the mapper count) and run concurrently -- then
  the reducer combines every item's result into one final answer.
  `input` is unchanged in meaning and type across every strategy; no
  existing `Workflow`/`NativeOrchestrator` signature changed.

No existing public API shape changed -- both are new `strategy` values
following the exact `_run_<strategy>`/`_arun_<strategy>` extension
pattern every prior strategy already used.

## [0.10.0] - 2026-08-11

### Added

- Two new native-orchestrator multi-agent strategies -- see
  [ADR-0011](docs/adr/0011-critic-and-consensus-strategies.md) for the
  full design, including why these two and not the other four still-📋
  strategies (debate, hierarchical, map-reduce, tree-of-thoughts).
- `Workflow.critic()`: two agents -- a generator (`steps[0]`) and a
  separate critic (`steps[1]`) -- iterate on a draft together, up to
  `max_rounds` (default 3), stopping early on the same
  `NO_CHANGES_NEEDED` sentinel `reflection` uses. Same shape as
  `reflection`, generalized to two distinct agents instead of one agent
  critiquing itself.
- `Workflow.consensus()`: the first agent added becomes a synthesizer;
  every agent added after it independently answers the same input
  concurrently (same `ThreadPoolExecutor`/`asyncio.gather` pattern as
  `parallel`), then the synthesizer combines their answers into one
  final response. Reuses the coordinator/worker split already shared by
  `planner`/`supervisor` -- requires at least 2 agents, unique names.

No existing public API shape changed -- both are new `strategy` values
on the existing `NativeOrchestrator`/`Workflow`, following the exact
`_run_<strategy>`/`_arun_<strategy>` extension pattern ADR-0007
established.

## [0.9.0] - 2026-08-11

### Added

- Hybrid/BM25 retrieval and re-ranking -- see
  [ADR-0010](docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md) for
  the full design.
- `BM25Retriever` (`requisite.rag.bm25`): standalone keyword retrieval,
  zero extra dependency (pure-Python Okapi BM25, no `rank-bm25`/numpy).
  Same public shape as `Retriever` (`add_texts`/`aadd_texts`/`retrieve`/
  `aretrieve`/`as_tool`).
- `HybridRetriever` (`requisite.rag.hybrid_retriever`): composes an
  embedding provider + vector store (dense) with an internal BM25 index
  (sparse), fusing results via Reciprocal Rank Fusion -- chosen over a
  normalized weighted-score blend since dense/BM25 scores aren't on
  comparable scales. `add_texts` chunks once and shares the same chunk
  ids across both sides, which fusion-by-id depends on.
- `BaseReranker` (`requisite.rag.base`) + `LLMReranker`
  (`requisite.rag.reranker`): listwise re-ranking via one
  `AI.chat_response`/`.achat_response(response_model=...)` call --
  reuses the framework's own already-integrated providers instead of a
  new cross-encoder ML dependency. Re-ranking is a standalone
  composable step, not wired into any retriever's constructor:
  `reranker.rerank(query, retriever.retrieve(query, top_k=20), top_k=5)`.
- `BM25Retriever`, `HybridRetriever`, `BaseReranker`, `LLMReranker` all
  exported from `requisite.rag`/`requisite` top-level, alongside
  `Retriever` -- none require an optional dependency.

No existing public API shape changed -- `Retriever` itself is
unmodified; the new classes are new implementations of the existing
`BaseRetriever`/new `BaseReranker` interfaces.

Verified against real Gemini in addition to the mocked test suite: a
`HybridRetriever` correctly surfaced both a keyword-only query (via
BM25) and a semantic-paraphrase query with no shared keywords (via
dense embeddings) from the same small corpus; `LLMReranker` correctly
re-scored and re-sorted a shuffled candidate list for a biology query,
putting the actually-relevant chunk first.

## [0.8.0] - 2026-08-11

### Added

- Streaming + tool calls together, across all 8 providers -- see
  [ADR-0009](docs/adr/0009-streaming-tool-calls.md) for the full design.
  `StreamChunk` (`requisite/core/interfaces.py`) gains `tool_calls:
  list[ToolCall]` and a `has_tool_calls` property; every provider
  accumulates any incremental/fragmented tool-call data internally and
  only attaches fully-assembled `ToolCall`s once complete, typically on
  the final chunk -- the same contract regardless of whether the
  underlying SDK streams arguments incrementally (OpenAI-family,
  Anthropic) or only ever delivers a tool call whole (Gemini, Ollama).
- `AI.stream_response`/`.astream_response`: new methods mirroring
  `chat_response`/`achat_response`, yielding the full `StreamChunk`
  sequence (including `tool_calls`) instead of bare text.
  `AI.stream`/`.astream` gain a `tools=` parameter too, matching
  `AI.chat`'s existing precedent of accepting tools but only returning
  text -- use `stream_response`/`astream_response` for the structured
  view.
- `BaseProvider.stream`/`.astream` gain `tools=` in the abstract
  signature, matching `chat`/`achat`.
- `OllamaProvider.stream`/`.astream` accept `tools=` for the first time
  -- previously the only provider whose streaming methods didn't even
  have the parameter.
- `AnthropicProvider.stream`/`.astream` switched from the SDK's
  `text_stream` convenience helper to raw event iteration
  (`content_block_start`/`content_block_delta`/`content_block_stop`),
  since only the raw event stream carries tool-call data.

No existing public API shape changed for callers who don't pass
`tools=` to a streaming call -- `AI.stream`/`.astream` still yield bare
text by default.

## [0.7.1] - 2026-08-11

### Fixed

- `RedisMemory`'s zero-config default URL was `redis://localhost:6379/0`.
  On Windows, resolving `"localhost"` tries the IPv6 loopback (`::1`)
  first and stalls for ~2s before falling back to IPv4 -- measured
  directly against a local Redis-compatible server (a fresh connection's
  first command: ~2.1s via `"localhost"` vs. ~0.02s via `"127.0.0.1"`).
  Changed the default (and the `REDIS_URL` fallback) to
  `redis://127.0.0.1:6379/0`, which skips DNS/address-family resolution
  entirely -- strictly faster everywhere, never slower. Found while
  smoke-testing `RedisMemory` against a real local server (Memurai) for
  the first time; the mocked unit test suite never exercised real DNS
  resolution, so this wasn't caught by tests -- it needed an actual
  network round-trip to surface.

## [0.7.0] - 2026-08-10

### Added

- Two new `BaseMemory` implementations: `SQLiteMemory`
  (`requisite.memory.sqlite`, zero extra dependency -- stdlib `sqlite3`)
  and `RedisMemory` (`requisite.memory.redis`, `pip install
  requisite-ai[redis]`, `redis>=5.0` -- ships both the sync `redis.Redis`
  client and, since 4.2+, the built-in `redis.asyncio.Redis` async client
  with no separate `aioredis` package needed). Both registered in
  `default_memory_registry` as `"sqlite"` / `"redis"`, so
  `Agent(memory=default_memory_registry.create("sqlite", db_path=...))`
  works the same way `"in_process"` already does. `SQLiteMemory` is
  exported from `requisite`/`requisite.memory` directly like
  `InProcessMemory`; `RedisMemory` follows the same not-eagerly-imported
  treatment as `PineconeVectorStore`/`WeaviateVectorStore` so importing
  `requisite` never requires `redis` to be installed.
- `RedisMemory.aload`/`.aappend`/`.aclear` are true async overrides
  backed by a separately-built `redis.asyncio.Redis` client, not the
  `BaseMemory` default's thread-wrapped sync fallback -- the case that
  base class's docstring calls out as worth overriding for.
- New `MemoryException(AIException)` for memory backend operation
  failures (connection/session setup, load/append/clear), matching
  `VectorStoreException`'s shape (`backend=` instead of `store=`).
- New optional dependency group `redis` (included in the `all` extra).
  `redis.*` added to `[tool.mypy.overrides]`'s ignore-missing-imports
  list alongside the other optional SDKs.
- `REDIS_URL` added to `.env.example`, read directly by `RedisMemory`
  (falls back to it when `url=` is omitted -- unlike the Pinecone/OpenAI
  SDKs, `redis-py` does not read this env var on its own, so the fallback
  is implemented in `RedisMemory` itself), not by `Settings`.

### Fixed

- `VectorStoreException` (added in 0.6.0) was never added to
  `requisite`'s top-level exports -- every other concrete exception was
  reachable as `from requisite import ...` except this one. Fixed
  alongside the `MemoryException` addition above.

Both new memory backends are new implementations of the existing
`BaseMemory` interface -- no public API shape changed for existing code.

## [0.6.2] - 2026-08-10

### Fixed

- `Agent.arun()`'s tool-calling loop awaited each of a turn's tool calls
  one at a time (`for call in response.tool_calls: await
  tool_instance.aexecute(...)`) even though providers already return the
  full list of independent tool calls for that turn up front. In an
  async context this was pure wasted latency -- three tools each taking
  200ms serialized to 600ms instead of running concurrently. Now uses
  `asyncio.gather` to run them concurrently; `asyncio.gather` preserves
  input order in its results regardless of completion order, so
  `tool_calls_executed` and the resulting `tool_result` messages stay
  deterministic. `Agent.run()` (the sync path) is unaffected -- sync
  execution has no concurrency to exploit here. See
  `tests/test_agents.py::test_agent_arun_executes_independent_tool_calls_concurrently`.

## [0.6.1] - 2026-08-08

### Fixed

- `mypy` was declared `mypy>=1.10` in `pyproject.toml`'s `dev` extra and
  `requirements.txt` -- an unbounded floor, not an exact pin -- despite
  `DEVELOPMENT.md` explicitly documenting "dev tool versions (`ruff`,
  `mypy`) are pinned exactly." In practice this meant CI/local installs
  had already silently drifted across a major version (mypy 1.x -> 2.x)
  with nobody deciding that deliberately -- the same class of risk the
  ruff 0.16.0 incident (0.3.2) was supposed to have closed for good.
  Verified `mypy==2.3.0` (current latest) is clean against `requisite
  --strict` and `examples --strict` before pinning to it exactly.
- Bumped the exact `ruff` pin from `0.14.0` to `0.14.14` -- the latest
  patch release within the same `0.14.x` line (no new default-enabled
  lint rules, unlike the `0.15`/`0.16` lines), verified clean against
  `ruff check .` / `ruff format --check .` before pinning.

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

[Unreleased]: https://github.com/requisite-ai/requisite-ai/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/requisite-ai/requisite-ai/compare/v0.6.0...v0.6.1
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
