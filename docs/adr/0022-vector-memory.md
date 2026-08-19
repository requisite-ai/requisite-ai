
# 0022. Vector-database-backed memory

Status: Accepted
Date: 2026-08-19

## Context

`ROADMAP.md`'s Memory section has one remaining 📋 line: *"Vector-database-backed
memory."* FEATURES.md is explicit about scope: *"Depends on RAG's vector
store work"* — and ADR-0001's original `BaseMemory` section names the
intended shape precisely: *"a generic vector-store-backed variant for
**similarity-scoped recall**."*

That phrasing settles a real ambiguity up front. "Vector-database-backed
memory" could mean two different things:
1. Store the same chronological message list in a vector database's
   storage layer.
2. Semantic top-k retrieval over past messages — find the most
   *relevant* prior turns for a query, not just the most recent ones.

(1) would be strictly worse than `SQLiteMemory`/`RedisMemory` for no
benefit — no ordering guarantee, an embedding-provider dependency, and
higher latency, in exchange for nothing a plain database doesn't already
do better. (2) is the actual, distinguishing capability, and it's what
ADR-0001's own "similarity-scoped recall" phrasing already committed to.

## Decision

### Semantic recall lives outside `BaseMemory`'s contract, not inside it

`requisite/memory/base.py`'s own module docstring draws this line
explicitly: *"memory is conversation-shaped storage... not retrieval,
ranking, or similarity search (that's RAG's job)."* Given that, and given
no other `BaseMemory` implementation would ever need a `query`/`top_k`
parameter on `load()` — the same "no second implementation exists to
justify the abstraction" reasoning ADR-0008 used to reject a
`Base<X>`+registry for `RateLimiter` applies here — semantic recall is
**not** added to `BaseMemory.load()`'s signature. `VectorMemory` exposes
it as two additional methods beyond the shared interface,
`load_relevant(session_id, query, top_k=...)` /
`aload_relevant(...)`, called directly by application code that wants
it. `Agent.run()`/`arun()`'s existing call sites
(`self._memory.load(...)`, `.append(...)`) never need to know these
methods exist — `VectorMemory` is a fully drop-in `BaseMemory`
implementation, exactly like `SQLiteMemory`/`RedisMemory`, with zero
changes to `Agent`.

### `VectorMemory` composes two things, not one

- A **chronological `BaseMemory` delegate** (`history_backend`, defaulting
  to `InProcessMemory()`; any existing backend works — `SQLiteMemory`/
  `RedisMemory` for persistence) satisfies the plain `load`/`append`/
  `clear` contract unchanged.
- A **`BaseEmbeddingProvider` + `BaseVectorStore` pair**, composed the
  same way `Retriever(embedding_provider=..., vector_store=...)` already
  does — per ADR-0005's documented preference for composable interfaces
  over one bundled thing. Every `append()` also embeds the message's
  `content` and stores it as a `Chunk` tagged
  `metadata={"session_id": ..., "role": ...}`; this is what
  `load_relevant` searches.

Chunk ids are `f"{session_id}:{turn_index}"`, where `turn_index` comes
from an in-memory counter cache (`self._counts`) seeded once per session
from `len(history_backend.load(session_id))` and incremented locally
after that — not re-derived via a full `load()` on every call (see
"Turn-index caching and locking" below for why the first version of this
ADR's design was wrong on this point). `clear(session_id)` uses the same
cache to know exactly which chunk ids to remove (`BaseVectorStore.delete`
has no filter-based bulk delete), then resets the counter to 0.

Underlying `VectorStoreException` (from the vector store) and
`MemoryException` (from the chronological delegate) propagate
unwrapped, rather than being re-wrapped into a third `VectorMemory`-specific
exception type. Both are already typed and meaningful; re-wrapping would
only obscure which layer actually failed.

### Turn-index caching and locking

An earlier version of this design recomputed `turn_index` via a full
`history_backend.load(session_id)` on *every* `append()`/`clear()` call,
reasoning that the chronological delegate was already "the source of
truth" so deriving from it was safer than a second counter. That
reasoning undersold the actual cost and risk:

- **Cost**: for a persistent `history_backend` (`SQLiteMemory`,
  `RedisMemory` — the ones this feature explicitly recommends), `load()`
  is a full table scan / `LRANGE` plus a `Message.model_validate_json()`
  deserialize per row. Doing that on every single `append()` just to
  learn a count made each append O(n) and a full conversation O(n²) —
  directly undermining the backends the feature's own docs recommend for
  real use.
- **Race**: `turn_index = len(load(...))` then `append(...)` is two
  non-atomic calls. Two concurrent appends to the same session could
  both read the same length and compute the same `turn_index`, silently
  colliding on the same vector-store chunk id (verified directly: for
  `InMemoryVectorStore` this overwrites one chunk, permanently losing the
  other from `load_relevant`, with no error raised anywhere).
- **`clear()` had the same race** against a concurrent `append()`: a
  message that lands between `clear()`'s delete-range snapshot and its
  final history reset could survive the delete, then have its id reused
  by the *next* append after clear — stale, supposedly-cleared content
  resurfacing under a session the caller believed was wiped.

Fixed by caching the counter and guarding it with a lock: `self._counts:
dict[str, int]`, a `threading.Lock` for sync calls and a separate
`asyncio.Lock` for async calls (an `asyncio.Lock` is required for the
async path specifically — holding a `threading.Lock` across an `await`
would block the entire event loop for any other coroutine waiting on the
same lock, not just the current one). `append()`/`aappend()` only hold
the lock for the counter read/increment plus the (typically fast)
`history.append()` call — the slow, network-bound embedding call happens
*after* releasing the lock, so one session's embedding latency doesn't
serialize every other session's appends behind it. `clear()`/`aclear()`
hold the lock across their *entire* body (delete + history reset +
counter reset), which is what actually closes the append-vs-clear race:
any concurrent `append()` blocks until `clear()` fully finishes, so no
append can straddle the clear boundary. `clear()` is infrequent enough
that this coarser locking is an acceptable cost, unlike `append()`.

Mixing sync and async calls concurrently on the *same* `VectorMemory`
instance isn't guarded (the two locks don't cross-exclude each other) —
an unusual usage pattern this codebase doesn't defend against elsewhere
either (e.g. `Agent.run()`/`arun()` don't guard against concurrent
same-session use across the sync/async API either).

### Write ordering: embed before history, not after

`append()`/`aappend()` embed the message and write it to the vector
store *before* committing it to `history_backend`, not after. The
original ordering (history first) meant a failed embedding call (rate
limit, network error, transient auth failure) left the message durably
logged in chronological history with no corresponding vector-store
entry — silently excluded from `load_relevant` forever, since nothing
re-embeds already-persisted history later. Embedding first means a
failed embed leaves no partial state: nothing is appended anywhere, the
exception propagates, and the caller can retry the whole `append()`
call cleanly. The turn_index allocated for the failed attempt is simply
never reused for a real message — harmless, since ids only need to be
unique, not contiguous.

### `BaseVectorStore.search`/`asearch` gains an optional `filter` param

The one real gap found while designing this: `BaseVectorStore` had no way
to scope a search to "only this session's chunks." `Chunk.metadata` is
already stored by all three concrete stores, but none of them filtered
by it. Two ways to close this gap were considered:

- **Per-session store isolation** (no interface change): one vector
  store/namespace/collection per `session_id`. Rejected — doesn't fit
  `VectorStoreRegistry.create(name, **kwargs)`'s construct-once shape
  (kwargs are fixed before any `session_id` is known), and
  `InMemoryVectorStore` has no natural per-session isolation without
  inventing a new store-of-stores wrapper just for this feature.
- **Add `filter` to `search`/`asearch`** (chosen): `filter: Optional[dict[str, Any]] = None`,
  exact equality on every key given, default `None` = today's unfiltered
  behavior — fully backward compatible, zero change for existing RAG
  callers. Real vector-DB SDKs already support exactly this pattern
  natively; this isn't inventing new capability, just exposing what's
  already there.

Implemented per store, sharing one `matches_filter(metadata, filter)`
helper (`requisite/rag/base.py`) for the exact-match predicate rather
than duplicating it in each store that filters client-side:
- **`InMemoryVectorStore`**: `matches_filter` applied over the in-memory
  dict before scoring.
- **`PineconeVectorStore`**: `filter=filter` passed straight through to
  `index.query(...)` — Pinecone's own metadata filter syntax accepts a
  plain equality dict natively.
- **`WeaviateVectorStore`**: **client-side filtering, not native.** The
  collection schema stores metadata as one opaque `metadata_json`
  property (see `_get_collection`'s fixed property list: `chunk_id`,
  `text`, `metadata_json`), not per-key queryable properties — there's
  no schema-compatible way to build a native `Filter.by_property(...)`
  for an arbitrary caller-supplied key without a breaking schema change
  to every collection already created by existing users.

  The first version of this fix over-fetched one fixed window
  (`max(top_k * 20, 200)` nearest-by-raw-distance candidates) and
  filtered it client-side. That silently under-returned results whenever
  a session's own chunks weren't in that raw-distance window — plausible
  in exactly the shared-collection, many-sessions setup `VectorMemory`
  itself creates, since it's the primary caller of filtered search. Fixed
  to **page through** the collection instead (`_FILTER_PAGE_SIZE = 200`
  per page, `_FILTER_MAX_PAGES = 10`, via `near_vector(..., offset=...)`):
  keep fetching successive pages, filtering each client-side, until
  `top_k` matches are found or the collection is exhausted or the page
  cap is hit. Still bounded (2,000 scanned candidates by default, not
  unbounded) and still not a true server-side filtered query, but it no
  longer misses a match just because it happened to rank outside one
  arbitrary fixed window — verified directly with a test where the only
  filter-matching chunk is deliberately placed on the second page.
  Weaviate users who need filtered search at real scale should widen
  their collection's schema for native filtering (see Follow-ups), not
  rely on this pagination indefinitely.

Also fixed while touching this code: `search(top_k=0, filter=...)`
previously returned one result instead of zero (the client-side filter
loop appended a match *before* checking `len(results) >= top_k`). All
three stores now short-circuit to `[]` for `top_k <= 0` before doing any
work.

## Alternatives considered

- **Overloading `BaseMemory.load()` with an optional `query` parameter**
  instead of new `load_relevant`/`aload_relevant` methods. Rejected —
  see "Decision" above: this would put a similarity-search concept
  directly on the interface every `BaseMemory` implementation shares,
  contradicting that module's own explicit "not retrieval... that's
  RAG's job" scoping, for a capability only one implementation would
  ever have.
- **Deriving `turn_index` fresh from `history_backend.load()` on every
  call, with no local cache.** This was the original design, rejected
  after being shipped and then found broken — see "Turn-index caching
  and locking" above for the O(n²) cost and the collision race it
  caused. The shipped design's local counter cache is still seeded from
  `history_backend` (not an independent, drift-prone second source of
  truth invented from scratch) — it's a cache of that one source, kept
  in sync by being the only code path that mutates it, not a competing
  ledger.
- **A per-session `threading.Lock` (a `dict[str, threading.Lock]`)**
  instead of one lock per `VectorMemory` instance. Would allow different
  sessions' appends to proceed fully concurrently; rejected for now as
  more complexity (lock-dict growth/cleanup) than justified — each
  `append()` only holds its instance-wide lock for a fast counter
  update plus one `history.append()` call, not the slow embedding call,
  so cross-session contention on one lock is small relative to the
  embedding-API latency that already dominates `append()`'s cost. A
  per-session lock is a reasonable future optimization if that
  assumption stops holding.
- **Re-wrapping underlying exceptions in a new `VectorMemory`-specific
  exception type.** Rejected — `VectorStoreException`/`MemoryException`
  are already typed, already carry `original_error`, and re-wrapping
  would obscure which layer (embedding/vector-store vs. chronological
  storage) actually failed.

## Consequences

### Positive

- Closes the last 📋 line in `ROADMAP.md`'s Memory section.
- Genuinely additive: `Agent` needs zero changes; the `BaseVectorStore`
  interface change is backward compatible (`filter=None` default);
  `VectorMemory` reuses every existing composable piece (`BaseMemory`
  delegates, `BaseEmbeddingProvider`, `BaseVectorStore`) rather than
  inventing new abstractions.
- `VectorMemory` has no hard optional dependency of its own — like
  `Retriever`, the optional-dependency risk lives entirely in whichever
  `embedding_provider`/`vector_store` the caller constructs and passes
  in — so it's a full top-level export (`from requisite import VectorMemory`),
  same tier as `InProcessMemory`/`SQLiteMemory`, not hidden behind a
  submodule import like `RedisMemory`.
- RAG's own `search`/`asearch` callers gain metadata-filtered search too
  (e.g. multi-tenant knowledge bases), as a side effect of the interface
  extension this feature needed anyway.

### Negative / risks

- Weaviate's filtered search is client-side (paginated fetch + Python-side
  filter), not a true server-side filtered query — bounded to 2,000
  scanned candidates by default, so a filter matching a vanishingly small
  fraction of a very large collection can still return fewer than
  `top_k` results. A real limitation at large collection sizes,
  documented rather than hidden, and meaningfully better than the
  original single-fixed-window version (see "Decision" above).
- `VectorMemory`'s turn-index cache/locks live in-memory on the
  `VectorMemory` instance itself, not in `history_backend`. Two separate
  `VectorMemory` instances pointed at the *same* persistent
  `history_backend` (e.g. two process replicas sharing one `SQLiteMemory`
  file, or one `RedisMemory` server) don't share a lock or a counter
  cache, so the concurrency fix only holds within a single instance, not
  across a multi-process deployment. Out of scope for this ADR — the
  same caveat already applies to `SQLiteMemory`'s own file-level
  concurrency and isn't unique to `VectorMemory`.
- `load_relevant`'s relevance quality depends entirely on the embedding
  provider's quality for short conversational snippets, which is a
  different regime than the longer documents RAG's `Retriever` usually
  embeds — no special handling added for this; a real-network smoke test
  (real Gemini embeddings, not a fake) is part of this feature's
  verification specifically to catch a "technically works, semantically
  useless" outcome before shipping.
- `load_relevant` is not automatically wired into `Agent`'s own
  context-building — an application has to call it explicitly. See
  Follow-ups.

### Follow-ups

- Wiring `load_relevant` into `Agent`'s own tool-calling loop
  automatically (e.g. injecting relevant past turns into context
  alongside the plain chronological window) — a materially bigger,
  `Agent`-level design question, not attempted here.
- A native, server-side filtered search for `WeaviateVectorStore`, if
  per-key queryable properties are added to its schema (a breaking
  change for existing collections, so deliberately not bundled into this
  feature).
- Threading `filter` through `Retriever.retrieve`/`aretrieve` for
  symmetry with `BaseVectorStore.search`/`asearch` — not needed by this
  feature (`VectorMemory` composes `BaseVectorStore` directly, bypassing
  `Retriever` entirely), so left as a natural, low-cost follow-up rather
  than scope-crept in here.
