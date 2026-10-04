# 0042. Access-controlled retrieval: filters on retrievers, a safe hybrid search, stable chunk ids

Status: Accepted
Date: 2026-10-04

## Context

A lab that put RAG behind access control (documents readable only by some
groups) hit these gaps. Each was checked in 0.40.0 source:

- `Retriever`, `HybridRetriever` and `BM25Retriever` had no `filter`
  argument on `retrieve`/`aretrieve`. Only `BaseVectorStore.search(...,
  filter=)` did, and `matches_filter` was exact equality per key, so "this
  chunk is readable by any of the user's groups" could not be one filter.
- **`HybridRetriever` was unsafe for restricted corpora.** It fused
  `vector_store.search(...)` with `BM25Index.search(...)`, and the BM25 index
  had no filter at all. A filtered dense search would still have returned
  restricted chunks through the keyword side.
- Chunk ids were random `uuid4`s and chunks carried no document id or
  ordinal, so a citation could not point at a stable passage.
- `as_tool()` returned `[score=0.812] text`, with no source to cite.

Out of scope on purpose: the authorization policy itself (who is in which
group, default-deny). The framework makes filtering possible and safe; the
application decides what to filter on.

## Decision

### Filter grammar (`requisite/rag/base.py::matches_filter`)

Backward compatible: a plain value is still exact equality. A dict whose keys
all start with `$` is an operator expression, and `$in` is the only operator:

- `{"groups": {"$in": ["eng", "hr"]}}` matches when the chunk's value is in
  the list, or, when the chunk's value is itself a list/tuple/set (the groups
  allowed to read it), when the two **overlap**. This is the semantics
  Pinecone already uses natively, so every store agrees.
- It fails closed. A key missing from the chunk never matches; an empty list
  matches nothing; any other `$` operator, or `$in` without a list, raises
  `ConfigurationException` (`validate_filter`, run up front so a bad filter
  fails even against an empty store). It never silently matches everything.

The grammar stays deliberately small so in-memory, Weaviate (client-side) and
Pinecone (native) behave identically. More operators can be added later
without breaking this.

### Where the filter is applied

- `BM25Index.search(..., filter=)` filters the corpus **before** scoring, so
  document frequencies and average length are computed over permitted chunks
  only. A restricted chunk can neither be returned nor shift the score of an
  allowed one (tested: an allowed chunk scores identically with and without
  ten restricted documents in the index). Post-filtering would have leaked
  through scoring and shrunk the candidate pool.
- `HybridRetriever` passes the same filter to both the dense and the BM25
  search before rank fusion.
- Dense stores already pre-filter. Pinecone receives the filter untouched
  (native `$in`); in-memory and Weaviate use `matches_filter`. Weaviate pages
  client-side with a bound, so a very selective filter may under-return but can
  never over-return.

### Retriever API

`filter=` on `BaseRetriever.retrieve/aretrieve` (the contract now says
implementations serving restricted data must apply it to every path, before
scoring or fusion), and on `Retriever`, `HybridRetriever`, `BM25Retriever`,
sync and async. `as_tool(filter=...)` binds the filter when the tool is
created; it is not a tool parameter, so the model can neither see nor change
it. The safe pattern is one tool per user or request.

### Stable chunk identity, opt-in

`add_texts` / `aadd_texts(..., doc_ids=[...])`. Given ids, a chunk's id is the
deterministic `"<doc_id>:<n>"` (re-ingesting overwrites instead of
duplicating, since every store upserts by id) and its metadata gains `doc_id`
and `chunk_index`. Without `doc_ids` nothing changes: random ids, no added
metadata. Errors: length mismatch, or a repeated id within one call.

The chunk-preparation loop was copy-pasted six times (three retrievers, sync
and async); it now lives once in `requisite/rag/_ingest.py::prepare_chunks`.

### Citations, opt-in

`as_tool(source_key="doc_id")` renders `[score=0.812 source=handbook-3#2] text`
(`#chunk_index` when present, `source=unknown` when the key is absent). The
default output is unchanged.

## Verified

- Security tests: a chunk containing the query's only exact keyword, readable
  only by another group, never appears through the dense path, the BM25 path
  or the fused hybrid result, sync or async. The unfiltered control does
  return it.
- Live, real Gemini embeddings through `HybridRetriever` over five documents:
  the unfiltered query returned all five including three restricted ones;
  filtering to `["all"]` returned only the two public documents; `["legal"]`
  returned only the legal one; `["all", "legal"]` the union.

## Alternatives considered

- **Post-filter BM25 results.** Rejected: restricted documents would still
  shape IDF and length normalisation, and the fused pool would shrink.
- **A full query language (`$gt`, `$or`, `$not`, ...).** Rejected for now:
  each operator must behave identically in three stores. `$in` covers
  membership, which was the gap.
- **Make `doc_ids` the default.** Rejected: it changes chunk ids and metadata
  for every existing user.
- **Let the model supply the filter as a tool argument.** Rejected: the filter
  is authorization, so it must come from the application.
- **Ship the policy layer (groups, default-deny).** Rejected: policy and
  security sensitive, and application-specific.

## Consequences

Positive: hybrid search is safe to use on access-controlled corpora; one
filter expresses group membership in every store; stable, citable chunk ids;
the retrievers share one ingest path.

Negative / risks:

- Filtering is still the caller's job. A call that omits `filter` sees
  everything; grounding or citation checks are not authorization (the lab's
  control run passed every citation check while leaking all five restricted
  documents).
- A third-party `BaseRetriever` that does not accept `filter=` fails when
  called with one. That is deliberate: it must not silently ignore it.
- Re-ingesting a document that got shorter leaves its old trailing chunks;
  delete the previously returned ids first. A `delete_document` helper is a
  follow-up, and matters for revoked text in restricted corpora.

Follow-ups: `delete_document(doc_id)`; further filter operators if a real need
appears; a `filter=` on rerankers/compressors is unnecessary (they only see
already-filtered results).
