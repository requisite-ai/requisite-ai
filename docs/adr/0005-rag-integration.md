
# 0005. RAG integration

Status: Accepted
Date: 2026-07-23

## Context

RAG was deferred at ADR-0001 with a note that it decomposes into several
independent extension points rather than one interface. Before
implementing, three decisions were made explicitly with the project
owner during roadmap planning: an in-memory default vector store with
Pinecone/Weaviate as optional integrations (not yet built), and a
retriever exposed to agents as a `CapabilityProvider` rather than a new
`Agent` constructor parameter. This ADR records those decisions plus the
implementation choices that followed from them.

## Decision

### Three interfaces, not one

`requisite.rag.base` defines three independent abstract interfaces,
confirming ADR-0001's prediction:

- `BaseEmbeddingProvider` -- text -> vectors (`embed`/`aembed`, plus
  `embed_one`/`aembed_one` convenience wrappers for the common
  single-string case).
- `BaseVectorStore` -- store and search vectors (`add`/`search`/`delete`,
  sync + async pairs).
- `BaseRetriever` -- what an application actually calls
  (`retrieve`/`aretrieve`). Deliberately independent of the other two at
  the interface level: the shipped `Retriever` composes an embedding
  provider and a vector store (dense retrieval), but a future hybrid or
  keyword-based (BM25) retriever might not use an embedding provider at
  all -- `BaseRetriever` doesn't assume one exists.

Each gets its own registry (`EmbeddingRegistry`, `VectorStoreRegistry`),
following the same plain-class, not-a-singleton pattern as every other
registry in the framework.

### Chunking: character-based, not token-aware, for v1

`chunk_text(text, chunk_size=1000, chunk_overlap=200)` splits on
character count, preferring a whitespace break near the boundary so
words aren't split mid-token, rather than counting model tokens via a
tokenizer library (e.g. `tiktoken`). This is an approximation --
character count and token count aren't the same thing, and the ratio
varies by language and model -- but it's a dependency-free one, and
consistent with `chunk_size`/`chunk_overlap` being simple, predictable
numbers a developer can reason about without knowing which model's
tokenizer applies. A token-aware chunker is a natural follow-up (see
below) once there's a concrete accuracy complaint against this default,
not before.

### In-memory default vector store; Pinecone/Weaviate not yet built

`InMemoryVectorStore` ships in core -- zero dependencies, cosine
similarity computed in pure Python (no numpy), mirroring
`InProcessMemory`'s role for conversation memory. This was the decided
default (over requiring a real vector DB from day one), for the same
reason `InProcessMemory` exists: RAG should be demonstrable with nothing
to provision.

**Pinecone and Weaviate integrations are not part of this ADR** --
`.env.example` already reserves `PINECONE_API_KEY`/`PINECONE_ENVIRONMENT`
and `WEAVIATE_URL`/`WEAVIATE_API_KEY` for them, and `BaseVectorStore` is
the interface either would implement, but no code exists yet. This is a
deliberate scope cut for this ADR, not an oversight -- see Follow-ups.

Pure-Python cosine similarity (no numpy) was chosen for the same reason
`capabilities/resolvers.py` uses `urllib.request` instead of `requests`:
avoiding a dependency for a reference implementation whose whole purpose
is being usable with nothing installed beyond `pydantic`. It doesn't
scale to large corpora (linear scan over every stored chunk on every
search) -- acceptable for the in-memory default's intended scale (a few
thousand chunks); Pinecone/Weaviate exist precisely for when that's not
enough.

### Retrievers are capabilities, not an `Agent` parameter

`Retriever.as_tool(name="knowledge_base", ...)` returns a
`~requisite.tools.base.Tool` wrapping `retrieve()`, formatted as scored
text blocks. The intended usage is registering it into
`CapabilityRegistry`:

```python
capabilities.register("knowledge_base", retriever.as_tool())
agent.requires("knowledge_base")
```

This was the explicitly decided design (over adding
`Agent(retriever=...)`), reusing the mechanism that already exists for
providers, MCP servers, and every other capability rather than adding a
fourth, competing way to attach something to an agent. It also means a
retriever and, say, an MCP-backed knowledge base tool are equally valid
providers of the same `"knowledge_base"` capability -- an application can
swap one for the other via `CapabilityRegistry.register(..., priority=...)`
exactly like swapping a stub weather provider for a paid one.

## Alternatives considered

- **A token-aware chunker (tiktoken or similar) for v1.** Rejected --
  adds a dependency and a model-specific coupling (which tokenizer
  matches which model) before there's a concrete case where character
  counting's approximation actually causes a problem.
- **Requiring Pinecone or Weaviate as the default vector store.**
  Rejected in the roadmap-planning decision -- would make RAG
  undemonstrable without external provisioning, unlike every other
  "default" in the framework (`InProcessMemory`, native capability
  resolvers, the `"native"` orchestrator).
- **`Agent(retriever=...)` as a fourth constructor mechanism.** Rejected
  in the roadmap-planning decision -- see "Retrievers are capabilities"
  above.
- **A single `BaseRAGPipeline` interface bundling embedding + vector
  store + retrieval together.** Rejected: it would force every retrieval
  strategy through the dense-retrieval shape, which is exactly what
  `BaseRetriever`'s independence from the other two interfaces is
  designed to avoid (a future hybrid/BM25 retriever doesn't need an
  embedding provider at all).

## Consequences

### Positive

- `agent.requires("knowledge_base")` works today with zero external
  services provisioned (in-memory store, an OpenAI or Gemini API key for
  embeddings) -- verified end-to-end with a deterministic fake embedding
  provider in tests, and the real `OpenAIEmbeddingProvider`/
  `GeminiEmbeddingProvider` classes tested against faked SDK modules.
- The three-interface decomposition means Pinecone/Weaviate (vector
  store), a new embedding provider, or a hybrid retriever can each be
  added independently later without touching the other two.
- Reusing the capability mechanism means no new documentation concept
  was needed to explain "how do I give an agent a knowledge base" --
  it's the same answer as "how do I give an agent a weather tool."

### Negative / risks

- `InMemoryVectorStore`'s linear-scan search doesn't scale past a modest
  corpus size -- by design, not by oversight, but worth remembering
  before someone reaches for it at a scale it wasn't meant for.
- Character-based chunking will occasionally produce chunks whose actual
  token count surprises someone tuning a chunk_size against a specific
  model's context window. Documented, not solved, in this ADR.
- RAG evaluation (retrieval precision/recall, faithfulness) is entirely
  out of scope here -- logged separately in `ROADMAP.md`'s Evaluation
  section, deliberately not built speculatively ahead of having
  something concrete to evaluate.

### Follow-ups

- Implement Pinecone and Weaviate `BaseVectorStore` implementations --
  the interface is ready; this ADR's scope cut, not a design gap.
- Consider a token-aware chunking option once a real accuracy complaint
  exists against the character-based default -- as an additional
  strategy, not a replacement (character-based chunking's zero-dependency
  property is worth keeping as an option even after a tokenizer-aware one
  exists).
- Hybrid/BM25 retrieval as a second `BaseRetriever` implementation,
  should a real use case need it -- `BaseRetriever`'s independence from
  `BaseEmbeddingProvider` was specifically designed to make this possible
  without interface changes.
- Retrieval evaluation, once there's a concrete retriever in real use to
  evaluate against a labeled set -- tracked in `ROADMAP.md`'s Evaluation
  section, not this ADR.
