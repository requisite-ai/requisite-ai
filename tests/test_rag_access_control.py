"""Access-controlled retrieval: filter grammar, filtered BM25, retriever ``filter=``,
stable chunk ids and citation-bearing tool output.

The security tests here encode the property the feature exists for: a chunk the
caller's filter excludes must never come back through ANY retrieval path (dense,
keyword, or fused hybrid), and must not even influence the scores of the chunks
that are allowed.
"""

from __future__ import annotations

from typing import Any

import pytest

from requisite.core.exceptions import ConfigurationException
from requisite.rag.base import Chunk, matches_filter
from requisite.rag.bm25 import BM25Index

# ---------------------------------------------------------------------------
# Filter grammar
# ---------------------------------------------------------------------------


def test_equality_filter_is_unchanged() -> None:
    assert matches_filter({"a": 1, "b": 2}, {"a": 1})
    assert not matches_filter({"a": 1}, {"a": 2})
    assert matches_filter({"a": 1}, None)
    assert matches_filter({"a": 1}, {})


def test_in_matches_scalar_metadata() -> None:
    assert matches_filter({"team": "eng"}, {"team": {"$in": ["eng", "hr"]}})
    assert not matches_filter({"team": "legal"}, {"team": {"$in": ["eng", "hr"]}})


def test_in_matches_when_list_metadata_overlaps() -> None:
    assert matches_filter({"groups": ["eng", "all"]}, {"groups": {"$in": ["hr", "all"]}})
    assert not matches_filter({"groups": ["eng"]}, {"groups": {"$in": ["hr", "all"]}})
    assert matches_filter({"groups": ("eng",)}, {"groups": {"$in": ["eng"]}})
    assert matches_filter({"groups": {"eng"}}, {"groups": {"$in": ["eng"]}})


def test_in_with_empty_list_matches_nothing() -> None:
    assert not matches_filter({"groups": ["eng"]}, {"groups": {"$in": []}})
    assert not matches_filter({"team": "eng"}, {"team": {"$in": []}})


def test_in_missing_key_never_matches() -> None:
    assert not matches_filter({}, {"groups": {"$in": ["eng"]}})
    assert not matches_filter({"other": 1}, {"groups": {"$in": ["eng"]}})


def test_filter_keys_are_anded_across_equality_and_in() -> None:
    meta = {"groups": ["eng"], "tenant": "t1"}
    assert matches_filter(meta, {"groups": {"$in": ["eng"]}, "tenant": "t1"})
    assert not matches_filter(meta, {"groups": {"$in": ["eng"]}, "tenant": "t2"})


def test_unknown_operator_raises_instead_of_matching() -> None:
    with pytest.raises(ConfigurationException, match=r"\$gt"):
        matches_filter({"n": 5}, {"n": {"$gt": 1}})


def test_in_requires_a_list() -> None:
    with pytest.raises(ConfigurationException, match=r"\$in"):
        matches_filter({"team": "eng"}, {"team": {"$in": "eng"}})


def test_plain_dict_value_without_operators_is_still_equality() -> None:
    assert matches_filter({"meta": {"x": 1}}, {"meta": {"x": 1}})
    assert not matches_filter({"meta": {"x": 1}}, {"meta": {"x": 2}})


# ---------------------------------------------------------------------------
# BM25Index: the filter applies BEFORE scoring
# ---------------------------------------------------------------------------


def _chunk(cid: str, text: str, **metadata: Any) -> Chunk:
    return Chunk(id=cid, text=text, metadata=metadata)


def test_bm25_filter_excludes_restricted_chunks() -> None:
    index = BM25Index()
    index.add(
        [
            _chunk("pub", "the quarterly forecast is published", groups=["all"]),
            _chunk("sec", "the quarterly forecast is confidential", groups=["exec"]),
        ]
    )

    results = index.search("quarterly forecast", top_k=5, filter={"groups": {"$in": ["all"]}})

    assert [r.chunk.id for r in results] == ["pub"]


def test_bm25_unfiltered_still_returns_everything() -> None:
    index = BM25Index()
    index.add([_chunk("a", "alpha beta"), _chunk("b", "alpha gamma")])
    assert {r.chunk.id for r in index.search("alpha", top_k=5)} == {"a", "b"}


def test_bm25_scores_of_allowed_chunks_ignore_restricted_ones() -> None:
    """IDF and average length must be computed over permitted chunks only, so
    restricted documents cannot shift an allowed document's score."""
    allowed = [
        _chunk("a1", "budget review for the platform team", groups=["all"]),
        _chunk("a2", "holiday schedule", groups=["all"]),
    ]
    restricted = [
        _chunk(f"r{i}", "budget budget budget layoffs plan " * 5, groups=["exec"])
        for i in range(10)
    ]
    filt = {"groups": {"$in": ["all"]}}

    alone = BM25Index()
    alone.add(allowed)
    mixed = BM25Index()
    mixed.add(allowed + restricted)

    score_alone = alone.search("budget", top_k=5, filter=filt)[0].score
    score_mixed = mixed.search("budget", top_k=5, filter=filt)[0].score

    assert score_mixed == pytest.approx(score_alone)


def test_bm25_empty_in_filter_returns_nothing() -> None:
    index = BM25Index()
    index.add([_chunk("a", "alpha", groups=["all"])])
    assert index.search("alpha", filter={"groups": {"$in": []}}) == []


# ---------------------------------------------------------------------------
# Retriever ``filter=`` (dense, BM25, hybrid) -- sync and async
# ---------------------------------------------------------------------------

from requisite.rag.bm25 import BM25Retriever  # noqa: E402
from requisite.rag.hybrid_retriever import HybridRetriever  # noqa: E402
from requisite.rag.retriever import Retriever  # noqa: E402
from requisite.rag.vectorstores.in_memory import InMemoryVectorStore  # noqa: E402
from tests.test_rag import FakeEmbeddingProvider  # noqa: E402

_PUBLIC = {"groups": ["all"]}
_SECRET = {"groups": ["exec"]}
_ALLOWED = {"groups": {"$in": ["all"]}}

_TEXTS = [
    "The cafeteria opens at nine and serves breakfast.",
    "Project Nightingale layoff list: the cafeteria staff are affected.",
    "Cafeteria menu: bread, cheese and cider.",
]
_METAS = [_PUBLIC, _SECRET, _PUBLIC]


def _dense() -> Retriever:
    r = Retriever(embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore())
    r.add_texts(_TEXTS, metadatas=_METAS)
    return r


def _hybrid() -> HybridRetriever:
    r = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    r.add_texts(_TEXTS, metadatas=_METAS)
    return r


def _bm25() -> BM25Retriever:
    r = BM25Retriever()
    r.add_texts(_TEXTS, metadatas=_METAS)
    return r


@pytest.mark.parametrize("build", [_dense, _hybrid, _bm25], ids=["dense", "hybrid", "bm25"])
def test_filtered_retrieve_never_returns_restricted_chunks(build: Any) -> None:
    retriever = build()
    # "cafeteria" is in all three texts, including the restricted one.
    results = retriever.retrieve("cafeteria", top_k=10, filter=_ALLOWED)

    assert results
    assert all("Nightingale" not in r.chunk.text for r in results)


@pytest.mark.parametrize("build", [_dense, _hybrid, _bm25], ids=["dense", "hybrid", "bm25"])
def test_unfiltered_retrieve_still_sees_everything(build: Any) -> None:
    """The control: without a filter the restricted chunk is retrievable, which is
    exactly why every call serving a restricted corpus must pass one."""
    results = build().retrieve("cafeteria", top_k=10)
    assert any("Nightingale" in r.chunk.text for r in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [_dense, _hybrid, _bm25], ids=["dense", "hybrid", "bm25"])
async def test_filtered_aretrieve_never_returns_restricted_chunks(build: Any) -> None:
    results = await build().aretrieve("cafeteria", top_k=10, filter=_ALLOWED)

    assert results
    assert all("Nightingale" not in r.chunk.text for r in results)


def test_hybrid_filter_reaches_the_keyword_side_for_an_exact_keyword_hit() -> None:
    """A rare token that appears ONLY in the restricted chunk is the strongest
    possible BM25 hit; it must still not leak through the fused result."""
    r = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    r.add_texts(
        ["general company news", "zyzzyva acquisition target codename"],
        metadatas=[_PUBLIC, _SECRET],
    )

    leaked = r.retrieve("zyzzyva", top_k=10, filter=_ALLOWED)

    assert all("zyzzyva" not in x.chunk.text for x in leaked)
    assert any("zyzzyva" in x.chunk.text for x in r.retrieve("zyzzyva", top_k=10))


def test_filter_with_no_matching_chunks_returns_empty_not_everything() -> None:
    assert _hybrid().retrieve("cafeteria", filter={"groups": {"$in": []}}) == []
    assert _dense().retrieve("cafeteria", filter={"groups": {"$in": ["nobody"]}}) == []


def test_malformed_filter_raises_on_every_retriever() -> None:
    for build in (_dense, _hybrid, _bm25):
        with pytest.raises(ConfigurationException):
            build().retrieve("cafeteria", filter={"groups": {"$gt": 1}})


def test_as_tool_binds_filter_and_hides_it_from_the_model() -> None:
    retriever = _hybrid()
    tool = retriever.as_tool(filter=_ALLOWED)

    output = tool.execute(query="cafeteria")

    assert "Nightingale" not in output
    assert list(tool.parameters_schema["properties"]) == ["query"]


# ---------------------------------------------------------------------------
# Stable chunk ids (opt-in ``doc_ids``)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_dense, _hybrid, _bm25], ids=["dense", "hybrid", "bm25"])
def test_doc_ids_give_deterministic_ids_and_citation_metadata(build: Any) -> None:
    retriever = build()
    ids = retriever.add_texts(["alpha beta gamma"], doc_ids=["handbook-3"])

    assert ids == ["handbook-3:0"]
    hit = retriever.retrieve("alpha", top_k=10, filter={"doc_id": "handbook-3"})[0]
    assert hit.chunk.id == "handbook-3:0"
    assert hit.chunk.metadata["doc_id"] == "handbook-3"
    assert hit.chunk.metadata["chunk_index"] == 0


def test_doc_ids_number_each_chunk_of_a_long_document() -> None:
    retriever = BM25Retriever()
    ids = retriever.add_texts(["word " * 400], doc_ids=["d"], chunk_size=200, chunk_overlap=20)

    assert len(ids) > 2
    assert ids == [f"d:{i}" for i in range(len(ids))]


def test_reingesting_same_doc_id_overwrites_instead_of_duplicating() -> None:
    retriever = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    retriever.add_texts(["version one of the policy"], doc_ids=["policy"])
    retriever.add_texts(["version two of the policy"], doc_ids=["policy"])

    results = retriever.retrieve("policy", top_k=10)
    assert len(results) == 1
    assert "two" in results[0].chunk.text


def test_without_doc_ids_ids_are_random_and_metadata_untouched() -> None:
    retriever = BM25Retriever()
    first = retriever.add_texts(["alpha"], metadatas=[{"k": "v"}])
    second = retriever.add_texts(["alpha"], metadatas=[{"k": "v"}])

    assert first != second
    hit = retriever.retrieve("alpha", top_k=1)[0]
    assert hit.chunk.metadata == {"k": "v"}


def test_doc_ids_validation() -> None:
    retriever = BM25Retriever()
    with pytest.raises(ConfigurationException, match="same length"):
        retriever.add_texts(["a", "b"], doc_ids=["only-one"])
    with pytest.raises(ConfigurationException, match="unique"):
        retriever.add_texts(["a", "b"], doc_ids=["x", "x"])


@pytest.mark.asyncio
async def test_aadd_texts_supports_doc_ids() -> None:
    dense = Retriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    hybrid = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(), vector_store=InMemoryVectorStore()
    )
    bm25 = BM25Retriever()
    for retriever in (dense, hybrid, bm25):
        assert await retriever.aadd_texts(["alpha"], doc_ids=["d1"]) == ["d1:0"]


def test_doc_ids_override_caller_supplied_doc_id_metadata() -> None:
    retriever = BM25Retriever()
    retriever.add_texts(["alpha"], metadatas=[{"doc_id": "stale", "k": 1}], doc_ids=["real"])
    hit = retriever.retrieve("alpha")[0]
    assert hit.chunk.metadata == {"doc_id": "real", "k": 1, "chunk_index": 0}


# ---------------------------------------------------------------------------
# Citation-bearing tool output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_dense, _hybrid, _bm25], ids=["dense", "hybrid", "bm25"])
def test_as_tool_default_output_format_is_unchanged(build: Any) -> None:
    out = build().as_tool().execute(query="cafeteria")
    assert out.startswith("[score=")
    assert "source=" not in out


@pytest.mark.parametrize("build", [_dense, _hybrid, _bm25], ids=["dense", "hybrid", "bm25"])
def test_as_tool_source_key_adds_citation(build: Any) -> None:
    retriever = build()
    retriever.add_texts(["cafeteria rules"], doc_ids=["rules-9"])

    out = retriever.as_tool(source_key="doc_id", filter={"doc_id": "rules-9"}).execute(
        query="cafeteria"
    )

    assert "source=rules-9#0]" in out


def test_as_tool_source_key_missing_on_chunk_says_unknown() -> None:
    out = _bm25().as_tool(source_key="doc_id").execute(query="cafeteria")
    assert "source=unknown]" in out


def test_as_tool_source_without_chunk_index_has_no_ordinal() -> None:
    r = BM25Retriever()
    r.add_texts(["alpha"], metadatas=[{"doc_id": "d"}])
    assert "source=d]" in r.as_tool(source_key="doc_id").execute(query="alpha")


# ---------------------------------------------------------------------------
# Other stores: $in reaches them correctly
# ---------------------------------------------------------------------------


def test_in_memory_store_supports_in_filter() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            Chunk(id="a", text="a", metadata={"groups": ["all"]}, embedding=[1.0, 0.0]),
            Chunk(id="b", text="b", metadata={"groups": ["exec"]}, embedding=[1.0, 0.0]),
        ]
    )
    results = store.search([1.0, 0.0], top_k=5, filter={"groups": {"$in": ["all"]}})
    assert [r.chunk.id for r in results] == ["a"]


def test_in_memory_store_rejects_bad_filter_even_when_empty() -> None:
    with pytest.raises(ConfigurationException):
        InMemoryVectorStore().search([1.0], filter={"x": {"$regex": "a"}})
