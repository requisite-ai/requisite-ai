"""
RAG example.

Shows building a retriever (embeddings + in-memory vector store),
adding documents, and exposing it to an agent as a capability so
`agent.requires("knowledge_base")` works exactly like any other capability.

Uses Gemini for both embeddings and chat by default. A retriever's
embedding provider is independent of an agent's chat provider, though --
swap in `OpenAIEmbeddingProvider` (from `requisite.rag.embeddings`) if
you'd rather embed with OpenAI while chatting with a different provider,
or vice versa.

    cp .env.example .env   # then fill in:
    #   GEMINI_API_KEY=...
    #   DEFAULT_PROVIDER=gemini
    #   MODEL=gemini-3.1-flash-lite
    python examples/rag_example.py
"""

from requisite import Agent
from requisite.capabilities import default_registry as capabilities
from requisite.config.settings import Settings
from requisite.rag import (
    BM25Retriever,
    HybridRetriever,
    LLMContextCompressor,
    LLMReranker,
    Retriever,
)
from requisite.rag.embeddings import GeminiEmbeddingProvider
from requisite.rag.vectorstores import InMemoryVectorStore

DOCS = [
    "Requisite is a provider-agnostic Python framework for building AI applications.",
    "Paris is the capital of France. The Eiffel Tower was completed in 1889.",
    "The Model Context Protocol (MCP) lets AI applications connect to external tools and data sources.",
    "The mitochondria is the powerhouse of the cell.",
    "REQ-4471 tracks the outstanding refund for order 88213.",
    "REQ-5502 is a separate support ticket about a delayed shipment, "
    "unrelated to biology. The mitochondria produces ATP through "
    "oxidative phosphorylation, which powers nearly all cellular activity.",
]


def main() -> None:
    settings = Settings()
    retriever = Retriever(
        embedding_provider=GeminiEmbeddingProvider(api_key=settings.api_key_for("gemini")),
        # Zero-dependency default, used here so this example runs without any
        # vector-database account. Swap in a real one the same way you'd swap
        # a provider -- construct it and pass it here instead:
        #   from requisite.rag.vectorstores.pinecone import PineconeVectorStore
        #   vector_store=PineconeVectorStore(api_key="...", index_name="demo", dimension=768)
        #   from requisite.rag.vectorstores.weaviate import WeaviateVectorStore
        #   vector_store=WeaviateVectorStore(url="...", api_key="...")
        vector_store=InMemoryVectorStore(),
    )

    retriever.add_texts(DOCS)

    # Direct use, no agent involved:
    results = retriever.retrieve("What is MCP?", top_k=1)
    print("Direct retrieval:")
    for scored_chunk in results:
        print(f"  [{scored_chunk.score:.3f}] {scored_chunk.chunk.text}")

    # Exposed as a capability -- an agent can't tell this apart from a
    # native tool or an MCP-backed one.
    capabilities.register("knowledge_base", retriever.as_tool())

    # No provider= given: uses whatever DEFAULT_PROVIDER / MODEL you've
    # configured in .env, so this runs unmodified regardless of which
    # chat provider you use.
    agent = Agent(name="Assistant")
    agent.requires("knowledge_base")

    agent_result = agent.run("What is MCP, according to the knowledge base?")
    print("\nAgent using the knowledge_base capability:")
    print(agent_result.content)


def bm25_example() -> None:
    """Keyword-only retrieval, zero dependency -- no embedding provider or
    API calls needed. Good for exact matches (ids, codes) dense embeddings
    can miss.
    """
    print("\n--- BM25Retriever (keyword-only) ---")
    retriever = BM25Retriever()
    retriever.add_texts(DOCS)
    results = retriever.retrieve("REQ-4471 refund", top_k=1)
    print(f"  [{results[0].score:.3f}] {results[0].chunk.text}")


def hybrid_and_rerank_example() -> None:
    """HybridRetriever fuses dense + BM25 results via Reciprocal Rank
    Fusion, so both a keyword-only query and a semantic-paraphrase query
    (no shared words with the doc at all) find the right chunk from the
    same index. LLMReranker then re-scores a candidate list with one
    structured-output LLM call, and LLMContextCompressor shrinks each
    surviving passage down to just the text relevant to the query.
    """
    print("\n--- HybridRetriever (dense + BM25, fused) ---")
    settings = Settings()
    retriever = HybridRetriever(
        embedding_provider=GeminiEmbeddingProvider(api_key=settings.api_key_for("gemini")),
        vector_store=InMemoryVectorStore(),
    )
    retriever.add_texts(DOCS)

    keyword_hit = retriever.retrieve("REQ-4471 refund status", top_k=1)[0]
    print(f"  keyword query  -> [{keyword_hit.score:.3f}] {keyword_hit.chunk.text}")

    semantic_hit = retriever.retrieve("Where is the famous Parisian iron tower?", top_k=1)[0]
    print(f"  semantic query -> [{semantic_hit.score:.3f}] {semantic_hit.chunk.text}")

    print("\n--- LLMReranker (listwise, one structured-output call) ---")
    query = "what powers a biological cell?"
    candidates = retriever.retrieve(query, top_k=5)
    reranker = LLMReranker(provider="gemini", model="gemini-3.1-flash-lite")
    reranked = reranker.rerank(query, candidates, top_k=2)
    print(f"  [{reranked[0].score:.1f}] {reranked[0].chunk.text}")

    print("\n--- LLMContextCompressor (shrinks each passage to what's relevant) ---")
    compressor = LLMContextCompressor(provider="gemini", model="gemini-3.1-flash-lite")
    compressed = compressor.compress(query, reranked)
    for scored_chunk in compressed:
        print(f"  [{scored_chunk.score:.1f}] {scored_chunk.chunk.text}")


if __name__ == "__main__":
    main()
    bm25_example()
    hybrid_and_rerank_example()
