"""
RAG example.

Shows building a retriever (embeddings + in-memory vector store),
adding documents, and exposing it to an agent as a capability so
`agent.requires("knowledge_base")` works exactly like any other capability.

Run with whichever provider(s) you've configured in `.env` -- this script
doesn't hardcode a provider. Embeddings and chat completion are picked
independently: you can chat with Gemini while embedding with OpenAI, or
any other combination, since a retriever's embedding provider has nothing
to do with an agent's chat provider.

    cp .env.example .env   # then fill in at least one of:
    #   OPENAI_API_KEY=...        (for OpenAI embeddings and/or chat)
    #   GEMINI_API_KEY=...        (for Gemini embeddings and/or chat)
    python examples/rag_example.py
"""

from requisite import Agent
from requisite.capabilities import default_registry as capabilities
from requisite.config.settings import Settings
from requisite.rag import Retriever
from requisite.rag.embeddings import GeminiEmbeddingProvider
from requisite.rag.vectorstores import InMemoryVectorStore


def main() -> None:
    settings = Settings()
    retriever = Retriever(
        embedding_provider=GeminiEmbeddingProvider(api_key=settings.api_key_for("gemini")),
        vector_store=InMemoryVectorStore(),
    )

    retriever.add_texts(
        [
            "Requisite is a provider-agnostic Python framework for building AI applications.",
            "Paris is the capital of France. The Eiffel Tower was completed in 1889.",
            "The Model Context Protocol (MCP) lets AI applications connect to external tools and data sources.",
        ]
    )

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


if __name__ == "__main__":
    main()
