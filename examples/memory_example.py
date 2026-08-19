"""
Conversation memory example: persisting history across separate run()
calls, with three interchangeable backends.

Run with:
    cp .env.example .env   # then fill in:
    #   GEMINI_API_KEY=...
    #   DEFAULT_PROVIDER=gemini
    #   MODEL=gemini-3.1-flash-lite
    python examples/memory_example.py
"""

import tempfile
from pathlib import Path

from requisite import Agent
from requisite.memory import InProcessMemory, VectorMemory
from requisite.memory.sqlite import SQLiteMemory
from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider
from requisite.rag.vectorstores import InMemoryVectorStore


def in_process_example() -> None:
    print("--- InProcessMemory (lost on restart, zero setup) ---")
    memory = InProcessMemory()
    agent = Agent(name="Assistant", memory=memory, session_id="user-42")

    print(agent.run("My favorite language is Python.").content)
    print(agent.run("What's my favorite language?").content)


def sqlite_example() -> None:
    print("\n--- SQLiteMemory (persists across restarts) ---")
    db_path = str(Path(tempfile.mkdtemp()) / "conversations.db")

    first_run = Agent(name="Assistant", memory=SQLiteMemory(db_path=db_path), session_id="user-42")
    print(first_run.run("My favorite color is teal.").content)

    # A brand new Agent + SQLiteMemory instance, pointed at the same file --
    # simulates the process restarting. It still recalls the earlier turn.
    second_run = Agent(name="Assistant", memory=SQLiteMemory(db_path=db_path), session_id="user-42")
    print(second_run.run("What's my favorite color?").content)

    # RedisMemory has the same shape as SQLiteMemory, not run here since it
    # needs a real Redis (or Redis-compatible, e.g. Memurai on Windows)
    # server -- sessions are shared across processes/machines instead of
    # one file:
    #
    #   from requisite.memory.redis import RedisMemory
    #   agent = Agent(
    #       name="Assistant",
    #       memory=RedisMemory(),  # defaults to redis://127.0.0.1:6379/0
    #       session_id="user-42",
    #   )


def vector_memory_example() -> None:
    print("\n--- VectorMemory (semantic recall, not just chronological) ---")
    # VectorMemory composes a plain chronological backend (InProcessMemory
    # by default -- swap in SQLiteMemory/RedisMemory for persistence) with
    # an embedding provider + vector store, so it's a drop-in BaseMemory
    # (Agent.run() sees plain chronological load()/append() like every
    # other backend) that *also* supports semantic top-k recall beyond
    # what any other backend can do.
    memory = VectorMemory(
        embedding_provider=GeminiEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
    )
    agent = Agent(name="Assistant", memory=memory, session_id="user-42")

    agent.run("My favorite color is teal.")
    agent.run("My favorite food is ramen.")
    agent.run("I'm planning a trip to Japan next spring.")

    print("Chronological history (memory.load):")
    for message in memory.load("user-42"):
        print(f"  [{message.role.value}] {message.content}")

    # load_relevant() is *not* on Agent -- it's an explicit, opt-in call
    # application code makes to pull the most semantically relevant past
    # turns for a query, distinct from the full chronological history above.
    print("\nSemantically relevant to 'What should I eat?':")
    for message in memory.load_relevant("user-42", "What should I eat?", top_k=1):
        print(f"  [{message.role.value}] {message.content}")


if __name__ == "__main__":
    in_process_example()
    sqlite_example()
    vector_memory_example()
