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
from requisite.memory import InProcessMemory
from requisite.memory.sqlite import SQLiteMemory


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


if __name__ == "__main__":
    in_process_example()
    sqlite_example()
