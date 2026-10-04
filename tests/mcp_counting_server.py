"""Tiny stdio MCP server for real-subprocess tests.

Appends one line per process start to the file named by ``MCP_SPAWN_LOG``, so a
test can count how many server subprocesses a client really spawned.
"""

import os

from requisite.mcp import MCPServer
from requisite.tools import tool

with open(os.environ["MCP_SPAWN_LOG"], "a", encoding="utf-8") as _log:
    _log.write(f"spawn {os.getpid()}\n")


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    MCPServer(name="counting", tools=[add]).run_stdio()
