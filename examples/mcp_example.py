"""
MCP integration example.

Shows connecting to a local (stdio) and a remote (Streamable HTTP) MCP
server, and bridging their tools into the capability system so
`agent.requires(...)` can't tell an MCP-backed tool from a native one.

Run with:
    GEMINI_API_KEY=... python examples/mcp_example.py
"""

import tempfile
from pathlib import Path

from requisite import Agent
from requisite.capabilities import default_registry as capabilities
from requisite.mcp import MCPClient


def main() -> None:
    # A local MCP server run as a subprocess over stdio. Replace the
    # command/args with any real MCP server you have available.
    #
    # The filesystem server validates its allowed directory at startup
    # and exits before completing the MCP handshake if it doesn't exist,
    # so use the OS temp dir rather than a hardcoded POSIX path like
    # "/tmp" (which isn't valid on Windows). A known demo file is written
    # into it so the agent has something real to read, rather than
    # guessing a filename that may not exist.
    demo_dir = Path(tempfile.gettempdir())
    demo_file = demo_dir / "requisite_mcp_demo.txt"
    demo_file.write_text("Requisite is a provider-agnostic framework for building AI agents.")

    filesystem = MCPClient.stdio(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(demo_dir)],
    )

    # Use its tools directly, without going through capabilities:
    tools = filesystem.discover_tools()
    print(f"Discovered {len(tools)} tool(s) from the filesystem server:")
    for discovered_tool in tools:
        print(f"  - {discovered_tool.name}: {discovered_tool.description}")

    # Bridge "read_file" into the capability system, so it's indistinguishable
    # from a native tool to any agent that requires it:
    filesystem.register_as_capability(capabilities, capability="read_file")

    agent = Agent(name="Assistant", provider="gemini")
    agent.requires("read_file")
    result = agent.run(
        f"Use the read_file tool to read '{demo_file.name}', then summarize what it says."
    )
    print(result.content)


if __name__ == "__main__":
    main()
