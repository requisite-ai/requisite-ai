"""
MCP integration example.

Shows connecting to a local (stdio) and a remote (Streamable HTTP) MCP
server, and bridging their tools into the capability system so
`agent.requires(...)` can't tell an MCP-backed tool from a native one.

Run with:
    OPENAI_API_KEY=sk-... python examples/mcp_example.py
"""

from requisite import Agent
from requisite.capabilities import default_registry as capabilities
from requisite.mcp import MCPClient


def main() -> None:
    # A local MCP server run as a subprocess over stdio. Replace the
    # command/args with any real MCP server you have available.
    filesystem = MCPClient.stdio(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    # Use its tools directly, without going through capabilities:
    tools = filesystem.discover_tools()
    print(f"Discovered {len(tools)} tool(s) from the filesystem server:")
    for discovered_tool in tools:
        print(f"  - {discovered_tool.name}: {discovered_tool.description}")

    # Or bridge one into the capability system, so it's indistinguishable
    # from a native tool to any agent that requires it:
    if tools:
        first_tool_name = tools[0].name
        filesystem.register_as_capability(capabilities, capability=first_tool_name)

        agent = Agent(name="Assistant", provider="openai")
        agent.requires(first_tool_name)
        result = agent.run(
            f"Use the {first_tool_name} tool to help me, then summarize what it returned."
        )
        print(result.content)


if __name__ == "__main__":
    main()
