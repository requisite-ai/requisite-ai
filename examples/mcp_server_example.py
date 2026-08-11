"""
MCP server example: expose Requisite tools and an agent as an MCP server.

The reverse direction of examples/mcp_example.py (which *connects to* an
MCP server) -- this script *is* one. Run it directly to serve over
stdio, or point requisite.mcp.MCPClient.stdio() at it from another
process/script to see the full round trip:

    from requisite.mcp import MCPClient
    client = MCPClient.stdio(name="requisite-demo", command="python",
                              args=["examples/mcp_server_example.py"])
    tools = client.discover_tools()

Run standalone with:
    GEMINI_API_KEY=... python examples/mcp_server_example.py
"""

from requisite import Agent
from requisite.mcp import MCPServer
from requisite.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 22C in {city}."


def main() -> None:
    assistant = Agent(
        name="assistant",
        provider="gemini",
        system_prompt="You are a helpful assistant. Answer in one sentence.",
    )

    server = MCPServer(name="requisite-demo", tools=[add, get_weather], agents=[assistant])
    server.run_stdio()


if __name__ == "__main__":
    main()
