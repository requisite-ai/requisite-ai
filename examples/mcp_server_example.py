"""
MCP server example: expose Requisite tools, an agent, a resource, and a
prompt as an MCP server.

The reverse direction of examples/mcp_example.py (which *connects to* an
MCP server) -- this script *is* one. Run it directly to serve over
stdio, or point requisite.mcp.MCPClient.stdio() at it from another
process/script to see the full round trip:

    from requisite.mcp import MCPClient
    client = MCPClient.stdio(name="requisite-demo", command="python",
                              args=["examples/mcp_server_example.py"])
    tools = client.discover_tools()
    resources = client.discover_resources()
    prompts = client.discover_prompts()

Run standalone with:
    GEMINI_API_KEY=... python examples/mcp_server_example.py
"""

from requisite import Agent
from requisite.core.interfaces import Message, Role
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

    server.add_resource(
        "memo://readme",
        name="readme",
        description="A short note about this demo server.",
        mime_type="text/plain",
        content="This MCP server is served by Requisite -- see examples/mcp_server_example.py.",
    )

    server.add_prompt(
        "weather_report",
        description="Ask for a one-sentence weather report for a city.",
        render=lambda args: [
            Message(role=Role.USER, content=f"What's the weather like in {args['city']}?")
        ],
    )

    server.run_stdio()


if __name__ == "__main__":
    main()
