"""
Agent example: an agent that autonomously decides to call a tool and
loops until it has a final answer.

Run with:
    OPENAI_API_KEY=sk-... python examples/agent_example.py
"""

from requisite import Agent
from requisite.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 22C in {city}."


def main() -> None:
    agent = Agent(
        name="Weather Agent",
        provider="gemini",
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant. Use the get_weather tool when needed.",
    )

    result = agent.run("What's the weather like in Tokyo? Answer in one sentence.")
    print(result.content)
    print(f"Tools used: {result.tool_calls_executed}, iterations: {result.iterations}")


if __name__ == "__main__":
    main()
