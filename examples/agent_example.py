"""
Agent example: an agent that autonomously decides to call a tool and
loops until it has a final answer.

Run with:
    OPENAI_API_KEY=sk-... python examples/agent_example.py
"""

import asyncio
import time

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


@tool
async def check_service_status(service: str) -> str:
    """Check whether a named service is currently operational. Takes ~2s."""
    await asyncio.sleep(2.0)
    return f"{service}: operational"


async def async_concurrent_tool_calls_example() -> None:
    """`Agent.arun()` runs a turn's independent tool calls concurrently
    (`asyncio.gather`), not one at a time -- if the model asks for
    `check_service_status` on three services in a single turn, all three
    ~2s calls overlap instead of serializing to ~6s. `Agent.run()` (sync)
    has no such concurrency to exploit -- see docs/adr and the 0.6.2
    CHANGELOG entry for the "why".
    """
    agent = Agent(
        name="Ops Agent",
        provider="gemini",
        tools=[check_service_status],
        system_prompt=(
            "You check the status of services the user names, one tool call per "
            "service, then summarize the results."
        ),
    )

    start = time.monotonic()
    result = await agent.arun(
        "Check the status of these three independent services: 'billing', 'auth', and 'search'."
    )
    elapsed = time.monotonic() - start

    print(f"\nTools used: {result.tool_calls_executed}")
    print(
        f"Elapsed: {elapsed:.1f}s (sequential would need ~{2 * len(result.tool_calls_executed)}s)"
    )
    print(result.content)


if __name__ == "__main__":
    main()
    asyncio.run(async_concurrent_tool_calls_example())
