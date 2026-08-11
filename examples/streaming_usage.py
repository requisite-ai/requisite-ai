"""
Streaming and async usage example.

Run with:
    OPENAI_API_KEY=sk-... python examples/streaming_usage.py
"""

import asyncio

from requisite import AI
from requisite.tools import tool


def sync_streaming_example() -> None:
    ai = AI()
    print("--- sync streaming ---")
    for token in ai.stream("Write a haiku about distributed systems."):
        print(token, end="", flush=True)
    print()


async def async_example() -> None:
    ai = AI(provider="gemini", model="gemini-3.1-flash-lite")

    print("--- async chat ---")
    text = await ai.achat("Give me one fun fact about the ocean.")
    print(text)

    print("--- async streaming ---")
    async for token in ai.astream("Count from one to five, one number per line."):
        print(token, end="", flush=True)
    print()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny and 22C in {city}."


def streaming_with_tools_example() -> None:
    """`AI.stream()`/`.astream()` accept `tools=` but only ever yield text,
    same as `AI.chat()` -- use `stream_response()`/`.astream_response()`
    for the full `StreamChunk` sequence, which carries `tool_calls` once
    the model completes one, typically on the final chunk.
    """
    ai = AI(provider="gemini", model="gemini-3.1-flash-lite")

    print("--- stream_response(tools=...) ---")
    for chunk in ai.stream_response("What's the weather in Tokyo?", tools=[get_weather]):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
        if chunk.has_tool_calls:
            print()
            for call in chunk.tool_calls:
                print(f"Model requested: {call.name}({call.arguments})")
    print()


if __name__ == "__main__":
    sync_streaming_example()
    streaming_with_tools_example()
    asyncio.run(async_example())
