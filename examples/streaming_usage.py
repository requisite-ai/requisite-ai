"""
Streaming and async usage example.

Run with:
    OPENAI_API_KEY=sk-... python examples/streaming_usage.py
"""

import asyncio

from requisite import AI


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


if __name__ == "__main__":
    sync_streaming_example()
    asyncio.run(async_example())
