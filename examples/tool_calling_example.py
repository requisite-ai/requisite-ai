"""
Tool calling example.

Run with:
    OPENAI_API_KEY=sk-... python examples/tool_calling_example.py
"""

from requisite import AI
from requisite.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real application this would call a weather API.
    return f"It's sunny and 22C in {city}."


def main() -> None:
    ai = AI()

    # The model decides whether/which tool to call; inspect tool_calls
    # on the full response to see what it requested.
    response = ai.chat_response("What's the weather like in Tokyo?", tools=[get_weather])

    if response.has_tool_calls:
        for call in response.tool_calls:
            print(f"Model requested: {call.name}({call.arguments})")
            if call.name == "get_weather":
                result = get_weather.tool.execute(**call.arguments)
                print(f"Tool result: {result}")
    else:
        print(response.content)


if __name__ == "__main__":
    main()
