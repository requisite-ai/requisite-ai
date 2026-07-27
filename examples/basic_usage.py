"""
Basic usage example.

Run with:
    OPENAI_API_KEY=sk-... python examples/basic_usage.py

Or copy `.env.example` to `.env` and fill in your keys.
"""

from requisite import AI


def main() -> None:
    # Uses DEFAULT_PROVIDER / MODEL from environment or .env (defaults to OpenAI).
    ai = AI()
    print(f"Using: {ai}")
    print(ai.chat("Explain LangGraph in one sentence."))

    # Switching providers is a configuration change, not a code change.
    gemini_ai = AI(provider="gemini", model="gemini-3.1-flash-lite")
    print(f"Using: {gemini_ai}")
    print(gemini_ai.chat("Explain LangGraph in one sentence."))

    # Full response object, with usage and provider metadata.
    response = ai.chat_response("What is retrieval-augmented generation?")
    print(response.content)
    print(
        f"model={response.model} provider={response.provider} tokens={response.usage.total_tokens}"
    )


if __name__ == "__main__":
    main()
