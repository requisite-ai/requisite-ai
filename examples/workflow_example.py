"""
Multi-agent workflow example.

Run with:
    OPENAI_API_KEY=sk-... python examples/workflow_example.py
"""

from requisite import Agent, Workflow


def main() -> None:
    research = Agent(
        name="Researcher",
        provider="gemini",
        system_prompt="You research topics and produce concise, factual bullet points.",
    )
    writer = Agent(
        name="Writer",
        provider="gemini",
        system_prompt="You turn research notes into a polished, engaging short summary.",
    )

    workflow = Workflow()
    workflow.add(research)
    workflow.add(writer)

    result = workflow.run("Research AI trends and write a short summary.")
    print("--- sequential (native) ---")
    print(result.content)

    # Run the same two agents in parallel against the same prompt instead.
    workflow.parallel()
    parallel_result = workflow.run("What is retrieval-augmented generation?")
    print("\n--- parallel (native) ---")
    print(parallel_result.content)

    # Switch the execution engine to langgraph -- same add()/run() API.
    # Requires: pip install langgraph
    workflow.sequential().use_langgraph()
    try:
        langgraph_result = workflow.run("Research AI trends and write a short summary.")
        print("\n--- sequential (langgraph) ---")
        print(langgraph_result.content)
    except Exception as exc:
        print(f"\nlanggraph backend not available: {exc}")


if __name__ == "__main__":
    main()
