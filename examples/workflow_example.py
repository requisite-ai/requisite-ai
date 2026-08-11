"""
Multi-agent workflow example.

Run with:
    GEMINI_API_KEY=... python examples/workflow_example.py

This script builds four agents (Researcher, Writer, Planner, Supervisor)
that all call the same Gemini API key -- on a free-tier key (15
requests/minute) their combined call rate can exceed the real quota even
though each agent looks fine on its own. One `RateLimiter` shared across
all four keeps the script under that quota by waiting for capacity
instead of letting Gemini reject the call with a 429. See the "Rate
limiting" section of README.md and docs/adr/0008-rate-limiting.md.
"""

from requisite import Agent, RateLimiter, Workflow

# Shared across every agent below since they all draw on the same Gemini
# API key/quota -- a limiter scoped to just one agent wouldn't help, since
# the quota is enforced against their combined call rate, not each one
# individually.
shared_rate_limit = RateLimiter(requests_per_minute=15)


def main() -> None:
    research = Agent(
        name="Researcher",
        provider="gemini",
        system_prompt="You research topics and produce concise, factual bullet points.",
        rate_limiter=shared_rate_limit,
    )
    writer = Agent(
        name="Writer",
        provider="gemini",
        system_prompt="You turn research notes into a polished, engaging short summary.",
        rate_limiter=shared_rate_limit,
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
    except Exception as exc:  # noqa: BLE001
        print(f"\nlanggraph backend not available: {exc}")

    # Reflection: a single agent critiques and revises its own output.
    reflection_workflow = Workflow().reflection()
    reflection_workflow.add(writer)
    reflection_result = reflection_workflow.run(
        "Write a one-sentence tagline for an open-source AI framework.", max_rounds=3
    )
    print("\n--- reflection (native) ---")
    print(reflection_result.content)
    print(f"(rounds of self-critique/revision: {len(reflection_result.steps)})")

    # Planner: the first agent decomposes the task into a plan; the rest
    # are workers it assigns subtasks to, by name.
    planner_agent = Agent(
        name="Planner",
        provider="gemini",
        system_prompt="You break tasks into an ordered plan of subtasks for your team.",
        rate_limiter=shared_rate_limit,
    )
    planner_workflow = Workflow().planner()
    planner_workflow.add(planner_agent).add(research).add(writer)
    planner_result = planner_workflow.run(
        "Research what MCP (Model Context Protocol) is and write a short explainer."
    )
    print("\n--- planner (native) ---")
    print(planner_result.content)
    print(
        f"(plan had {len(planner_result.steps)} step(s): "
        f"{[s.agent_name for s in planner_result.steps]})"
    )

    # Supervisor: the first agent delegates subtasks to named workers one
    # at a time, deciding for itself when the task is complete.
    supervisor_agent = Agent(
        name="Supervisor",
        provider="gemini",
        system_prompt="You coordinate a small team, delegating one subtask at a time.",
        rate_limiter=shared_rate_limit,
    )
    supervisor_workflow = Workflow().supervisor()
    supervisor_workflow.add(supervisor_agent).add(research).add(writer)
    supervisor_result = supervisor_workflow.run(
        "Research what LangGraph is and write a short summary."
    )
    print("\n--- supervisor (native) ---")
    print(supervisor_result.content)
    print(f"(delegated to: {[s.agent_name for s in supervisor_result.steps]})")

    # Critic: a separate agent reviews the writer's draft, distinct from
    # reflection's single agent critiquing itself.
    critic_agent = Agent(
        name="Critic",
        provider="gemini",
        system_prompt=(
            "You critique short taglines for clarity and punch. If a tagline is "
            "already excellent, respond with exactly NO_CHANGES_NEEDED."
        ),
        rate_limiter=shared_rate_limit,
    )
    critic_workflow = Workflow().critic()
    critic_workflow.add(writer).add(critic_agent)
    critic_result = critic_workflow.run(
        "Write a one-sentence tagline for an open-source AI framework.", max_rounds=3
    )
    print("\n--- critic (native) ---")
    print(critic_result.content)
    print(f"(rounds of generator/critic exchange: {len(critic_result.steps)})")

    # Consensus: several agents answer independently, then the first
    # agent synthesizes their answers into one.
    consensus_writer_a = Agent(name="WriterA", provider="gemini", rate_limiter=shared_rate_limit)
    consensus_writer_b = Agent(name="WriterB", provider="gemini", rate_limiter=shared_rate_limit)
    consensus_workflow = Workflow().consensus()
    consensus_workflow.add(writer).add(consensus_writer_a).add(consensus_writer_b)
    consensus_result = consensus_workflow.run(
        "In one sentence, what is the single biggest benefit of retrieval-augmented generation?"
    )
    print("\n--- consensus (native) ---")
    print(consensus_result.content)
    print(f"(synthesized from: {[s.agent_name for s in consensus_result.steps[:-1]]})")


if __name__ == "__main__":
    main()
