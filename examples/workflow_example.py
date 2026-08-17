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

    # Same supervisor strategy, run on the langgraph backend instead: a
    # real conditional graph (add_conditional_edges + a loop-back cycle)
    # rather than a Python loop -- see docs/adr/0016-langgraph-branching.md.
    # Requires: pip install langgraph
    try:
        supervisor_workflow.use_langgraph()
        supervisor_langgraph_result = supervisor_workflow.run(
            "Research what LangGraph is and write a short summary."
        )
        print("\n--- supervisor (langgraph) ---")
        print(supervisor_langgraph_result.content)
        print(f"(delegated to: {[s.agent_name for s in supervisor_langgraph_result.steps]})")
    except Exception as exc:  # noqa: BLE001
        print(f"\nlanggraph backend not available: {exc}")

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

    # Debate: a moderator judges two agents debating a real trade-off,
    # each round seeing the other's arguments from the previous round.
    debater_for = Agent(
        name="ForMonolith",
        provider="gemini",
        system_prompt="You argue FOR building a new backend as a monolith. Be concise.",
        rate_limiter=shared_rate_limit,
    )
    debater_against = Agent(
        name="ForMicroservices",
        provider="gemini",
        system_prompt="You argue FOR building a new backend as microservices. Be concise.",
        rate_limiter=shared_rate_limit,
    )
    moderator = Agent(
        name="Moderator",
        provider="gemini",
        system_prompt="You judge technical debates and deliver a balanced final verdict.",
        rate_limiter=shared_rate_limit,
    )
    debate_workflow = Workflow().debate()
    debate_workflow.add(moderator).add(debater_for).add(debater_against)
    debate_result = debate_workflow.run(
        "Should a small team building a new product start with a monolith or microservices?",
        max_rounds=2,
    )
    print("\n--- debate (native) ---")
    print(debate_result.content)
    print(f"(debate steps + verdict: {len(debate_result.steps)})")

    # Map-reduce: each mapper summarizes one item independently
    # (round-robin across mappers), then the reducer combines them.
    mapper_a = Agent(name="MapperA", provider="gemini", rate_limiter=shared_rate_limit)
    mapper_b = Agent(name="MapperB", provider="gemini", rate_limiter=shared_rate_limit)
    reducer = Agent(
        name="Reducer",
        provider="gemini",
        system_prompt="You combine several short summaries into one coherent overview.",
        rate_limiter=shared_rate_limit,
    )
    map_reduce_workflow = Workflow().map_reduce()
    map_reduce_workflow.add(reducer).add(mapper_a).add(mapper_b)
    map_reduce_result = map_reduce_workflow.run(
        "Summarize the key idea of each note in one sentence each.",
        map_items=[
            "RAG grounds LLM answers in retrieved documents to reduce hallucination.",
            "MCP standardizes how AI applications connect to external tools and data.",
            "LangGraph models agent workflows as graphs instead of linear chains.",
        ],
    )
    print("\n--- map_reduce (native) ---")
    print(map_reduce_result.content)
    print(f"(mapped by: {[s.agent_name for s in map_reduce_result.steps[:-1]]})")

    # Hierarchical: a top-level coordinator delegates to either a plain
    # agent or a named sub-team Workflow -- delegating to a Workflow
    # runs whatever strategy it's configured with, so a "team" can
    # itself be a supervisor delegating further, giving real 2-level
    # recursive hierarchy.
    team_lead = Agent(
        name="ResearchTeamLead",
        provider="gemini",
        system_prompt="You coordinate a small research team, delegating one subtask at a time.",
        rate_limiter=shared_rate_limit,
    )
    team_researcher = Agent(
        name="TeamResearcher", provider="gemini", rate_limiter=shared_rate_limit
    )
    research_team = Workflow(name="ResearchTeam").supervisor()
    research_team.add(team_lead).add(team_researcher)

    director = Agent(
        name="Director",
        provider="gemini",
        system_prompt=(
            "You coordinate a small organization, delegating one subtask at a time "
            "to either the Writer or the ResearchTeam."
        ),
        rate_limiter=shared_rate_limit,
    )
    hierarchical_workflow = Workflow().hierarchical()
    hierarchical_workflow.add(director).add(writer).add(research_team)
    hierarchical_result = hierarchical_workflow.run(
        "Research what the Model Context Protocol (MCP) is, then write a two-sentence summary.",
        max_rounds=4,
    )
    print("\n--- hierarchical (native) ---")
    print(hierarchical_result.content)
    print(
        "(delegated to: "
        f"{[getattr(s, 'agent_name', 'ResearchTeam (nested Workflow)') for s in hierarchical_result.steps]})"
    )

    # Tree-of-thoughts: the first agent evaluates candidate reasoning steps
    # the remaining agents (thinkers) generate; each level, `breadth`
    # candidates are scored together and pruned to the top `beam_width`
    # before continuing, for up to `max_depth` levels.
    tot_evaluator = Agent(
        name="Evaluator",
        provider="gemini",
        system_prompt=(
            "You evaluate candidate reasoning steps for a math word problem, "
            "scoring how promising and correct each one is."
        ),
        rate_limiter=shared_rate_limit,
    )
    tot_thinker = Agent(
        name="Thinker",
        provider="gemini",
        system_prompt="You propose the next reasoning step to solve a math word problem.",
        rate_limiter=shared_rate_limit,
    )
    tot_workflow = Workflow().tree_of_thoughts()
    tot_workflow.add(tot_evaluator).add(tot_thinker)
    tot_result = tot_workflow.run(
        "A train travels 60 miles in the first hour and 90 miles in the "
        "second hour. What is its average speed over the two hours?",
        breadth=3,
        beam_width=2,
        max_depth=3,
    )
    print("\n--- tree_of_thoughts (native) ---")
    print(tot_result.content)
    print(
        f"(generated {len(tot_result.steps)} candidate thoughts across up to 3 levels, "
        f"pruned to a beam of 2 each level)"
    )


if __name__ == "__main__":
    main()
