"""
First-party MCP servers as default capability providers.

Shows the two functions from `requisite.mcp.defaults`:

1. `register_github_mcp_capability` -- a ready-to-use provider backed by
   GitHub's official remote MCP server, registered at a higher priority
   than the built-in unauthenticated `search_github` REST resolver
   (see ADR-0020) so it takes over automatically when `GITHUB_TOKEN` is
   configured, with zero-code fallback otherwise.
2. `register_mcp_capability` -- the generic mechanism, illustrated here
   against a Postgres-shaped MCP server reading `DATABASE_URL`. There is
   no first-party default for databases (see ADR-0023) -- swap the
   `command`/`args` below for whatever database MCP server you use.

Run with:
    GITHUB_TOKEN=... python examples/mcp_default_capabilities.py
"""

import os

from requisite import Agent
from requisite.capabilities import default_registry as capabilities
from requisite.mcp import MCPClient, register_github_mcp_capability, register_mcp_capability


def main() -> None:
    # --- GitHub: a ready-to-use first-party default -----------------------
    registered = register_github_mcp_capability(capabilities)
    if registered:
        print("Registered GitHub's remote MCP server as the 'github' capability.")
    else:
        print(
            "GITHUB_TOKEN not set -- 'github' still resolves, but falls back to "
            "the unauthenticated REST resolver (search_github)."
        )

    agent = Agent(name="Assistant", provider="gemini")
    agent.requires("github")
    result = agent.run("Use the github tool to find a popular Python LLM agent framework.")
    print(result.content)

    # --- Databases: no first-party default, same mechanism ----------------
    # There's no single canonical database MCP server to hardcode as a
    # default (see ADR-0023) -- wire up whichever one your application
    # uses with the same `register_mcp_capability` helper GitHub's own
    # provider is built on:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        postgres = MCPClient.stdio(
            name="postgres",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres", database_url],
        )
        register_mcp_capability(
            capabilities,
            postgres,
            tool_name="query",  # replace with whatever tool your server exposes
            capability="database",
            priority=10,
        )
        agent.requires("database")
    else:
        print("DATABASE_URL not set -- skipping the database capability example.")


if __name__ == "__main__":
    main()
