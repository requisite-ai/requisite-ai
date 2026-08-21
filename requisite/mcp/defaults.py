"""
First-party MCP servers as default capability providers.

Unlike :mod:`requisite.capabilities.resolvers` -- whose reference
implementations are plain Python functions, cheap enough to register
eagerly at import time (see ``requisite/capabilities/__init__.py``) --
registering an MCP server as a capability provider means a real
subprocess/network handshake (:meth:`~requisite.mcp.base.BaseMCPClient.discover_tools`,
per ADR-0004's per-call-connection design). That cost, and the dependency
on an external server actually being reachable, means nothing here is
ever auto-registered at import time. Call these functions explicitly from
application setup code -- the same way you'd register any other
higher-priority provider per ``resolvers.py``'s own docstring.

See ``docs/adr/0023-mcp-default-capability-providers.md`` for the full
design reasoning.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from requisite.capabilities.registry import AvailabilityCheck, CapabilityRegistry
from requisite.mcp.base import BaseMCPClient
from requisite.mcp.client import MCPClient

logger = logging.getLogger("requisite.mcp.defaults")

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
"""GitHub's official remote MCP server (Streamable HTTP)."""


def register_mcp_capability(
    registry: CapabilityRegistry,
    client: BaseMCPClient,
    *,
    tool_name: str,
    capability: str,
    priority: int = 0,
    is_available: Optional[AvailabilityCheck] = None,
) -> bool:
    """Register one tool from an MCP server as a capability provider.

    Unlike :meth:`~requisite.mcp.base.BaseMCPClient.register_as_capability`,
    ``tool_name`` and ``capability`` may differ -- the discovered tool is
    renamed to ``capability`` (the same ``model_copy(update={"name": ...})``
    pattern :meth:`~requisite.agents.agent.Agent.requires` already uses)
    before being registered, since real-world MCP servers rarely expose a
    tool literally named e.g. ``"github"`` or ``"database"``.

    This is the generic mechanism for wiring up *any* first-party or
    third-party MCP server -- including a database MCP server, for which
    there is no single canonical default to hardcode. See
    :func:`register_github_mcp_capability` for a concrete, ready-to-use
    provider.

    Parameters
    ----------
    registry:
        The :class:`~requisite.capabilities.registry.CapabilityRegistry`
        to register into.
    client:
        A configured (but not yet connected) MCP client, e.g.
        ``MCPClient.stdio(...)`` or ``MCPClient.http(...)``.
    tool_name:
        The name of the tool as exposed by the MCP server.
    capability:
        The capability name to register it under.
    priority:
        Passed through to ``CapabilityRegistry.register``.
    is_available:
        Passed through to ``CapabilityRegistry.register``.

    Returns
    -------
    bool
        ``True`` if registration succeeded. ``False`` if the server
        couldn't be reached, or doesn't expose a tool named ``tool_name``
        -- logged as a warning either way, but never raised. This
        deliberately diverges from ``register_as_capability``'s
        raise-on-missing behavior: this function is meant for optional
        "default provider" wiring that must not crash application
        startup over an unreachable or misconfigured optional server.
    """
    try:
        tools = client.discover_tools()
    except Exception:
        logger.warning(
            "Could not reach MCP server '%s' to register capability '%s'; skipping.",
            client.name,
            capability,
            exc_info=True,
        )
        return False

    for discovered in tools:
        if discovered.name == tool_name:
            tool = (
                discovered
                if discovered.name == capability
                else discovered.model_copy(update={"name": capability})
            )
            registry.register(
                capability,
                tool,
                provider_name=f"mcp:{client.name}",
                priority=priority,
                is_available=is_available,
            )
            return True

    logger.warning(
        "MCP server '%s' does not expose a tool named '%s' for capability '%s'; "
        "skipping. Available: %s",
        client.name,
        tool_name,
        capability,
        [t.name for t in tools],
    )
    return False


def register_github_mcp_capability(
    registry: CapabilityRegistry,
    *,
    token: Optional[str] = None,
    tool_name: str = "search_repositories",
    priority: int = 10,
) -> bool:
    """Register GitHub's official remote MCP server as the ``"github"``
    capability, at a higher priority than the unauthenticated REST
    resolver (:func:`requisite.capabilities.resolvers.search_github`,
    registered at ``priority=0``) -- so it takes over automatically via
    ``CapabilityRegistry``'s existing priority resolution, per
    ADR-0020's follow-up commitment and ADR-0023.

    Parameters
    ----------
    registry:
        The :class:`~requisite.capabilities.registry.CapabilityRegistry`
        to register into.
    token:
        A GitHub personal access token. Defaults to the ``GITHUB_TOKEN``
        environment variable.
    tool_name:
        The name of the repository-search tool as exposed by GitHub's
        MCP server. Exposed as a parameter so a future rename on GitHub's
        side is a one-line fix, not a code change.
    priority:
        Registration priority. Defaults to ``10``, above the REST
        resolver's ``0``.

    Returns
    -------
    bool
        ``False`` without attempting any connection if no token is
        available (neither ``token=`` nor ``GITHUB_TOKEN`` is set) --
        GitHub's MCP endpoint requires authentication, so there is no
        unauthenticated fallback to attempt. Otherwise, the return value
        of :func:`register_mcp_capability`.
    """
    resolved_token = token or os.environ.get("GITHUB_TOKEN")
    if not resolved_token:
        return False

    client = MCPClient.http(
        name="github",
        url=GITHUB_MCP_URL,
        headers={"Authorization": f"Bearer {resolved_token}"},
    )
    return register_mcp_capability(
        registry,
        client,
        tool_name=tool_name,
        capability="github",
        priority=priority,
    )
