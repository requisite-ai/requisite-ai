"""
Capability resolution: ``agent.requires("weather")`` instead of
``agent.use_tool(specific_weather_impl)``.

Declaring a capability by name -- rather than binding to one concrete
tool -- lets the framework (or the surrounding plugin ecosystem) swap in
whichever implementation is available: a native Python function today,
an MCP server or paid API tomorrow, with zero changes to agent code.

``default_registry`` ships pre-populated with reference implementations
for ``"filesystem"``, ``"weather"``, and ``"internet_search"`` (see
:mod:`requisite.capabilities.resolvers`) so ``Agent.requires(...)`` is
demonstrable with no extra setup. Register your own, higher-priority
providers to have them take over automatically.
"""

from requisite.capabilities.registry import CapabilityProvider, CapabilityRegistry
from requisite.capabilities.resolvers import register_default_capabilities

default_registry = CapabilityRegistry()
register_default_capabilities(default_registry)

__all__ = [
    "CapabilityProvider",
    "CapabilityRegistry",
    "default_registry",
    "register_default_capabilities",
]
