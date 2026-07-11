"""
Capability resolution: declare *what* an agent needs, not *which*
implementation provides it.

This is the framework-agnostic layer on top of tool calling. Compare:

    agent.use_tool(weather)       # bound to one specific implementation
    agent.requires("weather")     # bound to a name; resolved at runtime

A capability (e.g. ``"weather"``, ``"internet_search"``, ``"filesystem"``)
can have multiple registered providers -- a native Python implementation,
an MCP server, a cloud API, a third-party plugin. :class:`CapabilityRegistry`
picks the highest-priority *available* one at resolution time, so:

* Application code written against capability names keeps working as the
  underlying ecosystem changes (a plugin replaces a stub, an MCP server
  becomes available, a paid API key gets added) with zero code changes.
* Multiple packages can compete to provide the same capability; the
  registry -- not a hardcoded import -- decides which one wins.

This module intentionally does not try to resolve *ambiguity* for you
beyond priority + availability ordering (see :meth:`CapabilityRegistry.resolve`).
Anything more elaborate (per-environment policies, cost-based routing,
user confirmation on conflicts) is a natural extension point, not
something this v1 bakes in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from requisite.core.exceptions import CapabilityException
from requisite.tools.base import Tool
from requisite.tools.registry import ToolLike, resolve_tool_like

logger = logging.getLogger("requisite.capabilities")

AvailabilityCheck = Callable[[], bool]


def _always_available() -> bool:
    return True


@dataclass
class CapabilityProvider:
    """A single registered implementation of a capability.

    Parameters
    ----------
    capability:
        The capability name this provider implements (e.g. ``"weather"``).
    provider_name:
        Unique identifier for *this implementation* of the capability
        (e.g. ``"open-meteo"``, ``"mcp:github"``, ``"acme-plugin"``),
        distinct from the capability name itself so multiple providers
        can compete for the same capability.
    tool:
        The underlying :class:`~requisite.tools.base.Tool` that
        actually implements the capability.
    priority:
        Higher priority wins when multiple providers for the same
        capability are available. Ties are broken by registration order
        (first registered wins).
    is_available:
        Called at resolution time to check whether this provider can
        currently be used (e.g. an API key is configured, an MCP server
        is reachable). Defaults to always-available.
    """

    capability: str
    provider_name: str
    tool: Tool
    priority: int = 0
    is_available: AvailabilityCheck = field(default=_always_available)


class CapabilityRegistry:
    """Registry mapping capability names to one or more competing providers.

    Examples
    --------
    >>> from requisite.capabilities import CapabilityRegistry
    >>> registry = CapabilityRegistry()
    >>> def get_weather(city: str) -> str:
    ...     '''Get the current weather for a city.'''
    ...     return f"Sunny in {city}"
    >>> registry.register("weather", get_weather, provider_name="stub")
    >>> registry.resolve("weather").execute(city="Paris")
    'Sunny in Paris'

    Registering a higher-priority provider that takes over when available:

    >>> def get_weather_pro(city: str) -> str:
    ...     '''Paid weather provider.'''
    ...     return f"Precise forecast for {city}"
    >>> registry.register(
    ...     "weather", get_weather_pro, provider_name="pro-api", priority=10,
    ...     is_available=lambda: False,  # e.g. no API key configured
    ... )
    >>> registry.resolve("weather").execute(city="Paris")  # falls back to 'stub'
    'Sunny in Paris'
    """

    def __init__(self) -> None:
        self._providers: dict[str, list[CapabilityProvider]] = {}

    def register(
        self,
        capability: str,
        tool_or_func: ToolLike,
        *,
        provider_name: Optional[str] = None,
        priority: int = 0,
        is_available: Optional[AvailabilityCheck] = None,
    ) -> CapabilityProvider:
        """Register an implementation for a named capability.

        Parameters
        ----------
        capability:
            Capability name, e.g. ``"weather"``, ``"internet_search"``,
            ``"filesystem"``, ``"github"``. Application/agent code refers
            to this name, never to ``provider_name`` directly.
        tool_or_func:
            A :class:`~requisite.tools.base.Tool`, a plain function, or
            an ``@tool``-decorated function implementing the capability.
        provider_name:
            Identifies this specific implementation among possibly several
            competing ones. Defaults to the resolved tool's name.
        priority:
            Higher wins when more than one provider for ``capability`` is
            available at resolution time.
        is_available:
            Optional callable checked at resolution time. Use this to
            gate a provider on, e.g., an API key being configured, or a
            remote service being reachable. Defaults to always-available.

        Returns
        -------
        CapabilityProvider
            The registered provider record.
        """
        tool = resolve_tool_like(tool_or_func)
        provider = CapabilityProvider(
            capability=capability,
            provider_name=provider_name or tool.name,
            tool=tool,
            priority=priority,
            is_available=is_available or _always_available,
        )
        self._providers.setdefault(capability, []).append(provider)
        logger.debug(
            "Registered capability provider '%s' for '%s' (priority=%d)",
            provider.provider_name,
            capability,
            priority,
        )
        return provider

    def resolve(self, capability: str) -> Tool:
        """Return the best available :class:`Tool` implementing ``capability``.

        Selection is by highest ``priority`` among providers whose
        ``is_available()`` currently returns ``True``; ties go to whichever
        was registered first.

        Raises
        ------
        requisite.core.exceptions.CapabilityException
            If no provider is registered for ``capability``, or every
            registered provider is currently unavailable.
        """
        providers = self._providers.get(capability)
        if not providers:
            raise CapabilityException(
                f"No provider registered for capability '{capability}'. "
                f"Registered capabilities: {self.list_capabilities()}",
            )

        ranked = sorted(providers, key=lambda p: -p.priority)
        for provider in ranked:
            if provider.is_available():
                logger.debug("Resolved capability '%s' -> '%s'", capability, provider.provider_name)
                return provider.tool

        unavailable = [p.provider_name for p in ranked]
        raise CapabilityException(
            f"Capability '{capability}' has registered providers {unavailable}, "
            f"but none are currently available. Check required API keys, "
            f"network access, or installed plugins.",
        )

    def providers_for(self, capability: str) -> list[CapabilityProvider]:
        """Return all registered providers for ``capability``, highest priority first."""
        return sorted(self._providers.get(capability, []), key=lambda p: -p.priority)

    def list_capabilities(self) -> list[str]:
        """Return the sorted list of capability names with at least one provider."""
        return sorted(self._providers)

    def unregister(self, capability: str, provider_name: str) -> None:
        """Remove a specific provider from a capability. No-op if not found."""
        providers = self._providers.get(capability)
        if not providers:
            return
        self._providers[capability] = [p for p in providers if p.provider_name != provider_name]

    def __contains__(self, capability: str) -> bool:
        return capability in self._providers
