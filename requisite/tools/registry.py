"""
Tool registry.

Mirrors :class:`requisite.providers.factory.ProviderRegistry`'s shape:
a plain, instantiable class (no singleton) that plugins, applications,
and tests can each have their own instance of.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Union

from requisite.core.exceptions import ToolException
from requisite.tools.base import Tool

logger = logging.getLogger("requisite.tools.registry")

ToolLike = Union[Tool, Callable[..., Any]]


def resolve_tool_like(tool_or_func: ToolLike) -> Tool:
    """Coerce a :class:`Tool`, a plain function, or an ``@tool``-decorated
    function into a :class:`Tool` instance.

    Shared by :class:`ToolRegistry` and
    :class:`requisite.capabilities.registry.CapabilityRegistry` so both
    accept the same range of "tool-like" values.
    """
    if isinstance(tool_or_func, Tool):
        return tool_or_func
    attached = getattr(tool_or_func, "tool", None)
    if isinstance(attached, Tool):
        return attached
    if callable(tool_or_func):
        return Tool.from_function(tool_or_func)
    raise ToolException(f"Cannot register {tool_or_func!r} as a tool.")


class ToolRegistry:
    """A registry of tools, keyed by name.

    Examples
    --------
    >>> from requisite.tools import tool, ToolRegistry
    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>> registry = ToolRegistry()
    >>> registry.register(add)
    >>> registry.get("add").execute(a=1, b=2)
    3
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool_or_func: ToolLike) -> Tool:
        """Register a tool, accepting either a :class:`Tool`, a plain
        function, or a function already decorated with ``@tool``.

        Returns
        -------
        Tool
            The registered :class:`Tool` instance (useful when a plain
            function was passed in and you want the wrapped form back).
        """
        resolved = resolve_tool_like(tool_or_func)
        if resolved.name in self._tools:
            logger.debug(
                "Overwriting existing tool registration: '%s'",
                resolved.name,
                extra={"tool": resolved.name},
            )
        self._tools[resolved.name] = resolved
        return resolved

    @staticmethod
    def _resolve(tool_or_func: ToolLike) -> Tool:
        """Deprecated alias for the module-level :func:`resolve_tool_like`. Kept
        for backward compatibility; new code should use :func:`resolve_tool_like`.
        """
        return resolve_tool_like(tool_or_func)

    def unregister(self, name: str) -> None:
        """Remove a tool registration, if present. No-op if absent."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        """Return the registered tool named ``name``.

        Raises
        ------
        requisite.core.exceptions.ToolException
            If no tool is registered under that name.
        """
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolException(
                f"No tool registered under '{name}'. Available: {self.list_names()}",
            ) from exc

    def list_names(self) -> list[str]:
        """Return the sorted list of registered tool names."""
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
