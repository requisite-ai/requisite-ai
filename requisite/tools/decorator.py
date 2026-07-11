"""
The ``@tool`` decorator.

Turns a plain Python function into something an agent or provider can
offer to an LLM for tool calling, with zero provider-specific code.

Examples
--------
>>> from requisite.tools import tool
>>>
>>> @tool
... def search_weather(city: str) -> str:
...     '''Look up the current weather for a city.'''
...     return f"Sunny in {city}"
>>>
>>> search_weather.tool.name
'search_weather'
>>> search_weather("Paris")
'Sunny in Paris'

With an explicit name/description:

>>> @tool(name="weather_lookup", description="Fetch current weather by city name.")
... def get_weather(city: str) -> str:
...     return f"Sunny in {city}"
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar, overload

from requisite.tools.base import Tool

F = TypeVar("F", bound=Callable[..., Any])


def _wrap(func: F, *, name: Optional[str], description: Optional[str]) -> F:
    built_tool = Tool.from_function(func, name=name, description=description)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper.tool = built_tool  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


@overload
def tool(func: F) -> F: ...


@overload
def tool(*, name: Optional[str] = None, description: Optional[str] = None) -> Callable[[F], F]: ...


def tool(
    func: Optional[F] = None, *, name: Optional[str] = None, description: Optional[str] = None
) -> Any:
    """Mark a function as an LLM-callable tool.

    Can be used bare (``@tool``) or with keyword arguments to override
    the name/description (``@tool(name=..., description=...)``).

    The decorated function remains directly callable as normal Python;
    the generated :class:`~requisite.tools.base.Tool` is attached as
    the ``.tool`` attribute for registration with a
    :class:`~requisite.tools.registry.ToolRegistry` or an
    :class:`~requisite.agents.agent.Agent`.
    """
    if func is not None:
        return _wrap(func, name=name, description=description)

    def decorator(inner: F) -> F:
        return _wrap(inner, name=name, description=description)

    return decorator
