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
from collections.abc import Callable
from typing import Any, ParamSpec, Protocol, TypeVar, overload

from requisite.tools.base import Tool

P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)


class ToolFunction(Protocol[P, R_co]):
    """The type of a function decorated with ``@tool``.

    Statically describes both halves of what ``@tool`` produces: the
    function remains directly callable with its original signature, and
    also carries a ``.tool`` attribute holding the generated
    :class:`~requisite.tools.base.Tool`. Declaring this as a real
    ``Protocol`` (rather than returning the original function type
    as-is) is what lets type checkers verify both
    ``search_weather("Paris")`` and ``search_weather.tool.execute(...)``
    / passing ``search_weather`` directly to ``tools=[...]`` -- accessing
    ``.tool`` on a plain, undecorated function is a real type error,
    exactly as it would be a real ``AttributeError`` at runtime.
    """

    tool: Tool

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


def _wrap(func: Callable[P, R], *, name: str | None, description: str | None) -> ToolFunction[P, R]:
    built_tool = Tool.from_function(func, name=name, description=description)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    wrapper.tool = built_tool  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


@overload
def tool(func: Callable[P, R]) -> ToolFunction[P, R]: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[P, R]], ToolFunction[P, R]]: ...


def tool(
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Mark a function as an LLM-callable tool.

    Can be used bare (``@tool``) or with keyword arguments to override
    the name/description (``@tool(name=..., description=...)``).

    The decorated function remains directly callable as normal Python;
    the generated :class:`~requisite.tools.base.Tool` is attached as
    the ``.tool`` attribute for registration with a
    :class:`~requisite.tools.registry.ToolRegistry` or an
    :class:`~requisite.agents.agent.Agent`. It can also be passed
    directly wherever a tool is accepted (``ai.chat(tools=[my_tool])``,
    ``Agent(tools=[my_tool])``) -- both resolve a decorated function's
    ``.tool`` automatically, the same as passing the ``Tool`` directly.
    """
    if func is not None:
        return _wrap(func, name=name, description=description)

    def decorator(inner: Callable[P, R]) -> ToolFunction[P, R]:
        return _wrap(inner, name=name, description=description)

    return decorator
