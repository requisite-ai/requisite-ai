"""
Tool: a Python callable exposed to an LLM for function/tool calling.

A :class:`Tool` bundles a callable with the metadata (name, description,
JSON Schema) a provider needs to offer it to the model, plus the
execution logic to actually run it once the model requests it.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from requisite.core.exceptions import ToolException
from requisite.tools.schema import function_to_parameters_schema

logger = logging.getLogger("requisite.tools")


class Tool(BaseModel):
    """A callable capability that can be offered to an LLM for tool calling.

    Parameters
    ----------
    name:
        Unique identifier the model uses to request this tool. Defaults
        to the wrapped function's ``__name__``.
    description:
        Explains to the model what the tool does and when to use it.
        Defaults to the function's docstring.
    parameters_schema:
        JSON Schema describing the tool's arguments, auto-derived from
        the function's type hints via
        :func:`requisite.tools.schema.function_to_parameters_schema`.
    func:
        The underlying Python callable. May be sync or async.

    Examples
    --------
    >>> def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>> tool = Tool.from_function(add)
    >>> tool.execute(a=2, b=3)
    5
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    func: Callable[..., Any]

    @classmethod
    def from_function(
        cls, func: Callable[..., Any], *, name: str | None = None, description: str | None = None
    ) -> Tool:
        """Build a :class:`Tool` from a plain Python function.

        Parameters
        ----------
        func:
            The function to wrap. Its signature and docstring drive the
            schema and description shown to the model.
        name:
            Overrides the tool name (defaults to ``func.__name__``).
        description:
            Overrides the tool description (defaults to ``func.__doc__``).
        """
        return cls(
            name=name or func.__name__,
            description=(description or inspect.getdoc(func) or "").strip(),
            parameters_schema=function_to_parameters_schema(func),
            func=func,
        )

    def execute(self, **kwargs: Any) -> Any:
        """Synchronously execute the tool with the given arguments.

        If the wrapped function is a coroutine function, it is run to
        completion via ``asyncio.run`` -- prefer :meth:`aexecute` from
        async contexts to avoid nested event loop issues.

        Raises
        ------
        requisite.core.exceptions.ToolException
            If the underlying function raises.
        """
        try:
            if inspect.iscoroutinefunction(self.func):
                return asyncio.run(self.func(**kwargs))
            return self.func(**kwargs)
        except Exception as exc:
            raise ToolException(
                f"Tool '{self.name}' raised an error: {exc}",
                details={"tool": self.name, "arguments": kwargs},
            ) from exc

    async def aexecute(self, **kwargs: Any) -> Any:
        """Asynchronously execute the tool with the given arguments."""
        try:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**kwargs)
            return await asyncio.to_thread(self.func, **kwargs)
        except Exception as exc:
            raise ToolException(
                f"Tool '{self.name}' raised an error: {exc}",
                details={"tool": self.name, "arguments": kwargs},
            ) from exc

    def to_openai_schema(self) -> dict[str, Any]:
        """Return this tool's definition in OpenAI's ``tools`` wire format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema or {"type": "object", "properties": {}},
            },
        }

    def to_gemini_schema(self) -> dict[str, Any]:
        """Return this tool's definition as a Gemini function declaration dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema or {"type": "object", "properties": {}},
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Return this tool's definition in Anthropic's tool-use wire format.

        Anthropic names the schema field ``input_schema`` rather than
        OpenAI/Gemini's ``parameters`` -- otherwise the same JSON Schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema or {"type": "object", "properties": {}},
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Tool(name={self.name!r})"
