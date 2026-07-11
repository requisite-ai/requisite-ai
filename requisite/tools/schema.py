"""
Function -> JSON Schema conversion.

Providers that support tool calling (OpenAI, Gemini, ...) require each
tool to be described as a JSON Schema. This module inspects a plain
Python function's signature and type hints and derives that schema
automatically, so developers never write JSON Schema by hand.

Supported type hints: ``str``, ``int``, ``float``, ``bool``, ``list``,
``dict``, ``Optional[X]`` / ``X | None``, and ``list[X]`` for the above.
Unrecognized annotations fall back to a permissive ``"string"`` schema
rather than raising, so tool registration never breaks on an exotic type.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, get_args, get_origin

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type_for(annotation: Any) -> dict[str, Any]:
    """Best-effort mapping from a Python type annotation to a JSON Schema fragment."""
    if annotation is inspect.Signature.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)

    # Optional[X] / X | None -> schema for X (nullability is handled via `required`)
    if origin is typing.Union:
        non_none_args = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none_args) == 1:
            return _json_type_for(non_none_args[0])
        return {"type": "string"}

    if origin in (list, typing.List):  # noqa: UP006
        args = get_args(annotation)
        item_schema = _json_type_for(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin in (dict, typing.Dict):  # noqa: UP006
        return {"type": "object"}

    json_type = _TYPE_MAP.get(annotation)
    if json_type:
        return {"type": json_type}

    return {"type": "string"}


def function_to_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema object describing ``func``'s parameters.

    Parameters
    ----------
    func:
        The function to inspect. ``self``/``cls`` are skipped automatically
        if present (e.g. for bound methods used as tools).

    Returns
    -------
    dict
        A JSON Schema object of type ``"object"`` with a ``properties``
        map and a ``required`` list, suitable for use as an OpenAI or
        Gemini function/tool parameter schema.

    Examples
    --------
    >>> def search_weather(city: str, days: int = 1) -> str:
    ...     '''Look up weather.'''
    ...     ...
    >>> schema = function_to_parameters_schema(search_weather)
    >>> schema["required"]
    ['city']
    """
    signature = inspect.signature(func)
    type_hints = typing.get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = type_hints.get(param_name, param.annotation)
        properties[param_name] = _json_type_for(annotation)

        if param.default is inspect.Signature.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
