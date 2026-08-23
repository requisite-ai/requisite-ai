"""Unit tests for tool calling: schema derivation, @tool decorator, ToolRegistry."""

from __future__ import annotations

from typing import Optional

import pytest

from requisite.core.exceptions import ToolException
from requisite.tools.base import Tool
from requisite.tools.decorator import tool
from requisite.tools.registry import ToolRegistry
from requisite.tools.schema import function_to_parameters_schema


def sample_search(city: str, days: int = 1, verbose: Optional[bool] = None) -> str:
    """Search the weather for a city."""
    return f"{city}:{days}:{verbose}"


def test_function_to_parameters_schema_marks_required_and_optional() -> None:
    schema = function_to_parameters_schema(sample_search)
    assert schema["type"] == "object"
    assert schema["required"] == ["city"]
    assert schema["properties"]["city"] == {"type": "string"}
    assert schema["properties"]["days"] == {"type": "integer"}
    assert schema["properties"]["verbose"] == {"type": "boolean"}


def sample_unresolvable_annotation(x: "NotARealTypeAnywhereInScope") -> str:  # noqa: F821
    """A function with a forward-reference annotation that can never
    resolve -- e.g. a TYPE_CHECKING-only import, or a typo."""
    return str(x)


def test_function_to_parameters_schema_degrades_gracefully_on_unresolvable_annotation() -> None:
    """Regression test: get_type_hints() resolves every annotation on a
    function at once and raises NameError if even one fails -- this must
    not crash tool registration, matching the module's own documented
    'never raises on an exotic type' guarantee. See
    docs/adr/0031-code-review-fixes.md."""
    schema = function_to_parameters_schema(sample_unresolvable_annotation)
    assert schema["properties"]["x"] == {"type": "string"}
    assert schema["required"] == ["x"]


def test_tool_decorator_on_function_with_unresolvable_annotation() -> None:
    @tool
    def bad_tool(x: "TotallyMissingType") -> str:  # noqa: F821
        return str(x)

    assert bad_tool.tool.parameters_schema["properties"]["x"] == {"type": "string"}


def test_tool_from_function_derives_name_and_description() -> None:
    built = Tool.from_function(sample_search)
    assert built.name == "sample_search"
    assert built.description == "Search the weather for a city."
    assert built.parameters_schema["required"] == ["city"]


def test_tool_execute_runs_the_function() -> None:
    built = Tool.from_function(sample_search)
    assert built.execute(city="Paris") == "Paris:1:None"


def test_tool_execute_wraps_errors() -> None:
    def boom() -> None:
        raise ValueError("nope")

    built = Tool.from_function(boom)
    with pytest.raises(ToolException):
        built.execute()


def test_tool_openai_schema_shape() -> None:
    built = Tool.from_function(sample_search)
    schema = built.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "sample_search"
    assert "parameters" in schema["function"]


def test_tool_decorator_bare() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add.tool.name == "add"
    assert add(2, 3) == 5
    assert add.tool.execute(a=2, b=3) == 5


def test_tool_decorator_with_overrides() -> None:
    @tool(name="custom_name", description="Custom description.")
    def whatever(x: int) -> int:
        return x

    assert whatever.tool.name == "custom_name"
    assert whatever.tool.description == "Custom description."


def test_registry_register_plain_function() -> None:
    registry = ToolRegistry()
    registry.register(sample_search)
    assert "sample_search" in registry
    assert registry.get("sample_search").execute(city="Rome") == "Rome:1:None"


def test_registry_register_decorated_function() -> None:
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    registry = ToolRegistry()
    registry.register(add)
    assert registry.get("add").execute(a=1, b=1) == 2


def test_registry_get_missing_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolException):
        registry.get("nope")


def test_registry_len_and_all() -> None:
    registry = ToolRegistry()
    registry.register(sample_search)
    assert len(registry) == 1
    assert registry.all()[0].name == "sample_search"


@pytest.mark.asyncio
async def test_tool_aexecute() -> None:
    built = Tool.from_function(sample_search)
    result = await built.aexecute(city="Tokyo")
    assert result == "Tokyo:1:None"
