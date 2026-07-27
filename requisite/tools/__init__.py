"""Tool calling: wrap Python functions so LLMs can invoke them."""

from requisite.tools.base import Tool
from requisite.tools.decorator import tool
from requisite.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolRegistry", "tool"]
