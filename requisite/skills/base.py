"""
Skills: reusable, higher-level capabilities.

A tool is typically a single function; a skill is a slightly heavier
capability that may wrap multiple operations, hold its own
configuration, and is meant to be composed into agents and reused
across an application (e.g. "SQL Query", "Search Internet", "Read File").

Every skill can also be exposed to an LLM as a tool via :meth:`BaseSkill.as_tool`,
so agents can offer skills for function calling without any extra glue.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from requisite.core.exceptions import SkillException
from requisite.tools.base import Tool

logger = logging.getLogger("requisite.skills")


class BaseSkill(ABC):
    """Abstract base class for all skills.

    Parameters
    ----------
    name:
        Unique, human-readable identifier for the skill.
    description:
        What the skill does and when to use it -- shown to the LLM when
        the skill is exposed as a tool.

    Examples
    --------
    >>> class UppercaseSkill(BaseSkill):
    ...     def __init__(self):
    ...         super().__init__(name="uppercase", description="Uppercase text.")
    ...     def run(self, text: str) -> str:
    ...         return text.upper()
    >>> skill = UppercaseSkill()
    >>> skill.run(text="hello")
    'HELLO'
    """

    def __init__(self, *, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the skill synchronously. Implemented by subclasses."""

    async def arun(self, **kwargs: Any) -> Any:
        """Execute the skill asynchronously.

        The default implementation runs :meth:`run` in a worker thread so
        that synchronous skills remain usable from async code without
        blocking the event loop. Override for a true async implementation.
        """
        try:
            return await asyncio.to_thread(self.run, **kwargs)
        except Exception as exc:
            raise SkillException(
                f"Skill '{self.name}' raised an error: {exc}",
                details={"skill": self.name, "arguments": kwargs},
            ) from exc

    def as_tool(self) -> Tool:
        """Expose this skill to an LLM as a :class:`~requisite.tools.base.Tool`.

        The tool's JSON Schema is derived from :meth:`run`'s signature,
        just like a plain function decorated with ``@tool``.
        """
        return Tool.from_function(self.run, name=self.name, description=self.description)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(name={self.name!r})"
