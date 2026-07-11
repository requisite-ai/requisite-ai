"""Skill registry -- mirrors :class:`requisite.tools.registry.ToolRegistry`'s shape."""

from __future__ import annotations

import logging

from requisite.core.exceptions import SkillException
from requisite.skills.base import BaseSkill

logger = logging.getLogger("requisite.skills.registry")


class SkillRegistry:
    """A registry of skills, keyed by name.

    Examples
    --------
    >>> registry = SkillRegistry()
    >>> registry.register(some_skill)  # doctest: +SKIP
    >>> registry.get("some_skill").run(...)  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> BaseSkill:
        """Register a skill instance. Returns the skill for convenient chaining."""
        if skill.name in self._skills:
            logger.debug("Overwriting existing skill registration: '%s'", skill.name)
        self._skills[skill.name] = skill
        return skill

    def unregister(self, name: str) -> None:
        """Remove a skill registration, if present. No-op if absent."""
        self._skills.pop(name, None)

    def get(self, name: str) -> BaseSkill:
        """Return the registered skill named ``name``.

        Raises
        ------
        requisite.core.exceptions.SkillException
            If no skill is registered under that name.
        """
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillException(
                f"No skill registered under '{name}'. Available: {self.list_names()}",
            ) from exc

    def list_names(self) -> list[str]:
        """Return the sorted list of registered skill names."""
        return sorted(self._skills)

    def all(self) -> list[BaseSkill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)
