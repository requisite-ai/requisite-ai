"""Prompt template registry -- mirrors the shape of every other registry in the framework."""

from __future__ import annotations

import logging
from typing import Union

from requisite.core.exceptions import PromptException
from requisite.prompts.template import ChatPromptTemplate, PromptTemplate

logger = logging.getLogger("requisite.prompts.registry")

AnyPromptTemplate = Union[PromptTemplate, ChatPromptTemplate]


class PromptTemplateRegistry:
    """A registry of named, reusable prompt templates.

    Unlike :class:`~requisite.providers.factory.ProviderRegistry` and
    similar, there's no framework-shipped default template -- there's no
    such thing as a universal default prompt. ``default_registry`` exists
    purely as a shared, opt-in place for an application (or a plugin) to
    register its own named templates once and reuse them everywhere.

    Examples
    --------
    >>> from requisite.prompts import PromptTemplate, PromptTemplateRegistry
    >>> registry = PromptTemplateRegistry()
    >>> registry.register("translate", PromptTemplate.from_template("Translate to {language}: {text}"))
    >>> registry.get("translate").format(language="French", text="Hello")
    'Translate to French: Hello'
    """

    def __init__(self) -> None:
        self._templates: dict[str, AnyPromptTemplate] = {}

    def register(self, name: str, template: AnyPromptTemplate) -> None:
        """Register a template under ``name``. Overwrites any existing registration."""
        if not name.strip():
            raise PromptException("Prompt template name must be a non-empty string.")
        if name in self._templates:
            logger.debug(
                "Overwriting existing prompt template registration: '%s'",
                name,
                extra={"prompt_template": name},
            )
        self._templates[name] = template

    def unregister(self, name: str) -> None:
        """Remove a template registration, if present. No-op if absent."""
        self._templates.pop(name, None)

    def get(self, name: str) -> AnyPromptTemplate:
        """Return the registered template named ``name``.

        Raises
        ------
        requisite.core.exceptions.PromptException
            If no template is registered under that name.
        """
        try:
            return self._templates[name]
        except KeyError as exc:
            raise PromptException(
                f"No prompt template registered under '{name}'. Available: {self.list_names()}",
            ) from exc

    def list_names(self) -> list[str]:
        """Return the sorted list of registered template names."""
        return sorted(self._templates)

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __len__(self) -> int:
        return len(self._templates)


default_registry = PromptTemplateRegistry()
