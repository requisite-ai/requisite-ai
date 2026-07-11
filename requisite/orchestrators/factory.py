"""
Orchestrator registry.

Mirrors :class:`requisite.providers.factory.ProviderRegistry`: a
plain, instantiable class (not a singleton) mapping backend names to
constructors, pre-populated with the framework's built-in backends.
Third-party packages can register their own orchestrator (e.g. a real
CrewAI or AutoGen integration) the same way new providers are added.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from requisite.core.exceptions import ConfigurationException
from requisite.orchestrators.base import BaseOrchestrator

logger = logging.getLogger("requisite.orchestrators.factory")

OrchestratorBuilder = Callable[..., BaseOrchestrator]


class OrchestratorRegistry:
    """A registry mapping orchestrator backend names to constructors."""

    def __init__(self) -> None:
        self._builders: dict[str, OrchestratorBuilder] = {}

    def register(self, name: str, builder: OrchestratorBuilder) -> None:
        """Register an orchestrator constructor under ``name``."""
        key = name.lower().strip()
        if not key:
            raise ConfigurationException("Orchestrator name must be a non-empty string.")
        self._builders[key] = builder
        logger.debug("Registered orchestrator backend '%s'", key)

    def available(self) -> list[str]:
        """Return the sorted list of currently registered backend names."""
        return sorted(self._builders)

    def create(self, name: str, **kwargs: Any) -> BaseOrchestrator:
        """Instantiate the orchestrator registered under ``name``.

        Raises
        ------
        ConfigurationException
            If no orchestrator is registered under ``name``.
        """
        key = name.lower().strip()
        builder = self._builders.get(key)
        if builder is None:
            raise ConfigurationException(
                f"Unknown orchestrator backend '{name}'. Available: {self.available()}",
            )
        return builder(**kwargs)


def _not_yet_implemented(backend: str, install_hint: str) -> OrchestratorBuilder:
    def _builder(**kwargs: Any) -> BaseOrchestrator:
        raise ConfigurationException(
            f"The '{backend}' orchestrator backend is on the roadmap but not yet "
            f"implemented in this release. {install_hint}",
        )

    return _builder


def _register_builtin_orchestrators(registry: OrchestratorRegistry) -> None:
    def _build_native(**kwargs: Any) -> BaseOrchestrator:
        from requisite.orchestrators.native import NativeOrchestrator

        return NativeOrchestrator(**kwargs)

    def _build_langgraph(**kwargs: Any) -> BaseOrchestrator:
        from requisite.orchestrators.langgraph_orchestrator import LangGraphOrchestrator

        return LangGraphOrchestrator(**kwargs)

    registry.register("native", _build_native)
    registry.register("langgraph", _build_langgraph)
    # Roadmap backends: registered now with a clear, actionable error so
    # `workflow.use_crewai()` / `workflow.use_autogen()` fail helpfully
    # today and can become real integrations later with no API change.
    registry.register(
        "crewai",
        _not_yet_implemented(
            "crewai", "Track progress or contribute at the project's GitHub repo."
        ),
    )
    registry.register(
        "autogen",
        _not_yet_implemented(
            "autogen", "Track progress or contribute at the project's GitHub repo."
        ),
    )


default_registry = OrchestratorRegistry()
_register_builtin_orchestrators(default_registry)
