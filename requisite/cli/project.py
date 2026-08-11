"""
Discovery of a scaffolded project's ``agent_registry``.

``AgentRegistry`` (unlike ``ProviderRegistry``/``CapabilityRegistry``) ships
with no ``default_registry`` instance -- agents are always user-constructed,
never built into the framework. ``requisite init`` resolves this for the
CLI by establishing a convention instead: a scaffolded project's
``agents.py`` defines a module-level ``agent_registry: AgentRegistry``
variable. ``requisite agents`` and ``requisite chat --agent`` both import
that file from the current directory via this module.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from requisite.agents.registry import AgentRegistry
from requisite.core.exceptions import ConfigurationException

logger = logging.getLogger("requisite.cli.project")

DEFAULT_MODULE_NAME = "agents.py"
REGISTRY_ATTR = "agent_registry"


def load_agent_registry(module_path: Path) -> AgentRegistry:
    """Import ``module_path`` and return its ``agent_registry`` variable.

    Parameters
    ----------
    module_path:
        Path to a Python file defining a module-level
        ``agent_registry: AgentRegistry`` variable -- the convention
        ``requisite init`` scaffolds into every new project's ``agents.py``.

    Returns
    -------
    AgentRegistry

    Raises
    ------
    requisite.core.exceptions.ConfigurationException
        If ``module_path`` doesn't exist, fails to import, or doesn't
        define an ``agent_registry`` of the right type.
    """
    if not module_path.is_file():
        raise ConfigurationException(
            f"No '{module_path.name}' found in {module_path.parent}. "
            f"Run 'requisite init <name>' to scaffold a project with an "
            f"example agent, or define an `agent_registry = AgentRegistry()` "
            f"in {module_path.name} yourself.",
        )

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ConfigurationException(f"Could not load '{module_path}' as a Python module.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigurationException(
            f"Error importing '{module_path}': {exc}",
        ) from exc

    registry = getattr(module, REGISTRY_ATTR, None)
    if not isinstance(registry, AgentRegistry):
        raise ConfigurationException(
            f"'{module_path}' has no `{REGISTRY_ATTR} = AgentRegistry()` "
            f"variable. Run 'requisite init <name>' to see the expected shape.",
        )
    return registry
