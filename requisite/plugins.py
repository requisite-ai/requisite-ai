"""
Entry-point plugin discovery.

Third-party packages register providers/tools/capabilities/etc. with
Requisite's registries the same way first-party code does --
``default_registry.register(...)`` -- typically from their own package's
``__init__.py`` (see ``CONTRIBUTING.md``'s "Adding a new capability
resolver" walkthrough). That's always worked; what's new here is
automating the *import* step: :func:`discover` finds every package that
declares itself under the ``"requisite.plugins"`` entry-point group and
imports it, so an application doesn't need to already know and
explicitly import every plugin it has installed.

This does **not** introduce a new registration mechanism -- there is
deliberately no ``Plugin`` base class (see
``docs/adr/0001-core-architecture-and-interfaces.md``'s Plugin discovery
section: "every real plugin still just calls ``registry.register(...)``
inside it"). A plugin's entry point points at either a plain module
(its top-level code registers on import, exactly as manual usage does
today) or a zero-argument callable (called once, after import).

:func:`discover` is never called automatically anywhere in this
package's own import chain -- nothing runs code from a package you
didn't explicitly ask to discover. Call it once, explicitly, typically
at application startup (or via ``requisite plugins`` on the CLI).
"""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points

from pydantic import BaseModel, Field

logger = logging.getLogger("requisite.plugins")

DEFAULT_GROUP = "requisite.plugins"


class PluginDiscoveryResult(BaseModel):
    """The outcome of a :func:`discover` call.

    Parameters
    ----------
    loaded:
        Names of entry points that imported (and, if callable, ran)
        successfully, in discovery order.
    failed:
        Maps the name of each entry point that raised to a string
        description of the error. A failure here never stops discovery
        of the remaining entry points -- see :func:`discover`.
    """

    loaded: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)


def discover(*, group: str = DEFAULT_GROUP) -> PluginDiscoveryResult:
    """Import every plugin registered under ``group``, letting each
    self-register with whichever framework registry it targets.

    Parameters
    ----------
    group:
        The entry-point group to scan. Defaults to ``"requisite.plugins"``.
        A plugin package declares itself under this group in its own
        ``pyproject.toml``::

            [project.entry-points."requisite.plugins"]
            my_plugin = "my_requisite_plugin"

    Returns
    -------
    PluginDiscoveryResult
        Every entry point's outcome -- both successes and failures.
        One plugin failing to import never prevents the others from
        being discovered; each failure is logged (at ``ERROR``) and
        recorded in the result rather than raised, since this is a
        batch of independent third-party packages, not a single call
        whose failure should abort the whole operation.

    Examples
    --------
    >>> from requisite.plugins import discover
    >>> result = discover()  # doctest: +SKIP
    >>> result.loaded  # doctest: +SKIP
    ['my_requisite_plugin']
    """
    result = PluginDiscoveryResult()
    for entry_point in entry_points(group=group):
        try:
            _load_and_run(entry_point)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Plugin '%s' failed to load: %s",
                entry_point.name,
                exc,
                extra={"plugin": entry_point.name},
            )
            result.failed[entry_point.name] = str(exc)
        else:
            logger.debug("Loaded plugin '%s'", entry_point.name, extra={"plugin": entry_point.name})
            result.loaded.append(entry_point.name)
    return result


def _load_and_run(entry_point: EntryPoint) -> None:
    loaded = entry_point.load()
    if callable(loaded):
        loaded()
