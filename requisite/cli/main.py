"""
``requisite`` CLI entry point: argument parsing, subcommand dispatch, and
the top-level error boundary.

Installed as the ``requisite`` console script and runnable via
``python -m requisite``. See ``requisite.cli.commands`` for what each
subcommand actually does.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Optional

from requisite import __version__
from requisite.capabilities import default_registry as default_capability_registry
from requisite.cli.commands import (
    cmd_agents,
    cmd_capabilities,
    cmd_chat,
    cmd_init,
    cmd_plugins,
    cmd_providers,
)
from requisite.cli.scaffold import PROVIDER_ENV_VARS
from requisite.config.settings import Settings
from requisite.core.exceptions import AIException
from requisite.plugins import DEFAULT_GROUP
from requisite.providers.factory import ProviderRegistry, default_registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="requisite",
        description="Scaffold a project, inspect registered providers/capabilities/agents/plugins, or run a quick chat.",
    )
    parser.add_argument("--version", action="version", version=f"requisite-ai {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Scaffold a new project")
    init_parser.add_argument("name", help="Directory to create ('.' for the current directory)")
    init_parser.add_argument(
        "--provider",
        default="openai",
        choices=sorted(PROVIDER_ENV_VARS),
        help="Provider to preconfigure (default: openai)",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Scaffold into a non-empty directory anyway"
    )

    subparsers.add_parser("providers", help="List registered providers")
    subparsers.add_parser("capabilities", help="List registered capabilities")

    plugins_parser = subparsers.add_parser(
        "plugins", help="Discover and load installed plugins (entry-point based)"
    )
    plugins_parser.add_argument(
        "--group",
        default=None,
        help=f"Entry-point group to scan (default: {DEFAULT_GROUP!r})",
    )

    agents_parser = subparsers.add_parser(
        "agents", help="List agents registered in the current project's agents.py"
    )
    agents_parser.add_argument(
        "--module", default=None, help="Path to the agents module (default: ./agents.py)"
    )

    chat_parser = subparsers.add_parser("chat", help="Run a quick chat, one-shot or interactive")
    chat_parser.add_argument(
        "prompt", nargs="?", default=None, help="One-shot prompt; omit for a REPL"
    )
    chat_parser.add_argument(
        "--provider", default=None, help="Provider name (default: Settings.default_provider)"
    )
    chat_parser.add_argument("--model", default=None, help="Model name (default: Settings.model)")
    chat_parser.add_argument(
        "--agent", default=None, help="Chat through a named agent from the project's agents.py"
    )
    chat_parser.add_argument(
        "--module", default=None, help="Path to the agents module (default: ./agents.py)"
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    registry: Optional[ProviderRegistry] = None,
    settings: Optional[Settings] = None,
) -> int:
    """Parse arguments, dispatch to a subcommand, and return an exit code.

    Parameters
    ----------
    argv:
        Argument list, excluding the program name. Defaults to
        ``sys.argv[1:]``.
    registry:
        :class:`~requisite.providers.factory.ProviderRegistry` used by the
        ``providers``/``chat`` subcommands. Defaults to the framework's
        built-in registry. Injectable so tests can register a fake
        provider without touching the shared ``default_registry``.
    settings:
        :class:`~requisite.config.settings.Settings` used by the
        ``providers``/``chat`` subcommands. Defaults to a freshly loaded
        instance (reads the environment / ``.env``). Injectable for the
        same reason as ``registry``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    provider_registry = registry or default_registry
    resolved_settings = settings if settings is not None else Settings()

    try:
        if args.command == "init":
            return cmd_init(args)
        if args.command == "providers":
            return cmd_providers(args, registry=provider_registry, settings=resolved_settings)
        if args.command == "capabilities":
            return cmd_capabilities(args, registry=default_capability_registry)
        if args.command == "plugins":
            return cmd_plugins(args)
        if args.command == "agents":
            return cmd_agents(args)
        if args.command == "chat":
            return cmd_chat(args, provider_registry=provider_registry, settings=resolved_settings)
    except AIException as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130

    raise AssertionError(
        f"Unhandled command: {args.command!r}"
    )  # pragma: no cover - argparse enforces choices


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
