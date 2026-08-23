"""
Implementations of each ``requisite`` subcommand.

Each ``cmd_*`` function takes the parsed :class:`argparse.Namespace`
(plus, where useful for testing, an injectable registry/settings) and
returns a process exit code. Output goes to ``stdout`` via ``print()`` --
a deliberate, scoped exception to the framework's "never ``print()``,
always ``logging``" convention: this output *is* the CLI's product, not a
diagnostic log. See ADR-0014.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Optional

from requisite.ai import AI
from requisite.capabilities.registry import CapabilityRegistry
from requisite.cli.project import DEFAULT_MODULE_NAME, load_agent_registry
from requisite.cli.scaffold import write_project
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.plugins import DEFAULT_GROUP, discover
from requisite.providers.factory import ProviderRegistry

logger = logging.getLogger("requisite.cli.commands")

# Module each provider's optional SDK is imported from -- used only to
# report "installed?" in `requisite providers`; the provider modules
# themselves already import lazily and independently of this mapping.
_PROVIDER_SDK_MODULES: dict[str, str] = {
    "openai": "openai",
    "gemini": "google.genai",
    "anthropic": "anthropic",
    "groq": "openai",
    "azure_openai": "openai",
    "openrouter": "openai",
    "together": "openai",
    "ollama": "ollama",
}


def _resolve_module_path(module_arg: Optional[str]) -> Path:
    if module_arg:
        return Path(module_arg)
    return Path.cwd() / DEFAULT_MODULE_NAME


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold a new project (``requisite init NAME``)."""
    dest = Path.cwd() if args.name == "." else Path(args.name)
    written = write_project(dest, args.provider, force=args.force)

    print(f"Created {len(written)} files in {dest}/:")
    for path in written:
        print(f"  {path.name}")

    print("\nNext steps:")
    if args.name != ".":
        print(f"  cd {args.name}")
    print("  pip install -r requirements.txt")
    print(f"  cp .env.example .env   # then fill in your {args.provider} API key")
    print("  python main.py          # or: requisite chat --agent assistant")
    return 0


def cmd_providers(
    args: argparse.Namespace, *, registry: ProviderRegistry, settings: Settings
) -> int:
    """List registered providers (``requisite providers``)."""
    names = registry.available()
    header = f"{'PROVIDER':<15}{'SDK INSTALLED':<16}{'API KEY SET':<12}"
    print(header)
    print("-" * len(header))
    for name in names:
        sdk_module = _PROVIDER_SDK_MODULES.get(name)
        sdk_installed = bool(sdk_module and importlib.util.find_spec(sdk_module) is not None)
        key_set = bool(settings.api_key_for(name))
        print(f"{name:<15}{'yes' if sdk_installed else 'no':<16}{'yes' if key_set else 'no':<12}")
    return 0


def cmd_capabilities(args: argparse.Namespace, *, registry: CapabilityRegistry) -> int:
    """List registered capabilities (``requisite capabilities``)."""
    names = registry.list_capabilities()
    if not names:
        print("No capabilities registered.")
        return 0
    for name in names:
        print(f"{name}:")
        for provider in registry.providers_for(name):
            status = "available" if provider.is_available() else "unavailable"
            print(f"  {provider.provider_name}  (priority={provider.priority}, {status})")
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    """List agents registered in the current project's ``agents.py`` (``requisite agents``)."""
    module_path = _resolve_module_path(args.module)
    registry = load_agent_registry(module_path)
    names = registry.list_names()
    if not names:
        print("No agents registered.")
        return 0
    print("Registered agents:")
    for name in names:
        print(f"  {name}")
    return 0


def cmd_plugins(args: argparse.Namespace) -> int:
    """Discover and load installed plugins (``requisite plugins``)."""
    group = args.group or DEFAULT_GROUP
    result = discover(group=group)

    if not result.loaded and not result.failed:
        print(f"No plugins found under entry-point group '{group}'.")
        return 0

    if result.loaded:
        print("Loaded:")
        for name in result.loaded:
            print(f"  {name}")

    if result.failed:
        print("Failed:")
        for name, error in result.failed.items():
            print(f"  {name}: {error}")
        return 1

    return 0


def cmd_chat(
    args: argparse.Namespace,
    *,
    provider_registry: ProviderRegistry,
    settings: Optional[Settings] = None,
) -> int:
    """Run a quick chat, one-shot or interactive (``requisite chat [PROMPT]``)."""
    if args.agent and (args.provider or args.model):
        raise ConfigurationException(
            "--provider/--model have no effect together with --agent -- a named agent "
            "already has its own configured provider/model. Drop --agent to use "
            "--provider/--model directly, or drop --provider/--model to use the "
            "agent's own configuration.",
        )
    if args.agent:
        module_path = _resolve_module_path(args.module)
        agent_registry = load_agent_registry(module_path)
        agent = agent_registry.get(args.agent)

        def ask(text: str) -> str:
            return agent.run(text).content
    else:
        ai = AI(
            provider=args.provider,
            model=args.model,
            settings=settings,
            registry=provider_registry,
        )

        def ask(text: str) -> str:
            reply: str = ai.chat(text)
            return reply

    if args.prompt:
        print(ask(args.prompt))
        return 0

    print("requisite chat -- type 'exit' or 'quit' to leave, Ctrl-D to end.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        stripped = line.strip()
        if stripped.lower() in {"exit", "quit"}:
            return 0
        if not stripped:
            continue
        print(ask(stripped))
