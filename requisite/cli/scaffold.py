"""
Project scaffolding for ``requisite init``.

Generates a minimal, runnable project: a trimmed ``.env.example``, a
``requirements.txt`` pinned to the chosen provider's extra, an
``agents.py`` establishing the ``agent_registry`` convention that
``requisite agents``/``requisite chat --agent`` rely on (see
``requisite.cli.project``), a ``main.py`` demonstrating the full
Settings -> AI -> Agent -> run flow in one file, and a quickstart
``README.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from requisite.core.exceptions import ConfigurationException

logger = logging.getLogger("requisite.cli.scaffold")

# Env var name(s) each provider needs, beyond DEFAULT_PROVIDER/MODEL --
# mirrors requisite/config/settings.py's field set.
PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "azure_openai": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "ollama": ["OLLAMA_API_KEY", "OLLAMA_HOST"],
}

# A reasonable default model per provider, matching README's table.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-6",
    "groq": "llama-3.3-70b-versatile",
    "azure_openai": "your-deployment-name",
    "openrouter": "openai/gpt-4o-mini",
    "together": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "ollama": "llama3.2",
}


def _env_example(provider: str) -> str:
    lines = [
        "# Copy this file to .env and fill in the key(s) below.",
        "# Settings (requisite.config.settings.Settings) reads this automatically.",
        "",
    ]
    for var in PROVIDER_ENV_VARS[provider]:
        lines.append(f"{var}=")
    lines += [
        "",
        f"DEFAULT_PROVIDER={provider}",
        f"MODEL={DEFAULT_MODELS[provider]}",
        "TEMPERATURE=0.2",
        "",
    ]
    return "\n".join(lines)


def _gitignore() -> str:
    return "__pycache__/\n*.pyc\n.env\n.venv/\n"


def _requirements(provider: str) -> str:
    return f"requisite-ai[{provider}]\n"


def _agents_py(provider: str) -> str:
    return f'''"""
Agents for this project.

`agent_registry` is the convention `requisite agents` and
`requisite chat --agent <name>` look for: a module-level
`AgentRegistry` populated with named `Agent` instances. Add more
agents by constructing them and calling `agent_registry.register(...)`.
"""

from requisite import Agent, AgentRegistry

agent_registry = AgentRegistry()

assistant = Agent(
    name="assistant",
    provider="{provider}",
    system_prompt="You are a helpful assistant.",
)
agent_registry.register(assistant)
'''


def _main_py() -> str:
    return '''"""
Quickstart: runs the "assistant" agent defined in agents.py.

    python main.py
"""

from agents import agent_registry


def main() -> None:
    assistant = agent_registry.get("assistant")
    result = assistant.run("Introduce yourself in one sentence.")
    print(result.content)


if __name__ == "__main__":
    main()
'''


def _readme(name: str, provider: str) -> str:
    return f"""# {name}

Scaffolded by `requisite init`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your {provider} API key
```

## Run

```bash
python main.py                 # runs the example agent once
requisite chat                 # interactive chat REPL
requisite chat --agent assistant  # chat through the example agent (tool calls work)
requisite agents                # list agents registered in agents.py
requisite providers             # list providers and whether each is configured
requisite capabilities          # list registered capabilities
```
"""


def write_project(dest: Path, provider: str, *, force: bool = False) -> list[Path]:
    """Scaffold a new project at ``dest``.

    Parameters
    ----------
    dest:
        Target directory. Created if missing. May already exist and be
        empty (or ``.``, the current directory) without ``force``.
    provider:
        Provider name to preconfigure (``.env.example``, ``agents.py``,
        ``requirements.txt``). One of ``PROVIDER_ENV_VARS``'s keys.
    force:
        If ``True``, write into ``dest`` even if it already contains files,
        overwriting any that collide by name.

    Returns
    -------
    list[Path]
        Paths of every file written, in creation order.

    Raises
    ------
    requisite.core.exceptions.ConfigurationException
        If ``provider`` is unknown, or ``dest`` exists, is non-empty, and
        ``force`` was not passed.
    """
    if provider not in PROVIDER_ENV_VARS:
        raise ConfigurationException(
            f"Unknown provider '{provider}'. Available: {sorted(PROVIDER_ENV_VARS)}",
        )

    dest.mkdir(parents=True, exist_ok=True)
    if not force and any(dest.iterdir()):
        raise ConfigurationException(
            f"'{dest}' already exists and is not empty. Pass --force to scaffold into it anyway.",
        )

    name = dest.resolve().name
    files = {
        ".env.example": _env_example(provider),
        ".gitignore": _gitignore(),
        "requirements.txt": _requirements(provider),
        "agents.py": _agents_py(provider),
        "main.py": _main_py(),
        "README.md": _readme(name, provider),
    }

    written: list[Path] = []
    for filename, content in files.items():
        path = dest / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
        logger.debug("Wrote %s", path, extra={"path": str(path)})
    return written
