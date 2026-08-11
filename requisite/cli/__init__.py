"""
The ``requisite`` command-line interface.

Installed as the ``requisite`` console script
(``[project.scripts]`` in ``pyproject.toml``) and runnable via
``python -m requisite``. See ``requisite.cli.main`` for the entry point
and ``requisite.cli.commands`` for each subcommand's implementation.
"""

from requisite.cli.main import main

__all__ = ["main"]
