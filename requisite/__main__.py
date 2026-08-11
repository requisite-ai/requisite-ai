"""Enables ``python -m requisite`` as an alternative to the ``requisite`` console script."""

import sys

from requisite.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
