"""Locating the repository and its default files."""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROFILE = Path("profile") / "mgb-1973.yaml"


def find_root(start: Path) -> Path:
    """The repository root: the nearest folder holding pyproject.toml."""
    start = start.resolve()
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"no pyproject.toml in {start} or above it")
