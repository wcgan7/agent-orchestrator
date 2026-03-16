"""Auto-detect Python virtual environments for a project directory.

Searches common locations for a usable venv and returns its path
information so that default project commands and worker env vars
can reference the correct Python environment without manual configuration.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Candidate directory names checked in project root and parent directory.
_VENV_CANDIDATE_NAMES: tuple[str, ...] = (".venv", "venv", "env")


@dataclass(frozen=True, slots=True)
class VenvInfo:
    """Detected virtual environment metadata."""

    path: Path
    """Absolute path to the venv root directory."""

    bin_dir: Path
    """Absolute path to the venv ``bin/`` directory."""

    source: str
    """Human-readable label describing how the venv was found."""


def _is_venv(path: Path) -> bool:
    """Return True if *path* looks like a valid Python venv (has ``bin/python``)."""
    try:
        return (path / "bin" / "python").exists()
    except OSError:
        return False


def detect_python_venv(project_dir: Path) -> VenvInfo | None:
    """Detect the best available Python virtual environment for *project_dir*.

    Searches in priority order:
    1. Project root candidates (``.venv``, ``venv``, ``env``).
    2. ``VIRTUAL_ENV`` environment variable.
    3. ``CONDA_PREFIX`` environment variable.
    4. Parent directory candidates.
    5. Poetry-managed venv (only if ``pyproject.toml`` contains ``[tool.poetry]``).

    Returns ``None`` if no usable venv is found.
    """
    # 1. Project root candidates
    for name in _VENV_CANDIDATE_NAMES:
        candidate = project_dir / name
        if _is_venv(candidate):
            resolved = candidate.resolve()
            return VenvInfo(
                path=resolved,
                bin_dir=resolved / "bin",
                source=f"project_dir/{name}",
            )

    # 2. VIRTUAL_ENV env var
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        venv_path = Path(virtual_env)
        if _is_venv(venv_path):
            return VenvInfo(
                path=venv_path.resolve(),
                bin_dir=venv_path.resolve() / "bin",
                source="env_var",
            )

    # 3. CONDA_PREFIX env var
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_path = Path(conda_prefix)
        if _is_venv(conda_path):
            return VenvInfo(
                path=conda_path.resolve(),
                bin_dir=conda_path.resolve() / "bin",
                source="conda",
            )

    # 4. Parent directory candidates
    parent = project_dir.parent
    if parent != project_dir:  # guard against filesystem root
        for name in _VENV_CANDIDATE_NAMES:
            candidate = parent / name
            if _is_venv(candidate):
                resolved = candidate.resolve()
                return VenvInfo(
                    path=resolved,
                    bin_dir=resolved / "bin",
                    source=f"parent_dir/{name}",
                )

    # 5. Poetry-managed venv (last resort — requires subprocess call)
    venv = _detect_poetry_venv(project_dir)
    if venv is not None:
        return venv

    return None


def _detect_poetry_venv(project_dir: Path) -> VenvInfo | None:
    """Attempt to detect a Poetry-managed venv via ``poetry env info -p``.

    Only called when ``pyproject.toml`` exists and contains ``[tool.poetry]``.
    """
    pyproject = project_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        text = pyproject.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "[tool.poetry]" not in text:
        return None

    try:
        result = subprocess.run(
            ["poetry", "env", "info", "-p"],
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    path_str = result.stdout.strip()
    if not path_str:
        return None

    venv_path = Path(path_str)
    if _is_venv(venv_path):
        return VenvInfo(
            path=venv_path.resolve(),
            bin_dir=venv_path.resolve() / "bin",
            source="poetry",
        )
    return None
