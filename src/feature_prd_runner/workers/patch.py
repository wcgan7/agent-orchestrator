"""Apply unified diff patches safely."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from ..git_utils import _path_is_allowed


def patch_paths_allowed(project_dir: Path, paths: list[str], allowed_files: list[str]) -> tuple[bool, list[str]]:
    disallowed = [p for p in paths if not _path_is_allowed(project_dir, p, allowed_files)]
    return (not disallowed), disallowed


def apply_patch_with_git(
    *,
    project_dir: Path,
    patch_text: str,
    run_dir: Optional[Path] = None,
) -> tuple[bool, str]:
    """Apply a patch using `git apply`.

    Returns:
        (ok, error_message)
    """
    if not patch_text.strip():
        return True, ""

    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "generated.patch").write_text(patch_text)

    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "--recount", "-"],
        cwd=project_dir,
        input=patch_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return True, ""
    detail = (result.stderr or result.stdout or "").strip()
    return False, detail or f"git apply failed with code {result.returncode}"

