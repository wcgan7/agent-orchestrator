from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from typing import Optional

try:
    from .constants import IGNORED_REVIEW_PATH_PREFIXES, STATE_DIR_NAME
except ImportError:  # pragma: no cover
    from constants import IGNORED_REVIEW_PATH_PREFIXES, STATE_DIR_NAME


def _ignore_file_has_entry(path: Path, ignore_entry: str) -> bool:
    if not path.exists():
        return False
    try:
        contents = path.read_text()
    except OSError:
        return False
    lines = {
        line.strip().rstrip("/")
        for line in contents.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    normalized_entry = ignore_entry.strip().rstrip("/")
    return normalized_entry in lines or STATE_DIR_NAME in lines


def _append_ignore_entry(path: Path, ignore_entry: str) -> None:
    contents = ""
    if path.exists():
        contents = path.read_text()
    if contents and not contents.endswith("\n"):
        contents += "\n"
    contents += ignore_entry + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def _ensure_gitignore(project_dir: Path, only_if_clean: bool = False) -> None:
    ignore_entry = f"{STATE_DIR_NAME}/"
    gitignore_path = project_dir / ".gitignore"
    if _ignore_file_has_entry(gitignore_path, ignore_entry):
        return
    if only_if_clean and _git_has_changes(project_dir):
        return
    try:
        _append_ignore_entry(gitignore_path, ignore_entry)
    except OSError as exc:
        print(f"Warning: unable to update .gitignore: {exc}")


def _git_current_branch(project_dir: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_branch_exists(project_dir: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _ensure_branch(project_dir: Path, branch: str) -> None:
    current = _git_current_branch(project_dir)
    if current == branch:
        return
    if _git_branch_exists(project_dir, branch):
        subprocess.run(["git", "checkout", branch], cwd=project_dir, check=True)
    else:
        subprocess.run(["git", "checkout", "-b", branch], cwd=project_dir, check=True)


def _git_has_changes(project_dir: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _path_is_ignored(path: str, ignore_patterns: Optional[list[str]] = None) -> bool:
    if not ignore_patterns:
        return False
    for pattern in ignore_patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        if any(char in pattern for char in "*?["):
            if fnmatch.fnmatch(path, pattern):
                return True
            continue
        normalized = pattern.rstrip("/")
        if pattern.endswith("/"):
            prefix = f"{normalized}/"
            if path == normalized or path.startswith(prefix):
                return True
        else:
            if path == normalized:
                return True
    return False


def _path_is_allowed(project_dir: Path, path: str, allowed_patterns: list[str]) -> bool:
    for pattern in allowed_patterns or []:
        pattern = str(pattern).strip()
        if not pattern:
            continue

        # Glob patterns
        if any(ch in pattern for ch in "*?["):
            if fnmatch.fnmatch(path, pattern):
                return True
            continue

        normalized = pattern.rstrip("/")

        # Treat directories as prefixes (even if the pattern omitted the trailing slash)
        candidate = project_dir / normalized
        if pattern.endswith("/") or (candidate.exists() and candidate.is_dir()):
            prefix = f"{normalized}/"
            if path == normalized or path.startswith(prefix):
                return True

        # Exact file match
        if path == normalized:
            return True

    return False


def _git_changed_files(
    project_dir: Path,
    include_untracked: bool = True,
    ignore_prefixes: Optional[list[str]] = None,
) -> list[str]:
    changed: set[str] = set()
    commands = [
        ["git", "diff", "--name-only"],
        ["git", "diff", "--name-only", "--staged"],
    ]
    if include_untracked:
        commands.append(["git", "ls-files", "--others", "--exclude-standard"])
    ignore_prefixes = ignore_prefixes or []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                if _path_is_ignored(line, ignore_prefixes):
                    continue
                changed.add(line)
    return sorted(changed)


def _snapshot_repo_changes(project_dir: Path) -> list[str]:
    return _git_changed_files(
        project_dir,
        include_untracked=True,
        ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
    )


def _diff_file_sets(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    before_set = set(before)
    after_set = set(after)
    added = sorted(after_set - before_set)   # new changes introduced by this run
    removed = sorted(before_set - after_set) # changes that got reverted/cleaned up
    return added, removed


def _is_prd_runner_artifact(path: str) -> bool:
    return path == STATE_DIR_NAME or path.startswith(f"{STATE_DIR_NAME}/")


def _filter_non_prd_runner_changes(paths: list[str]) -> list[str]:
    return [p for p in paths if not _is_prd_runner_artifact(p)]


def _is_only_allowed_gitignore_addition(project_dir: Path) -> bool:
    # You already have this helper:
    # _gitignore_change_is_prd_runner_only(project_dir)
    return _gitignore_change_is_prd_runner_only(project_dir)


def _validate_changes_for_mode(
    *,
    project_dir: Path,
    mode: str,  # "plan" | "plan_impl" | "review" | "implement"
    introduced_changes: list[str],
    allowed_files: Optional[list[str]] = None,
) -> tuple[bool, str, list[str]]:
    """
    Returns (ok, error_message, disallowed_paths).
    """
    # PLAN / PLAN_IMPL / REVIEW: no repo changes outside .prd_runner (and optional .gitignore addition)
    if mode in {"plan", "plan_impl", "review"}:
        non_runner = _filter_non_prd_runner_changes(introduced_changes)

        # allow a very specific .gitignore tweak
        if ".gitignore" in non_runner and _is_only_allowed_gitignore_addition(project_dir):
            non_runner = [p for p in non_runner if p != ".gitignore"]

        if non_runner:
            return (
                False,
                f"Run mode '{mode}' must not modify repo files outside {STATE_DIR_NAME}/ "
                f"(disallowed: {', '.join(non_runner)[:400]})",
                non_runner,
            )
        return True, "", []

    # IMPLEMENT: enforce allowed_files patterns (you mostly do this already)
    if mode == "implement":
        allowed_files = allowed_files or []
        if not allowed_files:
            # If allowed list is missing, treat as violation to avoid silent repo edits.
            return False, "IMPLEMENT mode missing allowed_files policy; refusing to proceed.", introduced_changes

        disallowed = [p for p in introduced_changes if not _path_is_allowed(project_dir, p, allowed_files)]
        if disallowed:
            # allow that same gitignore tweak
            if ".gitignore" in disallowed and _is_only_allowed_gitignore_addition(project_dir):
                disallowed = [p for p in disallowed if p != ".gitignore"]

        if disallowed:
            return (
                False,
                "Changes outside allowed files: " + ", ".join(disallowed)[:400],
                disallowed,
            )
        return True, "", []

    return False, f"Unknown mode '{mode}'", introduced_changes


def _git_diff_text(project_dir: Path, max_chars: int = 20000) -> tuple[str, bool]:
    sections: list[str] = []
    commands = [
        ("UNSTAGED DIFF", ["git", "diff"]),
        ("STAGED DIFF", ["git", "diff", "--staged"]),
    ]
    for label, command in commands:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        content = result.stdout.strip()
        if content:
            sections.append(f"{label}:\n{content}")
    diff_text = "\n\n".join(sections).strip()
    if not diff_text:
        return "", False
    if len(diff_text) <= max_chars:
        return diff_text, False
    return diff_text[:max_chars], True


def _git_diff_stat(project_dir: Path, max_chars: int = 4000) -> tuple[str, bool]:
    sections: list[str] = []
    commands = [
        ("UNSTAGED DIFFSTAT", ["git", "diff", "--stat"]),
        ("STAGED DIFFSTAT", ["git", "diff", "--stat", "--staged"]),
    ]
    for label, command in commands:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        content = result.stdout.strip()
        if content:
            sections.append(f"{label}:\n{content}")
    diff_stat = "\n\n".join(sections).strip()
    if not diff_stat:
        return "", False
    if len(diff_stat) <= max_chars:
        return diff_stat, False
    return diff_stat[:max_chars], True


def _git_status_porcelain(project_dir: Path, max_chars: int = 2000) -> tuple[str, bool]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "", False
    content = result.stdout.strip()
    if not content:
        return "", False
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars], True


def _git_is_ignored(project_dir: Path, path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _gitignore_change_is_prd_runner_only(project_dir: Path) -> bool:
    commands = [
        ["git", "diff", "--", ".gitignore"],
        ["git", "diff", "--staged", "--", ".gitignore"],
    ]
    changes: list[str] = []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            if line.startswith(("+++ ", "--- ", "@@ ")):
                continue
            if line.startswith("+") or line.startswith("-"):
                changes.append(line)
    if not changes:
        return False
    if any(line.startswith("-") for line in changes):
        return False
    allowed = {".prd_runner", ".prd_runner/"}
    additions = [line[1:].strip() for line in changes if line.startswith("+")]
    if not additions:
        return False
    return all(entry in allowed for entry in additions)


def _git_tracked_paths(project_dir: Path, path: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", path],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_commit_and_push(project_dir: Path, branch: str, message: str) -> None:
    if _git_tracked_paths(project_dir, ".prd_runner"):
        raise RuntimeError(".prd_runner is tracked; remove it from git history before committing")
    if not _git_is_ignored(project_dir, ".prd_runner"):
        _ensure_gitignore(project_dir)
        if not _git_is_ignored(project_dir, ".prd_runner"):
            raise RuntimeError(".prd_runner is not ignored; add it to .gitignore before committing")
    subprocess.run(
        ["git", "add", "-A", "--", "."],
        cwd=project_dir,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", message], cwd=project_dir, check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=project_dir, check=True)
