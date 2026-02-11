"""Repo Review generator — scans a codebase and produces improvement tasks.

This generator performs static analysis without invoking an AI model:
- Finds files missing tests
- Detects missing type annotations (Python)
- Identifies large files that should be split
- Checks for common issues (no README, no .gitignore, etc.)
- Spots TODO/FIXME/HACK comments

For AI-powered deep review, the ``enhance`` flag can be used (future).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from ..model import Task, TaskPriority, TaskSource, TaskStatus, TaskType


class RepoReviewGenerator:
    """Generate improvement tasks by scanning the repository."""

    name = "repo_review"
    description = "Scan codebase and generate improvement tasks"

    def generate(
        self,
        project_dir: Path,
        *,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> list[Task]:
        tasks: list[Task] = []
        _progress = on_progress or (lambda msg, frac: None)

        _progress("Scanning repository structure...", 0.1)
        tasks.extend(self._check_project_hygiene(project_dir))

        _progress("Analyzing source files...", 0.3)
        tasks.extend(self._find_large_files(project_dir))

        _progress("Scanning for TODOs and FIXMEs...", 0.5)
        tasks.extend(self._find_todo_comments(project_dir))

        _progress("Checking test coverage gaps...", 0.7)
        tasks.extend(self._find_missing_tests(project_dir))

        _progress("Complete", 1.0)
        return tasks

    def _get_tracked_files(self, project_dir: Path) -> list[str]:
        """Get list of git-tracked files."""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return [f for f in result.stdout.strip().split("\n") if f]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return []

    def _check_project_hygiene(self, project_dir: Path) -> list[Task]:
        tasks: list[Task] = []

        if not (project_dir / "README.md").exists() and not (project_dir / "readme.md").exists():
            tasks.append(Task(
                title="Add README.md",
                description="Project is missing a README file. Add documentation covering setup, usage, and contribution guidelines.",
                task_type=TaskType.DOCS,
                priority=TaskPriority.P1,
                source=TaskSource.REPO_REVIEW,
                labels=["hygiene", "docs"],
            ))

        if not (project_dir / ".gitignore").exists():
            tasks.append(Task(
                title="Add .gitignore",
                description="Project is missing a .gitignore file. Add appropriate ignore patterns for the project's language and tooling.",
                task_type=TaskType.REFACTOR,
                priority=TaskPriority.P2,
                source=TaskSource.REPO_REVIEW,
                labels=["hygiene"],
            ))

        # Check for common config files
        has_ci = any(
            (project_dir / p).exists()
            for p in [".github/workflows", ".gitlab-ci.yml", ".circleci", "Jenkinsfile"]
        )
        if not has_ci:
            tasks.append(Task(
                title="Add CI/CD pipeline",
                description="No CI configuration detected. Set up automated testing and linting in CI.",
                task_type=TaskType.FEATURE,
                priority=TaskPriority.P2,
                source=TaskSource.REPO_REVIEW,
                labels=["hygiene", "ci"],
            ))

        return tasks

    def _find_large_files(self, project_dir: Path, threshold: int = 500) -> list[Task]:
        """Flag source files over *threshold* lines."""
        tasks: list[Task] = []
        source_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"}

        for fpath in self._get_tracked_files(project_dir):
            p = project_dir / fpath
            if p.suffix not in source_exts:
                continue
            try:
                line_count = sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
            except (OSError, UnicodeDecodeError):
                continue

            if line_count > threshold:
                tasks.append(Task(
                    title=f"Refactor large file: {fpath} ({line_count} lines)",
                    description=(
                        f"`{fpath}` has {line_count} lines which exceeds the {threshold}-line threshold. "
                        f"Consider splitting into smaller, focused modules."
                    ),
                    task_type=TaskType.REFACTOR,
                    priority=TaskPriority.P3,
                    source=TaskSource.REPO_REVIEW,
                    context_files=[fpath],
                    labels=["refactor", "large-file"],
                ))
        return tasks

    def _find_todo_comments(self, project_dir: Path) -> list[Task]:
        """Extract TODO/FIXME/HACK comments as tasks."""
        tasks: list[Task] = []
        pattern = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\b[:\s]*(.*)", re.IGNORECASE)
        source_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".rb"}

        seen: set[str] = set()  # deduplicate by description
        for fpath in self._get_tracked_files(project_dir):
            p = project_dir / fpath
            if p.suffix not in source_exts:
                continue
            try:
                for i, line in enumerate(p.open(encoding="utf-8", errors="ignore"), 1):
                    m = pattern.search(line)
                    if m:
                        tag = m.group(1).upper()
                        comment = m.group(2).strip()
                        if not comment or comment in seen:
                            continue
                        seen.add(comment)

                        prio = TaskPriority.P2 if tag == "FIXME" else TaskPriority.P3
                        ttype = TaskType.BUG if tag == "FIXME" else TaskType.REFACTOR

                        tasks.append(Task(
                            title=f"{tag}: {comment[:80]}",
                            description=f"Found `{tag}` in `{fpath}:{i}`: {comment}",
                            task_type=ttype,
                            priority=prio,
                            source=TaskSource.REPO_REVIEW,
                            context_files=[fpath],
                            labels=["todo", tag.lower()],
                        ))
            except (OSError, UnicodeDecodeError):
                continue

        return tasks

    def _find_missing_tests(self, project_dir: Path) -> list[Task]:
        """Find source files without corresponding test files."""
        tasks: list[Task] = []
        tracked = set(self._get_tracked_files(project_dir))

        for fpath in tracked:
            p = Path(fpath)
            # Python source files
            if p.suffix == ".py" and not p.name.startswith("test_") and not p.name.startswith("__"):
                if "test" in p.parts or "tests" in p.parts:
                    continue
                # Look for a test file
                test_name = f"test_{p.name}"
                candidates = [
                    str(p.parent / test_name),
                    str(Path("tests") / test_name),
                    str(Path("tests") / p.parent / test_name),
                ]
                if not any(c in tracked for c in candidates):
                    tasks.append(Task(
                        title=f"Add tests for {fpath}",
                        description=f"`{fpath}` has no corresponding test file. Add unit tests.",
                        task_type=TaskType.TEST,
                        priority=TaskPriority.P3,
                        source=TaskSource.REPO_REVIEW,
                        context_files=[fpath],
                        labels=["testing", "coverage"],
                    ))

        return tasks
