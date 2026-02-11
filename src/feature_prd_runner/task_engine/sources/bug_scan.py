"""Bug scan generator — runs tests/lint/typecheck and creates fix tasks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from ..model import Task, TaskPriority, TaskSource, TaskType


class BugScanGenerator:
    """Run test suite and linting tools, create tasks for each failure."""

    name = "bug_scan"
    description = "Run tests, linting, and type checking to find issues"

    def generate(
        self,
        project_dir: Path,
        *,
        on_progress: Optional[Callable[[str, float], None]] = None,
        test_command: Optional[str] = None,
        lint_command: Optional[str] = None,
        typecheck_command: Optional[str] = None,
    ) -> list[Task]:
        tasks: list[Task] = []
        _progress = on_progress or (lambda msg, frac: None)

        # Auto-detect commands from project files
        if test_command is None:
            test_command = self._detect_test_command(project_dir)
        if lint_command is None:
            lint_command = self._detect_lint_command(project_dir)
        if typecheck_command is None:
            typecheck_command = self._detect_typecheck_command(project_dir)

        if test_command:
            _progress(f"Running tests: {test_command}", 0.2)
            tasks.extend(self._run_and_parse(project_dir, test_command, "test"))

        if lint_command:
            _progress(f"Running linter: {lint_command}", 0.5)
            tasks.extend(self._run_and_parse(project_dir, lint_command, "lint"))

        if typecheck_command:
            _progress(f"Running type checker: {typecheck_command}", 0.8)
            tasks.extend(self._run_and_parse(project_dir, typecheck_command, "typecheck"))

        _progress("Complete", 1.0)
        return tasks

    def _run_and_parse(
        self, project_dir: Path, command: str, category: str
    ) -> list[Task]:
        """Run a command and create a task if it fails."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return [Task(
                title=f"Fix {category}: command timed out",
                description=f"Command `{command}` timed out after 120s.",
                task_type=TaskType.BUG,
                priority=TaskPriority.P1,
                source=TaskSource.BUG_SCAN,
                labels=[category, "timeout"],
            )]
        except FileNotFoundError:
            return []

        if result.returncode == 0:
            return []

        # Truncate output for task description
        output = (result.stdout + "\n" + result.stderr).strip()
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"

        prio = TaskPriority.P1 if category == "test" else TaskPriority.P2

        return [Task(
            title=f"Fix {category} failures: {command}",
            description=(
                f"Command `{command}` failed with exit code {result.returncode}.\n\n"
                f"```\n{output}\n```"
            ),
            task_type=TaskType.BUG,
            priority=prio,
            source=TaskSource.BUG_SCAN,
            labels=[category, "automated"],
        )]

    def _detect_test_command(self, project_dir: Path) -> Optional[str]:
        if (project_dir / "pyproject.toml").exists() or (project_dir / "pytest.ini").exists():
            return "python -m pytest --tb=short -q"
        if (project_dir / "package.json").exists():
            return "npm test"
        if (project_dir / "go.mod").exists():
            return "go test ./..."
        if (project_dir / "Cargo.toml").exists():
            return "cargo test"
        return None

    def _detect_lint_command(self, project_dir: Path) -> Optional[str]:
        if (project_dir / "pyproject.toml").exists():
            return "python -m ruff check ."
        return None

    def _detect_typecheck_command(self, project_dir: Path) -> Optional[str]:
        if (project_dir / "pyproject.toml").exists():
            return "python -m mypy . --ignore-missing-imports"
        if (project_dir / "tsconfig.json").exists():
            return "npx tsc --noEmit"
        return None
