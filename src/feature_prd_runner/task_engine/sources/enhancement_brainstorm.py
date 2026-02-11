"""Enhancement brainstorm generator — AI-powered feature idea generation.

This generator analyzes the codebase structure and generates feature ideas.
The initial version uses heuristic analysis. The ``ai_enhanced`` mode
(future) would use an AI model for deeper suggestions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from ..model import Task, TaskPriority, TaskSource, TaskType


class EnhancementBrainstormGenerator:
    """Generate enhancement ideas by analyzing the codebase."""

    name = "enhancement_brainstorm"
    description = "Analyze codebase and suggest feature enhancements"

    def generate(
        self,
        project_dir: Path,
        *,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> list[Task]:
        tasks: list[Task] = []
        _progress = on_progress or (lambda msg, frac: None)

        _progress("Analyzing project structure...", 0.2)
        tasks.extend(self._suggest_from_structure(project_dir))

        _progress("Checking for common enhancements...", 0.6)
        tasks.extend(self._suggest_common_improvements(project_dir))

        _progress("Complete", 1.0)
        return tasks

    def _get_file_stats(self, project_dir: Path) -> dict[str, int]:
        """Count files by extension."""
        stats: dict[str, int] = {}
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for f in result.stdout.strip().split("\n"):
                    if f:
                        ext = Path(f).suffix
                        stats[ext] = stats.get(ext, 0) + 1
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return stats

    def _suggest_from_structure(self, project_dir: Path) -> list[Task]:
        tasks: list[Task] = []
        stats = self._get_file_stats(project_dir)

        # Suggest adding type annotations if Python project without mypy config
        py_count = stats.get(".py", 0)
        if py_count > 5:
            mypy_configs = ["mypy.ini", ".mypy.ini", "setup.cfg"]
            has_mypy = any((project_dir / f).exists() for f in mypy_configs)
            pyproject = project_dir / "pyproject.toml"
            if pyproject.exists():
                has_mypy = has_mypy or "[tool.mypy]" in pyproject.read_text(errors="ignore")
            if not has_mypy:
                tasks.append(Task(
                    title="Add mypy configuration for type checking",
                    description="Project has Python files but no mypy configuration. Adding type checking improves code quality.",
                    task_type=TaskType.REFACTOR,
                    priority=TaskPriority.P3,
                    source=TaskSource.ENHANCEMENT_BRAINSTORM,
                    labels=["enhancement", "typing"],
                ))

        # Suggest Docker setup if not present
        if not (project_dir / "Dockerfile").exists() and not (project_dir / "docker-compose.yml").exists():
            tasks.append(Task(
                title="Add Docker configuration",
                description="No Docker setup found. Adding Dockerfile and docker-compose.yml enables consistent dev environments and easier deployment.",
                task_type=TaskType.FEATURE,
                priority=TaskPriority.P3,
                source=TaskSource.ENHANCEMENT_BRAINSTORM,
                labels=["enhancement", "devops"],
            ))

        return tasks

    def _suggest_common_improvements(self, project_dir: Path) -> list[Task]:
        tasks: list[Task] = []

        # Check for logging setup
        has_logging_config = any(
            (project_dir / f).exists()
            for f in ["logging.yaml", "logging.conf", "logging.json"]
        )
        py_files_exist = list(project_dir.glob("**/*.py"))[:1]
        if py_files_exist and not has_logging_config:
            tasks.append(Task(
                title="Add structured logging configuration",
                description="Consider adding a centralized logging configuration for better observability in production.",
                task_type=TaskType.FEATURE,
                priority=TaskPriority.P3,
                source=TaskSource.ENHANCEMENT_BRAINSTORM,
                labels=["enhancement", "observability"],
            ))

        # Suggest pre-commit hooks
        if not (project_dir / ".pre-commit-config.yaml").exists():
            tasks.append(Task(
                title="Add pre-commit hooks",
                description="No pre-commit configuration found. Pre-commit hooks can automatically run linting, formatting, and security checks before each commit.",
                task_type=TaskType.REFACTOR,
                priority=TaskPriority.P3,
                source=TaskSource.ENHANCEMENT_BRAINSTORM,
                labels=["enhancement", "developer-experience"],
            ))

        return tasks
