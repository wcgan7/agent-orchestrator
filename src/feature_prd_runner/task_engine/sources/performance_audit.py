"""Performance audit generator — identifies slow paths, large bundles, and inefficiencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from ..model import Task, TaskPriority, TaskSource, TaskType


class PerformanceAuditGenerator:
    """Analyze codebase for performance issues and create improvement tasks."""

    name = "performance_audit"
    description = "Identify large files, N+1 patterns, large bundles, and performance anti-patterns"

    def generate(
        self,
        project_dir: Path,
        *,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> list[Task]:
        tasks: list[Task] = []
        _progress = on_progress or (lambda msg, frac: None)

        # 1. Large source files
        _progress("Checking for large source files...", 0.1)
        tasks.extend(self._check_large_files(project_dir))

        # 2. Bundle size
        _progress("Checking bundle size...", 0.3)
        tasks.extend(self._check_bundle_size(project_dir))

        # 3. N+1 query patterns
        _progress("Scanning for N+1 query patterns...", 0.5)
        tasks.extend(self._check_n_plus_one(project_dir))

        # 4. Missing indexes / slow patterns
        _progress("Checking for performance anti-patterns...", 0.7)
        tasks.extend(self._check_antipatterns(project_dir))

        _progress("Performance audit complete", 1.0)
        return tasks

    def _check_large_files(self, project_dir: Path) -> list[Task]:
        """Identify source files over 500 lines that may need splitting."""
        large_files: list[tuple[str, int]] = []
        source_exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}

        for path in project_dir.rglob("*"):
            if not path.is_file() or path.suffix not in source_exts:
                continue
            if any(part.startswith(".") or part == "node_modules" for part in path.parts):
                continue
            try:
                line_count = sum(1 for _ in open(path))
                if line_count > 500:
                    rel = str(path.relative_to(project_dir))
                    large_files.append((rel, line_count))
            except OSError:
                continue

        if large_files:
            large_files.sort(key=lambda x: x[1], reverse=True)
            file_list = "\n".join(f"- `{f}`: {n} lines" for f, n in large_files[:20])
            return [Task(
                title=f"Refactor {len(large_files)} large files (>500 lines)",
                description=(
                    f"Found {len(large_files)} source files exceeding 500 lines. "
                    "Consider splitting into smaller, focused modules.\n\n"
                    f"{file_list}"
                ),
                task_type=TaskType.PERFORMANCE,
                priority=TaskPriority.P2,
                source=TaskSource.PERFORMANCE_AUDIT,
                labels=["performance", "refactor", "automated"],
                context_files=[f for f, _ in large_files[:10]],
            )]
        return []

    def _check_bundle_size(self, project_dir: Path) -> list[Task]:
        """Check if frontend build output is unusually large."""
        dist_dirs = [
            project_dir / "dist",
            project_dir / "build",
            project_dir / "web" / "dist",
            project_dir / "frontend" / "dist",
        ]

        for dist_dir in dist_dirs:
            if not dist_dir.is_dir():
                continue
            total_size = 0
            large_assets: list[tuple[str, int]] = []
            for path in dist_dir.rglob("*"):
                if path.is_file():
                    size = path.stat().st_size
                    total_size += size
                    if size > 500_000 and path.suffix in (".js", ".css"):
                        rel = str(path.relative_to(project_dir))
                        large_assets.append((rel, size))

            if total_size > 2_000_000:
                large_assets.sort(key=lambda x: x[1], reverse=True)
                asset_list = "\n".join(
                    f"- `{f}`: {s // 1024}KB" for f, s in large_assets[:10]
                )
                return [Task(
                    title=f"Reduce bundle size ({total_size // 1024}KB total)",
                    description=(
                        f"Build output in `{dist_dir.relative_to(project_dir)}` is "
                        f"{total_size // 1024}KB. Consider code splitting and tree shaking.\n\n"
                        f"Largest assets:\n{asset_list}"
                    ),
                    task_type=TaskType.PERFORMANCE,
                    priority=TaskPriority.P2,
                    source=TaskSource.PERFORMANCE_AUDIT,
                    labels=["performance", "bundle-size", "automated"],
                )]
        return []

    def _check_n_plus_one(self, project_dir: Path) -> list[Task]:
        """Scan for potential N+1 query patterns in Python/JS code."""
        import re

        patterns = [
            (r'for\s+\w+\s+in\s+\w+.*:\s*\n\s+.*\.(query|execute|find|get)\s*\(', ".py",
             "Loop with per-iteration DB query"),
            (r'\.forEach\s*\(\s*(?:async\s*)?\(?.*\)?\s*=>\s*\{[^}]*(?:fetch|axios|query)\s*\(',
             (".js", ".ts", ".tsx"), "Loop with per-iteration API/DB call"),
        ]

        findings: list[str] = []
        for path in project_dir.rglob("*"):
            if not path.is_file():
                continue
            if any(part.startswith(".") or part == "node_modules" for part in path.parts):
                continue
            try:
                content = path.read_text(errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            for pattern, exts, desc in patterns:
                ext_tuple = exts if isinstance(exts, tuple) else (exts,)
                if path.suffix in ext_tuple:
                    if re.search(pattern, content, re.MULTILINE):
                        rel = str(path.relative_to(project_dir))
                        findings.append(f"- `{rel}`: {desc}")

        if findings:
            return [Task(
                title=f"Fix {len(findings)} potential N+1 query patterns",
                description=(
                    "Found code patterns that may cause N+1 query problems:\n\n"
                    + "\n".join(findings[:20])
                    + "\n\nBatch queries or use eager loading to fix."
                ),
                task_type=TaskType.PERFORMANCE,
                priority=TaskPriority.P1,
                source=TaskSource.PERFORMANCE_AUDIT,
                labels=["performance", "n-plus-one", "automated"],
            )]
        return []

    def _check_antipatterns(self, project_dir: Path) -> list[Task]:
        """Check for general performance anti-patterns."""
        import re

        checks = [
            (r'import\s+\*\s+from', (".py",), "Wildcard import (loads unused modules)"),
            (r'time\.sleep\s*\(\s*\d+\s*\)', (".py",), "Blocking sleep in code"),
            (r'JSON\.parse\s*\(\s*JSON\.stringify', (".js", ".ts", ".tsx"), "Deep clone via JSON (slow for large objects)"),
        ]

        findings: list[str] = []
        for path in project_dir.rglob("*"):
            if not path.is_file():
                continue
            if any(part.startswith(".") or part == "node_modules" for part in path.parts):
                continue
            try:
                content = path.read_text(errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            for pattern, exts, desc in checks:
                if path.suffix in exts:
                    matches = re.findall(pattern, content)
                    if matches:
                        rel = str(path.relative_to(project_dir))
                        findings.append(f"- `{rel}`: {desc} ({len(matches)}x)")

        if findings:
            return [Task(
                title="Fix performance anti-patterns",
                description=(
                    "Found performance anti-patterns:\n\n"
                    + "\n".join(findings[:20])
                ),
                task_type=TaskType.PERFORMANCE,
                priority=TaskPriority.P3,
                source=TaskSource.PERFORMANCE_AUDIT,
                labels=["performance", "anti-pattern", "automated"],
            )]
        return []
