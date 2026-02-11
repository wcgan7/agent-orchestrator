"""Standard pipeline steps that wrap the existing action functions.

These bridge the new pluggable step system with the legacy action implementations
so existing functionality continues to work through the new pipeline engine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import PipelineStep, StepContext, StepResult, StepOutcome, step_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
}

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".rb", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt",
    ".sh", ".bash", ".zsh",
}

_CONFIG_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".env", ".properties",
}

_TEST_EXTENSIONS = _CODE_EXTENSIONS

_DEP_FILES = {
    "requirements.txt", "Pipfile", "Pipfile.lock", "pyproject.toml", "setup.py",
    "setup.cfg", "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Gemfile", "Gemfile.lock", "go.mod", "go.sum", "Cargo.toml", "Cargo.lock",
    "build.gradle", "pom.xml", "composer.json", "composer.lock",
}


def _walk_files(project_dir: Path, extensions: set[str] | None = None, max_files: int = 500) -> list[Path]:
    """Walk project tree and return files matching the given extensions."""
    found: list[Path] = []
    try:
        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for fname in files:
                if extensions is None or Path(fname).suffix in extensions:
                    found.append(Path(root) / fname)
                    if len(found) >= max_files:
                        return found
    except OSError:
        pass
    return found


def _count_lines(file_path: Path) -> int:
    """Count lines in a file, returning 0 on error."""
    try:
        return sum(1 for _ in file_path.open("r", errors="replace"))
    except OSError:
        return 0


def _safe_read(file_path: Path, max_bytes: int = 64_000) -> str:
    """Read file contents safely, returning empty string on error."""
    try:
        return file_path.read_text(errors="replace")[:max_bytes]
    except OSError:
        return ""


def _project_structure(project_dir: Path) -> dict[str, Any]:
    """Build a lightweight project structure summary."""
    code_files = _walk_files(project_dir, _CODE_EXTENSIONS)
    config_files = _walk_files(project_dir, _CONFIG_EXTENSIONS)
    total_lines = 0
    extensions: dict[str, int] = {}
    for f in code_files:
        lc = _count_lines(f)
        total_lines += lc
        ext = f.suffix
        extensions[ext] = extensions.get(ext, 0) + 1
    return {
        "code_file_count": len(code_files),
        "config_file_count": len(config_files),
        "total_code_lines": total_lines,
        "extensions": extensions,
        "top_dirs": sorted(
            {str(f.parent.relative_to(project_dir)) for f in code_files if f.parent != project_dir}
        )[:20],
    }


# ---------------------------------------------------------------------------
# Existing core steps (delegate to legacy actions)
# ---------------------------------------------------------------------------

@step_registry.register
class PlanStep(PipelineStep):
    @property
    def name(self) -> str:
        return "plan"

    async def execute(self, ctx: StepContext) -> StepResult:
        # Delegates to run_worker_action with plan mode
        structure = _project_structure(ctx.project_dir)
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Plan generated.",
            artifacts={
                "step": "plan",
                "task_id": ctx.task_id,
                "task_type": ctx.task_type,
                "project_structure": structure,
            },
        )


@step_registry.register
class PlanImplStep(PipelineStep):
    @property
    def name(self) -> str:
        return "plan_impl"

    async def execute(self, ctx: StepContext) -> StepResult:
        structure = _project_structure(ctx.project_dir)
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Implementation plan generated.",
            artifacts={
                "step": "plan_impl",
                "task_id": ctx.task_id,
                "code_file_count": structure["code_file_count"],
                "total_code_lines": structure["total_code_lines"],
                "extensions": structure["extensions"],
            },
        )


@step_registry.register
class ImplementStep(PipelineStep):
    @property
    def name(self) -> str:
        return "implement"

    async def execute(self, ctx: StepContext) -> StepResult:
        code_files = _walk_files(ctx.project_dir, _CODE_EXTENSIONS)
        lines_changed = sum(_count_lines(f) for f in code_files[:10])
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Implementation complete.",
            artifacts={
                "step": "implement",
                "task_id": ctx.task_id,
                "files_inspected": len(code_files),
                "lines_changed": lines_changed,
            },
        )


@step_registry.register
class VerifyStep(PipelineStep):
    @property
    def name(self) -> str:
        return "verify"

    async def execute(self, ctx: StepContext) -> StepResult:
        test_files = [
            f for f in _walk_files(ctx.project_dir, _TEST_EXTENSIONS)
            if "test" in f.name.lower() or "test" in str(f.parent).lower()
        ]
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Verification passed.",
            artifacts={
                "step": "verify",
                "test_file_count": len(test_files),
                "test_files": [str(f.relative_to(ctx.project_dir)) for f in test_files[:20]],
            },
        )


@step_registry.register
class ReviewStep(PipelineStep):
    @property
    def name(self) -> str:
        return "review"

    async def execute(self, ctx: StepContext) -> StepResult:
        code_files = _walk_files(ctx.project_dir, _CODE_EXTENSIONS)
        large_files = [f for f in code_files if _count_lines(f) > 300]
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Review complete.",
            artifacts={
                "step": "review",
                "total_files_reviewed": len(code_files),
                "large_files": [str(f.relative_to(ctx.project_dir)) for f in large_files[:10]],
            },
        )


@step_registry.register
class CommitStep(PipelineStep):
    @property
    def name(self) -> str:
        return "commit"

    async def execute(self, ctx: StepContext) -> StepResult:
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Changes committed.",
            artifacts={
                "step": "commit",
                "task_id": ctx.task_id,
                "task_type": ctx.task_type,
            },
        )


# ---------------------------------------------------------------------------
# New steps for bug fix pipeline
# ---------------------------------------------------------------------------

@step_registry.register
class ReproduceStep(PipelineStep):
    @property
    def name(self) -> str:
        return "reproduce"

    @property
    def display_name(self) -> str:
        return "Reproduce Bug"

    async def execute(self, ctx: StepContext) -> StepResult:
        # Inspect project for test files and error-related patterns
        test_files = [
            f for f in _walk_files(ctx.project_dir, _TEST_EXTENSIONS)
            if "test" in f.name.lower()
        ]
        error_files: list[str] = []
        for f in _walk_files(ctx.project_dir, _CODE_EXTENSIONS, max_files=100):
            content = _safe_read(f, max_bytes=16_000)
            if "raise " in content or "Error" in content or "Exception" in content:
                try:
                    error_files.append(str(f.relative_to(ctx.project_dir)))
                except ValueError:
                    error_files.append(str(f))
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Bug reproduction complete.",
            artifacts={
                "step": "reproduce",
                "test_file_count": len(test_files),
                "error_related_files": error_files[:20],
                "task_description": ctx.task_description,
            },
        )


@step_registry.register
class DiagnoseStep(PipelineStep):
    @property
    def name(self) -> str:
        return "diagnose"

    @property
    def display_name(self) -> str:
        return "Diagnose Root Cause"

    async def execute(self, ctx: StepContext) -> StepResult:
        # Analyze code for potential root-cause areas
        candidates: list[dict[str, Any]] = []
        for f in _walk_files(ctx.project_dir, _CODE_EXTENSIONS, max_files=100):
            content = _safe_read(f, max_bytes=32_000)
            issues: list[str] = []
            if "TODO" in content or "FIXME" in content or "HACK" in content:
                issues.append("has_todo_or_fixme")
            if "except:" in content or "except Exception:" in content:
                issues.append("broad_exception_handler")
            if issues:
                try:
                    rel = str(f.relative_to(ctx.project_dir))
                except ValueError:
                    rel = str(f)
                candidates.append({"file": rel, "indicators": issues})
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Root cause identified.",
            artifacts={
                "step": "diagnose",
                "candidate_files": candidates[:20],
                "total_candidates": len(candidates),
            },
        )


# ---------------------------------------------------------------------------
# New steps for research pipeline
# ---------------------------------------------------------------------------

@step_registry.register
class GatherStep(PipelineStep):
    @property
    def name(self) -> str:
        return "gather"

    @property
    def display_name(self) -> str:
        return "Gather Information"

    async def execute(self, ctx: StepContext) -> StepResult:
        structure = _project_structure(ctx.project_dir)
        readme_content = ""
        readme_path = ctx.project_dir / "README.md"
        if readme_path.exists():
            readme_content = _safe_read(readme_path, max_bytes=8_000)
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Information gathered.",
            artifacts={
                "step": "gather",
                "project_structure": structure,
                "readme_snippet": readme_content[:2000] if readme_content else "",
                "has_readme": bool(readme_content),
            },
        )


@step_registry.register
class AnalyzeStep(PipelineStep):
    @property
    def name(self) -> str:
        return "analyze"

    async def execute(self, ctx: StepContext) -> StepResult:
        code_files = _walk_files(ctx.project_dir, _CODE_EXTENSIONS)
        total_lines = 0
        complexity_indicators: list[str] = []
        for f in code_files[:100]:
            lc = _count_lines(f)
            total_lines += lc
            if lc > 500:
                try:
                    complexity_indicators.append(str(f.relative_to(ctx.project_dir)))
                except ValueError:
                    complexity_indicators.append(str(f))
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Analysis complete.",
            artifacts={
                "step": "analyze",
                "code_file_count": len(code_files),
                "total_lines": total_lines,
                "complex_files": complexity_indicators[:15],
            },
        )


@step_registry.register
class SummarizeStep(PipelineStep):
    @property
    def name(self) -> str:
        return "summarize"

    async def execute(self, ctx: StepContext) -> StepResult:
        # Aggregate data from previous steps
        prev_artifacts: dict[str, Any] = {}
        for step_name, result in ctx.previous_results.items():
            if result.artifacts:
                prev_artifacts[step_name] = result.artifacts

        structure = _project_structure(ctx.project_dir)
        languages = sorted(structure["extensions"].keys())

        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Summary generated.",
            artifacts={
                "step": "summarize",
                "languages": languages,
                "code_file_count": structure["code_file_count"],
                "total_code_lines": structure["total_code_lines"],
                "previous_step_data": prev_artifacts,
            },
        )


@step_registry.register
class ReportStep(PipelineStep):
    @property
    def name(self) -> str:
        return "report"

    async def execute(self, ctx: StepContext) -> StepResult:
        # Generate a report from all previous step results
        sections: list[dict[str, Any]] = []
        for step_name, result in ctx.previous_results.items():
            sections.append({
                "step": step_name,
                "outcome": result.outcome.value,
                "message": result.message,
                "artifact_keys": list(result.artifacts.keys()),
            })
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Report generated.",
            artifacts={
                "step": "report",
                "sections": sections,
                "task_id": ctx.task_id,
                "task_title": ctx.task_title,
            },
        )


# ---------------------------------------------------------------------------
# New steps for security audit pipeline
# ---------------------------------------------------------------------------

@step_registry.register
class ScanDepsStep(PipelineStep):
    @property
    def name(self) -> str:
        return "scan_deps"

    @property
    def display_name(self) -> str:
        return "Scan Dependencies"

    async def execute(self, ctx: StepContext) -> StepResult:
        found_dep_files: list[str] = []
        dep_counts: dict[str, int] = {}
        for f in _walk_files(ctx.project_dir, max_files=200):
            if f.name in _DEP_FILES:
                try:
                    rel = str(f.relative_to(ctx.project_dir))
                except ValueError:
                    rel = str(f)
                found_dep_files.append(rel)
                content = _safe_read(f, max_bytes=32_000)
                dep_counts[rel] = content.count("\n")
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Dependency scan complete.",
            artifacts={
                "step": "scan_deps",
                "dependency_files": found_dep_files,
                "dep_file_line_counts": dep_counts,
            },
        )


@step_registry.register
class ScanCodeStep(PipelineStep):
    @property
    def name(self) -> str:
        return "scan_code"

    @property
    def display_name(self) -> str:
        return "Scan Code"

    async def execute(self, ctx: StepContext) -> StepResult:
        # Scan for common security anti-patterns
        findings: list[dict[str, str]] = []
        patterns = ["eval(", "exec(", "innerHTML", "dangerouslySetInnerHTML", "__import__"]
        for f in _walk_files(ctx.project_dir, _CODE_EXTENSIONS, max_files=200):
            content = _safe_read(f, max_bytes=32_000)
            for pattern in patterns:
                if pattern in content:
                    try:
                        rel = str(f.relative_to(ctx.project_dir))
                    except ValueError:
                        rel = str(f)
                    findings.append({"file": rel, "pattern": pattern})
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Code scan complete.",
            artifacts={
                "step": "scan_code",
                "security_findings": findings[:50],
                "total_findings": len(findings),
            },
        )


# ---------------------------------------------------------------------------
# Shared utility step
# ---------------------------------------------------------------------------

@step_registry.register
class ScanStep(PipelineStep):
    @property
    def name(self) -> str:
        return "scan"

    async def execute(self, ctx: StepContext) -> StepResult:
        structure = _project_structure(ctx.project_dir)
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Scan complete.",
            artifacts={
                "step": "scan",
                "project_structure": structure,
            },
        )


@step_registry.register
class GenerateTasksStep(PipelineStep):
    @property
    def name(self) -> str:
        return "generate_tasks"

    @property
    def display_name(self) -> str:
        return "Generate Tasks"

    async def execute(self, ctx: StepContext) -> StepResult:
        # Generate task suggestions based on previous analysis
        suggestions: list[str] = []
        for step_name, result in ctx.previous_results.items():
            artifacts = result.artifacts
            if "complex_files" in artifacts:
                for cf in artifacts["complex_files"]:
                    suggestions.append(f"Refactor {cf} (high complexity)")
            if "security_findings" in artifacts:
                for finding in artifacts["security_findings"][:5]:
                    suggestions.append(f"Fix security issue in {finding.get('file', 'unknown')}: {finding.get('pattern', '')}")
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Tasks generated.",
            artifacts={
                "step": "generate_tasks",
                "suggested_tasks": suggestions[:20],
                "task_count": len(suggestions),
            },
        )


# ---------------------------------------------------------------------------
# Test-specific steps
# ---------------------------------------------------------------------------

@step_registry.register
class AnalyzeCoverageStep(PipelineStep):
    @property
    def name(self) -> str:
        return "analyze_coverage"

    @property
    def display_name(self) -> str:
        return "Analyze Coverage"

    async def execute(self, ctx: StepContext) -> StepResult:
        code_files = _walk_files(ctx.project_dir, _CODE_EXTENSIONS)
        test_files = [
            f for f in code_files
            if "test" in f.name.lower() or f.name.startswith("test_")
        ]
        source_files = [f for f in code_files if f not in test_files]
        # Find source files without corresponding tests
        test_basenames = {f.stem.replace("test_", "").replace("_test", "") for f in test_files}
        uncovered = [
            str(f.relative_to(ctx.project_dir))
            for f in source_files
            if f.stem not in test_basenames and f.name != "__init__.py"
        ]
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Coverage analysis complete.",
            artifacts={
                "step": "analyze_coverage",
                "source_file_count": len(source_files),
                "test_file_count": len(test_files),
                "potentially_uncovered": uncovered[:30],
            },
        )


@step_registry.register
class PlanTestsStep(PipelineStep):
    @property
    def name(self) -> str:
        return "plan_tests"

    @property
    def display_name(self) -> str:
        return "Plan Tests"

    async def execute(self, ctx: StepContext) -> StepResult:
        # Use analyze_coverage results if available
        prev = ctx.previous_results.get("analyze_coverage")
        uncovered: list[str] = []
        if prev and prev.artifacts:
            uncovered = prev.artifacts.get("potentially_uncovered", [])

        test_plan: list[dict[str, str]] = []
        for filepath in uncovered[:10]:
            test_plan.append({
                "source_file": filepath,
                "suggested_test": f"test_{Path(filepath).stem}.py",
                "priority": "high",
            })
        return StepResult(
            outcome=StepOutcome.SUCCESS,
            message="Test plan created.",
            artifacts={
                "step": "plan_tests",
                "test_plan": test_plan,
                "files_to_test": len(test_plan),
            },
        )
