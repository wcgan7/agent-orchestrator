"""Worker adapter that dispatches pipeline steps to real Codex/Ollama providers."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...pipelines.registry import PipelineRegistry
from ...workers.config import get_workers_runtime_config, resolve_worker_for_step
from ...workers.diagnostics import test_worker
from ...workers.run import WorkerRunResult, run_worker
from ..domain.models import Task
from ..storage.container import Container
from .worker_adapter import StepResult

logger = logging.getLogger(__name__)

# Step category mapping
_PLANNING_STEPS = {"plan", "plan_impl", "analyze"}
_IMPL_STEPS = {"implement", "implement_fix", "prototype"}
_VERIFY_STEPS = {"verify", "benchmark", "reproduce"}
_REVIEW_STEPS = {"review"}
_REPORT_STEPS = {"report", "summarize"}
_SCAN_STEPS = {"scan", "scan_deps", "scan_code", "gather"}
_TASK_GEN_STEPS = {"generate_tasks", "diagnose"}
_MERGE_RESOLVE_STEPS = {"resolve_merge"}
_DEP_ANALYSIS_STEPS = {"analyze_deps"}
_STEP_TIMEOUT_ALIASES = {"implement_fix": "implement"}
_DEFAULT_STEP_TIMEOUT_SECONDS = 600

# ---------------------------------------------------------------------------
# Prompt layers
# ---------------------------------------------------------------------------

_PREAMBLE = (
    "You are an autonomous coding agent managed by a coordinator process.\n"
    "The coordinator is the final authority on task state — it assigns steps,\n"
    "tracks progress, and handles all git commits.\n\n"
    "## Human-blocking issues\n"
    "If you encounter a problem that genuinely cannot be resolved without human\n"
    "intervention, report it as a human-blocking issue. Valid reasons:\n"
    "specification is missing or contradictory, required credentials or access\n"
    "are unavailable. Do NOT escalate code-quality concerns, design preferences,\n"
    "refactoring suggestions, or review feedback — handle those within your\n"
    "step output."
)

_GUARDRAILS = (
    "## Guardrails\n"
    "- Do NOT commit, push, or rebase — the coordinator handles all commits.\n"
    "- Do NOT modify files under `.agent_orchestrator/` — those are coordinator state.\n"
    "- Do NOT suppress or down-rank review findings.\n"
    "- Prefer fixing issues over escalating; escalate only when truly stuck.\n"
    "- Be explicit about risks, uncertainty, and assumptions."
)

_LANGUAGE_STANDARDS: dict[str, str] = {
    "python": (
        "## Language standards — Python\n"
        "- Google-style docstrings; module-level docstring in every file.\n"
        "- Type hints (Python 3.10+ syntax). Aim for mypy strict compliance.\n"
        "- Format with ruff; lint with ruff check."
    ),
    "typescript": (
        "## Language standards — TypeScript\n"
        "- JSDoc on exported symbols. Strict tsconfig (no `any`).\n"
        "- Compile-check with tsc --noEmit. Lint with ESLint."
    ),
    "javascript": (
        "## Language standards — JavaScript\n"
        "- JSDoc on exported symbols.\n"
        "- Lint with ESLint; format with Prettier."
    ),
    "go": (
        "## Language standards — Go\n"
        "- Godoc conventions on exported symbols.\n"
        "- Format with gofmt; lint with golangci-lint."
    ),
    "rust": (
        "## Language standards — Rust\n"
        "- `///` doc comments on public items.\n"
        "- Format with cargo fmt; lint with cargo clippy."
    ),
}

_LANGUAGE_MARKERS: list[tuple[str, str]] = [
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("tsconfig.json", "typescript"),
    ("package.json", "javascript"),
    ("go.mod", "go"),
    ("Cargo.toml", "rust"),
]


def detect_project_languages(project_dir: Path) -> list[str]:
    """Return all detected project languages based on marker files.

    Multiple markers for the same language are deduplicated (e.g. pyproject.toml
    and setup.py both map to "python").  If a tsconfig.json is found alongside
    package.json, only "typescript" is returned (it subsumes "javascript").
    """
    seen: dict[str, None] = {}  # ordered set
    for marker, lang in _LANGUAGE_MARKERS:
        if (project_dir / marker).exists() and lang not in seen:
            seen[lang] = None
    langs = list(seen)
    # TypeScript subsumes JavaScript — drop the weaker signal
    if "typescript" in seen and "javascript" in seen:
        langs.remove("javascript")
    return langs


_LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "python": "Python",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "go": "Go",
    "rust": "Rust",
}


def _format_project_commands(
    project_commands: dict[str, dict[str, str]],
    project_languages: list[str],
) -> str:
    """Format project commands for detected languages into a prompt section."""
    _COMMAND_LABELS = {"test": "Test", "lint": "Lint", "typecheck": "Typecheck", "format": "Format"}
    blocks: list[tuple[str, list[str]]] = []
    for lang in project_languages:
        cmds = project_commands.get(lang)
        if not isinstance(cmds, dict):
            continue
        lines = []
        for key in ("test", "lint", "typecheck", "format"):
            val = cmds.get(key)
            if isinstance(val, str) and val.strip():
                lines.append(f"- {_COMMAND_LABELS[key]}: `{val.strip()}`")
        if lines:
            blocks.append((lang, lines))
    if not blocks:
        return ""
    parts = ["## Project commands"]
    if len(blocks) == 1:
        parts.extend(blocks[0][1])
    else:
        for lang, lines in blocks:
            display = _LANGUAGE_DISPLAY_NAMES.get(lang, lang.title())
            parts.append(f"### {display}")
            parts.extend(lines)
    return "\n".join(parts)


def _step_category(step: str) -> str:
    if step in _PLANNING_STEPS:
        return "planning"
    if step in _IMPL_STEPS:
        return "implementation"
    if step in _VERIFY_STEPS:
        return "verification"
    if step in _REVIEW_STEPS:
        return "review"
    if step in _REPORT_STEPS:
        return "reporting"
    if step in _SCAN_STEPS:
        return "scanning"
    if step in _TASK_GEN_STEPS:
        return "task_generation"
    if step in _MERGE_RESOLVE_STEPS:
        return "merge_resolution"
    if step in _DEP_ANALYSIS_STEPS:
        return "dependency_analysis"
    return "general"


_CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "planning": (
        "Create a scoped, independently testable plan for the following task.\n"
        "Describe a coherent technical approach. Do not assume infrastructure or\n"
        "services that are not already present. Planning does not modify\n"
        "repository code."
    ),
    "implementation": (
        "Implement the changes described in the following task.\n"
        "Complete the entire step fully — partial work leaves the repository in\n"
        "an inconsistent state. Update README or docs when observable behavior\n"
        "changes."
    ),
    "verification": (
        "Run the project's test, lint, and type-check commands for the following\n"
        "task. Do not bypass or skip tests. Report results accurately — do not\n"
        "mask failures. If you can identify the root cause of a failure, note it\n"
        "clearly so the next step can address it."
    ),
    "review": (
        "Review the implementation and list findings.\n"
        "Each finding must include a severity (critical / high / medium / low).\n"
        "Evaluate every acceptance criterion explicitly. Provide concrete\n"
        "evidence tied to files and diffs — do not speculate. Do not down-rank\n"
        "findings."
    ),
    "reporting": (
        "Produce a summary report for the following task.\n"
        "Tie conclusions to concrete evidence. Be explicit about risks and\n"
        "remaining uncertainty."
    ),
    "scanning": (
        "Scan and gather information for the following task.\n"
        "Report findings with severity and file locations. Provide concrete\n"
        "evidence only."
    ),
    "task_generation": (
        "Generate subtasks for the following task.\n"
        "Each subtask must be independently implementable. Include title,\n"
        "description, task_type, and priority. Cover the full scope without\n"
        "overlap.\n"
        "If a plan is provided, decompose it into ordered subtasks and specify\n"
        "depends_on indices for tasks that must complete before others can start."
    ),
    "merge_resolution": "Resolve the merge conflicts in the following files. Both tasks' objectives must be fulfilled in the resolution.",
    "dependency_analysis": (
        "Analyze task dependencies for this codebase.\n\n"
        "First, examine the project structure to understand what already exists:\n"
        "- Look at the directory layout and key files\n"
        "- Check existing modules, APIs, and shared code\n"
        "- Identify what infrastructure is already in place\n\n"
        "Then, given the pending tasks below, determine which tasks depend on others.\n"
        "A task B depends on task A if:\n"
        "- B requires code, APIs, schemas, or artifacts that task A will CREATE (not already existing)\n"
        "- B imports or builds on modules that task A will introduce\n"
        "- B cannot produce correct results without task A's changes being present\n\n"
        "Do NOT create a dependency if:\n"
        "- Both tasks touch the same area but don't actually need each other's output\n"
        "- The dependency is based on vague thematic similarity\n"
        "- The required code/API already exists in the codebase\n\n"
        "If tasks can safely run in parallel, leave them independent."
    ),
    "general": "Follow the task description and report results clearly.",
}

_CATEGORY_JSON_SCHEMAS: dict[str, str] = {
    "planning": '{"plan": "string describing the plan"}',
    "implementation": '{"patch": "unified diff of changes", "summary": "description of changes"}',
    "verification": '{"status": "pass|fail", "summary": "test results summary"}',
    "review": '{"findings": [{"severity": "critical|high|medium|low", "category": "string", "summary": "string", "file": "path", "line": 0, "suggested_fix": "string"}]}',
    "reporting": '{"summary": "detailed report text"}',
    "scanning": '{"findings": [{"severity": "critical|high|medium|low", "category": "string", "summary": "string", "file": "path"}]}',
    "task_generation": '{"tasks": [{"title": "string", "description": "string", "task_type": "feature|bugfix|research", "priority": "P0|P1|P2|P3", "depends_on": [0, 1]}]}',
    "merge_resolution": '{"status": "ok|error", "summary": "string"}',
    "dependency_analysis": '{"edges": [{"from": "task_id_first", "to": "task_id_depends", "reason": "why"}]}',
    "general": '{"status": "ok|error", "summary": "string"}',
}


def build_step_prompt(
    *,
    task: Task,
    step: str,
    attempt: int,
    is_codex: bool,
    project_languages: list[str] | None = None,
    project_commands: dict[str, dict[str, str]] | None = None,
) -> str:
    """Build a prompt from Task fields with step-specific instructions."""
    category = _step_category(step)
    instruction = _CATEGORY_INSTRUCTIONS[category]

    # Special prompt for dependency analysis
    if category == "dependency_analysis" and isinstance(task.metadata, dict):
        parts = [_PREAMBLE, "", instruction, ""]

        candidate_tasks = task.metadata.get("candidate_tasks")
        if isinstance(candidate_tasks, list) and candidate_tasks:
            parts.append("## Tasks to analyze")
            parts.append("")
            for ct in candidate_tasks:
                if not isinstance(ct, dict):
                    continue
                parts.append(f"- ID: {ct.get('id', '?')}")
                parts.append(f"  Title: {ct.get('title', '?')}")
                desc = str(ct.get("description") or "")[:200]
                if desc:
                    parts.append(f"  Description: {desc}")
                parts.append(f"  Type: {ct.get('task_type', 'feature')}")
                labels = ct.get("labels")
                if isinstance(labels, list) and labels:
                    parts.append(f"  Labels: {', '.join(str(l) for l in labels)}")
                parts.append("")

        existing_tasks = task.metadata.get("existing_tasks")
        if isinstance(existing_tasks, list) and existing_tasks:
            parts.append("## Already-scheduled tasks (may be blockers)")
            parts.append("")
            for et in existing_tasks:
                if not isinstance(et, dict):
                    continue
                parts.append(f"- ID: {et.get('id', '?')}")
                parts.append(f"  Title: {et.get('title', '?')}")
                parts.append(f"  Status: {et.get('status', '?')}")
                parts.append("")

        parts.append("## Rules")
        parts.append("- Only output edges where one task MUST complete before another can start.")
        parts.append("- Use the exact task IDs from above.")
        parts.append("- If all tasks are independent, return an empty edges array.")
        parts.append("- Do not create circular dependencies.")

        parts.append("")
        parts.append(_GUARDRAILS)

        if not is_codex:
            schema = _CATEGORY_JSON_SCHEMAS["dependency_analysis"]
            parts.append("")
            parts.append(f"Respond with valid JSON matching this schema: {schema}")

        return "\n".join(parts)

    parts = [_PREAMBLE, "", instruction, ""]

    parts.append(f"Task: {task.title}")
    if task.description:
        parts.append(f"Description: {task.description}")
    parts.append(f"Type: {task.task_type}")
    parts.append(f"Priority: {task.priority}")
    parts.append(f"Step: {step}")
    if attempt > 1:
        parts.append(f"Attempt: {attempt}")

    # Include review findings for fix steps
    review_findings = task.metadata.get("review_findings") if isinstance(task.metadata, dict) else None
    if review_findings and isinstance(review_findings, list):
        parts.append("")
        parts.append("Review findings to address:")
        for finding in review_findings:
            if isinstance(finding, dict):
                sev = finding.get("severity", "medium")
                summary = finding.get("summary", "")
                file_ = finding.get("file", "")
                line_ = finding.get("line", "")
                loc = f" ({file_}:{line_})" if file_ else ""
                parts.append(f"  - [{sev}] {summary}{loc}")

    # Include plan context for task generation
    plan_for_generation = task.metadata.get("plan_for_generation") if isinstance(task.metadata, dict) else None
    if plan_for_generation and category == "task_generation":
        parts.append("")
        parts.append("## Plan to decompose into subtasks")
        parts.append(str(plan_for_generation))
        parts.append("")
        parts.append(
            "Decompose this plan into ordered subtasks. Use the depends_on field "
            "(array of task indices) to specify execution order where needed."
        )

    # Include merge conflict context for resolve_merge step
    if category == "merge_resolution" and isinstance(task.metadata, dict):
        conflict_files = task.metadata.get("merge_conflict_files")
        if isinstance(conflict_files, dict):
            parts.append("")
            parts.append("Conflicted files (with <<<<<<< / ======= / >>>>>>> markers):")
            for fpath, content in conflict_files.items():
                parts.append(f"\n--- {fpath} ---")
                parts.append(content)

        other_tasks = task.metadata.get("merge_other_tasks")
        if isinstance(other_tasks, list) and other_tasks:
            parts.append("")
            parts.append("Other task(s) whose changes conflict with this task:")
            for info in other_tasks:
                parts.append(str(info))

        parts.append("")
        parts.append("Edit the conflicted files to resolve all conflicts. "
                      "Ensure BOTH this task's and the other task(s)' objectives are preserved.")

    # Inject language standards for implementation and review steps
    if project_languages and category in ("implementation", "review"):
        for lang in project_languages:
            lang_block = _LANGUAGE_STANDARDS.get(lang)
            if lang_block:
                parts.append("")
                parts.append(lang_block)

    # Inject project commands for implementation and verification steps
    if project_commands and project_languages and category in ("implementation", "verification"):
        cmds_block = _format_project_commands(project_commands, project_languages)
        if cmds_block:
            parts.append("")
            parts.append(cmds_block)

    parts.append("")
    parts.append(_GUARDRAILS)

    if not is_codex:
        # Add JSON schema instruction for ollama
        schema = _CATEGORY_JSON_SCHEMAS.get(category, _CATEGORY_JSON_SCHEMAS["general"])
        parts.append("")
        parts.append(f"Respond with valid JSON matching this schema: {schema}")

    return "\n".join(parts)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from text, handling markdown fences."""
    text = text.strip()
    # Try stripping markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner_lines = []
        started = False
        for line in lines:
            if not started:
                if line.strip().startswith("```"):
                    started = True
                    continue
            elif line.strip() == "```":
                break
            else:
                inner_lines.append(line)
        text = "\n".join(inner_lines).strip()

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class LiveWorkerAdapter:
    """Worker adapter that dispatches to real Codex/Ollama providers."""

    def __init__(self, container: Container) -> None:
        self._container = container

    @staticmethod
    def _coerce_timeout(value: Any, default: int = _DEFAULT_STEP_TIMEOUT_SECONDS) -> int:
        try:
            timeout = int(value)
        except (TypeError, ValueError):
            return default
        return timeout if timeout > 0 else default

    def _timeout_for_step(self, task: Task, step: str) -> int:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        overrides = metadata.get("step_timeouts")
        if isinstance(overrides, dict):
            for key in (step, _STEP_TIMEOUT_ALIASES.get(step)):
                if not key:
                    continue
                if key in overrides:
                    return self._coerce_timeout(overrides.get(key))

        try:
            template = PipelineRegistry().resolve_for_task_type(task.task_type)
        except Exception:
            return _DEFAULT_STEP_TIMEOUT_SECONDS

        step_timeouts = {sd.name: self._coerce_timeout(sd.timeout_seconds) for sd in template.steps}
        for key in (step, _STEP_TIMEOUT_ALIASES.get(step)):
            if key and key in step_timeouts:
                return step_timeouts[key]
        return _DEFAULT_STEP_TIMEOUT_SECONDS

    @staticmethod
    def _human_blocker_summary(issues: list[dict[str, str]]) -> str:
        count = len(issues)
        first = issues[0].get("summary", "").strip() if issues else ""
        if not first:
            return f"Human intervention required ({count} blocking issue{'s' if count != 1 else ''})."
        suffix = "issue" if count == 1 else "issues"
        return f"Human intervention required ({count} {suffix}): {first}"

    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        # 1. Resolve worker
        try:
            cfg = self._container.config.load()
            runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex exec")
            spec = resolve_worker_for_step(runtime, step)
            if spec.type in {"codex", "claude"}:
                task_model = str(getattr(task, "worker_model", "") or "").strip()
                if not task_model and isinstance(task.metadata, dict):
                    task_model = str(task.metadata.get("worker_model") or "").strip()
                default_model = str(getattr(runtime, "default_model", "") or "").strip() if spec.type == "codex" else ""
                effective_model = task_model or default_model or str(spec.model or "").strip()
                if effective_model and effective_model != (spec.model or ""):
                    spec = replace(spec, model=effective_model)
            available, reason = test_worker(spec)
            if not available:
                return StepResult(status="error", summary=f"Worker not available: {reason}")
        except (ValueError, KeyError) as exc:
            return StepResult(status="error", summary=f"Cannot resolve worker: {exc}")

        # 2. Build prompt
        worktree_path = task.metadata.get("worktree_dir") if isinstance(task.metadata, dict) else None
        project_dir = Path(worktree_path) if worktree_path else self._container.project_dir
        langs = detect_project_languages(project_dir)
        raw_commands = (cfg.get("project") or {}).get("commands") or {}
        project_commands = {
            lang: cmds for lang, cmds in raw_commands.items()
            if isinstance(cmds, dict)
        } or None
        prompt = build_step_prompt(
            task=task, step=step, attempt=attempt,
            is_codex=(spec.type in {"codex", "claude"}), project_languages=langs or None,
            project_commands=project_commands,
        )

        # 3. Execute
        run_dir = Path(tempfile.mkdtemp(dir=str(self._container.state_root)))
        progress_path = run_dir / "progress.json"
        timeout_seconds = self._timeout_for_step(task, step)
        try:
            result = run_worker(
                spec=spec,
                prompt=prompt,
                project_dir=project_dir,
                run_dir=run_dir,
                timeout_seconds=timeout_seconds,
                heartbeat_seconds=30,
                heartbeat_grace_seconds=15,
                progress_path=progress_path,
            )
        except Exception as exc:
            return StepResult(status="error", summary=f"Worker execution failed: {exc}")

        # 4. Map result
        return self._map_result(result, spec, step)

    def _map_result(self, result: WorkerRunResult, spec: Any, step: str) -> StepResult:
        if result.human_blocking_issues:
            return StepResult(
                status="human_blocked",
                summary=self._human_blocker_summary(result.human_blocking_issues),
                human_blocking_issues=result.human_blocking_issues,
            )
        if result.no_heartbeat:
            return StepResult(status="error", summary="Worker stalled (no heartbeat or output activity).")
        if result.timed_out:
            return StepResult(status="error", summary="Worker timed out")
        if result.exit_code != 0:
            summary = f"Worker exited with code {result.exit_code}"
            # Try to include stderr info
            if result.stderr_path:
                try:
                    err_text = Path(result.stderr_path).read_text(errors="replace").strip()
                    if err_text:
                        summary = err_text[:500]
                except Exception:
                    pass
            return StepResult(status="error", summary=summary)

        # Dependency analysis: always parse response text (both codex and ollama)
        category = _step_category(step)
        if category == "dependency_analysis" and result.response_text:
            return self._parse_dep_analysis_output(result.response_text)

        # Parse structured output for ollama
        if spec.type == "ollama" and result.response_text:
            return self._parse_ollama_output(result.response_text, step)

        return StepResult(status="ok")

    def _parse_dep_analysis_output(self, text: str) -> StepResult:
        parsed = _extract_json(text)
        if parsed is None:
            return StepResult(status="ok", dependency_edges=[])
        edges = parsed.get("edges")
        if isinstance(edges, list):
            return StepResult(status="ok", dependency_edges=edges)
        return StepResult(status="ok", dependency_edges=[])

    def _parse_ollama_output(self, text: str, step: str) -> StepResult:
        parsed = _extract_json(text)
        if parsed is None:
            return StepResult(status="ok", summary=text[:500] if text else None)

        category = _step_category(step)

        if category == "review" or category == "scanning":
            findings = parsed.get("findings")
            if isinstance(findings, list):
                return StepResult(status="ok", findings=findings)

        if category == "task_generation":
            tasks = parsed.get("tasks")
            if isinstance(tasks, list):
                return StepResult(status="ok", generated_tasks=tasks)

        if category == "verification":
            status = parsed.get("status", "ok")
            summary = parsed.get("summary")
            mapped_status = "ok" if status in ("ok", "pass") else "error"
            return StepResult(status=mapped_status, summary=summary)

        # For planning, implementation, reporting — extract summary
        summary = parsed.get("summary") or parsed.get("plan")
        return StepResult(status="ok", summary=str(summary) if summary else None)
