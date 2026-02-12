"""Worker adapter that dispatches pipeline steps to real Codex/Ollama providers."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from ...workers.config import get_workers_runtime_config, resolve_worker_for_step
from ...workers.diagnostics import test_worker
from ...workers.run import WorkerRunResult, run_worker
from ..domain.models import Task
from ..storage.container import V3Container
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
    return "general"


_CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "planning": "Create a plan for the following task.",
    "implementation": "Implement the changes described in the following task.",
    "verification": "Run tests and verification checks for the following task.",
    "review": "Review the implementation and list any findings.",
    "reporting": "Produce a summary report for the following task.",
    "scanning": "Scan and gather information for the following task.",
    "task_generation": "Generate subtasks for the following task.",
    "general": "Execute the following task step.",
}

_CATEGORY_JSON_SCHEMAS: dict[str, str] = {
    "planning": '{"plan": "string describing the plan"}',
    "implementation": '{"patch": "unified diff of changes", "summary": "description of changes"}',
    "verification": '{"status": "pass|fail", "summary": "test results summary"}',
    "review": '{"findings": [{"severity": "critical|high|medium|low", "category": "string", "summary": "string", "file": "path", "line": 0, "suggested_fix": "string"}]}',
    "reporting": '{"summary": "detailed report text"}',
    "scanning": '{"findings": [{"severity": "critical|high|medium|low", "category": "string", "summary": "string", "file": "path"}]}',
    "task_generation": '{"tasks": [{"title": "string", "description": "string", "task_type": "feature|bugfix|research", "priority": "P0|P1|P2|P3"}]}',
    "general": '{"status": "ok|error", "summary": "string"}',
}


def build_step_prompt(*, task: Task, step: str, attempt: int, is_codex: bool) -> str:
    """Build a prompt from Task fields with step-specific instructions."""
    category = _step_category(step)
    instruction = _CATEGORY_INSTRUCTIONS[category]

    parts = [instruction, ""]

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

    def __init__(self, container: V3Container) -> None:
        self._container = container

    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        # 1. Resolve worker
        try:
            cfg = self._container.config.load()
            runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex")
            spec = resolve_worker_for_step(runtime, step)
            available, reason = test_worker(spec)
            if not available:
                return StepResult(status="error", summary=f"Worker not available: {reason}")
        except (ValueError, KeyError) as exc:
            return StepResult(status="error", summary=f"Cannot resolve worker: {exc}")

        # 2. Build prompt
        prompt = build_step_prompt(task=task, step=step, attempt=attempt, is_codex=(spec.type == "codex"))

        # 3. Execute
        run_dir = Path(tempfile.mkdtemp(dir=str(self._container.v3_root)))
        progress_path = run_dir / "progress.json"
        try:
            result = run_worker(
                spec=spec,
                prompt=prompt,
                project_dir=self._container.project_dir,
                run_dir=run_dir,
                timeout_seconds=600,
                heartbeat_seconds=30,
                heartbeat_grace_seconds=15,
                progress_path=progress_path,
            )
        except Exception as exc:
            return StepResult(status="error", summary=f"Worker execution failed: {exc}")

        # 5. Map result
        return self._map_result(result, spec, step)

    def _map_result(self, result: WorkerRunResult, spec: Any, step: str) -> StepResult:
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

        # Parse structured output for ollama
        if spec.type == "ollama" and result.response_text:
            return self._parse_ollama_output(result.response_text, step)

        return StepResult(status="ok")

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
