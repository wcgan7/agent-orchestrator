"""Run Codex worker actions for planning, implementation, and review with allowlist enforcement."""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from ..constants import (
    ERROR_TYPE_CODEX_EXIT,
    ERROR_TYPE_DISALLOWED_FILES,
    ERROR_TYPE_HEARTBEAT_TIMEOUT,
    ERROR_TYPE_SHIFT_TIMEOUT,
    IGNORED_REVIEW_PATH_PREFIXES,
)
from ..git_utils import (
    _diff_file_sets,
    _git_changed_files,
    _git_diff_stat,
    _git_diff_text,
    _git_status_porcelain,
    _snapshot_repo_changes,
    _validate_changes_for_mode,
)
from ..io_utils import _load_data, _read_log_tail, _read_text_for_prompt, _render_json_for_prompt, _save_data
from ..models import (
    AllowlistViolation,
    NoIntroducedChanges,
    ProgressHumanBlockers,
    ResumePromptResult,
    ReviewResult,
    TaskStep,
    WorkerFailed,
    WorkerSucceeded,
)
from ..prompts import (
    _build_impl_plan_prompt,
    _build_phase_prompt,
    _build_plan_prompt,
    _build_resume_prompt,
    _build_review_prompt,
    _build_simple_review_prompt,
    _extract_prd_markers,
)
from ..prompts_local import (
    build_local_impl_plan_prompt,
    build_local_implement_prompt,
    build_local_plan_prompt,
    build_local_review_prompt,
)
from ..signals import build_allowed_files
from ..tasks import _impl_plan_path, _read_progress_human_blockers, _review_output_path
from ..utils import _hash_json_data
from ..validation import _extract_review_blocker_files
from ..workers import WorkersRuntimeConfig, get_workers_runtime_config, resolve_worker_for_step, run_worker
from ..workers.output import WorkerOutputError, extract_json_object, extract_paths_from_unified_diff, normalize_patch_text
from ..workers.patch import apply_patch_with_git, patch_paths_allowed
from .load_plan import load_plan
from .load_review import load_review
from .validate_plan import validate_plan
from .validate_review import validate_review


def _repo_file_list(project_dir: Path, max_lines: int = 250) -> str:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return "(git ls-files failed)"
    if result.returncode != 0:
        return (result.stderr or result.stdout or "(git ls-files failed)").strip()
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = "\n".join(lines[:max_lines])
    return f"{head}\n... ({len(lines)} total files; truncated to {max_lines})"


def _read_agents_text(project_dir: Path, max_chars: int = 12000) -> str:
    text, _ = _read_text_for_prompt(project_dir / "AGENTS.md", max_chars=max_chars)
    return text


def run_resume_prompt_action(
    *,
    user_prompt: str,
    project_dir: Path,
    run_dir: Path,
    run_id: str,
    codex_command: str,
    progress_path: Path,
    heartbeat_seconds: int,
    heartbeat_grace_seconds: int,
    shift_minutes: int,
    on_spawn: Optional[Callable[[int], None]] = None,
    workers_runtime: Optional[WorkersRuntimeConfig] = None,
) -> ResumePromptResult | ProgressHumanBlockers | WorkerFailed:
    """Run a standalone resume prompt action.

    Args:
        user_prompt: Human-provided instructions to be executed by the worker.
        project_dir: Repository root directory.
        run_dir: Per-run output directory.
        run_id: Current run identifier.
        codex_command: Command template used to run Codex.
        progress_path: Path to the progress file written by the worker.
        heartbeat_seconds: Heartbeat interval expected from the worker.
        heartbeat_grace_seconds: Allowed heartbeat staleness before terminating.
        shift_minutes: Timebox for the worker run.
        on_spawn: Optional callback invoked with the worker PID.

    Returns:
        A `ResumePromptResult` on success, or a `ProgressHumanBlockers`/`WorkerFailed` event on failure.
    """
    timeout_seconds = shift_minutes * 60
    workers_runtime = workers_runtime or get_workers_runtime_config(
        config={},
        codex_command_fallback=codex_command,
        cli_worker=None,
    )
    spec = resolve_worker_for_step(workers_runtime, TaskStep.RESUME_PROMPT)

    prompt: str
    if spec.type == "codex":
        prompt = _build_resume_prompt(
            user_prompt=user_prompt,
            progress_path=progress_path,
            run_id=run_id,
            heartbeat_seconds=heartbeat_seconds,
        )
    else:
        agents_text = _read_agents_text(project_dir)
        repo_file_list = _repo_file_list(project_dir)
        agents_block = f"Repository rules (AGENTS.md):\n{agents_text}\n" if agents_text.strip() else ""
        prompt = f"""You have been given the following instructions to complete:

{user_prompt}

You DO NOT have filesystem access. Return JSON only; do not write files.

{agents_block}

Repository file list (partial):
{repo_file_list}

Output (JSON only; no markdown fences, no extra text):
{{
  "patch": "diff --git a/path b/path\\n... (unified diff; may be empty if no changes needed)",
  "human_blocking_issues": [],
  "human_next_steps": [],
  "notes": "optional"
}}
"""

    run_result = run_worker(
        spec=spec,
        prompt=prompt,
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=timeout_seconds,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_grace_seconds=heartbeat_grace_seconds,
        progress_path=progress_path,
        expected_run_id=run_id,
        on_spawn=on_spawn,
    )

    stdout_tail = _read_log_tail(Path(run_result.stdout_path))
    stderr_tail = _read_log_tail(Path(run_result.stderr_path))

    if spec.type == "codex":
        human_blocking, human_next_steps = _read_progress_human_blockers(
            progress_path,
            expected_run_id=run_id,
        )
        if human_blocking:
            return ProgressHumanBlockers(
                run_id=run_id,
                issues=human_blocking,
                next_steps=human_next_steps,
            )
    else:
        try:
            obj = extract_json_object(run_result.response_text or stdout_tail)
            issues = obj.get("human_blocking_issues") or []
            next_steps = obj.get("human_next_steps") or []
            human_blocking = [str(x).strip() for x in issues if str(x).strip()] if isinstance(issues, list) else []
            human_next_steps = (
                [str(x).strip() for x in next_steps if str(x).strip()] if isinstance(next_steps, list) else []
            )
            if human_blocking:
                return ProgressHumanBlockers(run_id=run_id, issues=human_blocking, next_steps=human_next_steps)

            patch = normalize_patch_text(str(obj.get("patch") or ""))
            ok_apply, apply_err = apply_patch_with_git(project_dir=project_dir, patch_text=patch, run_dir=run_dir)
            if not ok_apply:
                return WorkerFailed(
                    step=TaskStep.RESUME_PROMPT,
                    run_id=run_id,
                    error_type="patch_apply_failed",
                    error_detail=apply_err or "Failed to apply patch",
                    stderr_tail=stderr_tail,
                    timed_out=bool(run_result.timed_out),
                    no_heartbeat=bool(run_result.no_heartbeat),
                )
        except WorkerOutputError as exc:
            return WorkerFailed(
                step=TaskStep.RESUME_PROMPT,
                run_id=run_id,
                error_type="output_parse_failed",
                error_detail=str(exc),
                stderr_tail=stderr_tail,
                timed_out=bool(run_result.timed_out),
                no_heartbeat=bool(run_result.no_heartbeat),
            )

    failure, error_type, error_detail = _classify_worker_failure(
        run_result={
            "exit_code": run_result.exit_code,
            "timed_out": run_result.timed_out,
            "no_heartbeat": run_result.no_heartbeat,
        },
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        worker_label="codex" if spec.type == "codex" else spec.type,
    )

    if failure:
        return WorkerFailed(
            step=TaskStep.RESUME_PROMPT,
            run_id=run_id,
            error_type=error_type or "worker_failed",
            error_detail=error_detail or "Resume prompt worker failed",
            stderr_tail=stderr_tail,
            timed_out=bool(run_result.timed_out),
            no_heartbeat=bool(run_result.no_heartbeat),
        )

    # Success - agent completed the instructions without blocking issues
    return ResumePromptResult(
        run_id=run_id,
        succeeded=True,
    )


def _copy_artifact_to_run_dir(artifact_path: Path, run_dir: Path) -> None:
    """Copy an artifact file to the run directory for easier debugging."""
    if artifact_path.exists():
        dest = run_dir / artifact_path.name
        try:
            shutil.copy2(artifact_path, dest)
        except Exception:
            pass  # Non-critical; silently ignore copy failures


def _classify_worker_failure(
    *,
    run_result: dict[str, Any],
    stdout_tail: str,
    stderr_tail: str,
    worker_label: str = "codex",
) -> tuple[bool, Optional[str], Optional[str]]:
    failure = run_result["exit_code"] != 0 or run_result["no_heartbeat"]
    error_detail = None
    error_type = None
    if run_result["no_heartbeat"]:
        error_detail = "No heartbeat received within grace period"
        error_type = ERROR_TYPE_HEARTBEAT_TIMEOUT
    elif run_result["timed_out"]:
        error_detail = "Shift timed out"
        error_type = ERROR_TYPE_SHIFT_TIMEOUT
    elif run_result["exit_code"] != 0:
        error_detail = f"{worker_label} worker exited with code {run_result['exit_code']}"
        error_type = ERROR_TYPE_CODEX_EXIT if worker_label == "codex" else "worker_exit"
        if stderr_tail.strip():
            error_detail = f"{error_detail}. stderr: {stderr_tail.strip()}"
        elif stdout_tail.strip():
            error_detail = f"{error_detail}. stdout: {stdout_tail.strip()}"
    return failure, error_type, error_detail


def run_worker_action(
    *,
    step: TaskStep,
    task: dict[str, Any],
    phase: Optional[dict[str, Any]],
    prd_path: Path,
    project_dir: Path,
    artifacts_dir: Path,
    phase_plan_path: Path,
    task_queue_path: Path,
    run_dir: Path,
    run_id: str,
    codex_command: str,
    user_prompt: Optional[str],
    progress_path: Path,
    events_path: Path,
    heartbeat_seconds: int,
    heartbeat_grace_seconds: int,
    shift_minutes: int,
    test_command: Optional[str],
    on_spawn: Optional[Callable[[int], None]] = None,
    simple_review: bool = False,
    workers_runtime: Optional[WorkersRuntimeConfig] = None,
) -> Any:
    """Run a worker action for the given task step.

    Args:
        step: Task step being executed (plan, plan-impl, implement, review).
        task: Task payload read from the durable task queue.
        phase: Optional phase metadata for the task.
        prd_path: Path to the PRD file.
        project_dir: Repository root directory.
        artifacts_dir: Directory for durable artifacts (plans, reviews, logs).
        phase_plan_path: Path to the phase plan YAML file.
        task_queue_path: Path to the task queue YAML file.
        run_dir: Per-run directory used to store manifests and logs.
        run_id: Current run identifier.
        codex_command: Command template used to run Codex.
        user_prompt: Optional extra prompt content to inject.
        progress_path: Progress file path used for worker heartbeat and results.
        events_path: Events jsonl path for logging.
        heartbeat_seconds: Heartbeat interval expected from the worker.
        heartbeat_grace_seconds: Allowed heartbeat staleness before terminating.
        shift_minutes: Timebox for the worker run.
        test_command: Optional test command used for plan validation context.
        on_spawn: Optional callback invoked with the worker PID.
        simple_review: Whether to use the simplified review schema/prompt.

    Returns:
        An event model describing the result (e.g., `WorkerSucceeded`, `WorkerFailed`,
        `AllowlistViolation`, `ReviewResult`).
    """
    pre_run_changed = _snapshot_repo_changes(project_dir)
    repo_dirty = bool(pre_run_changed)

    phase_id = phase.get("id") if phase else task.get("phase_id") or task.get("id")
    prd_text, prd_truncated = _read_text_for_prompt(prd_path)
    prd_markers = _extract_prd_markers(prd_text)

    prompt = ""
    allowed_files: list[str] = []
    mode = "implement"

    if task.get("type") == "plan":
        mode = "plan"
        workers_runtime = workers_runtime or get_workers_runtime_config(
            config={},
            codex_command_fallback=codex_command,
            cli_worker=None,
        )
        # Plan tasks are represented by task.type="plan" rather than a dedicated TaskStep.
        spec = resolve_worker_for_step(workers_runtime, "plan")
        agents_text = _read_agents_text(project_dir) if spec.type != "codex" else ""
        repo_file_list = _repo_file_list(project_dir) if spec.type != "codex" else ""
        if spec.type == "codex":
            prompt = _build_plan_prompt(
                prd_path=prd_path,
                phase_plan_path=phase_plan_path,
                task_queue_path=task_queue_path,
                events_path=events_path,
                progress_path=progress_path,
                run_id=run_id,
                user_prompt=user_prompt,
                heartbeat_seconds=heartbeat_seconds,
            )
        else:
            prompt = build_local_plan_prompt(
                prd_path=prd_path,
                prd_text=prd_text,
                phase_plan_path=phase_plan_path,
                task_queue_path=task_queue_path,
                repo_file_list=repo_file_list,
                agents_text=agents_text,
                user_prompt=user_prompt,
            )
    elif step == TaskStep.PLAN_IMPL:
        mode = "plan_impl"
        plan_path = _impl_plan_path(artifacts_dir, str(phase_id))
        workers_runtime = workers_runtime or get_workers_runtime_config(
            config={},
            codex_command_fallback=codex_command,
            cli_worker=None,
        )
        spec = resolve_worker_for_step(workers_runtime, step)
        agents_text = _read_agents_text(project_dir) if spec.type != "codex" else ""
        repo_file_list = _repo_file_list(project_dir) if spec.type != "codex" else ""
        phase_test_command = None
        if phase:
            phase_test_command = phase.get("test_command") or task.get("test_command") or test_command
        if spec.type == "codex":
            prompt = _build_impl_plan_prompt(
                phase=phase or {"id": phase_id, "acceptance_criteria": []},
                prd_path=prd_path,
                prd_text=prd_text,
                prd_truncated=prd_truncated,
                prd_markers=prd_markers,
                impl_plan_path=plan_path,
                user_prompt=user_prompt,
                progress_path=progress_path,
                run_id=run_id,
                test_command=phase_test_command,
                heartbeat_seconds=heartbeat_seconds,
                plan_expansion_request=task.get("plan_expansion_request") or [],
            )
        else:
            prompt = build_local_impl_plan_prompt(
                phase=phase or {"id": phase_id, "acceptance_criteria": []},
                prd_path=prd_path,
                prd_text=prd_text,
                prd_markers=prd_markers,
                impl_plan_path=plan_path,
                user_prompt=user_prompt,
                agents_text=agents_text,
                repo_file_list=repo_file_list,
                test_command=phase_test_command,
                plan_expansion_request=task.get("plan_expansion_request") or [],
            )
    elif step == TaskStep.REVIEW:
        mode = "review"
        if not phase:
            raise ValueError("Phase not found for review")
        workers_runtime = workers_runtime or get_workers_runtime_config(
            config={},
            codex_command_fallback=codex_command,
            cli_worker=None,
        )
        spec = resolve_worker_for_step(workers_runtime, step)
        agents_text = _read_agents_text(project_dir) if spec.type != "codex" else ""
        plan_path = _impl_plan_path(artifacts_dir, str(phase_id))
        plan_data = _load_data(plan_path, {})
        plan_text, plan_truncated = _render_json_for_prompt(plan_data)
        diff_text, diff_truncated = _git_diff_text(project_dir)
        diff_stat, diff_stat_truncated = _git_diff_stat(project_dir)
        status_text, status_truncated = _git_status_porcelain(project_dir)
        review_path = _review_output_path(artifacts_dir, str(phase_id))
        changed_files = _git_changed_files(
            project_dir,
            include_untracked=True,
            ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
        )
        tests_snapshot = task.get("last_verification") if isinstance(task, dict) else None
        if spec.type == "codex":
            prompt_builder = _build_simple_review_prompt if simple_review else _build_review_prompt
            prompt = prompt_builder(
                phase=phase,
                review_path=review_path,
                prd_path=prd_path,
                prd_text=prd_text,
                prd_truncated=prd_truncated,
                prd_markers=prd_markers,
                user_prompt=user_prompt,
                progress_path=progress_path,
                run_id=run_id,
                changed_files=changed_files,
                diff_text=diff_text,
                diff_truncated=diff_truncated,
                diff_stat=diff_stat,
                diff_stat_truncated=diff_stat_truncated,
                status_text=status_text,
                status_truncated=status_truncated,
                impl_plan_text=plan_text,
                impl_plan_truncated=plan_truncated,
                heartbeat_seconds=heartbeat_seconds,
                tests_snapshot=tests_snapshot,
            )
        else:
            prompt = build_local_review_prompt(
                phase=phase,
                review_path=review_path,
                prd_path=prd_path,
                prd_text=prd_text,
                prd_markers=prd_markers,
                user_prompt=user_prompt,
                agents_text=agents_text,
                changed_files=changed_files,
                diff_text=diff_text,
                diff_stat=diff_stat,
                status_text=status_text,
                impl_plan_text=plan_text if not plan_truncated else plan_text + "\n[runner] ... plan truncated ...\n",
                tests_snapshot=tests_snapshot,
                simple_review=simple_review,
            )
    else:
        if not phase:
            raise ValueError("Phase not found for implement")
        workers_runtime = workers_runtime or get_workers_runtime_config(
            config={},
            codex_command_fallback=codex_command,
            cli_worker=None,
        )
        spec = resolve_worker_for_step(workers_runtime, step)
        agents_text = _read_agents_text(project_dir) if spec.type != "codex" else ""
        plan_path = _impl_plan_path(artifacts_dir, str(phase_id))
        plan_data = _load_data(plan_path, {})
        tech_approach = ""
        if "technical_approach" in plan_data:
            raw = plan_data["technical_approach"]
            if isinstance(raw, list):
                tech_approach = "\n".join(str(x) for x in raw)
            else:
                tech_approach = str(raw)
        elif "steps" in plan_data:
            tech_approach = json.dumps(plan_data["steps"], indent=2)

        allowed_files = build_allowed_files(plan_data)
        if spec.type == "codex":
            prompt = _build_phase_prompt(
                prd_path=prd_path,
                phase=phase,
                task=task,
                events_path=events_path,
                progress_path=progress_path,
                run_id=run_id,
                user_prompt=user_prompt,
                impl_plan_path=plan_path,
                allowed_files=allowed_files,
                no_progress_attempts=int(task.get("no_progress_attempts", 0)),
                technical_approach_text=tech_approach,
                heartbeat_seconds=heartbeat_seconds,
                prompt_mode=task.get("prompt_mode"),
                last_verification=task.get("last_verification"),
                review_blockers=task.get("review_blockers"),
                review_blocker_files=task.get("review_blocker_files"),
            )
        else:
            plan_text, plan_truncated2 = _render_json_for_prompt(plan_data, max_chars=40_000)
            if plan_truncated2:
                plan_text += "\n[runner] ... plan truncated ...\n"
            repo_context_files: dict[str, str] = {}
            for rel in sorted({*(plan_data.get("files_to_change") or []), *(plan_data.get("new_files") or [])}):
                if not isinstance(rel, str) or not rel.strip():
                    continue
                rel_path = rel.strip().lstrip("./")
                abs_path = project_dir / rel_path
                if abs_path.exists() and abs_path.is_file():
                    content, truncated = _read_text_for_prompt(abs_path, max_chars=30_000)
                    if truncated:
                        content += "\n\n[runner] ... file truncated for prompt ...\n"
                    repo_context_files[rel_path] = content
            prompt = build_local_implement_prompt(
                prd_path=prd_path,
                phase=phase,
                task=task,
                impl_plan_path=plan_path,
                impl_plan_text=plan_text,
                allowed_files=allowed_files,
                agents_text=agents_text,
                repo_context_files=repo_context_files,
                user_prompt=user_prompt,
            )

    timeout_seconds = shift_minutes * 60
    run_result = run_worker(
        spec=spec,
        prompt=prompt,
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=timeout_seconds,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_grace_seconds=heartbeat_grace_seconds,
        progress_path=progress_path,
        expected_run_id=run_id,
        on_spawn=on_spawn,
    )

    stdout_tail = _read_log_tail(Path(run_result.stdout_path))
    stderr_tail = _read_log_tail(Path(run_result.stderr_path))

    # Non-agentic workers: interpret structured JSON output and write/apply results.
    if spec.type != "codex":
        try:
            obj = extract_json_object(run_result.response_text or stdout_tail)
            issues = obj.get("human_blocking_issues") or []
            next_steps = obj.get("human_next_steps") or []
            formatted_issues = [str(x).strip() for x in issues if str(x).strip()] if isinstance(issues, list) else []
            formatted_next = (
                [str(x).strip() for x in next_steps if str(x).strip()] if isinstance(next_steps, list) else []
            )
            if formatted_issues:
                return ProgressHumanBlockers(run_id=run_id, issues=formatted_issues, next_steps=formatted_next)

            if mode == "plan":
                phase_plan = obj.get("phase_plan")
                task_queue = obj.get("task_queue")
                if not isinstance(phase_plan, dict) or not isinstance(task_queue, dict):
                    raise WorkerOutputError("plan output must include 'phase_plan' and 'task_queue' objects")
                _save_data(phase_plan_path, phase_plan)
                _save_data(task_queue_path, task_queue)
            elif mode == "plan_impl":
                plan_path = _impl_plan_path(artifacts_dir, str(phase_id))
                impl_plan = obj.get("impl_plan")
                if not isinstance(impl_plan, dict):
                    raise WorkerOutputError("plan_impl output must include an 'impl_plan' object")
                _save_data(plan_path, impl_plan)
            elif mode == "review":
                review_path = _review_output_path(artifacts_dir, str(phase_id))
                review_obj = obj.get("review")
                if not isinstance(review_obj, dict):
                    raise WorkerOutputError("review output must include a 'review' object")
                _save_data(review_path, review_obj)
            elif mode == "implement":
                patch = normalize_patch_text(str(obj.get("patch") or ""))
                patch_paths = extract_paths_from_unified_diff(patch)
                ok_paths, disallowed = patch_paths_allowed(project_dir, patch_paths, allowed_files)
                if not ok_paths:
                    return AllowlistViolation(
                        run_id=run_id,
                        disallowed_paths=disallowed,
                        changed_files=[],
                        task_id=str(task.get("id")) if task.get("id") else None,
                        phase=str(phase_id) if phase_id is not None else None,
                    )
                ok_apply, apply_err = apply_patch_with_git(project_dir=project_dir, patch_text=patch, run_dir=run_dir)
                if not ok_apply:
                    return WorkerFailed(
                        step=step,
                        run_id=run_id,
                        error_type="patch_apply_failed",
                        error_detail=apply_err or "Failed to apply patch",
                        stderr_tail=stderr_tail,
                        timed_out=bool(run_result.timed_out),
                        no_heartbeat=bool(run_result.no_heartbeat),
                    )
        except WorkerOutputError as exc:
            return WorkerFailed(
                step=step,
                run_id=run_id,
                error_type="output_parse_failed",
                error_detail=str(exc),
                stderr_tail=stderr_tail,
                timed_out=bool(run_result.timed_out),
                no_heartbeat=bool(run_result.no_heartbeat),
            )

    # Agentic worker: read blocking issues from progress file.
    if spec.type == "codex":
        human_blocking, human_next_steps = _read_progress_human_blockers(
            progress_path,
            expected_run_id=run_id,
        )
        if human_blocking:
            return ProgressHumanBlockers(run_id=run_id, issues=human_blocking, next_steps=human_next_steps)

    failure, error_type, error_detail = _classify_worker_failure(
        run_result={
            "exit_code": run_result.exit_code,
            "timed_out": run_result.timed_out,
            "no_heartbeat": run_result.no_heartbeat,
        },
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        worker_label="codex" if spec.type == "codex" else spec.type,
    )

    post_run_changed = _snapshot_repo_changes(project_dir)
    introduced, removed = _diff_file_sets(pre_run_changed, post_run_changed)

    ok, msg, disallowed = _validate_changes_for_mode(
        project_dir=project_dir,
        mode=mode,
        introduced_changes=introduced,
        allowed_files=allowed_files if mode == "implement" else None,
    )

    disallowed_files: list[str] = []
    if not ok:
        disallowed_files = disallowed
        error_detail = msg
        error_type = ERROR_TYPE_DISALLOWED_FILES

    manifest = {
        "run_id": run_id,
        "task_id": task.get("id"),
        "mode": mode,
        "start_time": run_result.start_time,
        "end_time": run_result.end_time,
        "runtime_seconds": run_result.runtime_seconds,
        "exit_code": run_result.exit_code,
        "timed_out": bool(run_result.timed_out),
        "no_heartbeat": bool(run_result.no_heartbeat),
        "changed_files": post_run_changed,
        "disallowed_files": disallowed_files,
        "pre_run_snapshot": pre_run_changed,
        "post_run_snapshot": post_run_changed,
        "introduced_changes": introduced,
        "removed_changes": removed,
        "worker_provider": spec.name,
        "worker_type": spec.type,
    }
    _save_data(run_dir / "manifest.json", manifest)

    if not ok and mode == "implement":
        return AllowlistViolation(
            run_id=run_id,
            disallowed_paths=disallowed_files,
            changed_files=post_run_changed,
            task_id=str(task.get("id")) if task.get("id") else None,
            phase=str(phase_id) if phase_id is not None else None,
        )
    if not ok:
        failure = True

    if failure:
        return WorkerFailed(
            step=step,
            run_id=run_id,
            error_type=error_type or "worker_failed",
            error_detail=error_detail or "Worker failed",
            stderr_tail=stderr_tail,
            timed_out=bool(run_result.timed_out),
            no_heartbeat=bool(run_result.no_heartbeat),
            changed_files=post_run_changed,
            introduced_changes=introduced,
        )

    if task.get("type") == "plan":
        return WorkerSucceeded(
            step=step,
            run_id=run_id,
            changed_files=post_run_changed,
            introduced_changes=introduced,
            repo_dirty=repo_dirty,
        )

    if step == TaskStep.PLAN_IMPL:
        plan_path = _impl_plan_path(artifacts_dir, str(phase_id))
        _copy_artifact_to_run_dir(plan_path, run_dir)
        plan_data = load_plan(plan_path)
        if not plan_data:
            return WorkerSucceeded(
                step=step,
                run_id=run_id,
                changed_files=post_run_changed,
                introduced_changes=introduced,
                repo_dirty=repo_dirty,
                plan_valid=False,
                plan_issue="Implementation plan missing",
            )
        phase_test_command = None
        if phase:
            phase_test_command = phase.get("test_command") or task.get("test_command") or test_command
        plan_expansion_request = task.get("plan_expansion_request") or []
        plan_valid, plan_issue = validate_plan(
            plan_data,
            phase or {"id": phase_id, "acceptance_criteria": []},
            prd_markers=prd_markers,
            prd_truncated=prd_truncated,
            prd_has_content=bool(prd_text.strip()),
            expected_test_command=phase_test_command,
            plan_expansion_request=plan_expansion_request if plan_expansion_request else None,
        )
        plan_hash = _hash_json_data(plan_data)
        return WorkerSucceeded(
            step=step,
            run_id=run_id,
            changed_files=post_run_changed,
            introduced_changes=introduced,
            repo_dirty=repo_dirty,
            plan_valid=bool(plan_valid),
            plan_issue=plan_issue or None,
            impl_plan_path=str(plan_path),
            impl_plan_hash=plan_hash,
        )

    if step == TaskStep.REVIEW:
        review_path = _review_output_path(artifacts_dir, str(phase_id))
        _copy_artifact_to_run_dir(review_path, run_dir)
        review_data = load_review(review_path)
        changed_files = _git_changed_files(
            project_dir,
            include_untracked=True,
            ignore_prefixes=IGNORED_REVIEW_PATH_PREFIXES,
        )
        valid_review, review_issue = validate_review(
            review_data,
            phase or {"id": phase_id},
            changed_files=changed_files,
            prd_markers=prd_markers,
            prd_truncated=prd_truncated,
            prd_has_content=bool(prd_text.strip()),
            simple_review=simple_review,
        )
        if not valid_review:
            return WorkerFailed(
                run_id=run_id,
                step=step,
                error_type="invalid_review",
                error_detail=review_issue or "Review output invalid",
                changed_files=changed_files,
                task_id=str(task.get("id")) if task.get("id") else None,
                phase=str(phase_id) if phase_id is not None else None,
            )

        raw_issues = review_data.get("issues") or []
        formatted_issues: list[str] = []
        for item in raw_issues:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    formatted_issues.append(text)
                continue
            if isinstance(item, dict):
                severity = str(item.get("severity", "")).strip().upper()
                summary = str(item.get("summary") or item.get("text") or "").strip()
                if severity and summary:
                    formatted_issues.append(f"{severity}: {summary}")
                elif summary:
                    formatted_issues.append(summary)
                continue

        mergeable = bool(review_data.get("mergeable", True))
        return ReviewResult(
            run_id=run_id,
            mergeable=mergeable,
            issues=formatted_issues,
            review_path=str(review_path),
            task_id=str(task.get("id")) if task.get("id") else None,
            phase=str(phase_id) if phase_id is not None else None,
        )

    if step == TaskStep.IMPLEMENT:
        if not introduced:
            return NoIntroducedChanges(
                run_id=run_id,
                repo_dirty=repo_dirty,
                changed_files=post_run_changed,
                task_id=str(task.get("id")) if task.get("id") else None,
                phase=str(phase_id) if phase_id is not None else None,
            )
        return WorkerSucceeded(
            step=step,
            run_id=run_id,
            changed_files=post_run_changed,
            introduced_changes=introduced,
            repo_dirty=repo_dirty,
        )

    return WorkerSucceeded(
        step=step,
        run_id=run_id,
        changed_files=post_run_changed,
        introduced_changes=introduced,
        repo_dirty=repo_dirty,
    )
