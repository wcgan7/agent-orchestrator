from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

try:
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
        ReviewResult,
        TaskStep,
        WorkerFailed,
        WorkerSucceeded,
    )
    from ..prompts import (
        _build_impl_plan_prompt,
        _build_phase_prompt,
        _build_plan_prompt,
        _build_review_prompt,
        _extract_prd_markers,
    )
    from ..signals import build_allowed_files
    from ..tasks import _impl_plan_path, _read_progress_human_blockers, _review_output_path
    from ..utils import _hash_json_data
    from ..validation import _extract_review_blocker_files
    from ..worker import _run_codex_worker
    from .load_plan import load_plan
    from .load_review import load_review
    from .validate_plan import validate_plan
    from .validate_review import validate_review
except ImportError:  # pragma: no cover
    from constants import (
        ERROR_TYPE_CODEX_EXIT,
        ERROR_TYPE_DISALLOWED_FILES,
        ERROR_TYPE_HEARTBEAT_TIMEOUT,
        ERROR_TYPE_SHIFT_TIMEOUT,
        IGNORED_REVIEW_PATH_PREFIXES,
    )
    from git_utils import (
        _diff_file_sets,
        _git_changed_files,
        _git_diff_stat,
        _git_diff_text,
        _git_status_porcelain,
        _snapshot_repo_changes,
        _validate_changes_for_mode,
    )
    from io_utils import _load_data, _read_log_tail, _read_text_for_prompt, _render_json_for_prompt, _save_data
    from models import (
        AllowlistViolation,
        NoIntroducedChanges,
        ProgressHumanBlockers,
        ReviewResult,
        TaskStep,
        WorkerFailed,
        WorkerSucceeded,
    )
    from prompts import (
        _build_impl_plan_prompt,
        _build_phase_prompt,
        _build_plan_prompt,
        _build_review_prompt,
        _extract_prd_markers,
    )
    from signals import build_allowed_files
    from tasks import _impl_plan_path, _read_progress_human_blockers, _review_output_path
    from utils import _hash_json_data
    from validation import _extract_review_blocker_files
    from worker import _run_codex_worker
    from actions.load_plan import load_plan
    from actions.load_review import load_review
    from actions.validate_plan import validate_plan
    from actions.validate_review import validate_review


def _classify_worker_failure(
    *,
    run_result: dict[str, Any],
    stdout_tail: str,
    stderr_tail: str,
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
        error_detail = f"Codex CLI exited with code {run_result['exit_code']}"
        error_type = ERROR_TYPE_CODEX_EXIT
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
) -> Any:
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
    elif step == TaskStep.PLAN_IMPL:
        mode = "plan_impl"
        plan_path = _impl_plan_path(artifacts_dir, str(phase_id))
        phase_test_command = None
        if phase:
            phase_test_command = phase.get("test_command") or task.get("test_command") or test_command
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
    elif step == TaskStep.REVIEW:
        mode = "review"
        if not phase:
            raise ValueError("Phase not found for review")
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
        prompt = _build_review_prompt(
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
        if not phase:
            raise ValueError("Phase not found for implement")
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

    run_result = _run_codex_worker(
        command=codex_command,
        prompt=prompt,
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=shift_minutes * 60,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_grace_seconds=heartbeat_grace_seconds,
        progress_path=progress_path,
        expected_run_id=run_id,
        on_spawn=on_spawn,
    )

    stdout_tail = _read_log_tail(Path(run_result["stdout_path"]))
    stderr_tail = _read_log_tail(Path(run_result["stderr_path"]))

    human_blocking, human_next_steps = _read_progress_human_blockers(
        progress_path,
        expected_run_id=run_id,
    )
    if human_blocking:
        return ProgressHumanBlockers(run_id=run_id, issues=human_blocking, next_steps=human_next_steps)

    failure, error_type, error_detail = _classify_worker_failure(
        run_result=run_result,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
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
        "start_time": run_result["start_time"],
        "end_time": run_result["end_time"],
        "exit_code": run_result["exit_code"],
        "changed_files": post_run_changed,
        "disallowed_files": disallowed_files,
        "pre_run_snapshot": pre_run_changed,
        "post_run_snapshot": post_run_changed,
        "introduced_changes": introduced,
        "removed_changes": removed,
    }
    _save_data(run_dir / "manifest.json", manifest)

    if not ok and mode == "implement":
        return AllowlistViolation(
            run_id=run_id,
            step=step,
            disallowed_paths=disallowed_files,
            changed_files=post_run_changed,
            introduced_changes=introduced,
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
            timed_out=bool(run_result.get("timed_out")),
            no_heartbeat=bool(run_result.get("no_heartbeat")),
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
        plan_valid, plan_issue = validate_plan(
            plan_data,
            phase or {"id": phase_id, "acceptance_criteria": []},
            prd_markers=prd_markers,
            prd_truncated=prd_truncated,
            prd_has_content=bool(prd_text.strip()),
            expected_test_command=phase_test_command,
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
        )
        if not valid_review:
            return ReviewResult(
                run_id=run_id,
                valid=False,
                blocking_severities_present=False,
                issues=[],
                files=[],
                review_path=str(review_path),
                review_issue=review_issue,
            )
        issues = review_data.get("issues") or []
        blocking = [
            it
            for it in issues
            if isinstance(it, dict)
            and str(it.get("severity", "")).strip().lower() in {"critical", "high"}
        ]
        blocker_files = _extract_review_blocker_files(review_data)
        return ReviewResult(
            run_id=run_id,
            valid=True,
            blocking_severities_present=bool(blocking),
            issues=blocking,
            files=blocker_files,
            review_path=str(review_path),
            review_issue=None,
        )

    if step == TaskStep.IMPLEMENT:
        if not introduced:
            return NoIntroducedChanges(
                run_id=run_id,
                step=step,
                repo_dirty=repo_dirty,
                changed_files=post_run_changed,
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
