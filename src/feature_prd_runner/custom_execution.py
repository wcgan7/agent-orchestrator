"""Execute custom prompts and ad-hoc tasks with flexible controls.

This module provides the core functionality for executing custom prompts
outside the normal PLAN->IMPLEMENT->VERIFY->REVIEW->COMMIT cycle.
Supports AGENTS.md override for "superadmin" mode.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from .actions.run_worker import run_resume_prompt_action
from .constants import LOCK_FILE, STATE_DIR_NAME
from .io_utils import FileLock, _append_event, _load_data, _save_data, _update_progress
from .models import ProgressHumanBlockers, ResumePromptResult, TaskStep, WorkerFailed
from .state import _ensure_state_files, _finalize_run_state
from .utils import _now_iso


def _build_custom_prompt(
    user_prompt: str,
    progress_path: Path,
    run_id: str,
    heartbeat_seconds: Optional[int] = None,
    override_agents: bool = False,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """Build a custom prompt with optional AGENTS.md override.

    Args:
        user_prompt: The user's instructions to execute.
        progress_path: Path where progress should be written.
        run_id: Unique run identifier.
        heartbeat_seconds: Heartbeat interval for monitoring.
        override_agents: If True, user can bypass AGENTS.md rules.
        context: Optional context dict with additional info.

    Returns:
        The formatted prompt string to send to the worker.
    """
    heartbeat_block = ""
    if heartbeat_seconds:
        heartbeat_block = f"  Update heartbeat at least every {heartbeat_seconds} seconds.\n"

    # Build context block if provided
    context_block = ""
    if context:
        context_items = []
        if context.get("task_id"):
            context_items.append(f"  Task: {context['task_id']}")
        if context.get("phase_id"):
            context_items.append(f"  Phase: {context['phase_id']}")
        if context.get("files"):
            files_list = context["files"][:10]
            files_str = ", ".join(files_list)
            if len(context["files"]) > 10:
                files_str += f", ... ({len(context['files'])} total)"
            context_items.append(f"  Focus files: {files_str}")
        if context_items:
            context_block = "\n\nContext:\n" + "\n".join(context_items) + "\n"

    # AGENTS.md override notice
    agents_block = ""
    if override_agents:
        agents_block = """
IMPORTANT - SUPERADMIN MODE:
You are operating in SUPERADMIN mode. You have special privileges:
- You may bypass normal AGENTS.md rules if necessary to complete the task
- You may modify any files needed (not restricted by allowlists)
- You may skip documentation/testing requirements if time-sensitive
- However, still follow best practices where reasonable
- Explain any rule bypasses in your progress report

"""
    else:
        agents_block = """
Follow all repository rules in AGENTS.md.

"""

    prompt = f"""You have been given the following custom instructions to complete:

{user_prompt}
{context_block}
{agents_block}
Follow these instructions carefully. When you are done:
1. If you successfully completed the instructions, write a progress file indicating success.
2. If you cannot complete the instructions (e.g., need clarification, blocked by an issue),
   write a progress file with human_blocking_issues explaining what is blocking you.

Progress contract (REQUIRED):
- Write snapshot to: {progress_path}
  Required fields: run_id={run_id}, task_id="custom_exec", phase="custom_exec",
  actions (list of actions you took), claims (what you accomplished),
  next_steps, human_blocking_issues (if blocked), human_next_steps, heartbeat.
{heartbeat_block}
IMPORTANT: If you successfully complete the instructions, human_blocking_issues MUST be empty.
If you are blocked and cannot complete the instructions, human_blocking_issues MUST contain the blocking reason(s).
"""
    return prompt


def execute_custom_prompt(
    user_prompt: str,
    project_dir: Path,
    codex_command: str = "codex exec -",
    worker: Optional[str] = None,
    heartbeat_seconds: int = 120,
    heartbeat_grace_seconds: int = 300,
    shift_minutes: int = 45,
    override_agents: bool = False,
    context: Optional[dict[str, Any]] = None,
    then_continue: bool = False,
) -> tuple[bool, Optional[str]]:
    """Execute a custom prompt as a standalone action.

    Args:
        user_prompt: The user's instructions.
        project_dir: Project root directory.
        codex_command: Command to invoke the worker.
        heartbeat_seconds: Heartbeat interval.
        heartbeat_grace_seconds: Grace period for heartbeat timeout.
        shift_minutes: Timeout for the execution.
        override_agents: Enable superadmin mode (bypass AGENTS.md).
        context: Optional execution context.
        then_continue: If True, return to normal cycle after completion.

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    project_dir = project_dir.resolve()

    # Ensure state directory exists
    state_dir = project_dir / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)

    lock_path = state_dir / LOCK_FILE
    paths = _ensure_state_files(project_dir, project_dir / "dummy_prd.md")

    # Generate run ID
    custom_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    custom_run_id = f"{custom_run_id}-{uuid.uuid4().hex[:8]}-custom"
    custom_run_dir = paths["runs"] / custom_run_id
    custom_run_dir.mkdir(parents=True, exist_ok=True)
    custom_progress_path = custom_run_dir / "progress.json"

    logger.info("=" * 70)
    logger.info("EXECUTING CUSTOM PROMPT")
    if override_agents:
        logger.warning("SUPERADMIN MODE: AGENTS.md rules may be bypassed")
    logger.info("=" * 70)
    logger.info("Run ID: {}", custom_run_id)
    logger.info("Prompt: {}", user_prompt[:200] + ("..." if len(user_prompt) > 200 else ""))

    # Build custom prompt
    prompt_text = _build_custom_prompt(
        user_prompt=user_prompt,
        progress_path=custom_progress_path,
        run_id=custom_run_id,
        heartbeat_seconds=heartbeat_seconds,
        override_agents=override_agents,
        context=context,
    )

    # Write prompt to file for debugging
    prompt_file = custom_run_dir / "prompt.txt"
    prompt_file.write_text(prompt_text)

    # Initialize progress
    _update_progress(
        custom_progress_path,
        {
            "run_id": custom_run_id,
            "task_id": "custom_exec",
            "phase": "custom_exec",
            "human_blocking_issues": [],
            "human_next_steps": [],
        },
    )

    # Update run state
    with FileLock(lock_path):
        run_state = _load_data(paths["run_state"], {})
        run_state.update(
            {
                "status": "running",
                "current_task_id": "custom_exec",
                "current_phase_id": None,
                "run_id": custom_run_id,
                "last_run_id": custom_run_id,
                "updated_at": _now_iso(),
                "coordinator_pid": os.getpid(),
                "worker_pid": None,
                "coordinator_started_at": _now_iso(),
                "last_heartbeat": _now_iso(),
            }
        )
        _save_data(paths["run_state"], run_state)

    def _on_worker_spawn(pid: int) -> None:
        try:
            with FileLock(lock_path):
                rs = _load_data(paths["run_state"], {})
                if rs.get("status") == "running" and rs.get("run_id") == custom_run_id:
                    rs["worker_pid"] = pid
                    rs["last_heartbeat"] = _now_iso()
                    rs["updated_at"] = _now_iso()
                    _save_data(paths["run_state"], rs)
        except Exception:
            pass

    # Resolve workers config for the target project and run via resume prompt action.
    from .config import load_runner_config
    from .workers import get_workers_runtime_config, resolve_worker_for_step

    cfg, _cfg_err = load_runner_config(project_dir)
    workers_runtime = get_workers_runtime_config(
        config=cfg,
        codex_command_fallback=codex_command,
        cli_worker=worker,
    )
    spec = resolve_worker_for_step(workers_runtime, TaskStep.RESUME_PROMPT)

    # Codex providers receive the full progress-contract prompt; non-agentic providers
    # receive the raw instructions and return a patch-based JSON result.
    effective_prompt = prompt_text if spec.type == "codex" else user_prompt
    if spec.type != "codex" and override_agents:
        effective_prompt = (
            "IMPORTANT - SUPERADMIN MODE:\n"
            "- You may bypass normal AGENTS.md rules if necessary to complete the task\n"
            "- You may modify any files needed\n\n"
            + effective_prompt
        )

    custom_event = run_resume_prompt_action(
        user_prompt=effective_prompt,
        project_dir=project_dir,
        run_dir=custom_run_dir,
        run_id=custom_run_id,
        codex_command=codex_command,
        progress_path=custom_progress_path,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_grace_seconds=heartbeat_grace_seconds,
        shift_minutes=shift_minutes,
        on_spawn=_on_worker_spawn,
        workers_runtime=workers_runtime,
    )

    _append_event(paths["events"], custom_event.to_dict())

    # Check result
    success = False
    error_message = None

    if isinstance(custom_event, ResumePromptResult) and custom_event.succeeded:
        logger.info("✓ Custom prompt completed successfully")
        success = True
        _finalize_run_state(paths, lock_path, status="idle", last_error=None)
    elif isinstance(custom_event, ProgressHumanBlockers):
        logger.error("✗ Custom prompt blocked: {}", "; ".join(custom_event.issues))
        error_message = "; ".join(custom_event.issues)
        _finalize_run_state(
            paths,
            lock_path,
            status="blocked",
            last_error=error_message,
        )
    elif isinstance(custom_event, WorkerFailed):
        logger.error("✗ Custom prompt failed: {}", custom_event.error_detail)
        error_message = custom_event.error_detail
        _finalize_run_state(
            paths,
            lock_path,
            status="idle",
            last_error=error_message,
        )
    else:
        logger.error("✗ Custom prompt failed with unexpected event: {}", type(custom_event).__name__)
        error_message = f"Unexpected event: {type(custom_event).__name__}"
        _finalize_run_state(paths, lock_path, status="idle", last_error=error_message)

    if then_continue:
        logger.info("Continuing to normal implementation cycle...")

    return success, error_message
