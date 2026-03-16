"""Post-merge integration health checker.

Runs verification commands on the base branch after task merges to detect
cross-task regressions.  When failures are found the checker marks the
integration as *degraded* and optionally auto-generates a fix task.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..domain.models import Task, now_iso
from .live_worker_adapter import (
    _DEFAULT_PROJECT_COMMANDS,
    _apply_venv_to_defaults,
    _is_subpath,
    _resolve_command_paths,
    detect_project_languages,
)
from .venv_detector import detect_python_venv

if TYPE_CHECKING:
    from .service import OrchestratorService

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 10_000  # truncate captured stdout/stderr


@dataclass
class HealthCheckResult:
    """Outcome of a single post-merge health check."""

    passed: bool
    trigger_task_id: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


def _truncate(text: str, limit: int = _MAX_OUTPUT_BYTES) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


class IntegrationHealthChecker:
    """Run verification commands after merge and track integration health."""

    def __init__(self, service: OrchestratorService) -> None:
        self._service = service
        self._status: str = "healthy"  # "healthy" | "degraded"
        self._last_check_at: str | None = None
        self._last_check_task_id: str | None = None
        self._merge_count_since_check: int = 0
        self._failure_summary: str | None = None
        self._fix_task_id: str | None = None

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def load_state(self, state: dict[str, Any]) -> None:
        """Hydrate health state from persisted orchestrator state."""
        raw = state.get("integration_health")
        if not isinstance(raw, dict):
            return
        self._status = str(raw.get("status") or "healthy")
        if self._status not in ("healthy", "degraded"):
            self._status = "healthy"
        self._last_check_at = str(raw.get("last_check_at") or "").strip() or None
        self._last_check_task_id = str(raw.get("last_check_task_id") or "").strip() or None
        self._merge_count_since_check = max(0, int(raw.get("merge_count_since_check") or 0))
        self._failure_summary = str(raw.get("failure_summary") or "").strip() or None
        self._fix_task_id = str(raw.get("fix_task_id") or "").strip() or None

    def persist_state(self) -> dict[str, Any]:
        """Return health state dict suitable for embedding in orchestrator state."""
        return {
            "status": self._status,
            "last_check_at": self._last_check_at,
            "last_check_task_id": self._last_check_task_id,
            "merge_count_since_check": self._merge_count_since_check,
            "failure_summary": self._failure_summary,
            "fix_task_id": self._fix_task_id,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_merge(self) -> None:
        """Increment merge counter — called after each successful merge."""
        self._merge_count_since_check += 1

    def is_degraded(self) -> bool:
        """Return whether integration is currently degraded."""
        return self._status == "degraded"

    def get_state(self) -> dict[str, Any]:
        """Return current health state dict for API exposure."""
        return self.persist_state()

    def clear_degraded(self) -> None:
        """Reset health status back to healthy."""
        self._status = "healthy"
        self._failure_summary = None
        self._fix_task_id = None
        self._service.bus.emit(
            channel="tasks",
            event_type="integration_health.degraded_cleared",
            entity_id="integration_health",
            payload={"cleared_at": now_iso()},
        )

    def should_run(self) -> bool:
        """Decide whether a check should execute based on config and merge count."""
        cfg = self._get_config()
        mode = cfg.get("mode", "off")
        if mode == "off":
            return False
        if mode == "always":
            return True
        if mode == "periodic":
            interval = max(1, int(cfg.get("periodic_interval", 5) or 5))
            return self._merge_count_since_check >= interval
        return False

    def run_check(self, trigger_task_id: str, *, force: bool = False) -> HealthCheckResult | None:
        """Run verification commands on the base branch and update health state.

        Returns ``None`` when the check is skipped (mode is off or periodic
        interval not yet reached).  Pass ``force=True`` to bypass the
        mode/periodic gate (used by the post-merge hard gate).
        """
        if not force and not self.should_run():
            return None

        svc = self._service
        cfg = self._get_config()
        timeout = max(30, int(cfg.get("timeout_seconds", 300) or 300))

        test_command = self._resolve_test_command()
        if not test_command:
            logger.debug("integration_health: no test command resolved, skipping check")
            return None

        project_dir = svc.container.project_dir
        start = time.monotonic()
        try:
            proc = subprocess.run(
                test_command,
                shell=True,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exit_code = proc.returncode
            stdout = _truncate(proc.stdout or "")
            stderr = _truncate(proc.stderr or "")
        except subprocess.TimeoutExpired:
            exit_code = -1
            stdout = ""
            stderr = f"Test command timed out after {timeout}s"
        except Exception as exc:
            exit_code = -1
            stdout = ""
            stderr = f"Unexpected error running test command: {exc}"

        duration = time.monotonic() - start
        passed = exit_code == 0

        result = HealthCheckResult(
            passed=passed,
            trigger_task_id=trigger_task_id,
            command=test_command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(duration, 2),
        )

        self._last_check_at = now_iso()
        self._last_check_task_id = trigger_task_id
        self._merge_count_since_check = 0

        if passed:
            self._status = "healthy"
            self._failure_summary = None
            svc.bus.emit(
                channel="tasks",
                event_type="integration_health.check_passed",
                entity_id="integration_health",
                payload={
                    "trigger_task_id": trigger_task_id,
                    "command": test_command,
                    "duration_seconds": result.duration_seconds,
                },
            )
        else:
            self._status = "degraded"
            self._failure_summary = (stdout or stderr)[:2000]
            svc.bus.emit(
                channel="tasks",
                event_type="integration_health.check_failed",
                entity_id="integration_health",
                payload={
                    "trigger_task_id": trigger_task_id,
                    "command": test_command,
                    "exit_code": exit_code,
                    "duration_seconds": result.duration_seconds,
                    "failure_summary": self._failure_summary,
                },
            )
            if cfg.get("auto_fix_task", True):
                self._maybe_create_fix_task(result)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_config(self) -> dict[str, Any]:
        cfg = self._service.container.config.load()
        orchestrator_cfg = cfg.get("orchestrator") or {}
        raw = orchestrator_cfg.get("integration_health")
        return dict(raw) if isinstance(raw, dict) else {}

    def _resolve_test_command(self) -> str | None:
        """Build a test command string from project config or language defaults."""
        svc = self._service
        cfg = svc.container.config.load()

        # Check for explicit project-level commands first
        project_cfg = cfg.get("project") or {}
        raw_commands: dict[str, Any] = project_cfg.get("commands") or {}
        project_commands: dict[str, dict[str, str]] | None = {
            lang: cmds for lang, cmds in raw_commands.items()
            if isinstance(cmds, dict)
        } or None

        # Fall back to language-detected defaults
        project_dir = svc.container.project_dir
        venv_info = detect_python_venv(project_dir)
        langs = detect_project_languages(project_dir)
        if langs:
            if project_commands is None:
                project_commands = {}
            for lang in langs:
                if lang not in project_commands:
                    defaults = _DEFAULT_PROJECT_COMMANDS.get(lang)
                    if defaults:
                        lang_cmds = dict(defaults)
                        if lang == "python" and venv_info is not None:
                            rel_prefix = str(venv_info.path.relative_to(project_dir) / "bin") if _is_subpath(venv_info.path, project_dir) else str(venv_info.bin_dir)
                            lang_cmds = _apply_venv_to_defaults(lang_cmds, venv_info.bin_dir, rel_prefix)
                        project_commands[lang] = lang_cmds

        if not project_commands:
            return None

        project_commands = _resolve_command_paths(project_commands, svc.container.project_dir)

        # Pick the first available test command across detected languages
        for lang in (langs or list(project_commands.keys())):
            cmds = project_commands.get(lang)
            if isinstance(cmds, dict):
                test_cmd = cmds.get("test")
                if test_cmd:
                    return test_cmd

        return None

    def _maybe_create_fix_task(self, result: HealthCheckResult) -> None:
        """Auto-generate an integration fix task if one doesn't already exist."""
        svc = self._service
        # Deduplicate: skip if an open fix task already exists
        tasks = svc.container.tasks.list()
        for t in tasks:
            if t.status in ("done", "cancelled"):
                continue
            meta = t.metadata if isinstance(t.metadata, dict) else {}
            if meta.get("generated_from") == "integration_health_check":
                logger.debug(
                    "integration_health: skipping fix task creation, "
                    "open fix task %s already exists",
                    t.id,
                )
                self._fix_task_id = t.id
                return

        fix_task = Task(
            title="Fix integration regressions on base branch",
            description=(
                f"Post-merge health check failed after task {result.trigger_task_id}.\n\n"
                f"Command: {result.command}\n"
                f"Exit code: {result.exit_code}\n"
                f"Failures:\n{result.stdout[-5000:]}"
            ),
            task_type="chore",
            priority="P0",
            status="queued",
            source="generated",
            metadata={
                "generated_from": "integration_health_check",
                "trigger_task_id": result.trigger_task_id,
                "failure_snapshot": (result.stdout or result.stderr)[-2000:],
            },
        )
        svc.container.tasks.upsert(fix_task)
        self._fix_task_id = fix_task.id
        svc.bus.emit(
            channel="tasks",
            event_type="task.generated_from_pipeline",
            entity_id=fix_task.id,
            payload={
                "created_task_ids": [fix_task.id],
                "reason_code": "integration_health_check",
            },
        )
        logger.info(
            "integration_health: created fix task %s after failure triggered by %s",
            fix_task.id,
            result.trigger_task_id,
        )
