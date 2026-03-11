"""Core orchestration service for task lifecycle execution."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Optional, cast

from ...collaboration.modes import normalize_hitl_mode, should_gate
from ...pipelines.registry import PipelineRegistry
from ...workers.config import get_workers_runtime_config, resolve_worker_for_step
from ..domain.models import (
    PlanRefineJob,
    PlanRevision,
    PlanRevisionStatus,
    Priority,
    ReviewCycle,
    ReviewFinding,
    RunRecord,
    Task,
    now_iso,
)
from ..domain.scope_contract import normalize_scope_contract
from ..events.bus import EventBus
from ..storage.container import Container
from ...worker import WorkerCancelledError
from .dependency_manager import DependencyManager
from .environment_preflight import workers_environment_config
from .integration_health import IntegrationHealthChecker
from .reconciler import OrchestratorReconciler
from .live_worker_adapter import _VERIFY_STEPS
from .plan_manager import PlanManager
from .task_executor import TaskExecutor
from .workdoc_manager import WorkdocManager
from .worktree_manager import MergeOutcome, WorktreeManager
from .worker_adapter import DefaultWorkerAdapter, StepResult, WorkerAdapter

logger = logging.getLogger(__name__)


class OrchestratorService:
    """Coordinate task scheduling, execution, review, and commit flow."""

    # Compatibility shim for tests and callers that referenced the old class constant.
    _GENERIC_WORKDOC_TEMPLATE = WorkdocManager._GENERIC_WORKDOC_TEMPLATE

    _GATE_MAPPING: dict[str, str] = {
        "plan": "before_plan",
        "implement": "before_implement",
        "generate_tasks": "before_generate_tasks",
        "commit": "before_commit",
    }
    _BEFORE_DONE_RESUME_STEP = "__before_done__"
    _DECOMPOSITION_PIPELINES = {"plan_only", "repo_review", "security_audit"}
    _GATE_RESUME_STEP: dict[str, str] = {
        "before_plan": "plan",
        "before_implement": "implement",
        "before_generate_tasks": "generate_tasks",
        "before_done": _BEFORE_DONE_RESUME_STEP,
        "after_implement": "review",
        "before_commit": "commit",
    }
    _HUMAN_INTERVENTION_GATE = "human_intervention"
    _WAIT_KIND_APPROVAL = "approval_wait"
    _WAIT_KIND_INTERVENTION = "intervention_wait"
    _WAIT_KIND_AUTO_RECOVERY = "auto_recovery_wait"
    _SKIP_TO_PRECOMMIT_COMPATIBLE_BLOCKED_STEPS = {"verify", "benchmark"}
    _TASK_GENERATION_STATUS_VALUES = {"backlog", "queued"}
    _TASK_GENERATION_HITL_VALUES = {"inherit_parent", "autopilot", "supervised", "review_only"}
    _TASK_GENERATION_DEFAULTS_KEY = "task_generation_defaults"
    _TASK_GENERATION_OVERRIDE_KEY = "task_generation_override"
    _TASK_GENERATION_DEFAULTS = {
        "child_status": "backlog",
        "child_hitl_mode": "inherit_parent",
        "infer_deps": True,
    }
    _CANCELLED_CONTEXT_METADATA_KEYS = (
        "worktree_dir",
        "task_context",
        "preserved_branch",
        "preserved_base_branch",
        "preserved_base_sha",
        "preserved_head_sha",
        "preserved_merge_base_sha",
        "preserved_at",
        "review_context",
        "pending_precommit_approval",
        "review_stage",
        "cancel_cleanup_pending",
        "cancel_cleanup_deferred_at",
        "cancel_cleanup_reason",
        _TASK_GENERATION_OVERRIDE_KEY,
    )
    _ENVIRONMENT_FAILURE_SUMMARY_PATTERNS = (
        re.compile(r"environment preflight failed", re.IGNORECASE),
        re.compile(r"docker daemon/socket is unavailable", re.IGNORECASE),
        re.compile(r"node dependencies are missing", re.IGNORECASE),
        re.compile(r"prisma cli is missing", re.IGNORECASE),
        re.compile(r"network dns probe failed", re.IGNORECASE),
    )
    _ENVIRONMENT_RETRY_BASE_SECONDS = 15
    _ENVIRONMENT_RETRY_MAX_SECONDS = 300

    def __init__(
        self,
        container: Container,
        bus: EventBus,
        *,
        worker_adapter: WorkerAdapter | None = None,
    ) -> None:
        """Initialize the OrchestratorService.

        Args:
            container (Container): Container for this call.
            bus (EventBus): Bus for this call.
            worker_adapter (WorkerAdapter | None): Worker adapter for this call.
        """
        self.container = container
        self.bus = bus
        self.worker_adapter = worker_adapter or DefaultWorkerAdapter()
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._drain = False
        self._run_branch: Optional[str] = None
        self._pool: ThreadPoolExecutor | None = None
        self._pool_size: int = 0
        self._futures: dict[str, Future[Any]] = {}
        self._futures_lock = threading.Lock()
        self._merge_lock = threading.Lock()
        self._branch_lock = threading.Lock()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        self._last_tick_at: str | None = None
        self._last_dispatch_at: str | None = None
        self._last_tick_error: str | None = None
        self._consecutive_tick_failures: int = 0
        self._dispatch_blocked_reason: str | None = None
        self._last_reconcile_at: str | None = None
        self._last_reconcile_repairs: int = 0
        self._last_reconcile_monotonic: float = 0.0
        self._last_state_persist_monotonic: float = 0.0
        self._manual_run_active: int = 0
        self._reset_inflight: bool = False
        self._dependency_manager = DependencyManager(
            container,
            bus,
            worker_adapter_getter=lambda: self.worker_adapter,
        )
        self._workdoc_manager = WorkdocManager(
            container,
            bus,
            pipeline_id_resolver=self._pipeline_id_for_task,
        )
        self._plan_manager = PlanManager(self)
        self._worktree_manager = WorktreeManager(self)
        self._task_executor = TaskExecutor(self)
        self._reconciler = OrchestratorReconciler(self)
        self._integration_health = IntegrationHealthChecker(self)
        self._load_runtime_state()

    def _load_runtime_state(self) -> None:
        """Hydrate scheduler/reconciler heartbeat state from persisted storage."""
        state = self.container.db.load_orchestrator_state()
        if not state:
            # Backward compatibility: migrate legacy config blob into dedicated table once.
            cfg = self.container.config.load()
            raw = cfg.get("orchestrator_state")
            if isinstance(raw, dict):
                state = dict(raw)
                self.container.db.save_orchestrator_state(state)
                updater = getattr(self.container.config, "update", None)
                if callable(updater):
                    updater(lambda current: {k: v for k, v in current.items() if k != "orchestrator_state"})
                else:
                    cfg = dict(cfg)
                    cfg.pop("orchestrator_state", None)
                    self.container.config.save(cfg)
        last_tick = str(state.get("last_tick_at") or "").strip()
        last_dispatch = str(state.get("last_dispatch_at") or "").strip()
        last_error = str(state.get("last_tick_error") or "").strip()
        last_reconcile = str(state.get("last_reconcile_at") or "").strip()
        self._last_tick_at = last_tick or None
        self._last_dispatch_at = last_dispatch or None
        self._last_tick_error = last_error or None
        self._consecutive_tick_failures = self._coerce_nonnegative_int(
            state.get("consecutive_tick_failures"), 0, maximum=1_000_000
        )
        self._dispatch_blocked_reason = str(state.get("dispatch_blocked_reason") or "").strip() or None
        self._last_reconcile_at = last_reconcile or None
        self._last_reconcile_repairs = self._coerce_nonnegative_int(
            state.get("last_reconcile_repairs"), 0, maximum=1_000_000
        )
        self._integration_health.load_state(state)

    def _persist_runtime_state(self, *, force: bool = False) -> None:
        """Persist scheduler/reconciler heartbeat state to dedicated SQLite table."""
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_state_persist_monotonic) < 10.0:
            return
        state_payload = {
            "last_tick_at": self._last_tick_at,
            "last_dispatch_at": self._last_dispatch_at,
            "consecutive_tick_failures": int(self._consecutive_tick_failures),
            "last_tick_error": self._last_tick_error,
            "dispatch_blocked_reason": self._dispatch_blocked_reason,
            "last_reconcile_at": self._last_reconcile_at,
            "last_reconcile_repairs": int(self._last_reconcile_repairs),
            "integration_health": self._integration_health.persist_state(),
        }
        self.container.db.save_orchestrator_state(state_payload)
        self._last_state_persist_monotonic = now_mono

    @staticmethod
    def _execution_checkpoint(task: Task) -> dict[str, Any]:
        """Return the execution checkpoint dict from task metadata.

        Args:
            task: Task whose checkpoint to retrieve.

        Returns:
            Checkpoint dict, or an empty dict if none is stored.
        """
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        raw = task.metadata.get("execution_checkpoint")
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _save_execution_checkpoint(task: Task, checkpoint: dict[str, Any]) -> None:
        """Persist or clear the execution checkpoint in task metadata.

        Args:
            task: Task to update.
            checkpoint: Checkpoint dict to store, or empty dict to clear.
        """
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        if checkpoint:
            task.metadata["execution_checkpoint"] = checkpoint
        else:
            task.metadata.pop("execution_checkpoint", None)

    def _is_resume_requested(self, task: Task) -> bool:
        """Return whether an in-progress task has a pending resume request.

        Args:
            task: Task to check.
        """
        if task.status != "in_progress" or task.pending_gate:
            return False
        checkpoint = self._execution_checkpoint(task)
        return bool(str(checkpoint.get("resume_requested_at") or "").strip())

    def _clear_resume_request(self, task: Task) -> None:
        """Remove the resume_requested_at field from the task checkpoint.

        Args:
            task: Task whose resume request to clear.
        """
        checkpoint = self._execution_checkpoint(task)
        if not checkpoint:
            return
        checkpoint.pop("resume_requested_at", None)
        self._save_execution_checkpoint(task, checkpoint)

    def _consume_gate_resume_approval(self, task: Task, gate_name: str) -> bool:
        """Consume a gate approval if it matches, clearing the wait state.

        Args:
            task: Task with a pending gate.
            gate_name: Gate name that must match the approved checkpoint.

        Returns:
            ``True`` when the approval was consumed, ``False`` otherwise.
        """
        checkpoint = self._execution_checkpoint(task)
        approved_gate = str(checkpoint.get("approved_gate") or "").strip()
        if approved_gate != gate_name:
            return False
        checkpoint.pop("approved_gate", None)
        checkpoint.pop("resume_requested_at", None)
        checkpoint["resumed_at"] = now_iso()
        self._save_execution_checkpoint(task, checkpoint)
        self._clear_wait_state(task)
        self.container.tasks.upsert(task)
        self.bus.emit(
            channel="tasks",
            event_type="task.gate_resumed",
            entity_id=task.id,
            payload={"gate": gate_name},
        )
        return True

    def _mark_gate_waiting(self, task: Task, gate_name: str, *, resume_step: str | None = None) -> None:
        """Pause the task at a human gate and record checkpoint state.

        Args:
            task: Task to pause.
            gate_name: Identifier for the gate (e.g. ``"before_implement"``).
            resume_step: Pipeline step to resume from after approval.
        """
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        resolved_resume_step = resume_step if resume_step is not None else self._GATE_RESUME_STEP.get(gate_name, task.current_step or "")
        checkpoint = self._execution_checkpoint(task)
        checkpoint.update(
            {
                "gate": gate_name,
                "resume_step": resolved_resume_step or None,
                "run_id": task.run_ids[-1] if task.run_ids else None,
                "paused_at": now_iso(),
                "resume_requested_at": None,
                "approved_gate": None,
            }
        )
        if resolved_resume_step:
            task.metadata["retry_from_step"] = resolved_resume_step
        else:
            task.metadata.pop("retry_from_step", None)
        self._save_execution_checkpoint(task, checkpoint)
        self._set_wait_state(
            task,
            kind=self._WAIT_KIND_INTERVENTION if gate_name == self._HUMAN_INTERVENTION_GATE else self._WAIT_KIND_APPROVAL,
            step=resolved_resume_step or task.current_step,
            reason_code=gate_name,
            recoverable=False,
        )

    def _set_wait_state(
        self,
        task: Task,
        *,
        kind: str,
        step: str | None = None,
        reason_code: str | None = None,
        recoverable: bool | None = None,
        attempt: int | None = None,
        max_attempts: int | None = None,
        next_retry_at: str | None = None,
    ) -> None:
        """Attach structured wait-state metadata to a task.

        Args:
            task: Task to annotate.
            kind: Wait category (e.g. ``"approval"``, ``"intervention"``).
            step: Pipeline step the task is waiting at.
            reason_code: Machine-readable reason identifier.
            recoverable: Whether the wait can auto-resolve.
            attempt: Current retry attempt number.
            max_attempts: Maximum allowed retries.
            next_retry_at: ISO timestamp for the next retry window.
        """
        wait_state: dict[str, Any] = {"kind": str(kind).strip() or "none"}
        if step:
            wait_state["step"] = str(step).strip()
        if reason_code:
            wait_state["reason_code"] = str(reason_code).strip()
        if recoverable is not None:
            wait_state["recoverable"] = bool(recoverable)
        if attempt is not None:
            wait_state["attempt"] = int(attempt)
        if max_attempts is not None:
            wait_state["max_attempts"] = int(max_attempts)
        if next_retry_at:
            wait_state["next_retry_at"] = str(next_retry_at).strip()
        wait_state["updated_at"] = now_iso()
        task.wait_state = wait_state

    @staticmethod
    def _clear_wait_state(task: Task) -> None:
        """Remove the wait-state annotation from a task.

        Args:
            task: Task whose wait state to clear.
        """
        task.wait_state = None

    def _gate_for_step(
        self,
        *,
        task: Task,
        mode: str,
        steps: list[str],
        step: str,
        skip_before_implement_gate: bool = False,
    ) -> str | None:
        """Resolve whether a specific step should pause for a human gate."""
        normalized_mode = normalize_hitl_mode(mode)
        if normalized_mode != "supervised":
            return None
        if step == "implement":
            if skip_before_implement_gate:
                return None
            if step not in steps:
                return None
            idx = steps.index(step)
            if idx <= 0:
                return None
            meaningful_pre_work = any(str(prev or "").strip() not in {"review", "commit"} for prev in steps[:idx])
            return "before_implement" if meaningful_pre_work else None
        if step == "generate_tasks":
            pipeline_id = self._pipeline_id_for_task(task)
            if pipeline_id in self._DECOMPOSITION_PIPELINES:
                return "before_generate_tasks"
        return None

    def _should_before_done_gate(self, *, task: Task, mode: str, has_commit: bool) -> bool:
        """Return whether a task should pause at a final done gate."""
        if has_commit:
            return False
        if not self._should_gate(mode, "before_done"):
            return False
        return self._pipeline_id_for_task(task) not in self._DECOMPOSITION_PIPELINES

    @staticmethod
    def _parse_iso_epoch(value: Any) -> float | None:
        """Parse an ISO-8601 timestamp string into a UTC epoch float.

        Args:
            value: Raw timestamp value (string or None).

        Returns:
            Epoch seconds as a float, or ``None`` on invalid input.
        """
        raw = str(value or "").strip()
        if not raw:
            return None
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    @staticmethod
    def _coerce_nonnegative_int(value: Any, default: int, *, maximum: int) -> int:
        """Coerce a value to a non-negative integer within bounds.

        Args:
            value: Raw value to parse.
            default: Fallback when parsing fails.
            maximum: Upper bound to clamp the result.
        """
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        if parsed < 0:
            return 0
        if parsed > maximum:
            return maximum
        return parsed

    def _reconcile_interval_seconds(self) -> int:
        """Return the configured reconciliation interval in seconds."""
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        return self._coerce_nonnegative_int(
            orchestrator_cfg.get("reconcile_interval_seconds"), 30, maximum=3600
        )

    def _tick_stale_seconds(self) -> int:
        """Return the tick staleness threshold in seconds."""
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        return self._coerce_nonnegative_int(orchestrator_cfg.get("tick_stale_seconds"), 15, maximum=3600)

    def _tick_failure_threshold(self) -> int:
        """Return the consecutive tick failure count before alerting."""
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        return max(1, self._coerce_nonnegative_int(orchestrator_cfg.get("tick_failure_threshold"), 5, maximum=1000))

    def _lease_ttl_seconds(self) -> int:
        """Return the execution lease time-to-live in seconds."""
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        return max(15, self._coerce_nonnegative_int(orchestrator_cfg.get("lease_ttl_seconds"), 120, maximum=86400))

    def _acquire_execution_lease(self, task: Task) -> None:
        """Create a new execution lease for a task being dispatched.

        Args:
            task: Task to lease.
        """
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata.pop("execution_lease", None)
        ttl = self._lease_ttl_seconds()
        now_ts = now_iso()
        expires_ts = datetime.now(timezone.utc).timestamp() + ttl
        expires_at = datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat()
        lease = {
            "owner": "orchestrator",
            "acquired_at": now_ts,
            "heartbeat_at": now_ts,
            "expires_at": expires_at,
            "ttl_seconds": ttl,
        }
        self.container.db.save_execution_lease(task.id, lease)

    def _heartbeat_execution_lease(self, task: Task) -> None:
        """Refresh the heartbeat and expiry of an existing execution lease.

        Args:
            task: Task whose lease to extend.
        """
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata.pop("execution_lease", None)
        raw = self.container.db.load_execution_lease(task.id)
        lease = dict(raw) if isinstance(raw, dict) else {}
        ttl = self._lease_ttl_seconds()
        now_ts = now_iso()
        expires_ts = datetime.now(timezone.utc).timestamp() + ttl
        lease["owner"] = lease.get("owner") or "orchestrator"
        lease["heartbeat_at"] = now_ts
        lease["expires_at"] = datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat()
        lease["ttl_seconds"] = ttl
        if not lease.get("acquired_at"):
            lease["acquired_at"] = now_ts
        self.container.db.save_execution_lease(task.id, lease)

    def _release_execution_lease(self, task: Task) -> bool:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        legacy_removed = bool(task.metadata.pop("execution_lease", None))
        lease_removed = self.container.db.delete_execution_lease(task.id)
        return legacy_removed or lease_removed

    def _execution_lease_active(self, task: Task, *, now_ts: float | None = None) -> bool:
        raw = self.container.db.load_execution_lease(task.id)
        if isinstance(raw, dict):
            expires_epoch = self._parse_iso_epoch(raw.get("expires_at"))
            if expires_epoch is not None:
                return expires_epoch > (time.time() if now_ts is None else now_ts)

        # Legacy fallback for upgraded tasks that still carry metadata lease.
        if not isinstance(task.metadata, dict):
            return False
        legacy = task.metadata.get("execution_lease")
        if not isinstance(legacy, dict):
            return False
        expires_epoch = self._parse_iso_epoch(legacy.get("expires_at"))
        if expires_epoch is None:
            return False
        if expires_epoch > (time.time() if now_ts is None else now_ts):
            self.container.db.save_execution_lease(task.id, dict(legacy))
            task.metadata.pop("execution_lease", None)
            return True
        return False

    def _task_has_active_future(self, task_id: str, *, active_future_task_ids: set[str] | None = None) -> bool:
        if active_future_task_ids is not None:
            return task_id in active_future_task_ids
        with self._futures_lock:
            future = self._futures.get(task_id)
            return bool(future and not future.done())

    @staticmethod
    def _strip_cancelled_gate_artifacts(task: Task) -> bool:
        changed = False
        if task.pending_gate:
            task.pending_gate = None
            changed = True
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        for key in (
            "pending_precommit_approval",
            "review_stage",
            "execution_checkpoint",
            "requested_changes",
        ):
            if key in task.metadata:
                task.metadata.pop(key, None)
                changed = True
        return changed

    def _cleanup_cancelled_task_context(
        self,
        task: Task,
        *,
        active_future_task_ids: set[str] | None = None,
        force: bool = False,
    ) -> dict[str, bool]:
        """Best-effort cancelled-task cleanup with strict active-task guards.

        Cleanup is deferred while a task still has active execution state
        (future in-flight or unexpired execution lease).
        """
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        metadata = task.metadata

        has_active_future = self._task_has_active_future(task.id, active_future_task_ids=active_future_task_ids)
        has_active_lease = self._execution_lease_active(task)
        deferred = bool((not force) and (has_active_future or has_active_lease))
        metadata_changed = False
        worktree_removed = False
        branch_deleted = False
        lease_released = False

        if deferred:
            if metadata.get("cancel_cleanup_pending") is not True:
                metadata["cancel_cleanup_pending"] = True
                metadata_changed = True
            deferred_reason: list[str] = []
            if has_active_future:
                deferred_reason.append("active_future")
            if has_active_lease:
                deferred_reason.append("active_lease")
            reason_text = ",".join(deferred_reason) if deferred_reason else "active_execution"
            if str(metadata.get("cancel_cleanup_reason") or "") != reason_text:
                metadata["cancel_cleanup_reason"] = reason_text
                metadata_changed = True
            if not str(metadata.get("cancel_cleanup_deferred_at") or "").strip():
                metadata["cancel_cleanup_deferred_at"] = now_iso()
                metadata_changed = True
            return {
                "deferred": True,
                "metadata_changed": metadata_changed,
                "worktree_removed": False,
                "branch_deleted": False,
                "lease_released": False,
            }

        expected_worktree = self.container.state_root / "worktrees" / str(task.id)
        if expected_worktree.exists():
            if expected_worktree.is_symlink():
                logger.warning(
                    "Skipping cancelled-task cleanup for symlinked worktree path: task=%s path=%s",
                    task.id,
                    expected_worktree,
                )
            else:
                subprocess.run(
                    ["git", "worktree", "remove", str(expected_worktree), "--force"],
                    cwd=self.container.project_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if expected_worktree.exists():
                    shutil.rmtree(expected_worktree, ignore_errors=True)
                worktree_removed = not expected_worktree.exists()

        branch_name = f"task-{task.id}"
        if (self.container.project_dir / ".git").exists():
            branch_existed = self._local_branch_exists(branch_name)
            if branch_existed:
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    cwd=self.container.project_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                branch_deleted = not self._local_branch_exists(branch_name)

        lease_released = self._release_execution_lease(task)
        for key in self._CANCELLED_CONTEXT_METADATA_KEYS:
            if key in metadata:
                metadata.pop(key, None)
                metadata_changed = True

        return {
            "deferred": False,
            "metadata_changed": metadata_changed,
            "worktree_removed": worktree_removed,
            "branch_deleted": branch_deleted,
            "lease_released": lease_released,
        }

    def cancel_task(self, task_id: str, *, source: str = "api_cancel") -> Task:
        """Cancel a task via one canonical lifecycle path.

        Active tasks defer worktree cleanup; inactive tasks are cleaned up
        immediately with task-owned path guards.
        """
        with self._lock:
            task = self.container.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            task.status = "cancelled"
            task.current_agent_id = None
            self._strip_cancelled_gate_artifacts(task)
            cleanup = self._cleanup_cancelled_task_context(task)
            task.updated_at = now_iso()
            self.container.tasks.upsert(task)

        self.bus.emit(
            channel="tasks",
            event_type="task.cancelled",
            entity_id=task.id,
            payload={"status": "cancelled", "cleanup_deferred": bool(cleanup.get("deferred")), "source": source},
        )
        return task

    def reconcile(self, *, source: Literal["startup", "automatic", "manual"] = "manual") -> dict[str, Any]:
        """Apply runtime invariants and emit deterministic repair events."""
        summary = self._reconciler.run_once(source=source)
        self._last_reconcile_at = now_iso()
        self._last_reconcile_repairs = int(summary.get("repairs") or 0)
        self._persist_runtime_state(force=True)
        return summary

    def _apply_gate_wait_policies(self, orchestrator_cfg: dict[str, Any]) -> None:
        reminder_minutes = self._coerce_nonnegative_int(
            orchestrator_cfg.get("gate_reminder_minutes"), 30, maximum=10080
        )
        stale_minutes = self._coerce_nonnegative_int(
            orchestrator_cfg.get("gate_stale_minutes"), 0, maximum=10080
        )
        max_wait_minutes = self._coerce_nonnegative_int(
            orchestrator_cfg.get("gate_max_wait_minutes"), 0, maximum=43200
        )
        timeout_action = str(orchestrator_cfg.get("gate_timeout_action") or "none").strip().lower()
        if timeout_action not in {"none", "block"}:
            timeout_action = "none"
        if reminder_minutes <= 0 and stale_minutes <= 0 and (max_wait_minutes <= 0 or timeout_action == "none"):
            return

        now_ts = time.time()
        for task in self.container.tasks.list():
            if task.status != "in_progress" or not task.pending_gate:
                continue
            checkpoint = self._execution_checkpoint(task)
            gate = str(task.pending_gate or "").strip()
            changed = False

            paused_epoch = self._parse_iso_epoch(checkpoint.get("paused_at"))
            if paused_epoch is None:
                checkpoint["paused_at"] = now_iso()
                paused_epoch = self._parse_iso_epoch(checkpoint.get("paused_at")) or now_ts
                changed = True
            elapsed_seconds = max(0.0, now_ts - paused_epoch)
            elapsed_minutes = int(elapsed_seconds // 60)

            if reminder_minutes > 0:
                last_reminder_epoch = self._parse_iso_epoch(checkpoint.get("last_reminder_at"))
                should_remind = last_reminder_epoch is None or (now_ts - last_reminder_epoch) >= reminder_minutes * 60
                if should_remind and elapsed_seconds >= reminder_minutes * 60:
                    checkpoint["last_reminder_at"] = now_iso()
                    changed = True
                    self.bus.emit(
                        channel="tasks",
                        event_type="task.gate_reminder",
                        entity_id=task.id,
                        payload={"gate": gate, "elapsed_minutes": elapsed_minutes},
                    )

            if stale_minutes > 0 and elapsed_seconds >= stale_minutes * 60 and not checkpoint.get("stale_emitted_at"):
                checkpoint["stale_emitted_at"] = now_iso()
                changed = True
                self.bus.emit(
                    channel="tasks",
                    event_type="task.gate_stale",
                    entity_id=task.id,
                    payload={"gate": gate, "elapsed_minutes": elapsed_minutes},
                )

            if timeout_action == "block" and max_wait_minutes > 0 and elapsed_seconds >= max_wait_minutes * 60:
                checkpoint["timed_out_at"] = now_iso()
                checkpoint["timed_out_gate"] = gate
                self._save_execution_checkpoint(task, checkpoint)
                task.status = "blocked"
                task.error = f"Gate '{gate}' exceeded max wait policy ({max_wait_minutes}m)"
                task.pending_gate = None
                task.current_agent_id = None
                task.updated_at = now_iso()
                self.container.tasks.upsert(task)
                for run_id in reversed(task.run_ids):
                    run = self.container.runs.get(run_id)
                    if run is None:
                        continue
                    if run.status in {"waiting_gate", "in_progress"}:
                        run.status = "blocked"
                        run.finished_at = task.updated_at
                        run.summary = task.error
                        self.container.runs.upsert(run)
                        break
                self.bus.emit(
                    channel="tasks",
                    event_type="task.blocked",
                    entity_id=task.id,
                    payload={"error": task.error, "gate": gate, "policy": "gate_max_wait"},
                )
                continue

            if changed:
                self._save_execution_checkpoint(task, checkpoint)
                task.updated_at = now_iso()
                self.container.tasks.upsert(task)

    def _get_pool(self, desired_workers: int | None = None) -> ThreadPoolExecutor:
        if desired_workers is None:
            cfg = self.container.config.load()
            desired_workers = int(dict(cfg.get("orchestrator") or {}).get("concurrency", 2) or 2)
        if self._pool is not None and self._pool_size == desired_workers:
            return self._pool
        if self._pool is not None and self._pool_size != desired_workers:
            # Let existing tasks finish; shutdown(wait=False) won't cancel them.
            self._pool.shutdown(wait=False, cancel_futures=False)
            logger.info("Resizing thread pool from %d to %d workers", self._pool_size, desired_workers)
        self._pool = ThreadPoolExecutor(max_workers=desired_workers, thread_name_prefix="orchestrator-task")
        self._pool_size = desired_workers
        return self._pool

    def active_execution_blockers(self) -> dict[str, Any]:
        """Return a snapshot of active execution that blocks destructive clear/reset."""
        self._sweep_futures()
        now_ts = time.time()
        tasks = self.container.tasks.list()

        with self._futures_lock:
            active_future_task_ids = sorted(task_id for task_id, future in self._futures.items() if not future.done())

        in_progress_task_ids: list[str] = []
        active_lease_task_ids: list[str] = []
        reasons_by_task: dict[str, list[str]] = {}
        for task in tasks:
            if task.status != "in_progress":
                continue
            if not task.pending_gate:
                in_progress_task_ids.append(task.id)
                reasons_by_task.setdefault(task.id, []).append("in_progress")
            if self._execution_lease_active(task, now_ts=now_ts):
                active_lease_task_ids.append(task.id)
                reasons_by_task.setdefault(task.id, []).append("active_lease")

        for task_id in active_future_task_ids:
            reasons_by_task.setdefault(task_id, []).append("active_future")

        task_ids = sorted(reasons_by_task.keys())
        return {
            "count": len(task_ids),
            "task_ids": task_ids,
            "task_reasons": {task_id: sorted(set(reasons_by_task.get(task_id) or [])) for task_id in task_ids},
            "active_future_task_ids": active_future_task_ids,
            "active_lease_task_ids": sorted(set(active_lease_task_ids)),
            "in_progress_task_ids": sorted(set(in_progress_task_ids)),
        }

    def force_cancel_tasks(self, task_ids: list[str], *, source: str = "api_force_cancel") -> dict[str, Any]:
        """Best-effort cancel for a set of task ids, with per-id outcome reporting."""
        cancelled_task_ids: list[str] = []
        already_cancelled_task_ids: list[str] = []
        terminal_skipped_task_ids: list[str] = []
        missing_task_ids: list[str] = []
        failed: dict[str, str] = {}
        seen: set[str] = set()
        for raw_task_id in task_ids:
            task_id = str(raw_task_id or "").strip()
            if not task_id or task_id in seen:
                continue
            seen.add(task_id)
            task = self.container.tasks.get(task_id)
            if not task:
                missing_task_ids.append(task_id)
                continue
            if task.status == "cancelled":
                already_cancelled_task_ids.append(task_id)
                continue
            if task.status == "done":
                terminal_skipped_task_ids.append(task_id)
                continue
            try:
                self.cancel_task(task_id, source=source)
                cancelled_task_ids.append(task_id)
            except Exception as exc:  # pragma: no cover - defensive path
                failed[task_id] = str(exc)
        return {
            "cancelled_task_ids": sorted(cancelled_task_ids),
            "already_cancelled_task_ids": sorted(already_cancelled_task_ids),
            "terminal_skipped_task_ids": sorted(terminal_skipped_task_ids),
            "missing_task_ids": sorted(missing_task_ids),
            "failed": failed,
        }

    def wait_for_execution_quiescence(
        self,
        *,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> dict[str, Any]:
        """Wait until active execution blockers clear or timeout expires."""
        timeout_seconds = max(float(timeout), 0.0)
        poll_seconds = min(max(float(poll_interval), 0.05), 1.0)
        started = time.monotonic()
        blockers = self.active_execution_blockers()
        if int(blockers.get("count") or 0) == 0:
            return {"quiescent": True, "waited_seconds": 0.0, "blockers": blockers}

        deadline = started + timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                break
            time.sleep(min(poll_seconds, remaining))
            blockers = self.active_execution_blockers()
            if int(blockers.get("count") or 0) == 0:
                break

        waited_seconds = max(0.0, time.monotonic() - started)
        quiescent = int(blockers.get("count") or 0) == 0
        return {"quiescent": quiescent, "waited_seconds": waited_seconds, "blockers": blockers}

    def status(self) -> dict[str, Any]:
        """Build a status snapshot of queue depth and active worker usage.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        tasks = self.container.tasks.list()
        queue_depth = len([task for task in tasks if task.status == "queued"])
        in_progress = len([task for task in tasks if task.status == "in_progress"])
        scheduler_attached = bool(self._thread and self._thread.is_alive())
        with self._futures_lock:
            active_workers = len(self._futures)
        orchestrator_status = str(orchestrator_cfg.get("status", "running") or "running")
        tick_lag_seconds: int | None = None
        if self._last_tick_at:
            tick_epoch = self._parse_iso_epoch(self._last_tick_at)
            if tick_epoch is not None:
                tick_lag_seconds = max(0, int(time.time() - tick_epoch))
        stale_by_tick = bool(
            orchestrator_status == "running"
            and tick_lag_seconds is not None
            and tick_lag_seconds > self._tick_stale_seconds()
            and (queue_depth > 0 or in_progress > 0)
        )
        scheduler_stale = bool(
            orchestrator_status == "running"
            and (not scheduler_attached or stale_by_tick)
            and (queue_depth > 0 or in_progress > 0)
        )
        dispatch_reason = self._dispatch_blocked_reason or ("scheduler_stale" if scheduler_stale else None)
        return {
            "status": orchestrator_status,
            "queue_depth": queue_depth,
            "in_progress": in_progress,
            "active_workers": active_workers,
            "draining": self._drain,
            "run_branch": self._run_branch,
            "scheduler_attached": scheduler_attached,
            "scheduler_stale": scheduler_stale,
            "last_tick_at": self._last_tick_at,
            "last_dispatch_at": self._last_dispatch_at,
            "tick_lag_seconds": tick_lag_seconds,
            "consecutive_tick_failures": int(self._consecutive_tick_failures),
            "last_tick_error": self._last_tick_error,
            "dispatch_blocked_reason": dispatch_reason,
            "last_reconcile_at": self._last_reconcile_at,
            "reconcile_repairs": int(self._last_reconcile_repairs),
            "integration_health": self._integration_health.get_state(),
        }

    def control(self, action: str) -> dict[str, Any]:
        """Apply a control action and return updated orchestrator status.

        Args:
            action (str): Action for this call.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        ensure_worker = False
        reset_worker = False
        if action == "pause":
            orchestrator_cfg["status"] = "paused"
        elif action == "resume":
            orchestrator_cfg["status"] = "running"
            ensure_worker = True
        elif action == "drain":
            self._drain = True
            orchestrator_cfg["status"] = "running"
            ensure_worker = True
        elif action == "stop":
            self._stop.set()
            self._drain = False
            orchestrator_cfg["status"] = "stopped"
        elif action == "reset":
            # Non-destructive scheduler reset/reattach: keep lifecycle status as-is.
            reset_worker = True
        elif action == "reconcile":
            summary = self.reconcile(source="manual")
            payload = self.status()
            payload["reconcile"] = summary
            return payload
        else:
            raise ValueError(f"Unsupported control action: {action}")
        cfg["orchestrator"] = orchestrator_cfg
        self.container.config.save(cfg)
        self.bus.emit(channel="system", event_type="orchestrator.control", entity_id=self.container.project_id, payload={"action": action})
        if reset_worker:
            self._reset_scheduler_thread(timeout=2.0)
        elif ensure_worker:
            self.ensure_worker()
        status_payload = self.status()
        self._persist_runtime_state(force=True)
        return status_payload

    def ensure_worker(self) -> None:
        """Start the background scheduling loop when not already running."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            with self._futures_lock:
                inflight_futures = len([future for future in self._futures.values() if not future.done()])
            if inflight_futures == 0:
                self._recover_in_progress_tasks()
                self._cleanup_orphaned_worktrees()
                try:
                    self.reconcile(source="startup")
                except Exception:
                    logger.exception("Startup reconcile failed; scheduler will continue")
            else:
                logger.info(
                    "Skipping startup recovery/cleanup while %d task future(s) are in-flight",
                    inflight_futures,
                )
            self._last_reconcile_monotonic = time.monotonic()
            stop_token = threading.Event()
            self._stop = stop_token
            self._thread = threading.Thread(target=self._loop, args=(stop_token,), daemon=True, name="orchestrator")
            self._thread.start()
            if not self._watchdog_thread or not self._watchdog_thread.is_alive():
                self._watchdog_stop.clear()
                self._watchdog_thread = threading.Thread(
                    target=self._watchdog_loop,
                    daemon=True,
                    name="orchestrator-watchdog",
                )
                self._watchdog_thread.start()

    def shutdown(self, *, timeout: float = 10.0) -> None:
        """Stop the scheduler and wait for in-flight work up to ``timeout``.

        Args:
            timeout (float): Timeout for this call.
        """
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._watchdog_stop.set()
            watchdog = self._watchdog_thread

        if thread and thread.is_alive():
            thread.join(timeout=max(timeout, 0.0))
        if watchdog and watchdog.is_alive():
            watchdog.join(timeout=max(timeout, 0.0))

        with self._futures_lock:
            inflight = list(self._futures.values())
        if inflight and timeout > 0:
            wait(inflight, timeout=timeout)
        self._sweep_futures()

        pool = self._pool
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=False)
            self._pool = None
            self._pool_size = 0

        self._thread = None
        self._watchdog_thread = None
        self._persist_runtime_state(force=True)

    def _reset_scheduler_thread(self, *, timeout: float = 2.0) -> None:
        """Reset scheduler thread even if the old thread is still alive/stalled."""
        with self._lock:
            old_thread = self._thread
            self._stop.set()
        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=max(timeout, 0.0))
        with self._lock:
            self._thread = None
        self.ensure_worker()

    def _trigger_async_reset(self, *, reason: str) -> None:
        """Schedule one background reset attempt at a time."""
        with self._lock:
            if self._reset_inflight:
                return
            self._reset_inflight = True

        def _runner() -> None:
            try:
                self._reset_scheduler_thread(timeout=2.0)
            except Exception:
                logger.exception("Asynchronous scheduler reset failed (%s)", reason)
            finally:
                with self._lock:
                    self._reset_inflight = False

        threading.Thread(target=_runner, daemon=True, name="orchestrator-reset").start()

    def _watchdog_loop(self) -> None:
        """Auto-reattach scheduler when status indicates stale attachment."""
        while not self._watchdog_stop.wait(5.0):
            try:
                snapshot = self.status()
            except Exception:
                continue
            if snapshot.get("status") != "running":
                continue
            if not snapshot.get("scheduler_stale"):
                continue
            try:
                self._reset_scheduler_thread(timeout=2.0)
            except Exception:
                logger.exception("Watchdog reset attempt failed")

    def _recover_in_progress_tasks(self) -> None:
        tasks = self.container.tasks.list()
        task_by_id = {task.id: task for task in tasks}
        in_progress_ids = {task.id for task in tasks if task.status == "in_progress"}
        if not in_progress_ids:
            return

        completed_task_ids: set[str] = set()
        for run in self.container.runs.list():
            if run.task_id in in_progress_ids and run.status == "in_progress" and not run.finished_at:
                task = task_by_id.get(run.task_id)
                if task and task.pending_gate:
                    run.status = "waiting_gate"
                    run.finished_at = now_iso()
                    run.summary = run.summary or f"Paused at gate: {task.pending_gate}"
                    self.container.runs.upsert(run)
                    continue
                if task and self._is_resume_requested(task):
                    continue
                if task and self._run_contains_successful_commit(run):
                    self._finalize_recovered_completed_task(task, run)
                    completed_task_ids.add(task.id)
                    continue
                run.status = "interrupted"
                run.finished_at = now_iso()
                run.summary = run.summary or "Interrupted by orchestrator restart"
                self.container.runs.upsert(run)

        for task in tasks:
            if task.id not in in_progress_ids:
                continue
            if task.id in completed_task_ids:
                continue
            if isinstance(task.metadata, dict):
                self._mark_task_context_retained(task, reason="orchestrator_restart", expected_on_retry=False)
            if task.pending_gate or self._is_resume_requested(task):
                task.current_agent_id = None
                self.container.tasks.upsert(task)
                self.bus.emit(
                    channel="tasks",
                    event_type="task.recovered",
                    entity_id=task.id,
                    payload={"reason": "orchestrator_restart_waiting_gate"},
                )
                continue
            task.status = "queued"
            task.current_step = None
            task.current_agent_id = None
            task.pending_gate = None
            task.error = "Recovered from interrupted run"
            if isinstance(task.metadata, dict):
                task.metadata.pop("pipeline_phase", None)
            self.container.tasks.upsert(task)
            self.bus.emit(
                channel="tasks",
                event_type="task.recovered",
                entity_id=task.id,
                payload={"reason": "orchestrator_restart"},
            )

    @staticmethod
    def _run_contains_successful_commit(run: RunRecord) -> bool:
        """Return True when a run already recorded a successful commit step."""
        for step_data in reversed(run.steps or []):
            if not isinstance(step_data, dict):
                continue
            if str(step_data.get("step") or "").strip() != "commit":
                continue
            step_status = str(step_data.get("status") or "").strip().lower()
            return step_status == "ok"
        return False

    def _finalize_recovered_completed_task(self, task: Task, run: RunRecord) -> None:
        """Finalize a recovered in-progress task as done when commit already succeeded."""
        finished_at = now_iso()
        run.status = "done"
        run.finished_at = run.finished_at or finished_at
        run.summary = run.summary or "Pipeline completed"
        self.container.runs.upsert(run)

        task.status = "done"
        task.current_step = None
        task.current_agent_id = None
        task.pending_gate = None
        task.error = None
        task.updated_at = finished_at
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        for key in (
            "pipeline_phase",
            "execution_checkpoint",
            "pending_precommit_approval",
            "review_stage",
            "worktree_dir",
            "task_context",
        ):
            task.metadata.pop(key, None)
        self.container.tasks.upsert(task)
        self.bus.emit(
            channel="tasks",
            event_type="task.done",
            entity_id=task.id,
            payload={"source": "orchestrator_recovery"},
        )

    def _sweep_futures(self) -> None:
        """Remove completed futures and log any unexpected errors."""
        with self._futures_lock:
            done_ids = [tid for tid, f in self._futures.items() if f.done()]
            for tid in done_ids:
                fut = self._futures.pop(tid)
                exc = fut.exception()
                if exc:
                    logger.error("Task %s raised unexpected error: %s", tid, exc, exc_info=exc)

    def tick_once(self) -> bool:
        """Run one scheduler iteration and return whether work was dispatched.

        Returns:
            bool: `True` when the operation succeeds, otherwise `False`.
        """
        self._sweep_futures()
        with self._lock:
            cfg = self.container.config.load()
            orchestrator_cfg = dict(cfg.get("orchestrator") or {})
            if orchestrator_cfg.get("status", "running") != "running":
                self._dispatch_blocked_reason = "paused"
                return False
            if self._manual_run_active > 0:
                self._dispatch_blocked_reason = "manual_run"
                return False
            if self._integration_health.is_degraded() and orchestrator_cfg.get(
                "integration_health", {}
            ).get("blocking"):
                # Allow the auto-generated fix task to be dispatched; block
                # everything else to prevent cascading failures on a broken
                # base branch.
                fix_id = self._integration_health._fix_task_id
                fix_task = self.container.tasks.get(fix_id) if fix_id else None
                if not fix_task or fix_task.status != "queued":
                    self._dispatch_blocked_reason = "integration_degraded"
                    return False

            self._apply_gate_wait_policies(orchestrator_cfg)
            self._maybe_analyze_dependencies()

            max_in_progress = int(orchestrator_cfg.get("concurrency", 2) or 2)
            tasks_snapshot = self.container.tasks.list()
            queued_tasks = [task for task in tasks_snapshot if task.status == "queued"]
            running_tasks = [task for task in tasks_snapshot if task.status == "in_progress" and not task.pending_gate]
            claimed = self.container.tasks.claim_next_runnable(max_in_progress=max_in_progress)
            if not claimed:
                if len(running_tasks) >= max_in_progress and queued_tasks:
                    self._dispatch_blocked_reason = "concurrency_limit"
                elif queued_tasks:
                    waiting_gate = [task for task in queued_tasks if task.pending_gate]
                    if waiting_gate:
                        self._dispatch_blocked_reason = "waiting_gate"
                    else:
                        by_id = {task.id: task for task in tasks_snapshot}
                        terminal = {"done", "cancelled"}
                        has_dep_block = False
                        for task in queued_tasks:
                            unresolved = [
                                dep_id
                                for dep_id in task.blocked_by
                                if (by_id.get(dep_id) is None) or (by_id[dep_id].status not in terminal)
                            ]
                            if unresolved:
                                has_dep_block = True
                                break
                        self._dispatch_blocked_reason = "blocked_by_dependencies" if has_dep_block else "no_runnable_queued_tasks"
                else:
                    self._dispatch_blocked_reason = None
                return False
            self._dispatch_blocked_reason = None
            fresh_claimed = self.container.tasks.get(claimed.id) or claimed
            self._acquire_execution_lease(fresh_claimed)
            self.container.tasks.upsert(fresh_claimed)
            claimed = fresh_claimed

            self.bus.emit(channel="queue", event_type="task.claimed", entity_id=claimed.id, payload={"status": claimed.status})
            future = self._get_pool(desired_workers=max_in_progress).submit(self._execute_task, claimed)
            with self._futures_lock:
                self._futures[claimed.id] = future
            return True

    def run_task(self, task_id: str) -> Task:
        """Synchronously execute one task by id and return the final record.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            Task: Result produced by this call.
        """
        wait_existing = False
        manual_run_active = False
        existing_future: Future[Any] | None = None
        with self._lock:
            task = self.container.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            if task.pending_gate and task.status != "in_progress":
                raise ValueError(f"Task {task_id} is waiting for gate approval: {task.pending_gate}")
            # Make explicit run idempotent when a worker already started or finished
            # the same task; this avoids request races with the background loop.
            if task.status in {"in_review", "done"}:
                return task
            if task.status == "in_progress":
                with self._futures_lock:
                    existing_future = self._futures.get(task_id)
                if existing_future:
                    wait_existing = True
                elif self._is_resume_requested(task):
                    wait_existing = False
                    self._clear_resume_request(task)
                    task.updated_at = now_iso()
                    self.container.tasks.upsert(task)
                else:
                    wait_existing = True
            if task.status in {"cancelled"}:
                raise ValueError(f"Task {task_id} cannot be run from status={task.status}")

            if not wait_existing:
                terminal = {"done", "cancelled"}
                for dep_id in task.blocked_by:
                    dep = self.container.tasks.get(dep_id)
                    if dep is None or dep.status not in terminal:
                        raise ValueError(f"Task {task_id} has unresolved blocker {dep_id}")
                if task.status != "in_progress":
                    # Claim manual execution immediately to avoid a race where
                    # the background scheduler grabs this queued task first.
                    task.status = "in_progress"
                    task.pending_gate = None
                    task.current_agent_id = None
                    self._acquire_execution_lease(task)
                    self.container.tasks.upsert(task)
                self._manual_run_active += 1
                manual_run_active = True

        if wait_existing:
            future_to_wait = existing_future
            wait_deadline = time.monotonic() + 5.0
            while future_to_wait is None and time.monotonic() < wait_deadline:
                with self._futures_lock:
                    future_to_wait = self._futures.get(task_id)
                if future_to_wait is not None:
                    break
                observed = self.container.tasks.get(task_id)
                if not observed:
                    raise ValueError(f"Task disappeared during execution: {task_id}")
                if observed.status != "in_progress":
                    return observed
                time.sleep(0.05)
            if future_to_wait:
                future_to_wait.result()
            updated = self.container.tasks.get(task_id)
            if not updated:
                raise ValueError(f"Task disappeared during execution: {task_id}")
            return updated

        future: Future[Any] | None = None
        try:
            future = self._get_pool().submit(self._execute_task, task)
            with self._futures_lock:
                self._futures[task_id] = future
            future.result()
        finally:
            if future is not None:
                with self._futures_lock:
                    self._futures.pop(task_id, None)
            if manual_run_active:
                with self._lock:
                    self._manual_run_active = max(0, self._manual_run_active - 1)
        updated = self.container.tasks.get(task_id)
        if not updated:
            raise ValueError(f"Task disappeared during execution: {task_id}")
        return updated

    def _resolve_worker_lineage(self, task: Task, step: str) -> tuple[str | None, str | None]:
        try:
            cfg = self.container.config.load()
            runtime = get_workers_runtime_config(config=cfg, codex_command_fallback="codex exec")
            spec = resolve_worker_for_step(runtime, step)
        except Exception:
            return None, None
        return spec.name, spec.model

    def _local_branch_exists(self, branch_name: str) -> bool:
        normalized = str(branch_name or "").strip()
        if not normalized:
            return False
        try:
            result = subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{normalized}"],
                cwd=self.container.project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return False
        return result.returncode == 0

    def _pipeline_template_for_task(self, task: Task) -> Any | None:
        registry = PipelineRegistry()
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        pipeline_id = str(metadata.get("final_pipeline_id") or "").strip()
        if pipeline_id:
            try:
                return registry.get(pipeline_id)
            except Exception:
                pass
        try:
            return registry.resolve_for_task_type(task.task_type)
        except Exception:
            return None

    @classmethod
    def _normalize_task_generation_status(cls, value: Any, *, default: str = "backlog") -> str:
        raw = str(value or "").strip().lower()
        if raw in cls._TASK_GENERATION_STATUS_VALUES:
            return raw
        return default

    @classmethod
    def _normalize_task_generation_hitl_selection(cls, value: Any, *, default: str = "inherit_parent") -> str:
        raw = str(value or "").strip().lower()
        if raw == "collaborative":
            raw = "supervised"
        if raw in cls._TASK_GENERATION_HITL_VALUES:
            return raw
        return default

    @staticmethod
    def _coerce_task_generation_infer_deps(value: Any, *, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _system_task_generation_defaults_raw(self) -> dict[str, Any]:
        cfg = self.container.config.load()
        defaults_cfg = dict(cfg.get("defaults") or {})
        raw = defaults_cfg.get("task_generation")
        return dict(raw) if isinstance(raw, dict) else {}

    def supports_task_generation(self, task: Task) -> bool:
        """Return whether the task's effective pipeline includes generate_tasks."""
        template = self._pipeline_template_for_task(task)
        if task.pipeline_template:
            steps = [str(step).strip() for step in task.pipeline_template if str(step).strip()]
        elif template is not None:
            steps = [str(step).strip() for step in template.step_names() if str(step).strip()]
        else:
            steps = []
        return "generate_tasks" in steps

    def resolve_task_generation_policy(
        self,
        task: Task,
        *,
        request_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve generated-child policy from request/task/system/fallback precedence."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        fallback_defaults = dict(self._TASK_GENERATION_DEFAULTS)
        system_raw = self._system_task_generation_defaults_raw()
        task_defaults_raw = task.metadata.get(self._TASK_GENERATION_DEFAULTS_KEY)
        task_raw = dict(task_defaults_raw) if isinstance(task_defaults_raw, dict) else {}
        request_raw = dict(request_overrides) if isinstance(request_overrides, dict) else {}

        status: str
        status_source: str
        if "child_status" in request_raw and request_raw.get("child_status") is not None:
            status = self._normalize_task_generation_status(
                request_raw.get("child_status"),
                default=self._normalize_task_generation_status(
                    task_raw.get("child_status"),
                    default=self._normalize_task_generation_status(
                        system_raw.get("child_status"),
                        default=str(fallback_defaults["child_status"]),
                    ),
                ),
            )
            status_source = "request_override"
        elif "child_status" in task_raw:
            status = self._normalize_task_generation_status(
                task_raw.get("child_status"),
                default=self._normalize_task_generation_status(
                    system_raw.get("child_status"),
                    default=str(fallback_defaults["child_status"]),
                ),
            )
            status_source = "task_defaults"
        elif "child_status" in system_raw:
            status = self._normalize_task_generation_status(
                system_raw.get("child_status"),
                default=str(fallback_defaults["child_status"]),
            )
            status_source = "system_defaults"
        else:
            status = str(fallback_defaults["child_status"])
            status_source = "fallback"

        hitl_selection: str
        hitl_source: str
        if "child_hitl_mode" in request_raw and request_raw.get("child_hitl_mode") is not None:
            hitl_selection = self._normalize_task_generation_hitl_selection(
                request_raw.get("child_hitl_mode"),
                default=self._normalize_task_generation_hitl_selection(
                    task_raw.get("child_hitl_mode"),
                    default=self._normalize_task_generation_hitl_selection(
                        system_raw.get("child_hitl_mode"),
                        default=str(fallback_defaults["child_hitl_mode"]),
                    ),
                ),
            )
            hitl_source = "request_override"
        elif "child_hitl_mode" in task_raw:
            hitl_selection = self._normalize_task_generation_hitl_selection(
                task_raw.get("child_hitl_mode"),
                default=self._normalize_task_generation_hitl_selection(
                    system_raw.get("child_hitl_mode"),
                    default=str(fallback_defaults["child_hitl_mode"]),
                ),
            )
            hitl_source = "task_defaults"
        elif "child_hitl_mode" in system_raw:
            hitl_selection = self._normalize_task_generation_hitl_selection(
                system_raw.get("child_hitl_mode"),
                default=str(fallback_defaults["child_hitl_mode"]),
            )
            hitl_source = "system_defaults"
        else:
            hitl_selection = str(fallback_defaults["child_hitl_mode"])
            hitl_source = "fallback"

        infer_deps: bool
        infer_source: str
        if "infer_deps" in request_raw and request_raw.get("infer_deps") is not None:
            infer_deps = self._coerce_task_generation_infer_deps(
                request_raw.get("infer_deps"),
                default=self._coerce_task_generation_infer_deps(
                    task_raw.get("infer_deps"),
                    default=self._coerce_task_generation_infer_deps(
                        system_raw.get("infer_deps"),
                        default=bool(fallback_defaults["infer_deps"]),
                    ),
                ),
            )
            infer_source = "request_override"
        elif "infer_deps" in task_raw:
            infer_deps = self._coerce_task_generation_infer_deps(
                task_raw.get("infer_deps"),
                default=self._coerce_task_generation_infer_deps(
                    system_raw.get("infer_deps"),
                    default=bool(fallback_defaults["infer_deps"]),
                ),
            )
            infer_source = "task_defaults"
        elif "infer_deps" in system_raw:
            infer_deps = self._coerce_task_generation_infer_deps(
                system_raw.get("infer_deps"),
                default=bool(fallback_defaults["infer_deps"]),
            )
            infer_source = "system_defaults"
        else:
            infer_deps = bool(fallback_defaults["infer_deps"])
            infer_source = "fallback"

        if hitl_selection == "inherit_parent":
            resolved_hitl = normalize_hitl_mode(getattr(task, "hitl_mode", "autopilot"))
        else:
            resolved_hitl = normalize_hitl_mode(hitl_selection)

        return {
            "child_status": status,
            "child_hitl_mode": resolved_hitl,
            "child_hitl_mode_selection": hitl_selection,
            "infer_deps": infer_deps,
            "sources": {
                "child_status": status_source,
                "child_hitl_mode": hitl_source,
                "infer_deps": infer_source,
            },
        }

    def persist_task_generation_defaults(self, task: Task, *, effective_policy: dict[str, Any]) -> dict[str, Any]:
        """Persist normalized generation defaults onto a parent task."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        selection = self._normalize_task_generation_hitl_selection(
            effective_policy.get("child_hitl_mode_selection"),
            default=self._normalize_task_generation_hitl_selection(
                effective_policy.get("child_hitl_mode"),
                default=str(self._TASK_GENERATION_DEFAULTS["child_hitl_mode"]),
            ),
        )
        defaults = {
            "child_status": self._normalize_task_generation_status(
                effective_policy.get("child_status"),
                default=str(self._TASK_GENERATION_DEFAULTS["child_status"]),
            ),
            "child_hitl_mode": selection,
            "infer_deps": self._coerce_task_generation_infer_deps(
                effective_policy.get("infer_deps"),
                default=bool(self._TASK_GENERATION_DEFAULTS["infer_deps"]),
            ),
        }
        task.metadata[self._TASK_GENERATION_DEFAULTS_KEY] = defaults
        return defaults

    def set_pending_task_generation_override(self, task: Task, *, effective_policy: dict[str, Any] | None) -> None:
        """Set a one-shot generation policy override consumed by the next generate_tasks step."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        if not isinstance(effective_policy, dict):
            task.metadata.pop(self._TASK_GENERATION_OVERRIDE_KEY, None)
            return
        selection = self._normalize_task_generation_hitl_selection(
            effective_policy.get("child_hitl_mode_selection"),
            default=self._normalize_task_generation_hitl_selection(
                effective_policy.get("child_hitl_mode"),
                default=str(self._TASK_GENERATION_DEFAULTS["child_hitl_mode"]),
            ),
        )
        task.metadata[self._TASK_GENERATION_OVERRIDE_KEY] = {
            "child_status": self._normalize_task_generation_status(
                effective_policy.get("child_status"),
                default=str(self._TASK_GENERATION_DEFAULTS["child_status"]),
            ),
            "child_hitl_mode": selection,
            "infer_deps": self._coerce_task_generation_infer_deps(
                effective_policy.get("infer_deps"),
                default=bool(self._TASK_GENERATION_DEFAULTS["infer_deps"]),
            ),
        }

    def consume_pending_task_generation_override(self, task: Task) -> dict[str, Any] | None:
        """Consume and clear any queued one-shot generation policy override."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        raw = task.metadata.pop(self._TASK_GENERATION_OVERRIDE_KEY, None)
        if not isinstance(raw, dict):
            return None
        return {
            "child_status": self._normalize_task_generation_status(
                raw.get("child_status"),
                default=str(self._TASK_GENERATION_DEFAULTS["child_status"]),
            ),
            "child_hitl_mode": self._normalize_task_generation_hitl_selection(
                raw.get("child_hitl_mode"),
                default=str(self._TASK_GENERATION_DEFAULTS["child_hitl_mode"]),
            ),
            "infer_deps": self._coerce_task_generation_infer_deps(
                raw.get("infer_deps"),
                default=bool(self._TASK_GENERATION_DEFAULTS["infer_deps"]),
            ),
        }

    @staticmethod
    def _task_blocked_step(task: Task) -> str:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        return str(task.current_step or metadata.get("pipeline_phase") or "").strip()

    def _current_project_branch(self) -> str:
        """Resolve active branch in the project repository, falling back to HEAD."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.container.project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return "HEAD"
        branch = str(result.stdout or "").strip()
        if result.returncode == 0 and branch and branch != "HEAD":
            return branch
        return "HEAD"

    def _skip_material_base_ref(self) -> str:
        """Return baseline ref used to prove preserved work has meaningful diff."""
        run_branch = str(self._run_branch or "").strip()
        if run_branch:
            return run_branch
        return self._current_project_branch()

    def _branch_has_commits_ahead(self, *, branch: str, base_ref: str) -> bool:
        """Return whether branch has commits ahead of base_ref."""
        try:
            result = subprocess.run(
                ["git", "log", f"{base_ref}..refs/heads/{branch}", "--oneline"],
                cwd=self.container.project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return False
        if result.returncode != 0:
            return False
        return bool(str(result.stdout or "").strip())

    def can_skip_to_precommit(self, task: Task) -> tuple[bool, str | None]:
        """Return whether a blocked task can move directly to pre-commit review."""
        if task.status != "blocked":
            return False, "task_not_blocked"
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if metadata.get("pending_precommit_approval"):
            return False, "already_in_precommit_review"
        if task.pending_gate == self._HUMAN_INTERVENTION_GATE:
            return False, "human_intervention_gate"
        if metadata.get("human_blocking_issues"):
            return False, "human_blocking_issues"
        if metadata.get("merge_conflict"):
            return False, "merge_conflict"
        template = self._pipeline_template_for_task(task)
        if template is None:
            return False, "pipeline_unresolved"
        supports_skip = bool((getattr(template, "metadata", {}) or {}).get("supports_skip_to_precommit"))
        if not supports_skip:
            return False, "pipeline_not_supported"
        steps = [str(step).strip() for step in (task.pipeline_template or template.step_names()) if str(step).strip()]
        if "commit" not in steps:
            return False, "pipeline_without_commit"
        if not (self.container.project_dir / ".git").exists():
            return False, "git_required"
        blocked_step = self._task_blocked_step(task)
        if blocked_step not in self._SKIP_TO_PRECOMMIT_COMPATIBLE_BLOCKED_STEPS:
            return False, "blocked_step_not_supported"
        retained_worktree = self._resolve_retained_task_worktree(task)
        preserved_branch = str(metadata.get("preserved_branch") or "").strip()
        has_preserved_branch = bool(preserved_branch and self._local_branch_exists(preserved_branch))
        if retained_worktree is None and not has_preserved_branch:
            return False, "retry_context_unavailable"
        has_material_work = False
        if retained_worktree is not None:
            has_material_work = self._has_uncommitted_changes(retained_worktree) or self._has_commits_ahead(
                retained_worktree
            )
        if (not has_material_work) and has_preserved_branch:
            has_material_work = self._branch_has_commits_ahead(
                branch=preserved_branch,
                base_ref=self._skip_material_base_ref(),
            )
        if not has_material_work:
            return False, "no_task_changes"
        return True, None

    def _list_unmerged_files(self) -> list[str]:
        """Return currently unmerged paths in the project index."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=self.container.project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return []
        # Only discard on fatal git errors (exit >= 128); lower codes may
        # still carry valid stdout on some git versions.
        if result.returncode >= 128:
            return []
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    def _latest_run_commit_ref(self, task: Task) -> str | None:
        """Return newest commit SHA captured in task run steps, if present."""
        for run_id in reversed(task.run_ids):
            run = self.container.runs.get(run_id)
            if not run:
                continue
            for step_data in reversed(run.steps or []):
                if not isinstance(step_data, dict):
                    continue
                commit_ref = str(step_data.get("commit") or "").strip()
                if commit_ref:
                    return commit_ref
        return None

    def _is_ancestor_ref(self, ancestor_ref: str, base_ref: str) -> bool:
        """Return whether ancestor_ref is merged into base_ref."""
        if not ancestor_ref.strip():
            return False
        try:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor_ref, base_ref],
                cwd=self.container.project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return False
        return result.returncode == 0

    def _evaluate_manual_merge_finalize(
        self,
        task: Task,
    ) -> tuple[bool, str | None, str | None, str | None]:
        """Evaluate merge-conflict finalize eligibility and provide verification refs."""
        if task.status != "blocked":
            return False, "task_not_blocked", None, None
        blocked_step = self._task_blocked_step(task)
        if blocked_step != "commit":
            return False, "blocked_step_not_commit", None, None
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not metadata.get("merge_conflict"):
            return False, "not_merge_conflict_block", None, None
        if not (self.container.project_dir / ".git").exists():
            return False, "git_required", None, None
        if self._list_unmerged_files():
            return False, "unmerged_entries_present", None, None

        base_ref = self._run_branch or self._current_project_branch()
        candidate_refs: list[str] = []
        preserved_branch = str(metadata.get("preserved_branch") or "").strip()
        if preserved_branch and self._local_branch_exists(preserved_branch):
            candidate_refs.append(preserved_branch)
        raw_tc = metadata.get("task_context")
        task_context: dict[str, Any] = raw_tc if isinstance(raw_tc, dict) else {}
        task_branch = str(task_context.get("task_branch") or "").strip()
        if task_branch and self._local_branch_exists(task_branch) and task_branch not in candidate_refs:
            candidate_refs.append(task_branch)
        fallback_branch = f"task-{task.id}"
        if self._local_branch_exists(fallback_branch) and fallback_branch not in candidate_refs:
            candidate_refs.append(fallback_branch)
        latest_commit = self._latest_run_commit_ref(task)
        if latest_commit and latest_commit not in candidate_refs:
            candidate_refs.append(latest_commit)

        for candidate in candidate_refs:
            if self._is_ancestor_ref(candidate, base_ref):
                return True, None, base_ref, candidate
        return False, "merge_not_integrated", base_ref, None

    def can_finalize_merge_conflict(self, task: Task) -> tuple[bool, str | None]:
        """Return whether a blocked merge-conflict task can be finalized manually."""
        allowed, reason_code, _, _ = self._evaluate_manual_merge_finalize(task)
        return allowed, reason_code

    def finalize_merge_conflict(self, task_id: str, *, guidance: str = "") -> Task:
        """Mark a manually-resolved merge-conflict task as done after Git verification."""
        with self._lock:
            task = self.container.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            allowed, reason_code, base_ref, verified_ref = self._evaluate_manual_merge_finalize(task)
            if not allowed:
                raise ValueError(
                    f"Task {task_id} cannot finalize manual merge conflict resolution ({reason_code or 'not_allowed'})"
                )

            if not isinstance(task.metadata, dict):
                task.metadata = {}
            for key in (
                "merge_conflict",
                "merge_conflict_files",
                "merge_conflict_attempt",
                "merge_conflict_max_attempts",
                "merge_conflict_previous_error",
                "pending_precommit_approval",
                "review_stage",
                "pipeline_phase",
            ):
                task.metadata.pop(key, None)

            ts = now_iso()
            action_payload = {
                "action": "finalize_merge_conflict",
                "ts": ts,
                "guidance": str(guidance or "").strip(),
                "base_ref": str(base_ref or "").strip(),
                "verified_ref": str(verified_ref or "").strip(),
            }
            history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
            history.append(action_payload)
            task.metadata["last_manual_merge_finalize"] = action_payload
            task.status = "done"
            task.current_step = None
            task.pending_gate = None
            task.current_agent_id = None
            task.error = None
            task.updated_at = ts

            with self.container.transaction():
                latest_run = None
                for run_id in reversed(task.run_ids):
                    latest_run = self.container.runs.get(run_id)
                    if latest_run:
                        break
                if latest_run and latest_run.status != "done":
                    latest_run.status = "done"
                    latest_run.finished_at = latest_run.finished_at or ts
                    if not latest_run.summary:
                        latest_run.summary = "Completed after manual merge conflict resolution"
                    self.container.runs.upsert(latest_run)
                self.container.tasks.upsert(task)

            self.bus.emit(
                channel="tasks",
                event_type="task.done",
                entity_id=task.id,
                payload={"source": "manual_merge_finalize", "verified_ref": str(verified_ref or "").strip()},
            )
            return task

    def skip_task_to_precommit(self, task_id: str, *, guidance: str = "") -> Task:
        """Move an eligible blocked task into pre-commit review without rerunning steps."""
        with self._lock:
            task = self.container.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            allowed, reason_code = self.can_skip_to_precommit(task)
            if not allowed:
                raise ValueError(f"Task {task_id} cannot skip to pre-commit ({reason_code or 'not_allowed'})")
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            blocked_step = self._task_blocked_step(task)
            preserved_branch = str(task.metadata.get("preserved_branch") or "").strip()
            worktree_dir = self._resolve_retained_task_worktree(task)
            if worktree_dir is None and preserved_branch:
                try:
                    worktree_dir = self._create_worktree_from_branch(task, preserved_branch)
                except subprocess.CalledProcessError as exc:
                    raise ValueError(
                        f"Task {task_id} cannot skip to pre-commit (retry_context_attach_failed: {exc})"
                    ) from exc
            if worktree_dir is None:
                raise ValueError(f"Task {task_id} cannot skip to pre-commit (retry_context_unavailable)")
            context_ok, context_reason = self._task_executor._prepare_precommit_review_context(task, worktree_dir)
            if not context_ok:
                detail = context_reason.strip() or "failed_to_prepare_precommit_context"
                raise ValueError(f"Task {task_id} cannot skip to pre-commit ({detail})")

            task.metadata.pop("worktree_dir", None)
            task.metadata.pop("task_context", None)
            task.status = "in_review"
            task.pending_gate = None
            task.current_agent_id = None
            task.current_step = "review"
            task.error = None
            task.metadata["pipeline_phase"] = "review"
            task.metadata["pending_precommit_approval"] = True
            task.metadata["review_stage"] = "pre_commit"
            task.metadata["retry_from_step"] = "commit"
            action_ts = now_iso()
            action_payload = {
                "action": "skip_to_precommit",
                "ts": action_ts,
                "blocked_step": blocked_step,
                "reason_code": "skip_to_precommit",
                "guidance": str(guidance or "").strip(),
            }
            history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
            history.append(action_payload)
            task.metadata["last_skip_to_precommit"] = action_payload
            task.updated_at = action_ts
            self.container.tasks.upsert(task)
            self.bus.emit(
                channel="tasks",
                event_type="task.skip_to_precommit",
                entity_id=task.id,
                payload={
                    "blocked_step": blocked_step,
                    "reason_code": "skip_to_precommit",
                    "has_guidance": bool(str(guidance or "").strip()),
                },
            )
            self.bus.emit(
                channel="review",
                event_type="task.awaiting_human",
                entity_id=task.id,
                payload={"stage": "pre_commit", "source": "skip_to_precommit"},
            )
            return task

    def _active_plan_refine_job(self, task_id: str) -> PlanRefineJob | None:
        return self._plan_manager.active_plan_refine_job(task_id)

    def get_plan_document(self, task_id: str) -> dict[str, Any]:
        """Build plan-revision state plus active refine-job metadata for a task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        return self._plan_manager.get_plan_document(task_id)

    def create_plan_revision(
        self,
        *,
        task_id: str,
        content: str,
        source: Literal["worker_plan", "worker_refine", "human_edit", "import"],
        parent_revision_id: str | None = None,
        step: str | None = None,
        feedback_note: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        status: Literal["draft", "committed"] = "draft",
        created_at: str | None = None,
    ) -> PlanRevision:
        """Create and persist a plan revision for a task.

        Args:
            task_id (str): Identifier for the target task.
            content (str): Content for this call.
            source (Literal['worker_plan', 'worker_refine', 'human_edit', 'import']): Source for this call.
            parent_revision_id (str | None): Identifier for the related parent revision.
            step (str | None): Step for this call.
            feedback_note (str | None): Feedback note for this call.
            provider (str | None): Provider for this call.
            model (str | None): Model for this call.
            status (Literal['draft', 'committed']): Status for this call.
            created_at (str | None): Created at for this call.

        Returns:
            PlanRevision: Result produced by this call.
        """
        return self._plan_manager.create_plan_revision(
            task_id=task_id,
            content=content,
            source=source,
            parent_revision_id=parent_revision_id,
            step=step,
            feedback_note=feedback_note,
            provider=provider,
            model=model,
            status=status,
            created_at=created_at,
        )

    def queue_plan_refine_job(
        self,
        *,
        task_id: str,
        feedback: str,
        instructions: str | None = None,
        base_revision_id: str | None = None,
        priority: str = "normal",
    ) -> PlanRefineJob:
        """Queue a plan refinement job and schedule background processing.

        Args:
            task_id (str): Identifier for the target task.
            feedback (str): Feedback for this call.
            instructions (str | None): Instructions for this call.
            base_revision_id (str | None): Identifier for the related base revision.
            priority (str): Priority for this call.

        Returns:
            PlanRefineJob: Result produced by this call.
        """
        return self._plan_manager.queue_plan_refine_job(
            task_id=task_id,
            feedback=feedback,
            instructions=instructions,
            base_revision_id=base_revision_id,
            priority=priority,
        )

    def process_plan_refine_job(self, job_id: str) -> PlanRefineJob | None:
        """Execute one queued plan-refine job to completion.

        Args:
            job_id (str): Identifier for the target job.

        Returns:
            PlanRefineJob | None: Result produced by this call.
        """
        return self._plan_manager.process_plan_refine_job(job_id)

    def list_plan_refine_jobs(self, task_id: str) -> list[PlanRefineJob]:
        """List refine jobs for a task.

        Args:
            task_id (str): Task identifier whose refine-job history should be returned.

        Returns:
            list[PlanRefineJob]: Refine jobs for the task, ordered by repository behavior.

        Raises:
            ValueError: If the task does not exist.
        """
        return self._plan_manager.list_plan_refine_jobs(task_id)

    def get_plan_refine_job(self, task_id: str, job_id: str) -> PlanRefineJob:
        """Fetch one refine job and verify it belongs to the given task.

        Args:
            task_id (str): Parent task identifier expected for the refine job.
            job_id (str): Refine-job identifier to load.

        Returns:
            PlanRefineJob: The matching refine job owned by ``task_id``.

        Raises:
            ValueError: If the task does not exist.
            ValueError: If the job does not exist or belongs to a different task.
        """
        return self._plan_manager.get_plan_refine_job(task_id, job_id)

    def commit_plan_revision(self, task_id: str, revision_id: str) -> str:
        """Mark one plan revision as committed and sync task/workdoc metadata.

        Args:
            task_id (str): Task identifier that owns the plan revision.
            revision_id (str): Revision identifier to mark as committed.

        Returns:
            str: The committed revision identifier.

        Raises:
            ValueError: If the task does not exist.
            ValueError: If the revision does not exist for the task.
        """
        return self._plan_manager.commit_plan_revision(task_id, revision_id)

    def resolve_plan_text_for_generation(
        self,
        *,
        task_id: str,
        source: Literal["committed", "revision", "override", "latest"],
        revision_id: str | None = None,
        plan_override: str | None = None,
    ) -> tuple[str, str | None]:
        """Resolve plan text and optional revision id for task generation.

        Args:
            task_id (str): Task identifier whose plan should be resolved.
            source (Literal['committed', 'revision', 'override', 'latest']): Source
                strategy to use when selecting plan text.
            revision_id (str | None): Required when ``source='revision'``; ignored
                otherwise.
            plan_override (str | None): Required non-empty text when
                ``source='override'``.

        Returns:
            tuple[str, str | None]: A tuple of ``(plan_text, revision_id)``.
            ``revision_id`` is ``None`` only when ``source='override'``.

        Raises:
            ValueError: If the task does not exist.
            ValueError: If source-specific required inputs are missing.
            ValueError: If the requested revision/committed plan cannot be found.
        """
        return self._plan_manager.resolve_plan_text_for_generation(
            task_id=task_id,
            source=source,
            revision_id=revision_id,
            plan_override=plan_override,
        )

    def _resolve_task_plan_excerpt(self, task: Task, *, max_chars: int = 800) -> str:
        """Extract a bounded plan snippet for prompts and conflict resolution context."""
        return self._worktree_manager.resolve_task_plan_excerpt(task, max_chars=max_chars)

    def _format_task_objective_summary(self, task: Task, *, max_chars: int = 1600) -> str:
        """Build concise objective context for merge-conflict resolution."""
        return self._worktree_manager.format_task_objective_summary(task, max_chars=max_chars)

    def _loop(self, stop_token: threading.Event | None = None) -> None:
        token = stop_token or self._stop
        while not token.is_set():
            self._last_tick_at = now_iso()
            try:
                handled = self.tick_once()
                self._consecutive_tick_failures = 0
                self._last_tick_error = None
                if handled:
                    self._last_dispatch_at = now_iso()
            except Exception:
                handled = False
                self._consecutive_tick_failures += 1
                self._last_tick_error = "Scheduler tick failed"
                logger.exception("Scheduler loop tick failed; loop will continue")
                if self._consecutive_tick_failures >= self._tick_failure_threshold():
                    self.bus.emit(
                        channel="system",
                        event_type="orchestrator.critical",
                        entity_id=self.container.project_id,
                        payload={
                            "reason": "tick_failures_exceeded",
                            "consecutive_tick_failures": self._consecutive_tick_failures,
                        },
                    )
                    # Auto-reset scheduler attachment on repeated tick failures.
                    self._consecutive_tick_failures = 0
                    self._trigger_async_reset(reason="tick_failures_exceeded")
                    break
            with self._futures_lock:
                has_inflight = bool(self._futures)
            if self._drain and not handled and not has_inflight:
                self.control("pause")
                self._drain = False
                break
            now_mono = time.monotonic()
            interval = self._reconcile_interval_seconds()
            if interval > 0 and (now_mono - self._last_reconcile_monotonic) >= interval:
                try:
                    self.reconcile(source="automatic")
                except Exception:
                    logger.exception("Automatic reconciler run failed")
                self._last_reconcile_monotonic = now_mono
            self._persist_runtime_state()
            time.sleep(1 if handled else 2)

    def _create_worktree(self, task: Task) -> Optional[Path]:
        return self._worktree_manager.create_worktree(task)

    def _create_worktree_from_branch(self, task: Task, branch: str) -> Optional[Path]:
        return self._worktree_manager.create_worktree_from_branch(task, branch)

    def _task_context_metadata(self, task: Task) -> dict[str, Any]:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        raw = task.metadata.get("task_context")
        if isinstance(raw, dict):
            return raw
        context: dict[str, Any] = {}
        task.metadata["task_context"] = context
        return context

    def _record_task_context(self, task: Task, *, worktree_dir: Path | None, task_branch: str | None = None) -> None:
        """Persist active task context so retries can deterministically reattach."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        context = self._task_context_metadata(task)
        context_id = str(context.get("context_id") or "").strip()
        if not context_id:
            context["context_id"] = f"ctx-{task.id}-{int(time.time())}"
        if worktree_dir is not None:
            context["worktree_dir"] = str(worktree_dir)
            task.metadata["worktree_dir"] = str(worktree_dir)
        elif task.metadata.get("worktree_dir"):
            context["worktree_dir"] = str(task.metadata.get("worktree_dir"))
        branch_name = str(task_branch or context.get("task_branch") or f"task-{task.id}").strip()
        if branch_name:
            context["task_branch"] = branch_name
        # Snapshot the worktree HEAD before the task starts working so diffs
        # can exclude dependency-task commits that were merged beforehand.
        if worktree_dir is not None and not context.get("initial_head_sha"):
            try:
                head_result = subprocess.run(
                    ["git", "rev-parse", "--verify", "HEAD"],
                    cwd=worktree_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                if head_result.returncode == 0 and head_result.stdout.strip():
                    context["initial_head_sha"] = head_result.stdout.strip()
            except Exception:
                pass
        context["retained"] = False
        context["retained_reason"] = None
        context["retained_at"] = None
        context["expected_on_retry"] = False
        task.metadata["task_context"] = context

    def _mark_task_context_retained(self, task: Task, *, reason: str, expected_on_retry: bool = True) -> None:
        """Mark task context as retained after a recoverable blocked/interrupt path."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        context = self._task_context_metadata(task)
        context_id = str(context.get("context_id") or "").strip()
        if not context_id:
            context["context_id"] = f"ctx-{task.id}-{int(time.time())}"
        worktree_dir = str(task.metadata.get("worktree_dir") or context.get("worktree_dir") or "").strip()
        if worktree_dir:
            context["worktree_dir"] = worktree_dir
        task_branch = str(context.get("task_branch") or f"task-{task.id}").strip()
        if task_branch:
            context["task_branch"] = task_branch
        context["retained"] = True
        context["retained_reason"] = str(reason or "blocked")
        context["retained_at"] = now_iso()
        context["expected_on_retry"] = bool(expected_on_retry)
        task.metadata["task_context"] = context

    def _clear_task_context_retained(self, task: Task) -> None:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        raw = task.metadata.get("task_context")
        if not isinstance(raw, dict):
            return
        raw["retained"] = False
        raw["retained_reason"] = None
        raw["retained_at"] = None
        raw["expected_on_retry"] = False
        task.metadata["task_context"] = raw

    def _task_context_expected(self, task: Task) -> bool:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        raw = metadata.get("task_context")
        if not isinstance(raw, dict):
            return False
        return bool(raw.get("expected_on_retry"))

    def _resolve_retained_task_worktree(self, task: Task, raw_path: str | None = None) -> Path | None:
        """Resolve and validate retained task worktree path for retry attachment."""
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        context_raw = metadata.get("task_context")
        context = context_raw if isinstance(context_raw, dict) else {}
        candidate_raw = str(raw_path or context.get("worktree_dir") or metadata.get("worktree_dir") or "").strip()
        if not candidate_raw:
            return None
        try:
            candidate = Path(candidate_raw).expanduser().resolve()
            expected = (self.container.state_root / "worktrees" / str(task.id)).resolve()
        except Exception:
            return None
        if candidate != expected:
            return None
        if not candidate.exists() or not candidate.is_dir():
            return None
        try:
            inside_result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=candidate,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return None
        if inside_result.returncode != 0 or str(inside_result.stdout or "").strip().lower() != "true":
            return None
        expected_branch = str(context.get("task_branch") or metadata.get("preserved_branch") or f"task-{task.id}").strip()
        try:
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=candidate,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return None
        current_branch = str(branch_result.stdout or "").strip()
        if branch_result.returncode != 0 or not current_branch or current_branch == "HEAD":
            return None
        if expected_branch and current_branch != expected_branch:
            return None
        return candidate

    # ------------------------------------------------------------------

    def _pipeline_id_for_task(self, task: Task) -> str:
        """Resolve pipeline id for a task; fall back safely on failure."""
        try:
            return PipelineRegistry().resolve_for_task_type(task.task_type).id
        except Exception:
            return "feature"

    def _workdoc_template_for_task(self, task: Task) -> str:
        """Select the default workdoc template for the task's pipeline."""
        return self._workdoc_manager.workdoc_template_for_task(task)

    def _workdoc_section_for_step(self, task: Task, step: str) -> tuple[str, str | None] | None:
        """Resolve section heading/placeholder mapping for a step and task pipeline."""
        return self._workdoc_manager.workdoc_section_for_step(task, step)

    def _workdoc_canonical_path(self, task_id: str) -> Path:
        return self._workdoc_manager.workdoc_canonical_path(task_id)

    @staticmethod
    def _workdoc_worktree_path(project_dir: Path) -> Path:
        return WorkdocManager.workdoc_worktree_path(project_dir)

    def _init_workdoc(self, task: Task, project_dir: Path) -> Path:
        """Render the workdoc template, write canonical + worktree copies."""
        return self._workdoc_manager.init_workdoc(task, project_dir)

    @staticmethod
    def _cleanup_workdoc_for_commit(project_dir: Path) -> None:
        """Remove the worktree .workdoc.md before commit so git add -A won't stage it."""
        WorkdocManager.cleanup_workdoc_for_commit(project_dir)

    def _refresh_workdoc(self, task: Task, project_dir: Path) -> None:
        """Copy canonical workdoc to worktree so the worker sees the latest version."""
        self._workdoc_manager.refresh_workdoc(task, project_dir)

    def _sync_workdoc(
        self, task: Task, step: str, project_dir: Path, summary: str | None, attempt: int | None = None
    ) -> None:
        """Post-step sync: accept worker changes or fallback-append summary."""
        self._workdoc_manager.sync_workdoc(task, step, project_dir, summary, attempt)

    def _sync_workdoc_review(self, task: Task, cycle: ReviewCycle, project_dir: Path) -> None:
        """Append review cycle findings to the workdoc."""
        self._workdoc_manager.sync_workdoc_review(task, cycle, project_dir)

    def _append_retry_attempt_marker(
        self,
        task: Task,
        *,
        project_dir: Path,
        attempt: int,
        start_from_step: str | None = None,
    ) -> None:
        """Append retry attempt context into the canonical workdoc."""
        self._workdoc_manager.append_retry_attempt_marker(
            task,
            project_dir=project_dir,
            attempt=attempt,
            start_from_step=start_from_step,
        )

    def _validate_task_workdoc(self, task: Task) -> Path | None:
        """Return canonical workdoc path when present and UTF-8 readable, else ``None``."""
        canonical = self._workdoc_canonical_path(task.id)
        if not canonical.exists():
            return None
        try:
            canonical.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid workdoc encoding (expected UTF-8): {canonical}") from exc
        except OSError as exc:
            raise ValueError(f"Unreadable workdoc: {canonical} ({exc})") from exc
        return canonical

    def _block_for_invalid_workdoc(self, task: Task, run: RunRecord, *, step: str, detail: str) -> None:
        """Fail fast when a task has an unreadable or invalid canonical workdoc."""
        canonical = self._workdoc_canonical_path(task.id)
        task.status = "blocked"
        task.error = detail
        task.pending_gate = None
        task.current_step = step
        task.metadata["pipeline_phase"] = step
        task.metadata["invalid_workdoc_path"] = str(canonical)
        task.metadata["invalid_workdoc_error"] = detail
        self.container.tasks.upsert(task)
        self._finalize_run(task, run, status="blocked", summary=f"Blocked during {step}: invalid workdoc")
        self.bus.emit(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={"error": task.error},
        )

    def _read_canonical_workdoc(self, canonical: Path, *, task_id: str) -> str:
        """Read canonical workdoc text with consistent diagnostics."""
        try:
            return canonical.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid workdoc encoding for task {task_id} (expected UTF-8)") from exc
        except OSError as exc:
            raise ValueError(f"Unreadable workdoc for task {task_id}: {exc}") from exc

    def _read_worktree_workdoc(self, worktree: Path, *, task_id: str) -> str:
        """Read worktree workdoc text with consistent diagnostics."""
        try:
            return worktree.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid workdoc encoding in worktree for task {task_id} (expected UTF-8)") from exc
        except OSError as exc:
            raise ValueError(f"Unreadable worktree workdoc for task {task_id}: {exc}") from exc

    def _read_workdoc_pair(self, task: Task, project_dir: Path) -> tuple[str, str] | None:
        """Read canonical/worktree workdoc texts, returning None when either file is missing."""
        canonical = self._workdoc_canonical_path(task.id)
        if not canonical.exists():
            return None
        worktree = self._workdoc_worktree_path(project_dir)
        if not worktree.exists():
            raise ValueError(f"Missing worktree workdoc during sync for task {task.id}: {worktree}")
        canonical_text = self._read_canonical_workdoc(canonical, task_id=task.id)
        worktree_text = self._read_worktree_workdoc(worktree, task_id=task.id)
        return canonical_text, worktree_text

    @staticmethod
    def _clear_invalid_workdoc_markers(task: Task) -> None:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata.pop("invalid_workdoc_path", None)
        task.metadata.pop("invalid_workdoc_error", None)
        WorkdocManager.clear_sync_diagnostics(task)

    def _sync_workdoc_with_diagnostics(
        self, task: Task, step: str, project_dir: Path, summary: str | None, attempt: int | None = None
    ) -> None:
        """Sync workdoc and surface invalid/unreadable file diagnostics as ValueError."""
        self._workdoc_manager.sync_workdoc(
            task,
            step,
            project_dir,
            summary,
            attempt,
            read_workdoc_pair=lambda: self._read_workdoc_pair(task, project_dir),
        )

    def _refresh_workdoc_with_diagnostics(self, task: Task, project_dir: Path) -> None:
        """Refresh workdoc and surface invalid/unreadable file diagnostics as ValueError.

        Always recreates the worktree copy from canonical when canonical exists.
        """
        canonical = self._workdoc_canonical_path(task.id)
        if not canonical.exists():
            return
        canonical_text = self._read_canonical_workdoc(canonical, task_id=task.id)
        worktree = self._workdoc_worktree_path(project_dir)
        worktree.write_text(canonical_text, encoding="utf-8")

    def _block_for_missing_workdoc(self, task: Task, run: RunRecord, *, step: str) -> None:
        """Fail fast when a task loses its required canonical workdoc."""
        canonical = self._workdoc_canonical_path(task.id)
        task.status = "blocked"
        task.error = f"Missing required workdoc: {canonical.name}"
        task.pending_gate = None
        task.current_step = step
        task.metadata["pipeline_phase"] = step
        task.metadata["missing_workdoc_path"] = str(canonical)
        self.container.tasks.upsert(task)
        self._finalize_run(task, run, status="blocked", summary=f"Blocked during {step}: missing required workdoc")
        self.bus.emit(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={"error": task.error},
        )

    def _step_project_dir(self, task: Task) -> Path:
        """Resolve task worktree directory, falling back to the main project root."""
        worktree_path = task.metadata.get("worktree_dir") if isinstance(task.metadata, dict) else None
        return Path(worktree_path) if worktree_path else self.container.project_dir

    def get_workdoc(self, task_id: str) -> dict[str, Any]:
        """Read canonical workdoc for a task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            dict[str, Any]: Result produced by this call.
        """
        canonical = self._workdoc_canonical_path(task_id)
        if not canonical.exists():
            raise FileNotFoundError(f"Missing required workdoc for task {task_id}")
        content = self._read_canonical_workdoc(canonical, task_id=task_id)
        return {"task_id": task_id, "content": content, "exists": True}

    def _merge_and_cleanup(self, task: Task, worktree_dir: Path) -> MergeOutcome:
        return self._worktree_manager.merge_and_cleanup(task, worktree_dir)

    def approve_and_merge(self, task: Task) -> dict[str, Any]:
        """Merge a preserved branch to the base branch on user approval.

        Called when a blocked task is approved and its work was preserved on a
        branch (for example, after review-attempt limits were hit).

        Args:
            task (Task): Task carrying preserved branch metadata.

        Returns:
            dict[str, Any]: Merge outcome payload with ``status``. Includes
            ``commit_sha`` on successful merge. Failure statuses include
            ``merge_conflict``, ``dirty_overlapping``, and ``git_error``.
        """
        return self._worktree_manager.approve_and_merge(task)

    def _resolve_merge_conflict(self, task: Task, branch: str) -> bool:
        return self._worktree_manager.resolve_merge_conflict(task, branch)

    def _cleanup_orphaned_worktrees(self) -> None:
        self._worktree_manager.cleanup_orphaned_worktrees()

    def _role_for_task(self, task: Task) -> str:
        cfg = self.container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        by_type = dict(routing.get("task_type_roles") or {})
        default_role = str(routing.get("default_role") or "general")
        return str(by_type.get(task.task_type) or default_role)

    def _provider_override_for_role(self, role: str) -> Optional[str]:
        cfg = self.container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        overrides = dict(routing.get("role_provider_overrides") or {})
        raw = overrides.get(role)
        return str(raw) if raw else None

    def _choose_agent_for_task(self, task: Task) -> Optional[str]:
        desired_role = self._role_for_task(task)
        running = [agent for agent in self.container.agents.list() if agent.status == "running"]
        exact = [agent for agent in running if agent.role == desired_role]
        pool = exact or running
        if not pool:
            return None
        pool.sort(key=lambda agent: agent.last_seen_at)
        chosen = pool[0]
        override_provider = self._provider_override_for_role(chosen.role)
        if override_provider:
            task.metadata["provider_override"] = override_provider
        return chosen.id

    def _ensure_branch(self) -> Optional[str]:
        return self._worktree_manager.ensure_branch()

    def _commit_for_task(self, task: Task, working_dir: Optional[Path] = None) -> Optional[str]:
        return self._worktree_manager.commit_for_task(task, working_dir)

    def _has_uncommitted_changes(self, cwd: Path) -> bool:
        """Check whether the working tree has staged or unstaged changes.

        Returns True on git failure (no repo, etc.) to avoid false blocking.
        """
        return self._worktree_manager.has_uncommitted_changes(cwd)

    def _has_commits_ahead(self, cwd: Path) -> bool:
        """Check whether cwd's HEAD has commits beyond the base branch."""
        return self._worktree_manager.has_commits_ahead(cwd)

    def _preserve_worktree_work(self, task: Task, worktree_dir: Path) -> Any:
        """Commit agent edits in the worktree, remove the worktree but keep the branch.

        Returns a structured preserve outcome.
        """
        return self._worktree_manager.preserve_worktree_work(task, worktree_dir)

    def _exceeds_quality_gate(self, task: Task, findings: list[ReviewFinding]) -> bool:
        gate = dict(task.quality_gate or {})
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for finding in findings:
            if finding.status != "open":
                continue
            sev = finding.severity if finding.severity in counts else "low"
            counts[sev] += 1
        return any(counts[sev] > int(gate.get(sev, 0)) for sev in counts)

    def _build_review_history_summary(
        self, task_id: str, *, max_cycles: int = 5, max_chars: int = 4000
    ) -> list[dict[str, Any]]:
        """Build a compact summary of prior review cycles for prompt injection."""
        cycles = self.container.reviews.for_task(task_id)
        cycles.sort(key=lambda c: c.attempt)
        if len(cycles) > max_cycles:
            cycles = cycles[-max_cycles:]
        result: list[dict[str, Any]] = []
        total_chars = 0
        for cycle in cycles:
            entry: dict[str, Any] = {
                "attempt": cycle.attempt,
                "decision": cycle.decision,
                "findings": [],
            }
            for f in cycle.findings:
                f_entry = {"severity": f.severity, "summary": f.summary, "status": f.status}
                total_chars += len(f.summary) + 30
                if total_chars > max_chars:
                    break
                entry["findings"].append(f_entry)
            result.append(entry)
            if total_chars > max_chars:
                break
        return result

    _VERIFY_OUTPUT_TAIL_CHARS = 8000
    _NON_ACTIONABLE_VERIFY_REASON_CODES = {
        "tool_missing",
        "config_missing",
        "no_tests",
        "permission_denied",
        "resource_limit",
        "os_incompatibility",
        "infrastructure",
        "baseline_failure",
    }
    _NON_ACTIONABLE_VERIFY_SUMMARY_PATTERNS = (
        r"command not found",
        r"not installed",
        r"no tests? found",
        r"missing (config|configuration)",
        r"no such file or directory",
        r"permission denied",
        r"operation not permitted",
        r"network is unreachable",
        r"connection refused",
        r"timed out connecting",
        r"docker.*not (available|running)",
        r"requires .* not available",
    )
    _SCOPE_WRITE_STEPS = {"implement", "implement_fix", "prototype"}
    _SCOPE_IGNORE_PATH_SEGMENTS = {
        ".agent_orchestrator",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
    _SCOPE_IGNORE_PATHS = {".workdoc.md"}

    @staticmethod
    def _should_gate(mode: str, gate_name: str) -> bool:
        """Compatibility wrapper for HITL gate checks used by delegated executors."""
        return should_gate(mode, gate_name)

    def _scope_contract_for_task(self, task: Task) -> dict[str, Any] | None:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        normalized = normalize_scope_contract(task.metadata.get("scope_contract"))
        if normalized is None:
            return None
        task.metadata["scope_contract"] = normalized
        return normalized

    def _is_scope_restricted(self, task: Task) -> bool:
        contract = self._scope_contract_for_task(task)
        return bool(contract and str(contract.get("mode") or "open") == "restricted")

    @staticmethod
    def _rel_path(path: str) -> str:
        normalized = str(path or "").strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _is_path_allowed_by_scope(self, task: Task, rel_path: str) -> bool:
        contract = self._scope_contract_for_task(task)
        normalized = self._rel_path(rel_path).lower()
        if not contract:
            return True
        if str(contract.get("mode") or "open") != "restricted":
            return True
        forbidden = [self._rel_path(pat).lower() for pat in list(contract.get("forbidden_globs") or []) if str(pat).strip()]
        if any(fnmatch(normalized, pattern) for pattern in forbidden):
            return False
        allowed = [self._rel_path(pat).lower() for pat in list(contract.get("allowed_globs") or []) if str(pat).strip()]
        if not allowed:
            return False
        return any(fnmatch(normalized, pattern) for pattern in allowed)

    def _ensure_scope_contract_baseline_ref(self, task: Task, project_dir: Path) -> None:
        """Best-effort fill missing scope baseline_ref from current HEAD."""
        contract = self._scope_contract_for_task(task)
        if not contract:
            return
        if str(contract.get("baseline_ref") or "").strip():
            return
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return
        baseline_ref = str(result.stdout or "").strip() if result.returncode == 0 else ""
        if not baseline_ref:
            return
        contract["baseline_ref"] = baseline_ref
        task.metadata["scope_contract"] = contract

    @staticmethod
    def _parse_porcelain_changed_paths(output: str) -> list[str]:
        paths: list[str] = []
        for line in str(output or "").splitlines():
            if not line.strip():
                continue
            # porcelain format: XY <path> OR XY <old> -> <new>
            payload = line[3:] if len(line) > 3 else line
            payload = payload.strip()
            if " -> " in payload:
                payload = payload.split(" -> ", 1)[1].strip()
            cleaned = payload.strip().strip('"')
            rel = OrchestratorService._rel_path(cleaned)
            if rel:
                paths.append(rel)
        return OrchestratorService._dedupe_paths(paths)

    @staticmethod
    def _parse_plain_changed_paths(output: str) -> list[str]:
        paths: list[str] = []
        for line in str(output or "").splitlines():
            rel = OrchestratorService._rel_path(str(line or "").strip())
            if rel:
                paths.append(rel)
        return OrchestratorService._dedupe_paths(paths)

    @staticmethod
    def _dedupe_paths(paths: list[str]) -> list[str]:
        # preserve order while removing duplicates
        seen: set[str] = set()
        uniq: list[str] = []
        for item in paths:
            if item in seen:
                continue
            seen.add(item)
            uniq.append(item)
        return uniq

    def _list_worktree_changed_files(self, project_dir: Path) -> list[str]:
        tracked: list[str] = []
        try:
            tracked_result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
        except Exception:
            tracked_result = None
        if tracked_result and tracked_result.returncode == 0:
            tracked = self._parse_porcelain_changed_paths(str(tracked_result.stdout or ""))

        untracked: list[str] = []
        try:
            untracked_result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
        except Exception:
            untracked_result = None
        if untracked_result and untracked_result.returncode == 0:
            untracked = self._parse_plain_changed_paths(str(untracked_result.stdout or ""))

        return self._dedupe_paths(tracked + untracked)

    @classmethod
    def _is_scope_ignored_path(cls, rel_path: str) -> bool:
        normalized = cls._rel_path(rel_path).lower()
        if not normalized:
            return True
        if normalized in cls._SCOPE_IGNORE_PATHS:
            return True
        segments = [segment for segment in normalized.split("/") if segment]
        return any(segment in cls._SCOPE_IGNORE_PATH_SEGMENTS for segment in segments)

    def _detect_scope_violations(self, task: Task, project_dir: Path) -> list[str]:
        if not self._is_scope_restricted(task):
            return []
        changed = self._list_worktree_changed_files(project_dir)
        if not changed:
            return []
        filtered: list[str] = []
        for path in changed:
            normalized = self._rel_path(path)
            if self._is_scope_ignored_path(normalized):
                continue
            filtered.append(normalized)
        return [path for path in filtered if not self._is_path_allowed_by_scope(task, path)]

    def _block_for_scope_violation(
        self,
        task: Task,
        run: RunRecord,
        *,
        step: str,
        violations: list[str],
    ) -> None:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["scope_violation"] = {
            "step": step,
            "violations": violations,
            "ts": now_iso(),
        }
        task.metadata["verify_reason_code"] = "scope_violation"
        task.status = "blocked"
        task.pending_gate = None
        task.current_step = step
        task.error = (
            "Scope violation: modified out-of-scope files: "
            + ", ".join(violations[:6])
            + (" ..." if len(violations) > 6 else "")
        )
        self.container.tasks.upsert(task)
        run.status = "blocked"
        run.finished_at = now_iso()
        run.summary = f"Blocked during {step}"
        self.container.runs.upsert(run)
        self.bus.emit(
            channel="tasks",
            event_type="task.scope_violation",
            entity_id=task.id,
            payload={"step": step, "violations": violations},
        )
        self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})

    @staticmethod
    def _baseline_debt_signature(summary: str, changed_files: list[str]) -> str:
        normalized = f"{str(summary or '').strip()}||{'|'.join(sorted(set(changed_files)))}"
        return sha256(normalized.encode("utf-8", errors="replace")).hexdigest()

    def _create_scope_baseline_debt_task(
        self,
        task: Task,
        *,
        summary: str,
        changed_files: list[str],
    ) -> None:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        signature = self._baseline_debt_signature(summary, changed_files)
        existing = task.metadata.get("scope_baseline_debt_signatures")
        signatures = list(existing) if isinstance(existing, list) else []
        if signature in signatures:
            return
        signatures.append(signature)
        task.metadata["scope_baseline_debt_signatures"] = signatures

        debt_task = Task(
            title=f"Remediate baseline verification debt for {task.id}",
            description=(
                "Scoped task verification detected unchanged baseline failures outside allowed scope.\n\n"
                f"Origin task: {task.id}\n"
                f"Summary: {summary}\n"
                f"Changed files in scoped task: {', '.join(changed_files) if changed_files else '(none)'}"
            ),
            task_type="chore",
            priority="P1",
            status="backlog",
            source="generated",
            parent_id=task.id,
            metadata={
                "generated_from": "scope_baseline_debt",
                "origin_task_id": task.id,
                "failure_signature": signature,
            },
        )
        self.container.tasks.upsert(debt_task)
        if debt_task.id not in task.blocks:
            task.blocks.append(debt_task.id)
        self.container.tasks.upsert(task)
        self.bus.emit(
            channel="tasks",
            event_type="task.generated_from_pipeline",
            entity_id=task.id,
            payload={
                "created_task_ids": [debt_task.id],
                "reason_code": "baseline_failure",
            },
        )

    def _classify_verify_failure_for_scope(
        self,
        task: Task,
        *,
        step: str,
        summary: str | None,
        project_dir: Path,
    ) -> None:
        """Create baseline debt tasks when the LLM classifier flags baseline_failure."""
        if step not in _VERIFY_STEPS or not self._is_scope_restricted(task):
            return
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        existing_reason = str(task.metadata.get("verify_reason_code") or "").strip().lower()
        if existing_reason != "baseline_failure":
            return
        changed_files = self._list_worktree_changed_files(project_dir)
        task.metadata["verify_scope_classification"] = {
            "mode": "restricted",
            "classification": "unchanged_baseline_debt",
            "step": step,
            "summary": str(summary or "").strip(),
            "changed_files": changed_files,
            "ts": now_iso(),
        }
        self._create_scope_baseline_debt_task(task, summary=str(summary or "").strip(), changed_files=changed_files)

    def _defer_out_of_scope_review_findings(self, task: Task, findings: list[ReviewFinding]) -> list[dict[str, Any]]:
        """Mark open review findings outside restricted scope as deferred."""
        if not self._is_scope_restricted(task):
            return []
        deferred: list[dict[str, Any]] = []
        for finding in findings:
            if finding.status != "open":
                continue
            file_path = self._rel_path(str(finding.file or ""))
            if not file_path:
                continue
            if self._is_path_allowed_by_scope(task, file_path):
                continue
            finding.status = "deferred_out_of_scope"
            deferred.append(
                {
                    "id": finding.id,
                    "file": file_path,
                    "summary": finding.summary,
                    "severity": finding.severity,
                }
            )
        if deferred:
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            task.metadata["review_deferred_out_of_scope"] = deferred
        return deferred

    def _capture_verify_output(self, task: Task) -> None:
        """Read the tail of the verify step's stdout/stderr and stash it in metadata.

        This gives ``implement_fix`` concrete test/lint output to work with,
        not just the terse ``task.error`` summary.
        """
        last_logs = task.metadata.get("last_logs") if isinstance(task.metadata, dict) else None
        if not isinstance(last_logs, dict):
            return
        limit = self._VERIFY_OUTPUT_TAIL_CHARS
        parts: list[str] = []
        for key, label in (("stdout_path", "stdout"), ("stderr_path", "stderr")):
            raw = last_logs.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                p = Path(raw)
                if not p.is_file():
                    continue
                size = p.stat().st_size
                if size <= 0:
                    continue
                # Read only the tail to keep prompt size bounded.
                read_bytes = min(size, limit * 4)
                with open(p, "rb") as fh:
                    if read_bytes < size:
                        fh.seek(size - read_bytes)
                    text = fh.read(read_bytes).decode("utf-8", errors="replace")
                tail = text[-limit:] if len(text) > limit else text
                if tail.strip():
                    parts.append(f"--- {label} (last {len(tail)} chars) ---")
                    parts.append(tail.strip())
            except Exception:
                continue
        if parts:
            task.metadata["verify_output"] = "\n".join(parts)

    def _is_non_actionable_verify_failure(self, task: Task, summary: str | None) -> bool:
        """Classify whether verify failed because of environment or tooling issues."""
        if isinstance(task.metadata, dict):
            reason_code = str(task.metadata.get("verify_reason_code") or "").strip().lower()
            if reason_code in self._NON_ACTIONABLE_VERIFY_REASON_CODES:
                return True
        text = str(summary or "").strip().lower()
        if not text:
            return False
        return any(re.search(pattern, text) for pattern in self._NON_ACTIONABLE_VERIFY_SUMMARY_PATTERNS)

    def _mark_verify_degraded(
        self,
        task: Task,
        run: RunRecord,
        *,
        step: str,
        summary: str | None,
    ) -> None:
        """Persist and emit degraded verification context, then continue best-effort flow."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        entry = {
            "ts": now_iso(),
            "step": step,
            "reason_code": str(task.metadata.get("verify_reason_code") or "unknown"),
            "summary": str(summary or "Verification degraded due to environment/tooling constraints."),
        }
        issues = task.metadata.get("verify_degraded_issues")
        if not isinstance(issues, list):
            issues = []
        issues.append(entry)
        task.metadata["verify_degraded_issues"] = issues
        task.metadata["verify_degraded"] = entry
        task.metadata["verify_non_actionable_failure"] = True
        task.status = "in_progress"
        task.error = None
        task.pending_gate = None
        self._clear_wait_state(task)
        self.container.tasks.upsert(task)
        run.status = "in_progress"
        run.finished_at = None
        run.summary = None
        self.container.runs.upsert(run)
        self.bus.emit(
            channel="tasks",
            event_type="task.verify_degraded",
            entity_id=task.id,
            payload=entry,
        )

    def _consume_verify_non_actionable_flag(self, task: Task) -> bool:
        if not isinstance(task.metadata, dict):
            return False
        return bool(task.metadata.pop("verify_non_actionable_failure", False))

    def _select_post_fix_validation_step(self, steps: list[str]) -> str | None:
        """Pick the validation step that should run after implement_fix in review loops."""
        if "verify" in steps:
            return "verify"
        if "benchmark" in steps:
            return "benchmark"
        return None

    def _run_non_review_step(
        self,
        task: Task,
        run: RunRecord,
        step: str,
        attempt: int = 1,
    ) -> Literal["ok", "verify_failed", "verify_degraded", "auto_requeued", "human_blocked", "blocked"]:
        self._heartbeat_execution_lease(task)
        self.container.tasks.upsert(task)
        try:
            if self._validate_task_workdoc(task) is None:
                self._block_for_missing_workdoc(task, run, step=step)
                return "blocked"
        except ValueError as exc:
            self._block_for_invalid_workdoc(task, run, step=step, detail=str(exc))
            return "blocked"
        # Refresh workdoc before each step so the worker sees the latest version.
        step_project_dir = self._step_project_dir(task)
        try:
            self._refresh_workdoc_with_diagnostics(task, step_project_dir)
        except ValueError as exc:
            self._block_for_invalid_workdoc(task, run, step=step, detail=str(exc))
            return "blocked"

        # Ensure each verify invocation gets fresh classification context.
        if step in _VERIFY_STEPS and isinstance(task.metadata, dict):
            task.metadata.pop("verify_reason_code", None)
            task.metadata.pop("verify_environment_note", None)
            task.metadata.pop("verify_environment_kind", None)
            task.metadata.pop("verify_scope_classification", None)

        step_started = now_iso()
        try:
            result = self.worker_adapter.run_step(task=task, step=step, attempt=attempt)
        except WorkerCancelledError:
            raise self._Cancelled()

        # Post-step workdoc sync (before other bookkeeping that may upsert task).
        try:
            self._sync_workdoc_with_diagnostics(task, step, step_project_dir, result.summary, attempt=attempt)
        except ValueError as exc:
            self._block_for_invalid_workdoc(task, run, step=step, detail=str(exc))
            return "blocked"

        step_log: dict[str, Any] = {"step": step, "status": result.status, "ts": now_iso(), "started_at": step_started, "summary": result.summary}
        if result.human_blocking_issues:
            step_log["human_blocking_issues"] = result.human_blocking_issues
        # Preserve log file paths so historical step logs can be retrieved.
        last_logs = task.metadata.get("last_logs") if isinstance(task.metadata, dict) else None
        if isinstance(last_logs, dict):
            for key in ("stdout_path", "stderr_path", "progress_path"):
                if last_logs.get(key):
                    step_log[key] = last_logs[key]
        run.steps.append(step_log)
        self.container.runs.upsert(run)
        self.container.tasks.upsert(task)
        if result.human_blocking_issues:
            self._block_for_human_issues(task, run, step, result.summary, result.human_blocking_issues)
            return "human_blocked"
        if result.status != "ok":
            if step in _VERIFY_STEPS:
                self._classify_verify_failure_for_scope(
                    task,
                    step=step,
                    summary=result.summary,
                    project_dir=step_project_dir,
                )
            if step in _VERIFY_STEPS and self._is_non_actionable_verify_failure(task, result.summary):
                self._mark_verify_degraded(task, run, step=step, summary=result.summary)
                return "verify_degraded"
            if self._handle_recoverable_environment_failure(
                task,
                run,
                step=step,
                summary=result.summary,
            ):
                return "auto_requeued"
            if step in _VERIFY_STEPS:
                task.status = "in_progress"
                task.error = result.summary or f"{step} failed"
                task.pending_gate = None
                self._clear_wait_state(task)
                task.current_step = step
                self.container.tasks.upsert(task)
                return "verify_failed"
            task.status = "blocked"
            task.error = result.summary or f"{step} failed"
            task.pending_gate = None
            self._clear_wait_state(task)
            task.current_step = step
            self.container.tasks.upsert(task)
            run.status = "blocked"
            run.finished_at = now_iso()
            run.summary = f"Blocked during {step}"
            self.container.runs.upsert(run)
            self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})
            return "blocked"

        self._clear_environment_recovery_tracking(task, step=step)
        task.metadata.pop("human_blocking_issues", None)
        self._clear_wait_state(task)

        # Store plan output as first-class immutable plan revisions.
        if step in {"plan", "initiative_plan"} and result.summary:
            provider, model = self._resolve_worker_lineage(task, step)
            self.create_plan_revision(
                task_id=task.id,
                content=result.summary,
                source="worker_plan",
                step=step,
                provider=provider,
                model=model,
            )
            # Keep in-memory task metadata aligned so later upserts do not overwrite
            # the stored revision pointers/history.
            refreshed = self.container.tasks.get(task.id)
            if refreshed and isinstance(refreshed.metadata, dict):
                task.metadata = dict(refreshed.metadata)

        # Store step output for downstream prompt injection.
        if result.summary:
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            so = task.metadata.setdefault("step_outputs", {})
            # Plan text already bounded at 20KB by _normalize_planning_text.
            # Other outputs truncated to 4KB to prevent metadata bloat.
            max_len = 20_000 if step in {"plan", "initiative_plan"} else 4_000
            so[step] = result.summary[:max_len]

        # Handle generate_tasks: prefer generated tasks, but avoid silent no-op by recording warning metadata.
        if step == "generate_tasks":
            generated = list(result.generated_tasks or [])
            policy_override = self.consume_pending_task_generation_override(task)
            effective_policy = self.resolve_task_generation_policy(task, request_overrides=policy_override)
            if not generated:
                if not isinstance(task.metadata, dict):
                    task.metadata = {}
                task.metadata["generate_tasks_warning"] = result.summary or "generate_tasks produced no structured tasks"
                self.container.tasks.upsert(task)
                self.bus.emit(
                    channel="tasks",
                    event_type="task.generate_tasks_empty",
                    entity_id=task.id,
                    payload={
                        "warning": task.metadata["generate_tasks_warning"],
                        "effective_policy": effective_policy,
                    },
                )
                return "ok"
            created_ids = self._create_child_tasks(task, generated, effective_policy=effective_policy)
            self.bus.emit(
                channel="tasks",
                event_type="task.generated_from_pipeline",
                entity_id=task.id,
                payload={
                    "created_task_ids": created_ids,
                    "effective_policy": effective_policy,
                },
            )

        if step in self._SCOPE_WRITE_STEPS:
            scope_violations = self._detect_scope_violations(task, step_project_dir)
            if scope_violations:
                self._block_for_scope_violation(
                    task,
                    run,
                    step=step,
                    violations=scope_violations,
                )
                return "blocked"

        self._heartbeat_execution_lease(task)
        self.container.tasks.upsert(task)
        return "ok"

    def _clear_environment_recovery_tracking(self, task: Task, *, step: str) -> None:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata.pop("environment_auto_requeue_pending", None)
        task.metadata.pop("environment_next_retry_at", None)
        task.metadata.pop("environment_recovery_backoff_seconds", None)
        attempts_raw = task.metadata.get("environment_recovery_attempts_by_step")
        if isinstance(attempts_raw, dict):
            attempts = dict(attempts_raw)
            attempts.pop(step, None)
            if attempts:
                task.metadata["environment_recovery_attempts_by_step"] = attempts
            else:
                task.metadata.pop("environment_recovery_attempts_by_step", None)
        if isinstance(task.wait_state, dict):
            if str(task.wait_state.get("kind") or "").strip() == self._WAIT_KIND_AUTO_RECOVERY:
                wait_step = str(task.wait_state.get("step") or "").strip()
                if not wait_step or wait_step == step:
                    self._clear_wait_state(task)

    def _environment_recovery_settings(self) -> tuple[int, bool]:
        cfg = self.container.config.load()
        env_cfg = workers_environment_config(cfg)
        max_auto_retries = int(env_cfg.get("max_auto_retries", 3) or 0)
        if max_auto_retries < 0:
            max_auto_retries = 0
        auto_prepare = bool(env_cfg.get("auto_prepare", True))
        return max_auto_retries, auto_prepare

    def _is_environment_recoverable_failure(self, task: Task, summary: str | None) -> bool:
        text = str(summary or "").strip()
        if text and any(pattern.search(text) for pattern in self._ENVIRONMENT_FAILURE_SUMMARY_PATTERNS):
            return True
        if isinstance(task.metadata, dict):
            preflight = task.metadata.get("environment_preflight")
            if isinstance(preflight, dict):
                issues = preflight.get("issues")
                if isinstance(issues, list) and issues:
                    return True
        return False

    def _handle_recoverable_environment_failure(
        self,
        task: Task,
        run: RunRecord,
        *,
        step: str,
        summary: str | None,
    ) -> bool:
        if not self._is_environment_recoverable_failure(task, summary):
            return False

        max_auto_retries, _ = self._environment_recovery_settings()
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        attempts_map_raw = task.metadata.get("environment_recovery_attempts_by_step")
        attempts_map = dict(attempts_map_raw) if isinstance(attempts_map_raw, dict) else {}
        current_attempts = int(attempts_map.get(step) or 0)

        if current_attempts >= max_auto_retries:
            self._clear_environment_recovery_tracking(task, step=step)
            issues = task.metadata.get("environment_preflight")
            issue_summary = str(summary or f"Environment recovery retry limit reached for step '{step}'.").strip()
            details = str(issues) if isinstance(issues, dict) else ""
            escalation = [{"summary": issue_summary, "details": details}] if issue_summary else []
            self._block_for_human_issues(
                task,
                run,
                step,
                summary or f"Environment recovery retry limit reached for step '{step}'.",
                escalation or [{"summary": f"Environment recovery retry limit reached for step '{step}'."}],
            )
            return True

        attempts_map[step] = current_attempts + 1
        attempt_number = attempts_map[step]
        backoff_seconds = min(
            self._ENVIRONMENT_RETRY_MAX_SECONDS,
            self._ENVIRONMENT_RETRY_BASE_SECONDS * (2 ** max(0, attempt_number - 1)),
        )
        next_retry_at = (datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)).isoformat()
        task.metadata["environment_recovery_attempts_by_step"] = attempts_map
        task.metadata["environment_auto_requeue_pending"] = True
        task.metadata["environment_next_retry_at"] = next_retry_at
        task.metadata["environment_recovery_backoff_seconds"] = backoff_seconds
        task.metadata["environment_last_auto_recovery"] = {
            "step": step,
            "attempt": attempt_number,
            "max_auto_retries": max_auto_retries,
            "backoff_seconds": backoff_seconds,
            "next_retry_at": next_retry_at,
            "summary": str(summary or "").strip(),
            "ts": now_iso(),
        }
        task.pending_gate = None
        self._set_wait_state(
            task,
            kind=self._WAIT_KIND_AUTO_RECOVERY,
            step=step,
            reason_code="recoverable_environment_failure",
            recoverable=True,
            attempt=attempt_number,
            max_attempts=max_auto_retries,
            next_retry_at=next_retry_at,
        )
        task.current_step = step
        task.error = str(summary or f"Environment issue detected during {step}. Auto-requeue scheduled.")
        task.status = "queued"
        task.current_agent_id = None
        self.container.tasks.upsert(task)

        run.status = "error"
        run.finished_at = now_iso()
        run.summary = f"Auto-requeue: recoverable environment issue during {step}"
        self.container.runs.upsert(run)

        self.bus.emit(
            channel="tasks",
            event_type="task.auto_recovering",
            entity_id=task.id,
            payload={
                "step": step,
                "attempt": attempt_number,
                "max_auto_retries": max_auto_retries,
                "backoff_seconds": backoff_seconds,
                "next_retry_at": next_retry_at,
                "error": task.error,
            },
        )
        self.bus.emit(
            channel="tasks",
            event_type="task.queued",
            entity_id=task.id,
            payload={"reason": "environment_auto_recovery", "step": step},
        )
        return True

    def _create_child_tasks(
        self,
        parent: Task,
        task_defs: list[dict[str, Any]],
        *,
        effective_policy: dict[str, Any] | None = None,
    ) -> list[str]:
        resolved_policy = effective_policy or self.resolve_task_generation_policy(parent)
        apply_deps = bool(resolved_policy.get("infer_deps", True))
        return self._plan_manager.create_child_tasks(
            parent,
            task_defs,
            apply_deps=apply_deps,
            generation_policy=resolved_policy,
        )

    def generate_tasks_from_plan(
        self,
        task_id: str,
        plan_text: str,
        *,
        infer_deps: bool | None = True,
        generation_policy_overrides: dict[str, Any] | None = None,
        save_as_default: bool = False,
    ) -> tuple[list[str], dict[str, Any]]:
        """Generate child tasks from an explicit plan text.

        This supports a two-phase workflow: run a plan step, review the output,
        then explicitly trigger task generation from the plan.

        Args:
            task_id (str): Parent task identifier that will receive generated children.
            plan_text (str): Plan content to pass into the ``generate_tasks`` worker step.
            infer_deps (bool | None): Whether to apply ``depends_on`` links returned by
                the worker to the created child tasks.
            generation_policy_overrides (dict[str, Any] | None): Optional policy
                overrides (status/HITL/infer_deps) for generated children.
            save_as_default (bool): Persist effective generation policy on parent.

        Returns:
            tuple[list[str], dict[str, Any]]: Created child ids and effective policy.

        Raises:
            ValueError: If the parent task does not exist.
            ValueError: If worker execution fails or yields no valid tasks.
        """
        return self._plan_manager.generate_tasks_from_plan(
            task_id,
            plan_text,
            infer_deps=infer_deps,
            generation_policy_overrides=generation_policy_overrides,
            save_as_default=save_as_default,
        )

    def _findings_from_result(self, task: Task, review_attempt: int) -> tuple[list[ReviewFinding], StepResult]:
        try:
            result = self.worker_adapter.run_step(task=task, step="review", attempt=review_attempt)
        except WorkerCancelledError:
            raise self._Cancelled()
        raw_findings = list(result.findings or [])
        findings: list[ReviewFinding] = []
        for idx, finding in enumerate(raw_findings):
            if not isinstance(finding, dict):
                continue
            findings.append(
                ReviewFinding(
                    id=f"{task.id}-a{review_attempt}-{idx}",
                    task_id=task.id,
                    severity=str(finding.get("severity") or "medium"),
                    category=str(finding.get("category") or "quality"),
                    summary=str(finding.get("summary") or "Issue"),
                    file=finding.get("file"),
                    line=finding.get("line"),
                    suggested_fix=finding.get("suggested_fix"),
                    status=str(finding.get("status") or "open"),
                )
            )
        return findings, result

    def _block_for_human_issues(
        self,
        task: Task,
        run: RunRecord,
        step: str,
        summary: str | None,
        issues: list[dict[str, str]],
    ) -> None:
        task.status = "blocked"
        task.current_step = step
        task.pending_gate = self._HUMAN_INTERVENTION_GATE
        task.error = summary or "Human intervention required to continue"
        task.metadata["human_blocking_issues"] = issues
        task.metadata.pop("environment_auto_requeue_pending", None)
        task.metadata.pop("environment_next_retry_at", None)
        task.metadata.pop("environment_recovery_backoff_seconds", None)
        self._set_wait_state(
            task,
            kind=self._WAIT_KIND_INTERVENTION,
            step=step,
            reason_code="human_intervention_required",
            recoverable=False,
        )
        self.container.tasks.upsert(task)

        self._finalize_run(task, run, status="blocked", summary=f"Blocked during {step}: human intervention required")

        self.bus.emit(
            channel="tasks",
            event_type="task.gate_waiting",
            entity_id=task.id,
            payload={"gate": self._HUMAN_INTERVENTION_GATE, "step": step, "issues": issues},
        )
        self.bus.emit(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={
                "error": task.error,
                "gate": self._HUMAN_INTERVENTION_GATE,
                "step": step,
                "issues": issues,
            },
        )

    def _wait_for_gate(
        self,
        task: Task,
        gate_name: str,
        timeout: int = 3600,
        *,
        resume_step: str | None = None,
    ) -> bool:
        """Park execution at a gate and resume only after explicit approval.

        Returns:
            bool: ``True`` when this gate was already approved and execution can
            continue immediately, otherwise ``False`` after persisting wait state.
        """
        if self._consume_gate_resume_approval(task, gate_name):
            return True

        task.pending_gate = gate_name
        task.status = "in_progress"
        task.current_agent_id = None
        self._release_execution_lease(task)
        self._mark_gate_waiting(task, gate_name, resume_step=resume_step)
        task.updated_at = now_iso()
        self.container.tasks.upsert(task)
        if task.run_ids:
            run = self.container.runs.get(task.run_ids[-1])
            if run and run.status == "in_progress" and not run.finished_at:
                run.status = "waiting_gate"
                run.finished_at = task.updated_at
                run.summary = f"Paused at gate: {gate_name}"
                self.container.runs.upsert(run)
        self.bus.emit(
            channel="tasks",
            event_type="task.gate_waiting",
            entity_id=task.id,
            payload={"gate": gate_name, "timeout_seconds": timeout},
        )
        return False

    def _run_summarize_step(self, task: Task, run: RunRecord) -> None:
        """Auto-inject a summarize step using a lightweight LLM call."""
        fn = getattr(self.worker_adapter, "generate_run_summary", None)
        if fn is None:
            return
        try:
            worktree_path = task.metadata.get("worktree_dir") if isinstance(task.metadata, dict) else None
            project_dir = Path(worktree_path) if worktree_path else self.container.project_dir
            if not project_dir.is_dir():
                project_dir = self.container.project_dir
            summary_started = now_iso()
            summary_text = fn(task=task, run=run, project_dir=project_dir)
            if isinstance(summary_text, str) and summary_text.strip():
                run.steps.append({
                    "step": "summary",
                    "status": "ok",
                    "ts": now_iso(),
                    "started_at": summary_started,
                    "summary": summary_text,
                })
                self.container.runs.upsert(run)
        except Exception:
            logger.debug("Summarize step failed for task %s; skipping", task.id)

    def _finalize_run(self, task: Task, run: RunRecord, *, status: str, summary: str) -> None:
        """Run the summarize step, then set run status/summary/finished_at."""
        self._run_summarize_step(task, run)
        run.status = status
        run.finished_at = now_iso()
        run.summary = summary
        self.container.runs.upsert(run)

    def _maybe_analyze_dependencies(self) -> None:
        """Run automatic dependency analysis on unanalyzed ready tasks."""
        self._dependency_manager.maybe_analyze_dependencies()

    def _apply_dependency_edges(
        self,
        candidates: list[Task],
        edges: list[dict[str, str]],
        all_tasks: list[Task],
    ) -> None:
        """Apply inferred dependency edges with cycle detection."""
        self._dependency_manager.apply_dependency_edges(candidates, edges, all_tasks)

    class _Cancelled(Exception):
        """Raised when a task is cancelled mid-execution."""

    def _check_cancelled(self, task: Task) -> None:
        """Re-read task from storage and raise _Cancelled if user cancelled it."""
        fresh = self.container.tasks.get(task.id)
        if fresh and fresh.status == "cancelled":
            raise self._Cancelled()

    def _execute_task(self, task: Task) -> None:
        self._task_executor.execute_task(task)

    def _execute_task_inner(self, task: Task) -> None:
        self._task_executor.execute_task_inner(task)


def create_orchestrator(
    container: Container,
    bus: EventBus,
    *,
    worker_adapter: WorkerAdapter | None = None,
) -> OrchestratorService:
    """Build, start, and return an orchestrator instance for a container.

    Args:
        container (Container): Container for this call.
        bus (EventBus): Bus for this call.
        worker_adapter (WorkerAdapter | None): Worker adapter for this call.

    Returns:
        OrchestratorService: Result produced by this call.
    """
    if worker_adapter is None:
        from .live_worker_adapter import LiveWorkerAdapter

        worker_adapter = LiveWorkerAdapter(container)
    orchestrator = OrchestratorService(container, bus, worker_adapter=worker_adapter)
    orchestrator.ensure_worker()
    return orchestrator
