"""Define durable task state and structured event models emitted by the runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, cast


class TaskLifecycle(str, Enum):
    """Represent the high-level lifecycle state of a task."""

    READY = "ready"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskStep(str, Enum):
    """Enumerate the runner step currently being executed for a task."""

    RESUME_PROMPT = "resume_prompt"
    PLAN_IMPL = "plan_impl"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    REVIEW = "review"
    COMMIT = "commit"


class PromptMode(str, Enum):
    """Describe the prompt intent used for worker runs."""

    IMPLEMENT = "implement"
    FIX_TESTS = "fix_tests"
    FIX_VERIFY = "fix_verify"
    FIX_FORMAT = "fix_format"
    FIX_LINT = "fix_lint"
    FIX_TYPECHECK = "fix_typecheck"
    ADDRESS_REVIEW = "address_review"
    EXPAND_ALLOWLIST = "expand_allowlist"


@dataclass
class TaskState:
    """Store durable per-task state used to resume and route runner execution."""

    id: str
    type: str = "implement"
    phase_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    priority: int = 0
    deps: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    test_command: Optional[str] = None
    branch: Optional[str] = None

    lifecycle: TaskLifecycle = TaskLifecycle.READY
    step: TaskStep = TaskStep.PLAN_IMPL
    prompt_mode: Optional[PromptMode] = None

    impl_plan_path: Optional[str] = None
    impl_plan_hash: Optional[str] = None
    last_verification: Optional[dict[str, Any]] = None
    last_review_path: Optional[str] = None
    last_review_mergeable: Optional[bool] = None
    last_review_issues: list[str] = field(default_factory=list)
    commit_sha: Optional[str] = None
    review_blockers: list[str] = field(default_factory=list)
    review_blocker_files: list[str] = field(default_factory=list)

    block_reason: Optional[str] = None
    human_blocking_issues: list[str] = field(default_factory=list)
    human_next_steps: list[str] = field(default_factory=list)

    worker_attempts: int = 0
    plan_attempts: int = 0
    no_progress_attempts: int = 0
    test_fail_attempts: int = 0
    review_gen_attempts: int = 0
    review_fix_attempts: int = 0
    allowlist_expansion_attempts: int = 0

    last_run_id: Optional[str] = None
    last_error_type: Optional[str] = None
    last_error: Optional[str] = None
    context: list[str] = field(default_factory=list)

    last_changed_files: list[str] = field(default_factory=list)
    plan_expansion_request: list[str] = field(default_factory=list)

    blocked_intent: Optional[dict[str, Any]] = None
    blocked_at: Optional[str] = None
    auto_resume_attempts: int = 0
    manual_resume_attempts: int = 0

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        """Create a `TaskState` from a persisted dictionary.

        Args:
            data: Raw task state payload from durable storage.

        Returns:
            A `TaskState` instance with unknown keys preserved in `extra`.

        Raises:
            ValueError: If numeric fields cannot be coerced to integers.
        """
        extra = dict(data)

        def _pop(key: str, default: Any = None) -> Any:
            return extra.pop(key, default)

        def _coerce_enum(enum_cls: type[Enum], value: Any, default: Enum) -> Enum:
            if isinstance(value, enum_cls):
                return value
            try:
                return enum_cls(str(value))
            except Exception:
                return default

        lifecycle = _coerce_enum(TaskLifecycle, _pop("lifecycle", None), TaskLifecycle.READY)
        step = _coerce_enum(TaskStep, _pop("step", None), TaskStep.PLAN_IMPL)
        prompt_mode_value = _pop("prompt_mode", None)
        prompt_mode = None
        if prompt_mode_value:
            prompt_mode = _coerce_enum(PromptMode, prompt_mode_value, PromptMode.IMPLEMENT)

        return cls(
            id=str(_pop("id", "")),
            type=str(_pop("type", "implement")),
            phase_id=_pop("phase_id", None),
            title=_pop("title", None),
            description=_pop("description", None),
            priority=int(_pop("priority", 0) or 0),
            deps=list(_pop("deps", []) or []),
            acceptance_criteria=list(_pop("acceptance_criteria", []) or []),
            test_command=_pop("test_command", None),
            branch=_pop("branch", None),
            lifecycle=lifecycle,  # type: ignore[arg-type]
            step=step,  # type: ignore[arg-type]
            prompt_mode=prompt_mode,  # type: ignore[arg-type]
            impl_plan_path=_pop("impl_plan_path", None),
            impl_plan_hash=_pop("impl_plan_hash", None),
            last_verification=_pop("last_verification", None),
            last_review_path=_pop("last_review_path", None),
            last_review_mergeable=_pop("last_review_mergeable", None),
            last_review_issues=list(_pop("last_review_issues", []) or []),
            commit_sha=_pop("commit_sha", None),
            review_blockers=list(_pop("review_blockers", []) or []),
            review_blocker_files=list(_pop("review_blocker_files", []) or []),
            block_reason=_pop("block_reason", None),
            human_blocking_issues=list(_pop("human_blocking_issues", []) or []),
            human_next_steps=list(_pop("human_next_steps", []) or []),
            worker_attempts=int(_pop("worker_attempts", 0) or 0),
            plan_attempts=int(_pop("plan_attempts", 0) or 0),
            no_progress_attempts=int(_pop("no_progress_attempts", 0) or 0),
            test_fail_attempts=int(_pop("test_fail_attempts", 0) or 0),
            review_gen_attempts=int(_pop("review_gen_attempts", 0) or 0),
            review_fix_attempts=int(_pop("review_fix_attempts", 0) or 0),
            allowlist_expansion_attempts=int(_pop("allowlist_expansion_attempts", 0) or 0),
            last_run_id=_pop("last_run_id", None),
            last_error_type=_pop("last_error_type", None),
            last_error=_pop("last_error", None),
            context=list(_pop("context", []) or []),
            last_changed_files=list(_pop("last_changed_files", []) or []),
            plan_expansion_request=list(_pop("plan_expansion_request", []) or []),
            blocked_intent=_pop("blocked_intent", None),
            blocked_at=_pop("blocked_at", None),
            auto_resume_attempts=int(_pop("auto_resume_attempts", 0) or 0),
            manual_resume_attempts=int(_pop("manual_resume_attempts", 0) or 0),
            extra=extra,
        )

    def computed_status(self) -> str:
        if self.lifecycle == TaskLifecycle.DONE:
            return "done"
        if self.lifecycle == TaskLifecycle.WAITING_HUMAN:
            return "blocked"
        if self.lifecycle == TaskLifecycle.CANCELLED:
            return "cancelled"
        if self.step == TaskStep.IMPLEMENT:
            return "implementing"
        return str(self.step.value)

    @property
    def status(self) -> str:
        return self.computed_status()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task state as a plain dictionary for persistence.

        Returns:
            A dictionary suitable for writing to `task_queue.yaml`.
        """
        data = dict(self.extra)
        data.update(
            {
                "id": self.id,
                "type": self.type,
                "phase_id": self.phase_id,
                "title": self.title,
                "description": self.description,
                "priority": int(self.priority),
                "deps": list(self.deps),
                "acceptance_criteria": list(self.acceptance_criteria),
                "test_command": self.test_command,
                "branch": self.branch,
                "status": self.computed_status(),
                "lifecycle": self.lifecycle.value,
                "step": self.step.value,
                "prompt_mode": self.prompt_mode.value if self.prompt_mode else None,
                "impl_plan_path": self.impl_plan_path,
                "impl_plan_hash": self.impl_plan_hash,
                "last_verification": self.last_verification,
                "last_review_path": self.last_review_path,
                "last_review_mergeable": self.last_review_mergeable,
                "last_review_issues": list(self.last_review_issues),
                "commit_sha": self.commit_sha,
                "review_blockers": list(self.review_blockers),
                "review_blocker_files": list(self.review_blocker_files),
                "block_reason": self.block_reason,
                "human_blocking_issues": list(self.human_blocking_issues),
                "human_next_steps": list(self.human_next_steps),
                "worker_attempts": int(self.worker_attempts),
                "plan_attempts": int(self.plan_attempts),
                "no_progress_attempts": int(self.no_progress_attempts),
                "test_fail_attempts": int(self.test_fail_attempts),
                "review_gen_attempts": int(self.review_gen_attempts),
                "review_fix_attempts": int(self.review_fix_attempts),
                "allowlist_expansion_attempts": int(self.allowlist_expansion_attempts),
                "last_run_id": self.last_run_id,
                "last_error_type": self.last_error_type,
                "last_error": self.last_error,
                "context": list(self.context),
                "last_changed_files": list(self.last_changed_files),
                "plan_expansion_request": list(self.plan_expansion_request),
                "blocked_intent": self.blocked_intent,
                "blocked_at": self.blocked_at,
                "auto_resume_attempts": int(self.auto_resume_attempts),
                "manual_resume_attempts": int(self.manual_resume_attempts),
            }
        )
        return data


@dataclass
class Event:
    """Base class for structured events emitted by the runner."""

    event_type: str = field(init=False, default="event")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event as a dictionary, converting enums to values.

        Returns:
            A JSON-friendly dictionary representation of the event.
        """
        def _serialize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, list):
                return [_serialize(item) for item in value]
            if isinstance(value, dict):
                return {key: _serialize(val) for key, val in value.items()}
            return value

        data = cast(dict[str, Any], _serialize(asdict(self)))
        data["event_type"] = self.event_type
        return data


@dataclass
class WorkerSucceeded(Event):
    """Represent a successful worker run for a specific step."""

    run_id: str
    step: TaskStep
    changed_files: list[str] = field(default_factory=list)
    introduced_changes: bool = False
    repo_dirty: bool = False
    plan_valid: Optional[bool] = None
    plan_issue: Optional[str] = None
    impl_plan_path: Optional[str] = None
    impl_plan_hash: Optional[str] = None
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)
    captured_at: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="worker_succeeded")


@dataclass
class WorkerFailed(Event):
    """Represent a failed worker run for a specific step."""

    run_id: str
    step: TaskStep
    error_type: str
    error_detail: str
    changed_files: list[str] = field(default_factory=list)
    stderr_tail: str = ""
    timed_out: bool = False
    no_heartbeat: bool = False
    introduced_changes: bool = False
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)
    captured_at: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="worker_failed")


@dataclass
class ProgressHumanBlockers(Event):
    """Capture human-blocking issues reported by the worker."""

    run_id: str
    issues: list[str]
    next_steps: list[str]
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)
    captured_at: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="progress_human_blockers")


@dataclass
class AllowlistViolation(Event):
    """Report that the worker changed files outside the allowed allowlist."""

    run_id: str
    disallowed_paths: list[str]
    changed_files: list[str] = field(default_factory=list)
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)
    captured_at: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="allowlist_violation")


@dataclass
class NoIntroducedChanges(Event):
    """Report that the worker introduced no changes for a step."""

    run_id: str
    repo_dirty: bool
    changed_files: list[str] = field(default_factory=list)
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)
    captured_at: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="no_introduced_changes")


@dataclass
class VerificationResult(Event):
    """Capture the result of running VERIFY-stage commands."""

    run_id: str
    passed: bool
    command: Optional[str]
    exit_code: int
    log_path: Optional[str]
    log_tail: str
    captured_at: Optional[str] = None
    failing_paths: list[str] = field(default_factory=list)
    needs_allowlist_expansion: bool = False
    error_type: Optional[str] = None
    stage: Optional[str] = None  # "format", "lint", "typecheck", "tests"
    all_failing_paths: list[str] = field(default_factory=list)  # All files with issues (not just expansion)
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="verification_result")


@dataclass
class ReviewResult(Event):
    """Capture the result of a review step."""

    run_id: str
    mergeable: bool
    issues: list[str] = field(default_factory=list)
    review_path: Optional[str] = None
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)
    captured_at: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="review_result")


@dataclass
class CommitResult(Event):
    """Capture the result of COMMIT-stage git operations."""

    run_id: str
    committed: bool
    commit_sha: Optional[str] = None
    pushed: bool = False
    error: Optional[str] = None
    repo_clean: bool = False
    skipped: bool = False
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)
    captured_at: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="commit_result")


@dataclass
class ResumePromptResult(Event):
    """Capture the result of a standalone resume prompt action."""

    run_id: str
    succeeded: bool
    changed_files: list[str] = field(default_factory=list)
    error_detail: Optional[str] = None
    task_id: Optional[str] = field(default=None, kw_only=True)
    phase: Optional[str] = field(default=None, kw_only=True)
    captured_at: Optional[str] = field(default=None, kw_only=True)

    event_type: str = field(init=False, default="resume_prompt_result")
