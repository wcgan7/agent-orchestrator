from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskLifecycle(str, Enum):
    READY = "ready"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskStep(str, Enum):
    RESUME_PROMPT = "resume_prompt"
    PLAN_IMPL = "plan_impl"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    REVIEW = "review"
    COMMIT = "commit"


class PromptMode(str, Enum):
    IMPLEMENT = "implement"
    FIX_TESTS = "fix_tests"
    ADDRESS_REVIEW = "address_review"
    EXPAND_ALLOWLIST = "expand_allowlist"


@dataclass
class TaskState:
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

    def legacy_status(self) -> str:
        if self.lifecycle == TaskLifecycle.DONE:
            return "done"
        if self.lifecycle == TaskLifecycle.WAITING_HUMAN:
            return "blocked"
        if self.lifecycle == TaskLifecycle.CANCELLED:
            return "cancelled"
        if self.step == TaskStep.IMPLEMENT:
            return "implementing"
        return self.step.value

    def to_dict(self) -> dict[str, Any]:
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
                "lifecycle": self.lifecycle.value,
                "step": self.step.value,
                "prompt_mode": self.prompt_mode.value if self.prompt_mode else None,
                "impl_plan_path": self.impl_plan_path,
                "impl_plan_hash": self.impl_plan_hash,
                "last_verification": self.last_verification,
                "last_review_path": self.last_review_path,
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
        data["status"] = self.legacy_status()
        return data


@dataclass
class Event:
    event_type: str = field(init=False, default="event")

    def to_dict(self) -> dict[str, Any]:
        def _serialize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, list):
                return [_serialize(item) for item in value]
            if isinstance(value, dict):
                return {key: _serialize(val) for key, val in value.items()}
            return value

        data = _serialize(asdict(self))
        data["event_type"] = self.event_type
        return data


@dataclass
class WorkerSucceeded(Event):
    step: TaskStep
    run_id: str
    changed_files: list[str] = field(default_factory=list)
    introduced_changes: list[str] = field(default_factory=list)
    repo_dirty: bool = False
    plan_valid: Optional[bool] = None
    plan_issue: Optional[str] = None
    impl_plan_path: Optional[str] = None
    impl_plan_hash: Optional[str] = None

    event_type: str = field(init=False, default="worker_succeeded")


@dataclass
class WorkerFailed(Event):
    step: TaskStep
    run_id: str
    error_type: str
    error_detail: str
    stderr_tail: str
    timed_out: bool
    no_heartbeat: bool
    changed_files: list[str] = field(default_factory=list)
    introduced_changes: list[str] = field(default_factory=list)

    event_type: str = field(init=False, default="worker_failed")


@dataclass
class ProgressHumanBlockers(Event):
    run_id: str
    issues: list[str]
    next_steps: list[str]

    event_type: str = field(init=False, default="progress_human_blockers")


@dataclass
class AllowlistViolation(Event):
    run_id: str
    step: TaskStep
    disallowed_paths: list[str]
    changed_files: list[str] = field(default_factory=list)
    introduced_changes: list[str] = field(default_factory=list)

    event_type: str = field(init=False, default="allowlist_violation")


@dataclass
class NoIntroducedChanges(Event):
    run_id: str
    step: TaskStep
    repo_dirty: bool
    changed_files: list[str] = field(default_factory=list)

    event_type: str = field(init=False, default="no_introduced_changes")


@dataclass
class VerificationResult(Event):
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

    event_type: str = field(init=False, default="verification_result")


@dataclass
class ReviewResult(Event):
    run_id: str
    valid: bool
    blocking_severities_present: bool
    issues: list[dict[str, Any]]
    files: list[str]
    review_path: Optional[str] = None
    review_issue: Optional[str] = None

    event_type: str = field(init=False, default="review_result")


@dataclass
class CommitResult(Event):
    run_id: str
    pushed: bool
    error: Optional[str]
    repo_clean: bool

    event_type: str = field(init=False, default="commit_result")


@dataclass
class ResumePromptResult(Event):
    run_id: str
    succeeded: bool
    error_detail: Optional[str] = None

    event_type: str = field(init=False, default="resume_prompt_result")
