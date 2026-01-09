STATE_DIR_NAME = ".prd_runner"
RUN_STATE_FILE = "run_state.yaml"
TASK_QUEUE_FILE = "task_queue.yaml"
PHASE_PLAN_FILE = "phase_plan.yaml"
ARTIFACTS_DIR = "artifacts"
RUNS_DIR = "runs"
LOCK_FILE = ".lock"

DEFAULT_SHIFT_MINUTES = 45
DEFAULT_HEARTBEAT_SECONDS = 120
DEFAULT_HEARTBEAT_GRACE_SECONDS = 300
DEFAULT_MAX_ATTEMPTS = 5  # Increased default since we loop on tests
DEFAULT_MAX_AUTO_RESUMES = 3
DEFAULT_STOP_ON_BLOCKING_ISSUES = True
WINDOWS_LOCK_BYTES = 4096

TRANSIENT_ERROR_MARKERS = (
    "No heartbeat",
    "Shift timed out",
)

TASK_STATUS_TODO = "todo"
TASK_STATUS_DOING = "doing"
TASK_STATUS_PLAN_IMPL = "plan_impl"
TASK_STATUS_IMPLEMENTING = "implementing"
TASK_STATUS_TESTING = "testing"  # Kept for compatibility, mapped to implementing logic
TASK_STATUS_REVIEW = "review"
TASK_STATUS_DONE = "done"
TASK_STATUS_BLOCKED = "blocked"

TASK_IN_PROGRESS_STATUSES = {
    TASK_STATUS_DOING,
    "in_progress",
    TASK_STATUS_PLAN_IMPL,
    TASK_STATUS_IMPLEMENTING,
    TASK_STATUS_REVIEW,
}

TASK_RUN_CODEX_STATUSES = {
    TASK_STATUS_DOING,
    "in_progress",
    TASK_STATUS_PLAN_IMPL,
    TASK_STATUS_IMPLEMENTING,
    TASK_STATUS_REVIEW,
}

ERROR_TYPE_HEARTBEAT_TIMEOUT = "heartbeat_timeout"
ERROR_TYPE_SHIFT_TIMEOUT = "shift_timeout"
ERROR_TYPE_CODEX_EXIT = "codex_exit"
ERROR_TYPE_PLAN_MISSING = "plan_missing"
ERROR_TYPE_BLOCKING_ISSUES = "blocking_issues"
ERROR_TYPE_DISALLOWED_FILES = "disallowed_files"
AUTO_RESUME_ERROR_TYPES = {
    ERROR_TYPE_HEARTBEAT_TIMEOUT,
    ERROR_TYPE_SHIFT_TIMEOUT,
}

# Resolution steps help the user when the runner stops
BLOCKING_RESOLUTION_STEPS = {
    ERROR_TYPE_CODEX_EXIT: [
        "Verify Codex CLI is installed, authenticated, and reachable.",
        "Inspect the latest run logs for stderr output.",
    ],
    ERROR_TYPE_PLAN_MISSING: [
        "Open the PRD and regenerate the phase plan.",
        "Ensure phase_plan.yaml and task_queue.yaml are updated.",
    ],
    ERROR_TYPE_HEARTBEAT_TIMEOUT: [
        "Check Codex CLI connectivity and long-running command settings.",
        "Re-run the runner after the worker is healthy.",
    ],
    "review_attempts_exhausted": [
        "Open the review JSON and address all blocking issues.",
        "Re-run the runner once fixes are in place.",
    ],
    ERROR_TYPE_DISALLOWED_FILES: [
        "Revert or move the out-of-scope changes.",
        "Update the implementation plan to include needed files before re-running.",
    ],
    "git_push_failed": [
        "Check git remote/authentication and resolve conflicts.",
        "Push the branch manually, then re-run the runner.",
    ],
}

REVIEW_MIN_EVIDENCE_ITEMS = 2
REVIEW_MIN_ARCH_SUMMARY_ITEMS = 3
REVIEW_MAX_ARCH_SUMMARY_ITEMS = 8
REVIEW_MET_VALUES = {"yes", "no", "partial"}
REVIEW_ARCHITECTURE_CHECKS = [
    "right abstractions introduced",
    "responsibilities split cleanly",
    "failure modes handled and observable",
    "state consistent or idempotent",
    "matches project conventions",
]
MAX_REVIEW_ATTEMPTS = 3
MAX_NO_CHANGE_ATTEMPTS = 3
MAX_IMPL_PLAN_ATTEMPTS = 3
MAX_NO_PROGRESS_ATTEMPTS = 3  # Allowed "no-op" runs before blocking
MAX_MANUAL_RESUME_ATTEMPTS = 10
MAX_TEST_FAIL_ATTEMPTS = 3
REVIEW_SEVERITIES = {"critical", "high", "medium", "low"}
REVIEW_BLOCKING_SEVERITIES = {"critical", "high"}  # gate commit on these

IGNORED_REVIEW_PATH_PREFIXES = [
    ".prd_runner/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    ".venv/",
    "venv/",
    "dist/",
    "build/",
    ".eggs/",
    "htmlcov/",
    "*.egg-info",
    "*.egg-info/*",
    ".coverage",
    "*.pyc",
    "*.pyo",
]
