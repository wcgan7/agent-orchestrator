"""Define runner defaults, state file names, and shared constant values."""

STATE_DIR_NAME = ".prd_runner"
RUN_STATE_FILE = "run_state.yaml"
TASK_QUEUE_FILE = "task_queue.yaml"
PHASE_PLAN_FILE = "phase_plan.yaml"
ARTIFACTS_DIR = "artifacts"
RUNS_DIR = "runs"
CONFIG_FILE = "config.yaml"
LOCK_FILE = ".lock"

DEFAULT_SHIFT_MINUTES = 45
DEFAULT_HEARTBEAT_SECONDS = 120
DEFAULT_HEARTBEAT_GRACE_SECONDS = 300
DEFAULT_MAX_ATTEMPTS = 5  # Worker retries before human intervention
DEFAULT_MAX_AUTO_RESUMES = 3
DEFAULT_STOP_ON_BLOCKING_ISSUES = True
WINDOWS_LOCK_BYTES = 4096

TRANSIENT_ERROR_MARKERS = (
    "No heartbeat",
    "Shift timed out",
)

TASK_STATUS_TODO = "todo"
TASK_STATUS_PLAN_IMPL = "plan_impl"
TASK_STATUS_IMPLEMENTING = "implementing"
TASK_STATUS_REVIEW = "review"
TASK_STATUS_DONE = "done"
TASK_STATUS_BLOCKED = "blocked"

ERROR_TYPE_HEARTBEAT_TIMEOUT = "heartbeat_timeout"
ERROR_TYPE_SHIFT_TIMEOUT = "shift_timeout"
ERROR_TYPE_CODEX_EXIT = "codex_exit"
ERROR_TYPE_PLAN_MISSING = "plan_missing"
ERROR_TYPE_BLOCKING_ISSUES = "blocking_issues"
ERROR_TYPE_DISALLOWED_FILES = "disallowed_files"
ERROR_TYPE_TEST_TIMEOUT = "test_timeout"
ERROR_TYPE_STATE_CORRUPT = "state_corrupt"
ERROR_TYPE_PRD_MISMATCH = "prd_mismatch"
ERROR_TYPE_DIRTY_WORKTREE = "dirty_worktree"
ERROR_TYPE_STATE_RESET_FAILED = "state_reset_failed"
ERROR_TYPE_PRD_READ_FAILED = "prd_read_failed"
ERROR_TYPE_STATE_INVALID = "state_invalid"
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
    ERROR_TYPE_STATE_CORRUPT: [
        "Open the listed .prd_runner state file and fix/restore it (or delete it if safe).",
        "Re-run with --reset-state to start fresh (will archive existing .prd_runner).",
    ],
    ERROR_TYPE_STATE_RESET_FAILED: [
        "Check filesystem permissions and whether another process is using .prd_runner.",
        "Delete or move .prd_runner manually, then re-run with --reset-state.",
    ],
    ERROR_TYPE_STATE_INVALID: [
        "Open the listed .prd_runner file(s) and fix the invalid entries.",
        "Re-run with --reset-state to start fresh (will archive existing .prd_runner).",
    ],
    ERROR_TYPE_PRD_MISMATCH: [
        "Confirm you are running against the intended PRD.",
        "Re-run with --reset-state to start a new run for the new PRD.",
    ],
    ERROR_TYPE_PRD_READ_FAILED: [
        "Verify the PRD file exists and is readable.",
        "Fix permissions/path and re-run.",
    ],
    ERROR_TYPE_DIRTY_WORKTREE: [
        "Commit/stash your current changes or reset to a clean working tree.",
        "Re-run with --no-require-clean only if you intentionally want to proceed from a dirty tree.",
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
MAX_REVIEW_ATTEMPTS = 10
MAX_NO_CHANGE_ATTEMPTS = 3
MAX_IMPL_PLAN_ATTEMPTS = 10
MAX_NO_PROGRESS_ATTEMPTS = 3  # Allowed "no-op" runs before blocking
MAX_MANUAL_RESUME_ATTEMPTS = 10
MAX_TEST_FAIL_ATTEMPTS = 10
MAX_ALLOWLIST_EXPANSION_ATTEMPTS = 10
REVIEW_SEVERITIES = {"critical", "high", "medium", "low"}
REVIEW_BLOCKING_SEVERITIES = {"critical", "high", "medium"}  # gate commit on these

# Legacy ignored paths (kept for backwards compatibility)
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

# Language-specific ignored paths for code review
IGNORED_PATHS_BY_LANGUAGE: dict[str, list[str]] = {
    "python": [
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".tox/",
        ".venv/",
        "venv/",
        ".eggs/",
        "htmlcov/",
        "*.egg-info",
        "*.egg-info/*",
        ".coverage",
        "*.pyc",
        "*.pyo",
    ],
    "typescript": [
        "node_modules/",
        "dist/",
        "build/",
        ".next/",
        "out/",
        "coverage/",
        ".jest_cache/",
        ".turbo/",
        "*.js.map",
        "*.d.ts.map",
        ".tsbuildinfo",
        "*.tsbuildinfo",
    ],
    "nextjs": [
        "node_modules/",
        "dist/",
        "build/",
        ".next/",
        ".vercel/",
        "out/",
        "coverage/",
        ".jest_cache/",
        ".turbo/",
        "*.js.map",
        "*.d.ts.map",
        ".tsbuildinfo",
        "*.tsbuildinfo",
    ],
    "javascript": [
        "node_modules/",
        "dist/",
        "build/",
        "coverage/",
        ".jest_cache/",
    ],
    "go": [
        "vendor/",
        "bin/",
        "*.exe",
    ],
    "rust": [
        "target/",
        "Cargo.lock",
    ],
    "common": [
        ".prd_runner/",
        ".git/",
        ".DS_Store",
        "*.log",
    ],
}


def get_ignored_review_paths(language: str = "python") -> list[str]:
    """Get ignored paths for code review, combining language-specific and common paths.

    Args:
        language: The project language (python, typescript, javascript, go, rust).

    Returns:
        List of path prefixes/patterns to ignore during review.
    """
    language_paths = IGNORED_PATHS_BY_LANGUAGE.get(language, [])
    common_paths = IGNORED_PATHS_BY_LANGUAGE.get("common", [])
    return list(set(language_paths + common_paths))
