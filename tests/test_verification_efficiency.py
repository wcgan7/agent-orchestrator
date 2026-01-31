"""Test verification loop efficiency improvements.

These tests cover:
1. Granular PromptMode values for different verification failures
2. VerificationResult with stage and all_failing_paths
3. FSM storing detailed verification info in last_verification
4. FSM selecting the correct granular PromptMode
5. Stage-specific prompt builders with failing file lists
6. Minimal fix prompts for format/lint/typecheck issues
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from feature_prd_runner.fsm import _select_fix_prompt_mode, reduce_task
from feature_prd_runner.models import (
    PromptMode,
    TaskLifecycle,
    TaskState,
    TaskStep,
    VerificationResult,
)
from feature_prd_runner.prompts import (
    _build_minimal_allowlist_expansion_prompt,
    _build_minimal_fix_prompt,
    _build_minimal_review_fix_prompt,
    _build_phase_prompt,
)
from feature_prd_runner.utils import _now_iso


# --- Test PromptMode enum values ---


def test_prompt_mode_has_granular_fix_values() -> None:
    """Ensure PromptMode has granular values for different verification failures."""
    assert PromptMode.FIX_FORMAT.value == "fix_format"
    assert PromptMode.FIX_LINT.value == "fix_lint"
    assert PromptMode.FIX_TYPECHECK.value == "fix_typecheck"
    assert PromptMode.FIX_TESTS.value == "fix_tests"
    assert PromptMode.FIX_VERIFY.value == "fix_verify"


# --- Test VerificationResult model ---


def test_verification_result_has_stage_field() -> None:
    """Ensure VerificationResult includes stage field."""
    result = VerificationResult(
        run_id="run-1",
        passed=False,
        command="ruff check .",
        exit_code=1,
        log_path="/path/to/log",
        log_tail="error output",
        captured_at=_now_iso(),
        error_type="lint_failed",
        stage="lint",
        all_failing_paths=["src/foo.py", "src/bar.py"],
    )
    assert result.stage == "lint"
    assert result.all_failing_paths == ["src/foo.py", "src/bar.py"]


def test_verification_result_serialization() -> None:
    """Ensure VerificationResult serializes stage and all_failing_paths."""
    result = VerificationResult(
        run_id="run-1",
        passed=False,
        command="mypy .",
        exit_code=1,
        log_path="/path/to/log",
        log_tail="type errors",
        stage="typecheck",
        all_failing_paths=["src/types.py"],
    )
    data = result.to_dict()
    assert data["stage"] == "typecheck"
    assert data["all_failing_paths"] == ["src/types.py"]


# --- Test FSM _select_fix_prompt_mode function ---


def test_select_fix_prompt_mode_format() -> None:
    """Ensure format_failed maps to FIX_FORMAT."""
    assert _select_fix_prompt_mode("format_failed") == PromptMode.FIX_FORMAT


def test_select_fix_prompt_mode_lint() -> None:
    """Ensure lint_failed maps to FIX_LINT."""
    assert _select_fix_prompt_mode("lint_failed") == PromptMode.FIX_LINT


def test_select_fix_prompt_mode_typecheck() -> None:
    """Ensure typecheck_failed maps to FIX_TYPECHECK."""
    assert _select_fix_prompt_mode("typecheck_failed") == PromptMode.FIX_TYPECHECK


def test_select_fix_prompt_mode_tests() -> None:
    """Ensure tests_failed maps to FIX_TESTS."""
    assert _select_fix_prompt_mode("tests_failed") == PromptMode.FIX_TESTS


def test_select_fix_prompt_mode_unknown() -> None:
    """Ensure unknown error types default to FIX_VERIFY."""
    assert _select_fix_prompt_mode("unknown_error") == PromptMode.FIX_VERIFY
    assert _select_fix_prompt_mode(None) == PromptMode.FIX_VERIFY
    assert _select_fix_prompt_mode("") == PromptMode.FIX_VERIFY


# --- Test FSM reduce_task stores detailed verification info ---


def test_fsm_stores_verification_details_on_failure() -> None:
    """Ensure FSM stores error_type, stage, and all_failing_paths in last_verification."""
    task = TaskState(
        id="phase-1",
        type="implement",
        step=TaskStep.VERIFY,
        lifecycle=TaskLifecycle.READY,
    )
    event = VerificationResult(
        run_id="run-1",
        passed=False,
        command="ruff check .",
        exit_code=1,
        log_path="/path/to/lint.log",
        log_tail="src/foo.py:10:1: F401",
        captured_at=_now_iso(),
        error_type="lint_failed",
        stage="lint",
        all_failing_paths=["src/foo.py", "src/bar.py"],
    )
    caps = {"test_fail_attempts": 3}

    updated = reduce_task(task, event, caps=caps)

    assert updated.last_verification is not None
    assert updated.last_verification["error_type"] == "lint_failed"
    assert updated.last_verification["stage"] == "lint"
    assert updated.last_verification["all_failing_paths"] == ["src/foo.py", "src/bar.py"]


def test_fsm_selects_fix_lint_prompt_mode() -> None:
    """Ensure FSM sets prompt_mode to FIX_LINT for lint failures."""
    task = TaskState(
        id="phase-1",
        type="implement",
        step=TaskStep.VERIFY,
        lifecycle=TaskLifecycle.READY,
    )
    event = VerificationResult(
        run_id="run-1",
        passed=False,
        command="ruff check .",
        exit_code=1,
        log_path="/path/to/lint.log",
        log_tail="lint errors",
        error_type="lint_failed",
        stage="lint",
    )
    caps = {"test_fail_attempts": 3}

    updated = reduce_task(task, event, caps=caps)

    assert updated.step == TaskStep.IMPLEMENT
    assert updated.prompt_mode == PromptMode.FIX_LINT


def test_fsm_selects_fix_format_prompt_mode() -> None:
    """Ensure FSM sets prompt_mode to FIX_FORMAT for format failures."""
    task = TaskState(
        id="phase-1",
        type="implement",
        step=TaskStep.VERIFY,
        lifecycle=TaskLifecycle.READY,
    )
    event = VerificationResult(
        run_id="run-1",
        passed=False,
        command="ruff format --check .",
        exit_code=1,
        log_path="/path/to/format.log",
        log_tail="would reformat",
        error_type="format_failed",
        stage="format",
    )
    caps = {"test_fail_attempts": 3}

    updated = reduce_task(task, event, caps=caps)

    assert updated.step == TaskStep.IMPLEMENT
    assert updated.prompt_mode == PromptMode.FIX_FORMAT


def test_fsm_selects_fix_typecheck_prompt_mode() -> None:
    """Ensure FSM sets prompt_mode to FIX_TYPECHECK for typecheck failures."""
    task = TaskState(
        id="phase-1",
        type="implement",
        step=TaskStep.VERIFY,
        lifecycle=TaskLifecycle.READY,
    )
    event = VerificationResult(
        run_id="run-1",
        passed=False,
        command="mypy .",
        exit_code=1,
        log_path="/path/to/typecheck.log",
        log_tail="type errors",
        error_type="typecheck_failed",
        stage="typecheck",
    )
    caps = {"test_fail_attempts": 3}

    updated = reduce_task(task, event, caps=caps)

    assert updated.step == TaskStep.IMPLEMENT
    assert updated.prompt_mode == PromptMode.FIX_TYPECHECK


def test_fsm_selects_fix_tests_prompt_mode() -> None:
    """Ensure FSM sets prompt_mode to FIX_TESTS for test failures."""
    task = TaskState(
        id="phase-1",
        type="implement",
        step=TaskStep.VERIFY,
        lifecycle=TaskLifecycle.READY,
    )
    event = VerificationResult(
        run_id="run-1",
        passed=False,
        command="pytest",
        exit_code=1,
        log_path="/path/to/tests.log",
        log_tail="FAILED tests/test_foo.py",
        error_type="tests_failed",
        stage="tests",
    )
    caps = {"test_fail_attempts": 3}

    updated = reduce_task(task, event, caps=caps)

    assert updated.step == TaskStep.IMPLEMENT
    assert updated.prompt_mode == PromptMode.FIX_TESTS


# --- Test stage-specific prompt banners ---


def test_phase_prompt_fix_lint_banner() -> None:
    """Ensure phase prompt has lint-specific banner with failing files."""
    phase = {"id": "phase-1", "name": "Phase 1", "description": "Test phase"}
    task: dict[str, Any] = {"id": "phase-1", "context": []}
    last_verification = {
        "command": "ruff check .",
        "exit_code": 1,
        "log_path": "/path/to/lint.log",
        "log_tail": "src/foo.py:10:1: F401 `os` imported but unused",
        "error_type": "lint_failed",
        "stage": "lint",
        "all_failing_paths": ["src/foo.py", "src/bar.py"],
    }

    prompt = _build_phase_prompt(
        prd_path=Path("/path/to/prd.md"),
        phase=phase,
        task=task,
        events_path=Path("/path/to/events.jsonl"),
        progress_path=Path("/path/to/progress.json"),
        run_id="run-1",
        user_prompt=None,
        prompt_mode="fix_lint",
        last_verification=last_verification,
    )

    assert "LINT CHECK FAILING -- FIX THIS FIRST" in prompt
    assert "src/foo.py" in prompt
    assert "src/bar.py" in prompt
    assert "Fix ONLY the lint errors" in prompt


def test_phase_prompt_fix_format_banner() -> None:
    """Ensure phase prompt has format-specific banner with failing files."""
    phase = {"id": "phase-1", "name": "Phase 1", "description": "Test phase"}
    task: dict[str, Any] = {"id": "phase-1", "context": []}
    last_verification = {
        "command": "ruff format --check .",
        "exit_code": 1,
        "log_path": "/path/to/format.log",
        "log_tail": "Would reformat: src/foo.py",
        "error_type": "format_failed",
        "stage": "format",
        "all_failing_paths": ["src/foo.py"],
    }

    prompt = _build_phase_prompt(
        prd_path=Path("/path/to/prd.md"),
        phase=phase,
        task=task,
        events_path=Path("/path/to/events.jsonl"),
        progress_path=Path("/path/to/progress.json"),
        run_id="run-1",
        user_prompt=None,
        prompt_mode="fix_format",
        last_verification=last_verification,
    )

    assert "FORMAT CHECK FAILING -- FIX THIS FIRST" in prompt
    assert "src/foo.py" in prompt
    assert "Fix ONLY the formatting issues" in prompt


def test_phase_prompt_fix_typecheck_banner() -> None:
    """Ensure phase prompt has typecheck-specific banner with failing files."""
    phase = {"id": "phase-1", "name": "Phase 1", "description": "Test phase"}
    task: dict[str, Any] = {"id": "phase-1", "context": []}
    last_verification = {
        "command": "mypy .",
        "exit_code": 1,
        "log_path": "/path/to/typecheck.log",
        "log_tail": "src/foo.py:10: error: Incompatible types",
        "error_type": "typecheck_failed",
        "stage": "typecheck",
        "all_failing_paths": ["src/foo.py"],
    }

    prompt = _build_phase_prompt(
        prd_path=Path("/path/to/prd.md"),
        phase=phase,
        task=task,
        events_path=Path("/path/to/events.jsonl"),
        progress_path=Path("/path/to/progress.json"),
        run_id="run-1",
        user_prompt=None,
        prompt_mode="fix_typecheck",
        last_verification=last_verification,
    )

    assert "TYPE CHECK FAILING -- FIX THIS FIRST" in prompt
    assert "src/foo.py" in prompt
    assert "Fix ONLY the type errors" in prompt


def test_phase_prompt_fix_tests_banner() -> None:
    """Ensure phase prompt has test-specific banner allowing logic changes."""
    phase = {"id": "phase-1", "name": "Phase 1", "description": "Test phase"}
    task: dict[str, Any] = {"id": "phase-1", "context": []}
    last_verification = {
        "command": "pytest",
        "exit_code": 1,
        "log_path": "/path/to/tests.log",
        "log_tail": "FAILED tests/test_foo.py::test_bar",
        "error_type": "tests_failed",
        "stage": "tests",
        "all_failing_paths": ["tests/test_foo.py", "src/foo.py"],
    }

    prompt = _build_phase_prompt(
        prd_path=Path("/path/to/prd.md"),
        phase=phase,
        task=task,
        events_path=Path("/path/to/events.jsonl"),
        progress_path=Path("/path/to/progress.json"),
        run_id="run-1",
        user_prompt=None,
        prompt_mode="fix_tests",
        last_verification=last_verification,
    )

    assert "TESTS ARE FAILING -- FIX THIS FIRST" in prompt
    assert "tests/test_foo.py" in prompt
    assert "may require code logic changes" in prompt


# --- Test minimal fix prompt ---


def test_minimal_fix_prompt_lint() -> None:
    """Ensure minimal fix prompt for lint is focused and concise."""
    last_verification = {
        "command": "ruff check .",
        "exit_code": 1,
        "log_path": "/path/to/lint.log",
        "log_tail": "src/foo.py:10:1: F401 `os` imported but unused",
        "error_type": "lint_failed",
        "stage": "lint",
        "all_failing_paths": ["src/foo.py", "src/bar.py"],
    }

    prompt = _build_minimal_fix_prompt(
        prompt_mode="fix_lint",
        last_verification=last_verification,
        allowed_files=["src/foo.py", "src/bar.py"],
    )

    assert "LINT CHECK FAILING" in prompt
    assert "src/foo.py" in prompt
    assert "src/bar.py" in prompt
    assert "Fix ONLY the lint errors" in prompt
    assert "Do NOT modify code logic" in prompt
    # Ensure no PRD/acceptance criteria in minimal prompt
    assert "PRD:" not in prompt
    assert "acceptance criteria" not in prompt.lower()


def test_minimal_fix_prompt_format() -> None:
    """Ensure minimal fix prompt for format is focused and concise."""
    last_verification = {
        "command": "ruff format --check .",
        "exit_code": 1,
        "log_path": "/path/to/format.log",
        "log_tail": "Would reformat: src/foo.py",
        "error_type": "format_failed",
        "stage": "format",
        "all_failing_paths": ["src/foo.py"],
    }

    prompt = _build_minimal_fix_prompt(
        prompt_mode="fix_format",
        last_verification=last_verification,
        allowed_files=["src/foo.py"],
    )

    assert "FORMAT CHECK FAILING" in prompt
    assert "src/foo.py" in prompt
    assert "Fix ONLY the formatting issues" in prompt
    assert "Do NOT modify code logic" in prompt


def test_minimal_fix_prompt_typecheck() -> None:
    """Ensure minimal fix prompt for typecheck is focused and concise."""
    last_verification = {
        "command": "mypy .",
        "exit_code": 1,
        "log_path": "/path/to/typecheck.log",
        "log_tail": "src/foo.py:10: error: Incompatible types",
        "error_type": "typecheck_failed",
        "stage": "typecheck",
        "all_failing_paths": ["src/foo.py"],
    }

    prompt = _build_minimal_fix_prompt(
        prompt_mode="fix_typecheck",
        last_verification=last_verification,
        allowed_files=["src/foo.py"],
    )

    assert "TYPE CHECK FAILING" in prompt
    assert "src/foo.py" in prompt
    assert "Fix ONLY the type errors" in prompt
    assert "Do NOT modify code logic" in prompt


def test_minimal_fix_prompt_includes_progress_contract() -> None:
    """Ensure minimal fix prompt includes progress contract when provided."""
    last_verification = {
        "command": "ruff check .",
        "exit_code": 1,
        "log_path": "/path/to/lint.log",
        "log_tail": "lint errors",
        "all_failing_paths": ["src/foo.py"],
    }

    prompt = _build_minimal_fix_prompt(
        prompt_mode="fix_lint",
        last_verification=last_verification,
        progress_path=Path("/path/to/progress.json"),
        run_id="run-1",
        heartbeat_seconds=30,
    )

    assert "Progress contract" in prompt
    assert "run_id=run-1" in prompt
    assert "heartbeat" in prompt.lower()


# --- Test failing paths list truncation ---


def test_phase_prompt_truncates_long_failing_paths_list() -> None:
    """Ensure phase prompt truncates failing paths list if too long."""
    phase = {"id": "phase-1", "name": "Phase 1", "description": "Test phase"}
    task: dict[str, Any] = {"id": "phase-1", "context": []}
    # Create a list of 30 failing files
    failing_files = [f"src/file_{i}.py" for i in range(30)]
    last_verification = {
        "command": "ruff check .",
        "exit_code": 1,
        "log_path": "/path/to/lint.log",
        "log_tail": "many lint errors",
        "error_type": "lint_failed",
        "stage": "lint",
        "all_failing_paths": failing_files,
    }

    prompt = _build_phase_prompt(
        prd_path=Path("/path/to/prd.md"),
        phase=phase,
        task=task,
        events_path=Path("/path/to/events.jsonl"),
        progress_path=Path("/path/to/progress.json"),
        run_id="run-1",
        user_prompt=None,
        prompt_mode="fix_lint",
        last_verification=last_verification,
    )

    # Should show first 20 files
    assert "src/file_0.py" in prompt
    assert "src/file_19.py" in prompt
    # Should indicate more files exist
    assert "and 10 more files" in prompt
    # Should not show files beyond 20
    assert "src/file_25.py" not in prompt


# --- Test minimal allowlist expansion prompt ---


def test_minimal_allowlist_expansion_prompt_basic() -> None:
    """Ensure minimal expansion prompt is focused and concise."""
    phase = {"id": "phase-1", "name": "Phase 1"}
    current_allowlist = ["src/foo.py", "src/bar.py"]
    expansion_paths = ["tests/test_foo.py", "tests/test_bar.py"]

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=current_allowlist,
        expansion_paths=expansion_paths,
        error_type="typecheck_failed",
    )

    assert "UPDATE IMPLEMENTATION PLAN" in prompt
    assert "Allowlist Expansion" in prompt
    assert "tests/test_foo.py" in prompt
    assert "tests/test_bar.py" in prompt
    assert "type check errors" in prompt
    # Should NOT include PRD content
    assert "PRD content" not in prompt
    assert "Acceptance criteria" not in prompt


def test_minimal_allowlist_expansion_prompt_includes_current_files() -> None:
    """Ensure minimal expansion prompt shows current allowlist."""
    phase = {"id": "phase-1", "name": "Phase 1"}
    current_allowlist = ["src/existing.py"]
    expansion_paths = ["tests/new_test.py"]

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=current_allowlist,
        expansion_paths=expansion_paths,
    )

    assert "src/existing.py" in prompt
    assert "Current allowlist" in prompt


def test_minimal_allowlist_expansion_prompt_lint_context() -> None:
    """Ensure minimal expansion prompt has correct context for lint failures."""
    phase = {"id": "phase-1", "name": "Phase 1"}

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=[],
        expansion_paths=["src/foo.py"],
        error_type="lint_failed",
    )

    assert "lint errors" in prompt


def test_minimal_allowlist_expansion_prompt_format_context() -> None:
    """Ensure minimal expansion prompt has correct context for format failures."""
    phase = {"id": "phase-1", "name": "Phase 1"}

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=[],
        expansion_paths=["src/foo.py"],
        error_type="format_failed",
    )

    assert "formatting errors" in prompt


def test_minimal_allowlist_expansion_prompt_test_context() -> None:
    """Ensure minimal expansion prompt has correct context for test failures."""
    phase = {"id": "phase-1", "name": "Phase 1"}

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=[],
        expansion_paths=["tests/test_foo.py"],
        error_type="tests_failed",
    )

    assert "test failures" in prompt


def test_minimal_allowlist_expansion_prompt_with_log_tail() -> None:
    """Ensure minimal expansion prompt includes log excerpt when provided."""
    phase = {"id": "phase-1", "name": "Phase 1"}
    log_tail = "src/foo.py:10: error: Incompatible types"

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=[],
        expansion_paths=["src/foo.py"],
        error_type="typecheck_failed",
        log_tail=log_tail,
    )

    assert "Incompatible types" in prompt
    assert "verification output" in prompt


def test_minimal_allowlist_expansion_prompt_with_progress_contract() -> None:
    """Ensure minimal expansion prompt includes progress contract when provided."""
    phase = {"id": "phase-1", "name": "Phase 1"}

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=[],
        expansion_paths=["src/foo.py"],
        progress_path=Path("/path/to/progress.json"),
        run_id="run-1",
        heartbeat_seconds=30,
    )

    assert "Progress contract" in prompt
    assert "run_id=run-1" in prompt


def test_minimal_allowlist_expansion_prompt_truncates_large_allowlist() -> None:
    """Ensure minimal expansion prompt truncates large current allowlist."""
    phase = {"id": "phase-1", "name": "Phase 1"}
    # Create a list of 60 files
    current_allowlist = [f"src/file_{i}.py" for i in range(60)]

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=current_allowlist,
        expansion_paths=["tests/test_foo.py"],
    )

    # Should show first 50 files
    assert "src/file_0.py" in prompt
    assert "src/file_49.py" in prompt
    # Should indicate more files exist
    assert "and 10 more files" in prompt
    # Should not show files beyond 50
    assert "src/file_55.py" not in prompt


# --- Test minimal review fix prompt ---


def test_minimal_review_fix_prompt_basic() -> None:
    """Ensure minimal review fix prompt is focused and concise."""
    review_blockers = [
        "Function foo() is missing error handling",
        "Missing docstring for class Bar",
    ]
    review_blocker_files = ["src/foo.py", "src/bar.py"]

    prompt = _build_minimal_review_fix_prompt(
        review_blockers=review_blockers,
        review_blocker_files=review_blocker_files,
        allowed_files=["src/foo.py", "src/bar.py"],
    )

    assert "ADDRESS REVIEW BLOCKERS" in prompt
    assert "missing error handling" in prompt
    assert "Missing docstring" in prompt
    assert "src/foo.py" in prompt
    assert "src/bar.py" in prompt
    # Should NOT include PRD content
    assert "PRD content" not in prompt
    assert "Acceptance criteria" not in prompt


def test_minimal_review_fix_prompt_no_prd() -> None:
    """Ensure minimal review fix prompt does not include PRD references."""
    prompt = _build_minimal_review_fix_prompt(
        review_blockers=["Fix the bug"],
        review_blocker_files=["src/bug.py"],
    )

    assert "PRD:" not in prompt
    assert "PRD content" not in prompt
    assert "Phase:" not in prompt


def test_minimal_review_fix_prompt_with_progress_contract() -> None:
    """Ensure minimal review fix prompt includes progress contract when provided."""
    prompt = _build_minimal_review_fix_prompt(
        review_blockers=["Fix issue"],
        review_blocker_files=["src/foo.py"],
        progress_path=Path("/path/to/progress.json"),
        run_id="run-1",
        heartbeat_seconds=30,
    )

    assert "Progress contract" in prompt
    assert "run_id=run-1" in prompt
    assert "heartbeat" in prompt.lower()


def test_minimal_review_fix_prompt_includes_allowed_files() -> None:
    """Ensure minimal review fix prompt includes allowed files section."""
    allowed = ["src/foo.py", "src/bar.py", "tests/test_foo.py"]

    prompt = _build_minimal_review_fix_prompt(
        review_blockers=["Fix bug"],
        review_blocker_files=["src/foo.py"],
        allowed_files=allowed,
    )

    assert "Allowed files to edit" in prompt
    assert "src/foo.py" in prompt
    assert "src/bar.py" in prompt
    assert "tests/test_foo.py" in prompt


def test_minimal_review_fix_prompt_truncates_large_allowed_list() -> None:
    """Ensure minimal review fix prompt truncates large allowed files list."""
    allowed = [f"src/file_{i}.py" for i in range(40)]

    prompt = _build_minimal_review_fix_prompt(
        review_blockers=["Fix bug"],
        review_blocker_files=["src/foo.py"],
        allowed_files=allowed,
    )

    # Should show first 30 files
    assert "src/file_0.py" in prompt
    assert "src/file_29.py" in prompt
    # Should indicate more files exist
    assert "and 10 more files" in prompt
    # Should not show files beyond 30
    assert "src/file_35.py" not in prompt


def test_minimal_review_fix_prompt_instructions() -> None:
    """Ensure minimal review fix prompt has focused instructions."""
    prompt = _build_minimal_review_fix_prompt(
        review_blockers=["Fix the bug"],
        review_blocker_files=["src/foo.py"],
    )

    assert "Fix ONLY the blocking issues" in prompt
    assert "minimal changes" in prompt
    assert "Do NOT add new features" in prompt


# --- Edge case tests ---


def test_minimal_fix_prompt_handles_empty_verification() -> None:
    """Ensure minimal fix prompt handles empty/None verification gracefully."""
    prompt = _build_minimal_fix_prompt(
        prompt_mode="fix_lint",
        last_verification={},  # Empty verification
        allowed_files=["src/foo.py"],
    )

    assert "LINT CHECK FAILING" in prompt
    assert "(unknown)" in prompt  # Command should show unknown
    assert "src/foo.py" in prompt  # Allowed files should still appear


def test_minimal_fix_prompt_handles_missing_fields() -> None:
    """Ensure minimal fix prompt handles missing verification fields."""
    prompt = _build_minimal_fix_prompt(
        prompt_mode="fix_typecheck",
        last_verification={
            "command": "mypy .",
            # Missing: exit_code, log_path, log_tail, all_failing_paths
        },
        allowed_files=["src/foo.py"],
    )

    assert "TYPE CHECK FAILING" in prompt
    assert "mypy ." in prompt
    # Should handle missing fields gracefully
    assert "(no output captured)" in prompt or "log_tail" not in prompt.lower()


def test_minimal_review_fix_prompt_handles_empty_files() -> None:
    """Ensure minimal review fix prompt handles empty blocker files list."""
    prompt = _build_minimal_review_fix_prompt(
        review_blockers=["Fix the security issue"],
        review_blocker_files=[],  # Empty files list
        allowed_files=["src/foo.py"],
    )

    assert "ADDRESS REVIEW BLOCKERS" in prompt
    assert "security issue" in prompt
    # Should not crash, should render allowed files
    assert "src/foo.py" in prompt


def test_minimal_allowlist_expansion_handles_empty_allowlist() -> None:
    """Ensure minimal expansion prompt handles empty current allowlist."""
    phase = {"id": "phase-1", "name": "Phase 1"}

    prompt = _build_minimal_allowlist_expansion_prompt(
        phase=phase,
        impl_plan_path=Path("/path/to/plan.json"),
        current_allowlist=[],  # Empty current allowlist
        expansion_paths=["tests/test_foo.py"],
    )

    assert "UPDATE IMPLEMENTATION PLAN" in prompt
    assert "tests/test_foo.py" in prompt
    # Should not crash with empty allowlist


def test_minimal_fix_prompt_with_none_allowed_files() -> None:
    """Ensure minimal fix prompt handles None allowed_files."""
    prompt = _build_minimal_fix_prompt(
        prompt_mode="fix_format",
        last_verification={"command": "ruff format --check .", "all_failing_paths": ["src/foo.py"]},
        allowed_files=None,  # None instead of list
    )

    assert "FORMAT CHECK FAILING" in prompt
    # Should not crash


def test_fsm_fix_verify_fallback_for_unknown_error() -> None:
    """Ensure FSM uses FIX_VERIFY fallback for unknown error types."""
    # This tests the _select_fix_prompt_mode function edge cases
    assert _select_fix_prompt_mode("some_random_error") == PromptMode.FIX_VERIFY
    assert _select_fix_prompt_mode("") == PromptMode.FIX_VERIFY
    assert _select_fix_prompt_mode(None) == PromptMode.FIX_VERIFY
    assert _select_fix_prompt_mode("  ") == PromptMode.FIX_VERIFY  # Whitespace only
