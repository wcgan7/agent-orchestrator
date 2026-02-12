"""Tests for LiveWorkerAdapter — real worker dispatch, no silent fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from feature_prd_runner.v3.domain.models import Task
from feature_prd_runner.v3.orchestrator.live_worker_adapter import (
    LiveWorkerAdapter,
    _extract_json,
    build_step_prompt,
)
from feature_prd_runner.v3.orchestrator.worker_adapter import StepResult
from feature_prd_runner.v3.storage.container import V3Container
from feature_prd_runner.workers.config import WorkerProviderSpec
from feature_prd_runner.workers.run import WorkerRunResult


@pytest.fixture()
def container(tmp_path: Path) -> V3Container:
    return V3Container(tmp_path)


@pytest.fixture()
def adapter(container: V3Container) -> LiveWorkerAdapter:
    return LiveWorkerAdapter(container)


def _make_task(**kwargs) -> Task:
    defaults = dict(title="Test task", description="Do the thing", task_type="feature")
    defaults.update(kwargs)
    return Task(**defaults)


def _make_run_result(
    *,
    exit_code: int = 0,
    timed_out: bool = False,
    response_text: str = "",
) -> WorkerRunResult:
    return WorkerRunResult(
        provider="test",
        prompt_path="/tmp/claude/prompt.txt",
        stdout_path="/tmp/claude/stdout.log",
        stderr_path="/tmp/claude/stderr.log",
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-01T00:01:00Z",
        runtime_seconds=60,
        exit_code=exit_code,
        timed_out=timed_out,
        no_heartbeat=False,
        response_text=response_text,
    )


_CODEX_SPEC = WorkerProviderSpec(name="codex", type="codex", command="codex")
_OLLAMA_SPEC = WorkerProviderSpec(
    name="local", type="ollama", endpoint="http://localhost:11434", model="llama3"
)


# ---------------------------------------------------------------------------
# 1. Error when no worker is available
# ---------------------------------------------------------------------------


def test_error_when_no_worker_available(adapter: LiveWorkerAdapter) -> None:
    """When worker is not available, return error — never silently succeed."""
    task = _make_task()

    with patch(
        "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
        return_value=(False, "Executable not found in PATH: codex"),
    ):
        result = adapter.run_step(task=task, step="implement", attempt=1)

    assert result.status == "error"
    assert "not available" in (result.summary or "").lower()


def test_error_when_worker_cannot_be_resolved(adapter: LiveWorkerAdapter) -> None:
    """When worker resolution raises, return error with details."""
    task = _make_task()

    with patch(
        "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config",
        side_effect=ValueError("No workers section in config"),
    ):
        result = adapter.run_step(task=task, step="implement", attempt=1)

    assert result.status == "error"
    assert "cannot resolve" in (result.summary or "").lower()


# ---------------------------------------------------------------------------
# 2. Codex success maps to ok
# ---------------------------------------------------------------------------


def test_codex_success_maps_to_ok(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)

    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "ok"


# ---------------------------------------------------------------------------
# 3. Codex failure maps to error
# ---------------------------------------------------------------------------


def test_codex_failure_maps_to_error(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=1)

    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "error"
    assert "code 1" in (result.summary or "")


# ---------------------------------------------------------------------------
# 4. Codex timeout maps to error
# ---------------------------------------------------------------------------


def test_codex_timeout_maps_to_error(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=124, timed_out=True)

    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "error"
    assert "timed out" in (result.summary or "").lower()


# ---------------------------------------------------------------------------
# 5. Ollama review parses findings
# ---------------------------------------------------------------------------


def test_ollama_review_parses_findings(adapter: LiveWorkerAdapter) -> None:
    response_json = '{"findings": [{"severity": "high", "category": "security", "summary": "SQL injection"}]}'
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="review", attempt=1)

    assert result.status == "ok"
    assert result.findings is not None
    assert len(result.findings) == 1
    assert result.findings[0]["severity"] == "high"
    assert result.findings[0]["summary"] == "SQL injection"


# ---------------------------------------------------------------------------
# 6. Ollama generate_tasks parses tasks
# ---------------------------------------------------------------------------


def test_ollama_generate_tasks_parses_tasks(adapter: LiveWorkerAdapter) -> None:
    response_json = '{"tasks": [{"title": "Add tests", "description": "Write unit tests", "task_type": "feature", "priority": "P1"}]}'
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="generate_tasks", attempt=1)

    assert result.status == "ok"
    assert result.generated_tasks is not None
    assert len(result.generated_tasks) == 1
    assert result.generated_tasks[0]["title"] == "Add tests"


# ---------------------------------------------------------------------------
# 7. Ollama implement extracts summary
# ---------------------------------------------------------------------------


def test_ollama_implement_extracts_summary(adapter: LiveWorkerAdapter) -> None:
    response_json = '{"patch": "--- a/file.py\\n+++ b/file.py", "summary": "Added error handling"}'
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "ok"
    assert result.summary == "Added error handling"


# ---------------------------------------------------------------------------
# 8. Prompt includes task context
# ---------------------------------------------------------------------------


def test_prompt_includes_task_context() -> None:
    task = _make_task(title="Fix login bug", description="Users cannot log in", task_type="bugfix")
    prompt = build_step_prompt(task=task, step="implement", attempt=1, is_codex=True)

    assert "Fix login bug" in prompt
    assert "Users cannot log in" in prompt
    assert "bugfix" in prompt


# ---------------------------------------------------------------------------
# 9. Prompt includes review findings for fix steps
# ---------------------------------------------------------------------------


def test_prompt_includes_review_findings_for_fix() -> None:
    findings = [
        {"severity": "high", "summary": "Missing null check", "file": "auth.py", "line": 42}
    ]
    task = _make_task(metadata={"review_findings": findings})
    prompt = build_step_prompt(task=task, step="implement_fix", attempt=2, is_codex=True)

    assert "Missing null check" in prompt
    assert "auth.py" in prompt
    assert "Attempt: 2" in prompt


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def test_extract_json_plain() -> None:
    assert _extract_json('{"status": "ok"}') == {"status": "ok"}


def test_extract_json_with_markdown_fences() -> None:
    text = "```json\n{\"status\": \"ok\"}\n```"
    assert _extract_json(text) == {"status": "ok"}


def test_extract_json_with_surrounding_text() -> None:
    text = "Here is the result: {\"status\": \"ok\"} -- done"
    assert _extract_json(text) == {"status": "ok"}


def test_extract_json_invalid_returns_none() -> None:
    assert _extract_json("no json here") is None


# ---------------------------------------------------------------------------
# Prompt adds JSON schema suffix for ollama but not codex
# ---------------------------------------------------------------------------


def test_prompt_ollama_includes_json_schema() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="review", attempt=1, is_codex=False)
    assert "Respond with valid JSON" in prompt
    assert "findings" in prompt


def test_prompt_codex_no_json_schema() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="review", attempt=1, is_codex=True)
    assert "Respond with valid JSON" not in prompt


# ---------------------------------------------------------------------------
# Worker execution exception returns error
# ---------------------------------------------------------------------------


def test_worker_exception_returns_error(adapter: LiveWorkerAdapter) -> None:
    """When run_worker raises, return error — don't silently succeed."""
    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            side_effect=RuntimeError("Codex crashed"),
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "error"
    assert "Codex crashed" in (result.summary or "")


# ---------------------------------------------------------------------------
# Ollama verify step maps pass/fail
# ---------------------------------------------------------------------------


def test_ollama_verify_pass(adapter: LiveWorkerAdapter) -> None:
    response_json = '{"status": "pass", "summary": "All tests passed"}'
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="verify", attempt=1)

    assert result.status == "ok"
    assert result.summary == "All tests passed"


def test_ollama_verify_fail(adapter: LiveWorkerAdapter) -> None:
    response_json = '{"status": "fail", "summary": "3 tests failed"}'
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "feature_prd_runner.v3.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="verify", attempt=1)

    assert result.status == "error"
    assert result.summary == "3 tests failed"
