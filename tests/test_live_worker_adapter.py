"""Tests for LiveWorkerAdapter — real worker dispatch, no silent fallback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.orchestrator.live_worker_adapter import (
    LiveWorkerAdapter,
    _extract_json,
    _extract_json_value,
    _normalize_planning_text,
    build_step_prompt,
    detect_project_languages,
)
from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult
from agent_orchestrator.runtime.storage.container import Container
from agent_orchestrator.workers.config import WorkerProviderSpec
from agent_orchestrator.workers.run import WorkerRunResult


@pytest.fixture()
def container(tmp_path: Path) -> Container:
    return Container(tmp_path)


@pytest.fixture()
def adapter(container: Container) -> LiveWorkerAdapter:
    return LiveWorkerAdapter(container)


def _make_task(**kwargs) -> Task:
    defaults = dict(title="Test task", description="Do the thing", task_type="feature")
    defaults.update(kwargs)
    return Task(**defaults)


def _make_run_result(
    *,
    exit_code: int = 0,
    timed_out: bool = False,
    no_heartbeat: bool = False,
    response_text: str = "",
    human_blocking_issues: list[dict[str, str]] | None = None,
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
        no_heartbeat=no_heartbeat,
        response_text=response_text,
        human_blocking_issues=list(human_blocking_issues or []),
    )


_CODEX_SPEC = WorkerProviderSpec(name="codex", type="codex", command="codex")
_CLAUDE_SPEC = WorkerProviderSpec(name="claude", type="claude", command="claude -p")
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
        "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
        return_value=(False, "Executable not found in PATH: codex"),
    ):
        result = adapter.run_step(task=task, step="implement", attempt=1)

    assert result.status == "error"
    assert "not available" in (result.summary or "").lower()


def test_error_when_worker_cannot_be_resolved(adapter: LiveWorkerAdapter) -> None:
    """When worker resolution raises, return error with details."""
    task = _make_task()

    with patch(
        "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config",
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
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "ok"


# ---------------------------------------------------------------------------
# 3. Codex model selection prefers task override over runtime default
# ---------------------------------------------------------------------------


def test_codex_model_prefers_task_override(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)
    runtime = SimpleNamespace(default_model="gpt-5-codex-default")
    task = _make_task(worker_model="gpt-5-codex-task")

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config",
            return_value=runtime,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ) as run_worker_mock,
    ):
        result = adapter.run_step(task=task, step="implement", attempt=1)

    assert result.status == "ok"
    assert run_worker_mock.call_args.kwargs["spec"].model == "gpt-5-codex-task"


# ---------------------------------------------------------------------------
# 4. Codex model falls back to runtime default when task override absent
# ---------------------------------------------------------------------------


def test_codex_model_falls_back_to_runtime_default(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)
    runtime = SimpleNamespace(default_model="gpt-5-codex-default")

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config",
            return_value=runtime,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ) as run_worker_mock,
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "ok"
    assert run_worker_mock.call_args.kwargs["spec"].model == "gpt-5-codex-default"


# ---------------------------------------------------------------------------
# 5. Claude prompt mode matches codex mode (no schema injection)
# ---------------------------------------------------------------------------


def test_claude_uses_codex_prompt_mode(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CLAUDE_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.build_step_prompt",
            return_value="prompt",
        ) as build_prompt_mock,
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "ok"
    assert build_prompt_mock.call_args.kwargs["is_codex"] is True


# ---------------------------------------------------------------------------
# 6. Step timeout can be overridden per task metadata
# ---------------------------------------------------------------------------


def test_timeout_override_from_task_metadata(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)
    task = _make_task(metadata={"step_timeouts": {"implement": 42}})

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ) as run_worker_mock,
    ):
        result = adapter.run_step(task=task, step="implement", attempt=1)

    assert result.status == "ok"
    assert run_worker_mock.call_args.kwargs["timeout_seconds"] == 42


# ---------------------------------------------------------------------------
# 7. Template timeout is used when no metadata override exists
# ---------------------------------------------------------------------------


def test_timeout_defaults_from_pipeline_template(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)
    task = _make_task(task_type="bug")

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ) as run_worker_mock,
    ):
        result = adapter.run_step(task=task, step="reproduce", attempt=1)

    assert result.status == "ok"
    assert run_worker_mock.call_args.kwargs["timeout_seconds"] == 300


def test_heartbeat_defaults_are_forwarded(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ) as run_worker_mock,
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "ok"
    assert run_worker_mock.call_args.kwargs["heartbeat_seconds"] == 60
    assert run_worker_mock.call_args.kwargs["heartbeat_grace_seconds"] == 240


def test_heartbeat_settings_from_workers_config(container: Container, adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)
    cfg = container.config.load()
    cfg["workers"] = {"heartbeat_seconds": 75, "heartbeat_grace_seconds": 190}
    container.config.save(cfg)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ) as run_worker_mock,
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "ok"
    assert run_worker_mock.call_args.kwargs["heartbeat_seconds"] == 75
    assert run_worker_mock.call_args.kwargs["heartbeat_grace_seconds"] == 190


# ---------------------------------------------------------------------------
# 8. Codex failure maps to error
# ---------------------------------------------------------------------------


def test_codex_failure_maps_to_error(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=1)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "error"
    assert "code 1" in (result.summary or "")


# ---------------------------------------------------------------------------
# 9. Codex timeout maps to error
# ---------------------------------------------------------------------------


def test_codex_timeout_maps_to_error(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=124, timed_out=True)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "error"
    assert "timed out" in (result.summary or "").lower()


# ---------------------------------------------------------------------------
# 10. Codex no-heartbeat maps to explicit stall error
# ---------------------------------------------------------------------------


def test_codex_no_heartbeat_maps_to_stalled_error(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=1, no_heartbeat=True)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "error"
    assert "stalled" in (result.summary or "").lower()


# ---------------------------------------------------------------------------
# 8. Human-blocking issues map to dedicated status
# ---------------------------------------------------------------------------


def test_human_blocking_issues_map_to_human_blocked(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(
        exit_code=0,
        human_blocking_issues=[{"summary": "Need production API token"}],
    )

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="implement", attempt=1)

    assert result.status == "human_blocked"
    assert result.human_blocking_issues is not None
    assert result.human_blocking_issues[0]["summary"] == "Need production API token"
    assert "human intervention required" in (result.summary or "").lower()


# ---------------------------------------------------------------------------
# 9. Ollama review parses findings
# ---------------------------------------------------------------------------


def test_ollama_review_parses_findings(adapter: LiveWorkerAdapter) -> None:
    response_json = '{"findings": [{"severity": "high", "category": "security", "summary": "SQL injection"}]}'
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="review", attempt=1)

    assert result.status == "ok"
    assert result.findings is not None
    assert len(result.findings) == 1
    assert result.findings[0]["severity"] == "high"
    assert result.findings[0]["summary"] == "SQL injection"
    assert result.findings[0]["category"] == "security"


def test_ollama_review_normalizes_findings(adapter: LiveWorkerAdapter) -> None:
    response_json = (
        '{"findings": ['
        '{"severity": "info", "category": "style", "summary": "Looks great"},'
        '{"severity": "urgent", "category": "bug", "summary": "Breaks output", "line": "0"}'
        ']}'
    )
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="review", attempt=1)

    assert result.status == "ok"
    assert result.findings is not None
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["severity"] == "medium"  # unknown -> medium
    assert finding["category"] == "other"   # unknown -> other
    assert finding["line"] is None          # non-positive -> None
    assert finding["status"] == "open"      # default


# ---------------------------------------------------------------------------
# 10. Ollama generate_tasks parses tasks
# ---------------------------------------------------------------------------


def test_ollama_generate_tasks_parses_tasks(adapter: LiveWorkerAdapter) -> None:
    response_json = '{"tasks": [{"title": "Add tests", "description": "Write unit tests", "task_type": "feature", "priority": "P1"}]}'
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="generate_tasks", attempt=1)

    assert result.status == "ok"
    assert result.generated_tasks is not None
    assert len(result.generated_tasks) == 1
    assert result.generated_tasks[0]["title"] == "Add tests"


# ---------------------------------------------------------------------------
# 11. Ollama implement extracts summary
# ---------------------------------------------------------------------------


def test_ollama_implement_extracts_summary(adapter: LiveWorkerAdapter) -> None:
    response_json = '{"patch": "--- a/file.py\\n+++ b/file.py", "summary": "Added error handling"}'
    run_result = _make_run_result(exit_code=0, response_text=response_json)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
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
# 8b. implement prompt keeps task context minimal
# ---------------------------------------------------------------------------


def test_implement_prompt_omits_priority_step_attempt() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="implement", attempt=2, is_codex=True)

    assert "Priority:" not in prompt
    assert "Step:" not in prompt
    assert "Attempt:" not in prompt


def test_review_prompt_omits_priority_and_step() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="review", attempt=2, is_codex=True)

    assert "Priority:" not in prompt
    assert "Step:" not in prompt
    assert "Attempt: 2" in prompt


def test_verify_prompt_omits_priority_and_step() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="verify", attempt=2, is_codex=True)

    assert "Priority:" not in prompt
    assert "Step:" not in prompt
    assert "Attempt: 2" in prompt


def test_diagnose_prompt_omits_priority_and_step() -> None:
    task = _make_task(task_type="bug")
    prompt = build_step_prompt(task=task, step="diagnose", attempt=1, is_codex=True)

    assert "Priority:" not in prompt
    assert "Step:" not in prompt


def test_implement_prompt_omits_review_history() -> None:
    task = _make_task(metadata={
        "review_history": [
            {"attempt": 1, "decision": "changes_requested", "findings": [{"severity": "high", "summary": "x", "status": "open"}]}
        ]
    })
    prompt = build_step_prompt(task=task, step="implement", attempt=1, is_codex=True)

    assert "## Previous review cycle history" not in prompt


# ---------------------------------------------------------------------------
# 9. implement_fix prompt includes remediation payload
# ---------------------------------------------------------------------------


def test_prompt_fix_includes_review_findings_injection() -> None:
    findings = [
        {"severity": "high", "summary": "Missing null check", "file": "auth.py", "line": 42}
    ]
    task = _make_task(metadata={"review_findings": findings})
    prompt = build_step_prompt(task=task, step="implement_fix", attempt=2, is_codex=True)

    assert "## Issues to fix" in prompt
    assert "### Review findings to address" in prompt
    assert "Missing null check" in prompt
    assert "auth.py" in prompt
    assert "Priority:" not in prompt
    assert "Step:" not in prompt
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


def test_extract_json_value_array_from_markdown_fence() -> None:
    text = "```json\n[{\"title\":\"A\"}]\n```"
    parsed = _extract_json_value(text)
    assert isinstance(parsed, list)
    assert parsed[0]["title"] == "A"


def test_normalize_planning_text_strips_wrapper_and_followup() -> None:
    text = (
        "The plan has been prepared. Here's a summary of the key refinement.\n\n"
        "## Refined Plan\n"
        "- Keep timers as absolute timestamps.\n"
        "- Recompute on visibility change.\n\n"
        "---\n\n"
        "Should I write the full PLAN.md to the repo?"
    )
    cleaned = _normalize_planning_text(text)
    assert cleaned.startswith("## Refined Plan")
    assert "The plan has been prepared" not in cleaned
    assert "Should I write the full PLAN.md" not in cleaned


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


def test_prompt_planning_includes_shared_output_format() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="plan_refine", attempt=1, is_codex=True)
    assert "Output format (use these exact section headings)" in prompt
    assert "## Objective" in prompt
    assert "## Verification Plan" in prompt
    assert "Return the full rewritten plan, not a delta summary." in prompt


def test_prompt_ollama_diagnose_includes_diagnosis_schema() -> None:
    task = _make_task(task_type="bug")
    prompt = build_step_prompt(task=task, step="diagnose", attempt=1, is_codex=False)
    assert "Respond with valid JSON" in prompt
    assert '"diagnosis"' in prompt


# ---------------------------------------------------------------------------
# Worker execution exception returns error
# ---------------------------------------------------------------------------


def test_worker_exception_returns_error(adapter: LiveWorkerAdapter) -> None:
    """When run_worker raises, return error — don't silently succeed."""
    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
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
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
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
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_OLLAMA_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="verify", attempt=1)

    assert result.status == "error"
    assert result.summary == "3 tests failed"


def test_codex_verify_formatter_includes_reason_code_on_fail(adapter: LiveWorkerAdapter) -> None:
    raw_result = _make_run_result(exit_code=0, response_text="pytest output")
    fmt_result = _make_run_result(
        exit_code=0,
        response_text='{"status":"fail","reason_code":"type_error","summary":"mypy failed"}',
    )

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            side_effect=[raw_result, fmt_result],
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="verify", attempt=1)

    assert result.status == "error"
    assert result.summary is not None
    assert "mypy failed" in result.summary
    assert "reason_code=type_error" in result.summary


def test_codex_verify_formatter_environment_sets_note_with_reason(adapter: LiveWorkerAdapter) -> None:
    task = _make_task()
    raw_result = _make_run_result(exit_code=0, response_text="pytest output")
    fmt_result = _make_run_result(
        exit_code=0,
        response_text='{"status":"environment","reason_code":"permission_denied","summary":"Semaphore blocked"}',
    )

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            side_effect=[raw_result, fmt_result],
        ),
    ):
        result = adapter.run_step(task=task, step="verify", attempt=1)

    assert result.status == "ok"
    assert result.summary is not None
    assert "Semaphore blocked" in result.summary
    assert "reason_code=permission_denied" in result.summary
    assert task.metadata.get("verify_environment_note") == result.summary
    assert task.metadata.get("verify_reason_code") == "permission_denied"


def test_task_generation_parses_top_level_array(adapter: LiveWorkerAdapter) -> None:
    response = "```json\n[{\"title\":\"Task A\",\"task_type\":\"feature\",\"priority\":\"P1\"}]\n```"
    run_result = _make_run_result(exit_code=0, response_text=response)

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            return_value=run_result,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="generate_tasks", attempt=1)

    assert result.status == "ok"
    assert isinstance(result.generated_tasks, list)
    assert result.generated_tasks[0]["title"] == "Task A"


# ---------------------------------------------------------------------------
# Preamble and guardrails in prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step", ["plan", "implement", "review", "verify", "report"])
def test_prompt_includes_preamble(step: str) -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step=step, attempt=1, is_codex=True)
    assert "autonomous coding agent" in prompt
    assert "coordinator" in prompt
    assert "human-blocking issue" in prompt


@pytest.mark.parametrize("step", ["plan", "implement", "review", "verify", "report"])
def test_prompt_includes_guardrails(step: str) -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step=step, attempt=1, is_codex=True)
    assert "Do NOT commit" in prompt
    assert ".agent_orchestrator/" in prompt
    assert "suppress" in prompt.lower()


# ---------------------------------------------------------------------------
# Expanded category instructions
# ---------------------------------------------------------------------------


def test_planning_prompt_has_planning_rules() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="plan", attempt=1, is_codex=True)
    assert "independently testable" in prompt
    assert "does not modify" in prompt.lower()


def test_implementation_prompt_has_impl_rules() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="implement", attempt=1, is_codex=True)
    assert "Complete the step end-to-end" in prompt
    assert "source of truth" in prompt


def test_review_prompt_has_severity_guidance() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="review", attempt=1, is_codex=True)
    assert "severity" in prompt.lower()
    assert "acceptance criterion" in prompt.lower()
    assert "do not speculate" in prompt.lower()


def test_verification_prompt_has_testing_rules() -> None:
    task = _make_task()
    prompt = build_step_prompt(task=task, step="verify", attempt=1, is_codex=True)
    assert "test" in prompt.lower()
    assert "Do not fabricate results" in prompt
    assert "Do not silently skip checks" in prompt


# ---------------------------------------------------------------------------
# Dependency analysis prompt includes preamble and guardrails
# ---------------------------------------------------------------------------


def test_dep_analysis_prompt_includes_preamble_and_guardrails() -> None:
    task = _make_task(
        metadata={
            "candidate_tasks": [
                {"id": "t1", "title": "Add auth", "task_type": "feature"},
            ],
        },
    )
    prompt = build_step_prompt(task=task, step="analyze_deps", attempt=1, is_codex=True)
    assert "autonomous coding agent" in prompt
    assert "Do NOT commit" in prompt


# ---------------------------------------------------------------------------
# Language standards injection
# ---------------------------------------------------------------------------


def test_implementation_prompt_includes_python_standards() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="implement", attempt=1, is_codex=True,
        project_languages=["python"],
    )
    assert "Style guidelines" in prompt
    assert "Python defaults" in prompt


def test_review_prompt_includes_typescript_standards() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="review", attempt=1, is_codex=True,
        project_languages=["typescript"],
    )
    assert "Style guidelines" in prompt
    assert "TypeScript defaults" in prompt


def test_prompt_no_language_when_none() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="implement", attempt=1, is_codex=True, project_languages=None,
    )
    assert "Python defaults" not in prompt


def test_language_not_injected_for_planning() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="plan", attempt=1, is_codex=True,
        project_languages=["python"],
    )
    assert "Style guidelines" not in prompt


def test_mixed_language_prompt_includes_all_standards() -> None:
    """Full-stack repo: both Python and TypeScript defaults are injected."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="implement", attempt=1, is_codex=True,
        project_languages=["python", "typescript"],
    )
    assert "Python defaults" in prompt
    assert "TypeScript defaults" in prompt


# ---------------------------------------------------------------------------
# detect_project_languages
# ---------------------------------------------------------------------------


def test_detect_languages_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[build-system]")
    assert detect_project_languages(tmp_path) == ["python"]


def test_detect_languages_typescript(tmp_path: Path) -> None:
    (tmp_path / "tsconfig.json").write_text("{}")
    assert detect_project_languages(tmp_path) == ["typescript"]


def test_detect_languages_go(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example")
    assert detect_project_languages(tmp_path) == ["go"]


def test_detect_languages_rust(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]")
    assert detect_project_languages(tmp_path) == ["rust"]


def test_detect_languages_none(tmp_path: Path) -> None:
    assert detect_project_languages(tmp_path) == []


def test_detect_languages_mixed_python_typescript(tmp_path: Path) -> None:
    """Full-stack repo with both pyproject.toml and tsconfig.json."""
    (tmp_path / "pyproject.toml").write_text("[build-system]")
    (tmp_path / "tsconfig.json").write_text("{}")
    (tmp_path / "package.json").write_text("{}")
    langs = detect_project_languages(tmp_path)
    assert "python" in langs
    assert "typescript" in langs
    # JavaScript is subsumed by TypeScript
    assert "javascript" not in langs


def test_detect_languages_deduplicates(tmp_path: Path) -> None:
    """Both pyproject.toml and setup.py map to python — no duplicates."""
    (tmp_path / "pyproject.toml").write_text("[build-system]")
    (tmp_path / "setup.py").write_text("from setuptools import setup")
    assert detect_project_languages(tmp_path) == ["python"]


def test_detect_languages_typescript_subsumes_javascript(tmp_path: Path) -> None:
    """When tsconfig.json is present, package.json's 'javascript' is dropped."""
    (tmp_path / "tsconfig.json").write_text("{}")
    (tmp_path / "package.json").write_text("{}")
    assert detect_project_languages(tmp_path) == ["typescript"]


def test_detect_languages_javascript_alone(tmp_path: Path) -> None:
    """package.json without tsconfig.json → javascript."""
    (tmp_path / "package.json").write_text("{}")
    assert detect_project_languages(tmp_path) == ["javascript"]


# ---------------------------------------------------------------------------
# Project commands injection
# ---------------------------------------------------------------------------


def test_verification_prompt_includes_project_commands() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": ".venv/bin/pytest -n auto", "lint": ".venv/bin/ruff check ."}},
    )
    assert "## Project commands" in prompt
    assert ".venv/bin/pytest -n auto" in prompt
    assert ".venv/bin/ruff check ." in prompt


def test_implementation_prompt_includes_project_commands() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="implement", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": "pytest", "lint": "ruff check ."}},
    )
    assert "## Project commands" in prompt
    assert "pytest" in prompt


def test_prompt_no_commands_when_none() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands=None,
    )
    assert "Project commands" not in prompt


def test_prompt_partial_commands() -> None:
    """Only python test set → only test line appears, no lint/typecheck/format."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": "pytest -x"}},
    )
    assert "## Project commands" in prompt
    assert "Test:" in prompt
    assert "Lint:" not in prompt


def test_multi_language_commands() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python", "typescript"],
        project_commands={
            "python": {"test": "pytest", "lint": "ruff check ."},
            "typescript": {"test": "npm test", "lint": "npx eslint ."},
        },
    )
    assert "### Python" in prompt
    assert "### TypeScript" in prompt
    assert "pytest" in prompt
    assert "npm test" in prompt


def test_commands_filtered_by_detected_languages() -> None:
    """Config has python+go but only python detected → only python shown."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={
            "python": {"test": "pytest"},
            "go": {"test": "go test ./..."},
        },
    )
    assert "pytest" in prompt
    assert "go test" not in prompt


def test_commands_not_injected_for_planning() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="plan", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": "pytest"}},
    )
    assert "Project commands" not in prompt


def test_single_language_omits_subheading() -> None:
    """Single language uses flat list (no ### Python heading)."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": "pytest", "lint": "ruff check ."}},
    )
    assert "## Project commands" in prompt
    assert "### Python" not in prompt


def test_commands_not_injected_for_review() -> None:
    """Review step gets style guidelines but not project commands."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="review", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": "pytest"}},
    )
    assert "Style guidelines" in prompt
    assert "Project commands" not in prompt


def test_commands_empty_strings_skipped() -> None:
    """Empty or whitespace-only command values are not rendered."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": "", "lint": "   ", "typecheck": "mypy ."}},
    )
    assert "## Project commands" in prompt
    assert "Typecheck:" in prompt
    assert "Test:" not in prompt
    assert "Lint:" not in prompt


def test_multi_language_display_names_use_correct_casing() -> None:
    """TypeScript/JavaScript get proper casing, not .title() casing."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["typescript", "javascript"],
        project_commands={
            "typescript": {"test": "npm test"},
            "javascript": {"lint": "eslint ."},
        },
    )
    assert "### TypeScript" in prompt
    assert "### JavaScript" in prompt


def test_commands_non_string_values_skipped() -> None:
    """Non-string command values in config are silently skipped."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": 42, "lint": "ruff check ."}},  # type: ignore[dict-item]
    )
    assert "## Project commands" in prompt
    assert "Lint:" in prompt
    assert "Test:" not in prompt


def test_commands_non_dict_language_entry_skipped() -> None:
    """A non-dict language entry in project_commands is skipped."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="verify", attempt=1, is_codex=True,
        project_languages=["python", "go"],
        project_commands={"python": {"test": "pytest"}, "go": "not a dict"},  # type: ignore[dict-item]
    )
    assert "## Project commands" in prompt
    assert "pytest" in prompt
    # "go" entry silently skipped — only one language block, so no subheading
    assert "### Go" not in prompt


def test_run_step_reads_project_commands_from_config(container: Container, adapter: LiveWorkerAdapter) -> None:
    """run_step reads project.commands from config and passes them to the prompt."""
    # Write project commands to config
    cfg = container.config.load()
    cfg["project"] = {"commands": {"python": {"test": ".venv/bin/pytest -x"}}}
    container.config.save(cfg)

    # Create a pyproject.toml so python is detected
    (container.project_dir / "pyproject.toml").write_text("[build-system]")

    captured_prompt = {}
    run_result = _make_run_result(exit_code=0)

    def _capture_run_worker(**kwargs):
        captured_prompt["text"] = kwargs["prompt"]
        return run_result

    with (
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.get_workers_runtime_config"
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.resolve_worker_for_step",
            return_value=_CODEX_SPEC,
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.test_worker",
            return_value=(True, "ok"),
        ),
        patch(
            "agent_orchestrator.runtime.orchestrator.live_worker_adapter.run_worker",
            side_effect=_capture_run_worker,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="verify", attempt=1)

    assert result.status == "ok"
    prompt = captured_prompt["text"]
    assert "## Project commands" in prompt
    assert ".venv/bin/pytest -x" in prompt


# ---------------------------------------------------------------------------
# Step output injection in build_step_prompt
# ---------------------------------------------------------------------------


class TestStepOutputInjection:
    """Tests for prior step output injection into build_step_prompt."""

    def test_plan_output_not_injected_for_implementation(self) -> None:
        task = _make_task(metadata={"step_outputs": {"plan": "Use adapter pattern."}})
        prompt = build_step_prompt(task=task, step="implement", attempt=1, is_codex=True)
        assert "## Output from prior 'plan' step" not in prompt
        assert "Use adapter pattern." not in prompt

    def test_analyze_and_plan_not_injected_for_implementation(self) -> None:
        task = _make_task(metadata={
            "step_outputs": {"analyze": "Found 3 modules.", "plan": "Refactor X."}
        })
        prompt = build_step_prompt(task=task, step="implement", attempt=1, is_codex=True)
        assert "## Output from prior 'plan' step" not in prompt
        assert "## Output from prior 'analyze' step" not in prompt
        assert "Found 3 modules." not in prompt
        assert "Refactor X." not in prompt

    def test_verify_output_injected_for_review(self) -> None:
        task = _make_task(metadata={"step_outputs": {"verify": "All tests pass."}})
        prompt = build_step_prompt(task=task, step="review", attempt=1, is_codex=False)
        assert "## Output from prior 'verify' step" in prompt
        assert "All tests pass." in prompt

    def test_all_outputs_injected_for_reporting(self) -> None:
        task = _make_task(metadata={
            "step_outputs": {"gather": "Data collected.", "analyze": "Results look good."}
        })
        prompt = build_step_prompt(task=task, step="report", attempt=1, is_codex=True)
        assert "## Output from prior 'gather' step" in prompt
        assert "## Output from prior 'analyze' step" in prompt

    def test_plan_injected_for_task_generation(self) -> None:
        task = _make_task(metadata={"step_outputs": {"plan": "Split into 3 subtasks."}})
        prompt = build_step_prompt(task=task, step="generate_tasks", attempt=1, is_codex=True)
        assert "## Output from prior 'plan' step" in prompt
        assert "Split into 3 subtasks." in prompt

    def test_only_reproduce_injected_for_diagnose(self) -> None:
        task = _make_task(
            task_type="bug",
            metadata={"step_outputs": {"reproduce": "Fails on null payload", "plan": "Old plan text"}},
        )
        prompt = build_step_prompt(task=task, step="diagnose", attempt=1, is_codex=True)
        assert "## Output from prior 'reproduce' step" in prompt
        assert "Fails on null payload" in prompt
        assert "## Output from prior 'plan' step" not in prompt

    def test_no_injection_for_verification(self) -> None:
        task = _make_task(metadata={"step_outputs": {"plan": "Some plan."}})
        prompt = build_step_prompt(task=task, step="verify", attempt=1, is_codex=True)
        assert "## Output from prior" not in prompt

    def test_no_injection_for_planning(self) -> None:
        task = _make_task(metadata={"step_outputs": {"gather": "Info gathered."}})
        prompt = build_step_prompt(task=task, step="plan", attempt=1, is_codex=True)
        assert "## Output from prior" not in prompt

    def test_empty_values_skipped(self) -> None:
        task = _make_task(metadata={"step_outputs": {"plan": "", "analyze": "   "}})
        prompt = build_step_prompt(task=task, step="implement", attempt=1, is_codex=True)
        assert "## Output from prior" not in prompt

    def test_missing_step_outputs_no_crash(self) -> None:
        task = _make_task(metadata={})
        prompt = build_step_prompt(task=task, step="implement", attempt=1, is_codex=True)
        assert "## Output from prior" not in prompt

    def test_coexists_with_review_findings(self) -> None:
        task = _make_task(metadata={
            "step_outputs": {"plan": "Fix the bug."},
            "review_findings": [{"severity": "high", "summary": "Missing null check"}],
        })
        prompt = build_step_prompt(task=task, step="implement_fix", attempt=1, is_codex=True)
        assert "## Output from prior 'plan' step" not in prompt
        assert "Fix the bug." not in prompt
        assert "Missing null check" in prompt

    def test_fix_includes_verify_failure_and_output(self) -> None:
        task = _make_task(metadata={
            "verify_failure": "pytest failed",
            "verify_reason_code": "assertion_failure",
            "verify_output": "E   AssertionError: expected 1",
        })
        prompt = build_step_prompt(task=task, step="implement_fix", attempt=1, is_codex=True)
        assert "## Issues to fix" in prompt
        assert "### Verification failure" in prompt
        assert "pytest failed" in prompt
        assert "### Verification reason code" in prompt
        assert "assertion_failure" in prompt
        assert "### Verification output (test/lint/typecheck logs)" in prompt
        assert "AssertionError" in prompt

    def test_fix_includes_retry_and_requested_guidance(self) -> None:
        task = _make_task(metadata={
            "retry_guidance": {"previous_error": "timed out", "guidance": "narrow scope"},
            "requested_changes": {"guidance": "add edge-case test"},
        })
        prompt = build_step_prompt(task=task, step="implement_fix", attempt=2, is_codex=True)
        assert "## Issues to fix" in prompt
        assert "### Previous attempt error" in prompt
        assert "timed out" in prompt
        assert "### Feedback from previous attempt" in prompt
        assert "narrow scope" in prompt
        assert "### Requested changes from reviewer" in prompt
        assert "edge-case test" in prompt


# ---------------------------------------------------------------------------
# Workdoc prompt instructions
# ---------------------------------------------------------------------------


class TestWorkdocPromptInstructions:
    def test_prompt_includes_workdoc_instructions_for_plan(self) -> None:
        task = _make_task()
        prompt = build_step_prompt(task=task, step="plan", attempt=1, is_codex=True)
        assert "## Working Document" in prompt
        assert "## Plan" in prompt
        assert ".workdoc.md" in prompt

    def test_prompt_includes_workdoc_instructions_for_analyze(self) -> None:
        task = _make_task()
        prompt = build_step_prompt(task=task, step="analyze", attempt=1, is_codex=True)
        assert "## Working Document" in prompt
        assert "## Analysis" in prompt
        assert ".workdoc.md" in prompt
        assert "## Objective Interpreted" in prompt
        assert "## Implementation Phases" not in prompt

    def test_prompt_includes_workdoc_instructions_for_diagnose(self) -> None:
        task = _make_task(task_type="bug")
        prompt = build_step_prompt(task=task, step="diagnose", attempt=1, is_codex=True)
        assert "## Working Document" in prompt
        assert "## Analysis" in prompt
        assert ".workdoc.md" in prompt
        assert "## Root Cause Analysis" in prompt

    def test_prompt_includes_workdoc_instructions_for_implement(self) -> None:
        task = _make_task()
        prompt = build_step_prompt(task=task, step="implement", attempt=1, is_codex=True)
        assert "## Working Document" in prompt
        assert "## Implementation Log" in prompt
        assert ".workdoc.md" in prompt

    def test_prompt_includes_workdoc_instructions_for_prototype(self) -> None:
        task = _make_task(task_type="spike")
        prompt = build_step_prompt(task=task, step="prototype", attempt=1, is_codex=True)
        assert "## Working Document" in prompt
        assert "## Implementation Log" in prompt
        assert "prototype notes" in prompt
        assert "hypotheses tested" in prompt
        assert "Build a timeboxed prototype" in prompt

    def test_prompt_includes_workdoc_instructions_for_verify(self) -> None:
        task = _make_task()
        prompt = build_step_prompt(task=task, step="verify", attempt=1, is_codex=True)
        assert "## Working Document" in prompt
        assert "Do NOT modify `.workdoc.md`" in prompt

    def test_prompt_includes_workdoc_instructions_for_implement_fix(self) -> None:
        task = _make_task()
        prompt = build_step_prompt(task=task, step="implement_fix", attempt=1, is_codex=True)
        assert "## Working Document" in prompt
        assert "## Fix Log" in prompt
        assert "Do NOT modify `.workdoc.md`" in prompt

    def test_prompt_includes_workdoc_instructions_for_review(self) -> None:
        task = _make_task()
        prompt = build_step_prompt(task=task, step="review", attempt=1, is_codex=True)
        assert "## Working Document" in prompt
        assert "Do NOT modify" in prompt

    def test_prompt_no_workdoc_for_scan(self) -> None:
        task = _make_task()
        prompt = build_step_prompt(task=task, step="scan", attempt=1, is_codex=True)
        assert "## Working Document" not in prompt

    def test_prompt_no_workdoc_for_plan_refine(self) -> None:
        """plan_refine doesn't go through _run_non_review_step so should not get workdoc instructions."""
        task = _make_task()
        prompt = build_step_prompt(task=task, step="plan_refine", attempt=1, is_codex=True)
        assert "## Working Document" not in prompt
