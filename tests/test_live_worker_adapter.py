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
    detect_project_languages,
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
# 3. Step timeout can be overridden per task metadata
# ---------------------------------------------------------------------------


def test_timeout_override_from_task_metadata(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)
    task = _make_task(metadata={"step_timeouts": {"implement": 42}})

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
        ) as run_worker_mock,
    ):
        result = adapter.run_step(task=task, step="implement", attempt=1)

    assert result.status == "ok"
    assert run_worker_mock.call_args.kwargs["timeout_seconds"] == 42


# ---------------------------------------------------------------------------
# 4. Template timeout is used when no metadata override exists
# ---------------------------------------------------------------------------


def test_timeout_defaults_from_pipeline_template(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=0)
    task = _make_task(task_type="bug")

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
        ) as run_worker_mock,
    ):
        result = adapter.run_step(task=task, step="reproduce", attempt=1)

    assert result.status == "ok"
    assert run_worker_mock.call_args.kwargs["timeout_seconds"] == 300


# ---------------------------------------------------------------------------
# 5. Codex failure maps to error
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
# 6. Codex timeout maps to error
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
# 7. Codex no-heartbeat maps to explicit stall error
# ---------------------------------------------------------------------------


def test_codex_no_heartbeat_maps_to_stalled_error(adapter: LiveWorkerAdapter) -> None:
    run_result = _make_run_result(exit_code=1, no_heartbeat=True)

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
# 10. Ollama generate_tasks parses tasks
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
# 11. Ollama implement extracts summary
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
    assert ".prd_runner/" in prompt
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
    assert "entire step fully" in prompt
    assert "inconsistent state" in prompt


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
    assert "Do not bypass" in prompt
    assert "do not\nmask failures" in prompt.lower() or "do not mask failures" in prompt.lower()


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
    assert "Language standards" in prompt
    assert "Python" in prompt
    assert "ruff" in prompt


def test_review_prompt_includes_typescript_standards() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="review", attempt=1, is_codex=True,
        project_languages=["typescript"],
    )
    assert "Language standards" in prompt
    assert "TypeScript" in prompt
    assert "tsc" in prompt


def test_prompt_no_language_when_none() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="implement", attempt=1, is_codex=True, project_languages=None,
    )
    assert "Language standards" not in prompt


def test_language_not_injected_for_planning() -> None:
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="plan", attempt=1, is_codex=True,
        project_languages=["python"],
    )
    assert "Language standards" not in prompt


def test_mixed_language_prompt_includes_all_standards() -> None:
    """Full-stack repo: both Python and TypeScript standards are injected."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="implement", attempt=1, is_codex=True,
        project_languages=["python", "typescript"],
    )
    assert "Language standards \u2014 Python" in prompt
    assert "Language standards \u2014 TypeScript" in prompt
    assert "ruff" in prompt
    assert "tsc" in prompt


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
    """Review step gets language standards but not project commands."""
    task = _make_task()
    prompt = build_step_prompt(
        task=task, step="review", attempt=1, is_codex=True,
        project_languages=["python"],
        project_commands={"python": {"test": "pytest"}},
    )
    assert "Language standards" in prompt
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


def test_run_step_reads_project_commands_from_config(container: V3Container, adapter: LiveWorkerAdapter) -> None:
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
            side_effect=_capture_run_worker,
        ),
    ):
        result = adapter.run_step(task=_make_task(), step="verify", attempt=1)

    assert result.status == "ok"
    prompt = captured_prompt["text"]
    assert "## Project commands" in prompt
    assert ".venv/bin/pytest -x" in prompt
