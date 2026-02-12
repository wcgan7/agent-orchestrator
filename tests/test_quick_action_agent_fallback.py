"""Tests for quick action agent fallback via the workers subsystem."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from feature_prd_runner.v3.domain.models import QuickActionRun
from feature_prd_runner.v3.events.bus import EventBus
from feature_prd_runner.v3.quick_actions.executor import QuickActionExecutor
from feature_prd_runner.v3.storage.container import V3Container
from feature_prd_runner.workers.config import WorkerProviderSpec
from feature_prd_runner.workers.run import WorkerRunResult


def _make(tmp_path: Path) -> tuple[QuickActionExecutor, V3Container, MagicMock]:
    container = V3Container(tmp_path)
    mock_bus = MagicMock(spec=EventBus)
    executor = QuickActionExecutor(container, mock_bus)
    return executor, container, mock_bus


def _dummy_result(**overrides) -> WorkerRunResult:
    defaults = dict(
        provider="ollama:test",
        prompt_path="/tmp/prompt.txt",
        stdout_path="/tmp/stdout.log",
        stderr_path="/tmp/stderr.log",
        start_time="2025-01-01T00:00:00",
        end_time="2025-01-01T00:01:00",
        runtime_seconds=60,
        exit_code=0,
        timed_out=False,
        no_heartbeat=False,
        response_text="",
    )
    defaults.update(overrides)
    return WorkerRunResult(**defaults)


_PATCH_BASE = "feature_prd_runner.v3.quick_actions.executor"


def test_no_workers_configured(tmp_path: Path) -> None:
    """When resolve_worker_for_step raises ValueError, graceful message."""
    executor, container, _ = _make(tmp_path)
    run = QuickActionRun(prompt="explain auth flow")

    with patch(f"{_PATCH_BASE}.get_workers_runtime_config", side_effect=ValueError("no workers")):
        result = executor.execute(run)

    assert result.kind == "agent"
    assert result.status == "failed"
    assert "no worker" in (result.result_summary or "").lower()


def test_worker_unavailable(tmp_path: Path) -> None:
    """When test_worker returns (False, reason), show meaningful message."""
    executor, container, _ = _make(tmp_path)
    run = QuickActionRun(prompt="explain auth flow")

    mock_runtime = MagicMock()
    mock_spec = WorkerProviderSpec(name="test", type="ollama")

    with (
        patch(f"{_PATCH_BASE}.get_workers_runtime_config", return_value=mock_runtime),
        patch(f"{_PATCH_BASE}.resolve_worker_for_step", return_value=mock_spec),
        patch(f"{_PATCH_BASE}.test_worker", return_value=(False, "ollama not running")),
    ):
        result = executor.execute(run)

    assert result.status == "failed"
    assert "No worker available" in (result.result_summary or "")
    assert "ollama not running" in (result.result_summary or "")


def test_worker_success_ollama(tmp_path: Path) -> None:
    """Ollama worker returns response_text directly."""
    executor, container, _ = _make(tmp_path)
    run = QuickActionRun(prompt="explain auth flow")

    mock_runtime = MagicMock()
    mock_spec = WorkerProviderSpec(name="test", type="ollama")
    worker_result = _dummy_result(response_text="The auth flow works like this...")

    with (
        patch(f"{_PATCH_BASE}.get_workers_runtime_config", return_value=mock_runtime),
        patch(f"{_PATCH_BASE}.resolve_worker_for_step", return_value=mock_spec),
        patch(f"{_PATCH_BASE}.test_worker", return_value=(True, "ok")),
        patch(f"{_PATCH_BASE}.run_worker", return_value=worker_result),
    ):
        result = executor.execute(run)

    assert result.status == "completed"
    assert result.exit_code == 0
    assert "auth flow" in (result.result_summary or "").lower()


def test_worker_success_codex(tmp_path: Path) -> None:
    """Codex worker: response_text is empty, output read from stdout_path."""
    executor, container, _ = _make(tmp_path)
    run = QuickActionRun(prompt="implement feature X")

    stdout_file = tmp_path / "codex_stdout.log"
    stdout_file.write_text("Feature X implemented successfully.")

    mock_runtime = MagicMock()
    mock_spec = WorkerProviderSpec(name="test", type="codex")
    worker_result = _dummy_result(
        provider="codex",
        response_text="",
        stdout_path=str(stdout_file),
    )

    with (
        patch(f"{_PATCH_BASE}.get_workers_runtime_config", return_value=mock_runtime),
        patch(f"{_PATCH_BASE}.resolve_worker_for_step", return_value=mock_spec),
        patch(f"{_PATCH_BASE}.test_worker", return_value=(True, "ok")),
        patch(f"{_PATCH_BASE}.run_worker", return_value=worker_result),
    ):
        result = executor.execute(run)

    assert result.status == "completed"
    assert "Feature X implemented" in (result.result_summary or "")


def test_worker_timeout(tmp_path: Path) -> None:
    """When worker times out, run should be failed."""
    executor, container, _ = _make(tmp_path)
    run = QuickActionRun(prompt="long running task")

    mock_runtime = MagicMock()
    mock_spec = WorkerProviderSpec(name="test", type="ollama")
    worker_result = _dummy_result(timed_out=True, exit_code=124)

    with (
        patch(f"{_PATCH_BASE}.get_workers_runtime_config", return_value=mock_runtime),
        patch(f"{_PATCH_BASE}.resolve_worker_for_step", return_value=mock_spec),
        patch(f"{_PATCH_BASE}.test_worker", return_value=(True, "ok")),
        patch(f"{_PATCH_BASE}.run_worker", return_value=worker_result),
    ):
        result = executor.execute(run)

    assert result.status == "failed"
    assert "timed out" in (result.result_summary or "").lower()


def test_worker_exception(tmp_path: Path) -> None:
    """When run_worker raises, run should be failed with error message."""
    executor, container, _ = _make(tmp_path)
    run = QuickActionRun(prompt="do something")

    mock_runtime = MagicMock()
    mock_spec = WorkerProviderSpec(name="test", type="ollama")

    with (
        patch(f"{_PATCH_BASE}.get_workers_runtime_config", return_value=mock_runtime),
        patch(f"{_PATCH_BASE}.resolve_worker_for_step", return_value=mock_spec),
        patch(f"{_PATCH_BASE}.test_worker", return_value=(True, "ok")),
        patch(f"{_PATCH_BASE}.run_worker", side_effect=RuntimeError("connection lost")),
    ):
        result = executor.execute(run)

    assert result.status == "failed"
    assert "connection lost" in (result.result_summary or "").lower()


def test_truncates_long_output(tmp_path: Path) -> None:
    """Output longer than 4000 chars should be truncated."""
    executor, container, _ = _make(tmp_path)
    run = QuickActionRun(prompt="verbose task")

    long_text = "x" * 8000
    mock_runtime = MagicMock()
    mock_spec = WorkerProviderSpec(name="test", type="ollama")
    worker_result = _dummy_result(response_text=long_text)

    with (
        patch(f"{_PATCH_BASE}.get_workers_runtime_config", return_value=mock_runtime),
        patch(f"{_PATCH_BASE}.resolve_worker_for_step", return_value=mock_spec),
        patch(f"{_PATCH_BASE}.test_worker", return_value=(True, "ok")),
        patch(f"{_PATCH_BASE}.run_worker", return_value=worker_result),
    ):
        result = executor.execute(run)

    assert result.status == "completed"
    assert len(result.result_summary or "") <= 4000
