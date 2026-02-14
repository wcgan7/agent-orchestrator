from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.orchestrator.live_worker_adapter import LiveWorkerAdapter
from agent_orchestrator.runtime.storage.container import Container

RUN_INTEGRATION = os.getenv("AGENT_ORCHESTRATOR_RUN_INTEGRATION", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set AGENT_ORCHESTRATOR_RUN_INTEGRATION=1 to run integration tests",
)


def _write_fake_codex(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "out_file=\"$1\"\n"
        "shift\n"
        "printf '%s\\n' \"$@\" > \"$out_file\"\n"
        "cat >/dev/null\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _make_adapter(tmp_path: Path, *, default_model: str) -> tuple[LiveWorkerAdapter, Path]:
    container = Container(tmp_path)
    fake_worker = tmp_path / "fake-codex.sh"
    _write_fake_codex(fake_worker)

    cfg = container.config.load()
    cfg["workers"] = {
        "default": "codex",
        "default_model": default_model,
        "routing": {},
        "providers": {
            "codex": {
                "type": "codex",
                "command": f"{fake_worker} {{project_dir}}/.captured-worker-args.txt",
            }
        },
    }
    container.config.save(cfg)
    return LiveWorkerAdapter(container), container.project_dir / ".captured-worker-args.txt"


def test_integration_uses_workers_default_model_when_task_model_missing(tmp_path: Path) -> None:
    adapter, captured_args_path = _make_adapter(tmp_path, default_model="gpt-5-codex-default")

    task = Task(
        title="Integration model fallback",
        description="Ensure default model fallback is used",
        task_type="feature",
    )

    result = adapter.run_step(task=task, step="implement", attempt=1)

    assert result.status == "ok"
    captured = captured_args_path.read_text(encoding="utf-8")
    assert "--model" in captured
    assert "gpt-5-codex-default" in captured


def test_integration_task_model_overrides_default_model(tmp_path: Path) -> None:
    adapter, captured_args_path = _make_adapter(tmp_path, default_model="gpt-5-codex-default")

    task = Task(
        title="Integration model override",
        description="Ensure task-level model override is used",
        task_type="feature",
        worker_model="gpt-5-codex-task",
    )

    result = adapter.run_step(task=task, step="implement", attempt=1)

    assert result.status == "ok"
    captured = captured_args_path.read_text(encoding="utf-8")
    assert "--model" in captured
    assert "gpt-5-codex-task" in captured
    assert "gpt-5-codex-default" not in captured
