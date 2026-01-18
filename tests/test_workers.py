"""Test worker provider config and non-agentic (Ollama-style) execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from feature_prd_runner.actions.run_worker import run_worker_action
from feature_prd_runner.io_utils import _load_data, _save_data
from feature_prd_runner.models import AllowlistViolation, TaskStep, WorkerSucceeded
from feature_prd_runner.utils import _now_iso
from feature_prd_runner.workers import WorkerRunResult, get_workers_runtime_config


def test_workers_config_routing_resolves_provider() -> None:
    runtime = get_workers_runtime_config(
        config={
            "workers": {
                "default": "codex",
                "providers": {
                    "local": {"type": "ollama", "endpoint": "http://localhost:11434", "model": "x"},
                },
                "routing": {"plan": "local"},
            }
        },
        codex_command_fallback="codex exec -",
        cli_worker=None,
    )
    assert runtime.default_worker == "codex"
    assert runtime.routing["plan"] == "local"
    assert "codex" in runtime.providers
    assert runtime.providers["local"].type == "ollama"


def test_run_worker_action_ollama_plan_writes_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")

    state_dir = project_dir / ".prd_runner"
    artifacts_dir = state_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    run_dir = state_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)

    phase_plan_path = state_dir / "phase_plan.yaml"
    task_queue_path = state_dir / "task_queue.yaml"
    progress_path = run_dir / "progress.json"
    progress_path.write_text(json.dumps({"run_id": "run-1"}))

    runtime = get_workers_runtime_config(
        config={
            "workers": {
                "default": "codex",
                "providers": {
                    "local": {"type": "ollama", "endpoint": "http://localhost:11434", "model": "x"},
                },
                "routing": {"plan": "local"},
            }
        },
        codex_command_fallback="codex exec -",
        cli_worker=None,
    )

    def fake_run_worker(**kwargs: Any) -> WorkerRunResult:
        (run_dir / "stdout.log").write_text("")
        (run_dir / "stderr.log").write_text("")
        return WorkerRunResult(
            provider="ollama:x",
            prompt_path=str(run_dir / "prompt.txt"),
            stdout_path=str(run_dir / "stdout.log"),
            stderr_path=str(run_dir / "stderr.log"),
            start_time=_now_iso(),
            end_time=_now_iso(),
            runtime_seconds=1,
            exit_code=0,
            timed_out=False,
            no_heartbeat=False,
            response_text=json.dumps(
                {
                    "phase_plan": {"phases": [{"id": "phase-1", "name": "P1", "status": "todo"}]},
                    "task_queue": {
                        "tasks": [
                            {
                                "id": "phase-1",
                                "type": "implement",
                                "phase_id": "phase-1",
                                "status": "todo",
                                "lifecycle": "ready",
                                "step": "plan_impl",
                                "priority": 1,
                                "deps": [],
                                "description": "phase 1",
                            }
                        ]
                    },
                    "human_blocking_issues": [],
                    "human_next_steps": [],
                }
            ),
        )

    monkeypatch.setattr("feature_prd_runner.actions.run_worker.run_worker", fake_run_worker)

    task = {"id": "plan", "type": "plan", "context": [], "no_progress_attempts": 0}
    event = run_worker_action(
        step=TaskStep.PLAN_IMPL,  # plan tasks run under PLAN_IMPL in the FSM
        task=task,
        phase=None,
        prd_path=prd_path,
        project_dir=project_dir,
        artifacts_dir=artifacts_dir,
        phase_plan_path=phase_plan_path,
        task_queue_path=task_queue_path,
        run_dir=run_dir,
        run_id="run-1",
        codex_command="codex exec -",
        user_prompt=None,
        progress_path=progress_path,
        events_path=state_dir / "events.jsonl",
        heartbeat_seconds=10,
        heartbeat_grace_seconds=20,
        shift_minutes=1,
        test_command=None,
        workers_runtime=runtime,
    )

    assert isinstance(event, WorkerSucceeded)
    assert _load_data(phase_plan_path, {}).get("phases")
    assert _load_data(task_queue_path, {}).get("tasks")


def test_run_worker_action_ollama_implement_applies_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    (project_dir / "app.txt").write_text("hello\n")
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=project_dir, check=True)
    subprocess.run(["git", "add", "app.txt"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=project_dir, check=True)

    prd_path = tmp_path / "prd.md"
    prd_path.write_text("Spec\n")

    state_dir = project_dir / ".prd_runner"
    artifacts_dir = state_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    run_dir = state_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    progress_path = run_dir / "progress.json"
    progress_path.write_text(json.dumps({"run_id": "run-1"}))

    # Minimal impl plan allowlisting app.txt
    _save_data(
        artifacts_dir / "impl_plan_phase-1.json",
        {
            "phase_id": "phase-1",
            "spec_summary": ["x"],
            "files_to_change": ["app.txt"],
            "new_files": [],
            "technical_approach": ["x"],
            "design_notes": {"architecture": [], "data_flow": [], "invariants": [], "edge_cases": []},
            "test_plan": {"commands": [], "new_tests": [], "manual_checks": []},
            "migration_or_rollout": ["(none)"],
            "open_questions": [],
            "assumptions": [],
            "plan_deviations": [],
        },
    )

    runtime = get_workers_runtime_config(
        config={
            "workers": {
                "default": "codex",
                "providers": {
                    "local": {"type": "ollama", "endpoint": "http://localhost:11434", "model": "x"},
                },
                "routing": {"implement": "local"},
            }
        },
        codex_command_fallback="codex exec -",
        cli_worker=None,
    )

    patch = """diff --git a/app.txt b/app.txt
--- a/app.txt
+++ b/app.txt
@@ -1 +1 @@
-hello
+hello world
"""

    def fake_run_worker(**kwargs: Any) -> WorkerRunResult:
        (run_dir / "stdout.log").write_text("")
        (run_dir / "stderr.log").write_text("")
        return WorkerRunResult(
            provider="ollama:x",
            prompt_path=str(run_dir / "prompt.txt"),
            stdout_path=str(run_dir / "stdout.log"),
            stderr_path=str(run_dir / "stderr.log"),
            start_time=_now_iso(),
            end_time=_now_iso(),
            runtime_seconds=1,
            exit_code=0,
            timed_out=False,
            no_heartbeat=False,
            response_text=json.dumps({"patch": patch, "human_blocking_issues": [], "human_next_steps": []}),
        )

    monkeypatch.setattr("feature_prd_runner.actions.run_worker.run_worker", fake_run_worker)

    task = {"id": "phase-1", "type": "implement", "phase_id": "phase-1", "context": [], "no_progress_attempts": 0}
    phase = {"id": "phase-1", "name": "Phase 1", "description": ""}
    event = run_worker_action(
        step=TaskStep.IMPLEMENT,
        task=task,
        phase=phase,
        prd_path=prd_path,
        project_dir=project_dir,
        artifacts_dir=artifacts_dir,
        phase_plan_path=state_dir / "phase_plan.yaml",
        task_queue_path=state_dir / "task_queue.yaml",
        run_dir=run_dir,
        run_id="run-1",
        codex_command="codex exec -",
        user_prompt=None,
        progress_path=progress_path,
        events_path=state_dir / "events.jsonl",
        heartbeat_seconds=10,
        heartbeat_grace_seconds=20,
        shift_minutes=1,
        test_command=None,
        workers_runtime=runtime,
    )

    assert isinstance(event, WorkerSucceeded)
    assert (project_dir / "app.txt").read_text() == "hello world\n"


def test_run_worker_action_ollama_implement_blocks_disallowed_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    (project_dir / "allowed.txt").write_text("ok\n")

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")

    state_dir = project_dir / ".prd_runner"
    artifacts_dir = state_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    run_dir = state_dir / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    progress_path = run_dir / "progress.json"
    progress_path.write_text(json.dumps({"run_id": "run-1"}))

    _save_data(
        artifacts_dir / "impl_plan_phase-1.json",
        {
            "phase_id": "phase-1",
            "spec_summary": ["x"],
            "files_to_change": ["allowed.txt"],
            "new_files": [],
            "technical_approach": ["x"],
            "design_notes": {"architecture": [], "data_flow": [], "invariants": [], "edge_cases": []},
            "test_plan": {"commands": [], "new_tests": [], "manual_checks": []},
            "migration_or_rollout": ["(none)"],
            "open_questions": [],
            "assumptions": [],
            "plan_deviations": [],
        },
    )

    runtime = get_workers_runtime_config(
        config={
            "workers": {
                "default": "codex",
                "providers": {
                    "local": {"type": "ollama", "endpoint": "http://localhost:11434", "model": "x"},
                },
                "routing": {"implement": "local"},
            }
        },
        codex_command_fallback="codex exec -",
        cli_worker=None,
    )

    patch = """diff --git a/not_allowed.txt b/not_allowed.txt
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/not_allowed.txt
@@ -0,0 +1 @@
+nope
"""

    def fake_run_worker(**kwargs: Any) -> WorkerRunResult:
        (run_dir / "stdout.log").write_text("")
        (run_dir / "stderr.log").write_text("")
        return WorkerRunResult(
            provider="ollama:x",
            prompt_path=str(run_dir / "prompt.txt"),
            stdout_path=str(run_dir / "stdout.log"),
            stderr_path=str(run_dir / "stderr.log"),
            start_time=_now_iso(),
            end_time=_now_iso(),
            runtime_seconds=1,
            exit_code=0,
            timed_out=False,
            no_heartbeat=False,
            response_text=json.dumps({"patch": patch, "human_blocking_issues": [], "human_next_steps": []}),
        )

    monkeypatch.setattr("feature_prd_runner.actions.run_worker.run_worker", fake_run_worker)

    task = {"id": "phase-1", "type": "implement", "phase_id": "phase-1", "context": [], "no_progress_attempts": 0}
    phase = {"id": "phase-1", "name": "Phase 1", "description": ""}
    event = run_worker_action(
        step=TaskStep.IMPLEMENT,
        task=task,
        phase=phase,
        prd_path=prd_path,
        project_dir=project_dir,
        artifacts_dir=artifacts_dir,
        phase_plan_path=state_dir / "phase_plan.yaml",
        task_queue_path=state_dir / "task_queue.yaml",
        run_dir=run_dir,
        run_id="run-1",
        codex_command="codex exec -",
        user_prompt=None,
        progress_path=progress_path,
        events_path=state_dir / "events.jsonl",
        heartbeat_seconds=10,
        heartbeat_grace_seconds=20,
        shift_minutes=1,
        test_command=None,
        workers_runtime=runtime,
    )

    assert isinstance(event, AllowlistViolation)
