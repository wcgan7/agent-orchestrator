import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import orchestrator
from actions import run_worker
from io_utils import _load_data, _save_data
from models import AllowlistViolation, TaskStep
from state import _ensure_state_files
from utils import _now_iso


def test_missing_phase_blocks_without_branch(tmp_path, monkeypatch):
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")

    paths = _ensure_state_files(project_dir, prd_path)

    queue = {
        "updated_at": _now_iso(),
        "tasks": [
            {
                "id": "phase-missing",
                "type": "implement",
                "phase_id": "phase-missing",
                "status": "todo",
                "lifecycle": "ready",
                "step": "plan_impl",
                "priority": 1,
                "deps": [],
                "description": "missing phase",
            }
        ],
    }
    _save_data(paths["task_queue"], queue)
    _save_data(paths["phase_plan"], {"updated_at": _now_iso(), "phases": []})

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("_ensure_branch should not be called when phase is missing")

    monkeypatch.setattr(orchestrator, "_ensure_branch", _fail_if_called)

    orchestrator.run_feature_prd(
        project_dir=project_dir,
        prd_path=prd_path,
        max_iterations=1,
    )

    updated_queue = _load_data(paths["task_queue"], {})
    task = updated_queue["tasks"][0]
    assert task["lifecycle"] == "waiting_human"
    assert task["status"] == "blocked"
    assert task["block_reason"] == "PLAN_MISSING"


def test_allowlist_violation_writes_manifest(tmp_path, monkeypatch):
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"
    progress_path.write_text(json.dumps({"run_id": "run-1"}))
    events_path = tmp_path / "events.ndjson"

    def fake_run_codex_worker(**kwargs):
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        stdout_path.write_text("")
        stderr_path.write_text("")
        return {
            "command": "codex exec -",
            "prompt_path": str(run_dir / "prompt.txt"),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "start_time": _now_iso(),
            "end_time": _now_iso(),
            "runtime_seconds": 1,
            "exit_code": 0,
            "timed_out": False,
            "no_heartbeat": False,
            "last_heartbeat": None,
        }

    snapshots = {"count": 0}

    def fake_snapshot_repo_changes(_project_dir):
        snapshots["count"] += 1
        if snapshots["count"] == 1:
            return []
        return ["disallowed.py"]

    monkeypatch.setattr(run_worker, "_run_codex_worker", fake_run_codex_worker)
    monkeypatch.setattr(run_worker, "_snapshot_repo_changes", fake_snapshot_repo_changes)

    task = {
        "id": "phase-1",
        "type": "implement",
        "phase_id": "phase-1",
        "context": [],
        "no_progress_attempts": 0,
    }
    phase = {"id": "phase-1", "name": "Phase 1", "description": ""}

    event = run_worker.run_worker_action(
        step=TaskStep.IMPLEMENT,
        task=task,
        phase=phase,
        prd_path=prd_path,
        project_dir=project_dir,
        artifacts_dir=artifacts_dir,
        phase_plan_path=tmp_path / "phase_plan.yaml",
        task_queue_path=tmp_path / "task_queue.yaml",
        run_dir=run_dir,
        run_id="run-1",
        codex_command="codex exec -",
        user_prompt=None,
        progress_path=progress_path,
        events_path=events_path,
        heartbeat_seconds=10,
        heartbeat_grace_seconds=20,
        shift_minutes=1,
        test_command=None,
    )

    assert isinstance(event, AllowlistViolation)
    manifest = _load_data(run_dir / "manifest.json", {})
    assert "disallowed.py" in manifest.get("disallowed_files", [])
