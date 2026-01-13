import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from feature_prd_runner import runner
from feature_prd_runner.io_utils import _load_data, _save_data
from feature_prd_runner.state import _ensure_state_files
from feature_prd_runner.utils import _now_iso


def test_list_shows_tasks_and_phases(tmp_path, capsys) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")
    paths = _ensure_state_files(project_dir, prd_path)

    _save_data(
        paths["phase_plan"],
        {
            "updated_at": _now_iso(),
            "phases": [{"id": "phase-1", "name": "P1", "status": "todo", "description": ""}],
        },
    )
    _save_data(
        paths["task_queue"],
        {
            "updated_at": _now_iso(),
            "tasks": [
                {"id": "phase-1", "type": "implement", "phase_id": "phase-1", "status": "blocked", "lifecycle": "waiting_human", "step": "implement"},
            ],
        },
    )

    try:
        runner.main(["list", "--project-dir", str(project_dir)])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out
    assert "Phases:" in out
    assert "Tasks:" in out
    assert "phase-1" in out


def test_resume_marks_task_ready(tmp_path) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")
    paths = _ensure_state_files(project_dir, prd_path)

    _save_data(
        paths["task_queue"],
        {
            "updated_at": _now_iso(),
            "tasks": [
                {"id": "phase-1", "type": "implement", "phase_id": "phase-1", "status": "blocked", "lifecycle": "waiting_human", "step": "review"},
            ],
        },
    )

    try:
        runner.main(["resume", "phase-1", "--project-dir", str(project_dir), "--step", "implement"])
    except SystemExit as exc:
        assert exc.code == 0

    queue = _load_data(paths["task_queue"], {})
    task = queue["tasks"][0]
    assert task["lifecycle"] == "ready"
    assert task["step"] == "implement"
    assert task["status"] == "implement"


def test_resume_refuses_when_run_active_without_force(tmp_path, capsys) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")
    paths = _ensure_state_files(project_dir, prd_path)

    # Mark run_state as running with a live PID so it is considered active.
    run_state = _load_data(paths["run_state"], {})
    run_state.update({"status": "running", "run_id": "run-1", "worker_pid": os.getpid()})
    _save_data(paths["run_state"], run_state)

    _save_data(
        paths["task_queue"],
        {
            "updated_at": _now_iso(),
            "tasks": [
                {"id": "phase-1", "type": "implement", "phase_id": "phase-1", "status": "blocked", "lifecycle": "waiting_human", "step": "review"},
            ],
        },
    )

    try:
        runner.main(["resume", "phase-1", "--project-dir", str(project_dir), "--step", "implement"])
    except SystemExit as exc:
        assert exc.code == 2

    out = capsys.readouterr().out
    assert "active" in out.lower()


def test_resume_does_not_overwrite_corrupted_task_queue(tmp_path, capsys) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)

    prd_path = project_dir / "prd.md"
    prd_path.write_text("Spec\n")
    paths = _ensure_state_files(project_dir, prd_path)

    bad_yaml = "tasks: [this is not valid yaml\n"
    paths["task_queue"].write_text(bad_yaml)

    try:
        runner.main(["resume", "phase-1", "--project-dir", str(project_dir), "--step", "implement", "--force"])
    except SystemExit as exc:
        assert exc.code == 2

    assert paths["task_queue"].read_text() == bad_yaml
    assert "Unable to read task_queue.yaml" in capsys.readouterr().out
