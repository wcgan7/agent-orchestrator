"""Test worker heartbeat detection and timeout behavior."""

import json
import sys
from pathlib import Path

from agent_orchestrator.utils import _now_iso
from agent_orchestrator.worker import _run_codex_worker


def test_worker_uses_output_as_heartbeat(tmp_path: Path) -> None:
    """Ensure stdout/stderr activity counts as liveness when heartbeats lag."""
    project_dir = tmp_path / "repo"
    project_dir.mkdir()

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    progress_path = run_dir / "progress.json"
    progress_path.write_text(json.dumps({"run_id": "run-1", "heartbeat": _now_iso()}))

    # Emit stdout every ~1s for > heartbeat_grace_seconds without updating progress.json.
    command = (
        f"{sys.executable} -c "
        "\"import sys,time,pathlib; "
        "pathlib.Path(sys.argv[1]).read_text(); "
        "[(print('line %d' % i, flush=True), time.sleep(1)) for i in range(6)]\" "
        "{prompt_file}"
    )

    result = _run_codex_worker(
        command=command,
        prompt="hello",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=30,
        heartbeat_seconds=10,
        heartbeat_grace_seconds=3,
        progress_path=progress_path,
        expected_run_id="run-1",
    )

    assert result["exit_code"] == 0
    assert result["no_heartbeat"] is False


def test_plain_command_defaults_to_stdin_prompt(tmp_path: Path) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    progress_path = run_dir / "progress.json"
    progress_path.write_text(json.dumps({"run_id": "run-2", "heartbeat": _now_iso()}))

    command = f"{sys.executable} -c \"import sys; data=sys.stdin.read(); print('ok' if data.strip()=='hello' else 'bad')\""
    result = _run_codex_worker(
        command=command,
        prompt="hello",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=20,
        heartbeat_seconds=10,
        heartbeat_grace_seconds=5,
        progress_path=progress_path,
        expected_run_id="run-2",
    )

    assert result["exit_code"] == 0
    stdout = (run_dir / "stdout.log").read_text(encoding="utf-8")
    assert "ok" in stdout
