import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from feature_prd_runner.utils import _now_iso
from feature_prd_runner.worker import _run_codex_worker


def test_worker_uses_output_as_heartbeat(tmp_path: Path) -> None:
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

