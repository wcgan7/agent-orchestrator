from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_orchestrator.workers.config import WorkerProviderSpec
from agent_orchestrator.workers.run import run_worker

RUN_INTEGRATION = os.getenv("AGENT_ORCHESTRATOR_RUN_INTEGRATION", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set AGENT_ORCHESTRATOR_RUN_INTEGRATION=1 to run integration tests",
)


def _supports_effort(executable: str) -> bool:
    completed = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return "--effort" in f"{completed.stdout}\n{completed.stderr}".lower()


def _write_wrapper(path: Path, claude_path: str) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "args_file=\"$1\"\n"
        "shift\n"
        "printf '%s\\n' \"$@\" > \"$args_file\"\n"
        f"exec {shlex.quote(claude_path)} \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_integration_claude_provider_model_and_effort_flags(tmp_path: Path) -> None:
    claude_path = shutil.which("claude")
    if not claude_path:
        pytest.skip("Claude CLI not installed")

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"
    args_path = run_dir / "claude_args.txt"
    wrapper = tmp_path / "claude-wrapper.sh"
    _write_wrapper(wrapper, claude_path)

    result = run_worker(
        spec=WorkerProviderSpec(
            name="claude",
            type="claude",
            command=f"{wrapper} {{run_dir}}/claude_args.txt -p --output-format text",
            model="sonnet",
            reasoning_effort="medium",
        ),
        prompt="Reply exactly: OK",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=120,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=20,
        progress_path=progress_path,
    )

    assert result.exit_code == 0
    captured = args_path.read_text(encoding="utf-8")
    assert "--model" in captured
    assert "sonnet" in captured
    if _supports_effort(claude_path):
        assert "--effort" in captured
        assert "medium" in captured
    else:
        assert "--effort" not in captured
