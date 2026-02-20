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


def _supports_reasoning_effort(executable: str) -> bool:
    completed = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return "--reasoning-effort" in f"{completed.stdout}\n{completed.stderr}".lower()


def _write_wrapper(path: Path, codex_path: str) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--help\" ]; then\n"
        f"  exec {shlex.quote(codex_path)} --help\n"
        "fi\n"
        "args_file=\"$1\"\n"
        "shift\n"
        "printf '%s\\n' \"$@\" > \"$args_file\"\n"
        f"exec {shlex.quote(codex_path)} \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_integration_codex_provider_model_and_reasoning_flags(tmp_path: Path) -> None:
    codex_path = shutil.which("codex")
    if not codex_path:
        pytest.skip("Codex CLI not installed")

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"
    args_path = run_dir / "codex_args.txt"
    wrapper = tmp_path / "codex-wrapper.sh"
    _write_wrapper(wrapper, codex_path)

    result = run_worker(
        spec=WorkerProviderSpec(
            name="codex",
            type="codex",
            # Keep --full-auto pre-set so codex command builder doesn't inject it
            # ahead of our wrapper args_file parameter.
            command=f"{wrapper} {{run_dir}}/codex_args.txt --full-auto exec --skip-git-repo-check",
            model="gpt-5-codex",
            reasoning_effort="medium",
        ),
        prompt="Reply with exactly: OK",
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
    assert "gpt-5-codex" in captured
    if _supports_reasoning_effort(codex_path):
        assert "--reasoning-effort" in captured
        assert "medium" in captured
    else:
        assert "--reasoning-effort" not in captured
