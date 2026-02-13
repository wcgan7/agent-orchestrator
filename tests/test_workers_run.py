from __future__ import annotations

import json
from pathlib import Path

from feature_prd_runner.workers.config import WorkerProviderSpec
from feature_prd_runner.workers.run import WorkerRunResult, run_worker


def test_run_worker_extracts_human_blocking_issues_from_progress(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"

    def _fake_codex_worker(**_: object) -> dict[str, object]:
        progress_path.write_text(
            json.dumps(
                {
                    "heartbeat": "2025-01-01T00:00:00Z",
                    "human_blocking_issues": [
                        {"summary": "Need API token", "details": "Request access from ops"},
                        "Clarify acceptance criteria",
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {
            "prompt_path": str(run_dir / "prompt.txt"),
            "stdout_path": str(run_dir / "stdout.log"),
            "stderr_path": str(run_dir / "stderr.log"),
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T00:00:10Z",
            "runtime_seconds": 10,
            "exit_code": 0,
            "timed_out": False,
            "no_heartbeat": False,
        }

    monkeypatch.setattr(
        "feature_prd_runner.workers.run._run_codex_worker",
        _fake_codex_worker,
    )

    result = run_worker(
        spec=WorkerProviderSpec(name="codex", type="codex", command="codex"),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert len(result.human_blocking_issues) == 2
    assert result.human_blocking_issues[0]["summary"] == "Need API token"
    assert result.human_blocking_issues[1]["summary"] == "Clarify acceptance criteria"


def test_run_worker_extracts_human_blocking_issues_for_ollama(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"

    def _fake_ollama_generate(**_: object) -> WorkerRunResult:
        progress_path.write_text(
            json.dumps(
                {
                    "heartbeat": "2025-01-01T00:00:00Z",
                    "human_blocking_issues": [{"summary": "Need legal approval"}],
                }
            ),
            encoding="utf-8",
        )
        return WorkerRunResult(
            provider="ollama:test",
            prompt_path=str(run_dir / "prompt.txt"),
            stdout_path=str(run_dir / "stdout.log"),
            stderr_path=str(run_dir / "stderr.log"),
            start_time="2025-01-01T00:00:00Z",
            end_time="2025-01-01T00:00:01Z",
            runtime_seconds=1,
            exit_code=0,
            timed_out=False,
            no_heartbeat=False,
            response_text='{"status":"ok"}',
        )

    monkeypatch.setattr(
        "feature_prd_runner.workers.run._run_ollama_generate",
        _fake_ollama_generate,
    )

    result = run_worker(
        spec=WorkerProviderSpec(
            name="ollama-test",
            type="ollama",
            endpoint="http://localhost:11434",
            model="llama3",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert len(result.human_blocking_issues) == 1
    assert result.human_blocking_issues[0]["summary"] == "Need legal approval"
