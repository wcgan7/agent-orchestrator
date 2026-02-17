from __future__ import annotations

import json
from pathlib import Path

from agent_orchestrator.workers.config import WorkerProviderSpec
from agent_orchestrator.workers.run import WorkerRunResult, run_worker


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
        "agent_orchestrator.workers.run._run_codex_worker",
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
        "agent_orchestrator.workers.run._run_ollama_generate",
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


def test_run_worker_codex_adds_model_and_reasoning_flags(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"
    captured: dict[str, str] = {}

    def _fake_codex_worker(**kwargs: object) -> dict[str, object]:
        captured["command"] = str(kwargs.get("command") or "")
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
        "agent_orchestrator.workers.run._run_codex_worker",
        _fake_codex_worker,
    )
    monkeypatch.setattr(
        "agent_orchestrator.workers.run._codex_supports_reasoning_effort",
        lambda _exe: True,
    )

    run_worker(
        spec=WorkerProviderSpec(
            name="codex",
            type="codex",
            command="codex",
            model="gpt-5-codex",
            reasoning_effort="medium",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert "--model" in captured["command"]
    assert "gpt-5-codex" in captured["command"]
    assert "--reasoning-effort" in captured["command"]
    assert "medium" in captured["command"]


def test_run_worker_codex_skips_reasoning_flag_when_unsupported(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"
    captured: dict[str, str] = {}

    def _fake_codex_worker(**kwargs: object) -> dict[str, object]:
        captured["command"] = str(kwargs.get("command") or "")
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
        "agent_orchestrator.workers.run._run_codex_worker",
        _fake_codex_worker,
    )
    monkeypatch.setattr(
        "agent_orchestrator.workers.run._codex_supports_reasoning_effort",
        lambda _exe: False,
    )

    run_worker(
        spec=WorkerProviderSpec(
            name="codex",
            type="codex",
            command="codex",
            model="gpt-5-codex",
            reasoning_effort="medium",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert "--model" in captured["command"]
    assert "gpt-5-codex" in captured["command"]
    assert "--reasoning-effort" not in captured["command"]


def test_run_worker_claude_adds_model_and_effort_flags(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"
    captured: dict[str, str] = {}

    def _fake_worker(**kwargs: object) -> dict[str, object]:
        captured["command"] = str(kwargs.get("command") or "")
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
        "agent_orchestrator.workers.run._run_codex_worker",
        _fake_worker,
    )
    monkeypatch.setattr(
        "agent_orchestrator.workers.run._claude_supports_effort",
        lambda _exe: True,
    )

    run_worker(
        spec=WorkerProviderSpec(
            name="claude",
            type="claude",
            command="claude -p",
            model="sonnet",
            reasoning_effort="medium",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert "--model" in captured["command"]
    assert "sonnet" in captured["command"]
    assert "--effort" in captured["command"]
    assert "medium" in captured["command"]
    assert "--dangerously-skip-permissions" in captured["command"]


def test_run_worker_claude_skips_effort_when_unsupported(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"
    captured: dict[str, str] = {}

    def _fake_worker(**kwargs: object) -> dict[str, object]:
        captured["command"] = str(kwargs.get("command") or "")
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
        "agent_orchestrator.workers.run._run_codex_worker",
        _fake_worker,
    )
    monkeypatch.setattr(
        "agent_orchestrator.workers.run._claude_supports_effort",
        lambda _exe: False,
    )

    run_worker(
        spec=WorkerProviderSpec(
            name="claude",
            type="claude",
            command="claude -p",
            model="sonnet",
            reasoning_effort="high",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert "--model" in captured["command"]
    assert "sonnet" in captured["command"]
    assert "--effort" not in captured["command"]
    assert "--dangerously-skip-permissions" in captured["command"]


def test_run_worker_claude_defaults_to_stream_json_output(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"
    captured: dict[str, str] = {}

    def _fake_worker(**kwargs: object) -> dict[str, object]:
        captured["command"] = str(kwargs.get("command") or "")
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
        "agent_orchestrator.workers.run._run_codex_worker",
        _fake_worker,
    )
    monkeypatch.setattr(
        "agent_orchestrator.workers.run._claude_supports_effort",
        lambda _exe: True,
    )

    run_worker(
        spec=WorkerProviderSpec(
            name="claude",
            type="claude",
            command="claude -p",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert "--output-format" in captured["command"]
    assert "stream-json" in captured["command"]
    assert "--include-partial-messages" in captured["command"]
    assert "--verbose" in captured["command"]
    assert "--dangerously-skip-permissions" in captured["command"]


def test_run_worker_claude_stream_json_extracts_assistant_text(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"

    def _fake_worker(**_: object) -> dict[str, object]:
        stdout_path = run_dir / "stdout.log"
        stdout_path.write_text(
            "\n".join(
                [
                    '{"type":"system","subtype":"init"}',
                    '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}}',
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}',
                    '{"type":"result","result":"ok"}',
                ]
            ),
            encoding="utf-8",
        )
        return {
            "prompt_path": str(run_dir / "prompt.txt"),
            "stdout_path": str(stdout_path),
            "stderr_path": str(run_dir / "stderr.log"),
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T00:00:10Z",
            "runtime_seconds": 10,
            "exit_code": 0,
            "timed_out": False,
            "no_heartbeat": False,
        }

    monkeypatch.setattr(
        "agent_orchestrator.workers.run._run_codex_worker",
        _fake_worker,
    )

    result = run_worker(
        spec=WorkerProviderSpec(
            name="claude",
            type="claude",
            command="claude -p --verbose --output-format stream-json --include-partial-messages",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert result.response_text == "ok"
