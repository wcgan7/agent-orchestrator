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
    assert "--full-auto" in captured["command"]
    assert "--danger-full-access" not in captured["command"]


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


def test_run_worker_codex_host_access_injects_danger_flag(
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
            execution_mode="host_access",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert "--danger-full-access" in captured["command"]
    assert "--full-auto" not in captured["command"]


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
            execution_mode="host_access",
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
            execution_mode="host_access",
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
            execution_mode="host_access",
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


def test_run_worker_claude_sandboxed_does_not_force_danger_permission_flag(
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
            execution_mode="sandboxed",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert "--dangerously-skip-permissions" not in captured["command"]


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


def test_extract_claude_stream_json_usage_with_result_event() -> None:
    from agent_orchestrator.workers.run import _extract_claude_stream_json_usage

    stdout = "\n".join([
        '{"type":"system","subtype":"init"}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}',
        '{"type":"result","result":"hello","usage":{"input_tokens":150,"output_tokens":42,"cache_creation_input_tokens":10,"cache_read_input_tokens":5},"cost_usd":0.0037}',
    ])
    usage = _extract_claude_stream_json_usage(stdout)
    assert usage["provider_type"] == "claude"
    assert usage["input_tokens"] == 150
    assert usage["output_tokens"] == 42
    assert usage["cache_creation_input_tokens"] == 10
    assert usage["cache_read_input_tokens"] == 5
    assert abs(usage["cost_usd"] - 0.0037) < 1e-9


def test_extract_claude_stream_json_usage_no_result_event() -> None:
    from agent_orchestrator.workers.run import _extract_claude_stream_json_usage

    stdout = '{"type":"system","subtype":"init"}\n{"type":"assistant","message":{"content":[]}}'
    usage = _extract_claude_stream_json_usage(stdout)
    assert usage == {}


def test_extract_claude_stream_json_usage_empty_input() -> None:
    from agent_orchestrator.workers.run import _extract_claude_stream_json_usage

    assert _extract_claude_stream_json_usage("") == {}
    assert _extract_claude_stream_json_usage("   ") == {}


def test_run_worker_claude_extracts_token_usage(
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
            "\n".join([
                '{"type":"system","subtype":"init"}',
                '{"type":"assistant","message":{"content":[{"type":"text","text":"done"}]}}',
                '{"type":"result","result":"done","usage":{"input_tokens":200,"output_tokens":80},"cost_usd":0.005}',
            ]),
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

    monkeypatch.setattr("agent_orchestrator.workers.run._run_codex_worker", _fake_worker)

    result = run_worker(
        spec=WorkerProviderSpec(
            name="claude",
            type="claude",
            command="claude -p --verbose --output-format stream-json",
        ),
        prompt="test",
        project_dir=project_dir,
        run_dir=run_dir,
        timeout_seconds=60,
        heartbeat_seconds=30,
        heartbeat_grace_seconds=15,
        progress_path=progress_path,
    )

    assert result.token_usage["provider_type"] == "claude"
    assert result.token_usage["input_tokens"] == 200
    assert result.token_usage["output_tokens"] == 80
    assert abs(result.token_usage["cost_usd"] - 0.005) < 1e-9


def test_run_worker_codex_sets_provider_type_only(
    tmp_path: Path, monkeypatch
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress_path = run_dir / "progress.json"

    def _fake_worker(**_: object) -> dict[str, object]:
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

    monkeypatch.setattr("agent_orchestrator.workers.run._run_codex_worker", _fake_worker)
    monkeypatch.setattr("agent_orchestrator.workers.run._codex_supports_reasoning_effort", lambda _: False)

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

    assert result.token_usage == {"provider_type": "codex"}
    assert result.token_usage.get("input_tokens") is None
    assert result.token_usage.get("cost_usd") is None
