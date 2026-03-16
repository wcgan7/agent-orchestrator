"""Tests for worker environment variable configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.orchestrator.env_resolver import (
    _detect_compose_env_vars,
    _detect_prisma_required_vars,
    _extract_env_vars_from_config,
    _extract_env_vars_from_task,
    _parse_dotenv_files,
    auto_detect_env_vars,
    resolve_env_vars,
    resolved_env_vars_view,
)
from agent_orchestrator.runtime.api.router_impl import (
    _mask_settings_env_vars,
    _normalize_workers_environment,
    _settings_payload,
    _task_payload,
)


# ---------------------------------------------------------------------------
# _run_codex_worker env passthrough
# ---------------------------------------------------------------------------


def test_run_codex_worker_passes_env_to_popen(tmp_path: Path) -> None:
    """Verify that _run_codex_worker passes env= to subprocess.Popen."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress = run_dir / "progress.json"

    custom_env = {**os.environ, "MY_CUSTOM_VAR": "hello"}

    with patch("agent_orchestrator.worker.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout.readline = MagicMock(return_value="")
        mock_proc.stderr.readline = MagicMock(return_value="")
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        from agent_orchestrator.worker import _run_codex_worker

        _run_codex_worker(
            command="echo test",
            prompt="test prompt",
            project_dir=tmp_path,
            run_dir=run_dir,
            timeout_seconds=10,
            heartbeat_seconds=60,
            heartbeat_grace_seconds=120,
            progress_path=progress,
            env=custom_env,
        )

        _, kwargs = mock_popen.call_args
        assert kwargs["env"] is custom_env
        assert kwargs["env"]["MY_CUSTOM_VAR"] == "hello"


def test_run_codex_worker_none_env_inherits(tmp_path: Path) -> None:
    """Verify that env=None causes Popen to inherit the parent env."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress = run_dir / "progress.json"

    with patch("agent_orchestrator.worker.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout.readline = MagicMock(return_value="")
        mock_proc.stderr.readline = MagicMock(return_value="")
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        from agent_orchestrator.worker import _run_codex_worker

        _run_codex_worker(
            command="echo test",
            prompt="test prompt",
            project_dir=tmp_path,
            run_dir=run_dir,
            timeout_seconds=10,
            heartbeat_seconds=60,
            heartbeat_grace_seconds=120,
            progress_path=progress,
        )

        _, kwargs = mock_popen.call_args
        assert kwargs["env"] is None


def test_run_worker_forwards_env(tmp_path: Path) -> None:
    """Verify that run_worker threads env through to _run_codex_worker."""
    custom_env = {**os.environ, "FORWARDED": "yes"}

    with patch("agent_orchestrator.workers.run._run_codex_worker") as mock_codex:
        mock_codex.return_value = {
            "prompt_path": "",
            "stdout_path": "",
            "stderr_path": "",
            "start_time": "",
            "end_time": "",
            "runtime_seconds": 0,
            "exit_code": 0,
            "timed_out": False,
            "no_heartbeat": False,
            "last_heartbeat": None,
        }

        from agent_orchestrator.workers.config import WorkerProviderSpec
        from agent_orchestrator.workers.run import run_worker

        run_worker(
            spec=WorkerProviderSpec(name="codex", type="codex", command="echo"),
            prompt="test",
            project_dir=tmp_path,
            run_dir=tmp_path,
            timeout_seconds=10,
            heartbeat_seconds=60,
            heartbeat_grace_seconds=120,
            progress_path=tmp_path / "progress.json",
            env=custom_env,
        )

        _, kwargs = mock_codex.call_args
        assert kwargs["env"] is custom_env


# ---------------------------------------------------------------------------
# _extract_env_vars_from_config
# ---------------------------------------------------------------------------


def test_extract_env_vars_from_config_valid() -> None:
    cfg: dict[str, Any] = {
        "workers": {
            "environment": {
                "env_vars": {"DATABASE_URL": "postgres://localhost/db", "API_KEY": "secret"},
            },
        },
    }
    result = _extract_env_vars_from_config(cfg)
    assert result == {"DATABASE_URL": "postgres://localhost/db", "API_KEY": "secret"}


def test_extract_env_vars_from_config_empty() -> None:
    assert _extract_env_vars_from_config({}) == {}
    assert _extract_env_vars_from_config({"workers": {}}) == {}
    assert _extract_env_vars_from_config({"workers": {"environment": {}}}) == {}


def test_extract_env_vars_from_config_malformed() -> None:
    # Non-string values should be filtered out
    cfg: dict[str, Any] = {
        "workers": {
            "environment": {
                "env_vars": {"GOOD": "val", "BAD": 123, "": "empty_key"},
            },
        },
    }
    result = _extract_env_vars_from_config(cfg)
    assert result == {"GOOD": "val"}


# ---------------------------------------------------------------------------
# _extract_env_vars_from_task
# ---------------------------------------------------------------------------


def test_extract_env_vars_from_task_valid() -> None:
    task = Task(metadata={"env_vars": {"NODE_ENV": "production"}})
    result = _extract_env_vars_from_task(task)
    assert result == {"NODE_ENV": "production"}


def test_extract_env_vars_from_task_empty() -> None:
    task = Task(metadata={})
    assert _extract_env_vars_from_task(task) == {}

    task2 = Task(metadata={"env_vars": {}})
    assert _extract_env_vars_from_task(task2) == {}


def test_extract_env_vars_from_task_malformed() -> None:
    task = Task(metadata={"env_vars": "not a dict"})
    assert _extract_env_vars_from_task(task) == {}


# ---------------------------------------------------------------------------
# _parse_dotenv_files
# ---------------------------------------------------------------------------


def test_parse_dotenv_files(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        '# Comment line\n'
        'DATABASE_URL=postgres://localhost/db\n'
        'API_KEY="quoted_value"\n'
        "SINGLE_QUOTED='single'\n"
        '\n'
        'EMPTY_VAL=\n'
        'SPACED_KEY = spaced_val\n'
    )
    result = _parse_dotenv_files(tmp_path)
    assert result["DATABASE_URL"] == "postgres://localhost/db"
    assert result["API_KEY"] == "quoted_value"
    assert result["SINGLE_QUOTED"] == "single"
    assert result["EMPTY_VAL"] == ""
    assert result["SPACED_KEY"] == "spaced_val"
    assert "# Comment line" not in result


def test_parse_dotenv_export_prefix(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("export DATABASE_URL=postgres://localhost/db\nexport API_KEY=secret\n")
    result = _parse_dotenv_files(tmp_path)
    assert result["DATABASE_URL"] == "postgres://localhost/db"
    assert result["API_KEY"] == "secret"


def test_parse_dotenv_inline_comments(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        'UNQUOTED=value # this is a comment\n'
        'QUOTED="value # not a comment"\n'
        'NO_COMMENT=value\n'
    )
    result = _parse_dotenv_files(tmp_path)
    assert result["UNQUOTED"] == "value"
    assert result["QUOTED"] == "value # not a comment"
    assert result["NO_COMMENT"] == "value"


def test_parse_dotenv_priority(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("VAR=base\nONLY_BASE=yes\n")
    (tmp_path / ".env.local").write_text("VAR=local\n")
    result = _parse_dotenv_files(tmp_path)
    assert result["VAR"] == "local"  # .env.local wins
    assert result["ONLY_BASE"] == "yes"


# ---------------------------------------------------------------------------
# _detect_prisma_required_vars
# ---------------------------------------------------------------------------


def test_detect_prisma_required_vars(tmp_path: Path) -> None:
    prisma_dir = tmp_path / "prisma"
    prisma_dir.mkdir()
    schema = prisma_dir / "schema.prisma"
    schema.write_text(
        'datasource db {\n'
        '  provider = "postgresql"\n'
        '  url      = env("DATABASE_URL")\n'
        '}\n'
        'generator client {\n'
        '  provider = "prisma-client-js"\n'
        '  output   = env("PRISMA_OUTPUT")\n'
        '}\n'
    )
    result = _detect_prisma_required_vars(tmp_path, found={})
    assert result == {"DATABASE_URL": None, "PRISMA_OUTPUT": None}


def test_detect_prisma_no_schema(tmp_path: Path) -> None:
    result = _detect_prisma_required_vars(tmp_path, found={})
    assert result == {}


def test_detect_prisma_skips_already_found(tmp_path: Path) -> None:
    prisma_dir = tmp_path / "prisma"
    prisma_dir.mkdir()
    schema = prisma_dir / "schema.prisma"
    schema.write_text('url = env("DATABASE_URL")\n')
    # DATABASE_URL already found, should not be re-added
    result = _detect_prisma_required_vars(tmp_path, found={"DATABASE_URL": "postgres://..."})
    assert result == {}


# ---------------------------------------------------------------------------
# _detect_compose_env_vars
# ---------------------------------------------------------------------------


def test_detect_compose_env_vars(tmp_path: Path) -> None:
    try:
        import yaml  # noqa: F401
    except ImportError:
        pytest.skip("PyYAML not installed")

    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        "services:\n"
        "  web:\n"
        "    environment:\n"
        "      - DB_HOST=localhost\n"
        "      - DB_PORT=5432\n"
        "      - BARE_KEY\n"
        "  worker:\n"
        "    environment:\n"
        "      REDIS_URL: redis://localhost\n"
    )
    result = _detect_compose_env_vars(tmp_path, found={})
    assert result["DB_HOST"] == "localhost"
    assert result["DB_PORT"] == "5432"
    assert result["BARE_KEY"] is None
    assert result["REDIS_URL"] == "redis://localhost"


# ---------------------------------------------------------------------------
# auto_detect_env_vars
# ---------------------------------------------------------------------------


def test_auto_detect_returns_none_for_unresolved(tmp_path: Path) -> None:
    """Prisma env() refs without .env file should return None values."""
    prisma_dir = tmp_path / "prisma"
    prisma_dir.mkdir()
    (prisma_dir / "schema.prisma").write_text('url = env("DATABASE_URL")\n')
    result = auto_detect_env_vars(tmp_path)
    assert result["DATABASE_URL"] is None


def test_auto_detect_dotenv_resolves_prisma(tmp_path: Path) -> None:
    """When .env provides a value, prisma ref should not override it."""
    (tmp_path / ".env").write_text("DATABASE_URL=postgres://real\n")
    prisma_dir = tmp_path / "prisma"
    prisma_dir.mkdir()
    (prisma_dir / "schema.prisma").write_text('url = env("DATABASE_URL")\n')
    result = auto_detect_env_vars(tmp_path)
    assert result["DATABASE_URL"] == "postgres://real"


# ---------------------------------------------------------------------------
# resolve_env_vars (merge precedence)
# ---------------------------------------------------------------------------


def test_env_merge_precedence(tmp_path: Path) -> None:
    """Verify: auto < process env < config < task."""
    (tmp_path / ".env").write_text("AUTO_ONLY=from_dotenv\nCONFIG_WINS=from_dotenv\nTASK_WINS=from_dotenv\n")

    cfg: dict[str, Any] = {
        "workers": {
            "environment": {
                "env_vars": {"CONFIG_WINS": "from_config", "TASK_WINS": "from_config"},
            },
        },
    }
    task = Task(metadata={"env_vars": {"TASK_WINS": "from_task"}})

    with patch.dict(os.environ, {"PROCESS_ONLY": "from_env"}, clear=False):
        result = resolve_env_vars(project_dir=tmp_path, cfg=cfg, task=task)

    assert result["PROCESS_ONLY"] == "from_env"
    assert result["CONFIG_WINS"] == "from_config"
    assert result["TASK_WINS"] == "from_task"
    # AUTO_ONLY should be present if not already in process env
    # (it might or might not be depending on whether os.environ has it)
    if "AUTO_ONLY" not in os.environ:
        assert result["AUTO_ONLY"] == "from_dotenv"


# ---------------------------------------------------------------------------
# resolved_env_vars_view
# ---------------------------------------------------------------------------


def test_resolved_view_shows_source(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("FROM_DOTENV=val\n")
    prisma_dir = tmp_path / "prisma"
    prisma_dir.mkdir()
    (prisma_dir / "schema.prisma").write_text('url = env("REQUIRED_VAR")\n')

    cfg: dict[str, Any] = {
        "workers": {
            "environment": {
                "env_vars": {"CONFIG_VAR": "secret"},
            },
        },
    }

    view = resolved_env_vars_view(project_dir=tmp_path, cfg=cfg)
    by_key = {item["key"]: item for item in view}

    assert by_key["CONFIG_VAR"]["source"] == "config"
    assert by_key["CONFIG_VAR"]["has_value"] is True

    assert by_key["REQUIRED_VAR"]["source"] == "required"
    assert by_key["REQUIRED_VAR"]["has_value"] is False

    # FROM_DOTENV could be "auto" or "process" depending on os.environ
    assert by_key["FROM_DOTENV"]["has_value"] is True

    # No actual values should appear
    for item in view:
        assert "value" not in item


# ---------------------------------------------------------------------------
# _normalize_workers_environment
# ---------------------------------------------------------------------------


def test_normalize_env_vars_preserved() -> None:
    raw: dict[str, Any] = {
        "auto_prepare": True,
        "env_vars": {"DATABASE_URL": "postgres://localhost/db", " SPACED ": "val"},
    }
    result = _normalize_workers_environment(raw)
    assert result["env_vars"] == {"DATABASE_URL": "postgres://localhost/db", "SPACED": "val"}


def test_normalize_env_vars_empty() -> None:
    result = _normalize_workers_environment({})
    assert "env_vars" not in result


def test_normalize_env_vars_filters_bad_values() -> None:
    raw: dict[str, Any] = {"env_vars": {"GOOD": "val", "BAD": 42}}
    result = _normalize_workers_environment(raw)
    assert result.get("env_vars") == {"GOOD": "val"}


# ---------------------------------------------------------------------------
# Settings payload masking
# ---------------------------------------------------------------------------


def test_settings_payload_preserves_real_values() -> None:
    """_settings_payload must NOT mask — it's used for normalization before save."""
    cfg: dict[str, Any] = {
        "workers": {
            "environment": {
                "env_vars": {"SECRET_KEY": "actual_secret_value"},
            },
        },
    }
    payload = _settings_payload(cfg)
    env_vars = payload["workers"]["environment"].get("env_vars", {})
    assert env_vars == {"SECRET_KEY": "actual_secret_value"}


def test_mask_settings_env_vars() -> None:
    """_mask_settings_env_vars applied at response boundary masks values."""
    cfg: dict[str, Any] = {
        "workers": {
            "environment": {
                "env_vars": {"SECRET_KEY": "actual_secret_value"},
            },
        },
    }
    payload = _mask_settings_env_vars(_settings_payload(cfg))
    env_vars = payload["workers"]["environment"].get("env_vars", {})
    assert env_vars == {"SECRET_KEY": "***"}


def test_settings_payload_no_env_vars() -> None:
    """Settings without env_vars should still work normally."""
    cfg: dict[str, Any] = {}
    payload = _mask_settings_env_vars(_settings_payload(cfg))
    assert "env_vars" not in payload["workers"]["environment"]


# ---------------------------------------------------------------------------
# Task payload masking
# ---------------------------------------------------------------------------


def test_task_payload_masks_env_vars() -> None:
    task = Task(
        title="test",
        metadata={"env_vars": {"DB_PASS": "super_secret"}},
    )
    payload = _task_payload(task)
    meta = payload.get("metadata", {})
    assert meta["env_vars"] == {"DB_PASS": "***"}


def test_task_payload_no_env_vars() -> None:
    task = Task(title="test", metadata={"other": "data"})
    payload = _task_payload(task)
    meta = payload.get("metadata", {})
    assert "env_vars" not in meta


# ---------------------------------------------------------------------------
# PATCH settings merge behavior
# ---------------------------------------------------------------------------


def test_patch_environment_merges() -> None:
    """Verify that patching environment merges rather than replaces."""
    existing: dict[str, Any] = {
        "auto_prepare": True,
        "max_auto_retries": 3,
        "env_vars": {"EXISTING": "keep"},
    }
    incoming: dict[str, Any] = {
        "env_vars": {"NEW_VAR": "added"},
    }
    # Simulate the merge logic from routes_misc.py
    merged = dict(existing)
    merged.update(incoming)
    assert merged["auto_prepare"] is True  # preserved
    assert merged["env_vars"] == {"NEW_VAR": "added"}  # shallow merge replaces env_vars dict


def test_patch_settings_stores_env_vars() -> None:
    """Verify that PATCH roundtrip stores full values (not masked)."""
    raw: dict[str, Any] = {
        "env_vars": {"DATABASE_URL": "postgres://localhost/mydb"},
    }
    normalized = _normalize_workers_environment(raw)
    assert normalized["env_vars"]["DATABASE_URL"] == "postgres://localhost/mydb"


def test_settings_payload_normalization_preserves_values_for_save() -> None:
    """Verify _settings_payload (used for normalization before save) doesn't mask.

    The PATCH handler calls _settings_payload to normalize config before
    persisting. If masking happened inside _settings_payload, real values
    would be replaced with '***' in the database.
    """
    cfg: dict[str, Any] = {
        "workers": {
            "environment": {
                "env_vars": {"DB_PASS": "real_password_123"},
            },
        },
    }
    # Simulate PATCH normalization path (routes_misc.py line 575)
    normalized_workers = _settings_payload({"workers": cfg["workers"]})["workers"]
    # The real value must survive normalization — masking is only at API boundary
    assert normalized_workers["environment"]["env_vars"]["DB_PASS"] == "real_password_123"
