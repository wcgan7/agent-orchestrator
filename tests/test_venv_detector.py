"""Tests for Python venv auto-detection and command prefixing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_orchestrator.runtime.orchestrator.venv_detector import (
    VenvInfo,
    _is_venv,
    detect_python_venv,
)
from agent_orchestrator.runtime.orchestrator.live_worker_adapter import (
    _apply_venv_to_defaults,
)


def _make_venv(path: Path) -> None:
    """Create a minimal fake venv directory with ``bin/python``."""
    bin_dir = path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "python").touch()


# ---------------------------------------------------------------------------
# _is_venv
# ---------------------------------------------------------------------------


class TestIsVenv:
    def test_valid_venv(self, tmp_path: Path) -> None:
        _make_venv(tmp_path / ".venv")
        assert _is_venv(tmp_path / ".venv") is True

    def test_directory_without_bin_python(self, tmp_path: Path) -> None:
        (tmp_path / ".venv").mkdir()
        assert _is_venv(tmp_path / ".venv") is False

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert _is_venv(tmp_path / "nonexistent") is False

    def test_symlinked_venv(self, tmp_path: Path) -> None:
        _make_venv(tmp_path / "real_venv")
        (tmp_path / "linked_venv").symlink_to(tmp_path / "real_venv")
        assert _is_venv(tmp_path / "linked_venv") is True


# ---------------------------------------------------------------------------
# detect_python_venv — project root candidates
# ---------------------------------------------------------------------------


class TestDetectProjectRoot:
    def test_dot_venv_detected(self, tmp_path: Path) -> None:
        _make_venv(tmp_path / ".venv")
        result = detect_python_venv(tmp_path)
        assert result is not None
        assert result.source == "project_dir/.venv"
        assert result.bin_dir == (tmp_path / ".venv" / "bin").resolve()

    def test_venv_detected(self, tmp_path: Path) -> None:
        _make_venv(tmp_path / "venv")
        result = detect_python_venv(tmp_path)
        assert result is not None
        assert result.source == "project_dir/venv"

    def test_env_detected(self, tmp_path: Path) -> None:
        _make_venv(tmp_path / "env")
        result = detect_python_venv(tmp_path)
        assert result is not None
        assert result.source == "project_dir/env"

    def test_priority_dot_venv_over_venv(self, tmp_path: Path) -> None:
        _make_venv(tmp_path / ".venv")
        _make_venv(tmp_path / "venv")
        result = detect_python_venv(tmp_path)
        assert result is not None
        assert result.source == "project_dir/.venv"

    def test_priority_venv_over_env(self, tmp_path: Path) -> None:
        _make_venv(tmp_path / "venv")
        _make_venv(tmp_path / "env")
        result = detect_python_venv(tmp_path)
        assert result is not None
        assert result.source == "project_dir/venv"

    def test_invalid_venv_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Directory named .venv without bin/python is not detected."""
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        (tmp_path / ".venv").mkdir()
        # Use a subdirectory so parent-dir fallback doesn't find anything real
        project = tmp_path / "project"
        project.mkdir()
        (project / ".venv").mkdir()
        result = detect_python_venv(project)
        assert result is None


# ---------------------------------------------------------------------------
# detect_python_venv — env var detection
# ---------------------------------------------------------------------------


class TestDetectEnvVar:
    def test_virtual_env_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        venv_path = tmp_path / "external_venv"
        _make_venv(venv_path)
        monkeypatch.setenv("VIRTUAL_ENV", str(venv_path))
        # Use a project dir with no local venv so we fall through to env var
        project = tmp_path / "project"
        project.mkdir()
        result = detect_python_venv(project)
        assert result is not None
        assert result.source == "env_var"

    def test_virtual_env_nonexistent_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "nonexistent"))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        project = tmp_path / "project"
        project.mkdir()
        result = detect_python_venv(project)
        assert result is None

    def test_conda_prefix_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        conda_path = tmp_path / "conda_env"
        _make_venv(conda_path)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("CONDA_PREFIX", str(conda_path))
        project = tmp_path / "project"
        project.mkdir()
        result = detect_python_venv(project)
        assert result is not None
        assert result.source == "conda"

    def test_project_root_beats_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Project root candidates take priority over VIRTUAL_ENV."""
        _make_venv(tmp_path / ".venv")
        ext = tmp_path / "external"
        _make_venv(ext)
        monkeypatch.setenv("VIRTUAL_ENV", str(ext))
        result = detect_python_venv(tmp_path)
        assert result is not None
        assert result.source == "project_dir/.venv"


# ---------------------------------------------------------------------------
# detect_python_venv — parent directory fallback
# ---------------------------------------------------------------------------


class TestDetectParentDir:
    def test_parent_dir_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        _make_venv(tmp_path / ".venv")
        project = tmp_path / "subproject"
        project.mkdir()
        result = detect_python_venv(project)
        assert result is not None
        assert result.source == "parent_dir/.venv"


# ---------------------------------------------------------------------------
# detect_python_venv — no venv found
# ---------------------------------------------------------------------------


class TestDetectNone:
    def test_returns_none_when_nothing_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        project = tmp_path / "project"
        project.mkdir()
        result = detect_python_venv(project)
        assert result is None


# ---------------------------------------------------------------------------
# _apply_venv_to_defaults
# ---------------------------------------------------------------------------


class TestApplyVenvToDefaults:
    def test_binary_exists_prefixed(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "pytest").touch()
        result = _apply_venv_to_defaults(
            {"test": "pytest", "lint": "ruff check ."},
            bin_dir,
            ".venv/bin",
        )
        assert result["test"] == ".venv/bin/pytest"
        assert result["lint"] == "ruff check ."  # ruff not in venv

    def test_binary_with_args(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "ruff").touch()
        result = _apply_venv_to_defaults(
            {"lint": "ruff check ."},
            bin_dir,
            ".venv/bin",
        )
        assert result["lint"] == ".venv/bin/ruff check ."

    def test_all_missing_unchanged(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        defaults = {"test": "pytest", "lint": "ruff check ."}
        result = _apply_venv_to_defaults(defaults, bin_dir, ".venv/bin")
        assert result == defaults

    def test_already_has_path_separator_skipped(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "pytest").touch()
        result = _apply_venv_to_defaults(
            {"test": "./some/path/pytest"},
            bin_dir,
            ".venv/bin",
        )
        assert result["test"] == "./some/path/pytest"

    def test_mixed_tools(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "pytest").touch()
        (bin_dir / "mypy").touch()
        result = _apply_venv_to_defaults(
            {"test": "pytest", "lint": "ruff check .", "typecheck": "mypy ."},
            bin_dir,
            ".venv/bin",
        )
        assert result["test"] == ".venv/bin/pytest"
        assert result["lint"] == "ruff check ."
        assert result["typecheck"] == ".venv/bin/mypy ."


# ---------------------------------------------------------------------------
# env_resolver integration
# ---------------------------------------------------------------------------


class TestEnvResolverVenvIntegration:
    def test_venv_detected_injects_virtual_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_orchestrator.runtime.orchestrator.env_resolver import resolve_env_vars
        from agent_orchestrator.runtime.domain.models import Task

        _make_venv(tmp_path / ".venv")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        task = Task(id="t1", title="test", description="", status="queued")
        result = resolve_env_vars(project_dir=tmp_path, cfg={}, task=task)
        assert result["VIRTUAL_ENV"] == str((tmp_path / ".venv").resolve())

    def test_venv_prepends_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_orchestrator.runtime.orchestrator.env_resolver import resolve_env_vars
        from agent_orchestrator.runtime.domain.models import Task

        _make_venv(tmp_path / ".venv")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        task = Task(id="t1", title="test", description="", status="queued")
        result = resolve_env_vars(project_dir=tmp_path, cfg={}, task=task)
        bin_dir = str((tmp_path / ".venv" / "bin").resolve())
        assert result["PATH"].startswith(bin_dir + os.pathsep)

    def test_config_virtual_env_overrides_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_orchestrator.runtime.orchestrator.env_resolver import resolve_env_vars
        from agent_orchestrator.runtime.domain.models import Task

        _make_venv(tmp_path / ".venv")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        cfg = {"workers": {"environment": {"env_vars": {"VIRTUAL_ENV": "/custom/path"}}}}
        task = Task(id="t1", title="test", description="", status="queued")
        result = resolve_env_vars(project_dir=tmp_path, cfg=cfg, task=task)
        assert result["VIRTUAL_ENV"] == "/custom/path"

    def test_no_venv_env_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_orchestrator.runtime.orchestrator.env_resolver import resolve_env_vars
        from agent_orchestrator.runtime.domain.models import Task

        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        project = tmp_path / "project"
        project.mkdir()
        task = Task(id="t1", title="test", description="", status="queued")
        result = resolve_env_vars(project_dir=project, cfg={}, task=task)
        assert "VIRTUAL_ENV" not in result or result.get("VIRTUAL_ENV") == os.environ.get("VIRTUAL_ENV")

    def test_resolved_view_shows_venv_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_orchestrator.runtime.orchestrator.env_resolver import resolved_env_vars_view

        _make_venv(tmp_path / ".venv")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        view = resolved_env_vars_view(project_dir=tmp_path, cfg={})
        venv_entries = [e for e in view if e["key"] == "VIRTUAL_ENV"]
        assert len(venv_entries) == 1
        assert venv_entries[0]["source"] == "venv"
        assert venv_entries[0]["has_value"] is True
