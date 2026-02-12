"""Tests for the quick action execution system: shortcuts + executor."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from feature_prd_runner.v3.domain.models import QuickActionRun
from feature_prd_runner.v3.quick_actions.shortcuts import (
    ShortcutMatch,
    ShortcutRule,
    load_shortcuts,
    match_prompt,
)
from feature_prd_runner.v3.quick_actions.executor import QuickActionExecutor
from feature_prd_runner.v3.storage.container import V3Container
from feature_prd_runner.v3.events.bus import EventBus


# ---------------------------------------------------------------------------
# Shortcut matching tests
# ---------------------------------------------------------------------------


def test_shortcut_match_run_tests(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
    rules = load_shortcuts(tmp_path)
    match = match_prompt("run tests", rules, tmp_path)
    assert match.matched is True
    assert match.shortcut_name == "run_tests"
    assert match.confidence == 1.0


def test_shortcut_match_tests(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
    rules = load_shortcuts(tmp_path)
    match = match_prompt("tests", rules, tmp_path)
    assert match.matched is True
    assert match.shortcut_name == "run_tests"


def test_shortcut_match_lint(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]")
    rules = load_shortcuts(tmp_path)
    match = match_prompt("lint", rules, tmp_path)
    assert match.matched is True
    assert match.shortcut_name == "lint"


def test_shortcut_match_git_status(tmp_path: Path) -> None:
    rules = load_shortcuts(tmp_path)
    match = match_prompt("git status", rules, tmp_path)
    assert match.matched is True
    assert match.shortcut_name == "git_status"
    assert match.command == "git status"


def test_shortcut_no_match_freeform(tmp_path: Path) -> None:
    rules = load_shortcuts(tmp_path)
    match = match_prompt("explain the auth flow", rules, tmp_path)
    assert match.matched is False


def test_shortcut_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
    rules = load_shortcuts(tmp_path)
    match = match_prompt("Run Tests", rules, tmp_path)
    assert match.matched is True
    assert match.shortcut_name == "run_tests"


def test_shortcut_auto_test_resolves_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
    rules = load_shortcuts(tmp_path)
    match = match_prompt("test", rules, tmp_path)
    assert match.matched is True
    assert "pytest" in match.command


def test_shortcut_auto_test_resolves_node(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}")
    rules = load_shortcuts(tmp_path)
    match = match_prompt("test", rules, tmp_path)
    assert match.matched is True
    assert "npm test" in match.command


def test_shortcut_auto_test_no_project_files(tmp_path: Path) -> None:
    """When no project files exist, auto:test cannot resolve → no match."""
    rules = load_shortcuts(tmp_path)
    match = match_prompt("test", rules, tmp_path)
    assert match.matched is False


def test_user_override_replaces_builtin(tmp_path: Path) -> None:
    config_dir = tmp_path / ".prd_runner"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "quick_shortcuts.yaml").write_text(
        "- name: run_tests\n"
        "  patterns:\n"
        "    - '^tests?$'\n"
        "  command: make test\n"
    )
    rules = load_shortcuts(tmp_path)
    names = [r.name for r in rules]
    assert names.count("run_tests") == 1
    rule = next(r for r in rules if r.name == "run_tests")
    assert rule.command == "make test"


def test_user_addition_alongside_builtins(tmp_path: Path) -> None:
    config_dir = tmp_path / ".prd_runner"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "quick_shortcuts.yaml").write_text(
        "- name: deploy\n"
        "  patterns:\n"
        "    - '^deploy$'\n"
        "  command: make deploy\n"
    )
    rules = load_shortcuts(tmp_path)
    names = [r.name for r in rules]
    assert "deploy" in names
    assert "git_status" in names  # builtins still present


# ---------------------------------------------------------------------------
# Executor tests
# ---------------------------------------------------------------------------


def _make_executor(tmp_path: Path) -> tuple[QuickActionExecutor, V3Container, MagicMock]:
    container = V3Container(tmp_path)
    mock_bus = MagicMock(spec=EventBus)
    executor = QuickActionExecutor(container, mock_bus)
    return executor, container, mock_bus


def test_executor_shortcut_success(tmp_path: Path) -> None:
    executor, container, mock_bus = _make_executor(tmp_path)
    run = QuickActionRun(prompt="echo hello")

    # Add a custom shortcut that matches "echo hello"
    config_dir = tmp_path / ".prd_runner"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "quick_shortcuts.yaml").write_text(
        "- name: echo_hello\n"
        "  patterns:\n"
        "    - '^echo hello$'\n"
        "  command: echo hello\n"
    )

    result = executor.execute(run)
    assert result.status == "completed"
    assert result.kind == "shortcut"
    assert result.exit_code == 0
    assert "hello" in (result.result_summary or "")


def test_executor_shortcut_failure(tmp_path: Path) -> None:
    executor, container, mock_bus = _make_executor(tmp_path)

    config_dir = tmp_path / ".prd_runner"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "quick_shortcuts.yaml").write_text(
        "- name: fail_cmd\n"
        "  patterns:\n"
        "    - '^fail$'\n"
        "  command: 'false'\n"
    )

    run = QuickActionRun(prompt="fail")
    result = executor.execute(run)
    assert result.status == "failed"
    assert result.kind == "shortcut"
    assert result.exit_code != 0


def test_executor_agent_fallback(tmp_path: Path) -> None:
    executor, container, mock_bus = _make_executor(tmp_path)
    run = QuickActionRun(prompt="explain the auth flow in detail")
    result = executor.execute(run)
    assert result.kind == "agent"
    # Without workers configured, it should fail gracefully
    assert result.status == "failed"
    assert "no worker" in (result.result_summary or "").lower()


def test_executor_status_flow(tmp_path: Path) -> None:
    """Verify the executor upserts through queued → running → completed."""
    executor, container, mock_bus = _make_executor(tmp_path)

    config_dir = tmp_path / ".prd_runner"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "quick_shortcuts.yaml").write_text(
        "- name: hello\n"
        "  patterns:\n"
        "    - '^hello$'\n"
        "  command: echo hi\n"
    )

    run = QuickActionRun(prompt="hello")
    result = executor.execute(run)

    # The run should end in completed
    assert result.status == "completed"
    # Check the persisted version
    persisted = container.quick_actions.get(result.id)
    assert persisted is not None
    assert persisted.status == "completed"


def test_executor_bus_events_emitted(tmp_path: Path) -> None:
    executor, container, mock_bus = _make_executor(tmp_path)

    config_dir = tmp_path / ".prd_runner"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "quick_shortcuts.yaml").write_text(
        "- name: hw\n"
        "  patterns:\n"
        "    - '^hw$'\n"
        "  command: echo hi\n"
    )

    run = QuickActionRun(prompt="hw")
    executor.execute(run)

    event_types = [call.kwargs.get("event_type") or call[1].get("event_type", "") for call in mock_bus.emit.call_args_list]
    assert "quick_action.started" in event_types
    assert "quick_action.completed" in event_types


def test_executor_git_status_shortcut(tmp_path: Path) -> None:
    """The built-in 'git status' shortcut executes directly."""
    executor, container, mock_bus = _make_executor(tmp_path)
    run = QuickActionRun(prompt="git status")
    result = executor.execute(run)
    assert result.kind == "shortcut"
    assert result.command == "git status"
    # git status will likely fail in a non-git dir, but the command ran
    assert result.exit_code is not None
