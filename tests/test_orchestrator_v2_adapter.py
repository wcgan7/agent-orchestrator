"""Tests for v2 task adapter execution path in orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from feature_prd_runner import orchestrator
from feature_prd_runner.task_engine.engine import TaskEngine
from feature_prd_runner.task_engine.model import TaskStatus


def test_v2_tasks_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FEATURE_PRD_USE_V2_TASKS", raising=False)
    monkeypatch.delenv("FEATURE_PRD_DISABLE_V2_TASKS", raising=False)
    assert orchestrator._v2_tasks_enabled() is True


def test_v2_tasks_can_be_temporarily_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FEATURE_PRD_DISABLE_V2_TASKS", "true")
    assert orchestrator._v2_tasks_enabled() is False


@pytest.mark.anyio
async def test_v2_adapter_success_moves_to_in_review_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    state_dir = project_dir / ".prd_runner"
    state_dir.mkdir()

    engine = TaskEngine(state_dir)
    task = engine.create_task(title="Implement profile page")
    engine.transition_task(task.id, "ready")

    monkeypatch.delenv("FEATURE_PRD_AUTO_APPROVE_REVIEW", raising=False)
    monkeypatch.setattr(orchestrator, "execute_custom_prompt", lambda **_: (True, None))

    run_state: dict[str, object] = {
        "status": "idle",
        "current_task_id": None,
        "run_id": None,
        "last_error": None,
    }

    handled = orchestrator._run_next_v2_task(
        project_dir=project_dir,
        state_dir=state_dir,
        run_state=run_state,
        shift_minutes=30,
        heartbeat_seconds=5,
        heartbeat_grace_seconds=120,
        override_agents=False,
    )
    assert handled is True

    updated = engine.get_task(task.id)
    assert updated is not None
    assert updated.status == TaskStatus.IN_REVIEW
    assert run_state["status"] == "blocked"
    assert run_state["current_task_id"] == task.id

    events = engine.get_task_events(task.id, limit=20)
    types = [evt.get("type") for evt in events]
    assert "task.step_started" in types
    assert "task.step_completed" in types
    assert "task.awaiting_human_review" in types


@pytest.mark.anyio
async def test_v2_adapter_success_auto_approve_moves_to_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    state_dir = project_dir / ".prd_runner"
    state_dir.mkdir()

    engine = TaskEngine(state_dir)
    task = engine.create_task(title="Implement account settings")
    engine.transition_task(task.id, "ready")

    monkeypatch.setenv("FEATURE_PRD_AUTO_APPROVE_REVIEW", "true")
    monkeypatch.setattr(orchestrator, "execute_custom_prompt", lambda **_: (True, None))

    run_state: dict[str, object] = {
        "status": "idle",
        "current_task_id": None,
        "run_id": None,
        "last_error": None,
    }

    handled = orchestrator._run_next_v2_task(
        project_dir=project_dir,
        state_dir=state_dir,
        run_state=run_state,
        shift_minutes=30,
        heartbeat_seconds=5,
        heartbeat_grace_seconds=120,
        override_agents=False,
    )
    assert handled is True

    updated = engine.get_task(task.id)
    assert updated is not None
    assert updated.status == TaskStatus.DONE
    assert run_state["status"] == "idle"
    assert run_state["current_task_id"] is None
    assert run_state["run_id"] is None
