"""Tests for the enhanced task model (tasks/model.py)."""

from __future__ import annotations

import pytest

from feature_prd_runner.task_engine.model import (
    EffortEstimate,
    Task,
    TaskPriority,
    TaskSource,
    TaskStatus,
    TaskType,
    from_legacy_task,
)


class TestTaskCreation:
    def test_default_values(self) -> None:
        t = Task(title="Test task")
        assert t.title == "Test task"
        assert t.task_type == TaskType.FEATURE
        assert t.priority == TaskPriority.P2
        assert t.status == TaskStatus.BACKLOG
        assert t.labels == []
        assert t.blocked_by == []
        assert t.blocks == []
        assert t.assignee is None
        assert t.source == TaskSource.MANUAL
        assert t.id.startswith("task-")
        assert len(t.id) == 13  # "task-" + 8 hex chars

    def test_custom_values(self) -> None:
        t = Task(
            title="Fix login bug",
            description="Users can't log in with email",
            task_type=TaskType.BUG,
            priority=TaskPriority.P0,
            labels=["auth", "urgent"],
            effort=EffortEstimate.S,
            context_files=["src/auth.py"],
        )
        assert t.task_type == TaskType.BUG
        assert t.priority == TaskPriority.P0
        assert t.labels == ["auth", "urgent"]
        assert t.effort == EffortEstimate.S
        assert t.context_files == ["src/auth.py"]

    def test_id_generation_unique(self) -> None:
        ids = {Task().id for _ in range(100)}
        assert len(ids) == 100


class TestTaskSerialization:
    def test_round_trip(self) -> None:
        t = Task(
            title="Write docs",
            task_type=TaskType.DOCS,
            priority=TaskPriority.P3,
            status=TaskStatus.READY,
            labels=["docs"],
            effort=EffortEstimate.M,
            acceptance_criteria=["README updated"],
            metadata={"key": "value"},
        )
        d = t.to_dict()
        assert d["task_type"] == "docs"
        assert d["priority"] == "P3"
        assert d["status"] == "ready"
        assert d["effort"] == "M"
        assert d["metadata"] == {"key": "value"}

        restored = Task.from_dict(d)
        assert restored.title == t.title
        assert restored.task_type == t.task_type
        assert restored.priority == t.priority
        assert restored.status == t.status
        assert restored.labels == t.labels
        assert restored.effort == t.effort
        assert restored.metadata == t.metadata

    def test_from_dict_unknown_enum_defaults(self) -> None:
        d = {
            "id": "task-abc",
            "title": "Test",
            "task_type": "nonexistent_type",
            "priority": "PXXX",
            "status": "unknown_status",
        }
        t = Task.from_dict(d)
        assert t.task_type == TaskType.FEATURE  # default
        assert t.priority == TaskPriority.P2  # default
        assert t.status == TaskStatus.BACKLOG  # default

    def test_from_dict_missing_fields(self) -> None:
        t = Task.from_dict({"id": "task-min"})
        assert t.id == "task-min"
        assert t.title == ""
        assert t.description == ""

    def test_from_dict_none_lists(self) -> None:
        t = Task.from_dict({"id": "task-x", "labels": None, "blocked_by": None})
        assert t.labels == []
        assert t.blocked_by == []


class TestTaskStatusHelpers:
    def test_touch(self) -> None:
        t = Task(title="Test")
        old_ts = t.updated_at
        import time
        time.sleep(0.01)
        t.touch()
        assert t.updated_at >= old_ts

    def test_transition(self) -> None:
        t = Task(title="Test", status=TaskStatus.READY)
        t.transition(TaskStatus.IN_PROGRESS)
        assert t.status == TaskStatus.IN_PROGRESS
        assert t.completed_at is None

    def test_transition_done_sets_completed_at(self) -> None:
        t = Task(title="Test", status=TaskStatus.IN_PROGRESS)
        t.transition(TaskStatus.DONE)
        assert t.status == TaskStatus.DONE
        assert t.completed_at is not None

    def test_is_terminal(self) -> None:
        assert Task(status=TaskStatus.DONE).is_terminal
        assert Task(status=TaskStatus.CANCELLED).is_terminal
        assert not Task(status=TaskStatus.READY).is_terminal
        assert not Task(status=TaskStatus.IN_PROGRESS).is_terminal

    def test_is_actionable(self) -> None:
        assert Task(status=TaskStatus.READY).is_actionable
        assert not Task(status=TaskStatus.READY, blocked_by=["dep-1"]).is_actionable
        assert not Task(status=TaskStatus.BACKLOG).is_actionable


class TestTaskAssignment:
    def test_assign(self) -> None:
        t = Task(title="Test")
        t.assign("agent-impl-1", "agent")
        assert t.assignee == "agent-impl-1"
        assert t.assignee_type == "agent"

    def test_unassign(self) -> None:
        t = Task(title="Test", assignee="bob", assignee_type="human")
        t.unassign()
        assert t.assignee is None
        assert t.assignee_type is None


class TestTaskDependencies:
    def test_add_remove_blocked_by(self) -> None:
        t = Task(title="Test")
        t.add_blocked_by("dep-1")
        assert "dep-1" in t.blocked_by
        t.add_blocked_by("dep-1")  # idempotent
        assert t.blocked_by.count("dep-1") == 1
        t.remove_blocked_by("dep-1")
        assert "dep-1" not in t.blocked_by

    def test_add_remove_blocks(self) -> None:
        t = Task(title="Test")
        t.add_blocks("child-1")
        assert "child-1" in t.blocks
        t.remove_blocks("child-1")
        assert "child-1" not in t.blocks


class TestPrioritySort:
    def test_sort_key_ordering(self) -> None:
        assert TaskPriority.P0.sort_key < TaskPriority.P1.sort_key
        assert TaskPriority.P1.sort_key < TaskPriority.P2.sort_key
        assert TaskPriority.P2.sort_key < TaskPriority.P3.sort_key


class TestLegacyMigration:
    def test_basic_migration(self) -> None:
        legacy = {
            "id": "phase-1",
            "phase_id": "phase-1",
            "title": "Setup database",
            "description": "Create DB schema",
            "lifecycle": "ready",
            "step": "plan_impl",
            "priority": 0,
            "acceptance_criteria": ["Tables created", "Migrations pass"],
            "last_changed_files": ["src/db.py"],
        }
        task = from_legacy_task(legacy)
        assert task.title == "Setup database"
        assert task.description == "Create DB schema"
        assert task.task_type == TaskType.FEATURE
        assert task.status == TaskStatus.READY
        assert task.acceptance_criteria == ["Tables created", "Migrations pass"]
        assert task.context_files == ["src/db.py"]
        assert task.legacy_phase_id == "phase-1"
        assert task.legacy_task_id == "phase-1"
        assert task.source == TaskSource.LEGACY_MIGRATION
        assert "migrated" in task.labels

    def test_running_lifecycle_maps_to_in_progress(self) -> None:
        task = from_legacy_task({"id": "t1", "lifecycle": "running"})
        assert task.status == TaskStatus.IN_PROGRESS

    def test_waiting_human_maps_to_blocked(self) -> None:
        task = from_legacy_task({"id": "t1", "lifecycle": "waiting_human"})
        assert task.status == TaskStatus.BLOCKED

    def test_done_maps_to_done(self) -> None:
        task = from_legacy_task({"id": "t1", "lifecycle": "done"})
        assert task.status == TaskStatus.DONE

    def test_high_priority_mapping(self) -> None:
        task = from_legacy_task({"id": "t1", "priority": 2})
        assert task.priority == TaskPriority.P0
