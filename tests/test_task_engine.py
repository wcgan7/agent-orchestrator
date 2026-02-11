"""Tests for the task engine (tasks/engine.py) and store (tasks/store.py)."""

from __future__ import annotations

import pytest
from pathlib import Path

from feature_prd_runner.task_engine.model import (
    Task,
    TaskPriority,
    TaskSource,
    TaskStatus,
    TaskType,
)
from feature_prd_runner.task_engine.engine import TaskEngine
from feature_prd_runner.task_engine.store import TaskStore


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".prd_runner"
    d.mkdir()
    return d


@pytest.fixture
def engine(state_dir: Path) -> TaskEngine:
    return TaskEngine(state_dir)


@pytest.fixture
def store(state_dir: Path) -> TaskStore:
    return TaskStore(state_dir)


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

class TestTaskStore:
    def test_empty_read(self, store: TaskStore) -> None:
        tasks = store.read_snapshot()
        assert tasks == []

    def test_add_and_read(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            tx.add(Task(id="t1", title="First"))
            tx.add(Task(id="t2", title="Second"))

        tasks = store.read_snapshot()
        assert len(tasks) == 2
        assert tasks[0].id == "t1"
        assert tasks[1].id == "t2"

    def test_duplicate_add_raises(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            tx.add(Task(id="t1", title="First"))
            with pytest.raises(ValueError, match="already exists"):
                tx.add(Task(id="t1", title="Duplicate"))

    def test_get_one(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            tx.add(Task(id="t1", title="Test"))

        t = store.get_one("t1")
        assert t is not None
        assert t.title == "Test"

        assert store.get_one("nonexistent") is None

    def test_update(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            tx.add(Task(id="t1", title="Old title"))

        with store.transaction() as tx:
            result = tx.update("t1", {"title": "New title"})
            assert result is not None
            assert result.title == "New title"

        t = store.get_one("t1")
        assert t is not None
        assert t.title == "New title"

    def test_update_nonexistent(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            assert tx.update("nope", {"title": "x"}) is None

    def test_soft_remove(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            tx.add(Task(id="t1", title="Test", status=TaskStatus.READY))

        with store.transaction() as tx:
            assert tx.remove("t1")

        t = store.get_one("t1")
        assert t is not None
        assert t.status == TaskStatus.CANCELLED

    def test_hard_remove(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            tx.add(Task(id="t1", title="Test"))

        with store.transaction() as tx:
            assert tx.hard_remove("t1")

        assert store.get_one("t1") is None

    def test_find_filters(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            tx.add(Task(id="t1", title="Bug fix", task_type=TaskType.BUG, priority=TaskPriority.P0, labels=["auth"]))
            tx.add(Task(id="t2", title="Feature", task_type=TaskType.FEATURE, priority=TaskPriority.P2))
            tx.add(Task(id="t3", title="Docs update", task_type=TaskType.DOCS, labels=["auth"]))

        with store.transaction() as tx:
            bugs = tx.find(task_type="bug")
            assert len(bugs) == 1
            assert bugs[0].id == "t1"

            p0 = tx.find(priority="P0")
            assert len(p0) == 1

            auth = tx.find(label="auth")
            assert len(auth) == 2

            searched = tx.find(search="docs")
            assert len(searched) == 1
            assert searched[0].id == "t3"

    def test_reorder(self, store: TaskStore) -> None:
        with store.transaction() as tx:
            tx.add(Task(id="t1", title="First"))
            tx.add(Task(id="t2", title="Second"))
            tx.add(Task(id="t3", title="Third"))

        with store.transaction() as tx:
            tx.reorder(["t3", "t1", "t2"])

        tasks = store.read_snapshot()
        assert [t.id for t in tasks] == ["t3", "t1", "t2"]

    def test_persistence_survives_reload(self, state_dir: Path) -> None:
        store1 = TaskStore(state_dir)
        with store1.transaction() as tx:
            tx.add(Task(id="t1", title="Persistent"))

        store2 = TaskStore(state_dir)
        tasks = store2.read_snapshot()
        assert len(tasks) == 1
        assert tasks[0].title == "Persistent"


# ---------------------------------------------------------------------------
# Engine CRUD tests
# ---------------------------------------------------------------------------

class TestEngineCRUD:
    def test_create_task(self, engine: TaskEngine) -> None:
        t = engine.create_task(
            title="Add login",
            description="OAuth2 login flow",
            task_type="feature",
            priority="P1",
            labels=["auth"],
            acceptance_criteria=["Login works", "Logout works"],
        )
        assert t.title == "Add login"
        assert t.task_type == TaskType.FEATURE
        assert t.priority == TaskPriority.P1
        assert t.labels == ["auth"]
        assert len(t.acceptance_criteria) == 2

        fetched = engine.get_task(t.id)
        assert fetched is not None
        assert fetched.title == "Add login"

    def test_create_with_parent(self, engine: TaskEngine) -> None:
        parent = engine.create_task(title="Parent feature")
        child = engine.create_task(title="Subtask", parent_id=parent.id)
        assert child.parent_id == parent.id

        # Parent should have child in children_ids
        updated_parent = engine.get_task(parent.id)
        assert updated_parent is not None
        assert child.id in updated_parent.children_ids

    def test_list_tasks_with_filters(self, engine: TaskEngine) -> None:
        engine.create_task(title="Bug", task_type="bug", priority="P0")
        engine.create_task(title="Feature", task_type="feature", priority="P2")
        engine.create_task(title="Docs", task_type="docs")

        all_tasks = engine.list_tasks()
        assert len(all_tasks) == 3

        bugs = engine.list_tasks(task_type="bug")
        assert len(bugs) == 1
        assert bugs[0].title == "Bug"

    def test_update_task(self, engine: TaskEngine) -> None:
        t = engine.create_task(title="Original")
        updated = engine.update_task(t.id, {"title": "Updated", "priority": "P0"})
        assert updated is not None
        assert updated.title == "Updated"
        assert updated.priority == TaskPriority.P0

    def test_delete_task(self, engine: TaskEngine) -> None:
        t = engine.create_task(title="To delete")
        assert engine.delete_task(t.id)
        fetched = engine.get_task(t.id)
        assert fetched is not None
        assert fetched.status == TaskStatus.CANCELLED

    def test_bulk_create(self, engine: TaskEngine) -> None:
        tasks = [
            Task(title="Task 1"),
            Task(title="Task 2"),
            Task(title="Task 3"),
        ]
        created = engine.bulk_create(tasks)
        assert len(created) == 3
        assert len(engine.list_tasks()) == 3


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    def test_valid_transition(self, engine: TaskEngine) -> None:
        t = engine.create_task(title="Test")
        # backlog → ready
        result = engine.transition_task(t.id, "ready")
        assert result is not None
        assert result.status == TaskStatus.READY

        # ready → in_progress
        result = engine.transition_task(t.id, "in_progress")
        assert result is not None
        assert result.status == TaskStatus.IN_PROGRESS

    def test_invalid_transition_raises(self, engine: TaskEngine) -> None:
        t = engine.create_task(title="Test")
        # backlog → done should fail
        with pytest.raises(ValueError, match="Cannot transition"):
            engine.transition_task(t.id, "done")

    def test_done_unblocks_dependents(self, engine: TaskEngine) -> None:
        t1 = engine.create_task(title="First")
        t2 = engine.create_task(title="Second")

        engine.transition_task(t1.id, "ready")
        engine.transition_task(t2.id, "ready")
        engine.add_dependency(t2.id, t1.id)

        t2_check = engine.get_task(t2.id)
        assert t2_check is not None
        assert t2_check.status == TaskStatus.BLOCKED

        # Complete t1 → t2 should become ready
        engine.transition_task(t1.id, "in_progress")
        engine.transition_task(t1.id, "in_review")
        engine.transition_task(t1.id, "done")

        t2_after = engine.get_task(t2.id)
        assert t2_after is not None
        assert t2_after.status == TaskStatus.READY
        assert t1.id not in t2_after.blocked_by


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

class TestDependencies:
    def test_add_dependency(self, engine: TaskEngine) -> None:
        t1 = engine.create_task(title="First")
        t2 = engine.create_task(title="Second")
        engine.transition_task(t1.id, "ready")
        engine.transition_task(t2.id, "ready")

        engine.add_dependency(t2.id, t1.id)

        t2_check = engine.get_task(t2.id)
        assert t2_check is not None
        assert t1.id in t2_check.blocked_by

        t1_check = engine.get_task(t1.id)
        assert t1_check is not None
        assert t2.id in t1_check.blocks

    def test_self_dependency_raises(self, engine: TaskEngine) -> None:
        t = engine.create_task(title="Self")
        with pytest.raises(ValueError, match="cannot depend on itself"):
            engine.add_dependency(t.id, t.id)

    def test_cycle_detection(self, engine: TaskEngine) -> None:
        t1 = engine.create_task(title="A")
        t2 = engine.create_task(title="B")
        t3 = engine.create_task(title="C")

        engine.add_dependency(t2.id, t1.id)  # B depends on A
        engine.add_dependency(t3.id, t2.id)  # C depends on B

        # A depends on C would create A→C→B→A cycle
        with pytest.raises(ValueError, match="cycle"):
            engine.add_dependency(t1.id, t3.id)

    def test_remove_dependency(self, engine: TaskEngine) -> None:
        t1 = engine.create_task(title="First")
        t2 = engine.create_task(title="Second")
        engine.transition_task(t1.id, "ready")
        engine.transition_task(t2.id, "ready")

        engine.add_dependency(t2.id, t1.id)
        assert engine.get_task(t2.id).status == TaskStatus.BLOCKED

        engine.remove_dependency(t2.id, t1.id)
        t2_after = engine.get_task(t2.id)
        assert t2_after is not None
        assert t1.id not in t2_after.blocked_by
        assert t2_after.status == TaskStatus.READY

    def test_get_ready_tasks(self, engine: TaskEngine) -> None:
        t1 = engine.create_task(title="Ready P0", priority="P0")
        t2 = engine.create_task(title="Ready P2", priority="P2")
        t3 = engine.create_task(title="Blocked")

        engine.transition_task(t1.id, "ready")
        engine.transition_task(t2.id, "ready")
        engine.transition_task(t3.id, "ready")
        engine.add_dependency(t3.id, t1.id)

        ready = engine.get_ready_tasks()
        ready_ids = [t.id for t in ready]
        assert t1.id in ready_ids
        assert t2.id in ready_ids
        assert t3.id not in ready_ids
        # P0 should come first
        assert ready_ids.index(t1.id) < ready_ids.index(t2.id)

    def test_execution_order(self, engine: TaskEngine) -> None:
        t1 = engine.create_task(title="A")
        t2 = engine.create_task(title="B")
        t3 = engine.create_task(title="C")

        engine.add_dependency(t2.id, t1.id)
        engine.add_dependency(t3.id, t2.id)

        batches = engine.get_execution_order()
        assert len(batches) >= 2
        # t1 must be in first batch, t3 must be in last batch
        assert t1.id in batches[0]
        assert t3.id in batches[-1]

    def test_dependency_graph(self, engine: TaskEngine) -> None:
        t1 = engine.create_task(title="A")
        t2 = engine.create_task(title="B")
        engine.add_dependency(t2.id, t1.id)

        graph = engine.get_dependency_graph()
        assert t1.id in graph[t2.id]


# ---------------------------------------------------------------------------
# Board view
# ---------------------------------------------------------------------------

class TestBoardView:
    def test_get_board(self, engine: TaskEngine) -> None:
        engine.create_task(title="Backlog item")
        t2 = engine.create_task(title="Ready item")
        engine.transition_task(t2.id, "ready")
        t3 = engine.create_task(title="In progress")
        engine.transition_task(t3.id, "ready")
        engine.transition_task(t3.id, "in_progress")

        board = engine.get_board()
        assert "backlog" in board
        assert "ready" in board
        assert "in_progress" in board
        assert "done" in board
        assert len(board["backlog"]) == 1
        assert len(board["ready"]) == 1
        assert len(board["in_progress"]) == 1

    def test_cancelled_hidden_from_board(self, engine: TaskEngine) -> None:
        t = engine.create_task(title="To cancel")
        engine.delete_task(t.id)

        board = engine.get_board()
        all_ids = [
            task["id"]
            for col in board.values()
            for task in col
        ]
        assert t.id not in all_ids


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

class TestAssignment:
    def test_assign_and_unassign(self, engine: TaskEngine) -> None:
        t = engine.create_task(title="Test")
        engine.assign_task(t.id, "agent-impl-1", "agent")
        fetched = engine.get_task(t.id)
        assert fetched is not None
        assert fetched.assignee == "agent-impl-1"
        assert fetched.assignee_type == "agent"

        engine.unassign_task(t.id)
        fetched2 = engine.get_task(t.id)
        assert fetched2 is not None
        assert fetched2.assignee is None
