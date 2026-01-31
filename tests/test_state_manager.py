"""Tests for state_manager module."""

import pytest
from pathlib import Path

from feature_prd_runner.state_manager import (
    StateSnapshot,
    state_transaction,
    load_task_state_quick,
)
from feature_prd_runner.io_utils import _load_data, _save_data


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Create a temporary state directory."""
    state_dir = tmp_path / ".prd_runner"
    state_dir.mkdir(parents=True)
    return state_dir


@pytest.fixture
def paths(state_dir: Path) -> dict[str, Path]:
    """Create standard paths for testing."""
    return {
        "lock": state_dir / "run.lock",
        "task_queue": state_dir / "task_queue.yaml",
        "phase_plan": state_dir / "phase_plan.yaml",
    }


@pytest.fixture
def sample_queue() -> dict:
    """Sample task queue data."""
    return {
        "tasks": [
            {
                "id": "task-001",
                "status": "todo",
                "lifecycle": "ready",
                "step": "plan_impl",
                "phase_id": "phase-1",
            },
            {
                "id": "task-002",
                "status": "done",
                "lifecycle": "done",
                "step": "commit",
                "phase_id": "phase-2",
            },
        ],
    }


@pytest.fixture
def sample_plan() -> dict:
    """Sample phase plan data."""
    return {
        "phases": [
            {"id": "phase-1", "name": "Phase 1", "status": "todo"},
            {"id": "phase-2", "name": "Phase 2", "status": "done"},
        ],
    }


class TestStateSnapshot:
    """Tests for StateSnapshot dataclass."""

    def test_find_task_existing(self, sample_queue: dict):
        """Test finding an existing task."""
        from feature_prd_runner.tasks import _normalize_tasks

        tasks = _normalize_tasks(sample_queue)
        snapshot = StateSnapshot(queue=sample_queue, tasks=tasks)

        task = snapshot.find_task("task-001")
        assert task is not None
        assert task["id"] == "task-001"

    def test_find_task_nonexistent(self, sample_queue: dict):
        """Test finding a nonexistent task."""
        from feature_prd_runner.tasks import _normalize_tasks

        tasks = _normalize_tasks(sample_queue)
        snapshot = StateSnapshot(queue=sample_queue, tasks=tasks)

        task = snapshot.find_task("nonexistent")
        assert task is None

    def test_phase_for_task_with_phases(self, sample_queue: dict, sample_plan: dict):
        """Test getting phase for task when phases are loaded."""
        from feature_prd_runner.tasks import _normalize_tasks, _normalize_phases

        tasks = _normalize_tasks(sample_queue)
        phases = _normalize_phases(sample_plan)
        snapshot = StateSnapshot(
            queue=sample_queue, tasks=tasks, plan=sample_plan, phases=phases
        )

        task = snapshot.find_task("task-001")
        phase = snapshot.phase_for_task(task)
        assert phase is not None
        assert phase["id"] == "phase-1"

    def test_phase_for_task_without_phases(self, sample_queue: dict):
        """Test getting phase for task when phases are not loaded."""
        from feature_prd_runner.tasks import _normalize_tasks

        tasks = _normalize_tasks(sample_queue)
        snapshot = StateSnapshot(queue=sample_queue, tasks=tasks)

        task = snapshot.find_task("task-001")
        phase = snapshot.phase_for_task(task)
        assert phase is None


class TestStateTransaction:
    """Tests for state_transaction context manager."""

    def test_basic_load_and_save(self, paths: dict[str, Path], sample_queue: dict):
        """Test basic load and save functionality."""
        _save_data(paths["task_queue"], sample_queue)

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
        ) as state:
            assert len(state.tasks) == 2
            task = state.find_task("task-001")
            assert task is not None
            task["status"] = "running"

        # Verify changes were saved
        reloaded = _load_data(paths["task_queue"], {})
        tasks = reloaded.get("tasks", [])
        task_001 = next((t for t in tasks if t.get("id") == "task-001"), None)
        assert task_001 is not None
        assert task_001["status"] == "running"

    def test_load_without_plan(self, paths: dict[str, Path], sample_queue: dict):
        """Test loading without phase plan (efficiency mode)."""
        _save_data(paths["task_queue"], sample_queue)

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
            load_plan=False,
        ) as state:
            assert state.plan is None
            assert state.phases is None
            assert state.tasks is not None

    def test_load_with_plan(
        self, paths: dict[str, Path], sample_queue: dict, sample_plan: dict
    ):
        """Test loading with phase plan."""
        _save_data(paths["task_queue"], sample_queue)
        _save_data(paths["phase_plan"], sample_plan)

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
            paths["phase_plan"],
            load_plan=True,
        ) as state:
            assert state.plan is not None
            assert state.phases is not None
            assert len(state.phases) == 2

    def test_save_plan_when_modified(
        self, paths: dict[str, Path], sample_queue: dict, sample_plan: dict
    ):
        """Test saving phase plan when save_plan=True."""
        _save_data(paths["task_queue"], sample_queue)
        _save_data(paths["phase_plan"], sample_plan)

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
            paths["phase_plan"],
            load_plan=True,
            save_plan=True,
        ) as state:
            phase = state.phases[0]
            phase["status"] = "in_progress"

        # Verify plan was saved
        reloaded = _load_data(paths["phase_plan"], {})
        phases = reloaded.get("phases", [])
        phase_1 = next((p for p in phases if p.get("id") == "phase-1"), None)
        assert phase_1 is not None
        assert phase_1["status"] == "in_progress"

    def test_no_save_plan_by_default(
        self, paths: dict[str, Path], sample_queue: dict, sample_plan: dict
    ):
        """Test that phase plan is not saved by default."""
        _save_data(paths["task_queue"], sample_queue)
        _save_data(paths["phase_plan"], sample_plan)

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
            paths["phase_plan"],
            load_plan=True,
            save_plan=False,
        ) as state:
            phase = state.phases[0]
            phase["status"] = "modified_but_not_saved"

        # Verify plan was NOT saved
        reloaded = _load_data(paths["phase_plan"], {})
        phases = reloaded.get("phases", [])
        phase_1 = next((p for p in phases if p.get("id") == "phase-1"), None)
        assert phase_1 is not None
        assert phase_1["status"] == "todo"  # Original value

    def test_requires_phase_plan_path_when_load_plan(
        self, paths: dict[str, Path], sample_queue: dict
    ):
        """Test that ValueError is raised when load_plan=True without path."""
        _save_data(paths["task_queue"], sample_queue)

        with pytest.raises(ValueError, match="phase_plan_path is required"):
            with state_transaction(
                paths["lock"],
                paths["task_queue"],
                None,
                load_plan=True,
            ) as state:
                pass

    def test_creates_empty_queue_if_missing(self, paths: dict[str, Path]):
        """Test handling of missing task queue file."""
        with state_transaction(
            paths["lock"],
            paths["task_queue"],
        ) as state:
            assert state.queue == {}
            assert state.tasks == []

    def test_exception_does_not_save(
        self, paths: dict[str, Path], sample_queue: dict
    ):
        """Test that state is not saved when exception occurs."""
        _save_data(paths["task_queue"], sample_queue)

        try:
            with state_transaction(
                paths["lock"],
                paths["task_queue"],
            ) as state:
                task = state.find_task("task-001")
                task["status"] = "should_not_be_saved"
                raise RuntimeError("Intentional error")
        except RuntimeError:
            pass

        # Verify changes were NOT saved due to exception
        reloaded = _load_data(paths["task_queue"], {})
        tasks = reloaded.get("tasks", [])
        task_001 = next((t for t in tasks if t.get("id") == "task-001"), None)
        assert task_001 is not None
        assert task_001["status"] == "todo"  # Original value

    def test_adds_task_to_queue(self, paths: dict[str, Path], sample_queue: dict):
        """Test adding a new task."""
        _save_data(paths["task_queue"], sample_queue)

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
        ) as state:
            new_task = {
                "id": "task-003",
                "status": "todo",
                "lifecycle": "ready",
                "step": "plan_impl",
            }
            state.tasks.append(new_task)

        # Verify task was added
        reloaded = _load_data(paths["task_queue"], {})
        tasks = reloaded.get("tasks", [])
        assert len(tasks) == 3
        task_003 = next((t for t in tasks if t.get("id") == "task-003"), None)
        assert task_003 is not None

    def test_concurrent_safe_with_lock(self, paths: dict[str, Path], sample_queue: dict):
        """Test that lock prevents concurrent access issues."""
        import threading
        import time

        _save_data(paths["task_queue"], sample_queue)

        results = []

        def increment_counter(thread_id: int):
            with state_transaction(
                paths["lock"],
                paths["task_queue"],
            ) as state:
                task = state.find_task("task-001")
                if task:
                    current = task.get("counter", 0)
                    time.sleep(0.01)  # Simulate some work
                    task["counter"] = current + 1
                    results.append(thread_id)

        threads = [threading.Thread(target=increment_counter, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 5 threads should have completed
        assert len(results) == 5

        # Counter should be exactly 5 (no lost updates)
        reloaded = _load_data(paths["task_queue"], {})
        tasks = reloaded.get("tasks", [])
        task_001 = next((t for t in tasks if t.get("id") == "task-001"), None)
        assert task_001 is not None
        assert task_001.get("counter") == 5


class TestLoadTaskStateQuick:
    """Tests for load_task_state_quick function."""

    def test_load_existing_task(self, paths: dict[str, Path], sample_queue: dict):
        """Test loading an existing task."""
        _save_data(paths["task_queue"], sample_queue)

        task = load_task_state_quick(
            paths["lock"],
            paths["task_queue"],
            "task-001",
        )
        assert task is not None
        assert task["id"] == "task-001"
        # Status is normalized from lifecycle/step by _normalize_tasks
        assert task["status"] == "plan_impl"

    def test_load_nonexistent_task(self, paths: dict[str, Path], sample_queue: dict):
        """Test loading a nonexistent task."""
        _save_data(paths["task_queue"], sample_queue)

        task = load_task_state_quick(
            paths["lock"],
            paths["task_queue"],
            "nonexistent",
        )
        assert task is None

    def test_returns_copy(self, paths: dict[str, Path], sample_queue: dict):
        """Test that a copy is returned, not a reference."""
        _save_data(paths["task_queue"], sample_queue)

        task = load_task_state_quick(
            paths["lock"],
            paths["task_queue"],
            "task-001",
        )
        assert task is not None

        # Modify the returned copy
        task["status"] = "modified"

        # Original should be unchanged (still normalized value)
        task_again = load_task_state_quick(
            paths["lock"],
            paths["task_queue"],
            "task-001",
        )
        # Status is normalized from lifecycle/step by _normalize_tasks
        assert task_again["status"] == "plan_impl"


class TestStateTransactionEdgeCases:
    """Edge case tests for state_transaction."""

    def test_empty_phases_list(self, paths: dict[str, Path], sample_queue: dict):
        """Test handling of empty phases list."""
        _save_data(paths["task_queue"], sample_queue)
        _save_data(paths["phase_plan"], {"phases": []})

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
            paths["phase_plan"],
            load_plan=True,
        ) as state:
            assert state.phases == []

    def test_malformed_queue(self, paths: dict[str, Path]):
        """Test handling of malformed queue data."""
        _save_data(paths["task_queue"], {"tasks": "not_a_list"})

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
        ) as state:
            # _normalize_tasks should handle this gracefully
            assert state.tasks == []

    def test_malformed_plan(self, paths: dict[str, Path], sample_queue: dict):
        """Test handling of malformed plan data."""
        _save_data(paths["task_queue"], sample_queue)
        _save_data(paths["phase_plan"], {"phases": "not_a_list"})

        with state_transaction(
            paths["lock"],
            paths["task_queue"],
            paths["phase_plan"],
            load_plan=True,
        ) as state:
            # _normalize_phases should handle this gracefully
            assert state.phases == []

    def test_multiple_sequential_transactions(
        self, paths: dict[str, Path], sample_queue: dict
    ):
        """Test that multiple sequential transactions work correctly."""
        _save_data(paths["task_queue"], sample_queue)

        # First transaction - update lifecycle to trigger status change
        with state_transaction(
            paths["lock"],
            paths["task_queue"],
        ) as state1:
            task = state1.find_task("task-001")
            task["lifecycle"] = "running"
            task["step"] = "implement"

        # Second transaction (sequential, not nested)
        with state_transaction(
            paths["lock"],
            paths["task_queue"],
        ) as state2:
            task = state2.find_task("task-001")
            # Should see the lifecycle from first transaction
            assert task["lifecycle"] == "running"
            # Status is normalized from lifecycle/step (implement -> implementing)
            assert task["status"] == "implementing"
            task["lifecycle"] = "done"
            task["step"] = "commit"

        # Final result should be from second transaction
        reloaded = _load_data(paths["task_queue"], {})
        tasks = reloaded.get("tasks", [])
        task_001 = next((t for t in tasks if t.get("id") == "task-001"), None)
        assert task_001 is not None
        assert task_001["lifecycle"] == "done"
