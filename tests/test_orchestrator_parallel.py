"""Tests verifying parallel task execution in the orchestrator."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.events import EventBus
from agent_orchestrator.runtime.orchestrator import OrchestratorService
from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult
from agent_orchestrator.runtime.storage.container import Container


def _service(tmp_path: Path, concurrency: int = 2) -> tuple[Container, OrchestratorService, EventBus]:
    container = Container(tmp_path)
    if concurrency != 2:
        cfg = container.config.load()
        cfg["orchestrator"] = {"concurrency": concurrency}
        container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)
    return container, service, bus


# ---------------------------------------------------------------------------
# 1. tick_once dispatches to thread pool (non-blocking)
# ---------------------------------------------------------------------------


def test_tick_dispatches_to_thread_pool(tmp_path: Path) -> None:
    """Two ready tasks dispatched via two tick_once calls should run
    concurrently in separate threads."""
    barrier = threading.Barrier(2, timeout=5)
    call_threads: list[str] = []
    lock = threading.Lock()

    class SlowAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            with lock:
                call_threads.append(threading.current_thread().name)
            # Both tasks must reach this barrier before either can proceed
            barrier.wait()
            return StepResult(status="ok")

    container = Container(tmp_path)
    cfg = container.config.load()
    cfg["orchestrator"] = {"concurrency": 2, "auto_deps": False}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=SlowAdapter())

    t1 = Task(title="Task A", task_type="chore", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
    t2 = Task(title="Task B", task_type="chore", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    # Dispatch both tasks
    assert service.tick_once() is True
    assert service.tick_once() is True

    # Wait for both to complete
    deadline = time.time() + 10
    while time.time() < deadline:
        with service._futures_lock:
            if all(f.done() for f in service._futures.values()):
                break
        time.sleep(0.1)

    service._sweep_futures()

    # Both tasks should have run in pool threads (not the main thread)
    assert len(call_threads) >= 2
    assert all("orchestrator-task" in name for name in call_threads)

    # Both tasks should be done
    for task_id in [t1.id, t2.id]:
        task = container.tasks.get(task_id)
        assert task is not None
        assert task.status == "done"


# ---------------------------------------------------------------------------
# 2. Concurrency cap is respected
# ---------------------------------------------------------------------------


def test_concurrency_cap_respected(tmp_path: Path) -> None:
    """With concurrency=2 and 3 ready tasks, the third tick should not
    claim a task until one finishes."""
    gate = threading.Event()

    class BlockingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            gate.wait(timeout=10)
            return StepResult(status="ok")

    container = Container(tmp_path)
    cfg = container.config.load()
    cfg["orchestrator"] = {"concurrency": 2, "auto_deps": False}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=BlockingAdapter())

    for i in range(3):
        t = Task(title=f"Task {i}", task_type="chore", status="queued",
                 approval_mode="auto_approve", hitl_mode="autopilot")
        container.tasks.upsert(t)

    # First two ticks claim tasks
    assert service.tick_once() is True
    assert service.tick_once() is True

    # Third tick should fail — concurrency cap reached (2 in_progress in storage)
    assert service.tick_once() is False

    # Release the gate so tasks complete
    gate.set()

    # Wait for futures to finish
    deadline = time.time() + 10
    while time.time() < deadline:
        with service._futures_lock:
            if all(f.done() for f in service._futures.values()):
                break
        time.sleep(0.1)

    service._sweep_futures()

    # Now the third task can be claimed
    assert service.tick_once() is True

    gate.set()  # Already set, just ensuring
    deadline = time.time() + 10
    while time.time() < deadline:
        with service._futures_lock:
            if all(f.done() for f in service._futures.values()):
                break
        time.sleep(0.1)

    service._sweep_futures()

    # All three should be done
    tasks = container.tasks.list()
    done_count = sum(1 for t in tasks if t.status == "done")
    assert done_count == 3


# ---------------------------------------------------------------------------
# 3. Same-repo tasks now run in parallel (worktrees provide isolation)
# ---------------------------------------------------------------------------


def test_same_repo_tasks_run_in_parallel(tmp_path: Path) -> None:
    """Two tasks with the same repo_path can now run concurrently
    because worktrees provide isolation."""
    barrier = threading.Barrier(2, timeout=5)

    class BarrierAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            barrier.wait()
            return StepResult(status="ok")

    container = Container(tmp_path)
    cfg = container.config.load()
    cfg["orchestrator"] = {"concurrency": 4, "auto_deps": False}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=BarrierAdapter())

    t1 = Task(title="Task A", task_type="chore", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot",
              metadata={"repo_path": "/shared/repo"})
    t2 = Task(title="Task B", task_type="chore", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot",
              metadata={"repo_path": "/shared/repo"})
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    # Both ticks should claim tasks — no repo conflict blocking
    assert service.tick_once() is True
    assert service.tick_once() is True

    # Wait for both to complete (barrier ensures they ran concurrently)
    deadline = time.time() + 10
    while time.time() < deadline:
        with service._futures_lock:
            if all(f.done() for f in service._futures.values()):
                break
        time.sleep(0.1)
    service._sweep_futures()

    for tid in [t1.id, t2.id]:
        task = container.tasks.get(tid)
        assert task is not None
        assert task.status == "done"


# ---------------------------------------------------------------------------
# 4. Different repos can run in parallel
# ---------------------------------------------------------------------------


def test_different_repos_run_in_parallel(tmp_path: Path) -> None:
    """Tasks targeting different repos should run concurrently."""
    barrier = threading.Barrier(2, timeout=5)

    class BarrierAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            barrier.wait()
            return StepResult(status="ok")

    container = Container(tmp_path)
    cfg = container.config.load()
    cfg["orchestrator"] = {"concurrency": 2, "auto_deps": False}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=BarrierAdapter())

    t1 = Task(title="Task A", task_type="chore", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot",
              metadata={"repo_path": "/repo/alpha"})
    t2 = Task(title="Task B", task_type="chore", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot",
              metadata={"repo_path": "/repo/beta"})
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    assert service.tick_once() is True
    assert service.tick_once() is True

    # Both should complete (the barrier requires both threads to arrive)
    deadline = time.time() + 10
    while time.time() < deadline:
        with service._futures_lock:
            if all(f.done() for f in service._futures.values()):
                break
        time.sleep(0.1)

    service._sweep_futures()

    for task_id in [t1.id, t2.id]:
        task = container.tasks.get(task_id)
        assert task is not None
        assert task.status == "done"


# ---------------------------------------------------------------------------
# 5. drain waits for in-flight tasks
# ---------------------------------------------------------------------------


def test_drain_waits_for_inflight(tmp_path: Path) -> None:
    """Drain should not pause until all in-flight tasks complete."""
    gate = threading.Event()
    task_started = threading.Event()

    class BlockingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            task_started.set()
            gate.wait(timeout=10)
            return StepResult(status="ok")

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=BlockingAdapter())

    t = Task(title="Slow task", task_type="chore", status="queued",
             approval_mode="auto_approve", hitl_mode="autopilot")
    container.tasks.upsert(t)

    # Run _loop in a background thread
    loop_thread = threading.Thread(target=service._loop, daemon=True)
    loop_thread.start()

    # Wait for task to start executing
    assert task_started.wait(timeout=5)

    # Request drain — the loop should NOT pause yet because the task is in-flight
    service._drain = True

    # Verify the loop is still running (not paused)
    time.sleep(0.5)
    cfg = container.config.load()
    assert cfg.get("orchestrator", {}).get("status", "running") == "running"

    # Release the task so it finishes
    gate.set()

    # Wait for the loop to exit (drain should trigger pause and break)
    loop_thread.join(timeout=10)
    assert not loop_thread.is_alive(), "Loop should have exited after drain"

    cfg = container.config.load()
    assert cfg.get("orchestrator", {}).get("status") == "paused"


# ---------------------------------------------------------------------------
# 6. run_task still returns synchronously
# ---------------------------------------------------------------------------


def test_run_task_still_synchronous(tmp_path: Path) -> None:
    """run_task() should block until the task completes and return the result."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Sync task",
        task_type="chore",
        status="queued",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"


# ---------------------------------------------------------------------------
# 7. Unexpected exception in _execute_task sets task to blocked
# ---------------------------------------------------------------------------


def test_future_exception_sets_task_blocked(tmp_path: Path) -> None:
    """If the worker adapter raises, the task should be marked blocked."""

    class CrashingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            raise RuntimeError("kaboom")

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=CrashingAdapter())

    task = Task(title="Crash task", task_type="chore", status="queued",
                approval_mode="auto_approve", hitl_mode="autopilot")
    container.tasks.upsert(task)

    assert service.tick_once() is True

    # Wait for the future to complete
    deadline = time.time() + 10
    while time.time() < deadline:
        with service._futures_lock:
            if all(f.done() for f in service._futures.values()):
                break
        time.sleep(0.1)

    service._sweep_futures()

    updated = container.tasks.get(task.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.error == "Internal error during execution"


# ---------------------------------------------------------------------------
# 8. status() reports active_workers
# ---------------------------------------------------------------------------


def test_status_reports_active_workers(tmp_path: Path) -> None:
    """status() should include active_workers count."""
    gate = threading.Event()

    class BlockingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            gate.wait(timeout=10)
            return StepResult(status="ok")

    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=BlockingAdapter())

    t = Task(title="Active task", task_type="chore", status="queued",
             approval_mode="auto_approve", hitl_mode="autopilot")
    container.tasks.upsert(t)

    assert service.status()["active_workers"] == 0

    service.tick_once()
    assert service.status()["active_workers"] == 1

    gate.set()
    deadline = time.time() + 10
    while time.time() < deadline:
        with service._futures_lock:
            if all(f.done() for f in service._futures.values()):
                break
        time.sleep(0.1)
    service._sweep_futures()

    assert service.status()["active_workers"] == 0
