"""Tests for git worktree-based same-repo concurrency in the orchestrator."""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.events import EventBus
from agent_orchestrator.runtime.orchestrator import OrchestratorService
from agent_orchestrator.runtime.orchestrator.live_worker_adapter import build_step_prompt
from agent_orchestrator.runtime.orchestrator.worktree_manager import WorktreeManager
from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult
from agent_orchestrator.runtime.storage.container import Container


def _git_init(path: Path) -> None:
    """Initialize a git repo with an initial commit."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True, text=True)
    (path / "README.md").write_text("# init\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True, text=True)


def _service(
    tmp_path: Path,
    *,
    adapter: object | None = None,
    concurrency: int = 4,
    git: bool = True,
) -> tuple[Container, OrchestratorService, EventBus]:
    if git:
        _git_init(tmp_path)
    container = Container(tmp_path)
    cfg = container.config.load()
    cfg["orchestrator"] = {"concurrency": concurrency, "auto_deps": False}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=adapter) if adapter else OrchestratorService(container, bus)
    return container, service, bus


def _wait_futures(service: OrchestratorService, timeout: float = 10) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with service._futures_lock:
            if all(f.done() for f in service._futures.values()):
                break
        time.sleep(0.1)
    service._sweep_futures()


# ---------------------------------------------------------------------------
# 1. Worktree is created for a task and cleaned up after completion
# ---------------------------------------------------------------------------


def test_worktree_created_for_task(tmp_path: Path) -> None:
    """Task execution in a git repo creates a worktree directory; after
    completion it is cleaned up."""
    worktree_paths: list[Optional[Path]] = []

    class SpyAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt:
                worktree_paths.append(Path(wt))
                if step == "implement":
                    (Path(wt) / "change.txt").write_text("impl\n")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=SpyAdapter())
    task = Task(
        title="WT task",
        task_type="chore",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    service.tick_once()
    _wait_futures(service)

    # Worktree was created during execution
    assert len(worktree_paths) >= 1
    wt_path = worktree_paths[0]
    assert wt_path is not None

    # Worktree should be cleaned up after task completes
    assert not wt_path.exists()

    # Task should be done
    updated = container.tasks.get(task.id)
    assert updated is not None
    assert updated.status == "done"
    assert "worktree_dir" not in updated.metadata


def test_create_worktree_reuses_existing_task_branch_without_b_flag_failure(tmp_path: Path) -> None:
    """Creating a task worktree should succeed even when task branch already exists."""
    container, service, _ = _service(tmp_path, git=True)
    task = Task(title="Branch reuse", task_type="chore", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(task)

    branch = f"task-{task.id}"
    subprocess.run(
        ["git", "branch", branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    worktree_dir = service._create_worktree(task)
    assert worktree_dir is not None
    assert worktree_dir.exists()
    current_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=worktree_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert current_branch == branch

    subprocess.run(
        ["git", "worktree", "remove", str(worktree_dir), "--force"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_create_worktree_recovers_from_branch_exists_race(tmp_path: Path) -> None:
    """If `worktree add -b` races with branch creation, fallback should still succeed."""
    container, service, _ = _service(tmp_path, git=True)
    task = Task(title="Branch race", task_type="chore", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(task)

    branch = f"task-{task.id}"
    subprocess.run(
        ["git", "branch", branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    # Simulate TOCTOU: branch exists in git but the pre-check returns false.
    with patch.object(WorktreeManager, "_local_branch_exists", return_value=False):
        worktree_dir = service._create_worktree(task)
    assert worktree_dir is not None
    assert worktree_dir.exists()

    current_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=worktree_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert current_branch == branch

    subprocess.run(
        ["git", "worktree", "remove", str(worktree_dir), "--force"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_create_worktree_retries_transient_git_lock_error(tmp_path: Path) -> None:
    """Worktree creation should recover from transient git lock contention."""
    container, service, _ = _service(tmp_path, git=True)
    task = Task(title="Transient lock", task_type="chore", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(task)

    index_lock = tmp_path / ".git" / "index.lock"
    index_lock.write_text("locked\n", encoding="utf-8")

    def _release_lock() -> None:
        time.sleep(0.12)
        if index_lock.exists():
            index_lock.unlink()

    releaser = threading.Thread(target=_release_lock, daemon=True)
    releaser.start()
    worktree_dir = service._create_worktree(task)
    releaser.join(timeout=1.0)

    assert worktree_dir is not None
    assert worktree_dir.exists()
    assert not index_lock.exists()

    subprocess.run(
        ["git", "worktree", "remove", str(worktree_dir), "--force"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_create_worktree_from_branch_retries_transient_git_lock_error(tmp_path: Path) -> None:
    """Retaching from preserved branch should recover from transient git lock contention."""
    container, service, _ = _service(tmp_path, git=True)
    task = Task(title="Transient lock preserved", task_type="chore", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(task)
    branch = f"task-{task.id}"
    subprocess.run(
        ["git", "branch", branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    index_lock = tmp_path / ".git" / "index.lock"
    index_lock.write_text("locked\n", encoding="utf-8")

    def _release_lock() -> None:
        time.sleep(0.12)
        if index_lock.exists():
            index_lock.unlink()

    releaser = threading.Thread(target=_release_lock, daemon=True)
    releaser.start()
    worktree_dir = service._create_worktree_from_branch(task, branch)
    releaser.join(timeout=1.0)

    assert worktree_dir is not None
    assert worktree_dir.exists()
    assert not index_lock.exists()

    subprocess.run(
        ["git", "worktree", "remove", str(worktree_dir), "--force"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# 2. Concurrent same-repo tasks run in parallel via worktrees
# ---------------------------------------------------------------------------


def test_concurrent_same_repo_tasks(tmp_path: Path) -> None:
    """Two tasks targeting the same repo run concurrently via worktrees
    (barrier proves concurrency)."""
    barrier = threading.Barrier(2, timeout=5)

    class BarrierAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / f"{task.id}.txt").write_text("impl\n")
                barrier.wait()  # Proves both tasks reach implement concurrently
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=BarrierAdapter(), concurrency=4)

    t1 = Task(title="Task A", task_type="chore", status="queued", hitl_mode="autopilot",
              metadata={"repo_path": str(tmp_path)})
    t2 = Task(title="Task B", task_type="chore", status="queued", hitl_mode="autopilot",
              metadata={"repo_path": str(tmp_path)})
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    # Both should be claimable now (no repo conflict blocking)
    assert service.tick_once() is True
    assert service.tick_once() is True

    _wait_futures(service)

    # Both tasks should be done — the barrier would have timed out if they
    # didn't run concurrently
    for tid in [t1.id, t2.id]:
        task = container.tasks.get(tid)
        assert task is not None
        assert task.status == "done"


def test_ensure_worker_skips_recovery_for_inflight_manual_run(tmp_path: Path) -> None:
    """Starting scheduler must not recover/requeue a task currently running via run_task()."""
    entered_implement = threading.Event()
    release_implement = threading.Event()

    class BlockingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                entered_implement.set()
                if not release_implement.wait(timeout=10):
                    raise RuntimeError("timeout waiting for implement release")
                (Path(wt) / f"{task.id}.txt").write_text("impl\n")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=BlockingAdapter(), concurrency=2)
    task = Task(title="Manual run", task_type="chore", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(task)

    holder: dict[str, Task] = {}
    errors: list[Exception] = []

    def _run_manual() -> None:
        try:
            holder["task"] = service.run_task(task.id)
        except Exception as exc:  # pragma: no cover - assertion checks this list
            errors.append(exc)

    runner = threading.Thread(target=_run_manual, daemon=True)
    runner.start()
    try:
        assert entered_implement.wait(timeout=5), "manual run never reached implement"
        service.ensure_worker()

        mid = container.tasks.get(task.id)
        assert mid is not None
        assert mid.status == "in_progress"
        assert "Recovered from interrupted run" not in str(mid.error or "")
    finally:
        release_implement.set()
        runner.join(timeout=15)
        service.shutdown(timeout=2)

    assert not errors
    assert "task" in holder
    final = container.tasks.get(task.id)
    assert final is not None
    assert final.status == "done"


# ---------------------------------------------------------------------------
# 3. Task branch is merged to run branch
# ---------------------------------------------------------------------------


def test_task_branch_merged_to_run_branch(tmp_path: Path) -> None:
    """After task completes, its commits appear on the run branch."""

    class FileWriter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step in ("plan", "implement"):
                (Path(wt) / f"{task.id}.txt").write_text(f"work by {task.id}\n")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=FileWriter())
    task = Task(
        title="Merge test",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    # The file should exist on the run branch (project_dir)
    expected_file = tmp_path / f"{task.id}.txt"
    assert expected_file.exists()
    assert expected_file.read_text().strip() == f"work by {task.id}"


# ---------------------------------------------------------------------------
# 4. Merge conflict is resolved by worker
# ---------------------------------------------------------------------------


def test_merge_conflict_resolved_by_worker(tmp_path: Path) -> None:
    """Two tasks modify the same file concurrently; first merges cleanly,
    second conflicts; worker is dispatched and resolves the conflict."""
    resolve_called = threading.Event()
    # Barrier ensures both tasks write before either starts committing
    write_barrier = threading.Barrier(2, timeout=5)

    class ConflictAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "shared.txt").write_text(f"content by {task.title}\n")
                write_barrier.wait()
            if step == "resolve_merge":
                conflict_files = task.metadata.get("merge_conflict_files", {})
                for fpath in conflict_files:
                    full = tmp_path / fpath
                    full.write_text("resolved content\n")
                resolve_called.set()
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=ConflictAdapter(), concurrency=2)

    t1 = Task(title="Alpha", task_type="feature", status="queued", hitl_mode="autopilot")
    t2 = Task(title="Beta", task_type="feature", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    assert service.tick_once() is True
    assert service.tick_once() is True

    _wait_futures(service, timeout=15)

    # The resolve_merge step should have been called for the conflicting task
    assert resolve_called.is_set()

    # The shared file should have the resolved content
    assert (tmp_path / "shared.txt").read_text().strip() == "resolved content"

    # Both tasks should be done
    for tid in [t1.id, t2.id]:
        task = container.tasks.get(tid)
        assert task is not None
        assert task.status == "done"


# ---------------------------------------------------------------------------
# 5. Merge conflict fallback on worker failure
# ---------------------------------------------------------------------------


def test_merge_conflict_fallback_on_worker_failure(tmp_path: Path) -> None:
    """If the conflict-resolution worker fails, the merge is aborted and
    task.metadata['merge_conflict'] is set."""
    write_barrier = threading.Barrier(2, timeout=5)

    class FailingResolveAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "shared.txt").write_text(f"content by {task.title}\n")
                write_barrier.wait()
            if step == "resolve_merge":
                return StepResult(status="error", summary="Cannot resolve")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=FailingResolveAdapter(), concurrency=2)

    t1 = Task(title="Alpha", task_type="feature", status="queued", hitl_mode="autopilot")
    t2 = Task(title="Beta", task_type="feature", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    assert service.tick_once() is True
    assert service.tick_once() is True

    _wait_futures(service, timeout=15)

    # One task should have merge_conflict set and be blocked
    tasks = [container.tasks.get(t1.id), container.tasks.get(t2.id)]
    conflict_tasks = [t for t in tasks if t and t.metadata.get("merge_conflict")]
    assert len(conflict_tasks) == 1, "Exactly one task should have merge_conflict flag"
    assert conflict_tasks[0].status == "blocked"
    assert "merge conflict" in (conflict_tasks[0].error or "").lower()

    # The conflicted task's branch should be preserved for recovery
    branch_name = f"task-{conflict_tasks[0].id}"
    branches = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert branch_name in branches, "Task branch should be preserved when merge fails"

    # The non-conflicting task should be done
    ok_tasks = [t for t in tasks if t and not t.metadata.get("merge_conflict")]
    assert len(ok_tasks) == 1
    assert ok_tasks[0].status == "done"


# ---------------------------------------------------------------------------
# 6. Worktree is cleaned up on failure
# ---------------------------------------------------------------------------


def test_worktree_cleanup_on_failure(tmp_path: Path) -> None:
    """If task fails mid-execution, blocked state keeps live context for retry."""
    worktree_paths: list[Path] = []

    class CrashAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt:
                worktree_paths.append(Path(wt))
            raise RuntimeError("boom")

    container, service, _ = _service(tmp_path, adapter=CrashAdapter())
    task = Task(title="Crash WT", task_type="chore", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(task)

    service.tick_once()
    _wait_futures(service)

    # Worktree should remain because blocked tasks retain retry context.
    for wt in worktree_paths:
        assert wt.exists(), f"Worktree {wt} should be retained for retry"

    updated = container.tasks.get(task.id)
    assert updated is not None
    assert updated.status == "blocked"
    context = updated.metadata.get("task_context")
    assert isinstance(context, dict)
    assert context.get("retained") is True


# ---------------------------------------------------------------------------
# 7. No worktree without .git
# ---------------------------------------------------------------------------


def test_no_worktree_without_git(tmp_path: Path) -> None:
    """Non-git project dir skips worktrees, runs directly."""
    worktree_used = []

    class SpyAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            worktree_used.append(task.metadata.get("worktree_dir"))
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=SpyAdapter(), git=False)
    task = Task(title="No git", task_type="chore", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    # No worktree should have been set
    assert all(wt is None for wt in worktree_used)


# ---------------------------------------------------------------------------
# 8. Orphaned worktree cleanup on startup
# ---------------------------------------------------------------------------


def test_orphaned_worktree_cleanup(tmp_path: Path) -> None:
    """Leftover worktree dirs from previous runs are cleaned up on startup."""
    _git_init(tmp_path)
    container = Container(tmp_path)

    # Simulate orphaned worktree by creating one via git
    orphan_dir = container.state_root / "worktrees" / "orphan-task"
    subprocess.run(
        ["git", "worktree", "add", str(orphan_dir), "-b", "task-orphan-task"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert orphan_dir.exists()

    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)

    # ensure_worker triggers cleanup
    service.ensure_worker()
    service._stop.set()
    time.sleep(0.5)

    # The orphaned worktree should be removed
    assert not orphan_dir.exists()

    # The orphaned branch should also be removed
    branches = subprocess.run(
        ["git", "branch", "--list", "task-orphan-task"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert branches == ""


# ---------------------------------------------------------------------------
# 9. Non-commit pipeline cleans up worktree without leaking branch
# ---------------------------------------------------------------------------


def test_non_commit_pipeline_cleans_worktree(tmp_path: Path) -> None:
    """Research pipeline (no commit step) still creates and cleans up its
    worktree, and does not leak a task branch."""
    worktree_paths: list[Path] = []

    class SpyAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt:
                worktree_paths.append(Path(wt))
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=SpyAdapter())
    task = Task(
        title="Research task",
        task_type="research",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done"

    # Worktree was used and cleaned up
    assert len(worktree_paths) >= 1
    for wt in worktree_paths:
        assert not wt.exists()

    # No task branch should remain
    branches = subprocess.run(
        ["git", "branch", "--list", f"task-{task.id}"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert branches == ""

    # Metadata should be clean
    updated = container.tasks.get(task.id)
    assert updated is not None
    assert "worktree_dir" not in updated.metadata


# ---------------------------------------------------------------------------
# 10. Blocked task keeps retained context in metadata
# ---------------------------------------------------------------------------


def test_blocked_task_metadata_cleaned(tmp_path: Path) -> None:
    """When a task blocks during a pipeline step, live worktree context is retained."""

    class FailOnImplement:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            if step == "implement":
                return StepResult(status="error", summary="implement failed")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=FailOnImplement())
    task = Task(
        title="Block test",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert result.error == "implement failed"

    # worktree_dir stays in metadata so retries can reattach deterministically.
    assert result.metadata.get("worktree_dir")
    worktree_dir = container.state_root / "worktrees" / task.id
    assert worktree_dir.exists()

    context = result.metadata.get("task_context")
    assert isinstance(context, dict)
    assert context.get("retained") is True
    assert context.get("expected_on_retry") is True

    # Task branch is retained.
    branches = subprocess.run(
        ["git", "branch", "--list", f"task-{task.id}"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert f"task-{task.id}" in branches


# ---------------------------------------------------------------------------
# 11. _create_worktree failure falls back to direct execution
# ---------------------------------------------------------------------------


def test_worktree_creation_failure_falls_back(tmp_path: Path) -> None:
    """If git worktree add fails (e.g., branch already exists), the task
    should still attempt to run without a worktree rather than crash."""
    _git_init(tmp_path)
    container = Container(tmp_path)

    # Pre-create the branch so worktree add will fail
    task = Task(
        title="Fallback test",
        task_type="chore",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)
    subprocess.run(
        ["git", "branch", f"task-{task.id}"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )

    cfg = container.config.load()
    cfg["orchestrator"] = {"concurrency": 2}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)

    # _create_worktree will raise CalledProcessError because the branch exists.
    # _execute_task catches all exceptions and marks as blocked.
    service.tick_once()
    _wait_futures(service)

    updated = container.tasks.get(task.id)
    assert updated is not None
    # Task should be blocked (worktree creation failure causes an exception)
    assert updated.status == "blocked"


# ---------------------------------------------------------------------------
# 12. resolve_merge receives conflict metadata and runs in project_dir
# ---------------------------------------------------------------------------


def test_resolve_merge_receives_metadata_and_runs_in_project_dir(tmp_path: Path) -> None:
    """When resolve_merge is dispatched, the worker receives conflict files and
    other task info in task.metadata, and worktree_dir is cleared so the worker
    runs in project_dir (where the merge conflict lives)."""
    write_barrier = threading.Barrier(2, timeout=5)
    resolve_metadata: list[dict] = []

    class InspectingAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "shared.txt").write_text(f"content by {task.title}\n")
                write_barrier.wait()
            if step == "resolve_merge":
                # Capture a snapshot of metadata at the time of the call
                resolve_metadata.append({
                    "has_conflict_files": "merge_conflict_files" in task.metadata,
                    "conflict_files": dict(task.metadata.get("merge_conflict_files", {})),
                    "other_tasks": list(task.metadata.get("merge_other_tasks", [])),
                    "current_objective": task.metadata.get("merge_current_objective"),
                    "other_objectives": list(task.metadata.get("merge_other_objectives", [])),
                    "worktree_dir": task.metadata.get("worktree_dir"),
                })
                # Resolve the conflict
                conflict_files = task.metadata.get("merge_conflict_files", {})
                for fpath in conflict_files:
                    full = tmp_path / fpath
                    full.write_text("resolved\n")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=InspectingAdapter(), concurrency=2)

    t1 = Task(title="Alpha", task_type="feature", status="queued", hitl_mode="autopilot")
    t2 = Task(title="Beta", task_type="feature", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    assert service.tick_once() is True
    assert service.tick_once() is True
    _wait_futures(service, timeout=15)

    # resolve_merge should have been called with conflict metadata
    assert len(resolve_metadata) == 1
    meta = resolve_metadata[0]
    assert meta["has_conflict_files"] is True
    assert "shared.txt" in meta["conflict_files"]
    # The conflict file should contain merge markers
    content = meta["conflict_files"]["shared.txt"]
    assert "<<<<<<<" in content or "=======" in content

    # worktree_dir should NOT be set during resolve_merge (worker uses project_dir)
    assert meta["worktree_dir"] is None

    # other_tasks should contain the first task's info (it merged before the conflict)
    assert len(meta["other_tasks"]) >= 1
    other_text = " ".join(meta["other_tasks"])
    assert "Alpha" in other_text or "Beta" in other_text
    assert isinstance(meta["current_objective"], str) and meta["current_objective"].strip()
    assert len(meta["other_objectives"]) >= 1

    # After completion, worktree_dir metadata should be cleaned up
    for tid in [t1.id, t2.id]:
        task = container.tasks.get(tid)
        assert task is not None
        assert "worktree_dir" not in task.metadata
        assert "merge_conflict_files" not in task.metadata
        assert "merge_other_tasks" not in task.metadata
        assert "merge_current_objective" not in task.metadata
        assert "merge_other_objectives" not in task.metadata


# ---------------------------------------------------------------------------
# 13. resolve_merge worker exception is handled safely
# ---------------------------------------------------------------------------


def test_resolve_merge_worker_exception_handled(tmp_path: Path) -> None:
    """If the resolve_merge worker raises an exception (not just returns error),
    the merge is aborted cleanly and metadata is cleaned up."""
    write_barrier = threading.Barrier(2, timeout=5)

    class ExplodingResolveAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "shared.txt").write_text(f"content by {task.title}\n")
                write_barrier.wait()
            if step == "resolve_merge":
                raise RuntimeError("worker crashed during resolve")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=ExplodingResolveAdapter(), concurrency=2)

    t1 = Task(title="Alpha", task_type="feature", status="queued", hitl_mode="autopilot")
    t2 = Task(title="Beta", task_type="feature", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(t1)
    container.tasks.upsert(t2)

    assert service.tick_once() is True
    assert service.tick_once() is True
    _wait_futures(service, timeout=15)

    # One task should have merge_conflict flag set and be blocked
    tasks = [container.tasks.get(t1.id), container.tasks.get(t2.id)]
    conflict_tasks = [t for t in tasks if t and t.metadata.get("merge_conflict")]
    assert len(conflict_tasks) == 1
    assert conflict_tasks[0].status == "blocked"
    assert "merge conflict" in (conflict_tasks[0].error or "").lower()

    # Conflict metadata should be cleaned up from ALL tasks
    for t in tasks:
        assert t is not None
        assert "merge_conflict_files" not in t.metadata
        assert "merge_other_tasks" not in t.metadata
        assert "merge_current_objective" not in t.metadata
        assert "merge_other_objectives" not in t.metadata

    # The git repo should not be in a dirty merge state
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    # Filter out the .prd-runner state files
    git_lines = [line for line in status.split("\n") if line and ".prd-runner" not in line]
    # No unmerged files should remain
    unmerged = [line for line in git_lines if line.startswith("U") or line.startswith("AA")]
    assert unmerged == [], f"Git repo has unresolved merge state: {unmerged}"


# ---------------------------------------------------------------------------
# 14. build_step_prompt includes conflict context for resolve_merge
# ---------------------------------------------------------------------------


def test_build_step_prompt_resolve_merge() -> None:
    """build_step_prompt includes conflict file contents and other task info
    for the resolve_merge step."""
    task = Task(
        title="Fix auth",
        description="Add JWT support",
        task_type="feature",
        metadata={
            "merge_conflict_files": {
                "auth.py": "<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>> task-xyz",
            },
            "merge_other_tasks": ["- Add OAuth: Implement OAuth2 flow"],
            "merge_current_objective": "- Task: Fix auth\n  Description: Add JWT support",
            "merge_other_objectives": ["- Task: Add OAuth\n  Description: Implement OAuth2 flow"],
        },
    )

    prompt = build_step_prompt(task=task, step="resolve_merge", attempt=1)

    assert "Resolve merge conflicts for this task" in prompt
    assert "auth.py" in prompt
    assert "<<<<<<< HEAD" in prompt
    assert "Add OAuth" in prompt
    assert "Current task objective context" in prompt
    assert "Other task objective context" in prompt
    assert "BOTH" in prompt

    # Resolve merge is not a structured-output step; no JSON schema contract appended.
    prompt_ollama = build_step_prompt(task=task, step="resolve_merge", attempt=1)
    assert "Respond with valid JSON matching this schema" not in prompt_ollama


# ---------------------------------------------------------------------------
# 15. No-changes guard blocks task when implementation produces no file changes
# ---------------------------------------------------------------------------


def test_no_changes_blocks_task(tmp_path: Path) -> None:
    """Task blocks when implementation produces no file changes (e.g. worker
    invoked without write permissions)."""

    class NoOpAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            return StepResult(status="ok")  # Does NOT write any files

    container, service, _ = _service(tmp_path, adapter=NoOpAdapter())
    task = Task(
        title="No-op task",
        task_type="chore",  # chore pipeline: implement, verify, commit
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert "No file changes" in (result.error or "")


# ---------------------------------------------------------------------------
# 16a. Missing canonical workdoc blocks before review/commit regardless of phase
# ---------------------------------------------------------------------------


def test_missing_workdoc_blocks_before_review_phase(tmp_path: Path) -> None:
    """Feature pipeline should block before review if canonical workdoc disappears."""

    class DeleteWorkdocBeforeReviewAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step in ("plan", "implement"):
                (Path(wt) / "change.txt").write_text("impl\n")
            if step == "verify":
                workdoc_path = task.metadata.get("workdoc_path")
                if isinstance(workdoc_path, str) and workdoc_path:
                    Path(workdoc_path).unlink(missing_ok=True)
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=DeleteWorkdocBeforeReviewAdapter())
    task = Task(
        title="Block before review",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert result.current_step == "review"
    assert result.error and "Missing required workdoc" in result.error
    assert container.reviews.for_task(task.id) == []


def test_missing_workdoc_blocks_before_commit_phase(tmp_path: Path) -> None:
    """Commit-only path should block before commit if canonical workdoc disappears."""

    class DeleteWorkdocBeforeCommitAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "change.txt").write_text("impl\n")
            if step == "verify":
                workdoc_path = task.metadata.get("workdoc_path")
                if isinstance(workdoc_path, str) and workdoc_path:
                    Path(workdoc_path).unlink(missing_ok=True)
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=DeleteWorkdocBeforeCommitAdapter())
    task = Task(
        title="Block before commit",
        task_type="chore",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert result.current_step == "commit"
    assert result.error and "Missing required workdoc" in result.error


# ---------------------------------------------------------------------------
# 16. Review cap retains live blocked context
# ---------------------------------------------------------------------------


def test_review_cap_preserves_branch(tmp_path: Path) -> None:
    """When review cap is hit, blocked tasks retain live worktree + branch context."""

    class FailingReviewAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "docstrings.py").write_text("# 216 docstrings added\n")
            if step == "review":
                return StepResult(
                    status="ok",
                    findings=[{"severity": "high", "summary": "Issue found", "status": "open"}],
                )
            if step == "implement_fix":
                # Simulate fix attempt that doesn't resolve the issue
                pass
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=FailingReviewAdapter())
    cfg = container.config.load()
    cfg["orchestrator"]["max_review_attempts"] = 2
    container.config.save(cfg)

    task = Task(
        title="Add docstrings",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert "Review attempt cap exceeded" in (result.error or "")

    # Live context should be retained on blocked tasks.
    assert not result.metadata.get("preserved_branch")
    context = result.metadata.get("task_context")
    assert isinstance(context, dict)
    assert context.get("retained") is True
    assert context.get("expected_on_retry") is True

    retained_worktree = str(result.metadata.get("worktree_dir") or "")
    assert retained_worktree
    assert Path(retained_worktree).exists()

    # Branch should be retained.
    branch_name = f"task-{task.id}"
    branches = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert branch_name in branches, "Task branch should be preserved"

    # The file changes should exist in retained worktree.
    assert (Path(retained_worktree) / "docstrings.py").exists()
    assert "216 docstrings" in (Path(retained_worktree) / "docstrings.py").read_text()


# ---------------------------------------------------------------------------
# 16b. Pre-commit review requires preserved task-scoped context
# ---------------------------------------------------------------------------


def test_precommit_review_blocks_when_context_preserve_fails(tmp_path: Path) -> None:
    """Pre-commit pause must fail closed if task-scoped context cannot be preserved."""

    class ReviewOnlyAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "feature.py").write_text("# staged change\n")
            if step == "review":
                return StepResult(status="ok", findings=[])
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=ReviewOnlyAdapter())
    service._preserve_worktree_work = lambda task, worktree_dir: {  # type: ignore[method-assign]
        "status": "failed",
        "reason_code": "forced",
        "commit_sha": None,
        "base_sha": None,
        "head_sha": None,
    }

    task = Task(
        title="Fail preserve before pre-commit review",
        task_type="feature",
        status="queued",
        hitl_mode="review_only",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert "Failed to preserve task-scoped changes for pre-commit review" in (result.error or "")
    assert not result.metadata.get("pending_precommit_approval")
    assert not result.metadata.get("preserved_branch")


def test_precommit_review_requires_preserved_branch_context(tmp_path: Path) -> None:
    """Pre-commit in_review tasks must carry preserved branch context for change review."""

    class ReviewOnlyAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "feature.py").write_text("# staged change\n")
            if step == "review":
                return StepResult(status="ok", findings=[])
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=ReviewOnlyAdapter())
    task = Task(
        title="Pre-commit context task",
        task_type="feature",
        status="queued",
        hitl_mode="review_only",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "in_review"
    assert result.metadata.get("pending_precommit_approval") is True
    preserved_branch = str(result.metadata.get("preserved_branch") or "").strip()
    assert preserved_branch == f"task-{task.id}"
    preserved_base_branch = str(result.metadata.get("preserved_base_branch") or "").strip()
    assert preserved_base_branch
    assert preserved_base_branch != "HEAD"
    assert str(result.metadata.get("preserved_base_sha") or "").strip()
    assert str(result.metadata.get("preserved_head_sha") or "").strip()
    assert str(result.metadata.get("preserved_at") or "").strip()
    review_context = result.metadata.get("review_context")
    assert isinstance(review_context, dict)
    assert str(review_context.get("base_sha") or "").strip()
    assert str(review_context.get("head_sha") or "").strip()
    assert "worktree_dir" not in result.metadata
    branches = subprocess.run(
        ["git", "branch", "--list", preserved_branch],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert preserved_branch in branches


# ---------------------------------------------------------------------------
# 17. approve_and_merge merges preserved branch to run branch
# ---------------------------------------------------------------------------


def test_approve_merges_preserved_branch(tmp_path: Path) -> None:
    """approve_and_merge merges a manually preserved branch and cleans metadata."""
    container, service, _ = _service(tmp_path)
    service._ensure_branch()

    task = Task(title="Feature work", task_type="feature", status="blocked", hitl_mode="autopilot")
    container.tasks.upsert(task)

    branch_name = f"task-{task.id}"
    subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "feature.py").write_text("# new feature\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "feature work"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "checkout", service._run_branch or "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    task.metadata["preserved_branch"] = branch_name
    container.tasks.upsert(task)

    merge_result = service.approve_and_merge(task)
    assert merge_result["status"] == "ok"
    assert "commit_sha" in merge_result

    # Changes should now be on the run branch
    assert (tmp_path / "feature.py").exists()
    assert "new feature" in (tmp_path / "feature.py").read_text()

    # Branch should be deleted
    branches = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert branches == "", "Branch should be deleted after merge"

    # preserved_branch should be cleared
    updated = container.tasks.get(task.id)
    assert updated is not None
    assert "preserved_branch" not in updated.metadata


# ---------------------------------------------------------------------------
# 18. approve_and_merge is a no-op without preserved_branch
# ---------------------------------------------------------------------------


def test_approve_without_preserved_branch(tmp_path: Path) -> None:
    """approve_and_merge on a task without preserved_branch returns ok, no-op."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Normal task",
        task_type="chore",
        status="done",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.approve_and_merge(task)
    assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# 19. finally block preserves work on unexpected failure
# ---------------------------------------------------------------------------


def test_finally_preserves_on_unexpected_failure(tmp_path: Path) -> None:
    """If the adapter raises an exception after writing files, the finally
    block keeps live task context for retry."""

    class CrashAfterWriteAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                (Path(wt) / "work.txt").write_text("important work\n")
                return StepResult(status="ok")
            if step == "verify":
                raise RuntimeError("unexpected crash during verify")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=CrashAfterWriteAdapter())
    task = Task(
        title="Crash task",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    service.tick_once()
    _wait_futures(service)

    updated = container.tasks.get(task.id)
    assert updated is not None
    assert updated.status == "blocked"

    context = updated.metadata.get("task_context")
    assert isinstance(context, dict)
    assert context.get("retained") is True
    assert not updated.metadata.get("preserved_branch")
    branch_name = f"task-{task.id}"
    branches = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert branch_name in branches, "Branch should stay available for retry"

    # Worktree dir should remain for retry.
    worktree_dir = container.state_root / "worktrees" / task.id
    assert worktree_dir.exists()

    assert (worktree_dir / "work.txt").exists()
    assert "important work" in (worktree_dir / "work.txt").read_text()


# ---------------------------------------------------------------------------
# 20. Review cap with no file changes still retains context
# ---------------------------------------------------------------------------


def test_review_cap_no_changes_cleans_worktree(tmp_path: Path) -> None:
    """When review cap is hit, blocked tasks still keep retry context."""

    class NoWriteReviewAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                # Write only the workdoc (which gets cleaned before commit)
                # This simulates "no real changes" since workdoc is removed
                pass
            if step == "review":
                return StepResult(
                    status="ok",
                    findings=[{"severity": "high", "summary": "Issue", "status": "open"}],
                )
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=NoWriteReviewAdapter())
    cfg = container.config.load()
    cfg["orchestrator"]["max_review_attempts"] = 1
    container.config.save(cfg)

    task = Task(
        title="No write task",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    # Task blocks due to no changes guard (before review even runs)
    # or due to review cap — either way, retry context should remain.
    assert result.status == "blocked"
    worktree_dir = container.state_root / "worktrees" / task.id
    assert worktree_dir.exists(), "Worktree directory should be retained for retry"
    assert result.metadata.get("worktree_dir"), "worktree_dir should remain in metadata"


# ---------------------------------------------------------------------------
# 21. Orphan cleanup skips preserved branches
# ---------------------------------------------------------------------------


def test_orphan_cleanup_skips_preserved_branches(tmp_path: Path) -> None:
    """_cleanup_orphaned_worktrees removes the worktree dir but keeps the
    branch if a task references it as preserved_branch."""
    _git_init(tmp_path)
    container = Container(tmp_path)

    # Create a task with preserved_branch metadata
    task = Task(
        title="Preserved task",
        task_type="feature",
        status="blocked",
        metadata={"preserved_branch": "task-preserved-id"},
    )
    container.tasks.upsert(task)

    # Create worktree + branch via git
    orphan_dir = container.state_root / "worktrees" / "preserved-id"
    subprocess.run(
        ["git", "worktree", "add", str(orphan_dir), "-b", "task-preserved-id"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert orphan_dir.exists()

    cfg = container.config.load()
    cfg["orchestrator"] = {"concurrency": 2, "auto_deps": False}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)

    service._cleanup_orphaned_worktrees()

    # Worktree dir should be removed
    assert not orphan_dir.exists()

    # Branch should still exist (not deleted)
    branches = subprocess.run(
        ["git", "branch", "--list", "task-preserved-id"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert "task-preserved-id" in branches, "Preserved branch should not be deleted"


# ---------------------------------------------------------------------------
# 22. Orphan cleanup skips task-context referenced worktrees
# ---------------------------------------------------------------------------


def test_orphan_cleanup_skips_referenced_task_context_worktree(tmp_path: Path) -> None:
    """cleanup_orphaned_worktrees must not remove worktrees referenced by blocked task context."""
    _git_init(tmp_path)
    container = Container(tmp_path)

    worktree_dir = container.state_root / "worktrees" / "task-context-id"
    subprocess.run(
        ["git", "worktree", "add", str(worktree_dir), "-b", "task-task-context-id"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert worktree_dir.exists()

    task = Task(
        id="task-context-id",
        title="Retained context task",
        task_type="feature",
        status="blocked",
        metadata={
            "worktree_dir": str(worktree_dir),
            "task_context": {
                "context_id": "ctx-1",
                "worktree_dir": str(worktree_dir),
                "task_branch": "task-task-context-id",
                "retained": True,
                "expected_on_retry": True,
            },
        },
    )
    container.tasks.upsert(task)

    cfg = container.config.load()
    cfg["orchestrator"] = {"concurrency": 2, "auto_deps": False}
    container.config.save(cfg)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)
    service._cleanup_orphaned_worktrees()

    assert worktree_dir.exists(), "Referenced retained worktree must not be removed"
    branches = subprocess.run(
        ["git", "branch", "--list", "task-task-context-id"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert "task-task-context-id" in branches


# ---------------------------------------------------------------------------
# 23. Retry after retained blocked context runs successfully
# ---------------------------------------------------------------------------


def test_retry_after_preserved_branch(tmp_path: Path) -> None:
    """When a task with a preserved_branch is retried, the worktree is created
    from the preserved branch so prior committed work carries forward."""
    call_count = {"review": 0, "implement": 0}

    class PassOnSecondRunAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt and step == "implement":
                call_count["implement"] += 1
                # Write different content on each run so there are always new changes
                (Path(wt) / "output.py").write_text(f"# code v{call_count['implement']}\n")
            if step == "review":
                call_count["review"] += 1
                if call_count["review"] <= 1:
                    return StepResult(
                        status="ok",
                        findings=[{"severity": "high", "summary": "Bad", "status": "open"}],
                    )
                return StepResult(status="ok", findings=[])
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=PassOnSecondRunAdapter())
    cfg = container.config.load()
    cfg["orchestrator"]["max_review_attempts"] = 1
    container.config.save(cfg)

    task = Task(
        title="Retry task",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    # First run — hits review cap and retains live task context
    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert not result.metadata.get("preserved_branch")
    assert result.metadata.get("worktree_dir")
    first_worktree = Path(str(result.metadata.get("worktree_dir")))
    assert first_worktree.exists()

    # Retry — should reuse preserved branch and carry forward prior work
    task = container.tasks.get(task.id)
    assert task is not None
    task.status = "queued"
    task.error = None
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "done", f"Expected done but got {result.status}: {result.error}"
    assert "preserved_branch" not in result.metadata
    assert "worktree_dir" not in result.metadata
    assert "task_context" not in result.metadata
    assert not first_worktree.exists()


def test_retry_with_expected_missing_context_fails_closed(tmp_path: Path) -> None:
    """Retry should fail closed when expected retained context is missing."""
    container, service, _ = _service(tmp_path)
    missing_dir = container.state_root / "worktrees" / "missing-task"
    task = Task(
        title="Missing retained context",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
        metadata={
            "worktree_dir": str(missing_dir),
            "task_context": {
                "context_id": "ctx-missing",
                "worktree_dir": str(missing_dir),
                "task_branch": "task-missing-task",
                "retained": True,
                "expected_on_retry": True,
            },
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert "Retained task context is missing" in (result.error or "")
    assert result.metadata.get("task_context", {}).get("retained") is True


def test_retry_with_missing_preserved_branch_fails_closed(tmp_path: Path) -> None:
    """Retry should fail closed when preserved branch context cannot be attached."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Missing preserved branch",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
        metadata={
            "preserved_branch": "task-missing-preserved",
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert "Retained task context is missing" in (result.error or "")


def test_retry_with_non_git_retained_worktree_fails_closed(tmp_path: Path) -> None:
    """Retry should fail closed when retained worktree path exists but is not a git worktree."""
    container, service, _ = _service(tmp_path)
    bad_dir = container.state_root / "worktrees" / "bad-worktree"
    bad_dir.mkdir(parents=True, exist_ok=True)
    task = Task(
        id="bad-worktree",
        title="Invalid retained worktree",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
        metadata={
            "worktree_dir": str(bad_dir),
            "task_context": {
                "context_id": "ctx-bad",
                "worktree_dir": str(bad_dir),
                "task_branch": "task-bad-worktree",
                "retained": True,
                "expected_on_retry": True,
            },
        },
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert "Retained task context is missing" in (result.error or "")


def test_retry_from_step_preserves_prior_workdoc_context(tmp_path: Path) -> None:
    """Retry from a later step should keep prior workdoc sections from earlier attempts."""

    class RetryFromStepAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            if step == "plan":
                return StepResult(status="ok", summary="Attempt 1 plan context")
            if step == "implement":
                return StepResult(status="error", summary="Stop after planning")
            if step == "verify":
                return StepResult(status="ok", summary="Verify after retry")
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=RetryFromStepAdapter())
    task = Task(
        title="Retry from verify keeps plan",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
        pipeline_template=["plan", "implement", "verify"],
    )
    container.tasks.upsert(task)

    first = service.run_task(task.id)
    assert first.status == "blocked"
    canonical = container.state_root / "workdocs" / f"{task.id}.md"
    before_retry = canonical.read_text(encoding="utf-8")
    assert "Attempt 1 plan context" in before_retry

    task = container.tasks.get(task.id)
    assert task is not None
    task.status = "queued"
    task.error = None
    task.metadata["retry_from_step"] = "verify"
    container.tasks.upsert(task)

    second = service.run_task(task.id)
    assert second.status == "done"
    after_retry = canonical.read_text(encoding="utf-8")
    assert "Attempt 1 plan context" in after_retry
    assert "Verify after retry" in after_retry


def test_retry_after_preserved_branch_keeps_prior_workdoc_context(tmp_path: Path) -> None:
    """Retry on preserved branch should append to workdoc instead of resetting it."""
    call_count = {"review": 0, "plan": 0, "implement": 0}

    class PreservedBranchContextAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if step == "plan":
                call_count["plan"] += 1
                return StepResult(status="ok", summary=f"Plan context run {call_count['plan']}")
            if wt and step == "implement":
                call_count["implement"] += 1
                (Path(wt) / "output.py").write_text(f"# code v{call_count['implement']}\n")
                return StepResult(status="ok", summary=f"Implementation run {call_count['implement']}")
            if step == "review":
                call_count["review"] += 1
                if call_count["review"] <= 1:
                    return StepResult(
                        status="ok",
                        findings=[{"severity": "high", "summary": "Needs follow-up", "status": "open"}],
                    )
                return StepResult(status="ok", findings=[])
            return StepResult(status="ok")

    container, service, _ = _service(tmp_path, adapter=PreservedBranchContextAdapter())
    cfg = container.config.load()
    cfg["orchestrator"]["max_review_attempts"] = 1
    container.config.save(cfg)

    task = Task(
        title="Preserved branch retry keeps workdoc",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    first = service.run_task(task.id)
    assert first.status == "blocked"
    assert not first.metadata.get("preserved_branch")
    assert first.metadata.get("worktree_dir")
    canonical = container.state_root / "workdocs" / f"{task.id}.md"
    before_retry = canonical.read_text(encoding="utf-8")
    assert "Plan context run 1" in before_retry

    task = container.tasks.get(task.id)
    assert task is not None
    task.status = "queued"
    task.error = None
    container.tasks.upsert(task)

    second = service.run_task(task.id)
    assert second.status == "done"
    after_retry = canonical.read_text(encoding="utf-8")
    assert "Plan context run 1" in after_retry
    assert "Plan context run 2" in after_retry


def test_retry_from_preserved_branch_no_new_uncommitted_changes(tmp_path: Path) -> None:
    """When retrying from a preserved branch and retry_from_step causes
    implement to be skipped, the 'No file changes' check must not block —
    the branch already carries committed work ahead of the run branch."""
    from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult as SR

    class NoopAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> SR:
            return SR(status="ok")

    container, service, _ = _service(tmp_path, adapter=NoopAdapter())

    task = Task(
        title="Skip-impl retry",
        task_type="feature",
        status="queued",
        hitl_mode="autopilot",
        pipeline_template=["plan", "implement", "verify", "commit"],
    )
    container.tasks.upsert(task)

    # Manually create a preserved branch with committed work ahead of the
    # run branch so we don't need a complex first-run failure scenario.
    service._ensure_branch()
    branch = f"task-{task.id}"
    subprocess.run(
        ["git", "checkout", "-b", branch],
        cwd=container.project_dir,
        check=True, capture_output=True, text=True,
    )
    (container.project_dir / "impl_output.py").write_text("# prior work\n")
    subprocess.run(["git", "add", "-A"], cwd=container.project_dir, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "prior implementation"],
        cwd=container.project_dir,
        check=True, capture_output=True, text=True,
    )
    # Switch back to run branch
    subprocess.run(
        ["git", "checkout", service._run_branch],
        cwd=container.project_dir,
        check=True, capture_output=True, text=True,
    )

    # Set up the task as if it was blocked with a preserved branch and
    # retry_from_step=verify (skipping plan and implement).
    task.status = "queued"
    task.metadata["preserved_branch"] = branch
    task.metadata["retry_from_step"] = "verify"
    container.tasks.upsert(task)

    # The retry creates a worktree from the preserved branch, skips plan
    # and implement, runs verify (no-op), then reaches the commit step.
    # The has_commits_ahead check detects prior committed work on the branch
    # even though there are no new uncommitted changes.
    result = service.run_task(task.id)
    assert result.status == "done", f"Expected done but got {result.status}: {result.error}"


def test_skip_task_to_precommit_transitions_blocked_verify_task(tmp_path: Path) -> None:
    """Eligible blocked verify tasks can move directly into pre-commit review."""
    container, service, _ = _service(tmp_path, git=True)
    service._ensure_branch()
    run_branch = service._run_branch or subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    task = Task(
        title="Skip to precommit",
        task_type="feature",
        status="blocked",
        hitl_mode="autopilot",
    )
    task.current_step = "verify"
    container.tasks.upsert(task)

    task_branch = f"task-{task.id}"
    subprocess.run(
        ["git", "checkout", "-b", task_branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "skip_precommit.txt").write_text("ready for commit\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "task work"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "checkout", run_branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    stored = container.tasks.get(task.id)
    assert stored is not None
    stored.error = "verify failed due missing local dependency"
    stored.metadata["preserved_branch"] = task_branch
    container.tasks.upsert(stored)

    allowed, reason = service.can_skip_to_precommit(stored)
    assert allowed is True
    assert reason is None

    updated = service.skip_task_to_precommit(task.id, guidance="Accept risk and proceed to commit.")
    assert updated.status == "in_review"
    assert updated.current_step == "review"
    assert updated.error is None
    assert updated.pending_gate is None
    assert updated.metadata.get("pending_precommit_approval") is True
    assert updated.metadata.get("review_stage") == "pre_commit"
    assert updated.metadata.get("retry_from_step") == "commit"
    assert "worktree_dir" not in updated.metadata
    assert "task_context" not in updated.metadata
    review_context = updated.metadata.get("review_context")
    assert isinstance(review_context, dict)
    assert review_context.get("preserved_branch") == task_branch
    actions = updated.metadata.get("human_review_actions")
    assert isinstance(actions, list) and actions
    assert actions[-1].get("action") == "skip_to_precommit"


def test_can_skip_to_precommit_rejects_unsupported_blocked_step(tmp_path: Path) -> None:
    """Skip-to-precommit is only available for blocked verify/benchmark steps."""
    container, service, _ = _service(tmp_path, git=True)
    task = Task(
        title="Blocked at plan",
        task_type="feature",
        status="blocked",
        hitl_mode="autopilot",
    )
    task.current_step = "plan"
    container.tasks.upsert(task)

    allowed, reason = service.can_skip_to_precommit(task)
    assert allowed is False
    assert reason == "blocked_step_not_supported"


def test_can_skip_to_precommit_does_not_depend_on_error_phrase(tmp_path: Path) -> None:
    """Eligibility should depend on state/context, not free-form error strings."""
    container, service, _ = _service(tmp_path, git=True)
    service._ensure_branch()
    run_branch = service._run_branch or subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    task = Task(
        title="Error phrase resilience",
        task_type="feature",
        status="blocked",
        hitl_mode="autopilot",
    )
    task.current_step = "verify"
    container.tasks.upsert(task)

    task_branch = f"task-{task.id}"
    subprocess.run(
        ["git", "checkout", "-b", task_branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "error_phrase.txt").write_text("material change\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "material work"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "checkout", run_branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    stored = container.tasks.get(task.id)
    assert stored is not None
    stored.error = "Internal error during execution"
    stored.metadata["preserved_branch"] = task_branch
    container.tasks.upsert(stored)

    allowed, reason = service.can_skip_to_precommit(stored)
    assert allowed is True
    assert reason is None


def test_cancel_cleanup_defers_while_execution_lease_active_then_reconciles(tmp_path: Path) -> None:
    """Cancelled cleanup must defer during active lease and finalize on reconcile."""
    container, service, _ = _service(tmp_path, git=True)
    task = Task(
        title="Deferred cancel cleanup",
        task_type="feature",
        status="blocked",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    retained_branch = f"task-{task.id}"
    retained_worktree = container.state_root / "worktrees" / task.id
    subprocess.run(
        ["git", "worktree", "add", str(retained_worktree), "-b", retained_branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    stored = container.tasks.get(task.id)
    assert stored is not None
    stored.metadata = {
        "worktree_dir": str(retained_worktree),
        "task_context": {
            "context_id": "ctx-deferred-cancel",
            "worktree_dir": str(retained_worktree),
            "task_branch": retained_branch,
            "retained": True,
            "expected_on_retry": True,
        },
    }
    container.tasks.upsert(stored)
    service._acquire_execution_lease(stored)

    cancelled = service.cancel_task(task.id, source="test_deferred")
    assert cancelled.status == "cancelled"
    assert cancelled.metadata.get("cancel_cleanup_pending") is True
    assert retained_worktree.exists()
    branch_listing = subprocess.run(
        ["git", "branch", "--list", retained_branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert retained_branch in branch_listing

    latest = container.tasks.get(task.id)
    assert latest is not None
    service._release_execution_lease(latest)

    service.reconcile(source="manual")
    updated = container.tasks.get(task.id)
    assert updated is not None
    assert updated.status == "cancelled"
    assert not retained_worktree.exists()
    branch_listing_after = subprocess.run(
        ["git", "branch", "--list", retained_branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch_listing_after == ""
    metadata = updated.metadata if isinstance(updated.metadata, dict) else {}
    assert "worktree_dir" not in metadata
    assert "task_context" not in metadata
    assert "cancel_cleanup_pending" not in metadata


def test_cancel_cleanup_never_removes_out_of_scope_worktree_paths(tmp_path: Path) -> None:
    """Cancelled cleanup should clear metadata but never delete out-of-scope paths."""
    container, service, _ = _service(tmp_path, git=True)
    outside = tmp_path / "outside-worktree"
    outside.mkdir(parents=True, exist_ok=True)
    marker = outside / "keep.txt"
    marker.write_text("must survive cancel cleanup\n", encoding="utf-8")

    task = Task(
        title="Out of scope cleanup guard",
        task_type="feature",
        status="blocked",
        hitl_mode="autopilot",
        metadata={
            "worktree_dir": str(outside),
            "task_context": {
                "context_id": "ctx-outside",
                "worktree_dir": str(outside),
                "task_branch": "external-branch",
                "retained": True,
                "expected_on_retry": True,
            },
        },
    )
    container.tasks.upsert(task)

    cancelled = service.cancel_task(task.id, source="test_outside_guard")
    assert cancelled.status == "cancelled"
    assert outside.exists()
    assert marker.exists()
    updated = container.tasks.get(task.id)
    assert updated is not None
    metadata = updated.metadata if isinstance(updated.metadata, dict) else {}
    assert "worktree_dir" not in metadata
    assert "task_context" not in metadata
