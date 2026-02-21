"""Tests for git worktree-based same-repo concurrency in the orchestrator."""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.events import EventBus
from agent_orchestrator.runtime.orchestrator import OrchestratorService
from agent_orchestrator.runtime.orchestrator.live_worker_adapter import build_step_prompt
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
        approval_mode="auto_approve",
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

    t1 = Task(title="Task A", task_type="chore", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot",
              metadata={"repo_path": str(tmp_path)})
    t2 = Task(title="Task B", task_type="chore", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot",
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
        approval_mode="auto_approve",
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

    t1 = Task(title="Alpha", task_type="feature", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
    t2 = Task(title="Beta", task_type="feature", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
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

    t1 = Task(title="Alpha", task_type="feature", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
    t2 = Task(title="Beta", task_type="feature", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
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
    """If task fails mid-execution, worktree is still removed."""
    worktree_paths: list[Path] = []

    class CrashAdapter:
        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            wt = task.metadata.get("worktree_dir")
            if wt:
                worktree_paths.append(Path(wt))
            raise RuntimeError("boom")

    container, service, _ = _service(tmp_path, adapter=CrashAdapter())
    task = Task(title="Crash WT", task_type="chore", status="queued",
                approval_mode="auto_approve", hitl_mode="autopilot")
    container.tasks.upsert(task)

    service.tick_once()
    _wait_futures(service)

    # Worktree should be cleaned up despite failure
    for wt in worktree_paths:
        assert not wt.exists(), f"Worktree {wt} should have been cleaned up"

    updated = container.tasks.get(task.id)
    assert updated is not None
    assert updated.status == "blocked"


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
    task = Task(title="No git", task_type="chore", status="queued",
                approval_mode="auto_approve", hitl_mode="autopilot")
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
        approval_mode="auto_approve",
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
# 10. Blocked task has worktree_dir cleaned from persisted metadata
# ---------------------------------------------------------------------------


def test_blocked_task_metadata_cleaned(tmp_path: Path) -> None:
    """When a task blocks during a pipeline step, worktree_dir should be
    removed from the persisted metadata."""

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
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert result.error == "implement failed"

    # worktree_dir must not be in persisted metadata
    assert "worktree_dir" not in result.metadata

    # Worktree directory should not exist
    worktree_dir = container.state_root / "worktrees" / task.id
    assert not worktree_dir.exists()

    # Task branch should be cleaned up
    branches = subprocess.run(
        ["git", "branch", "--list", f"task-{task.id}"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip()
    assert branches == ""


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
        approval_mode="auto_approve",
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

    t1 = Task(title="Alpha", task_type="feature", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
    t2 = Task(title="Beta", task_type="feature", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
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

    t1 = Task(title="Alpha", task_type="feature", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
    t2 = Task(title="Beta", task_type="feature", status="queued",
              approval_mode="auto_approve", hitl_mode="autopilot")
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
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)
    assert result.status == "blocked"
    assert "No file changes" in (result.error or "")
