from __future__ import annotations

import logging
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional

from ...collaboration.modes import should_gate
from ...pipelines.registry import PipelineRegistry
from ..domain.models import ReviewCycle, ReviewFinding, RunRecord, Task, now_iso
from ..events.bus import EventBus
from ..storage.container import V3Container
from .worker_adapter import DefaultWorkerAdapter, StepResult, WorkerAdapter

logger = logging.getLogger(__name__)


def _has_cycle(adj: dict[str, list[str]], from_id: str, to_id: str) -> bool:
    """Return True if adding an edge from_id→to_id would create a cycle.

    Checks whether to_id can already reach from_id via existing edges.
    """
    visited: set[str] = set()
    stack = [to_id]
    while stack:
        node = stack.pop()
        if node == from_id:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj.get(node, []))
    return False


class OrchestratorService:
    _GATE_MAPPING: dict[str, str] = {
        "plan": "before_plan",
        "implement": "before_implement",
        "review": "after_implement",
        "commit": "before_commit",
    }
    _HUMAN_INTERVENTION_GATE = "human_intervention"

    def __init__(
        self,
        container: V3Container,
        bus: EventBus,
        *,
        worker_adapter: WorkerAdapter | None = None,
    ) -> None:
        self.container = container
        self.bus = bus
        self.worker_adapter = worker_adapter or DefaultWorkerAdapter()
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._drain = False
        self._run_branch: Optional[str] = None
        self._pool: ThreadPoolExecutor | None = None
        self._futures: dict[str, Future] = {}
        self._futures_lock = threading.Lock()
        self._merge_lock = threading.Lock()
        self._branch_lock = threading.Lock()

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            cfg = self.container.config.load()
            max_workers = int(dict(cfg.get("orchestrator") or {}).get("concurrency", 2) or 2)
            self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="v3-task")
        return self._pool

    def status(self) -> dict[str, Any]:
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        tasks = self.container.tasks.list()
        queue_depth = len([task for task in tasks if task.status == "ready"])
        in_progress = len([task for task in tasks if task.status == "in_progress"])
        with self._futures_lock:
            active_workers = len(self._futures)
        return {
            "status": orchestrator_cfg.get("status", "running"),
            "queue_depth": queue_depth,
            "in_progress": in_progress,
            "active_workers": active_workers,
            "draining": self._drain,
            "run_branch": self._run_branch,
        }

    def control(self, action: str) -> dict[str, Any]:
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        if action == "pause":
            orchestrator_cfg["status"] = "paused"
        elif action == "resume":
            orchestrator_cfg["status"] = "running"
        elif action == "drain":
            self._drain = True
            orchestrator_cfg["status"] = "running"
        elif action == "stop":
            self._stop.set()
            orchestrator_cfg["status"] = "stopped"
        else:
            raise ValueError(f"Unsupported control action: {action}")
        cfg["orchestrator"] = orchestrator_cfg
        self.container.config.save(cfg)
        self.bus.emit(channel="system", event_type="orchestrator.control", entity_id=self.container.project_id, payload={"action": action})
        return self.status()

    def ensure_worker(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._recover_in_progress_tasks()
            self._cleanup_orphaned_worktrees()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="v3-orchestrator")
            self._thread.start()

    def shutdown(self, *, timeout: float = 10.0) -> None:
        with self._lock:
            self._stop.set()
            thread = self._thread

        if thread and thread.is_alive():
            thread.join(timeout=max(timeout, 0.0))

        with self._futures_lock:
            inflight = list(self._futures.values())
        if inflight and timeout > 0:
            wait(inflight, timeout=timeout)

        pool = self._pool
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=False)
            self._pool = None

        with self._futures_lock:
            self._futures.clear()
        self._thread = None

    def _recover_in_progress_tasks(self) -> None:
        tasks = self.container.tasks.list()
        in_progress_ids = {task.id for task in tasks if task.status == "in_progress"}
        if not in_progress_ids:
            return

        for run in self.container.runs.list():
            if run.task_id in in_progress_ids and run.status == "in_progress" and not run.finished_at:
                run.status = "interrupted"
                run.finished_at = now_iso()
                run.summary = run.summary or "Interrupted by orchestrator restart"
                self.container.runs.upsert(run)

        for task in tasks:
            if task.id not in in_progress_ids:
                continue
            task.status = "ready"
            task.current_step = None
            task.current_agent_id = None
            task.pending_gate = None
            task.error = "Recovered from interrupted run"
            self.container.tasks.upsert(task)
            self.bus.emit(
                channel="tasks",
                event_type="task.recovered",
                entity_id=task.id,
                payload={"reason": "orchestrator_restart"},
            )

    def _sweep_futures(self) -> None:
        """Remove completed futures and log any unexpected errors."""
        with self._futures_lock:
            done_ids = [tid for tid, f in self._futures.items() if f.done()]
            for tid in done_ids:
                fut = self._futures.pop(tid)
                exc = fut.exception()
                if exc:
                    logger.error("Task %s raised unexpected error: %s", tid, exc, exc_info=exc)

    def tick_once(self) -> bool:
        self._sweep_futures()

        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        if orchestrator_cfg.get("status", "running") != "running":
            return False

        self._maybe_analyze_dependencies()

        max_in_progress = int(orchestrator_cfg.get("concurrency", 2) or 2)
        claimed = self.container.tasks.claim_next_runnable(max_in_progress=max_in_progress)
        if not claimed:
            return False

        self.bus.emit(channel="queue", event_type="task.claimed", entity_id=claimed.id, payload={"status": claimed.status})
        future = self._get_pool().submit(self._execute_task, claimed)
        with self._futures_lock:
            self._futures[claimed.id] = future
        return True

    def run_task(self, task_id: str) -> Task:
        wait_existing = False
        with self._lock:
            task = self.container.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            if task.pending_gate and task.status != "in_progress":
                raise ValueError(f"Task {task_id} is waiting for gate approval: {task.pending_gate}")
            # Make explicit run idempotent when a worker already started or finished
            # the same task; this avoids request races with the background loop.
            if task.status in {"in_review", "done"}:
                return task
            if task.status == "in_progress":
                wait_existing = True
            if task.status in {"cancelled"}:
                raise ValueError(f"Task {task_id} cannot be run from status={task.status}")

            if not wait_existing:
                terminal = {"done", "cancelled"}
                for dep_id in task.blocked_by:
                    dep = self.container.tasks.get(dep_id)
                    if dep is None or dep.status not in terminal:
                        raise ValueError(f"Task {task_id} has unresolved blocker {dep_id}")
                task.status = "ready"
                self.container.tasks.upsert(task)

        if wait_existing:
            with self._futures_lock:
                existing_future = self._futures.get(task_id)
            if existing_future:
                existing_future.result()
            updated = self.container.tasks.get(task_id)
            if not updated:
                raise ValueError(f"Task disappeared during execution: {task_id}")
            return updated

        future = self._get_pool().submit(self._execute_task, task)
        with self._futures_lock:
            self._futures[task_id] = future
        try:
            future.result()
        finally:
            with self._futures_lock:
                self._futures.pop(task_id, None)
        updated = self.container.tasks.get(task_id)
        if not updated:
            raise ValueError(f"Task disappeared during execution: {task_id}")
        return updated

    def _loop(self) -> None:
        while not self._stop.is_set():
            handled = self.tick_once()
            with self._futures_lock:
                has_inflight = bool(self._futures)
            if self._drain and not handled and not has_inflight:
                self.control("pause")
                self._drain = False
                break
            time.sleep(1 if handled else 2)

    def _create_worktree(self, task: Task) -> Optional[Path]:
        git_dir = self.container.project_dir / ".git"
        if not git_dir.exists():
            return None
        self._ensure_branch()  # ensure run branch exists as merge target
        worktree_dir = self.container.v3_root / "worktrees" / task.id
        branch = f"task-{task.id}"
        subprocess.run(
            ["git", "worktree", "add", str(worktree_dir), "-b", branch],
            cwd=self.container.project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return worktree_dir

    def _merge_and_cleanup(self, task: Task, worktree_dir: Path) -> None:
        branch = f"task-{task.id}"
        merge_failed = False
        with self._merge_lock:
            try:
                subprocess.run(
                    ["git", "merge", branch, "--no-edit"],
                    cwd=self.container.project_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                resolved = self._resolve_merge_conflict(task, branch)
                if not resolved:
                    subprocess.run(
                        ["git", "merge", "--abort"],
                        cwd=self.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    merge_failed = True
                    task.metadata["merge_conflict"] = True
        # Always clean up worktree
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_dir), "--force"],
            cwd=self.container.project_dir,
            capture_output=True,
            text=True,
        )
        # Only delete branch if merge succeeded; preserve it for recovery on failure
        if not merge_failed:
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=self.container.project_dir,
                capture_output=True,
                text=True,
            )

    def _resolve_merge_conflict(self, task: Task, branch: str) -> bool:
        saved_worktree_dir = task.metadata.get("worktree_dir")
        try:
            # Get conflicted files
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=self.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            conflicted_files = [f for f in result.stdout.strip().split("\n") if f]
            if not conflicted_files:
                return False

            # Read conflicted file contents
            conflict_contents: dict[str, str] = {}
            for fpath in conflicted_files:
                full = self.container.project_dir / fpath
                if full.exists():
                    conflict_contents[fpath] = full.read_text(errors="replace")

            # Identify other recently completed tasks whose changes may conflict
            other_tasks_info: list[str] = []
            for other in self.container.tasks.list():
                if other.id != task.id and other.status == "done":
                    other_tasks_info.append(f"- {other.title}: {other.description}")

            # Build resolve prompt and store in task metadata.
            # Temporarily clear worktree_dir so the worker runs in project_dir
            # (where the merge conflict lives), not the worktree.
            task.metadata.pop("worktree_dir", None)
            task.metadata["merge_conflict_files"] = conflict_contents
            task.metadata["merge_other_tasks"] = other_tasks_info
            self.container.tasks.upsert(task)

            # Dispatch worker to resolve
            step_result = self.worker_adapter.run_step(task=task, step="resolve_merge", attempt=1)

            if step_result.status != "ok":
                return False

            # Stage and commit the resolution
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "--no-edit"],
                cwd=self.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except Exception:
            logger.exception("Failed to resolve merge conflict for task %s", task.id)
            return False
        finally:
            # Always clean up conflict metadata and restore worktree_dir
            task.metadata.pop("merge_conflict_files", None)
            task.metadata.pop("merge_other_tasks", None)
            if saved_worktree_dir:
                task.metadata["worktree_dir"] = saved_worktree_dir

    def _cleanup_orphaned_worktrees(self) -> None:
        worktrees_dir = self.container.v3_root / "worktrees"
        if not worktrees_dir.exists():
            return
        if not (self.container.project_dir / ".git").exists():
            return
        for child in worktrees_dir.iterdir():
            if child.is_dir():
                branch_name = f"task-{child.name}"
                subprocess.run(
                    ["git", "worktree", "remove", str(child), "--force"],
                    cwd=self.container.project_dir,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    cwd=self.container.project_dir,
                    capture_output=True,
                    text=True,
                )

    def _role_for_task(self, task: Task) -> str:
        cfg = self.container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        by_type = dict(routing.get("task_type_roles") or {})
        default_role = str(routing.get("default_role") or "general")
        return str(by_type.get(task.task_type) or default_role)

    def _provider_override_for_role(self, role: str) -> Optional[str]:
        cfg = self.container.config.load()
        routing = dict(cfg.get("agent_routing") or {})
        overrides = dict(routing.get("role_provider_overrides") or {})
        raw = overrides.get(role)
        return str(raw) if raw else None

    def _choose_agent_for_task(self, task: Task) -> Optional[str]:
        desired_role = self._role_for_task(task)
        running = [agent for agent in self.container.agents.list() if agent.status == "running"]
        exact = [agent for agent in running if agent.role == desired_role]
        pool = exact or running
        if not pool:
            return None
        pool.sort(key=lambda agent: agent.last_seen_at)
        chosen = pool[0]
        override_provider = self._provider_override_for_role(chosen.role)
        if override_provider:
            task.metadata["provider_override"] = override_provider
        return chosen.id

    def _ensure_branch(self) -> Optional[str]:
        if self._run_branch:
            return self._run_branch
        with self._branch_lock:
            # Double-check after acquiring lock
            if self._run_branch:
                return self._run_branch
            git_dir = self.container.project_dir / ".git"
            if not git_dir.exists():
                return None
            branch = f"orchestrator-run-{int(time.time())}"
            try:
                subprocess.run(["git", "checkout", "-B", branch], cwd=self.container.project_dir, check=True, capture_output=True, text=True)
                self._run_branch = branch
                return branch
            except subprocess.CalledProcessError:
                return None

    def _commit_for_task(self, task: Task, working_dir: Optional[Path] = None) -> Optional[str]:
        cwd = working_dir or self.container.project_dir
        if not (cwd / ".git").exists() and not (self.container.project_dir / ".git").exists():
            return None
        if working_dir is None:
            self._ensure_branch()
        try:
            subprocess.run(["git", "add", "-A"], cwd=cwd, check=True, capture_output=True, text=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", f"task({task.id}): {task.title[:60]}"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
            return sha
        except subprocess.CalledProcessError:
            return None

    def _exceeds_quality_gate(self, task: Task, findings: list[ReviewFinding]) -> bool:
        gate = dict(task.quality_gate or {})
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for finding in findings:
            if finding.status != "open":
                continue
            sev = finding.severity if finding.severity in counts else "low"
            counts[sev] += 1
        return any(counts[sev] > int(gate.get(sev, 0)) for sev in counts)

    def _run_non_review_step(self, task: Task, run: RunRecord, step: str, attempt: int = 1) -> bool:
        result = self.worker_adapter.run_step(task=task, step=step, attempt=attempt)
        step_log: dict[str, Any] = {"step": step, "status": result.status, "ts": now_iso(), "summary": result.summary}
        if result.human_blocking_issues:
            step_log["human_blocking_issues"] = result.human_blocking_issues
        run.steps.append(step_log)
        task.current_step = step
        self.container.tasks.upsert(task)
        if result.human_blocking_issues:
            self._block_for_human_issues(task, run, step, result.summary, result.human_blocking_issues)
            return False
        if result.status != "ok":
            task.status = "blocked"
            task.error = result.summary or f"{step} failed"
            task.pending_gate = None
            task.current_step = step
            self.container.tasks.upsert(task)
            run.status = "blocked"
            run.finished_at = now_iso()
            run.summary = f"Blocked during {step}"
            self.container.runs.upsert(run)
            self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})
            return False

        task.metadata.pop("human_blocking_issues", None)

        # Handle generate_tasks: create child tasks from step output
        if step == "generate_tasks" and result.generated_tasks:
            self._create_child_tasks(task, result.generated_tasks)

        return True

    def _create_child_tasks(self, parent: Task, task_defs: list[dict[str, Any]]) -> list[str]:
        created_ids: list[str] = []
        for item in task_defs:
            if not isinstance(item, dict):
                continue
            child = Task(
                title=str(item.get("title") or "Generated task"),
                description=str(item.get("description") or ""),
                task_type=str(item.get("task_type") or "feature"),
                priority=str(item.get("priority") or parent.priority),
                parent_id=parent.id,
                source="generated",
                labels=list(item.get("labels") or []),
                metadata=dict(item.get("metadata") or {}),
            )
            self.container.tasks.upsert(child)
            created_ids.append(child.id)
            self.bus.emit(
                channel="tasks",
                event_type="task.created",
                entity_id=child.id,
                payload={"parent_id": parent.id, "source": "generate_tasks"},
            )
        if created_ids:
            parent.children_ids.extend(created_ids)
            self.container.tasks.upsert(parent)
        return created_ids

    def _findings_from_result(self, task: Task, review_attempt: int) -> tuple[list[ReviewFinding], StepResult]:
        result = self.worker_adapter.run_step(task=task, step="review", attempt=review_attempt)
        raw_findings = list(result.findings or [])
        findings: list[ReviewFinding] = []
        for idx, finding in enumerate(raw_findings):
            if not isinstance(finding, dict):
                continue
            findings.append(
                ReviewFinding(
                    id=f"{task.id}-a{review_attempt}-{idx}",
                    task_id=task.id,
                    severity=str(finding.get("severity") or "medium"),
                    category=str(finding.get("category") or "quality"),
                    summary=str(finding.get("summary") or "Issue"),
                    file=finding.get("file"),
                    line=finding.get("line"),
                    suggested_fix=finding.get("suggested_fix"),
                    status=str(finding.get("status") or "open"),
                )
            )
        return findings, result

    def _block_for_human_issues(
        self,
        task: Task,
        run: RunRecord,
        step: str,
        summary: str | None,
        issues: list[dict[str, str]],
    ) -> None:
        task.status = "blocked"
        task.current_step = step
        task.pending_gate = self._HUMAN_INTERVENTION_GATE
        task.error = summary or "Human intervention required to continue"
        task.metadata["human_blocking_issues"] = issues
        self.container.tasks.upsert(task)

        run.status = "blocked"
        run.finished_at = now_iso()
        run.summary = f"Blocked during {step}: human intervention required"
        self.container.runs.upsert(run)

        self.bus.emit(
            channel="tasks",
            event_type="task.gate_waiting",
            entity_id=task.id,
            payload={"gate": self._HUMAN_INTERVENTION_GATE, "step": step, "issues": issues},
        )
        self.bus.emit(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={
                "error": task.error,
                "gate": self._HUMAN_INTERVENTION_GATE,
                "step": step,
                "issues": issues,
            },
        )

    def _wait_for_gate(self, task: Task, gate_name: str, timeout: int = 3600) -> bool:
        """Block until ``pending_gate`` is cleared (via approve-gate API).

        Returns True if gate was approved, False on timeout/stop/cancel.
        """
        task.pending_gate = gate_name
        task.updated_at = now_iso()
        self.container.tasks.upsert(task)
        self.bus.emit(
            channel="tasks",
            event_type="task.gate_waiting",
            entity_id=task.id,
            payload={"gate": gate_name},
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop.is_set():
                return False
            fresh = self.container.tasks.get(task.id)
            if fresh is None or fresh.status == "cancelled":
                return False
            if fresh.pending_gate is None:
                return True
            time.sleep(1)
        return False

    def _abort_for_gate(self, task: Task, run: RunRecord, gate_name: str) -> None:
        """Mark task as blocked because a gate was not approved."""
        task.status = "blocked"
        task.error = f"Gate '{gate_name}' was not approved in time"
        task.pending_gate = None
        self.container.tasks.upsert(task)
        run.status = "blocked"
        run.finished_at = now_iso()
        run.summary = f"Blocked at gate: {gate_name}"
        self.container.runs.upsert(run)
        self.bus.emit(
            channel="tasks",
            event_type="task.blocked",
            entity_id=task.id,
            payload={"error": task.error},
        )

    def _maybe_analyze_dependencies(self) -> None:
        """Run automatic dependency analysis on unanalyzed ready tasks."""
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        if not orchestrator_cfg.get("auto_deps", True):
            return

        all_tasks = self.container.tasks.list()
        candidates = [
            t for t in all_tasks
            if t.status == "ready"
            and not (isinstance(t.metadata, dict) and t.metadata.get("deps_analyzed"))
            and t.source != "prd_import"
        ]

        # Mark all candidates analyzed regardless of outcome
        def _mark_analyzed(tasks: list[Task]) -> None:
            for t in tasks:
                if not isinstance(t.metadata, dict):
                    t.metadata = {}
                t.metadata["deps_analyzed"] = True
                self.container.tasks.upsert(t)

        if len(candidates) < 2:
            _mark_analyzed(candidates)
            return

        # Gather already-analyzed non-terminal tasks as context
        terminal = {"done", "cancelled"}
        existing = [
            t for t in all_tasks
            if isinstance(t.metadata, dict) and t.metadata.get("deps_analyzed")
            and t.status not in terminal
        ]

        # Build synthetic task with metadata for the worker
        candidate_data = [
            {
                "id": t.id,
                "title": t.title,
                "description": (t.description or "")[:200],
                "task_type": t.task_type,
                "labels": t.labels,
            }
            for t in candidates
        ]
        existing_data = [
            {"id": t.id, "title": t.title, "status": t.status}
            for t in existing
        ]

        synthetic = Task(
            title="Dependency analysis",
            description="Analyze task dependencies",
            task_type="research",
            source="system",
            metadata={
                "candidate_tasks": candidate_data,
                "existing_tasks": existing_data,
            },
        )

        try:
            result = self.worker_adapter.run_step(task=synthetic, step="analyze_deps", attempt=1)
            if result.status == "ok" and result.dependency_edges:
                self._apply_dependency_edges(candidates, result.dependency_edges, all_tasks)
        except Exception:
            logger.exception("Dependency analysis failed; tasks will run without inferred deps")
        finally:
            _mark_analyzed(candidates)

    def _apply_dependency_edges(
        self,
        candidates: list[Task],
        edges: list[dict[str, str]],
        all_tasks: list[Task],
    ) -> None:
        """Apply inferred dependency edges with cycle detection."""
        task_map: dict[str, Task] = {}
        # All tasks as context for resolving IDs outside candidate set
        for t in all_tasks:
            task_map[t.id] = t
        # Overlay candidate objects (same Python objects that _mark_analyzed will touch)
        for t in candidates:
            task_map[t.id] = t

        # Build adjacency list from existing blocked_by relationships
        adj: dict[str, list[str]] = {}
        for t in task_map.values():
            for dep_id in t.blocked_by:
                adj.setdefault(dep_id, []).append(t.id)

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            from_id = edge.get("from", "")
            to_id = edge.get("to", "")
            reason = edge.get("reason", "")

            if not from_id or not to_id:
                continue
            if from_id not in task_map or to_id not in task_map:
                continue
            if from_id == to_id:
                continue

            # Cycle check
            if _has_cycle(adj, from_id, to_id):
                logger.warning("Skipping edge %s→%s: would create cycle", from_id, to_id)
                continue

            from_task = task_map[from_id]
            to_task = task_map[to_id]

            if from_id not in to_task.blocked_by:
                to_task.blocked_by.append(from_id)
            if to_id not in from_task.blocks:
                from_task.blocks.append(to_id)

            # Store inferred deps for traceability
            if not isinstance(to_task.metadata, dict):
                to_task.metadata = {}
            inferred = to_task.metadata.setdefault("inferred_deps", [])
            inferred.append({"from": from_id, "reason": reason})

            # Update adjacency for subsequent cycle checks
            adj.setdefault(from_id, []).append(to_id)

            self.container.tasks.upsert(from_task)
            self.container.tasks.upsert(to_task)
            self.bus.emit(
                channel="tasks",
                event_type="task.dependency_inferred",
                entity_id=to_id,
                payload={"from": from_id, "to": to_id, "reason": reason},
            )

    def _execute_task(self, task: Task) -> None:
        try:
            self._execute_task_inner(task)
        except Exception:
            logger.exception("Unexpected error executing task %s", task.id)
            task.status = "blocked"
            task.error = "Internal error during execution"
            self.container.tasks.upsert(task)

    def _execute_task_inner(self, task: Task) -> None:
        worktree_dir: Optional[Path] = None
        try:
            worktree_dir = self._create_worktree(task)
            if worktree_dir:
                task.metadata["worktree_dir"] = str(worktree_dir)
                self.container.tasks.upsert(task)

            task_branch = f"task-{task.id}" if worktree_dir else self._ensure_branch()
            run = RunRecord(task_id=task.id, status="in_progress", started_at=now_iso(), branch=task_branch)
            run.steps = []
            self.container.runs.upsert(run)

            cfg = self.container.config.load()
            max_review_attempts = int(dict(cfg.get("orchestrator") or {}).get("max_review_attempts", 3) or 3)

            # Resolve pipeline template from registry
            registry = PipelineRegistry()
            template = registry.resolve_for_task_type(task.task_type)
            steps = task.pipeline_template if task.pipeline_template else template.step_names()
            task.pipeline_template = steps
            has_review = "review" in steps
            has_commit = "commit" in steps

            task.run_ids.append(run.id)
            task.current_step = steps[0] if steps else None
            task.status = "in_progress"
            task.current_agent_id = self._choose_agent_for_task(task)
            self.container.tasks.upsert(task)
            self.bus.emit(
                channel="tasks",
                event_type="task.started",
                entity_id=task.id,
                payload={"run_id": run.id, "agent_id": task.current_agent_id},
            )

            mode = getattr(task, "hitl_mode", "autopilot") or "autopilot"

            # Phase 1: Run all pre-review/pre-commit steps
            for step in steps:
                if step in ("review", "commit"):
                    continue
                gate_name = self._GATE_MAPPING.get(step)
                if gate_name and should_gate(mode, gate_name):
                    if not self._wait_for_gate(task, gate_name):
                        self._abort_for_gate(task, run, gate_name)
                        return
                if not self._run_non_review_step(task, run, step, attempt=1):
                    return

            # Phase 2: Review loop (only if template has "review")
            if has_review:
                gate_name = self._GATE_MAPPING.get("review")
                if gate_name and should_gate(mode, gate_name):
                    if not self._wait_for_gate(task, gate_name):
                        self._abort_for_gate(task, run, gate_name)
                        return

                review_attempt = 0
                review_passed = False

                while review_attempt < max_review_attempts:
                    review_attempt += 1
                    task.current_step = "review"
                    self.container.tasks.upsert(task)
                    findings, review_result = self._findings_from_result(task, review_attempt)
                    if review_result.human_blocking_issues:
                        self._block_for_human_issues(
                            task,
                            run,
                            "review",
                            review_result.summary,
                            review_result.human_blocking_issues,
                        )
                        return
                    if review_result.status != "ok":
                        task.status = "blocked"
                        task.error = review_result.summary or "Review step failed"
                        task.pending_gate = None
                        task.current_step = "review"
                        self.container.tasks.upsert(task)
                        run.status = "blocked"
                        run.finished_at = now_iso()
                        run.summary = "Blocked during review"
                        self.container.runs.upsert(run)
                        self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})
                        return
                    open_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                    for finding in findings:
                        if finding.status == "open" and finding.severity in open_counts:
                            open_counts[finding.severity] += 1
                    cycle = ReviewCycle(
                        task_id=task.id,
                        attempt=review_attempt,
                        findings=findings,
                        open_counts=open_counts,
                        decision="changes_requested" if self._exceeds_quality_gate(task, findings) else "approved",
                    )
                    self.container.reviews.append(cycle)
                    run.steps.append({"step": "review", "status": cycle.decision, "ts": now_iso(), "open_counts": open_counts})
                    self.bus.emit(
                        channel="review",
                        event_type="task.reviewed",
                        entity_id=task.id,
                        payload={"attempt": review_attempt, "decision": cycle.decision, "open_counts": open_counts},
                    )

                    if cycle.decision == "approved":
                        review_passed = True
                        break

                    if review_attempt >= max_review_attempts:
                        break

                    # Attach open findings so the worker knows what to fix
                    open_findings = [f.to_dict() for f in findings if f.status == "open"]
                    task.metadata["review_findings"] = open_findings
                    for fix_step in ["implement_fix", "verify"]:
                        task.retry_count += 1
                        self.container.tasks.upsert(task)
                        if not self._run_non_review_step(task, run, fix_step, attempt=review_attempt):
                            return
                    task.metadata.pop("review_findings", None)

                if not review_passed:
                    task.status = "blocked"
                    task.error = "Review attempt cap exceeded"
                    task.current_step = "review"
                    self.container.tasks.upsert(task)
                    run.status = "blocked"
                    run.finished_at = now_iso()
                    run.summary = "Blocked due to unresolved review findings"
                    self.container.runs.upsert(run)
                    self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})
                    return

            # Phase 3: Commit (only if template has "commit")
            if has_commit:
                if should_gate(mode, "before_commit"):
                    if not self._wait_for_gate(task, "before_commit"):
                        self._abort_for_gate(task, run, "before_commit")
                        return

                commit_sha = self._commit_for_task(task, worktree_dir)
                run.steps.append({"step": "commit", "status": "ok", "ts": now_iso(), "commit": commit_sha})

                # Merge worktree branch back to run branch
                if worktree_dir:
                    self._merge_and_cleanup(task, worktree_dir)
                    worktree_dir = None  # prevent double-cleanup in finally

                # If merge conflict couldn't be resolved, block the task
                if task.metadata.get("merge_conflict"):
                    task.status = "blocked"
                    task.error = "Merge conflict could not be resolved automatically"
                    task.metadata["unmerged_branch"] = f"task-{task.id}"
                    self.container.tasks.upsert(task)
                    run.status = "blocked"
                    run.finished_at = now_iso()
                    run.summary = "Blocked due to unresolved merge conflict"
                    self.container.runs.upsert(run)
                    self.bus.emit(
                        channel="tasks",
                        event_type="task.blocked",
                        entity_id=task.id,
                        payload={"error": task.error},
                    )
                    return

                if task.approval_mode == "auto_approve":
                    task.status = "done"
                    task.current_step = None
                    run.status = "done"
                    run.summary = "Completed with auto-approve"
                    self.bus.emit(channel="tasks", event_type="task.done", entity_id=task.id, payload={"commit": commit_sha})
                else:
                    task.status = "in_review"
                    task.current_step = None
                    run.status = "in_review"
                    run.summary = "Awaiting human review"
                    self.bus.emit(channel="review", event_type="task.awaiting_human", entity_id=task.id, payload={"commit": commit_sha})
            else:
                # Templates without commit (research, repo_review, security_audit, review)
                # Clean up worktree if present — no merge needed for non-commit pipelines
                if worktree_dir:
                    subprocess.run(
                        ["git", "worktree", "remove", str(worktree_dir), "--force"],
                        cwd=self.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "branch", "-D", f"task-{task.id}"],
                        cwd=self.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    worktree_dir = None  # prevent double-cleanup in finally

                task.status = "done"
                task.current_step = None
                run.status = "done"
                run.summary = "Pipeline completed"
                self.bus.emit(channel="tasks", event_type="task.done", entity_id=task.id, payload={})

            task.error = None
            task.metadata.pop("worktree_dir", None)
            self.container.tasks.upsert(task)
            run.finished_at = now_iso()
            self.container.runs.upsert(run)
        finally:
            # Clean up worktree on any failure path
            if worktree_dir and worktree_dir.exists():
                subprocess.run(
                    ["git", "worktree", "remove", str(worktree_dir), "--force"],
                    cwd=self.container.project_dir,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "branch", "-D", f"task-{task.id}"],
                    cwd=self.container.project_dir,
                    capture_output=True,
                    text=True,
                )
            if task.metadata.pop("worktree_dir", None):
                self.container.tasks.upsert(task)


def create_orchestrator(
    container: V3Container,
    bus: EventBus,
    *,
    worker_adapter: WorkerAdapter | None = None,
) -> OrchestratorService:
    if worker_adapter is None:
        from .live_worker_adapter import LiveWorkerAdapter

        worker_adapter = LiveWorkerAdapter(container)
    orchestrator = OrchestratorService(container, bus, worker_adapter=worker_adapter)
    orchestrator.ensure_worker()
    return orchestrator
