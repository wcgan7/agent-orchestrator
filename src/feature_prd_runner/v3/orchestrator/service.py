from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Optional

from ..domain.models import ReviewCycle, ReviewFinding, RunRecord, Task, now_iso
from ..events.bus import EventBus
from ..storage.container import V3Container
from .worker_adapter import DefaultWorkerAdapter, WorkerAdapter


class OrchestratorService:
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

    def status(self) -> dict[str, Any]:
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        tasks = self.container.tasks.list()
        queue_depth = len([task for task in tasks if task.status == "ready"])
        in_progress = len([task for task in tasks if task.status == "in_progress"])
        return {
            "status": orchestrator_cfg.get("status", "running"),
            "queue_depth": queue_depth,
            "in_progress": in_progress,
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
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="v3-orchestrator")
            self._thread.start()

    def tick_once(self) -> bool:
        cfg = self.container.config.load()
        orchestrator_cfg = dict(cfg.get("orchestrator") or {})
        if orchestrator_cfg.get("status", "running") != "running":
            return False

        max_in_progress = int(orchestrator_cfg.get("concurrency", 2) or 2)
        conflicts = self._active_repo_conflicts()
        claimed = self.container.tasks.claim_next_runnable(max_in_progress=max_in_progress, repo_conflicts=conflicts)
        if not claimed:
            return False

        self.bus.emit(channel="queue", event_type="task.claimed", entity_id=claimed.id, payload={"status": claimed.status})
        self._execute_task(claimed)
        return True

    def run_task(self, task_id: str) -> Task:
        with self._lock:
            task = self.container.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            if task.status in {"done", "cancelled", "in_review", "in_progress"}:
                raise ValueError(f"Task {task_id} cannot be run from status={task.status}")
            terminal = {"done", "cancelled"}
            for dep_id in task.blocked_by:
                dep = self.container.tasks.get(dep_id)
                if dep is None or dep.status not in terminal:
                    raise ValueError(f"Task {task_id} has unresolved blocker {dep_id}")
            task.status = "ready"
            self.container.tasks.upsert(task)
            self._execute_task(task)
            updated = self.container.tasks.get(task_id)
            if not updated:
                raise ValueError(f"Task disappeared during execution: {task_id}")
            return updated

    def _loop(self) -> None:
        while not self._stop.is_set():
            handled = self.tick_once()
            if self._drain and not handled:
                self.control("pause")
                self._drain = False
                break
            time.sleep(1 if handled else 2)

    def _active_repo_conflicts(self) -> set[str]:
        conflicts: set[str] = set()
        for task in self.container.tasks.list():
            if task.status == "in_progress":
                repo_path = str(task.metadata.get("repo_path") or "")
                if repo_path:
                    conflicts.add(repo_path)
        return conflicts

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

    def _commit_for_task(self, task: Task) -> Optional[str]:
        if not (self.container.project_dir / ".git").exists():
            return None
        self._ensure_branch()
        try:
            subprocess.run(["git", "add", "-A"], cwd=self.container.project_dir, check=True, capture_output=True, text=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", f"task({task.id}): {task.title[:60]}"],
                cwd=self.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.container.project_dir, check=True, capture_output=True, text=True).stdout.strip()
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
        run.steps.append({"step": step, "status": result.status, "ts": now_iso(), "summary": result.summary})
        task.current_step = step
        self.container.tasks.upsert(task)
        if result.status != "ok":
            task.status = "blocked"
            task.error = result.summary or f"{step} failed"
            task.current_step = step
            self.container.tasks.upsert(task)
            run.status = "blocked"
            run.finished_at = now_iso()
            run.summary = f"Blocked during {step}"
            self.container.runs.upsert(run)
            self.bus.emit(channel="tasks", event_type="task.blocked", entity_id=task.id, payload={"error": task.error})
            return False
        return True

    def _findings_from_result(self, task: Task, review_attempt: int) -> list[ReviewFinding]:
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
        return findings

    def _execute_task(self, task: Task) -> None:
        run = RunRecord(task_id=task.id, status="in_progress", started_at=now_iso(), branch=self._ensure_branch())
        run.steps = []
        self.container.runs.upsert(run)

        cfg = self.container.config.load()
        max_review_attempts = int(dict(cfg.get("orchestrator") or {}).get("max_review_attempts", 3) or 3)

        task.run_ids.append(run.id)
        task.current_step = "plan"
        task.status = "in_progress"
        task.current_agent_id = self._choose_agent_for_task(task)
        self.container.tasks.upsert(task)
        self.bus.emit(
            channel="tasks",
            event_type="task.started",
            entity_id=task.id,
            payload={"run_id": run.id, "agent_id": task.current_agent_id},
        )

        for base_step in ["plan", "implement", "verify"]:
            if not self._run_non_review_step(task, run, base_step, attempt=1):
                return

        review_attempt = 0
        review_passed = False

        while review_attempt < max_review_attempts:
            review_attempt += 1
            task.current_step = "review"
            self.container.tasks.upsert(task)
            findings = self._findings_from_result(task, review_attempt)
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

            for fix_step in ["implement_fix", "verify"]:
                task.retry_count += 1
                self.container.tasks.upsert(task)
                if not self._run_non_review_step(task, run, fix_step, attempt=review_attempt):
                    return

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

        commit_sha = self._commit_for_task(task)
        run.steps.append({"step": "commit", "status": "ok", "ts": now_iso(), "commit": commit_sha})

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

        task.error = None
        self.container.tasks.upsert(task)
        run.finished_at = now_iso()
        self.container.runs.upsert(run)


def create_orchestrator(
    container: V3Container,
    bus: EventBus,
    *,
    worker_adapter: WorkerAdapter | None = None,
) -> OrchestratorService:
    orchestrator = OrchestratorService(container, bus, worker_adapter=worker_adapter)
    orchestrator.ensure_worker()
    return orchestrator
