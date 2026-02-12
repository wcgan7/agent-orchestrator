from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Optional

from ...collaboration.modes import should_gate
from ...pipelines.registry import PipelineRegistry
from ..domain.models import ReviewCycle, ReviewFinding, RunRecord, Task, now_iso
from ..events.bus import EventBus
from ..storage.container import V3Container
from .worker_adapter import DefaultWorkerAdapter, WorkerAdapter


class OrchestratorService:
    _GATE_MAPPING: dict[str, str] = {
        "plan": "before_plan",
        "implement": "before_implement",
        "review": "after_implement",
        "commit": "before_commit",
    }

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
            # Make explicit run idempotent when a worker already started or finished
            # the same task; this avoids request races with the background loop.
            if task.status in {"in_progress", "in_review", "done"}:
                return task
            if task.status in {"cancelled"}:
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

    def _execute_task(self, task: Task) -> None:
        run = RunRecord(task_id=task.id, status="in_progress", started_at=now_iso(), branch=self._ensure_branch())
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
        else:
            # Templates without commit (research, repo_review, security_audit, review)
            task.status = "done"
            task.current_step = None
            run.status = "done"
            run.summary = "Pipeline completed"
            self.bus.emit(channel="tasks", event_type="task.done", entity_id=task.id, payload={})

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
    if worker_adapter is None:
        from .live_worker_adapter import LiveWorkerAdapter

        worker_adapter = LiveWorkerAdapter(container)
    orchestrator = OrchestratorService(container, bus, worker_adapter=worker_adapter)
    orchestrator.ensure_worker()
    return orchestrator
