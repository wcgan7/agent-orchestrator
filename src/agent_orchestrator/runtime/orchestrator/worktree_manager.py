"""Git worktree, branch, and merge helpers for orchestrator tasks."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .service import OrchestratorService

logger = logging.getLogger(__name__)


class WorktreeManager:
    """Coordinate task branch/worktree lifecycle and merge conflict handling."""

    def __init__(self, service: OrchestratorService) -> None:
        """Bind the manager to the owning orchestrator service state."""
        self._service = service

    def create_worktree(self, task: Any) -> Optional[Path]:
        """Create a task-specific worktree and branch when git metadata exists.

        Returns ``None`` for non-git projects so callers can fall back to
        in-place execution in the primary repository directory.
        """
        svc = self._service
        git_dir = svc.container.project_dir / ".git"
        if not git_dir.exists():
            return None
        self.ensure_branch()
        task_id = str(task.id)
        worktree_dir = svc.container.state_root / "worktrees" / task_id
        branch = f"task-{task_id}"
        subprocess.run(
            ["git", "worktree", "add", str(worktree_dir), "-b", branch],
            cwd=svc.container.project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return worktree_dir

    def merge_and_cleanup(self, task: Any, worktree_dir: Path) -> None:
        """Merge task work into the run branch, then remove transient worktree.

        On merge conflicts the method attempts automated resolution and marks
        ``task.metadata["merge_conflict"]`` when conflict handling fails.
        """
        svc = self._service
        branch = f"task-{task.id}"
        merge_failed = False
        with svc._merge_lock:
            try:
                subprocess.run(
                    ["git", "merge", branch, "--no-edit"],
                    cwd=svc.container.project_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                resolved = self.resolve_merge_conflict(task, branch)
                if not resolved:
                    subprocess.run(
                        ["git", "merge", "--abort"],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    merge_failed = True
                    task.metadata["merge_conflict"] = True
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_dir), "--force"],
            cwd=svc.container.project_dir,
            capture_output=True,
            text=True,
        )
        if not merge_failed:
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=svc.container.project_dir,
                capture_output=True,
                text=True,
            )

    def approve_and_merge(self, task: Any) -> dict[str, Any]:
        """Merge a preserved task branch after manual review approval.

        Returns a status payload for API handlers and persists metadata cleanup
        when the branch no longer exists or merges successfully.
        """
        svc = self._service
        branch = task.metadata.get("preserved_branch")
        if not branch:
            return {"status": "ok"}

        result = subprocess.run(
            ["git", "branch", "--list", branch],
            cwd=svc.container.project_dir,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            task.metadata.pop("preserved_branch", None)
            svc.container.tasks.upsert(task)
            return {"status": "ok"}

        self.ensure_branch()

        with svc._merge_lock:
            try:
                subprocess.run(
                    ["git", "merge", branch, "--no-edit"],
                    cwd=svc.container.project_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                resolved = self.resolve_merge_conflict(task, branch)
                if not resolved:
                    subprocess.run(
                        ["git", "merge", "--abort"],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )
                    task.metadata["merge_conflict"] = True
                    svc.container.tasks.upsert(task)
                    return {"status": "merge_conflict"}

            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=svc.container.project_dir,
                capture_output=True,
                text=True,
            ).stdout.strip()

        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=svc.container.project_dir,
            capture_output=True,
            text=True,
        )
        task.metadata.pop("preserved_branch", None)
        task.metadata.pop("merge_conflict", None)
        svc.container.tasks.upsert(task)
        return {"status": "ok", "commit_sha": sha}

    def resolve_merge_conflict(self, task: Any, branch: str) -> bool:
        """Try worker-assisted conflict resolution for the currently running merge.

        The method snapshots conflict context into task metadata so the worker
        prompt includes file-level conflict text and related task objectives.
        """
        svc = self._service
        saved_worktree_dir = task.metadata.get("worktree_dir")
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=svc.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            conflicted_files = [f for f in result.stdout.strip().split("\n") if f]
            if not conflicted_files:
                return False

            conflict_contents: dict[str, str] = {}
            for fpath in conflicted_files:
                full = svc.container.project_dir / fpath
                if full.exists():
                    conflict_contents[fpath] = full.read_text(errors="replace")

            other_tasks_info: list[str] = []
            other_objectives: list[str] = []
            for other in svc.container.tasks.list():
                if other.id != task.id and other.status == "done":
                    other_tasks_info.append(f"- {other.title}: {other.description}")
                    other_objectives.append(self.format_task_objective_summary(other))

            task.metadata.pop("worktree_dir", None)
            task.metadata["merge_conflict_files"] = conflict_contents
            task.metadata["merge_other_tasks"] = other_tasks_info
            task.metadata["merge_current_objective"] = self.format_task_objective_summary(task)
            task.metadata["merge_other_objectives"] = other_objectives
            svc.container.tasks.upsert(task)

            step_result = svc.worker_adapter.run_step(task=task, step="resolve_merge", attempt=1)
            if step_result.status != "ok":
                return False

            subprocess.run(
                ["git", "add", "-A"],
                cwd=svc.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "--no-edit"],
                cwd=svc.container.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except Exception:
            logger.exception("Failed to resolve merge conflict for task %s", task.id)
            return False
        finally:
            task.metadata.pop("merge_conflict_files", None)
            task.metadata.pop("merge_other_tasks", None)
            task.metadata.pop("merge_current_objective", None)
            task.metadata.pop("merge_other_objectives", None)
            if saved_worktree_dir:
                task.metadata["worktree_dir"] = saved_worktree_dir

    def cleanup_orphaned_worktrees(self) -> None:
        """Remove leftover task worktrees and delete unneeded task branches."""
        svc = self._service
        worktrees_dir = svc.container.state_root / "worktrees"
        if not worktrees_dir.exists():
            return
        if not (svc.container.project_dir / ".git").exists():
            return
        preserved_branches: set[str] = set()
        for t in svc.container.tasks.list():
            pb = t.metadata.get("preserved_branch") if isinstance(t.metadata, dict) else None
            if pb:
                preserved_branches.add(str(pb))
        for child in worktrees_dir.iterdir():
            if child.is_dir():
                branch_name = f"task-{child.name}"
                subprocess.run(
                    ["git", "worktree", "remove", str(child), "--force"],
                    cwd=svc.container.project_dir,
                    capture_output=True,
                    text=True,
                )
                if branch_name not in preserved_branches:
                    subprocess.run(
                        ["git", "branch", "-D", branch_name],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )

    def ensure_branch(self) -> Optional[str]:
        """Create or reuse the shared orchestrator run branch in a thread-safe way."""
        svc = self._service
        if svc._run_branch:
            return svc._run_branch
        with svc._branch_lock:
            if svc._run_branch:
                return svc._run_branch
            git_dir = svc.container.project_dir / ".git"
            if not git_dir.exists():
                return None
            branch = f"orchestrator-run-{int(time.time())}"
            try:
                subprocess.run(
                    ["git", "checkout", "-B", branch],
                    cwd=svc.container.project_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                svc._run_branch = branch
                return branch
            except subprocess.CalledProcessError:
                return None

    def commit_for_task(self, task: Any, working_dir: Optional[Path] = None) -> Optional[str]:
        """Stage and commit task changes, returning the commit SHA on success.

        Returns ``None`` when there is no git repository or commit creation
        fails (for example when there are no staged changes).
        """
        svc = self._service
        cwd = working_dir or svc.container.project_dir
        if not (cwd / ".git").exists() and not (svc.container.project_dir / ".git").exists():
            return None
        if working_dir is None:
            self.ensure_branch()
        try:
            subprocess.run(["git", "add", "-A"], cwd=cwd, check=True, capture_output=True, text=True)
            subprocess.run(
                ["git", "commit", "-m", f"task({task.id}): {task.title[:60]}"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            return sha
        except subprocess.CalledProcessError:
            return None

    def has_uncommitted_changes(self, cwd: Path) -> bool:
        """Return whether git reports staged or unstaged changes for ``cwd``."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return True

    def preserve_worktree_work(self, task: Any, worktree_dir: Path) -> bool:
        """Persist task edits by keeping branch history but removing worktree dir.

        This is used when human review is required before final merge into the
        orchestrator run branch.
        """
        svc = self._service
        branch = f"task-{task.id}"
        try:
            svc._cleanup_workdoc_for_commit(worktree_dir)
            self.commit_for_task(task, worktree_dir)

            base_ref = svc._run_branch or "HEAD"
            try:
                result = subprocess.run(
                    ["git", "log", f"{base_ref}..{branch}", "--oneline"],
                    cwd=svc.container.project_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if not result.stdout.strip():
                    return False
            except subprocess.CalledProcessError:
                pass

            subprocess.run(
                ["git", "worktree", "remove", str(worktree_dir), "--force"],
                cwd=svc.container.project_dir,
                capture_output=True,
                text=True,
            )
            task.metadata["preserved_branch"] = branch
            svc.container.tasks.upsert(task)
            return True
        except Exception:
            logger.exception("Failed to preserve worktree work for task %s", task.id)
            return False

    def resolve_task_plan_excerpt(self, task: Any, *, max_chars: int = 800) -> str:
        """Extract bounded plan text used in merge-conflict prompt context."""
        svc = self._service
        if not isinstance(task.metadata, dict):
            return ""

        for key in ("committed_plan_revision_id", "latest_plan_revision_id"):
            rev_id = str(task.metadata.get(key) or "").strip()
            if not rev_id:
                continue
            revision = svc.container.plan_revisions.get(rev_id)
            if revision and revision.task_id == task.id and str(revision.content or "").strip():
                return str(revision.content).strip()[:max_chars]

        step_outputs = task.metadata.get("step_outputs")
        if isinstance(step_outputs, dict):
            plan_text = str(step_outputs.get("plan") or "").strip()
            if plan_text:
                return plan_text[:max_chars]
        return ""

    def format_task_objective_summary(self, task: Any, *, max_chars: int = 1600) -> str:
        """Compose a compact objective summary for conflict-resolution prompts."""
        lines = [f"- Task: {task.title}"]
        if task.description:
            lines.append(f"  Description: {task.description}")
        plan_excerpt = self.resolve_task_plan_excerpt(task)
        if plan_excerpt:
            lines.append("  Plan excerpt:")
            lines.append("  ---")
            lines.append(plan_excerpt)
            lines.append("  ---")
        return "\n".join(lines)[:max_chars]
