"""Git worktree, branch, and merge helpers for orchestrator tasks."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypedDict

from ..domain.models import now_iso

if TYPE_CHECKING:
    from .service import OrchestratorService

logger = logging.getLogger(__name__)


class PreserveOutcome(TypedDict):
    """Structured preserve result for deterministic blocked/review handling."""

    status: str
    reason_code: str
    commit_sha: str | None
    base_sha: str | None
    head_sha: str | None


class WorktreeManager:
    """Coordinate task branch/worktree lifecycle and merge conflict handling."""

    def __init__(self, service: OrchestratorService) -> None:
        """Bind the manager to the owning orchestrator service state."""
        self._service = service

    _WORKTREE_ADD_MAX_ATTEMPTS = 3
    _WORKTREE_ADD_RETRY_SLEEP_SECONDS = 0.05
    _TRANSIENT_WORKTREE_ADD_ERROR_HINTS = (
        "already checked out at",
        "is already checked out",
        "worktree is already registered",
        "another git process seems to be running",
        "index.lock",
        "unable to create",
        "cannot lock ref",
        "could not lock",
        "resource temporarily unavailable",
    )

    @staticmethod
    def _local_branch_exists(project_dir: Path, branch_name: str) -> bool:
        """Return whether a local branch currently exists."""
        normalized = str(branch_name or "").strip()
        if not normalized:
            return False
        try:
            result = subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{normalized}"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return False
        return result.returncode == 0

    def _prepare_worktree_dir(self, worktree_dir: Path) -> None:
        """Ensure target worktree path is clear before adding a new worktree."""
        svc = self._service
        if not worktree_dir.exists():
            return
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_dir), "--force"],
            cwd=svc.container.project_dir,
            capture_output=True,
            text=True,
        )
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir, ignore_errors=True)

    @classmethod
    def _is_transient_worktree_add_error(cls, stderr: str) -> bool:
        lowered = str(stderr or "").lower()
        if not lowered:
            return False
        return any(hint in lowered for hint in cls._TRANSIENT_WORKTREE_ADD_ERROR_HINTS)

    def _add_worktree_with_retry(
        self,
        *,
        worktree_dir: Path,
        branch: str,
        prefer_create_branch: bool,
    ) -> None:
        """Add task worktree with bounded retries for git-lock/registration races."""
        svc = self._service
        last_exc: subprocess.CalledProcessError | None = None
        for attempt in range(1, self._WORKTREE_ADD_MAX_ATTEMPTS + 1):
            self._prepare_worktree_dir(worktree_dir)
            create_branch = bool(
                prefer_create_branch and not self._local_branch_exists(svc.container.project_dir, branch)
            )
            cmd = ["git", "worktree", "add", str(worktree_dir)]
            if create_branch:
                cmd.extend(["-b", branch])
            else:
                cmd.append(branch)
            try:
                subprocess.run(
                    cmd,
                    cwd=svc.container.project_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return
            except subprocess.CalledProcessError as exc:
                resolved_exc = exc
                stderr = str(getattr(exc, "stderr", "") or "")
                # Handle TOCTOU branch races: branch appeared after our local check.
                if create_branch and "already exists" in stderr.lower():
                    try:
                        self._prepare_worktree_dir(worktree_dir)
                        subprocess.run(
                            ["git", "worktree", "add", str(worktree_dir), branch],
                            cwd=svc.container.project_dir,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        return
                    except subprocess.CalledProcessError as race_exc:
                        resolved_exc = race_exc
                        stderr = str(getattr(race_exc, "stderr", "") or "")
                last_exc = resolved_exc
                if not self._is_transient_worktree_add_error(stderr):
                    raise resolved_exc
                # Best-effort prune clears stale worktree registrations before retry.
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=svc.container.project_dir,
                    capture_output=True,
                    text=True,
                )
                time.sleep(self._WORKTREE_ADD_RETRY_SLEEP_SECONDS * attempt)
        if last_exc is not None:
            raise last_exc

    @staticmethod
    def _clear_preserved_context_metadata(task: Any) -> None:
        """Remove preserved-branch metadata keys after merge/reattach."""
        if not isinstance(getattr(task, "metadata", None), dict):
            return
        for key in (
            "preserved_branch",
            "preserved_base_branch",
            "preserved_base_sha",
            "preserved_head_sha",
            "preserved_merge_base_sha",
            "preserved_at",
        ):
            task.metadata.pop(key, None)

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
        self._add_worktree_with_retry(
            worktree_dir=worktree_dir,
            branch=branch,
            prefer_create_branch=True,
        )
        return worktree_dir

    def create_worktree_from_branch(self, task: Any, branch: str) -> Optional[Path]:
        """Create a worktree by checking out an existing branch (no ``-b`` flag).

        Used when retrying a task that has a preserved branch so that prior
        committed work is carried forward into the new worktree.
        """
        svc = self._service
        git_dir = svc.container.project_dir / ".git"
        if not git_dir.exists():
            return None
        self.ensure_branch()
        task_id = str(task.id)
        worktree_dir = svc.container.state_root / "worktrees" / task_id
        self._add_worktree_with_retry(
            worktree_dir=worktree_dir,
            branch=branch,
            prefer_create_branch=False,
        )
        return worktree_dir

    def merge_and_cleanup(self, task: Any, worktree_dir: Path) -> None:
        """Merge task work into the base branch, then remove transient worktree.

        On merge conflicts the method attempts automated resolution and marks
        ``task.metadata["merge_conflict"]`` when conflict handling fails.
        """
        svc = self._service
        branch = f"task-{task.id}"
        merge_failed = False
        with svc._merge_lock:
            try:
                subprocess.run(
                    ["git", "merge", branch, "--ff", "--no-edit"],
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
        if not merge_failed:
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
            svc._integration_health.record_merge()

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
            self._clear_preserved_context_metadata(task)
            svc.container.tasks.upsert(task)
            return {"status": "ok"}

        self.ensure_branch()

        with svc._merge_lock:
            try:
                subprocess.run(
                    ["git", "merge", branch, "--ff", "--no-edit"],
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
        svc._integration_health.record_merge()
        self._clear_preserved_context_metadata(task)
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
            peers = [other for other in svc.container.tasks.list() if other.id != task.id and other.status == "done"]
            if not peers:
                peers = [other for other in svc.container.tasks.list() if other.id != task.id]
            for other in peers:
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
        referenced_branches: set[str] = set()
        referenced_worktrees: set[Path] = set()
        terminal_statuses = {"done", "cancelled"}
        for t in svc.container.tasks.list():
            metadata = t.metadata if isinstance(t.metadata, dict) else {}
            if t.status not in terminal_statuses:
                task_context_raw = metadata.get("task_context")
                task_context = task_context_raw if isinstance(task_context_raw, dict) else {}
                for key in ("worktree_dir",):
                    raw_path = str(task_context.get(key) or "").strip()
                    if raw_path:
                        try:
                            referenced_worktrees.add(Path(raw_path).expanduser().resolve())
                        except Exception:
                            pass
                raw_worktree_dir = str(metadata.get("worktree_dir") or "").strip()
                if raw_worktree_dir:
                    try:
                        referenced_worktrees.add(Path(raw_worktree_dir).expanduser().resolve())
                    except Exception:
                        pass
                for branch_key in ("task_branch", "preserved_branch"):
                    branch = str(task_context.get(branch_key) or "").strip()
                    if branch:
                        referenced_branches.add(branch)
                review_context_raw = metadata.get("review_context")
                review_context = review_context_raw if isinstance(review_context_raw, dict) else {}
                review_branch = str(review_context.get("preserved_branch") or "").strip()
                if review_branch:
                    referenced_branches.add(review_branch)
            pb = str(metadata.get("preserved_branch") or "").strip()
            if pb:
                referenced_branches.add(pb)
        for child in worktrees_dir.iterdir():
            if child.is_dir():
                try:
                    child_resolved = child.resolve()
                except Exception:
                    child_resolved = child
                if child_resolved in referenced_worktrees:
                    continue
                branch_name = f"task-{child.name}"
                subprocess.run(
                    ["git", "worktree", "remove", str(child), "--force"],
                    cwd=svc.container.project_dir,
                    capture_output=True,
                    text=True,
                )
                if branch_name not in referenced_branches:
                    subprocess.run(
                        ["git", "branch", "-D", branch_name],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                    )

    def ensure_branch(self) -> Optional[str]:
        """Record the user's current branch as the orchestrator base branch.

        Previously this created an ephemeral ``orchestrator-run-*`` branch.
        Now it simply reads the current branch name so that task merges land
        directly on whatever branch the user was on (e.g. ``main``).
        """
        svc = self._service
        if svc._run_branch:
            return svc._run_branch
        with svc._branch_lock:
            if svc._run_branch:
                return svc._run_branch
            git_dir = svc.container.project_dir / ".git"
            if not git_dir.exists():
                return None
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=svc.container.project_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                branch = result.stdout.strip()
                if branch and branch != "HEAD":
                    svc._run_branch = branch
                    return branch
                return None
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

    def has_commits_ahead(self, cwd: Path) -> bool:
        """Return whether ``cwd``'s HEAD has commits beyond the base branch.

        Used to detect prior committed work on a task branch (e.g. from a
        preserved branch) even when there are no uncommitted changes.
        """
        svc = self._service
        base_ref = str(svc._run_branch or "").strip()
        if not base_ref:
            try:
                current_branch = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=svc.container.project_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                candidate = str(current_branch.stdout or "").strip()
                if current_branch.returncode == 0 and candidate and candidate != "HEAD":
                    base_ref = candidate
            except Exception:
                base_ref = ""
        if not base_ref:
            base_ref = "HEAD"
        try:
            result = subprocess.run(
                ["git", "log", f"{base_ref}..HEAD", "--oneline"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            # Err on the side of preserving work — matches has_uncommitted_changes
            # which also returns the conservative default (True) on git errors.
            return True

    def preserve_worktree_work(self, task: Any, worktree_dir: Path) -> PreserveOutcome:
        """Persist task edits by keeping branch history but removing worktree dir.

        This is used when human review is required before final merge into the
        base branch.
        """
        svc = self._service
        branch = f"task-{task.id}"
        try:
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            had_uncommitted = self.has_uncommitted_changes(worktree_dir)
            had_commits_ahead = self.has_commits_ahead(worktree_dir)
            has_material_work = had_uncommitted or had_commits_ahead
            if not has_material_work:
                return {
                    "status": "no_changes",
                    "reason_code": "no_task_changes",
                    "commit_sha": None,
                    "base_sha": None,
                    "head_sha": None,
                }

            before_head = ""
            before_head_result = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=worktree_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if before_head_result.returncode == 0:
                before_head = before_head_result.stdout.strip()
            svc._cleanup_workdoc_for_commit(worktree_dir)
            commit_sha = self.commit_for_task(task, worktree_dir)
            after_head = ""
            after_head_result = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=worktree_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if after_head_result.returncode == 0:
                after_head = after_head_result.stdout.strip()

            if had_uncommitted and (not commit_sha or not after_head or after_head == before_head):
                return {
                    "status": "failed",
                    "reason_code": "dirty_not_preserved",
                    "commit_sha": commit_sha,
                    "base_sha": None,
                    "head_sha": after_head or None,
                }
            if not commit_sha and not had_commits_ahead:
                return {
                    "status": "failed",
                    "reason_code": "preserve_commit_missing",
                    "commit_sha": None,
                    "base_sha": None,
                    "head_sha": after_head or None,
                }

            base_ref = str(svc._run_branch or "").strip()
            if not base_ref:
                try:
                    current_branch_result = subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )
                    candidate = current_branch_result.stdout.strip()
                    if current_branch_result.returncode == 0 and candidate and candidate != "HEAD":
                        base_ref = candidate
                except Exception:
                    base_ref = ""
            if not base_ref:
                base_ref = "HEAD"
            try:
                result = subprocess.run(
                    ["git", "log", f"{base_ref}..{branch}", "--oneline"],
                    cwd=svc.container.project_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if not result.stdout.strip():
                    return {
                        "status": "failed",
                        "reason_code": "branch_ahead_missing",
                        "commit_sha": commit_sha,
                        "base_sha": None,
                        "head_sha": after_head or None,
                    }
            except subprocess.CalledProcessError:
                pass

            subprocess.run(
                ["git", "worktree", "remove", str(worktree_dir), "--force"],
                cwd=svc.container.project_dir,
                capture_output=True,
                text=True,
            )
            task.metadata["preserved_branch"] = branch
            if base_ref != "HEAD":
                task.metadata["preserved_base_branch"] = str(base_ref)
            else:
                task.metadata.pop("preserved_base_branch", None)
            task.metadata["preserved_at"] = now_iso()

            try:
                base_sha_result = subprocess.run(
                    ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
                    cwd=svc.container.project_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                base_sha = base_sha_result.stdout.strip() if base_sha_result.returncode == 0 else ""
                if base_sha:
                    task.metadata["preserved_base_sha"] = base_sha
                else:
                    task.metadata.pop("preserved_base_sha", None)
            except Exception:
                task.metadata.pop("preserved_base_sha", None)

            try:
                head_sha_result = subprocess.run(
                    ["git", "rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}"],
                    cwd=svc.container.project_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                head_sha = head_sha_result.stdout.strip() if head_sha_result.returncode == 0 else ""
                if head_sha:
                    task.metadata["preserved_head_sha"] = head_sha
                else:
                    task.metadata.pop("preserved_head_sha", None)
            except Exception:
                task.metadata.pop("preserved_head_sha", None)

            base_sha_for_merge = str(task.metadata.get("preserved_base_sha") or "").strip()
            head_sha_for_merge = str(task.metadata.get("preserved_head_sha") or "").strip()
            if base_sha_for_merge and head_sha_for_merge:
                try:
                    merge_base_result = subprocess.run(
                        ["git", "merge-base", base_sha_for_merge, head_sha_for_merge],
                        cwd=svc.container.project_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )
                    merge_base_sha = merge_base_result.stdout.strip() if merge_base_result.returncode == 0 else ""
                    if merge_base_sha:
                        task.metadata["preserved_merge_base_sha"] = merge_base_sha
                    else:
                        task.metadata.pop("preserved_merge_base_sha", None)
                except Exception:
                    task.metadata.pop("preserved_merge_base_sha", None)
            else:
                task.metadata.pop("preserved_merge_base_sha", None)

            svc.container.tasks.upsert(task)
            return {
                "status": "preserved",
                "reason_code": "ok",
                "commit_sha": commit_sha or None,
                "base_sha": str(task.metadata.get("preserved_base_sha") or "").strip() or None,
                "head_sha": str(task.metadata.get("preserved_head_sha") or "").strip() or None,
            }
        except Exception:
            logger.exception("Failed to preserve worktree work for task %s", task.id)
            return {
                "status": "failed",
                "reason_code": "exception",
                "commit_sha": None,
                "base_sha": None,
                "head_sha": None,
            }

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
