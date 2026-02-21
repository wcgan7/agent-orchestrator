"""Repository interfaces for runtime persistence abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..domain.models import AgentRecord, PlanRefineJob, PlanRevision, ReviewCycle, RunRecord, Task, TerminalSession


class TaskRepository(ABC):
    """Persistence contract for orchestrator task records."""
    @abstractmethod
    def list(self) -> List[Task]:
        """List every persisted task record.

        Returns:
            List[Task]: All task records currently stored for the project.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, task_id: str) -> Optional[Task]:
        """Fetch a task by id, or ``None`` when no record exists.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            Optional[Task]: Requested value when available; otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, task: Task) -> Task:
        """Create or update a task record.

        Args:
            task (Task): Task model to persist.

        Returns:
            Task: Persisted task record after the write operation.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """Delete a task by id and return whether anything was removed.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            bool: `True` when the operation succeeds, otherwise `False`.
        """
        raise NotImplementedError

    @abstractmethod
    def claim_next_runnable(self, *, max_in_progress: int) -> Optional[Task]:
        """Atomically claim and mark the next runnable task as in-progress.

        Args:
            max_in_progress (int): Upper bound for concurrently in-progress tasks.

        Returns:
            Optional[Task]: Newly claimed task, or `None` when no runnable task is available.
        """
        raise NotImplementedError


class RunRepository(ABC):
    """Persistence contract for execution run records."""
    @abstractmethod
    def list(self) -> List[RunRecord]:
        """List every persisted run record.

        Returns:
            List[RunRecord]: All run records currently stored for the project.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> Optional[RunRecord]:
        """Fetch a run by id, or ``None`` when no record exists.

        Args:
            run_id (str): Identifier for the target run.

        Returns:
            Optional[RunRecord]: Requested value when available; otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, run: RunRecord) -> RunRecord:
        """Create or update a run record.

        Args:
            run (RunRecord): Run model to persist.

        Returns:
            RunRecord: Persisted run record after the write operation.
        """
        raise NotImplementedError


class AgentRepository(ABC):
    """Persistence contract for agent presence and lifecycle records."""
    @abstractmethod
    def list(self) -> List[AgentRecord]:
        """List every persisted agent record.

        Returns:
            List[AgentRecord]: All agent lifecycle records currently stored.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, agent_id: str) -> Optional[AgentRecord]:
        """Fetch an agent by id, or ``None`` when no record exists.

        Args:
            agent_id (str): Identifier for the target agent.

        Returns:
            Optional[AgentRecord]: Requested value when available; otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, agent: AgentRecord) -> AgentRecord:
        """Create or update an agent record.

        Args:
            agent (AgentRecord): Agent record to persist.

        Returns:
            AgentRecord: Persisted agent record after the write operation.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, agent_id: str) -> bool:
        """Delete an agent by id and return whether one existed.

        Args:
            agent_id (str): Identifier for the target agent.

        Returns:
            bool: `True` when the operation succeeds, otherwise `False`.
        """
        raise NotImplementedError


class TerminalSessionRepository(ABC):
    """Persistence contract for terminal session metadata."""
    @abstractmethod
    def list(self) -> List[TerminalSession]:
        """List every persisted terminal session.

        Returns:
            List[TerminalSession]: All terminal session records for the project.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> Optional[TerminalSession]:
        """Fetch a terminal session by id, or ``None`` when no record exists.

        Args:
            session_id (str): Terminal session identifier.

        Returns:
            Optional[TerminalSession]: Matching terminal session when present; otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, session: TerminalSession) -> TerminalSession:
        """Create or update a terminal session record.

        Args:
            session (TerminalSession): Terminal session metadata to persist.

        Returns:
            TerminalSession: Persisted terminal session record after the write operation.
        """
        raise NotImplementedError


class ReviewRepository(ABC):
    """Persistence contract for review cycles and findings."""
    @abstractmethod
    def list(self) -> List[ReviewCycle]:
        """List every persisted review cycle.

        Returns:
            List[ReviewCycle]: All review cycles currently stored.
        """
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[ReviewCycle]:
        """List review cycles that belong to a specific task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            List[ReviewCycle]: Review cycles associated with ``task_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def append(self, cycle: ReviewCycle) -> ReviewCycle:
        """Append a new review cycle entry.

        Args:
            cycle (ReviewCycle): Completed review cycle to append.

        Returns:
            ReviewCycle: Persisted review cycle record.
        """
        raise NotImplementedError


class EventRepository(ABC):
    """Persistence contract for runtime event streams."""
    @abstractmethod
    def append(self, *, channel: str, event_type: str, entity_id: str, payload: dict[str, Any], project_id: str) -> dict[str, Any]:
        """Append an event envelope and return the persisted record.

        Args:
            channel (str): Event channel name (for example ``task`` or ``terminal``).
            event_type (str): Event type label within the channel namespace.
            entity_id (str): Identifier for the related entity.
            payload (dict[str, Any]): JSON-serializable event payload.
            project_id (str): Identifier for the related project.

        Returns:
            dict[str, Any]: Persisted event envelope including id and timestamp metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def list_recent(self, limit: int = 100) -> List[dict[str, Any]]:
        """List the most recent events, capped at ``limit`` records.

        Args:
            limit (int): Maximum number of newest event records to return.

        Returns:
            List[dict[str, Any]]: Most recent event envelopes, newest-last by storage order.
        """
        raise NotImplementedError


class PlanRevisionRepository(ABC):
    """Persistence contract for plan revision history."""
    @abstractmethod
    def list(self) -> List[PlanRevision]:
        """List every persisted plan revision.

        Returns:
            List[PlanRevision]: All persisted plan revisions.
        """
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[PlanRevision]:
        """List plan revisions that belong to a specific task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            List[PlanRevision]: Plan revisions associated with ``task_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, revision_id: str) -> Optional[PlanRevision]:
        """Fetch a plan revision by id, or ``None`` when no record exists.

        Args:
            revision_id (str): Plan revision identifier.

        Returns:
            Optional[PlanRevision]: Matching plan revision when present; otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, revision: PlanRevision) -> PlanRevision:
        """Create or update a plan revision.

        Args:
            revision (PlanRevision): Plan revision record to persist.

        Returns:
            PlanRevision: Persisted plan revision after the write operation.
        """
        raise NotImplementedError


class PlanRefineJobRepository(ABC):
    """Persistence contract for asynchronous plan-refine jobs."""
    @abstractmethod
    def list(self) -> List[PlanRefineJob]:
        """List every persisted plan-refine job.

        Returns:
            List[PlanRefineJob]: All persisted plan-refine jobs.
        """
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[PlanRefineJob]:
        """List plan-refine jobs that belong to a specific task.

        Args:
            task_id (str): Identifier for the target task.

        Returns:
            List[PlanRefineJob]: Plan-refine jobs associated with ``task_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, job_id: str) -> Optional[PlanRefineJob]:
        """Fetch a plan-refine job by id, or ``None`` when no record exists.

        Args:
            job_id (str): Plan-refine job identifier.

        Returns:
            Optional[PlanRefineJob]: Matching job when present; otherwise `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, job: PlanRefineJob) -> PlanRefineJob:
        """Create or update a refine job.

        Args:
            job (PlanRefineJob): Plan-refine job record to persist.

        Returns:
            PlanRefineJob: Persisted plan-refine job after the write operation.
        """
        raise NotImplementedError
