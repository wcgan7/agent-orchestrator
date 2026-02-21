"""Repository interfaces for runtime persistence abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..domain.models import AgentRecord, PlanRefineJob, PlanRevision, ReviewCycle, RunRecord, Task, TerminalSession


class TaskRepository(ABC):
    """Persistence contract for orchestrator task records."""
    @abstractmethod
    def list(self) -> List[Task]:
        """List every persisted task record."""
        raise NotImplementedError

    @abstractmethod
    def get(self, task_id: str) -> Optional[Task]:
        """Fetch a task by id, or ``None`` when no record exists."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, task: Task) -> Task:
        """Create or update a task record."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """Delete a task by id and return whether anything was removed."""
        raise NotImplementedError

    @abstractmethod
    def claim_next_runnable(self, *, max_in_progress: int) -> Optional[Task]:
        """Atomically claim and mark the next runnable task as in-progress."""
        raise NotImplementedError


class RunRepository(ABC):
    """Persistence contract for execution run records."""
    @abstractmethod
    def list(self) -> List[RunRecord]:
        """List every persisted run record."""
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> Optional[RunRecord]:
        """Fetch a run by id, or ``None`` when no record exists."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, run: RunRecord) -> RunRecord:
        """Create or update a run record."""
        raise NotImplementedError


class AgentRepository(ABC):
    """Persistence contract for agent presence and lifecycle records."""
    @abstractmethod
    def list(self) -> List[AgentRecord]:
        """List every persisted agent record."""
        raise NotImplementedError

    @abstractmethod
    def get(self, agent_id: str) -> Optional[AgentRecord]:
        """Fetch an agent by id, or ``None`` when no record exists."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, agent: AgentRecord) -> AgentRecord:
        """Create or update an agent record."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, agent_id: str) -> bool:
        """Delete an agent by id and return whether one existed."""
        raise NotImplementedError


class TerminalSessionRepository(ABC):
    """Persistence contract for terminal session metadata."""
    @abstractmethod
    def list(self) -> List[TerminalSession]:
        """List every persisted terminal session."""
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> Optional[TerminalSession]:
        """Fetch a terminal session by id, or ``None`` when no record exists."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, session: TerminalSession) -> TerminalSession:
        """Create or update a terminal session record."""
        raise NotImplementedError


class ReviewRepository(ABC):
    """Persistence contract for review cycles and findings."""
    @abstractmethod
    def list(self) -> List[ReviewCycle]:
        """List every persisted review cycle."""
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[ReviewCycle]:
        """List review cycles that belong to a specific task."""
        raise NotImplementedError

    @abstractmethod
    def append(self, cycle: ReviewCycle) -> ReviewCycle:
        """Append a new review cycle entry."""
        raise NotImplementedError


class EventRepository(ABC):
    """Persistence contract for runtime event streams."""
    @abstractmethod
    def append(self, *, channel: str, event_type: str, entity_id: str, payload: dict[str, Any], project_id: str) -> dict[str, Any]:
        """Append an event envelope and return the persisted record."""
        raise NotImplementedError

    @abstractmethod
    def list_recent(self, limit: int = 100) -> List[dict[str, Any]]:
        """List the most recent events, capped at ``limit`` records."""
        raise NotImplementedError


class PlanRevisionRepository(ABC):
    """Persistence contract for plan revision history."""
    @abstractmethod
    def list(self) -> List[PlanRevision]:
        """List every persisted plan revision."""
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[PlanRevision]:
        """List plan revisions that belong to a specific task."""
        raise NotImplementedError

    @abstractmethod
    def get(self, revision_id: str) -> Optional[PlanRevision]:
        """Fetch a plan revision by id, or ``None`` when no record exists."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, revision: PlanRevision) -> PlanRevision:
        """Create or update a plan revision."""
        raise NotImplementedError


class PlanRefineJobRepository(ABC):
    """Persistence contract for asynchronous plan-refine jobs."""
    @abstractmethod
    def list(self) -> List[PlanRefineJob]:
        """List every persisted plan-refine job."""
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[PlanRefineJob]:
        """List plan-refine jobs that belong to a specific task."""
        raise NotImplementedError

    @abstractmethod
    def get(self, job_id: str) -> Optional[PlanRefineJob]:
        """Fetch a plan-refine job by id, or ``None`` when no record exists."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, job: PlanRefineJob) -> PlanRefineJob:
        """Create or update a refine job."""
        raise NotImplementedError
