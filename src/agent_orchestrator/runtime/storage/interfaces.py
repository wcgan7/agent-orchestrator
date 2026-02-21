"""Repository interfaces for runtime persistence abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..domain.models import AgentRecord, PlanRefineJob, PlanRevision, ReviewCycle, RunRecord, Task, TerminalSession


class TaskRepository(ABC):
    """Represents TaskRepository."""
    @abstractmethod
    def list(self) -> List[Task]:
        """Return list."""
        raise NotImplementedError

    @abstractmethod
    def get(self, task_id: str) -> Optional[Task]:
        """Return get."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, task: Task) -> Task:
        """Return upsert."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """Return delete."""
        raise NotImplementedError

    @abstractmethod
    def claim_next_runnable(self, *, max_in_progress: int) -> Optional[Task]:
        """Return claim next runnable."""
        raise NotImplementedError


class RunRepository(ABC):
    """Represents RunRepository."""
    @abstractmethod
    def list(self) -> List[RunRecord]:
        """Return list."""
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> Optional[RunRecord]:
        """Return get."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, run: RunRecord) -> RunRecord:
        """Return upsert."""
        raise NotImplementedError


class AgentRepository(ABC):
    """Represents AgentRepository."""
    @abstractmethod
    def list(self) -> List[AgentRecord]:
        """Return list."""
        raise NotImplementedError

    @abstractmethod
    def get(self, agent_id: str) -> Optional[AgentRecord]:
        """Return get."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, agent: AgentRecord) -> AgentRecord:
        """Return upsert."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, agent_id: str) -> bool:
        """Return delete."""
        raise NotImplementedError


class TerminalSessionRepository(ABC):
    """Represents TerminalSessionRepository."""
    @abstractmethod
    def list(self) -> List[TerminalSession]:
        """Return list."""
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> Optional[TerminalSession]:
        """Return get."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, session: TerminalSession) -> TerminalSession:
        """Return upsert."""
        raise NotImplementedError


class ReviewRepository(ABC):
    """Represents ReviewRepository."""
    @abstractmethod
    def list(self) -> List[ReviewCycle]:
        """Return list."""
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[ReviewCycle]:
        """Return for task."""
        raise NotImplementedError

    @abstractmethod
    def append(self, cycle: ReviewCycle) -> ReviewCycle:
        """Return append."""
        raise NotImplementedError


class EventRepository(ABC):
    """Represents EventRepository."""
    @abstractmethod
    def append(self, *, channel: str, event_type: str, entity_id: str, payload: dict[str, Any], project_id: str) -> dict[str, Any]:
        """Return append."""
        raise NotImplementedError

    @abstractmethod
    def list_recent(self, limit: int = 100) -> List[dict[str, Any]]:
        """Return list recent."""
        raise NotImplementedError


class PlanRevisionRepository(ABC):
    """Represents PlanRevisionRepository."""
    @abstractmethod
    def list(self) -> List[PlanRevision]:
        """Return list."""
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[PlanRevision]:
        """Return for task."""
        raise NotImplementedError

    @abstractmethod
    def get(self, revision_id: str) -> Optional[PlanRevision]:
        """Return get."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, revision: PlanRevision) -> PlanRevision:
        """Return upsert."""
        raise NotImplementedError


class PlanRefineJobRepository(ABC):
    """Represents PlanRefineJobRepository."""
    @abstractmethod
    def list(self) -> List[PlanRefineJob]:
        """Return list."""
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> List[PlanRefineJob]:
        """Return for task."""
        raise NotImplementedError

    @abstractmethod
    def get(self, job_id: str) -> Optional[PlanRefineJob]:
        """Return get."""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, job: PlanRefineJob) -> PlanRefineJob:
        """Return upsert."""
        raise NotImplementedError
