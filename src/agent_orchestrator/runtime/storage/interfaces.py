from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..domain.models import AgentRecord, PlanRefineJob, PlanRevision, QuickActionRun, ReviewCycle, RunRecord, Task


class TaskRepository(ABC):
    @abstractmethod
    def list(self) -> list[Task]:
        raise NotImplementedError

    @abstractmethod
    def get(self, task_id: str) -> Optional[Task]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, task: Task) -> Task:
        raise NotImplementedError

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def claim_next_runnable(self, *, max_in_progress: int) -> Optional[Task]:
        raise NotImplementedError


class RunRepository(ABC):
    @abstractmethod
    def list(self) -> list[RunRecord]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, run: RunRecord) -> RunRecord:
        raise NotImplementedError


class AgentRepository(ABC):
    @abstractmethod
    def list(self) -> list[AgentRecord]:
        raise NotImplementedError

    @abstractmethod
    def get(self, agent_id: str) -> Optional[AgentRecord]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, agent: AgentRecord) -> AgentRecord:
        raise NotImplementedError

    @abstractmethod
    def delete(self, agent_id: str) -> bool:
        raise NotImplementedError


class QuickActionRepository(ABC):
    @abstractmethod
    def list(self) -> list[QuickActionRun]:
        raise NotImplementedError

    @abstractmethod
    def get(self, quick_action_id: str) -> Optional[QuickActionRun]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, quick_action: QuickActionRun) -> QuickActionRun:
        raise NotImplementedError


class ReviewRepository(ABC):
    @abstractmethod
    def list(self) -> list[ReviewCycle]:
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> list[ReviewCycle]:
        raise NotImplementedError

    @abstractmethod
    def append(self, cycle: ReviewCycle) -> ReviewCycle:
        raise NotImplementedError


class EventRepository(ABC):
    @abstractmethod
    def append(self, *, channel: str, event_type: str, entity_id: str, payload: dict[str, Any], project_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        raise NotImplementedError


class PlanRevisionRepository(ABC):
    @abstractmethod
    def list(self) -> list[PlanRevision]:
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> list[PlanRevision]:
        raise NotImplementedError

    @abstractmethod
    def get(self, revision_id: str) -> Optional[PlanRevision]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, revision: PlanRevision) -> PlanRevision:
        raise NotImplementedError


class PlanRefineJobRepository(ABC):
    @abstractmethod
    def list(self) -> list[PlanRefineJob]:
        raise NotImplementedError

    @abstractmethod
    def for_task(self, task_id: str) -> list[PlanRefineJob]:
        raise NotImplementedError

    @abstractmethod
    def get(self, job_id: str) -> Optional[PlanRefineJob]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, job: PlanRefineJob) -> PlanRefineJob:
        raise NotImplementedError
