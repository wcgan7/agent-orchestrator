from __future__ import annotations

from typing import Any

from ..storage.interfaces import EventRepository
from .ws import hub


class EventBus:
    def __init__(self, repo: EventRepository, project_id: str) -> None:
        self._repo = repo
        self._project_id = project_id

    def emit(self, *, channel: str, event_type: str, entity_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = self._repo.append(
            channel=channel,
            event_type=event_type,
            entity_id=entity_id,
            payload=payload,
            project_id=self._project_id,
        )
        hub.publish_sync(event)
        return event
