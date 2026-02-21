"""Event bus wrapper that persists and broadcasts runtime events."""

from __future__ import annotations

from typing import Any

from ..storage.interfaces import EventRepository
from .ws import hub


class EventBus:
    """Persist runtime events and fan them out to websocket subscribers."""
    def __init__(self, repo: EventRepository, project_id: str) -> None:
        self._repo = repo
        self._project_id = project_id

    def emit(self, *, channel: str, event_type: str, entity_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append an event to storage and publish it to connected clients."""
        event = self._repo.append(
            channel=channel,
            event_type=event_type,
            entity_id=entity_id,
            payload=payload,
            project_id=self._project_id,
        )
        hub.publish_sync(event)
        return event
