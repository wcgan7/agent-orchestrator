"""Shared dependency context for runtime API route registration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from ..events.bus import EventBus
from ..orchestrator.service import OrchestratorService
from ..storage.container import Container
from ..terminal.service import TerminalService


@dataclass(frozen=True)
class RouteDeps:
    """Route registration dependency bundle."""

    resolve_container: Callable[[Optional[str]], Container]
    resolve_orchestrator: Callable[[Optional[str]], OrchestratorService]
    job_store: dict[str, dict[str, Any]]
    ctx: Callable[[Optional[str]], tuple[Container, EventBus, OrchestratorService]]
    terminal_ctx: Callable[[Optional[str]], tuple[Container, EventBus, TerminalService]]
    load_feedback_records: Callable[[Container], list[dict[str, Any]]]
    save_feedback_records: Callable[[Container, list[dict[str, Any]]], None]
    load_comment_records: Callable[[Container], list[dict[str, Any]]]
    save_comment_records: Callable[[Container, list[dict[str, Any]]], None]
    prune_in_memory_jobs: Callable[[], None]
    upsert_import_job: Callable[[Container, dict[str, Any]], None]
    fetch_import_job: Callable[[Container, str], Optional[dict[str, Any]]]
