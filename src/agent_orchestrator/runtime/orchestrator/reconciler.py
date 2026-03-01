"""Periodic/manual reconciler wrapper around runtime invariants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .invariants import apply_runtime_invariants

if TYPE_CHECKING:
    from .service import OrchestratorService


class OrchestratorReconciler:
    """Apply runtime invariant checks and deterministic self-heal actions."""

    def __init__(self, service: "OrchestratorService") -> None:
        self._service = service

    def run_once(self, *, source: Literal["startup", "automatic", "manual"]) -> dict[str, Any]:
        """Run one reconciliation pass and return summary details."""
        with self._service._futures_lock:
            active_ids = {
                task_id
                for task_id, future in self._service._futures.items()
                if not future.done()
            }
        return apply_runtime_invariants(
            self._service,
            active_future_task_ids=active_ids,
            source=source,
        )
