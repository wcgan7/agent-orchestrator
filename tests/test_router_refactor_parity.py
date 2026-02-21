from __future__ import annotations

import re
from pathlib import Path

from fastapi.routing import APIRoute

from agent_orchestrator.runtime.api.router import create_router


EXPECTED_API_ROUTE_TABLE: set[tuple[str, str]] = {
    ("DELETE", "/api/agents/{agent_id}"),
    ("DELETE", "/api/projects/pinned/{project_id}"),
    ("DELETE", "/api/tasks/{task_id}/dependencies/{dep_id}"),
    ("GET", "/api/agents"),
    ("GET", "/api/agents/types"),
    ("GET", "/api/collaboration/comments/{task_id}"),
    ("GET", "/api/collaboration/feedback/{task_id}"),
    ("GET", "/api/collaboration/modes"),
    ("GET", "/api/collaboration/presence"),
    ("GET", "/api/collaboration/timeline/{task_id}"),
    ("GET", "/api/import/{job_id}"),
    ("GET", "/api/metrics"),
    ("GET", "/api/orchestrator/status"),
    ("GET", "/api/phases"),
    ("GET", "/api/projects"),
    ("GET", "/api/projects/browse"),
    ("GET", "/api/projects/pinned"),
    ("GET", "/api/review-queue"),
    ("GET", "/api/settings"),
    ("GET", "/api/tasks"),
    ("GET", "/api/tasks/board"),
    ("GET", "/api/tasks/execution-order"),
    ("GET", "/api/tasks/{task_id}"),
    ("GET", "/api/tasks/{task_id}/diff"),
    ("GET", "/api/tasks/{task_id}/logs"),
    ("GET", "/api/tasks/{task_id}/plan"),
    ("GET", "/api/tasks/{task_id}/plan/jobs"),
    ("GET", "/api/tasks/{task_id}/plan/jobs/{job_id}"),
    ("GET", "/api/tasks/{task_id}/workdoc"),
    ("GET", "/api/terminal/session"),
    ("GET", "/api/terminal/session/{session_id}/logs"),
    ("GET", "/api/workers/health"),
    ("GET", "/api/workers/routing"),
    ("PATCH", "/api/settings"),
    ("PATCH", "/api/tasks/{task_id}"),
    ("POST", "/api/agents/spawn"),
    ("POST", "/api/agents/{agent_id}/pause"),
    ("POST", "/api/agents/{agent_id}/remove"),
    ("POST", "/api/agents/{agent_id}/resume"),
    ("POST", "/api/agents/{agent_id}/terminate"),
    ("POST", "/api/collaboration/comments"),
    ("POST", "/api/collaboration/comments/{comment_id}/resolve"),
    ("POST", "/api/collaboration/feedback"),
    ("POST", "/api/collaboration/feedback/{feedback_id}/dismiss"),
    ("POST", "/api/import/prd/commit"),
    ("POST", "/api/import/prd/preview"),
    ("POST", "/api/orchestrator/control"),
    ("POST", "/api/projects/pinned"),
    ("POST", "/api/review/{task_id}/approve"),
    ("POST", "/api/review/{task_id}/request-changes"),
    ("POST", "/api/tasks"),
    ("POST", "/api/tasks/analyze-dependencies"),
    ("POST", "/api/tasks/classify-pipeline"),
    ("POST", "/api/tasks/{task_id}/approve-gate"),
    ("POST", "/api/tasks/{task_id}/cancel"),
    ("POST", "/api/tasks/{task_id}/dependencies"),
    ("POST", "/api/tasks/{task_id}/generate-tasks"),
    ("POST", "/api/tasks/{task_id}/plan/commit"),
    ("POST", "/api/tasks/{task_id}/plan/refine"),
    ("POST", "/api/tasks/{task_id}/plan/revisions"),
    ("POST", "/api/tasks/{task_id}/reset-dep-analysis"),
    ("POST", "/api/tasks/{task_id}/retry"),
    ("POST", "/api/tasks/{task_id}/run"),
    ("POST", "/api/tasks/{task_id}/transition"),
    ("POST", "/api/terminal/session"),
    ("POST", "/api/terminal/session/{session_id}/input"),
    ("POST", "/api/terminal/session/{session_id}/resize"),
    ("POST", "/api/terminal/session/{session_id}/stop"),
}


def _runtime_route_table() -> set[tuple[str, str]]:
    router = create_router(lambda _project_dir: None, lambda _project_dir: None, {})
    return {
        (method, route.path)
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
        if method not in {"HEAD", "OPTIONS"}
    }


def test_runtime_route_table_parity() -> None:
    assert _runtime_route_table() == EXPECTED_API_ROUTE_TABLE


def test_api_reference_matches_runtime_routes() -> None:
    docs = Path("docs/API_REFERENCE.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"### `([A-Z]+) (/api[^`]+)`", docs))
    assert documented == EXPECTED_API_ROUTE_TABLE
