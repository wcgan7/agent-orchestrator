"""Tests for quick-run behavior and promotion contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from feature_prd_runner.server.api import create_app


@pytest.fixture
def app(tmp_path: Path):
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / ".prd_runner").mkdir()
    return create_app(project_dir=project_dir, enable_cors=False)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
class TestQuickRunPromotionFlow:
    async def test_quick_run_does_not_create_task_until_promoted(self, client: AsyncClient) -> None:
        with patch("feature_prd_runner.custom_execution.execute_custom_prompt", return_value=(True, None)):
            quick_resp = await client.post(
                "/api/v2/quick-runs",
                json={
                    "prompt": "Run a one-off cleanup command",
                    "override_agents": False,
                },
            )
        assert quick_resp.status_code == 200
        quick_payload = quick_resp.json()
        quick_run = quick_payload["quick_run"]
        quick_run_id = quick_run["id"]
        assert quick_payload["success"] is True
        assert quick_run["promoted_task_id"] is None

        tasks_resp = await client.get("/api/v2/tasks")
        assert tasks_resp.status_code == 200
        assert tasks_resp.json()["total"] == 0

        quick_get_resp = await client.get(f"/api/v2/quick-runs/{quick_run_id}")
        assert quick_get_resp.status_code == 200
        assert quick_get_resp.json()["promoted_task_id"] is None

        events_resp = await client.get(f"/api/v2/quick-runs/{quick_run_id}/events")
        assert events_resp.status_code == 200
        event_types = [evt["type"] for evt in events_resp.json()["events"]]
        assert "quick_run.started" in event_types
        assert "quick_run.completed" in event_types

    async def test_promote_quick_run_creates_task(self, client: AsyncClient) -> None:
        with patch("feature_prd_runner.custom_execution.execute_custom_prompt", return_value=(True, None)):
            quick_resp = await client.post(
                "/api/v2/quick-runs",
                json={
                    "prompt": "Add retry handling to upload endpoint",
                    "override_agents": False,
                },
            )
        quick_run_id = quick_resp.json()["quick_run"]["id"]

        promote_resp = await client.post(
            f"/api/v2/quick-runs/{quick_run_id}/promote",
            json={"title": "Promoted upload retry task", "task_type": "feature", "priority": "P1"},
        )
        assert promote_resp.status_code == 200
        promoted_task_id = promote_resp.json()["task_id"]
        assert isinstance(promoted_task_id, str) and promoted_task_id

        quick_get_resp = await client.get(f"/api/v2/quick-runs/{quick_run_id}")
        assert quick_get_resp.status_code == 200
        assert quick_get_resp.json()["promoted_task_id"] == promoted_task_id

        tasks_resp = await client.get("/api/v2/tasks")
        assert tasks_resp.status_code == 200
        assert tasks_resp.json()["total"] == 1
        task = tasks_resp.json()["tasks"][0]
        assert task["id"] == promoted_task_id
        assert task["source"] == "promoted_quick_action"

        recent_events_resp = await client.get("/api/v2/quick-runs/events/recent")
        assert recent_events_resp.status_code == 200
        recent_event_types = [evt["type"] for evt in recent_events_resp.json()["events"]]
        assert "quick_run.promoted" in recent_event_types
