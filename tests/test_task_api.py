"""Tests for the V2 Task API endpoints."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from httpx import AsyncClient, ASGITransport

from feature_prd_runner.server.api import create_app


@pytest.fixture
def app(tmp_path: Path):
    """Create a test app with a temp project directory."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    state_dir = project_dir / ".prd_runner"
    state_dir.mkdir()
    return create_app(project_dir=project_dir, enable_cors=False)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
class TestTaskCRUD:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v2/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks"] == []
        assert data["total"] == 0

    async def test_create_and_get(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v2/tasks", json={
            "title": "Implement auth",
            "description": "OAuth2 login",
            "task_type": "feature",
            "priority": "P1",
            "labels": ["auth"],
        })
        assert resp.status_code == 201
        task = resp.json()["task"]
        assert task["title"] == "Implement auth"
        assert task["task_type"] == "feature"
        assert task["priority"] == "P1"
        task_id = task["id"]

        # Get it back
        resp = await client.get(f"/api/v2/tasks/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["task"]["title"] == "Implement auth"

    async def test_get_nonexistent(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v2/tasks/nope")
        assert resp.status_code == 404

    async def test_list_with_filters(self, client: AsyncClient) -> None:
        await client.post("/api/v2/tasks", json={"title": "Bug", "task_type": "bug"})
        await client.post("/api/v2/tasks", json={"title": "Feature", "task_type": "feature"})

        resp = await client.get("/api/v2/tasks?task_type=bug")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        assert resp.json()["tasks"][0]["title"] == "Bug"

    async def test_update(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v2/tasks", json={"title": "Old"})
        task_id = resp.json()["task"]["id"]

        resp = await client.patch(f"/api/v2/tasks/{task_id}", json={"title": "New", "priority": "P0"})
        assert resp.status_code == 200
        assert resp.json()["task"]["title"] == "New"
        assert resp.json()["task"]["priority"] == "P0"

    async def test_delete(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v2/tasks", json={"title": "To delete"})
        task_id = resp.json()["task"]["id"]

        resp = await client.delete(f"/api/v2/tasks/{task_id}")
        assert resp.status_code == 200

        resp = await client.get(f"/api/v2/tasks/{task_id}")
        assert resp.json()["task"]["status"] == "cancelled"

    async def test_bulk_create(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v2/tasks/bulk", json={
            "tasks": [
                {"title": "Task 1"},
                {"title": "Task 2"},
                {"title": "Task 3"},
            ]
        })
        assert resp.status_code == 201
        assert resp.json()["total"] == 3


@pytest.mark.anyio
class TestTransitionsAndAssignment:
    async def test_transition(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v2/tasks", json={"title": "Test"})
        task_id = resp.json()["task"]["id"]

        # backlog → ready
        resp = await client.post(f"/api/v2/tasks/{task_id}/transition", json={"status": "ready"})
        assert resp.status_code == 200
        assert resp.json()["task"]["status"] == "ready"

    async def test_invalid_transition(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v2/tasks", json={"title": "Test"})
        task_id = resp.json()["task"]["id"]

        # backlog → done should fail
        resp = await client.post(f"/api/v2/tasks/{task_id}/transition", json={"status": "done"})
        assert resp.status_code == 400

    async def test_assign_unassign(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v2/tasks", json={"title": "Test"})
        task_id = resp.json()["task"]["id"]

        resp = await client.post(f"/api/v2/tasks/{task_id}/assign", json={
            "assignee": "agent-impl-1",
            "assignee_type": "agent",
        })
        assert resp.status_code == 200
        assert resp.json()["task"]["assignee"] == "agent-impl-1"

        resp = await client.post(f"/api/v2/tasks/{task_id}/unassign")
        assert resp.status_code == 200
        assert resp.json()["task"]["assignee"] is None


@pytest.mark.anyio
class TestDependencies:
    async def test_add_and_get_dependency(self, client: AsyncClient) -> None:
        r1 = await client.post("/api/v2/tasks", json={"title": "First"})
        r2 = await client.post("/api/v2/tasks", json={"title": "Second"})
        id1 = r1.json()["task"]["id"]
        id2 = r2.json()["task"]["id"]

        resp = await client.post(f"/api/v2/tasks/{id2}/dependencies", json={"depends_on": id1})
        assert resp.status_code == 200

        resp = await client.get(f"/api/v2/tasks/{id2}/dependencies")
        assert resp.status_code == 200
        assert id1 in resp.json()["graph"].get(id2, [])

    async def test_cycle_returns_400(self, client: AsyncClient) -> None:
        r1 = await client.post("/api/v2/tasks", json={"title": "A"})
        r2 = await client.post("/api/v2/tasks", json={"title": "B"})
        id1 = r1.json()["task"]["id"]
        id2 = r2.json()["task"]["id"]

        await client.post(f"/api/v2/tasks/{id2}/dependencies", json={"depends_on": id1})
        resp = await client.post(f"/api/v2/tasks/{id1}/dependencies", json={"depends_on": id2})
        assert resp.status_code == 400

    async def test_remove_dependency(self, client: AsyncClient) -> None:
        r1 = await client.post("/api/v2/tasks", json={"title": "A"})
        r2 = await client.post("/api/v2/tasks", json={"title": "B"})
        id1 = r1.json()["task"]["id"]
        id2 = r2.json()["task"]["id"]

        await client.post(f"/api/v2/tasks/{id2}/dependencies", json={"depends_on": id1})
        resp = await client.delete(f"/api/v2/tasks/{id2}/dependencies/{id1}")
        assert resp.status_code == 200


@pytest.mark.anyio
class TestBoardAndOrdering:
    async def test_board_view(self, client: AsyncClient) -> None:
        await client.post("/api/v2/tasks", json={"title": "Backlog"})
        r2 = await client.post("/api/v2/tasks", json={"title": "Ready"})
        await client.post(f"/api/v2/tasks/{r2.json()['task']['id']}/transition", json={"status": "ready"})

        resp = await client.get("/api/v2/tasks/board")
        assert resp.status_code == 200
        cols = resp.json()["columns"]
        assert len(cols["backlog"]) == 1
        assert len(cols["ready"]) == 1

    async def test_execution_order(self, client: AsyncClient) -> None:
        r1 = await client.post("/api/v2/tasks", json={"title": "A"})
        r2 = await client.post("/api/v2/tasks", json={"title": "B"})
        id1 = r1.json()["task"]["id"]
        id2 = r2.json()["task"]["id"]
        await client.post(f"/api/v2/tasks/{id2}/dependencies", json={"depends_on": id1})

        resp = await client.get("/api/v2/tasks/execution-order")
        assert resp.status_code == 200
        batches = resp.json()["batches"]
        assert len(batches) >= 1

    async def test_reorder(self, client: AsyncClient) -> None:
        r1 = await client.post("/api/v2/tasks", json={"title": "A"})
        r2 = await client.post("/api/v2/tasks", json={"title": "B"})
        id1 = r1.json()["task"]["id"]
        id2 = r2.json()["task"]["id"]

        resp = await client.post("/api/v2/tasks/reorder", json={"task_ids": [id2, id1]})
        assert resp.status_code == 200

    async def test_ready_tasks(self, client: AsyncClient) -> None:
        r1 = await client.post("/api/v2/tasks", json={"title": "Ready"})
        await client.post(f"/api/v2/tasks/{r1.json()['task']['id']}/transition", json={"status": "ready"})

        resp = await client.get("/api/v2/tasks/ready")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1
