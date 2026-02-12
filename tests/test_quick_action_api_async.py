"""Tests for async quick action execution via the API."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from feature_prd_runner.server import create_app


def _poll_until(client: TestClient, quick_action_id: str, target_statuses: set[str], timeout: float = 5.0) -> dict:
    """Poll GET /api/v3/quick-actions/{id} until status is in target_statuses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/v3/quick-actions/{quick_action_id}")
        assert resp.status_code == 200
        qa = resp.json()["quick_action"]
        if qa["status"] in target_statuses:
            return qa
        time.sleep(0.1)
    raise TimeoutError(f"Quick action {quick_action_id} did not reach {target_statuses} within {timeout}s")


def test_post_returns_queued(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.post("/api/v3/quick-actions", json={"prompt": "explain something"})
        assert resp.status_code == 200
        qa = resp.json()["quick_action"]
        assert qa["status"] == "queued"


def test_completes_async(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.post("/api/v3/quick-actions", json={"prompt": "explain something"})
        qa_id = resp.json()["quick_action"]["id"]
        result = _poll_until(client, qa_id, {"completed", "failed"})
        assert result["status"] == "failed"  # no workers configured → graceful failure
        assert result["kind"] == "agent"


def test_shortcut_via_api(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.post("/api/v3/quick-actions", json={"prompt": "git status"})
        qa_id = resp.json()["quick_action"]["id"]
        result = _poll_until(client, qa_id, {"completed", "failed"})
        assert result["kind"] == "shortcut"
        assert result["command"] == "git status"


def test_promotion_still_works_immediately(tmp_path: Path) -> None:
    """Promote a quick action right after POST, before execution completes."""
    app = create_app(project_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.post("/api/v3/quick-actions", json={"prompt": "do something"})
        qa_id = resp.json()["quick_action"]["id"]
        promote = client.post(f"/api/v3/quick-actions/{qa_id}/promote", json={})
        assert promote.status_code == 200
        assert promote.json()["already_promoted"] is False
