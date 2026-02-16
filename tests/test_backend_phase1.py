from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from agent_orchestrator.server.api import create_app
from agent_orchestrator.runtime.orchestrator import DefaultWorkerAdapter
from agent_orchestrator.runtime.domain.models import QuickActionRun, Task
from agent_orchestrator.runtime.storage.bootstrap import ensure_state_root
from agent_orchestrator.runtime.storage.container import Container


def test_cutover_archives_legacy_state(tmp_path: Path) -> None:
    legacy_root = tmp_path / ".agent_orchestrator"
    legacy_root.mkdir(parents=True)
    (legacy_root / "task_queue.yaml").write_text("tasks: []\n", encoding="utf-8")
    (legacy_root / "run_state.yaml").write_text("{}\n", encoding="utf-8")

    state_root = ensure_state_root(tmp_path)

    assert state_root == tmp_path / ".agent_orchestrator"
    assert (state_root / "config.yaml").exists()
    config = (state_root / "config.yaml").read_text(encoding="utf-8")
    assert "schema_version: 3" in config

    archives = sorted(tmp_path.glob(".agent_orchestrator_legacy_*"))
    assert len(archives) == 1
    assert (archives[0] / "task_queue.yaml").exists()


def test_cutover_archives_any_non_runtime_state(tmp_path: Path) -> None:
    legacy_root = tmp_path / ".agent_orchestrator"
    legacy_root.mkdir(parents=True)
    (legacy_root / "custom_legacy_blob.yaml").write_text("legacy: true\n", encoding="utf-8")

    ensure_state_root(tmp_path)

    archives = sorted(tmp_path.glob(".agent_orchestrator_legacy_*"))
    assert len(archives) == 1
    assert (archives[0] / "custom_legacy_blob.yaml").exists()


def test_cutover_forces_schema_version_3(tmp_path: Path) -> None:
    state_root = tmp_path / ".agent_orchestrator"
    state_root.mkdir(parents=True)
    (state_root / "config.yaml").write_text("schema_version: 2\n", encoding="utf-8")

    ensure_state_root(tmp_path)

    config_text = (tmp_path / ".agent_orchestrator" / "config.yaml").read_text(encoding="utf-8")
    assert "schema_version: 3" in config_text


def test_task_dependency_guard_blocks_ready_transition(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        blocker = client.post(
            "/api/tasks",
            json={"title": "Blocker", "status": "backlog", "approval_mode": "auto_approve", "metadata": {"scripted_findings": [[]]}},
        ).json()["task"]
        blocked = client.post("/api/tasks", json={"title": "Blocked", "status": "backlog"}).json()["task"]

        dep_resp = client.post(
            f"/api/tasks/{blocked['id']}/dependencies",
            json={"depends_on": blocker["id"]},
        )
        assert dep_resp.status_code == 200

        transition = client.post(
            f"/api/tasks/{blocked['id']}/transition",
            json={"status": "queued"},
        )
        assert transition.status_code == 400
        assert "Unresolved blocker" in transition.text

        # Queue and run the blocker so it completes
        client.post(f"/api/tasks/{blocker['id']}/transition", json={"status": "queued"})
        done = client.post(f"/api/tasks/{blocker['id']}/run")
        assert done.status_code == 200
        assert done.json()["task"]["status"] == "done"
        ok_transition = client.post(
            f"/api/tasks/{blocked['id']}/transition",
            json={"status": "queued"},
        )
        assert ok_transition.status_code == 200
        assert ok_transition.json()["task"]["status"] == "queued"


def test_patch_rejects_direct_status_changes(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"title": "Patch guarded"}).json()["task"]
        response = client.patch(f"/api/tasks/{task['id']}", json={"status": "done"})
        assert response.status_code == 400
        assert "cannot be changed via PATCH" in response.text


def test_review_actions_require_in_review_state(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"title": "Needs status guard"}).json()["task"]
        approve = client.post(f"/api/review/{task['id']}/approve", json={})
        assert approve.status_code == 400
        changes = client.post(f"/api/review/{task['id']}/request-changes", json={"guidance": "x"})
        assert changes.status_code == 400


def test_quick_action_promotion_is_singleton(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        qrun = client.post("/api/quick-actions", json={"prompt": "Do thing"}).json()["quick_action"]

        first = client.post(f"/api/quick-actions/{qrun['id']}/promote", json={})
        assert first.status_code == 200
        assert first.json()["already_promoted"] is False
        task_id = first.json()["task"]["id"]

        second = client.post(f"/api/quick-actions/{qrun['id']}/promote", json={})
        assert second.status_code == 200
        assert second.json()["already_promoted"] is True
        assert second.json()["task"]["id"] == task_id

        tasks = client.get("/api/tasks").json()["tasks"]
        assert [task["id"] for task in tasks].count(task_id) == 1


def test_quick_action_pending_limit_returns_429(tmp_path: Path) -> None:
    container = Container(tmp_path)
    for idx in range(32):
        container.quick_actions.upsert(QuickActionRun(prompt=f"Run {idx}", status="running"))

    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        response = client.post("/api/quick-actions", json={"prompt": "git status"})
        assert response.status_code == 429


def test_project_pin_requires_git_unless_override(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()

    with TestClient(app) as client:
        rejected = client.post("/api/projects/pinned", json={"path": str(plain_dir)})
        assert rejected.status_code == 400

        accepted = client.post(
            "/api/projects/pinned",
            json={"path": str(plain_dir), "allow_non_git": True},
        )
        assert accepted.status_code == 200
        listing = client.get("/api/projects/pinned").json()["items"]
        assert len(listing) == 1


def test_import_preview_commit_creates_dependency_chain(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        preview = client.post(
            "/api/import/prd/preview",
            json={"content": "- First\n- Second\n- Third", "default_priority": "P1"},
        )
        assert preview.status_code == 200
        job_id = preview.json()["job_id"]

        commit = client.post("/api/import/prd/commit", json={"job_id": job_id})
        assert commit.status_code == 200
        created_ids = commit.json()["created_task_ids"]
        assert len(created_ids) == 3

        tasks = {item["id"]: item for item in client.get("/api/tasks").json()["tasks"]}
        assert tasks[created_ids[0]]["blocked_by"] == []
        assert tasks[created_ids[1]]["blocked_by"] == [created_ids[0]]
        assert tasks[created_ids[2]]["blocked_by"] == [created_ids[1]]

        job = client.get(f"/api/import/{job_id}")
        assert job.status_code == 200
        assert job.json()["job"]["created_task_ids"] == created_ids


def test_import_job_persists_when_in_memory_cache_is_empty(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        preview = client.post(
            "/api/import/prd/preview",
            json={"content": "- Persist me"},
        )
        assert preview.status_code == 200
        job_id = preview.json()["job_id"]

        app.state.import_jobs.clear()
        loaded = client.get(f"/api/import/{job_id}")
        assert loaded.status_code == 200
        assert loaded.json()["job"]["id"] == job_id


def test_health_endpoints_available(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"


def test_agent_remove_supports_delete_and_post(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        first = client.post("/api/agents/spawn", json={"role": "general", "capacity": 1})
        assert first.status_code == 200
        first_id = first.json()["agent"]["id"]

        delete_resp = client.delete(f"/api/agents/{first_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["removed"] is True

        second = client.post("/api/agents/spawn", json={"role": "general", "capacity": 1})
        assert second.status_code == 200
        second_id = second.json()["agent"]["id"]

        post_resp = client.post(f"/api/agents/{second_id}/remove")
        assert post_resp.status_code == 200
        assert post_resp.json()["removed"] is True

        missing = client.delete("/api/agents/does-not-exist")
        assert missing.status_code == 404


def test_workers_health_and_routing_endpoints(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        health = client.get("/api/workers/health")
        assert health.status_code == 200
        providers = health.json()["providers"]
        names = {item["name"] for item in providers}
        assert "codex" in names
        assert "claude" in names
        assert "ollama" in names

        routing = client.get("/api/workers/routing")
        assert routing.status_code == 200
        payload = routing.json()
        assert payload["default"] == "codex"
        assert any(item["step"] == "implement" for item in payload["rows"])


def test_legacy_compat_endpoints_available(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"title": "Compat seed"}).json()["task"]

        metrics = client.get("/api/metrics")
        assert metrics.status_code == 200
        assert "phases_total" in metrics.json()

        phases = client.get("/api/phases")
        assert phases.status_code == 200
        assert isinstance(phases.json(), list)

        agent_types = client.get("/api/agents/types")
        assert agent_types.status_code == 200
        assert len(agent_types.json()["types"]) > 0

        modes = client.get("/api/collaboration/modes")
        assert modes.status_code == 200
        assert len(modes.json()["modes"]) > 0

        add_feedback = client.post(
            "/api/collaboration/feedback",
            json={"task_id": created["id"], "summary": "Need stricter checks"},
        )
        assert add_feedback.status_code == 200
        feedback_id = add_feedback.json()["feedback"]["id"]

        add_comment = client.post(
            "/api/collaboration/comments",
            json={"task_id": created["id"], "file_path": "main.py", "line_number": 1, "body": "Looks good"},
        )
        assert add_comment.status_code == 200
        comment_id = add_comment.json()["comment"]["id"]

        feedback_list = client.get(f"/api/collaboration/feedback/{created['id']}")
        assert feedback_list.status_code == 200
        assert any(item["id"] == feedback_id for item in feedback_list.json()["feedback"])

        comment_list = client.get(f"/api/collaboration/comments/{created['id']}")
        assert comment_list.status_code == 200
        assert any(item["id"] == comment_id for item in comment_list.json()["comments"])

        dismissed = client.post(f"/api/collaboration/feedback/{feedback_id}/dismiss")
        assert dismissed.status_code == 200
        assert dismissed.json()["feedback"]["status"] == "addressed"

        resolved = client.post(f"/api/collaboration/comments/{comment_id}/resolve")
        assert resolved.status_code == 200
        assert resolved.json()["comment"]["resolved"] is True

        timeline = client.get(f"/api/collaboration/timeline/{created['id']}")
        assert timeline.status_code == 200
        assert len(timeline.json()["events"]) >= 1
        types = {event["type"] for event in timeline.json()["events"]}
        assert "feedback" in types
        assert "comment" in types


def test_settings_endpoint_round_trip(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        baseline = client.get("/api/settings")
        assert baseline.status_code == 200
        assert baseline.json()["orchestrator"]["concurrency"] == 2
        assert baseline.json()["orchestrator"]["auto_deps"] is True
        assert baseline.json()["orchestrator"]["max_review_attempts"] == 10
        assert baseline.json()["agent_routing"]["default_role"] == "general"
        assert baseline.json()["defaults"]["quality_gate"]["high"] == 0
        assert baseline.json()["workers"]["default"] == "codex"
        assert baseline.json()["workers"]["heartbeat_seconds"] == 60
        assert baseline.json()["workers"]["heartbeat_grace_seconds"] == 240
        assert baseline.json()["workers"]["providers"]["codex"]["type"] == "codex"

        updated = client.patch(
            "/api/settings",
            json={
                "orchestrator": {"concurrency": 5, "auto_deps": False, "max_review_attempts": 4},
                "agent_routing": {
                    "default_role": "reviewer",
                    "task_type_roles": {"bug": "debugger"},
                    "role_provider_overrides": {"reviewer": "openai"},
                },
                "defaults": {"quality_gate": {"critical": 1, "high": 2, "medium": 3, "low": 4}},
                "workers": {
                    "default": "ollama-dev",
                    "default_model": "gpt-5-codex",
                    "heartbeat_seconds": 90,
                    "heartbeat_grace_seconds": 360,
                    "routing": {"plan": "codex", "implement": "ollama-dev"},
                    "providers": {
                        "codex": {
                            "type": "codex",
                            "command": "codex exec",
                            "model": "gpt-5-codex",
                            "reasoning_effort": "high",
                        },
                        "ollama-dev": {
                            "type": "ollama",
                            "endpoint": "http://localhost:11434",
                            "model": "llama3.1:8b",
                            "temperature": 0.2,
                            "num_ctx": 8192,
                        },
                        "claude": {
                            "type": "claude",
                            "command": "claude -p",
                            "model": "sonnet",
                            "reasoning_effort": "medium",
                        },
                    },
                },
            },
        )
        assert updated.status_code == 200
        body = updated.json()
        assert body["orchestrator"]["concurrency"] == 5
        assert body["orchestrator"]["auto_deps"] is False
        assert body["orchestrator"]["max_review_attempts"] == 4
        assert body["agent_routing"]["default_role"] == "reviewer"
        assert body["agent_routing"]["task_type_roles"]["bug"] == "debugger"
        assert body["agent_routing"]["role_provider_overrides"]["reviewer"] == "openai"
        assert body["defaults"]["quality_gate"]["critical"] == 1
        assert body["defaults"]["quality_gate"]["high"] == 2
        assert body["defaults"]["quality_gate"]["medium"] == 3
        assert body["defaults"]["quality_gate"]["low"] == 4
        assert body["workers"]["default"] == "ollama-dev"
        assert body["workers"]["default_model"] == "gpt-5-codex"
        assert body["workers"]["heartbeat_seconds"] == 90
        assert body["workers"]["heartbeat_grace_seconds"] == 360
        assert body["workers"]["routing"]["plan"] == "codex"
        assert body["workers"]["routing"]["implement"] == "ollama-dev"
        assert body["workers"]["providers"]["codex"]["type"] == "codex"
        assert body["workers"]["providers"]["codex"]["model"] == "gpt-5-codex"
        assert body["workers"]["providers"]["codex"]["reasoning_effort"] == "high"
        assert body["workers"]["providers"]["ollama-dev"]["type"] == "ollama"
        assert body["workers"]["providers"]["ollama-dev"]["model"] == "llama3.1:8b"
        assert body["workers"]["providers"]["claude"]["type"] == "claude"
        assert body["workers"]["providers"]["claude"]["command"] == "claude -p"
        assert body["workers"]["providers"]["claude"]["model"] == "sonnet"
        assert body["workers"]["providers"]["claude"]["reasoning_effort"] == "medium"

        reloaded = client.get("/api/settings")
        assert reloaded.status_code == 200
        assert reloaded.json() == body


def test_create_task_worker_model_round_trip(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"title": "Model override", "worker_model": "gpt-5-codex"})
        assert created.status_code == 200
        task = created.json()["task"]
        assert task["worker_model"] == "gpt-5-codex"

        loaded = client.get(f"/api/tasks/{task['id']}")
        assert loaded.status_code == 200
        assert loaded.json()["task"]["worker_model"] == "gpt-5-codex"


def test_project_commands_settings_round_trip(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        # Baseline: empty commands
        baseline = client.get("/api/settings")
        assert baseline.status_code == 200
        assert baseline.json()["project"]["commands"] == {}

        # Set python commands
        resp = client.patch(
            "/api/settings",
            json={"project": {"commands": {"python": {"test": "pytest -n auto", "lint": "ruff check ."}}}},
        )
        assert resp.status_code == 200
        cmds = resp.json()["project"]["commands"]
        assert cmds["python"]["test"] == "pytest -n auto"
        assert cmds["python"]["lint"] == "ruff check ."

        # Reload and verify persistence
        reloaded = client.get("/api/settings")
        assert reloaded.json()["project"]["commands"] == cmds

        # Merge: add typecheck, leave test and lint untouched
        resp2 = client.patch(
            "/api/settings",
            json={"project": {"commands": {"python": {"typecheck": "mypy ."}}}},
        )
        assert resp2.status_code == 200
        cmds2 = resp2.json()["project"]["commands"]["python"]
        assert cmds2["test"] == "pytest -n auto"
        assert cmds2["lint"] == "ruff check ."
        assert cmds2["typecheck"] == "mypy ."

        # Remove a field by setting to empty string
        resp3 = client.patch(
            "/api/settings",
            json={"project": {"commands": {"python": {"lint": ""}}}},
        )
        assert resp3.status_code == 200
        cmds3 = resp3.json()["project"]["commands"]["python"]
        assert "lint" not in cmds3
        assert cmds3["test"] == "pytest -n auto"
        assert cmds3["typecheck"] == "mypy ."

        # Remove all fields for a language → language entry removed
        resp4 = client.patch(
            "/api/settings",
            json={"project": {"commands": {"python": {"test": "", "typecheck": ""}}}},
        )
        assert resp4.status_code == 200
        assert resp4.json()["project"]["commands"] == {}


def test_project_commands_language_key_normalized(tmp_path: Path) -> None:
    """Uppercase/mixed-case language keys are normalized to lowercase."""
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        resp = client.patch(
            "/api/settings",
            json={"project": {"commands": {"PYTHON": {"test": "pytest"}, "TypeScript": {"lint": "eslint ."}}}},
        )
        assert resp.status_code == 200
        cmds = resp.json()["project"]["commands"]
        assert "python" in cmds
        assert "typescript" in cmds
        assert "PYTHON" not in cmds
        assert "TypeScript" not in cmds


def test_project_commands_empty_language_key_ignored(tmp_path: Path) -> None:
    """Empty string language key is silently ignored."""
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        resp = client.patch(
            "/api/settings",
            json={"project": {"commands": {"": {"test": "pytest"}, "python": {"lint": "ruff check ."}}}},
        )
        assert resp.status_code == 200
        cmds = resp.json()["project"]["commands"]
        assert "" not in cmds
        assert cmds["python"]["lint"] == "ruff check ."


def test_claim_lock_prevents_double_claim(tmp_path: Path) -> None:
    container = Container(tmp_path)
    task = Task(title="Concurrent claim", status="queued")
    container.tasks.upsert(task)

    def _claim() -> str | None:
        claimed = container.tasks.claim_next_runnable(max_in_progress=4)
        return claimed.id if claimed else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: _claim(), range(2)))

    claimed_ids = [task_id for task_id in outcomes if task_id is not None]
    assert claimed_ids == [task.id]


def test_state_machine_allows_and_rejects_expected_transitions(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"title": "FSM", "status": "backlog"}).json()["task"]
        task_id = task["id"]

        valid = client.post(f"/api/tasks/{task_id}/transition", json={"status": "queued"})
        assert valid.status_code == 200

        invalid = client.post(f"/api/tasks/{task_id}/transition", json={"status": "done"})
        assert invalid.status_code == 400
        assert "Invalid transition" in invalid.text


def test_api_surfaces_human_blocking_issues_on_task_and_timeline(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post(
            "/api/tasks",
            json={
                "title": "Needs credentials",
                "status": "backlog",
                "approval_mode": "auto_approve",
                "metadata": {
                    "scripted_steps": {
                        "plan": {
                            "status": "human_blocked",
                            "summary": "Need production API token",
                            "human_blocking_issues": [
                                {
                                    "summary": "Need production API token",
                                    "details": "Grant read-only access",
                                    "action": "Provide token",
                                }
                            ],
                        }
                    }
                },
            },
        ).json()["task"]

        run_resp = client.post(f"/api/tasks/{created['id']}/run")
        assert run_resp.status_code == 200
        task = run_resp.json()["task"]
        assert task["status"] == "blocked"
        assert task["pending_gate"] == "human_intervention"
        assert len(task.get("human_blocking_issues") or []) == 1
        assert task["human_blocking_issues"][0]["summary"] == "Need production API token"

        timeline = client.get(f"/api/collaboration/timeline/{created['id']}")
        assert timeline.status_code == 200
        events = timeline.json()["events"]
        assert any((event.get("human_blocking_issues") or []) for event in events)


def test_retry_clears_pending_gate_and_human_blockers(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post(
            "/api/tasks",
            json={
                "title": "Needs credentials",
                "status": "backlog",
                "approval_mode": "auto_approve",
                "metadata": {
                    "scripted_steps": {
                        "plan": {
                            "status": "human_blocked",
                            "summary": "Need production API token",
                            "human_blocking_issues": [{"summary": "Need production API token"}],
                        }
                    }
                },
            },
        ).json()["task"]

        run_resp = client.post(f"/api/tasks/{created['id']}/run")
        assert run_resp.status_code == 200
        blocked_task = run_resp.json()["task"]
        assert blocked_task["status"] == "blocked"
        assert blocked_task["pending_gate"] == "human_intervention"
        assert blocked_task.get("human_blocking_issues")

        retry_resp = client.post(f"/api/tasks/{created['id']}/retry")
        assert retry_resp.status_code == 200
        retried_task = retry_resp.json()["task"]
        assert retried_task["status"] == "queued"
        assert retried_task["pending_gate"] is None
        assert retried_task.get("human_blocking_issues") == []

        rerun_resp = client.post(f"/api/tasks/{created['id']}/run")
        assert rerun_resp.status_code == 200


def test_review_queue_request_changes_and_approve(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={"title": "Needs review", "status": "backlog", "approval_mode": "human_review", "metadata": {"scripted_findings": [[]]}},
        ).json()["task"]
        ran = client.post(f"/api/tasks/{task['id']}/run")
        assert ran.status_code == 200
        assert ran.json()["task"]["status"] == "in_review"

        queue = client.get("/api/review-queue").json()
        assert queue["total"] == 1
        assert queue["tasks"][0]["id"] == task["id"]

        request_changes = client.post(
            f"/api/review/{task['id']}/request-changes",
            json={"guidance": "Please adjust tests"},
        )
        assert request_changes.status_code == 200
        assert request_changes.json()["task"]["status"] == "queued"

        client.post(f"/api/tasks/{task['id']}/run")
        approved = client.post(f"/api/review/{task['id']}/approve", json={})
        assert approved.status_code == 200
        assert approved.json()["task"]["status"] == "done"


def test_get_task_plan(tmp_path: Path) -> None:
    """GET /api/tasks/{id}/plan returns stored plan history."""
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        # Create a task with plans in metadata
        task = client.post(
            "/api/tasks",
            json={
                "title": "Plan query test",
                "metadata": {
                    "plans": [
                        {"step": "plan", "ts": "2025-01-01T00:00:00Z", "content": "Plan A"},
                        {"step": "plan_impl", "ts": "2025-01-01T00:01:00Z", "content": "Plan B"},
                    ]
                },
            },
        ).json()["task"]

        resp = client.get(f"/api/tasks/{task['id']}/plan")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["plans"]) == 2
        assert body["latest"]["content"] == "Plan B"
        assert body["latest"]["step"] == "plan_impl"

        # Non-existent task → 404
        resp_404 = client.get("/api/tasks/nonexistent/plan")
        assert resp_404.status_code == 404

        # Task with no plans → empty list
        task_no_plan = client.post("/api/tasks", json={"title": "No plan"}).json()["task"]
        resp_empty = client.get(f"/api/tasks/{task_no_plan['id']}/plan")
        assert resp_empty.status_code == 200
        assert resp_empty.json()["plans"] == []
        assert resp_empty.json()["latest"] is None


def test_generate_tasks_endpoint(tmp_path: Path) -> None:
    """POST /api/tasks/{id}/generate-tasks creates child tasks from stored plan."""
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        # Create a task with a plan and scripted generation output
        task = client.post(
            "/api/tasks",
            json={
                "title": "Generate from plan",
                "status": "backlog",
                "metadata": {
                    "plans": [
                        {"step": "plan", "ts": "2025-01-01T00:00:00Z", "content": "Build auth"},
                    ],
                    "scripted_generated_tasks": [
                        {"title": "Login endpoint", "task_type": "feature", "priority": "P1"},
                        {"title": "Session store", "task_type": "feature", "priority": "P2"},
                    ],
                },
            },
        ).json()["task"]

        resp = client.post(f"/api/tasks/{task['id']}/generate-tasks", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["created_task_ids"]) == 2
        assert len(body["children"]) == 2
        assert body["children"][0]["title"] == "Login endpoint"
        assert body["children"][1]["title"] == "Session store"
        assert body["task"]["children_ids"] == body["created_task_ids"]


def test_generate_tasks_endpoint_returns_400_when_worker_outputs_no_tasks(tmp_path: Path) -> None:
    """Explicit generate should fail loudly when worker returns no task list."""
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Generate no output",
                "status": "backlog",
                "metadata": {
                    "plans": [
                        {"step": "plan", "ts": "2025-01-01T00:00:00Z", "content": "Build auth"},
                    ],
                    "scripted_steps": {
                        "generate_tasks": {"status": "ok", "summary": "No decomposition possible."}
                    },
                },
            },
        ).json()["task"]

        resp = client.post(f"/api/tasks/{task['id']}/generate-tasks", json={"source": "latest"})
        assert resp.status_code == 400
        assert "no generated tasks" in resp.json()["detail"].lower()


def test_generate_tasks_with_override(tmp_path: Path) -> None:
    """POST with plan_override doesn't require stored plan."""
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Override plan test",
                "metadata": {
                    "scripted_generated_tasks": [
                        {"title": "From override", "task_type": "feature"},
                    ],
                },
            },
        ).json()["task"]

        # No stored plan, no override → 400
        resp_fail = client.post(
            f"/api/tasks/{task['id']}/generate-tasks",
            json={},
        )
        assert resp_fail.status_code == 400
        assert "No plan revision exists" in resp_fail.json()["detail"]

        # With override → success
        resp = client.post(
            f"/api/tasks/{task['id']}/generate-tasks",
            json={"source": "override", "plan_override": "Custom plan text"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["created_task_ids"]) == 1
        assert resp.json()["children"][0]["title"] == "From override"


def test_plan_refine_job_lifecycle_and_commit(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Iterative plan",
                "metadata": {
                    "plans": [{"step": "plan", "ts": "2025-01-01T00:00:00Z", "content": "Initial plan"}],
                    "scripted_steps": {
                        "plan_refine": {"status": "ok", "summary": "Refined plan with rollout"}
                    },
                },
            },
        ).json()["task"]

        initial_doc = client.get(f"/api/tasks/{task['id']}/plan").json()
        assert len(initial_doc["revisions"]) == 1
        base_revision_id = initial_doc["latest_revision_id"]

        queued = client.post(
            f"/api/tasks/{task['id']}/plan/refine",
            json={"base_revision_id": base_revision_id, "feedback": "Add rollout and risk checks"},
        )
        assert queued.status_code == 200
        job_id = queued.json()["job"]["id"]

        status = ""
        for _ in range(50):
            job = client.get(f"/api/tasks/{task['id']}/plan/jobs/{job_id}").json()["job"]
            status = job["status"]
            if status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert status == "completed"
        result_revision_id = job["result_revision_id"]
        assert result_revision_id

        doc = client.get(f"/api/tasks/{task['id']}/plan").json()
        assert doc["latest_revision_id"] == result_revision_id
        assert len(doc["revisions"]) == 2
        refined = next(item for item in doc["revisions"] if item["id"] == result_revision_id)
        assert refined["source"] == "worker_refine"
        assert refined["parent_revision_id"] == base_revision_id

        commit = client.post(
            f"/api/tasks/{task['id']}/plan/commit",
            json={"revision_id": result_revision_id},
        )
        assert commit.status_code == 200
        assert commit.json()["committed_revision_id"] == result_revision_id

        doc_after_commit = client.get(f"/api/tasks/{task['id']}/plan").json()
        assert doc_after_commit["committed_revision_id"] == result_revision_id


def test_plan_refine_failure_path(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Broken refine",
                "metadata": {
                    "plans": [{"step": "plan", "ts": "2025-01-01T00:00:00Z", "content": "Base"}],
                    "scripted_steps": {"plan_refine": {"status": "error", "summary": "worker failed"}},
                },
            },
        ).json()["task"]

        queued = client.post(
            f"/api/tasks/{task['id']}/plan/refine",
            json={"feedback": "Refine this"},
        )
        assert queued.status_code == 200
        job_id = queued.json()["job"]["id"]

        status = ""
        for _ in range(50):
            job = client.get(f"/api/tasks/{task['id']}/plan/jobs/{job_id}").json()["job"]
            status = job["status"]
            if status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert status == "failed"
        assert "worker failed" in str(job.get("error") or "")


def test_plan_refine_rejects_summary_style_output(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Reject summary refine",
                "metadata": {
                    "plans": [{"step": "plan", "ts": "2025-01-01T00:00:00Z", "content": "Base plan with details"}],
                    "scripted_steps": {
                        "plan_refine": {
                            "status": "ok",
                            "summary": (
                                "The plan has been prepared. Here's a summary of key changes.\n\n"
                                "## Refined Plan — Key Changes\n- Tightened scope\n\n"
                                "Should I write the full PLAN.md to the repo?"
                            ),
                        }
                    },
                },
            },
        ).json()["task"]

        before = client.get(f"/api/tasks/{task['id']}/plan").json()
        assert len(before["revisions"]) == 1

        queued = client.post(
            f"/api/tasks/{task['id']}/plan/refine",
            json={"feedback": "Please provide full rewrite"},
        )
        assert queued.status_code == 200
        job_id = queued.json()["job"]["id"]

        status = ""
        for _ in range(50):
            job = client.get(f"/api/tasks/{task['id']}/plan/jobs/{job_id}").json()["job"]
            status = job["status"]
            if status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert status == "failed"
        assert "summary-style output" in str(job.get("error") or "").lower()

        after = client.get(f"/api/tasks/{task['id']}/plan").json()
        assert len(after["revisions"]) == 1


def test_plan_refine_emit_error_does_not_mark_job_failed(tmp_path: Path, monkeypatch) -> None:
    from agent_orchestrator.runtime.events.bus import EventBus

    original_emit = EventBus.emit

    def flaky_emit(self, *, channel: str, event_type: str, entity_id: str, payload: dict | None = None):
        if event_type == "plan.refine.completed":
            raise RuntimeError("synthetic bus failure")
        return original_emit(self, channel=channel, event_type=event_type, entity_id=entity_id, payload=payload)

    monkeypatch.setattr(EventBus, "emit", flaky_emit)

    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Refine resilient finalize",
                "metadata": {
                    "plans": [{"step": "plan", "ts": "2025-01-01T00:00:00Z", "content": "Base plan"}],
                    "scripted_steps": {"plan_refine": {"status": "ok", "summary": "Refined output"}},
                },
            },
        ).json()["task"]

        queued = client.post(
            f"/api/tasks/{task['id']}/plan/refine",
            json={"feedback": "Please refine"},
        )
        assert queued.status_code == 200
        job_id = queued.json()["job"]["id"]

        status = ""
        for _ in range(50):
            job = client.get(f"/api/tasks/{task['id']}/plan/jobs/{job_id}").json()["job"]
            status = job["status"]
            if status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)

        assert status == "completed"
        assert job["result_revision_id"]
        doc = client.get(f"/api/tasks/{task['id']}/plan").json()
        assert any(item["id"] == job["result_revision_id"] for item in doc["revisions"])


def test_generate_tasks_with_explicit_plan_sources(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Source selection",
                "metadata": {
                    "plans": [{"step": "plan", "ts": "2025-01-01T00:00:00Z", "content": "Base plan"}],
                    "scripted_generated_tasks": [{"title": "From selected source", "task_type": "feature"}],
                },
            },
        ).json()["task"]

        plan_doc = client.get(f"/api/tasks/{task['id']}/plan").json()
        latest_revision_id = plan_doc["latest_revision_id"]

        manual = client.post(
            f"/api/tasks/{task['id']}/plan/revisions",
            json={"content": "Manual plan text", "parent_revision_id": latest_revision_id, "feedback_note": "manual tweak"},
        )
        assert manual.status_code == 200
        manual_revision_id = manual.json()["revision"]["id"]

        bad_revision = client.post(
            f"/api/tasks/{task['id']}/generate-tasks",
            json={"source": "revision"},
        )
        assert bad_revision.status_code == 400

        good_revision = client.post(
            f"/api/tasks/{task['id']}/generate-tasks",
            json={"source": "revision", "revision_id": manual_revision_id},
        )
        assert good_revision.status_code == 200
        assert good_revision.json()["source"] == "revision"
        assert good_revision.json()["source_revision_id"] == manual_revision_id

        commit = client.post(
            f"/api/tasks/{task['id']}/plan/commit",
            json={"revision_id": manual_revision_id},
        )
        assert commit.status_code == 200

        committed_source = client.post(
            f"/api/tasks/{task['id']}/generate-tasks",
            json={"source": "committed"},
        )
        assert committed_source.status_code == 200
        assert committed_source.json()["source_revision_id"] == manual_revision_id
