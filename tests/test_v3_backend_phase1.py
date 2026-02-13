from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from feature_prd_runner.server.api import create_app
from feature_prd_runner.v3.orchestrator import DefaultWorkerAdapter
from feature_prd_runner.v3.domain.models import QuickActionRun, Task
from feature_prd_runner.v3.storage.bootstrap import ensure_v3_state_root
from feature_prd_runner.v3.storage.container import V3Container


def test_cutover_archives_legacy_state(tmp_path: Path) -> None:
    legacy_root = tmp_path / ".prd_runner"
    legacy_root.mkdir(parents=True)
    (legacy_root / "task_queue.yaml").write_text("tasks: []\n", encoding="utf-8")
    (legacy_root / "run_state.yaml").write_text("{}\n", encoding="utf-8")

    v3_root = ensure_v3_state_root(tmp_path)

    assert v3_root == tmp_path / ".prd_runner" / "v3"
    assert (v3_root / "config.yaml").exists()
    config = (v3_root / "config.yaml").read_text(encoding="utf-8")
    assert "schema_version: 3" in config

    archives = sorted(tmp_path.glob(".prd_runner_legacy_*"))
    assert len(archives) == 1
    assert (archives[0] / "task_queue.yaml").exists()


def test_cutover_archives_any_non_v3_prd_runner_state(tmp_path: Path) -> None:
    legacy_root = tmp_path / ".prd_runner"
    legacy_root.mkdir(parents=True)
    (legacy_root / "custom_legacy_blob.yaml").write_text("legacy: true\n", encoding="utf-8")

    ensure_v3_state_root(tmp_path)

    archives = sorted(tmp_path.glob(".prd_runner_legacy_*"))
    assert len(archives) == 1
    assert (archives[0] / "custom_legacy_blob.yaml").exists()


def test_cutover_forces_schema_version_3(tmp_path: Path) -> None:
    v3_root = tmp_path / ".prd_runner" / "v3"
    v3_root.mkdir(parents=True)
    (v3_root / "config.yaml").write_text("schema_version: 2\n", encoding="utf-8")

    ensure_v3_state_root(tmp_path)

    config_text = (tmp_path / ".prd_runner" / "v3" / "config.yaml").read_text(encoding="utf-8")
    assert "schema_version: 3" in config_text


def test_task_dependency_guard_blocks_ready_transition(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        blocker = client.post(
            "/api/v3/tasks",
            json={"title": "Blocker", "approval_mode": "auto_approve", "metadata": {"scripted_findings": [[]]}},
        ).json()["task"]
        blocked = client.post("/api/v3/tasks", json={"title": "Blocked"}).json()["task"]

        dep_resp = client.post(
            f"/api/v3/tasks/{blocked['id']}/dependencies",
            json={"depends_on": blocker["id"]},
        )
        assert dep_resp.status_code == 200

        transition = client.post(
            f"/api/v3/tasks/{blocked['id']}/transition",
            json={"status": "ready"},
        )
        assert transition.status_code == 400
        assert "Unresolved blocker" in transition.text

        done = client.post(f"/api/v3/tasks/{blocker['id']}/run")
        assert done.status_code == 200
        assert done.json()["task"]["status"] == "done"
        ok_transition = client.post(
            f"/api/v3/tasks/{blocked['id']}/transition",
            json={"status": "ready"},
        )
        assert ok_transition.status_code == 200
        assert ok_transition.json()["task"]["status"] == "ready"


def test_patch_rejects_direct_status_changes(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post("/api/v3/tasks", json={"title": "Patch guarded"}).json()["task"]
        response = client.patch(f"/api/v3/tasks/{task['id']}", json={"status": "done"})
        assert response.status_code == 400
        assert "cannot be changed via PATCH" in response.text


def test_review_actions_require_in_review_state(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post("/api/v3/tasks", json={"title": "Needs status guard"}).json()["task"]
        approve = client.post(f"/api/v3/review/{task['id']}/approve", json={})
        assert approve.status_code == 400
        changes = client.post(f"/api/v3/review/{task['id']}/request-changes", json={"guidance": "x"})
        assert changes.status_code == 400


def test_quick_action_promotion_is_singleton(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        qrun = client.post("/api/v3/quick-actions", json={"prompt": "Do thing"}).json()["quick_action"]

        first = client.post(f"/api/v3/quick-actions/{qrun['id']}/promote", json={})
        assert first.status_code == 200
        assert first.json()["already_promoted"] is False
        task_id = first.json()["task"]["id"]

        second = client.post(f"/api/v3/quick-actions/{qrun['id']}/promote", json={})
        assert second.status_code == 200
        assert second.json()["already_promoted"] is True
        assert second.json()["task"]["id"] == task_id

        tasks = client.get("/api/v3/tasks").json()["tasks"]
        assert [task["id"] for task in tasks].count(task_id) == 1


def test_quick_action_pending_limit_returns_429(tmp_path: Path) -> None:
    container = V3Container(tmp_path)
    for idx in range(32):
        container.quick_actions.upsert(QuickActionRun(prompt=f"Run {idx}", status="running"))

    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        response = client.post("/api/v3/quick-actions", json={"prompt": "git status"})
        assert response.status_code == 429


def test_project_pin_requires_git_unless_override(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()

    with TestClient(app) as client:
        rejected = client.post("/api/v3/projects/pinned", json={"path": str(plain_dir)})
        assert rejected.status_code == 400

        accepted = client.post(
            "/api/v3/projects/pinned",
            json={"path": str(plain_dir), "allow_non_git": True},
        )
        assert accepted.status_code == 200
        listing = client.get("/api/v3/projects/pinned").json()["items"]
        assert len(listing) == 1


def test_import_preview_commit_creates_dependency_chain(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        preview = client.post(
            "/api/v3/import/prd/preview",
            json={"content": "- First\n- Second\n- Third", "default_priority": "P1"},
        )
        assert preview.status_code == 200
        job_id = preview.json()["job_id"]

        commit = client.post("/api/v3/import/prd/commit", json={"job_id": job_id})
        assert commit.status_code == 200
        created_ids = commit.json()["created_task_ids"]
        assert len(created_ids) == 3

        tasks = {item["id"]: item for item in client.get("/api/v3/tasks").json()["tasks"]}
        assert tasks[created_ids[0]]["blocked_by"] == []
        assert tasks[created_ids[1]]["blocked_by"] == [created_ids[0]]
        assert tasks[created_ids[2]]["blocked_by"] == [created_ids[1]]

        job = client.get(f"/api/v3/import/{job_id}")
        assert job.status_code == 200
        assert job.json()["job"]["created_task_ids"] == created_ids


def test_import_job_persists_when_in_memory_cache_is_empty(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        preview = client.post(
            "/api/v3/import/prd/preview",
            json={"content": "- Persist me"},
        )
        assert preview.status_code == 200
        job_id = preview.json()["job_id"]

        app.state.import_jobs.clear()
        loaded = client.get(f"/api/v3/import/{job_id}")
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


def test_legacy_compat_endpoints_available(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post("/api/v3/tasks", json={"title": "Compat seed"}).json()["task"]

        metrics = client.get("/api/v3/metrics")
        assert metrics.status_code == 200
        assert "phases_total" in metrics.json()

        phases = client.get("/api/v3/phases")
        assert phases.status_code == 200
        assert isinstance(phases.json(), list)

        agent_types = client.get("/api/v3/agents/types")
        assert agent_types.status_code == 200
        assert len(agent_types.json()["types"]) > 0

        modes = client.get("/api/v3/collaboration/modes")
        assert modes.status_code == 200
        assert len(modes.json()["modes"]) > 0

        add_feedback = client.post(
            "/api/v3/collaboration/feedback",
            json={"task_id": created["id"], "summary": "Need stricter checks"},
        )
        assert add_feedback.status_code == 200
        feedback_id = add_feedback.json()["feedback"]["id"]

        add_comment = client.post(
            "/api/v3/collaboration/comments",
            json={"task_id": created["id"], "file_path": "main.py", "line_number": 1, "body": "Looks good"},
        )
        assert add_comment.status_code == 200
        comment_id = add_comment.json()["comment"]["id"]

        feedback_list = client.get(f"/api/v3/collaboration/feedback/{created['id']}")
        assert feedback_list.status_code == 200
        assert any(item["id"] == feedback_id for item in feedback_list.json()["feedback"])

        comment_list = client.get(f"/api/v3/collaboration/comments/{created['id']}")
        assert comment_list.status_code == 200
        assert any(item["id"] == comment_id for item in comment_list.json()["comments"])

        dismissed = client.post(f"/api/v3/collaboration/feedback/{feedback_id}/dismiss")
        assert dismissed.status_code == 200
        assert dismissed.json()["feedback"]["status"] == "addressed"

        resolved = client.post(f"/api/v3/collaboration/comments/{comment_id}/resolve")
        assert resolved.status_code == 200
        assert resolved.json()["comment"]["resolved"] is True

        timeline = client.get(f"/api/v3/collaboration/timeline/{created['id']}")
        assert timeline.status_code == 200
        assert len(timeline.json()["events"]) >= 1
        types = {event["type"] for event in timeline.json()["events"]}
        assert "feedback" in types
        assert "comment" in types


def test_settings_endpoint_round_trip(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        baseline = client.get("/api/v3/settings")
        assert baseline.status_code == 200
        assert baseline.json()["orchestrator"]["concurrency"] == 2
        assert baseline.json()["orchestrator"]["auto_deps"] is True
        assert baseline.json()["orchestrator"]["max_review_attempts"] == 3
        assert baseline.json()["agent_routing"]["default_role"] == "general"
        assert baseline.json()["defaults"]["quality_gate"]["high"] == 0
        assert baseline.json()["workers"]["default"] == "codex"
        assert baseline.json()["workers"]["providers"]["codex"]["type"] == "codex"

        updated = client.patch(
            "/api/v3/settings",
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
                    "routing": {"plan": "codex", "implement": "ollama-dev"},
                    "providers": {
                        "codex": {"type": "codex", "command": "codex"},
                        "ollama-dev": {
                            "type": "ollama",
                            "endpoint": "http://localhost:11434",
                            "model": "llama3.1:8b",
                            "temperature": 0.2,
                            "num_ctx": 8192,
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
        assert body["workers"]["routing"]["plan"] == "codex"
        assert body["workers"]["routing"]["implement"] == "ollama-dev"
        assert body["workers"]["providers"]["codex"]["type"] == "codex"
        assert body["workers"]["providers"]["ollama-dev"]["type"] == "ollama"
        assert body["workers"]["providers"]["ollama-dev"]["model"] == "llama3.1:8b"

        reloaded = client.get("/api/v3/settings")
        assert reloaded.status_code == 200
        assert reloaded.json() == body


def test_project_commands_settings_round_trip(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        # Baseline: empty commands
        baseline = client.get("/api/v3/settings")
        assert baseline.status_code == 200
        assert baseline.json()["project"]["commands"] == {}

        # Set python commands
        resp = client.patch(
            "/api/v3/settings",
            json={"project": {"commands": {"python": {"test": "pytest -n auto", "lint": "ruff check ."}}}},
        )
        assert resp.status_code == 200
        cmds = resp.json()["project"]["commands"]
        assert cmds["python"]["test"] == "pytest -n auto"
        assert cmds["python"]["lint"] == "ruff check ."

        # Reload and verify persistence
        reloaded = client.get("/api/v3/settings")
        assert reloaded.json()["project"]["commands"] == cmds

        # Merge: add typecheck, leave test and lint untouched
        resp2 = client.patch(
            "/api/v3/settings",
            json={"project": {"commands": {"python": {"typecheck": "mypy ."}}}},
        )
        assert resp2.status_code == 200
        cmds2 = resp2.json()["project"]["commands"]["python"]
        assert cmds2["test"] == "pytest -n auto"
        assert cmds2["lint"] == "ruff check ."
        assert cmds2["typecheck"] == "mypy ."

        # Remove a field by setting to empty string
        resp3 = client.patch(
            "/api/v3/settings",
            json={"project": {"commands": {"python": {"lint": ""}}}},
        )
        assert resp3.status_code == 200
        cmds3 = resp3.json()["project"]["commands"]["python"]
        assert "lint" not in cmds3
        assert cmds3["test"] == "pytest -n auto"
        assert cmds3["typecheck"] == "mypy ."

        # Remove all fields for a language → language entry removed
        resp4 = client.patch(
            "/api/v3/settings",
            json={"project": {"commands": {"python": {"test": "", "typecheck": ""}}}},
        )
        assert resp4.status_code == 200
        assert resp4.json()["project"]["commands"] == {}


def test_project_commands_language_key_normalized(tmp_path: Path) -> None:
    """Uppercase/mixed-case language keys are normalized to lowercase."""
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v3/settings",
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
            "/api/v3/settings",
            json={"project": {"commands": {"": {"test": "pytest"}, "python": {"lint": "ruff check ."}}}},
        )
        assert resp.status_code == 200
        cmds = resp.json()["project"]["commands"]
        assert "" not in cmds
        assert cmds["python"]["lint"] == "ruff check ."


def test_claim_lock_prevents_double_claim(tmp_path: Path) -> None:
    container = V3Container(tmp_path)
    task = Task(title="Concurrent claim", status="ready")
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
        task = client.post("/api/v3/tasks", json={"title": "FSM"}).json()["task"]
        task_id = task["id"]

        valid = client.post(f"/api/v3/tasks/{task_id}/transition", json={"status": "ready"})
        assert valid.status_code == 200

        invalid = client.post(f"/api/v3/tasks/{task_id}/transition", json={"status": "done"})
        assert invalid.status_code == 400
        assert "Invalid transition" in invalid.text


def test_api_surfaces_human_blocking_issues_on_task_and_timeline(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post(
            "/api/v3/tasks",
            json={
                "title": "Needs credentials",
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

        run_resp = client.post(f"/api/v3/tasks/{created['id']}/run")
        assert run_resp.status_code == 200
        task = run_resp.json()["task"]
        assert task["status"] == "blocked"
        assert task["pending_gate"] == "human_intervention"
        assert len(task.get("human_blocking_issues") or []) == 1
        assert task["human_blocking_issues"][0]["summary"] == "Need production API token"

        timeline = client.get(f"/api/v3/collaboration/timeline/{created['id']}")
        assert timeline.status_code == 200
        events = timeline.json()["events"]
        assert any((event.get("human_blocking_issues") or []) for event in events)


def test_retry_clears_pending_gate_and_human_blockers(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post(
            "/api/v3/tasks",
            json={
                "title": "Needs credentials",
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

        run_resp = client.post(f"/api/v3/tasks/{created['id']}/run")
        assert run_resp.status_code == 200
        blocked_task = run_resp.json()["task"]
        assert blocked_task["status"] == "blocked"
        assert blocked_task["pending_gate"] == "human_intervention"
        assert blocked_task.get("human_blocking_issues")

        retry_resp = client.post(f"/api/v3/tasks/{created['id']}/retry")
        assert retry_resp.status_code == 200
        retried_task = retry_resp.json()["task"]
        assert retried_task["status"] == "ready"
        assert retried_task["pending_gate"] is None
        assert retried_task.get("human_blocking_issues") == []

        rerun_resp = client.post(f"/api/v3/tasks/{created['id']}/run")
        assert rerun_resp.status_code == 200


def test_review_queue_request_changes_and_approve(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/v3/tasks",
            json={"title": "Needs review", "approval_mode": "human_review", "metadata": {"scripted_findings": [[]]}},
        ).json()["task"]
        ran = client.post(f"/api/v3/tasks/{task['id']}/run")
        assert ran.status_code == 200
        assert ran.json()["task"]["status"] == "in_review"

        queue = client.get("/api/v3/review-queue").json()
        assert queue["total"] == 1
        assert queue["tasks"][0]["id"] == task["id"]

        request_changes = client.post(
            f"/api/v3/review/{task['id']}/request-changes",
            json={"guidance": "Please adjust tests"},
        )
        assert request_changes.status_code == 200
        assert request_changes.json()["task"]["status"] == "ready"

        client.post(f"/api/v3/tasks/{task['id']}/run")
        approved = client.post(f"/api/v3/review/{task['id']}/approve", json={})
        assert approved.status_code == 200
        assert approved.json()["task"]["status"] == "done"
