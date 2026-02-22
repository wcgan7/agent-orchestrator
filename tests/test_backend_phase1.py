from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from agent_orchestrator.server.api import create_app
from agent_orchestrator.runtime.orchestrator import DefaultWorkerAdapter
from agent_orchestrator.runtime.domain.models import RunRecord, Task, now_iso
from agent_orchestrator.runtime.orchestrator.worker_adapter import StepResult
from agent_orchestrator.runtime.storage.bootstrap import ensure_state_root
from agent_orchestrator.runtime.storage.container import Container


class _ClassifierWorkerAdapter(DefaultWorkerAdapter):
    def __init__(self, *, summary: str, status: str = "ok") -> None:
        super().__init__()
        self._summary = summary
        self._status = status

    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        if step == "pipeline_classify":
            return StepResult(status=self._status, summary=self._summary)
        return super().run_step(task=task, step=step, attempt=attempt)


class _PrdImportWorkerAdapter(DefaultWorkerAdapter):
    """Deterministic PRD import adapter for tests without runtime scripted metadata."""

    def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
        if step == "generate_tasks":
            parsed = task.metadata.get("parsed_prd") if isinstance(task.metadata, dict) else None
            candidates = parsed.get("task_candidates") if isinstance(parsed, dict) else None
            if not isinstance(candidates, list) or not candidates:
                lines = [line.strip() for line in str(task.description or "").splitlines()]
                fallback = [line[2:].strip() for line in lines if line.startswith("- ") and line[2:].strip()]
                candidates = [{"title": title, "priority": "P2"} for title in fallback]
            if isinstance(candidates, list):
                generated: list[dict[str, object]] = []
                prev: str | None = None
                for idx, item in enumerate(candidates, start=1):
                    if not isinstance(item, dict):
                        continue
                    ref_id = f"prd_{idx}"
                    generated.append(
                        {
                            "id": ref_id,
                            "title": str(item.get("title") or f"Imported PRD task {idx}"),
                            "task_type": "feature",
                            "priority": str(item.get("priority") or "P2"),
                            "depends_on": [prev] if prev else [],
                            "metadata": {
                                "generated_ref_id": ref_id,
                                "generated_depends_on": [prev] if prev else [],
                                "source_chunk_id": item.get("source_chunk_id"),
                                "source_section_path": item.get("source_section_path"),
                            },
                        }
                    )
                    prev = ref_id
                return StepResult(status="ok", generated_tasks=generated)
        return super().run_step(task=task, step=step, attempt=attempt)


def _create_plan_revision(client: TestClient, task_id: str, content: str, *, parent_revision_id: str | None = None) -> str:
    payload: dict[str, object] = {"content": content}
    if parent_revision_id:
        payload["parent_revision_id"] = parent_revision_id
    resp = client.post(f"/api/tasks/{task_id}/plan/revisions", json=payload)
    assert resp.status_code == 200
    return str(resp.json()["revision"]["id"])


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


def test_classify_pipeline_endpoint_high_confidence(tmp_path: Path) -> None:
    app = create_app(
        project_dir=tmp_path,
        worker_adapter=_ClassifierWorkerAdapter(
            summary='{"pipeline_id":"docs","confidence":"high","reason":"Documentation-only request."}'
        ),
    )
    with TestClient(app) as client:
        resp = client.post(
            "/api/tasks/classify-pipeline",
            json={"title": "Update README", "description": "Refresh setup instructions"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pipeline_id"] == "docs"
        assert body["task_type"] == "docs"
        assert body["confidence"] == "high"


def test_classify_pipeline_endpoint_invalid_output_downgrades_to_low(tmp_path: Path) -> None:
    app = create_app(
        project_dir=tmp_path,
        worker_adapter=_ClassifierWorkerAdapter(summary="not-json-output"),
    )
    with TestClient(app) as client:
        resp = client.post(
            "/api/tasks/classify-pipeline",
            json={"title": "Something vague", "description": "Need help"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence"] == "low"
        assert body["pipeline_id"] == "feature"


def test_create_task_rejects_auto_without_high_confidence_classification(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        resp = client.post(
            "/api/tasks",
            json={"title": "Auto create", "task_type": "auto"},
        )
        assert resp.status_code == 400
        assert "high-confidence classifier result" in resp.json()["detail"]


def test_create_task_accepts_auto_with_high_confidence_classification(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        resp = client.post(
            "/api/tasks",
            json={
                "title": "Auto create docs",
                "task_type": "auto",
                "classifier_pipeline_id": "docs",
                "classifier_confidence": "high",
                "classifier_reason": "Docs-only update",
            },
        )
        assert resp.status_code == 200
        body = resp.json()["task"]
        assert body["task_type"] == "docs"
        metadata = body.get("metadata") or {}
        assert metadata["classifier_pipeline_id"] == "docs"
        assert metadata["classifier_confidence"] == "high"
        assert metadata["final_pipeline_id"] == "docs"


def test_create_task_drops_invalid_classifier_metadata_for_manual_task(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        resp = client.post(
            "/api/tasks",
            json={
                "title": "Manual task with bogus classifier metadata",
                "task_type": "feature",
                "classifier_pipeline_id": "not_a_pipeline",
                "classifier_confidence": "high",
                "classifier_reason": "bogus",
                "was_user_override": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()["task"]
        metadata = body.get("metadata") or {}
        assert "classifier_pipeline_id" not in metadata
        assert "classifier_confidence" not in metadata
        assert "classifier_reason" not in metadata
        assert "was_user_override" not in metadata
        assert metadata["final_pipeline_id"] == "feature"


def test_tasks_board_uses_status_aware_ordering(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        seeded = {
            "backlog_high": client.post("/api/tasks", json={"title": "Backlog high", "priority": "P1"}).json()["task"]["id"],
            "backlog_urgent": client.post("/api/tasks", json={"title": "Backlog urgent", "priority": "P0"}).json()["task"]["id"],
            "in_progress_old": client.post("/api/tasks", json={"title": "In progress old", "priority": "P1"}).json()["task"]["id"],
            "in_progress_new": client.post("/api/tasks", json={"title": "In progress new", "priority": "P1"}).json()["task"]["id"],
            "in_review_old": client.post("/api/tasks", json={"title": "In review old", "priority": "P1"}).json()["task"]["id"],
            "in_review_new": client.post("/api/tasks", json={"title": "In review new", "priority": "P1"}).json()["task"]["id"],
            "blocked_old": client.post("/api/tasks", json={"title": "Blocked old", "priority": "P1"}).json()["task"]["id"],
            "blocked_new": client.post("/api/tasks", json={"title": "Blocked new", "priority": "P1"}).json()["task"]["id"],
            "done_old": client.post("/api/tasks", json={"title": "Done old", "priority": "P2"}).json()["task"]["id"],
            "done_new": client.post("/api/tasks", json={"title": "Done new", "priority": "P3"}).json()["task"]["id"],
            "done_p0": client.post("/api/tasks", json={"title": "Done p0", "priority": "P0"}).json()["task"]["id"],
            "done_p1": client.post("/api/tasks", json={"title": "Done p1", "priority": "P1"}).json()["task"]["id"],
            "cancelled_old": client.post("/api/tasks", json={"title": "Cancelled old", "priority": "P1"}).json()["task"]["id"],
            "cancelled_new": client.post("/api/tasks", json={"title": "Cancelled new", "priority": "P1"}).json()["task"]["id"],
        }
        tasks_path = tmp_path / ".agent_orchestrator" / "tasks.yaml"
        payload = yaml.safe_load(tasks_path.read_text(encoding="utf-8")) or {}
        records = payload.get("tasks") or []
        by_id = {str(item.get("id")): item for item in records if isinstance(item, dict)}

        for task_id, fields in {
            seeded["backlog_high"]: {
                "status": "backlog",
                "created_at": "2026-01-02T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
            },
            seeded["backlog_urgent"]: {
                "status": "backlog",
                "created_at": "2026-01-10T00:00:00+00:00",
                "updated_at": "2026-01-10T00:00:00+00:00",
            },
            seeded["in_progress_old"]: {
                "status": "in_progress",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-05T00:00:00+00:00",
            },
            seeded["in_progress_new"]: {
                "status": "in_progress",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-06T00:00:00+00:00",
            },
            seeded["in_review_old"]: {
                "status": "in_review",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-05T00:00:00+00:00",
            },
            seeded["in_review_new"]: {
                "status": "in_review",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-06T00:00:00+00:00",
            },
            seeded["blocked_old"]: {
                "status": "blocked",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-03T00:00:00+00:00",
            },
            seeded["blocked_new"]: {
                "status": "blocked",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-04T00:00:00+00:00",
            },
            seeded["done_old"]: {
                "status": "done",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-05T00:00:00+00:00",
            },
            seeded["done_new"]: {
                "status": "done",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-07T00:00:00+00:00",
            },
            seeded["done_p0"]: {
                "status": "done",
                "created_at": "2026-01-02T00:00:00+00:00",
                "updated_at": "2026-01-06T00:00:00+00:00",
            },
            seeded["done_p1"]: {
                "status": "done",
                "created_at": "2026-01-03T00:00:00+00:00",
                "updated_at": "2026-01-06T00:00:00+00:00",
            },
            seeded["cancelled_old"]: {
                "status": "cancelled",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
            },
            seeded["cancelled_new"]: {
                "status": "cancelled",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-03T00:00:00+00:00",
            },
        }.items():
            assert task_id in by_id
            by_id[task_id].update(fields)

        tasks_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        board = client.get("/api/tasks/board")
        assert board.status_code == 200
        columns = board.json()["columns"]

        assert [item["id"] for item in columns["backlog"]] == [seeded["backlog_urgent"], seeded["backlog_high"]]
        assert [item["id"] for item in columns["in_progress"]] == [seeded["in_progress_new"], seeded["in_progress_old"]]
        assert [item["id"] for item in columns["in_review"]] == [seeded["in_review_old"], seeded["in_review_new"]]
        assert [item["id"] for item in columns["blocked"]] == [seeded["blocked_new"], seeded["blocked_old"]]
        assert [item["id"] for item in columns["done"]] == [
            seeded["done_new"],
            seeded["done_p0"],
            seeded["done_p1"],
            seeded["done_old"],
        ]
        assert [item["id"] for item in columns["cancelled"]] == [seeded["cancelled_new"], seeded["cancelled_old"]]


def test_tasks_board_sorting_is_deterministic_for_ties_and_missing_timestamps(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        done_new = client.post("/api/tasks", json={"title": "Done with timestamp", "priority": "P2"}).json()["task"]["id"]
        done_bad_a = client.post("/api/tasks", json={"title": "Done malformed A", "priority": "P2"}).json()["task"]["id"]
        done_bad_b = client.post("/api/tasks", json={"title": "Done malformed B", "priority": "P2"}).json()["task"]["id"]
        backlog_tie_a = client.post("/api/tasks", json={"title": "Backlog tie A", "priority": "P2"}).json()["task"]["id"]
        backlog_tie_b = client.post("/api/tasks", json={"title": "Backlog tie B", "priority": "P2"}).json()["task"]["id"]

        done_tie_ids = sorted([done_bad_a, done_bad_b])
        backlog_tie_ids = sorted([backlog_tie_a, backlog_tie_b])

        tasks_path = tmp_path / ".agent_orchestrator" / "tasks.yaml"
        payload = yaml.safe_load(tasks_path.read_text(encoding="utf-8")) or {}
        records = payload.get("tasks") or []
        by_id = {str(item.get("id")): item for item in records if isinstance(item, dict)}

        for task_id, fields in {
            done_new: {
                "status": "done",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-04T00:00:00+00:00",
            },
            done_bad_a: {
                "status": "done",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "not-a-timestamp",
            },
            done_bad_b: {
                "status": "done",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "not-a-timestamp",
            },
            backlog_tie_a: {
                "status": "backlog",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            backlog_tie_b: {
                "status": "backlog",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
        }.items():
            assert task_id in by_id
            by_id[task_id].update(fields)

        tasks_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        board = client.get("/api/tasks/board")
        assert board.status_code == 200
        columns = board.json()["columns"]

        assert [item["id"] for item in columns["done"]] == [done_new, *done_tie_ids]
        assert [item["id"] for item in columns["backlog"]] == backlog_tie_ids


def test_terminal_session_is_singleton_per_project(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        first = client.post("/api/terminal/session", json={"cols": 100, "rows": 30})
        assert first.status_code == 200
        first_session = first.json()["session"]
        assert first_session["status"] in {"running", "starting"}

        second = client.post("/api/terminal/session", json={"cols": 120, "rows": 40})
        assert second.status_code == 200
        second_session = second.json()["session"]
        assert second_session["id"] == first_session["id"]


def test_terminal_session_io_logs_and_stop(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        start = client.post("/api/terminal/session", json={})
        assert start.status_code == 200
        session = start.json()["session"]
        session_id = session["id"]

        write = client.post(f"/api/terminal/session/{session_id}/input", json={"data": "echo hello\n"})
        assert write.status_code == 200

        output_text = ""
        for _ in range(40):
            logs = client.get(f"/api/terminal/session/{session_id}/logs?offset=0&max_bytes=262144")
            assert logs.status_code == 200
            output_text = str(logs.json().get("output") or "")
            if "hello" in output_text:
                break
            time.sleep(0.05)
        assert "hello" in output_text

        stop = client.post(f"/api/terminal/session/{session_id}/stop", json={"signal": "TERM"})
        assert stop.status_code == 200


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
    app = create_app(project_dir=tmp_path, worker_adapter=_PrdImportWorkerAdapter())
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


def test_import_preview_stores_original_and_parsed_prd_artifacts(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        content = "Feature: checkout\n\n- Add API endpoint\n- Add UI flow\n"
        preview = client.post(
            "/api/import/prd/preview",
            json={"content": content, "default_priority": "P1"},
        )
        assert preview.status_code == 200
        payload = preview.json()
        assert payload["preview"]["chunk_count"] >= 1
        assert payload["preview"]["strategy"] in {"heading_section", "paragraph_fallback", "token_window"}

        job_id = payload["job_id"]
        job_resp = client.get(f"/api/import/{job_id}")
        assert job_resp.status_code == 200
        job = job_resp.json()["job"]
        assert job["original_prd"]["content"] == content
        assert job["original_prd"]["checksum_sha256"]
        assert job["parsed_prd"]["chunk_count"] >= 1
        assert len(job["parsed_prd"]["task_candidates"]) >= 1


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


def test_get_task_includes_timing_summary_for_completed_runs(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    container = Container(tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"title": "Timing complete runs"}).json()["task"]
        task = container.tasks.get(created["id"])
        assert task is not None

        run_one = RunRecord(
            task_id=task.id,
            status="done",
            started_at="2026-02-21T10:00:00Z",
            finished_at="2026-02-21T10:00:30Z",
        )
        run_two = RunRecord(
            task_id=task.id,
            status="done",
            started_at="2026-02-21T10:02:00Z",
            finished_at="2026-02-21T10:03:00Z",
        )
        container.runs.upsert(run_one)
        container.runs.upsert(run_two)
        task.run_ids = [run_one.id, run_two.id]
        container.tasks.upsert(task)

        loaded = client.get(f"/api/tasks/{task.id}")
        assert loaded.status_code == 200
        timing = loaded.json()["task"]["timing_summary"]
        assert timing["is_running"] is False
        assert timing["active_run_started_at"] is None
        assert timing["total_completed_seconds"] == 90.0
        assert timing["first_started_at"] == "2026-02-21T10:00:00+00:00"
        assert timing["last_finished_at"] == "2026-02-21T10:03:00+00:00"


def test_get_task_timing_summary_keeps_completed_total_while_running(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    container = Container(tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"title": "Timing running run"}).json()["task"]
        task = container.tasks.get(created["id"])
        assert task is not None

        completed = RunRecord(
            task_id=task.id,
            status="done",
            started_at="2026-02-21T11:00:00Z",
            finished_at="2026-02-21T11:01:00Z",
        )
        active = RunRecord(
            task_id=task.id,
            status="in_progress",
            started_at="2026-02-21T11:05:00Z",
            finished_at=None,
        )
        container.runs.upsert(completed)
        container.runs.upsert(active)
        task.run_ids = [completed.id, active.id]
        container.tasks.upsert(task)

        loaded = client.get(f"/api/tasks/{task.id}")
        assert loaded.status_code == 200
        timing = loaded.json()["task"]["timing_summary"]
        assert timing["is_running"] is True
        assert timing["active_run_started_at"] == "2026-02-21T11:05:00+00:00"
        assert timing["total_completed_seconds"] == 60.0
        assert timing["first_started_at"] == "2026-02-21T11:00:00+00:00"
        assert timing["last_finished_at"] == "2026-02-21T11:01:00+00:00"


def test_get_task_timing_summary_ignores_malformed_timestamps(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    container = Container(tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"title": "Timing malformed"}).json()["task"]
        task = container.tasks.get(created["id"])
        assert task is not None

        bad_completed = RunRecord(
            task_id=task.id,
            status="done",
            started_at="not-a-timestamp",
            finished_at="2026-02-21T12:01:00Z",
        )
        bad_running = RunRecord(
            task_id=task.id,
            status="in_progress",
            started_at="bad-start",
            finished_at=None,
        )
        container.runs.upsert(bad_completed)
        container.runs.upsert(bad_running)
        task.run_ids = [bad_completed.id, bad_running.id]
        container.tasks.upsert(task)

        loaded = client.get(f"/api/tasks/{task.id}")
        assert loaded.status_code == 200
        timing = loaded.json()["task"]["timing_summary"]
        assert timing["is_running"] is False
        assert timing["active_run_started_at"] is None
        assert timing["total_completed_seconds"] == 0.0
        assert timing["first_started_at"] is None
        assert timing["last_finished_at"] == "2026-02-21T12:01:00+00:00"


def test_get_task_timing_summary_handles_mixed_timezone_formats(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    container = Container(tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"title": "Timing mixed timezone formats"}).json()["task"]
        task = container.tasks.get(created["id"])
        assert task is not None

        aware_run = RunRecord(
            task_id=task.id,
            status="done",
            started_at="2026-02-21T10:00:00Z",
            finished_at="2026-02-21T10:00:30+00:00",
        )
        naive_run = RunRecord(
            task_id=task.id,
            status="done",
            started_at="2026-02-21T10:01:00",
            finished_at="2026-02-21T10:02:00",
        )
        active_naive = RunRecord(
            task_id=task.id,
            status="in_progress",
            started_at="2026-02-21T10:03:00",
            finished_at=None,
        )
        container.runs.upsert(aware_run)
        container.runs.upsert(naive_run)
        container.runs.upsert(active_naive)
        task.run_ids = [aware_run.id, naive_run.id, active_naive.id]
        container.tasks.upsert(task)

        loaded = client.get(f"/api/tasks/{task.id}")
        assert loaded.status_code == 200
        timing = loaded.json()["task"]["timing_summary"]
        assert timing["is_running"] is True
        assert timing["active_run_started_at"] == "2026-02-21T10:03:00+00:00"
        assert timing["total_completed_seconds"] == 90.0
        assert timing["first_started_at"] == "2026-02-21T10:00:00+00:00"
        assert timing["last_finished_at"] == "2026-02-21T10:02:00+00:00"


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


def test_delete_terminal_task_cleans_relationship_references(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    container = Container(tmp_path)

    task_to_delete = Task(title="Terminal delete target", status="done")
    child = Task(title="Child task", status="done", parent_id=task_to_delete.id)
    predecessor = Task(title="Predecessor", status="done", blocks=[task_to_delete.id], children_ids=[task_to_delete.id])
    dependent = Task(title="Dependent", status="done", blocked_by=[task_to_delete.id])
    parent_ref = Task(title="Parent ref", status="done", children_ids=[task_to_delete.id])

    container.tasks.upsert(task_to_delete)
    container.tasks.upsert(child)
    container.tasks.upsert(predecessor)
    container.tasks.upsert(dependent)
    container.tasks.upsert(parent_ref)

    with TestClient(app) as client:
        resp = client.delete(f"/api/tasks/{task_to_delete.id}")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["deleted"] is True
        assert payload["task_id"] == task_to_delete.id

    fresh = Container(tmp_path)
    assert fresh.tasks.get(task_to_delete.id) is None
    assert fresh.tasks.get(child.id) is not None
    assert fresh.tasks.get(child.id).parent_id is None
    assert task_to_delete.id not in (fresh.tasks.get(predecessor.id).blocks or [])
    assert task_to_delete.id not in (fresh.tasks.get(predecessor.id).children_ids or [])
    assert task_to_delete.id not in (fresh.tasks.get(dependent.id).blocked_by or [])
    assert task_to_delete.id not in (fresh.tasks.get(parent_ref.id).children_ids or [])


def test_delete_non_terminal_task_rejected(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"title": "Queued task", "status": "queued"}).json()["task"]
        resp = client.delete(f"/api/tasks/{created['id']}")
        assert resp.status_code == 400
        assert "Only terminal tasks" in resp.json()["detail"]


def test_clear_tasks_archives_state_and_reinitializes_board(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"title": "Archive me", "status": "queued"}).json()["task"]

        clear_resp = client.post("/api/tasks/clear")
        assert clear_resp.status_code == 200
        body = clear_resp.json()
        assert body["cleared"] is True
        archived_to = str(body["archived_to"] or "")
        assert archived_to
        assert "Archived previous runtime state to" in body["message"]

        archived_path = Path(archived_to)
        assert archived_path.exists()
        assert archived_path.parent == tmp_path / ".agent_orchestrator_archive"
        archived_tasks = (archived_path / "tasks.yaml").read_text(encoding="utf-8")
        assert created["id"] in archived_tasks

        board = client.get("/api/tasks/board")
        assert board.status_code == 200
        columns = board.json()["columns"]
        assert columns["backlog"] == []
        assert columns["queued"] == []
        assert columns["in_progress"] == []
        assert columns["in_review"] == []
        assert columns["blocked"] == []
        assert columns["done"] == []
        assert columns["cancelled"] == []

        fresh_tasks = (tmp_path / ".agent_orchestrator" / "tasks.yaml").read_text(encoding="utf-8")
        assert created["id"] not in fresh_tasks


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
        actions_after_changes = request_changes.json()["task"]["human_review_actions"]
        assert len(actions_after_changes) == 1
        assert actions_after_changes[0]["action"] == "request_changes"
        assert actions_after_changes[0]["guidance"] == "Please adjust tests"

        client.post(f"/api/tasks/{task['id']}/run")
        approved = client.post(f"/api/review/{task['id']}/approve", json={})
        assert approved.status_code == 200
        assert approved.json()["task"]["status"] == "done"
        execution_summary = approved.json()["task"].get("execution_summary") or {}
        assert execution_summary.get("run_status") == "done"
        actions_after_approve = approved.json()["task"]["human_review_actions"]
        assert len(actions_after_approve) == 2
        assert actions_after_approve[0]["action"] == "request_changes"
        assert actions_after_approve[1]["action"] == "approve"

        container = app.state.containers[str(tmp_path.resolve())]
        latest_run = None
        task_after = container.tasks.get(task["id"])
        assert task_after is not None
        for run_id in reversed(task_after.run_ids):
            latest_run = container.runs.get(run_id)
            if latest_run:
                break
        assert latest_run is not None
        assert latest_run.status == "done"
        assert latest_run.finished_at is not None


def test_task_execution_summary_prefers_terminal_run_for_done_task(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post(
            "/api/tasks",
            json={"title": "Summary run selection", "status": "backlog", "approval_mode": "auto_approve"},
        ).json()["task"]
        run_resp = client.post(f"/api/tasks/{created['id']}/run")
        assert run_resp.status_code == 200
        assert run_resp.json()["task"]["status"] == "done"

        container = app.state.containers[str(tmp_path.resolve())]
        task = container.tasks.get(created["id"])
        assert task is not None
        assert task.run_ids
        completed_run = container.runs.get(task.run_ids[-1])
        assert completed_run is not None
        assert completed_run.status == "done"

        stale_run = RunRecord(
            task_id=task.id,
            status="in_progress",
            started_at=now_iso(),
            finished_at=None,
            summary="Stale in-progress run",
            steps=[{"step": "implement", "status": "ok", "summary": "stale step", "started_at": now_iso(), "ts": now_iso()}],
        )
        container.runs.upsert(stale_run)
        task.run_ids.append(stale_run.id)
        container.tasks.upsert(task)

        task_detail = client.get(f"/api/tasks/{task.id}")
        assert task_detail.status_code == 200
        execution_summary = task_detail.json()["task"].get("execution_summary") or {}
        assert execution_summary.get("run_id") == completed_run.id
        assert execution_summary.get("run_status") == "done"


def test_task_execution_summary_reconciles_single_stale_run_for_done_task(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post(
            "/api/tasks",
            json={"title": "Single stale run", "status": "done", "approval_mode": "auto_approve"},
        ).json()["task"]

        container = app.state.containers[str(tmp_path.resolve())]
        task = container.tasks.get(created["id"])
        assert task is not None

        stale_run = RunRecord(
            task_id=task.id,
            status="in_progress",
            started_at=now_iso(),
            finished_at=None,
            summary="Stale in-progress run",
            steps=[{"step": "implement", "status": "ok", "summary": "stale step", "started_at": now_iso(), "ts": now_iso()}],
        )
        container.runs.upsert(stale_run)
        task.status = "done"
        task.run_ids = [stale_run.id]
        container.tasks.upsert(task)

        task_detail = client.get(f"/api/tasks/{task.id}")
        assert task_detail.status_code == 200
        execution_summary = task_detail.json()["task"].get("execution_summary") or {}
        assert execution_summary.get("run_id") == stale_run.id
        assert execution_summary.get("run_status") == "done"


def test_get_task_plan(tmp_path: Path) -> None:
    """GET /api/tasks/{id}/plan returns revision-based plan history."""
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"title": "Plan query test"}).json()["task"]
        first_revision_id = _create_plan_revision(client, task["id"], "Plan A")
        _create_plan_revision(client, task["id"], "Plan B", parent_revision_id=first_revision_id)

        resp = client.get(f"/api/tasks/{task['id']}/plan")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["revisions"]) == 2
        assert body["latest_revision_id"] == body["revisions"][-1]["id"]
        assert body["revisions"][-1]["content"] == "Plan B"

        # Non-existent task → 404
        resp_404 = client.get("/api/tasks/nonexistent/plan")
        assert resp_404.status_code == 404

        # Task with no revisions → empty list
        task_no_plan = client.post("/api/tasks", json={"title": "No plan"}).json()["task"]
        resp_empty = client.get(f"/api/tasks/{task_no_plan['id']}/plan")
        assert resp_empty.status_code == 200
        assert resp_empty.json()["revisions"] == []
        assert resp_empty.json()["latest_revision_id"] is None


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
                    "scripted_generated_tasks": [
                        {"title": "Login endpoint", "task_type": "feature", "priority": "P1"},
                        {"title": "Session store", "task_type": "feature", "priority": "P2"},
                    ],
                },
            },
        ).json()["task"]
        _create_plan_revision(client, task["id"], "Build auth")

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
                    "scripted_steps": {
                        "generate_tasks": {"status": "ok", "summary": "No decomposition possible."}
                    },
                },
            },
        ).json()["task"]
        _create_plan_revision(client, task["id"], "Build auth")

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
                    "scripted_steps": {
                        "plan_refine": {"status": "ok", "summary": "Refined plan with rollout"}
                    },
                },
            },
        ).json()["task"]
        _create_plan_revision(client, task["id"], "Initial plan")

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


def test_initiative_plan_refine_uses_initiative_refine_step(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Initiative refine",
                "task_type": "initiative_plan",
                "metadata": {
                    "scripted_steps": {
                        "initiative_plan_refine": {"status": "ok", "summary": "Refined initiative plan with updated sequencing"}
                    },
                },
            },
        ).json()["task"]
        _create_plan_revision(client, task["id"], "Initial initiative plan")

        initial_doc = client.get(f"/api/tasks/{task['id']}/plan").json()
        base_revision_id = initial_doc["latest_revision_id"]

        queued = client.post(
            f"/api/tasks/{task['id']}/plan/refine",
            json={"base_revision_id": base_revision_id, "feedback": "Adjust sequencing by risk"},
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
        doc = client.get(f"/api/tasks/{task['id']}/plan").json()
        refined = next(item for item in doc["revisions"] if item["id"] == result_revision_id)
        assert refined["step"] == "initiative_plan_refine"


def test_plan_refine_failure_path(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post(
            "/api/tasks",
            json={
                "title": "Broken refine",
                "metadata": {
                    "scripted_steps": {"plan_refine": {"status": "error", "summary": "worker failed"}},
                },
            },
        ).json()["task"]
        _create_plan_revision(client, task["id"], "Base")

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
                    "scripted_steps": {"plan_refine": {"status": "ok", "summary": "Refined output"}},
                },
            },
        ).json()["task"]
        _create_plan_revision(client, task["id"], "Base plan")

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
                    "scripted_generated_tasks": [{"title": "From selected source", "task_type": "feature"}],
                },
            },
        ).json()["task"]
        _create_plan_revision(client, task["id"], "Base plan")

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


# ---------------------------------------------------------------------------
# Workdoc API
# ---------------------------------------------------------------------------


def test_get_workdoc_returns_content(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"title": "Workdoc task"}).json()["task"]
        task_id = task["id"]

        # Manually init the workdoc via the orchestrator service
        from agent_orchestrator.runtime.orchestrator.service import OrchestratorService
        from agent_orchestrator.runtime.events.bus import EventBus
        from agent_orchestrator.runtime.storage.container import Container

        container = Container(tmp_path)
        bus = EventBus(container.events, container.project_id)
        svc = OrchestratorService(container, bus, worker_adapter=DefaultWorkerAdapter())
        t = container.tasks.get(task_id)
        assert t is not None
        svc._init_workdoc(t, tmp_path)

        resp = client.get(f"/api/tasks/{task_id}/workdoc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert data["exists"] is True
        assert "Workdoc task" in data["content"]


def test_get_workdoc_returns_409_when_missing(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"title": "No workdoc", "status": "backlog"}).json()["task"]
        task_id = task["id"]

        resp = client.get(f"/api/tasks/{task_id}/workdoc")
        assert resp.status_code == 409
        assert "Missing required workdoc" in resp.json()["detail"]


def test_get_workdoc_404_for_unknown_task(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        resp = client.get("/api/tasks/nonexistent-id/workdoc")
        assert resp.status_code == 404
