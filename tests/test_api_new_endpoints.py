#!/usr/bin/env python3
"""Tests for the new API endpoints (Batches 1-6).

Covers: explain, inspect, dry-run, doctor, workers, correct, require,
task-logs, metrics-export, and advanced run options.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip(
        "FastAPI TestClient requires httpx. Install with: pip install -e '.[test,server]'",
        allow_module_level=True,
    )

from feature_prd_runner.server.api import create_app


# --------------- Fixtures ---------------


@pytest.fixture
def test_project(tmp_path: Path):
    """Create a test project with .prd_runner state and tasks."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    state_dir = project_dir / ".prd_runner"
    state_dir.mkdir()
    (state_dir / "runs").mkdir()
    (state_dir / "artifacts").mkdir()

    # run_state with an active run
    (state_dir / "run_state.yaml").write_text(
        "status: idle\nrun_id: test-run-1\n"
    )

    # task_queue with blocked and ready tasks
    (state_dir / "task_queue.yaml").write_text(
        """tasks:
  - id: task-001
    type: feature
    phase_id: phase-1
    step: implement
    lifecycle: waiting_human
    status: blocked
    worker_attempts: 3
    last_error: "Test failure: assert False"
    last_error_type: test_failed
    block_reason: "Tests keep failing"
    context:
      - src/main.py
    human_blocking_issues:
      - "Tests are not passing"
    human_next_steps:
      - "Review the test output"
  - id: task-002
    type: feature
    phase_id: phase-1
    step: plan_impl
    lifecycle: ready
    status: ready
    worker_attempts: 0
  - id: task-003
    type: feature
    phase_id: phase-1
    step: verify
    lifecycle: done
    status: done
    worker_attempts: 1
"""
    )

    (state_dir / "phase_plan.yaml").write_text(
        """phases:
  - id: phase-1
    name: Core
    description: Core feature
    status: running
    deps: []
"""
    )

    # Create a run directory with progress.json
    run_dir = state_dir / "runs" / "test-run-1"
    run_dir.mkdir(parents=True)
    progress = {
        "run_id": "test-run-1",
        "task_id": "task-001",
        "phase": "phase-1",
        "step": "implement",
        "status": "running",
        "started_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T01:00:00Z",
        "messages_from_human": [],
        "messages_to_human": [],
    }
    (run_dir / "progress.json").write_text(json.dumps(progress))
    (run_dir / "stdout.log").write_text("line1\nline2\nline3\n")
    (run_dir / "stderr.log").write_text("err1\nerr2\n")

    # Initialize git repo for doctor checks
    import subprocess
    subprocess.run(["git", "init"], cwd=project_dir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=project_dir, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=project_dir, capture_output=True,
    )

    return project_dir


@pytest.fixture
def client(test_project: Path):
    """Create a FastAPI test client."""
    app = create_app(project_dir=test_project, enable_cors=True)
    return TestClient(app)


# --------------- Batch 1: Explain + Inspect ---------------


class TestExplainEndpoint:
    """Tests for GET /api/tasks/{task_id}/explain."""

    def test_explain_blocked_task(self, client: TestClient):
        response = client.get("/api/tasks/task-001/explain")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-001"
        assert "blocked" in data["explanation"].lower() or "waiting_human" in data["explanation"].lower()
        assert data["is_blocked"] is True

    def test_explain_ready_task(self, client: TestClient):
        response = client.get("/api/tasks/task-002/explain")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-002"
        assert "not blocked" in data["explanation"].lower() or data["is_blocked"] is False

    def test_explain_not_found(self, client: TestClient):
        response = client.get("/api/tasks/nonexistent/explain")
        assert response.status_code == 200
        data = response.json()
        assert "not found" in data["explanation"].lower()

    def test_explain_no_state(self, client: TestClient, tmp_path: Path):
        app = create_app(project_dir=tmp_path / "nope")
        c = TestClient(app)
        response = c.get("/api/tasks/x/explain")
        assert response.status_code == 404


class TestTraceEndpoint:
    """Tests for GET /api/tasks/{task_id}/trace."""

    def test_trace_no_events(self, client: TestClient):
        response = client.get("/api/tasks/task-001/trace")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_trace_with_events(self, client: TestClient, test_project: Path):
        """When events.jsonl has matching events, they are returned."""
        events_path = test_project / ".prd_runner" / "artifacts" / "events.jsonl"
        events_path.write_text(
            '{"event_type":"worker_started","task_id":"task-001","timestamp":"2025-01-01T00:00:00Z"}\n'
            '{"event_type":"worker_failed","task_id":"task-001","timestamp":"2025-01-01T00:05:00Z","error_type":"test_failed"}\n'
            '{"event_type":"worker_started","task_id":"task-002","timestamp":"2025-01-01T00:10:00Z"}\n'
        )
        response = client.get("/api/tasks/task-001/trace")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["event_type"] == "worker_started"
        assert data[1]["event_type"] == "worker_failed"

    def test_trace_with_limit(self, client: TestClient, test_project: Path):
        events_path = test_project / ".prd_runner" / "artifacts" / "events.jsonl"
        lines = [f'{{"event_type":"step_{i}","task_id":"task-001","timestamp":"2025-01-01T00:{i:02d}:00Z"}}\n' for i in range(10)]
        events_path.write_text("".join(lines))
        response = client.get("/api/tasks/task-001/trace?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_trace_no_state(self, client: TestClient, tmp_path: Path):
        app = create_app(project_dir=tmp_path / "nope")
        c = TestClient(app)
        response = c.get("/api/tasks/x/trace")
        assert response.status_code == 200
        assert response.json() == []


class TestInspectEndpoint:
    """Tests for GET /api/tasks/{task_id}/inspect."""

    def test_inspect_task(self, client: TestClient):
        response = client.get("/api/tasks/task-001/inspect")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-001"
        assert data["lifecycle"] == "waiting_human"
        assert data["step"] == "implement"
        assert data["worker_attempts"] == 3
        assert data["last_error"] == "Test failure: assert False"
        assert data["last_error_type"] == "test_failed"
        assert isinstance(data["context"], list)
        assert isinstance(data["metadata"], dict)

    def test_inspect_not_found(self, client: TestClient):
        response = client.get("/api/tasks/nonexistent/inspect")
        assert response.status_code == 404


# --------------- Batch 2: Dry-Run + Doctor ---------------


class TestDryRunEndpoint:
    """Tests for GET /api/dry-run."""

    def test_dry_run_with_ready_task(self, client: TestClient):
        response = client.get("/api/dry-run")
        assert response.status_code == 200
        data = response.json()
        assert "project_dir" in data
        assert "state_dir" in data
        assert data["next"] is not None
        assert data["next"]["action"] == "process_task"
        assert data["next"]["task_id"] == "task-002"
        assert data["would_spawn_codex"] is True

    def test_dry_run_no_state(self, client: TestClient, tmp_path: Path):
        app = create_app(project_dir=tmp_path / "no_state")
        c = TestClient(app)
        response = c.get("/api/dry-run")
        assert response.status_code == 200
        data = response.json()
        assert data["next"]["action"] == "init"
        assert len(data["warnings"]) > 0


class TestDoctorEndpoint:
    """Tests for GET /api/doctor."""

    def test_doctor_basic(self, client: TestClient):
        response = client.get("/api/doctor")
        assert response.status_code == 200
        data = response.json()
        assert "checks" in data
        assert "warnings" in data
        assert "errors" in data
        assert "exit_code" in data
        assert data["checks"]["state_dir"]["status"] == "pass"
        assert data["checks"]["git"]["status"] == "pass"

    def test_doctor_with_codex_check(self, client: TestClient):
        response = client.get("/api/doctor?check_codex=true")
        assert response.status_code == 200
        data = response.json()
        assert "codex" in data["checks"]

    def test_doctor_no_state(self, client: TestClient, tmp_path: Path):
        app = create_app(project_dir=tmp_path / "no_state")
        c = TestClient(app)
        response = c.get("/api/doctor")
        assert response.status_code == 200
        data = response.json()
        assert data["checks"]["state_dir"]["status"] == "fail"
        assert data["exit_code"] == 1


# --------------- Batch 3: Workers ---------------


class TestWorkersEndpoint:
    """Tests for GET /api/workers and POST /api/workers/{name}/test."""

    def test_list_workers(self, client: TestClient):
        response = client.get("/api/workers")
        assert response.status_code == 200
        data = response.json()
        assert "default_worker" in data
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_test_worker_not_found(self, client: TestClient):
        response = client.post("/api/workers/nonexistent/test")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["message"].lower()


# --------------- Batch 4: Correct + Require ---------------


class TestCorrectionEndpoint:
    """Tests for POST /api/tasks/{task_id}/correct."""

    def test_send_correction(self, client: TestClient, test_project: Path):
        # Need an active run
        state_dir = test_project / ".prd_runner"
        (state_dir / "run_state.yaml").write_text(
            "status: running\nrun_id: test-run-1\n"
        )

        response = client.post(
            "/api/tasks/task-001/correct",
            json={
                "issue": "Wrong variable name used",
                "file_path": "src/main.py",
                "suggested_fix": "Use 'user_name' instead of 'name'",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "correction" in data["message"].lower()

    def test_send_correction_no_run(self, client: TestClient, test_project: Path):
        # Clear run_id
        state_dir = test_project / ".prd_runner"
        (state_dir / "run_state.yaml").write_text("status: idle\n")

        response = client.post(
            "/api/tasks/task-001/correct",
            json={"issue": "Something wrong"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestRequirementEndpoint:
    """Tests for POST /api/requirements."""

    def test_send_requirement(self, client: TestClient, test_project: Path):
        state_dir = test_project / ".prd_runner"
        (state_dir / "run_state.yaml").write_text(
            "status: running\nrun_id: test-run-1\n"
        )

        response = client.post(
            "/api/requirements",
            json={
                "requirement": "Must support pagination",
                "task_id": "task-001",
                "priority": "high",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["priority"] == "high"

    def test_send_requirement_no_run(self, client: TestClient, test_project: Path):
        state_dir = test_project / ".prd_runner"
        (state_dir / "run_state.yaml").write_text("status: idle\n")

        response = client.post(
            "/api/requirements",
            json={"requirement": "Must be fast"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


# --------------- Batch 5: Task Logs + Metrics Export ---------------


class TestTaskLogsEndpoint:
    """Tests for GET /api/tasks/{task_id}/logs."""

    def test_get_task_logs(self, client: TestClient):
        response = client.get("/api/tasks/task-001/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-001"
        assert isinstance(data["logs"], dict)

    def test_get_task_logs_no_match(self, client: TestClient, tmp_path: Path):
        """Task with no matching run and no active run returns empty logs."""
        # Create a project with no runs at all
        project_dir = tmp_path / "empty_project"
        project_dir.mkdir()
        state_dir = project_dir / ".prd_runner"
        state_dir.mkdir()
        (state_dir / "runs").mkdir()
        (state_dir / "run_state.yaml").write_text("status: idle\n")
        (state_dir / "task_queue.yaml").write_text("tasks: []\n")
        (state_dir / "phase_plan.yaml").write_text("phases: []\n")

        app = create_app(project_dir=project_dir)
        c = TestClient(app)
        response = c.get("/api/tasks/nonexistent/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "nonexistent"
        assert len(data["logs"]) == 0

    def test_get_task_logs_with_step_filter(self, client: TestClient):
        response = client.get("/api/tasks/task-001/logs?step=verify")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-001"


class TestMetricsExportEndpoint:
    """Tests for GET /api/metrics/export."""

    def test_export_csv(self, client: TestClient):
        response = client.get("/api/metrics/export?format=csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "metrics.csv" in response.headers.get("content-disposition", "")
        lines = response.text.strip().split("\n")
        assert len(lines) == 2  # header + values

    def test_export_html(self, client: TestClient):
        response = client.get("/api/metrics/export?format=html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<table" in response.text


# --------------- Batch 6: Advanced Run Options ---------------


class TestAdvancedRunOptions:
    """Tests for extended StartRunRequest fields."""

    def test_start_run_with_advanced_options(self, client: TestClient):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()

            response = client.post(
                "/api/runs/start",
                json={
                    "mode": "full_prd",
                    "content": "# Feature: Test\n\n## Requirements\n1. Test",
                    "language": "python",
                    "reset_state": True,
                    "require_clean": False,
                    "commit_enabled": False,
                    "push_enabled": False,
                    "parallel": True,
                    "max_workers": 5,
                    "ensure_ruff": "fix",
                    "ensure_deps": "install",
                    "ensure_deps_command": "pip install -r requirements.txt",
                    "shift_minutes": 30,
                    "max_task_attempts": 3,
                    "max_review_attempts": 5,
                    "worker": "ollama",
                    "codex_command": "/usr/local/bin/codex exec -",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # Verify the advanced flags were mapped to CLI args
            call_args = mock_popen.call_args
            cmd = call_args[0][0]
            assert "--language" in cmd
            assert "python" in cmd
            assert "--reset-state" in cmd
            assert "--no-require-clean" in cmd
            assert "--no-commit" in cmd
            assert "--no-push" in cmd
            assert "--parallel" in cmd
            assert "--max-workers" in cmd
            assert "5" in cmd
            assert "--ensure-ruff" in cmd
            assert "fix" in cmd
            assert "--ensure-deps" in cmd
            assert "install" in cmd
            assert "--ensure-deps-command" in cmd
            assert "--shift-minutes" in cmd
            assert "30" in cmd
            assert "--max-task-attempts" in cmd
            assert "3" in cmd
            assert "--max-review-attempts" in cmd
            assert "--worker" in cmd
            assert "ollama" in cmd
            assert "--codex-command" in cmd

    def test_start_run_defaults_dont_add_flags(self, client: TestClient):
        """Verify default values don't add unnecessary flags."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()

            response = client.post(
                "/api/runs/start",
                json={
                    "mode": "full_prd",
                    "content": "# Feature: Test\n\n## Requirements\n1. Test",
                },
            )

            assert response.status_code == 200
            call_args = mock_popen.call_args
            cmd = call_args[0][0]
            # Default values should not appear
            assert "--language" not in cmd
            assert "--reset-state" not in cmd
            assert "--no-require-clean" not in cmd
            assert "--parallel" not in cmd
