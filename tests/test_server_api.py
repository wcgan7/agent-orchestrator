#!/usr/bin/env python3
"""Test the FastAPI server endpoints, especially the run launcher.

To run these tests, install with:
    pip install -e ".[test,server]"

Requirements:
- httpx (for FastAPI TestClient)
- anthropic (mocked in tests)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import after path setup
try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip(
        "FastAPI TestClient requires httpx. Install with: pip install -e '.[test,server]'",
        allow_module_level=True,
    )

from feature_prd_runner.server.api import create_app


@pytest.fixture
def test_project(tmp_path: Path):
    """Create a test project directory with .prd_runner state."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create .prd_runner directory structure
    state_dir = project_dir / ".prd_runner"
    state_dir.mkdir()
    (state_dir / "runs").mkdir()
    (state_dir / "artifacts").mkdir()

    # Create minimal state files
    run_state = {
        "status": "idle",
        "run_id": "test-run-1",
    }
    (state_dir / "run_state.yaml").write_text(f"status: idle\nrun_id: test-run-1\n")

    task_queue = {"tasks": []}
    (state_dir / "task_queue.yaml").write_text("tasks: []\n")

    phase_plan = {"phases": []}
    (state_dir / "phase_plan.yaml").write_text("phases: []\n")

    return project_dir


@pytest.fixture
def client(test_project: Path):
    """Create a FastAPI test client."""
    app = create_app(project_dir=test_project, enable_cors=True)
    return TestClient(app)


def test_start_run_full_prd_mode(client: TestClient, test_project: Path):
    """Test starting a run with full PRD mode."""
    prd_content = """# Feature: Test Feature

## Overview
This is a test feature.

## Requirements
1. Requirement 1
2. Requirement 2

## Success Criteria
- Works correctly
"""

    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        response = client.post(
            "/api/runs/start",
            json={
                "mode": "full_prd",
                "content": prd_content,
                "test_command": "npm test",
                "build_command": "npm run build",
                "verification_profile": "none",
                "auto_approve_plans": False,
                "auto_approve_changes": False,
                "auto_approve_commits": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["run_id"] is not None
        assert data["prd_path"] is not None

        # Verify PRD was saved
        prd_path = Path(data["prd_path"])
        assert prd_path.exists()
        assert prd_content in prd_path.read_text()

        # Verify subprocess was called
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd_args = call_args[0][0]

        assert "feature-prd-runner" in cmd_args
        assert "run" in cmd_args
        assert "--prd-file" in cmd_args
        assert str(prd_path) in cmd_args
        assert "--test-command" in cmd_args
        assert "npm test" in cmd_args
        assert "--build-cmd" in cmd_args
        assert "npm run build" in cmd_args


def test_start_run_quick_prompt_mode(client: TestClient, test_project: Path):
    """Test starting a run with quick prompt mode (PRD generation)."""
    prompt = "Add a user profile page with avatar upload"

    # Mock the Anthropic API call
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="# Feature: User Profile\n\nGenerated PRD...")]

    with patch("subprocess.Popen") as mock_popen, patch(
        "anthropic.Anthropic"
    ) as mock_anthropic:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        response = client.post(
            "/api/runs/start",
            json={
                "mode": "quick_prompt",
                "content": prompt,
                "test_command": None,
                "build_command": None,
                "verification_profile": "none",
                "auto_approve_plans": False,
                "auto_approve_changes": False,
                "auto_approve_commits": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["run_id"] is not None
        assert data["prd_path"] is not None

        # Verify PRD generation was called
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert prompt in call_kwargs["messages"][0]["content"]

        # Verify generated PRD was saved
        prd_path = Path(data["prd_path"])
        assert prd_path.exists()
        assert "Generated PRD" in prd_path.read_text()

        # Verify subprocess was called
        mock_popen.assert_called_once()


def test_start_run_invalid_mode(client: TestClient):
    """Test starting a run with invalid mode."""
    response = client.post(
        "/api/runs/start",
        json={
            "mode": "invalid_mode",
            "content": "Some content",
            "verification_profile": "none",
        },
    )

    assert response.status_code == 200  # API returns 200 with success=false
    data = response.json()
    assert data["success"] is False
    assert "Invalid mode" in data["message"]


def test_start_run_empty_content(client: TestClient):
    """Test that empty content is handled (frontend validates, but backend should handle gracefully)."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        # Empty content should still work - it will create an empty PRD
        response = client.post(
            "/api/runs/start",
            json={
                "mode": "full_prd",
                "content": "",
                "verification_profile": "none",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Backend doesn't validate empty content, that's frontend's job
        assert data["success"] is True


def test_start_run_prd_generation_failure(client: TestClient):
    """Test handling of PRD generation failures."""
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        response = client.post(
            "/api/runs/start",
            json={
                "mode": "quick_prompt",
                "content": "Add a feature",
                "verification_profile": "none",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Failed to generate PRD" in data["message"]


def test_start_run_with_verification_profile(client: TestClient, test_project: Path):
    """Test starting a run with python verification profile."""
    prd_content = "# Feature: Test"

    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        response = client.post(
            "/api/runs/start",
            json={
                "mode": "full_prd",
                "content": prd_content,
                "verification_profile": "python",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify verification profile was passed
        call_args = mock_popen.call_args[0][0]
        assert "--verification-profile" in call_args
        assert "python" in call_args


def test_start_run_subprocess_failure(client: TestClient, test_project: Path):
    """Test handling of subprocess spawn failures."""
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.side_effect = Exception("Failed to spawn process")

        response = client.post(
            "/api/runs/start",
            json={
                "mode": "full_prd",
                "content": "# Test",
                "verification_profile": "none",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Failed to start run" in data["message"]


def test_prd_file_saved_in_correct_location(client: TestClient, test_project: Path):
    """Test that generated PRDs are saved in the correct directory."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        response = client.post(
            "/api/runs/start",
            json={
                "mode": "full_prd",
                "content": "# Feature",
                "verification_profile": "none",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify PRD is in generated_prds directory
        prd_path = Path(data["prd_path"])
        assert prd_path.parent.name == "generated_prds"
        assert prd_path.parent.parent.name == ".prd_runner"
        assert prd_path.name.startswith("prd_")
        assert prd_path.suffix == ".md"
