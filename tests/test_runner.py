#!/usr/bin/env python3
"""
Feature PRD Runner Tests
========================
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from feature_prd_runner import runner


def test_phase_prompt_includes_readme_and_resume() -> None:
    prompt = runner._build_phase_prompt(
        prd_path=Path("/tmp/prd.md"),
        phase={"id": "phase-1", "description": "Do things"},
        task={"id": "phase-1", "context": []},
        events_path=Path("/tmp/events.ndjson"),
        progress_path=Path("/tmp/progress.json"),
        run_id="run-1",
        user_prompt="Prioritize error handling",
    )

    assert "README.md" in prompt
    assert "Special instructions" in prompt


def test_review_prompt_mentions_requirements() -> None:
    prompt = runner._build_review_prompt(
        phase={"id": "phase-1", "acceptance_criteria": ["AC1"]},
        review_path=Path("/tmp/review.json"),
        prd_path=Path("/tmp/prd.md"),
        prd_text="## Requirements\nREQ-1: Do thing\n",
        prd_truncated=False,
        prd_markers=["Requirements", "REQ-1"],
        user_prompt=None,
    )

    assert "PRD:" in prompt
    assert "acceptance criteria" in prompt.lower()
    assert "acceptance_criteria_checklist" in prompt
    assert "spec_summary" in prompt
    assert "changed_files" in prompt
    assert "Diff (from coordinator)" in prompt
    assert "Diffstat (from coordinator)" in prompt
    assert "Git status (from coordinator)" in prompt
    assert "design_assessment" in prompt
    assert "architecture_checklist" in prompt
    assert "spec_traceability" in prompt
    assert "logic_risks" in prompt
    assert "Review instructions" in prompt


def test_plan_prompt_includes_resume_prompt() -> None:
    prompt = runner._build_plan_prompt(
        prd_path=Path("/tmp/prd.md"),
        phase_plan_path=Path("/tmp/phase_plan.yaml"),
        task_queue_path=Path("/tmp/task_queue.yaml"),
        events_path=Path("/tmp/events.ndjson"),
        progress_path=Path("/tmp/progress.json"),
        run_id="run-1",
        user_prompt="Focus on migrations",
    )

    assert "Special instructions" in prompt


def test_read_progress_blocking_issues(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "human_blocking_issues": ["Need secrets"],
                "human_next_steps": ["Add secrets to env"],
            }
        )
    )

    issues, steps = runner._read_progress_human_blockers(
        progress_path, expected_run_id="run-1"
    )

    assert issues == ["Need secrets"]
    assert steps == ["Add secrets to env"]

    issues, steps = runner._read_progress_human_blockers(
        progress_path, expected_run_id="run-2"
    )

    assert issues == []
    assert steps == []
