#!/usr/bin/env python3
"""
Feature PRD Runner Tests
========================
"""

import sys
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_prd_runner import runner


class TestFeaturePrdRunner:
    """Basic tests for prompt composition and helpers."""

    def __init__(self) -> None:
        self.test_dir: Path | None = None
        self.passed = 0
        self.failed = 0

    def setup(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="feature_prd_runner_test_"))

    def teardown(self) -> None:
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def assert_true(self, condition: bool, message: str) -> None:
        if condition:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            print(f"  ✗ {message}")

    def test_phase_prompt_includes_readme_and_resume(self) -> None:
        print("\n[Test] Phase prompt includes README and resume prompt")

        prompt = runner._build_phase_prompt(
            prd_path=Path("/tmp/prd.md"),
            phase={"id": "phase-1", "description": "Do things"},
            task={"id": "phase-1", "context": []},
            events_path=Path("/tmp/events.ndjson"),
            progress_path=Path("/tmp/progress.json"),
            run_id="run-1",
            user_prompt="Prioritize error handling",
        )

        self.assert_true("README.md" in prompt, "README update requirement present")
        self.assert_true("Special instructions" in prompt, "Resume prompt included")

    def test_review_prompt_mentions_requirements(self) -> None:
        print("\n[Test] Review prompt mentions requirements")

        prompt = runner._build_review_prompt(
            phase={"id": "phase-1", "acceptance_criteria": ["AC1"]},
            review_path=Path("/tmp/review.json"),
            prd_path=Path("/tmp/prd.md"),
            user_prompt=None,
        )

        self.assert_true("PRD:" in prompt, "PRD path included")
        self.assert_true("acceptance criteria" in prompt.lower(), "Acceptance criteria included")
        self.assert_true("Review instructions" in prompt, "Review instructions included")

    def test_plan_prompt_includes_resume_prompt(self) -> None:
        print("\n[Test] Plan prompt includes resume prompt")

        prompt = runner._build_plan_prompt(
            prd_path=Path("/tmp/prd.md"),
            phase_plan_path=Path("/tmp/phase_plan.yaml"),
            task_queue_path=Path("/tmp/task_queue.yaml"),
            events_path=Path("/tmp/events.ndjson"),
            progress_path=Path("/tmp/progress.json"),
            run_id="run-1",
            user_prompt="Focus on migrations",
        )

        self.assert_true("Special instructions" in prompt, "Resume prompt included")

    def run_all(self) -> int:
        print("\n" + "=" * 70)
        print("FEATURE PRD RUNNER TESTS")
        print("=" * 70)

        try:
            self.setup()

            self.test_phase_prompt_includes_readme_and_resume()
            self.test_review_prompt_mentions_requirements()
            self.test_plan_prompt_includes_resume_prompt()

        finally:
            self.teardown()

        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")

        if self.failed == 0:
            print("\n✅ All tests passed!")
            return 0

        print(f"\n❌ {self.failed} test(s) failed")
        return 1


def main() -> int:
    suite = TestFeaturePrdRunner()
    return suite.run_all()


if __name__ == "__main__":
    exit(main())
