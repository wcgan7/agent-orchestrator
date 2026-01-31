"""Integration tests for TypeScript verification flow."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from feature_prd_runner.actions import run_verify
from feature_prd_runner.io_utils import _load_data


class TestTypeScriptVerification:
    """Tests for TypeScript project verification."""

    def test_jest_test_failures_extract_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure Jest test failures extract the correct file paths."""
        project_dir = tmp_path / "repo"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        # Create all files referenced in the test output
        (project_dir / "src" / "utils.test.ts").write_text("test('x', () => {})")
        (project_dir / "src" / "auth.test.ts").write_text("test('x', () => {})")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Simulate Jest output with failing tests
        jest_output = """FAIL src/utils.test.ts
  ● Test suite failed to run

    TypeError: Cannot read property 'foo' of undefined

      1 | import { foo } from './utils';
      2 |
    > 3 | test('should work', () => {
        |                           ^

      at Object.<anonymous> (src/utils.test.ts:3:27)

FAIL src/auth.test.ts
  ● login › should validate credentials

    expect(received).toBe(expected)

    Expected: true
    Received: false

      at Object.<anonymous> (src/auth.test.ts:15:18)

Test Suites: 2 failed, 0 passed, 2 total
Tests:       2 failed, 0 passed, 2 total
"""

        def fake_run_command(
            command: str,
            project_dir: Path,
            log_path: Path,
            timeout_seconds: int | None = None,
        ) -> dict[str, object]:
            _ = project_dir
            _ = timeout_seconds
            log_path.write_text(jest_output)
            return {
                "command": command,
                "exit_code": 1,
                "log_path": str(log_path),
                "timed_out": False,
            }

        monkeypatch.setattr(run_verify, "_run_command", fake_run_command)

        event = run_verify.run_verify_action(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            run_dir=run_dir,
            phase={"id": "phase-1"},
            task={"id": "phase-1", "phase_id": "phase-1"},
            run_id="run-1",
            plan_data={"files_to_change": ["src/other.ts"], "new_files": []},
            default_test_command="npm test",
            language="typescript",
            timeout_seconds=10,
        )

        assert event.passed is False
        assert "src/utils.test.ts" in (event.failing_paths or [])
        assert "src/auth.test.ts" in (event.failing_paths or [])

    def test_eslint_errors_trigger_allowlist_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure ESLint errors trigger allowlist expansion."""
        project_dir = tmp_path / "repo"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        # Create all files referenced in the test output
        (project_dir / "src" / "utils.ts").write_text("const x = 1;")
        (project_dir / "src" / "auth.ts").write_text("const y = 2;")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Simulate ESLint output with relative paths (stylish format)
        eslint_output = """src/utils.ts
  1:7   error  'x' is assigned a value but never used  @typescript-eslint/no-unused-vars
  5:1   error  Missing return type                      @typescript-eslint/explicit-function-return-type

src/auth.ts
  10:5  error  Unexpected any                           @typescript-eslint/no-explicit-any

✖ 3 problems (3 errors, 0 warnings)
"""

        def fake_run_command(
            command: str,
            project_dir: Path,
            log_path: Path,
            timeout_seconds: int | None = None,
        ) -> dict[str, object]:
            _ = project_dir
            _ = timeout_seconds
            if "test" in command:
                log_path.write_text("All tests passed\n")
                return {
                    "command": command,
                    "exit_code": 0,
                    "log_path": str(log_path),
                    "timed_out": False,
                }
            log_path.write_text(eslint_output)
            return {
                "command": command,
                "exit_code": 1,
                "log_path": str(log_path),
                "timed_out": False,
            }

        monkeypatch.setattr(run_verify, "_run_command", fake_run_command)

        event = run_verify.run_verify_action(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            run_dir=run_dir,
            phase={"id": "phase-1"},
            task={"id": "phase-1", "phase_id": "phase-1"},
            run_id="run-1",
            plan_data={"files_to_change": ["src/other.ts"], "new_files": []},
            default_test_command="npm test",
            default_lint_command="npx eslint .",
            language="typescript",
            timeout_seconds=10,
        )

        assert event.passed is False
        assert event.needs_allowlist_expansion is True
        # Should extract relative paths from absolute paths
        failing = event.failing_paths or []
        assert any("utils.ts" in p for p in failing)
        assert any("auth.ts" in p for p in failing)

    def test_tsc_type_errors_extract_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure tsc type errors extract file paths correctly."""
        project_dir = tmp_path / "repo"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        (project_dir / "src" / "components").mkdir()
        # Create all files referenced in the test output
        (project_dir / "src" / "utils.ts").write_text("const x: number = 'string';")
        (project_dir / "src" / "auth.ts").write_text("const y = undefined;")
        (project_dir / "src" / "components" / "Button.tsx").write_text("export const Button = {};")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Simulate tsc output
        tsc_output = """src/utils.ts(1,7): error TS2322: Type 'string' is not assignable to type 'number'.
src/auth.ts(25,10): error TS2345: Argument of type 'undefined' is not assignable to type 'string'.
src/components/Button.tsx(42,5): error TS2739: Type '{}' is missing the following properties from type 'Props': onClick, label

Found 3 errors.
"""

        def fake_run_command(
            command: str,
            project_dir: Path,
            log_path: Path,
            timeout_seconds: int | None = None,
        ) -> dict[str, object]:
            _ = project_dir
            _ = timeout_seconds
            if "test" in command or "eslint" in command:
                log_path.write_text("ok\n")
                return {
                    "command": command,
                    "exit_code": 0,
                    "log_path": str(log_path),
                    "timed_out": False,
                }
            log_path.write_text(tsc_output)
            return {
                "command": command,
                "exit_code": 1,
                "log_path": str(log_path),
                "timed_out": False,
            }

        monkeypatch.setattr(run_verify, "_run_command", fake_run_command)

        event = run_verify.run_verify_action(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            run_dir=run_dir,
            phase={"id": "phase-1"},
            task={"id": "phase-1", "phase_id": "phase-1"},
            run_id="run-1",
            plan_data={"files_to_change": ["src/other.ts"], "new_files": []},
            default_test_command="npm test",
            default_lint_command="npx eslint .",
            default_typecheck_command="npx tsc --noEmit",
            language="typescript",
            timeout_seconds=10,
        )

        assert event.passed is False
        failing = event.failing_paths or []
        assert "src/utils.ts" in failing
        assert "src/auth.ts" in failing
        assert "src/components/Button.tsx" in failing

    def test_prettier_format_errors_extract_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure Prettier format errors extract file paths correctly."""
        project_dir = tmp_path / "repo"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        (project_dir / "src" / "components").mkdir()
        (project_dir / "src" / "auth").mkdir()
        # Create all files referenced in the test output
        (project_dir / "src" / "utils.ts").write_text("const x=1")
        (project_dir / "src" / "components" / "Button.tsx").write_text("export const Button = {}")
        (project_dir / "src" / "auth" / "login.ts").write_text("export const login = ()")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Simulate Prettier output
        prettier_output = """Checking formatting...
[warn] src/utils.ts
[warn] src/components/Button.tsx
[warn] src/auth/login.ts
[warn] Code style issues found in 3 files. Run Prettier to fix.
"""

        def fake_run_command(
            command: str,
            project_dir: Path,
            log_path: Path,
            timeout_seconds: int | None = None,
        ) -> dict[str, object]:
            _ = project_dir
            _ = timeout_seconds
            if "test" in command or "eslint" in command or "tsc" in command:
                log_path.write_text("ok\n")
                return {
                    "command": command,
                    "exit_code": 0,
                    "log_path": str(log_path),
                    "timed_out": False,
                }
            log_path.write_text(prettier_output)
            return {
                "command": command,
                "exit_code": 1,
                "log_path": str(log_path),
                "timed_out": False,
            }

        monkeypatch.setattr(run_verify, "_run_command", fake_run_command)

        event = run_verify.run_verify_action(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            run_dir=run_dir,
            phase={"id": "phase-1"},
            task={"id": "phase-1", "phase_id": "phase-1"},
            run_id="run-1",
            plan_data={"files_to_change": ["src/other.ts"], "new_files": []},
            default_test_command="npm test",
            default_lint_command="npx eslint .",
            default_typecheck_command="npx tsc --noEmit",
            default_format_command="npx prettier --check .",
            language="typescript",
            timeout_seconds=10,
        )

        assert event.passed is False
        failing = event.failing_paths or []
        assert "src/utils.ts" in failing
        assert "src/components/Button.tsx" in failing
        assert "src/auth/login.ts" in failing

    def test_vitest_failures_use_jest_parser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure Vitest output (similar to Jest) is parsed correctly."""
        project_dir = tmp_path / "repo"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        # Create all files referenced in the test output
        (project_dir / "src" / "utils.test.ts").write_text("test('x', () => {})")
        (project_dir / "src" / "math.test.ts").write_text("test('y', () => {})")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Vitest output format - uses FAIL lines like Jest
        vitest_output = """FAIL src/utils.test.ts > add > should add numbers
AssertionError: expected 3 to equal 4

  - Expected: 4
  + Received: 3

    at src/utils.test.ts:5:18

FAIL src/math.test.ts > multiply > should multiply numbers
AssertionError: expected 6 to equal 8

    at src/math.test.ts:10:18

Test Files  2 failed
"""

        def fake_run_command(
            command: str,
            project_dir: Path,
            log_path: Path,
            timeout_seconds: int | None = None,
        ) -> dict[str, object]:
            _ = project_dir
            _ = timeout_seconds
            log_path.write_text(vitest_output)
            return {
                "command": command,
                "exit_code": 1,
                "log_path": str(log_path),
                "timed_out": False,
            }

        monkeypatch.setattr(run_verify, "_run_command", fake_run_command)

        event = run_verify.run_verify_action(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            run_dir=run_dir,
            phase={"id": "phase-1"},
            task={"id": "phase-1", "phase_id": "phase-1"},
            run_id="run-1",
            plan_data={"files_to_change": ["src/other.ts"], "new_files": []},
            default_test_command="vitest run",
            language="typescript",
            timeout_seconds=10,
        )

        assert event.passed is False
        failing = event.failing_paths or []
        assert "src/utils.test.ts" in failing
        assert "src/math.test.ts" in failing

    def test_npm_install_ensure_deps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure npm install is used for TypeScript dependency installation."""
        project_dir = tmp_path / "repo"
        project_dir.mkdir()

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        calls: list[str] = []

        def fake_run_command(
            command: str,
            project_dir: Path,
            log_path: Path,
            timeout_seconds: int | None = None,
        ) -> dict[str, object]:
            _ = project_dir
            _ = timeout_seconds
            calls.append(command)
            log_path.write_text("ok\n")
            return {
                "command": command,
                "exit_code": 0,
                "log_path": str(log_path),
                "timed_out": False,
            }

        monkeypatch.setattr(run_verify, "_run_command", fake_run_command)

        event = run_verify.run_verify_action(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            run_dir=run_dir,
            phase={"id": "phase-1"},
            task={"id": "phase-1", "phase_id": "phase-1"},
            run_id="run-1",
            plan_data={"files_to_change": ["src/index.ts"], "new_files": []},
            default_test_command="npm test",
            ensure_deps="install",
            ensure_deps_command="npm install",
            language="typescript",
            timeout_seconds=10,
        )

        assert event.passed is True
        assert calls[0] == "npm install"
        assert calls[1] == "npm test"


class TestLanguageFallback:
    """Tests for language fallback behavior."""

    def test_python_remains_default_when_not_specified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure Python parsing is used when language not specified."""
        project_dir = tmp_path / "repo"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        # Create all files referenced in the test output
        (project_dir / "src" / "utils.py").write_text("import os")
        (project_dir / "src" / "auth.py").write_text("x = 1")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Python ruff output
        ruff_output = """src/utils.py:1:1: F401 `os` imported but unused
src/auth.py:10:5: E501 Line too long
"""

        def fake_run_command(
            command: str,
            project_dir: Path,
            log_path: Path,
            timeout_seconds: int | None = None,
        ) -> dict[str, object]:
            _ = project_dir
            _ = timeout_seconds
            if "test" in command:
                log_path.write_text("ok\n")
                return {
                    "command": command,
                    "exit_code": 0,
                    "log_path": str(log_path),
                    "timed_out": False,
                }
            log_path.write_text(ruff_output)
            return {
                "command": command,
                "exit_code": 1,
                "log_path": str(log_path),
                "timed_out": False,
            }

        monkeypatch.setattr(run_verify, "_run_command", fake_run_command)

        # Don't specify language - should default to python
        event = run_verify.run_verify_action(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            run_dir=run_dir,
            phase={"id": "phase-1"},
            task={"id": "phase-1", "phase_id": "phase-1"},
            run_id="run-1",
            plan_data={"files_to_change": ["src/other.py"], "new_files": []},
            default_test_command="pytest",
            default_lint_command="ruff check .",
            timeout_seconds=10,
        )

        assert event.passed is False
        failing = event.failing_paths or []
        assert "src/utils.py" in failing
        assert "src/auth.py" in failing
