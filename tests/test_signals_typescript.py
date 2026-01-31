"""Tests for TypeScript/JavaScript signal extraction functions."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_prd_runner.signals import (
    extract_jest_failed_test_files,
    extract_eslint_repo_paths,
    extract_tsc_repo_paths,
    extract_prettier_repo_paths,
    extract_js_stacktrace_repo_paths,
)


class TestExtractJestFailedTestFiles:
    """Tests for extract_jest_failed_test_files function."""

    def test_extracts_fail_lines(self, tmp_path: Path) -> None:
        """Extract failing test paths from FAIL lines."""
        # Create test files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.test.ts").write_text("test")
        (tmp_path / "__tests__").mkdir()
        (tmp_path / "__tests__" / "auth.test.ts").write_text("test")

        output = """
PASS src/other.test.ts
FAIL src/utils.test.ts
  ● adds numbers correctly
FAIL __tests__/auth.test.ts
  ● login should work
"""
        paths = extract_jest_failed_test_files(output, tmp_path)
        assert "src/utils.test.ts" in paths
        assert "__tests__/auth.test.ts" in paths

    def test_extracts_from_stack_traces(self, tmp_path: Path) -> None:
        """Extract file paths from Jest stack traces."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "services").mkdir(parents=True)
        (tmp_path / "src" / "services" / "auth.ts").write_text("test")

        output = """
Error: Expected value to be truthy
    at Object.<anonymous> (src/services/auth.ts:42:15)
    at processTicksAndRejections (node:internal/process/task_queues:95:5)
"""
        paths = extract_jest_failed_test_files(output, tmp_path)
        assert "src/services/auth.ts" in paths

    def test_extracts_error_locations(self, tmp_path: Path) -> None:
        """Extract file paths from error location lines."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.ts").write_text("test")

        output = """
● Test suite failed to run

    src/utils.ts:42:5 - error TS2322: Type 'string' is not assignable
"""
        paths = extract_jest_failed_test_files(output, tmp_path)
        assert "src/utils.ts" in paths

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        """Skip paths from node_modules."""
        output = """
FAIL node_modules/some-pkg/test.ts
    at Object.<anonymous> (node_modules/jest/lib/runner.ts:25:18)
"""
        paths = extract_jest_failed_test_files(output, tmp_path)
        assert len(paths) == 0

    def test_handles_tsx_files(self, tmp_path: Path) -> None:
        """Handle .tsx extension for React tests."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "Button.test.tsx").write_text("test")

        output = """
FAIL src/Button.test.tsx
  ● renders correctly
"""
        paths = extract_jest_failed_test_files(output, tmp_path)
        assert "src/Button.test.tsx" in paths

    def test_empty_input(self, tmp_path: Path) -> None:
        """Handle empty input."""
        assert extract_jest_failed_test_files("", tmp_path) == []
        assert extract_jest_failed_test_files(None, tmp_path) == []  # type: ignore


class TestExtractEslintRepoPaths:
    """Tests for extract_eslint_repo_paths function."""

    def test_extracts_file_headers_stylish(self, tmp_path: Path) -> None:
        """Extract paths from stylish formatter output."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.ts").write_text("test")
        (tmp_path / "src" / "index.ts").write_text("test")

        output = """
src/utils.ts
  12:5   error  'foo' is defined but never used  @typescript-eslint/no-unused-vars
  25:10  warning  Unexpected console statement   no-console

src/index.ts
  8:1  error  Missing return type  @typescript-eslint/explicit-function-return-type
"""
        paths = extract_eslint_repo_paths(output, tmp_path)
        assert "src/utils.ts" in paths
        assert "src/index.ts" in paths

    def test_extracts_compact_format(self, tmp_path: Path) -> None:
        """Extract paths from compact formatter output."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.ts").write_text("test")

        output = """
src/utils.ts:12:5: 'foo' is defined but never used @typescript-eslint/no-unused-vars
src/utils.ts:25:10: Unexpected console statement no-console
"""
        paths = extract_eslint_repo_paths(output, tmp_path)
        assert "src/utils.ts" in paths

    def test_handles_absolute_paths(self, tmp_path: Path) -> None:
        """Handle absolute paths in output."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.ts").write_text("test")

        output = f"""
{tmp_path}/src/utils.ts
  12:5   error  'foo' is defined but never used
"""
        paths = extract_eslint_repo_paths(output, tmp_path)
        assert "src/utils.ts" in paths

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        """Skip node_modules paths."""
        output = """
node_modules/some-pkg/index.ts
  1:1  error  Something wrong
"""
        paths = extract_eslint_repo_paths(output, tmp_path)
        assert len(paths) == 0

    def test_handles_jsx_tsx(self, tmp_path: Path) -> None:
        """Handle .jsx and .tsx extensions."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "Button.tsx").write_text("test")
        (tmp_path / "src" / "App.jsx").write_text("test")

        output = """
src/Button.tsx
  5:1  error  Missing prop validation

src/App.jsx
  10:1  warning  No return type
"""
        paths = extract_eslint_repo_paths(output, tmp_path)
        assert "src/Button.tsx" in paths
        assert "src/App.jsx" in paths

    def test_empty_input(self, tmp_path: Path) -> None:
        """Handle empty input."""
        assert extract_eslint_repo_paths("", tmp_path) == []


class TestExtractTscRepoPaths:
    """Tests for extract_tsc_repo_paths function."""

    def test_extracts_paren_format(self, tmp_path: Path) -> None:
        """Extract paths from tsc parenthesis format."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.ts").write_text("test")

        output = """
src/utils.ts(12,5): error TS2322: Type 'string' is not assignable to type 'number'.
src/utils.ts(25,10): error TS2345: Argument of type 'undefined' is not assignable...
"""
        paths = extract_tsc_repo_paths(output, tmp_path)
        assert "src/utils.ts" in paths

    def test_extracts_colon_format(self, tmp_path: Path) -> None:
        """Extract paths from tsc colon format."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.ts").write_text("test")

        output = """
src/index.ts:12:5 - error TS2322: Type 'string' is not assignable to type 'number'.
"""
        paths = extract_tsc_repo_paths(output, tmp_path)
        assert "src/index.ts" in paths

    def test_handles_tsx_files(self, tmp_path: Path) -> None:
        """Handle .tsx extension."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "Button.tsx").write_text("test")

        output = """
src/Button.tsx(5,10): error TS2339: Property 'onClick' does not exist on type...
"""
        paths = extract_tsc_repo_paths(output, tmp_path)
        assert "src/Button.tsx" in paths

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        """Skip node_modules paths."""
        output = """
node_modules/@types/node/index.d.ts(12,5): error TS2322: ...
"""
        paths = extract_tsc_repo_paths(output, tmp_path)
        assert len(paths) == 0

    def test_empty_input(self, tmp_path: Path) -> None:
        """Handle empty input."""
        assert extract_tsc_repo_paths("", tmp_path) == []


class TestExtractPrettierRepoPaths:
    """Tests for extract_prettier_repo_paths function."""

    def test_extracts_warn_lines(self, tmp_path: Path) -> None:
        """Extract paths from Prettier warn lines."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.ts").write_text("test")
        (tmp_path / "src" / "Button.tsx").write_text("test")

        output = """
Checking formatting...
[warn] src/utils.ts
[warn] src/Button.tsx
[warn] Code style issues found in 2 files. Run Prettier to fix.
"""
        paths = extract_prettier_repo_paths(output, tmp_path)
        assert "src/utils.ts" in paths
        assert "src/Button.tsx" in paths

    def test_handles_various_extensions(self, tmp_path: Path) -> None:
        """Handle various file extensions Prettier supports."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.js").write_text("test")
        (tmp_path / "styles.css").write_text("test")
        (tmp_path / "config.json").write_text("test")

        output = """
[warn] src/index.js
[warn] styles.css
[warn] config.json
"""
        paths = extract_prettier_repo_paths(output, tmp_path)
        assert "src/index.js" in paths
        assert "styles.css" in paths
        assert "config.json" in paths

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        """Skip node_modules paths."""
        output = """
[warn] node_modules/some-pkg/index.js
"""
        paths = extract_prettier_repo_paths(output, tmp_path)
        assert len(paths) == 0

    def test_empty_input(self, tmp_path: Path) -> None:
        """Handle empty input."""
        assert extract_prettier_repo_paths("", tmp_path) == []


class TestExtractJsStacktraceRepoPaths:
    """Tests for extract_js_stacktrace_repo_paths function."""

    def test_extracts_stack_frames(self, tmp_path: Path) -> None:
        """Extract file paths from stack trace frames."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "services").mkdir(parents=True)
        (tmp_path / "src" / "services" / "auth.ts").write_text("test")
        (tmp_path / "__tests__").mkdir()
        (tmp_path / "__tests__" / "auth.test.ts").write_text("test")

        output = """
Error: Expected value to be truthy
    at AuthService.login (src/services/auth.ts:42:15)
    at Object.<anonymous> (__tests__/auth.test.ts:25:18)
    at processTicksAndRejections (node:internal/process/task_queues:95:5)
"""
        paths = extract_js_stacktrace_repo_paths(output, tmp_path)
        assert "src/services/auth.ts" in paths
        assert "__tests__/auth.test.ts" in paths

    def test_skips_node_internals(self, tmp_path: Path) -> None:
        """Skip node: internal paths."""
        output = """
    at processTicksAndRejections (node:internal/process/task_queues:95:5)
    at node:async_hooks:333:5
"""
        paths = extract_js_stacktrace_repo_paths(output, tmp_path)
        assert len(paths) == 0

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        """Skip node_modules paths."""
        output = """
    at Module._compile (node_modules/ts-node/src/index.ts:1:1)
    at Object.<anonymous> (node_modules/jest/lib/runner.ts:25:18)
"""
        paths = extract_js_stacktrace_repo_paths(output, tmp_path)
        assert len(paths) == 0

    def test_handles_various_extensions(self, tmp_path: Path) -> None:
        """Handle various JS/TS extensions."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.mjs").write_text("test")
        (tmp_path / "src" / "config.cjs").write_text("test")

        output = """
    at loadConfig (src/utils.mjs:10:5)
    at parseConfig (src/config.cjs:25:10)
"""
        paths = extract_js_stacktrace_repo_paths(output, tmp_path)
        assert "src/utils.mjs" in paths
        assert "src/config.cjs" in paths

    def test_empty_input(self, tmp_path: Path) -> None:
        """Handle empty input."""
        assert extract_js_stacktrace_repo_paths("", tmp_path) == []
