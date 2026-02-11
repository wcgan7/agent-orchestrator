"""Tests for Next.js signal extractors and parser registry integration."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_prd_runner.signals import (
    extract_next_build_repo_paths,
    extract_eslint_repo_paths,
    extract_jest_failed_test_files,
    extract_js_stacktrace_repo_paths,
    extract_prettier_repo_paths,
    extract_tsc_repo_paths,
    get_format_parser,
    get_lint_parser,
    get_test_parser,
    get_traceback_parser,
    get_typecheck_parser,
)


class TestExtractNextBuildRepoPaths:
    """Tests for extract_next_build_repo_paths function."""

    def test_extracts_type_error_paths(self, tmp_path: Path) -> None:
        """Extract paths from Next.js type error output."""
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "page.tsx").write_text("export default function Home() {}")

        text = """\
./app/page.tsx:12:5
Type error: Property 'foo' does not exist on type 'Props'.
"""
        result = extract_next_build_repo_paths(text, tmp_path)
        assert "app/page.tsx" in result

    def test_extracts_embedded_tsc_errors(self, tmp_path: Path) -> None:
        """Extract paths from embedded tsc-style errors in next build output."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.ts").write_text("export const x = 1;")

        text = """\
src/utils.ts(42,5): error TS2322: Type 'string' is not assignable to type 'number'.
"""
        result = extract_next_build_repo_paths(text, tmp_path)
        assert "src/utils.ts" in result

    def test_filters_node_modules(self, tmp_path: Path) -> None:
        """Skip node_modules paths."""
        text = """\
./node_modules/some-package/index.tsx:1:1
Type error: something
"""
        result = extract_next_build_repo_paths(text, tmp_path)
        assert result == []

    def test_empty_input(self, tmp_path: Path) -> None:
        """Return empty list for empty input."""
        assert extract_next_build_repo_paths("", tmp_path) == []
        assert extract_next_build_repo_paths(None, tmp_path) == []  # type: ignore

    def test_extracts_paths_without_leading_dot_slash(self, tmp_path: Path) -> None:
        """Extract paths without leading ./ prefix."""
        (tmp_path / "components").mkdir()
        (tmp_path / "components" / "Button.tsx").write_text("export default function Button() {}")

        text = """\
components/Button.tsx:5:3
Type error: Missing return type.
"""
        result = extract_next_build_repo_paths(text, tmp_path)
        assert "components/Button.tsx" in result


class TestNextJsParserRegistry:
    """Tests for parser registry functions returning correct parsers for nextjs."""

    def test_get_test_parser_for_nextjs(self) -> None:
        """get_test_parser returns Jest parser for nextjs language."""
        parser = get_test_parser("npm test", "nextjs")
        assert parser is extract_jest_failed_test_files

    def test_get_test_parser_nextjs_default(self) -> None:
        """get_test_parser returns Jest parser for nextjs even with empty command."""
        parser = get_test_parser("", "nextjs")
        assert parser is extract_jest_failed_test_files

    def test_get_lint_parser_next_lint_command(self) -> None:
        """get_lint_parser detects 'next lint' command."""
        parser = get_lint_parser("npx next lint", "nextjs")
        assert parser is extract_eslint_repo_paths

    def test_get_lint_parser_nextjs_default(self) -> None:
        """get_lint_parser defaults to ESLint parser for nextjs language."""
        parser = get_lint_parser("", "nextjs")
        assert parser is extract_eslint_repo_paths

    def test_get_typecheck_parser_next_build_command(self) -> None:
        """get_typecheck_parser detects 'next build' command."""
        parser = get_typecheck_parser("npx next build", "nextjs")
        assert parser is extract_next_build_repo_paths

    def test_get_typecheck_parser_nextjs_default(self) -> None:
        """get_typecheck_parser defaults to tsc parser for nextjs language."""
        parser = get_typecheck_parser("", "nextjs")
        assert parser is extract_tsc_repo_paths

    def test_get_format_parser_for_nextjs(self) -> None:
        """get_format_parser returns Prettier parser for nextjs language."""
        parser = get_format_parser("npx prettier --check .", "nextjs")
        assert parser is extract_prettier_repo_paths

    def test_get_format_parser_nextjs_default(self) -> None:
        """get_format_parser defaults to Prettier parser for nextjs language."""
        parser = get_format_parser("", "nextjs")
        assert parser is extract_prettier_repo_paths

    def test_get_traceback_parser_for_nextjs(self) -> None:
        """get_traceback_parser returns JS stack trace parser for nextjs."""
        parser = get_traceback_parser("nextjs")
        assert parser is extract_js_stacktrace_repo_paths
