"""Tests for the language detection module."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_prd_runner.language import (
    detect_language,
    detect_test_framework,
    detect_linter,
    detect_typechecker,
    get_default_verify_commands,
    get_default_deps_command,
    get_ignored_paths,
    get_verify_profile_for_language,
)


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detects_typescript_from_package_json_with_typescript_dep(self, tmp_path: Path) -> None:
        """Detect TypeScript when typescript is in devDependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-project",
            "devDependencies": {"typescript": "^5.0.0"}
        }))

        assert detect_language(tmp_path) == "typescript"

    def test_detects_typescript_from_package_json_with_typescript_in_deps(self, tmp_path: Path) -> None:
        """Detect TypeScript when typescript is in dependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-project",
            "dependencies": {"typescript": "^5.0.0"}
        }))

        assert detect_language(tmp_path) == "typescript"

    def test_detects_typescript_from_tsconfig(self, tmp_path: Path) -> None:
        """Detect TypeScript when tsconfig.json exists alongside package.json."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test-project"}))
        tsconfig = tmp_path / "tsconfig.json"
        tsconfig.write_text("{}")

        assert detect_language(tmp_path) == "typescript"

    def test_detects_javascript_from_package_json_without_typescript(self, tmp_path: Path) -> None:
        """Detect JavaScript when package.json exists without TypeScript."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-project",
            "dependencies": {"express": "^4.0.0"}
        }))

        assert detect_language(tmp_path) == "javascript"

    def test_detects_python_from_pyproject_toml(self, tmp_path: Path) -> None:
        """Detect Python from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        assert detect_language(tmp_path) == "python"

    def test_detects_python_from_setup_py(self, tmp_path: Path) -> None:
        """Detect Python from setup.py."""
        setup_py = tmp_path / "setup.py"
        setup_py.write_text("from setuptools import setup\nsetup()")

        assert detect_language(tmp_path) == "python"

    def test_detects_python_from_requirements_txt(self, tmp_path: Path) -> None:
        """Detect Python from requirements.txt."""
        req = tmp_path / "requirements.txt"
        req.write_text("flask==2.0.0\n")

        assert detect_language(tmp_path) == "python"

    def test_detects_go_from_go_mod(self, tmp_path: Path) -> None:
        """Detect Go from go.mod."""
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("module example.com/test\n\ngo 1.21\n")

        assert detect_language(tmp_path) == "go"

    def test_detects_rust_from_cargo_toml(self, tmp_path: Path) -> None:
        """Detect Rust from Cargo.toml."""
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "test"\n')

        assert detect_language(tmp_path) == "rust"

    def test_returns_unknown_for_empty_directory(self, tmp_path: Path) -> None:
        """Return 'unknown' for directory without manifest files."""
        assert detect_language(tmp_path) == "unknown"

    def test_handles_malformed_package_json(self, tmp_path: Path) -> None:
        """Handle malformed package.json gracefully."""
        package_json = tmp_path / "package.json"
        package_json.write_text("{ invalid json }")

        # Should still return javascript since package.json exists
        assert detect_language(tmp_path) == "javascript"


class TestDetectTestFramework:
    """Tests for detect_test_framework function."""

    def test_detects_pytest(self) -> None:
        """Detect pytest from command."""
        assert detect_test_framework("pytest -v", "python") == "pytest"
        assert detect_test_framework("python -m pytest tests/", "python") == "pytest"

    def test_detects_unittest(self) -> None:
        """Detect unittest from command."""
        assert detect_test_framework("python -m unittest discover", "python") == "unittest"

    def test_detects_jest(self) -> None:
        """Detect Jest from command."""
        assert detect_test_framework("jest --coverage", "typescript") == "jest"
        assert detect_test_framework("npx jest", "javascript") == "jest"

    def test_detects_vitest(self) -> None:
        """Detect Vitest from command."""
        assert detect_test_framework("vitest run", "typescript") == "vitest"
        assert detect_test_framework("npx vitest", "typescript") == "vitest"

    def test_detects_mocha(self) -> None:
        """Detect Mocha from command."""
        assert detect_test_framework("mocha tests/", "javascript") == "mocha"

    def test_detects_go_test(self) -> None:
        """Detect go test from command."""
        assert detect_test_framework("go test ./...", "go") == "go"

    def test_detects_cargo_test(self) -> None:
        """Detect cargo test from command."""
        assert detect_test_framework("cargo test", "rust") == "cargo"

    def test_infers_jest_from_npm_test_for_js(self) -> None:
        """Infer Jest as default for npm test in JS/TS projects."""
        assert detect_test_framework("npm test", "typescript") == "jest"
        assert detect_test_framework("yarn test", "javascript") == "jest"

    def test_returns_unknown_for_empty_command(self) -> None:
        """Return 'unknown' for empty command."""
        assert detect_test_framework("", "python") == "unknown"
        assert detect_test_framework(None, "typescript") == "unknown"  # type: ignore


class TestDetectLinter:
    """Tests for detect_linter function."""

    def test_detects_ruff(self) -> None:
        """Detect ruff from command."""
        assert detect_linter("ruff check .", "python") == "ruff"

    def test_detects_eslint(self) -> None:
        """Detect ESLint from command."""
        assert detect_linter("eslint .", "typescript") == "eslint"
        assert detect_linter("npx eslint src/", "javascript") == "eslint"

    def test_detects_golangci_lint(self) -> None:
        """Detect golangci-lint from command."""
        assert detect_linter("golangci-lint run", "go") == "golangci-lint"

    def test_detects_clippy(self) -> None:
        """Detect clippy from command."""
        assert detect_linter("cargo clippy", "rust") == "clippy"

    def test_returns_unknown_for_empty_command(self) -> None:
        """Return 'unknown' for empty command."""
        assert detect_linter("", "python") == "unknown"


class TestDetectTypechecker:
    """Tests for detect_typechecker function."""

    def test_detects_mypy(self) -> None:
        """Detect mypy from command."""
        assert detect_typechecker("mypy src/", "python") == "mypy"

    def test_detects_pyright(self) -> None:
        """Detect pyright from command."""
        assert detect_typechecker("pyright .", "python") == "pyright"

    def test_detects_tsc(self) -> None:
        """Detect tsc from command."""
        assert detect_typechecker("tsc --noEmit", "typescript") == "tsc"
        assert detect_typechecker("npx tsc --noEmit", "typescript") == "tsc"

    def test_returns_unknown_for_empty_command(self) -> None:
        """Return 'unknown' for empty command."""
        assert detect_typechecker("", "python") == "unknown"


class TestGetDefaultVerifyCommands:
    """Tests for get_default_verify_commands function."""

    def test_python_defaults(self) -> None:
        """Check Python default commands."""
        defaults = get_default_verify_commands("python")
        assert defaults["test_command"] == "pytest -v"
        assert defaults["lint_command"] == "ruff check ."
        assert defaults["format_command"] == "ruff format --check ."

    def test_typescript_defaults(self) -> None:
        """Check TypeScript default commands."""
        defaults = get_default_verify_commands("typescript")
        assert defaults["test_command"] == "npm test"
        assert defaults["lint_command"] == "npx eslint ."
        assert defaults["format_command"] == "npx prettier --check ."
        assert defaults["typecheck_command"] == "npx tsc --noEmit"

    def test_unknown_returns_none_values(self) -> None:
        """Check unknown language returns None values."""
        defaults = get_default_verify_commands("unknown")
        assert defaults["test_command"] is None
        assert defaults["lint_command"] is None


class TestGetDefaultDepsCommand:
    """Tests for get_default_deps_command function."""

    def test_python_deps_command(self) -> None:
        """Check Python default deps command."""
        assert get_default_deps_command("python") == 'python -m pip install -e ".[test]"'

    def test_typescript_deps_command(self) -> None:
        """Check TypeScript default deps command."""
        assert get_default_deps_command("typescript") == "npm install"

    def test_go_deps_command(self) -> None:
        """Check Go default deps command."""
        assert get_default_deps_command("go") == "go mod download"

    def test_unknown_returns_none(self) -> None:
        """Check unknown language returns None."""
        assert get_default_deps_command("unknown") is None


class TestGetIgnoredPaths:
    """Tests for get_ignored_paths function."""

    def test_python_includes_python_specific_paths(self) -> None:
        """Check Python ignored paths include Python-specific dirs."""
        paths = get_ignored_paths("python")
        assert "__pycache__/" in paths
        assert ".pytest_cache/" in paths
        assert ".mypy_cache/" in paths
        assert "*.pyc" in paths

    def test_typescript_includes_node_specific_paths(self) -> None:
        """Check TypeScript ignored paths include Node-specific dirs."""
        paths = get_ignored_paths("typescript")
        assert "node_modules/" in paths
        assert ".jest_cache/" in paths
        assert "*.js.map" in paths

    def test_all_languages_include_common_paths(self) -> None:
        """Check all languages include common ignored paths."""
        for lang in ["python", "typescript", "javascript", "go", "rust", "unknown"]:
            paths = get_ignored_paths(lang)  # type: ignore
            assert ".prd_runner/" in paths
            assert ".git/" in paths


class TestDetectLanguageNextJs:
    """Tests for Next.js detection in detect_language function."""

    def test_detects_nextjs_from_next_dependency(self, tmp_path: Path) -> None:
        """Detect Next.js when 'next' is in dependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-project",
            "dependencies": {"next": "^14.0.0", "react": "^18.0.0"},
            "devDependencies": {"typescript": "^5.0.0"}
        }))

        assert detect_language(tmp_path) == "nextjs"

    def test_detects_nextjs_from_next_dev_dependency(self, tmp_path: Path) -> None:
        """Detect Next.js when 'next' is in devDependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-project",
            "devDependencies": {"next": "^14.0.0", "typescript": "^5.0.0"}
        }))

        assert detect_language(tmp_path) == "nextjs"

    def test_detects_nextjs_from_next_config_js(self, tmp_path: Path) -> None:
        """Detect Next.js from next.config.js file."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test-project"}))
        (tmp_path / "next.config.js").write_text("module.exports = {};")

        assert detect_language(tmp_path) == "nextjs"

    def test_detects_nextjs_from_next_config_ts(self, tmp_path: Path) -> None:
        """Detect Next.js from next.config.ts file."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test-project"}))
        (tmp_path / "next.config.ts").write_text("export default {};")

        assert detect_language(tmp_path) == "nextjs"

    def test_detects_nextjs_from_next_config_mjs(self, tmp_path: Path) -> None:
        """Detect Next.js from next.config.mjs file."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test-project"}))
        (tmp_path / "next.config.mjs").write_text("export default {};")

        assert detect_language(tmp_path) == "nextjs"

    def test_nextjs_priority_over_typescript(self, tmp_path: Path) -> None:
        """Next.js detection takes priority over TypeScript."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-project",
            "dependencies": {"next": "^14.0.0", "react": "^18.0.0"},
            "devDependencies": {"typescript": "^5.0.0"}
        }))
        (tmp_path / "tsconfig.json").write_text("{}")

        assert detect_language(tmp_path) == "nextjs"

    def test_detects_nextjs_from_malformed_package_json_with_next_config(self, tmp_path: Path) -> None:
        """Detect Next.js from next.config.js when package.json is malformed."""
        package_json = tmp_path / "package.json"
        package_json.write_text("{ invalid json }")
        (tmp_path / "next.config.js").write_text("module.exports = {};")

        assert detect_language(tmp_path) == "nextjs"

    def test_malformed_package_json_without_next_config_returns_javascript(self, tmp_path: Path) -> None:
        """Malformed package.json without next.config falls back to javascript."""
        package_json = tmp_path / "package.json"
        package_json.write_text("{ invalid json }")

        assert detect_language(tmp_path) == "javascript"


class TestNextJsDefaults:
    """Tests for Next.js default commands and configuration."""

    def test_verify_commands(self) -> None:
        """Check Next.js default verify commands."""
        defaults = get_default_verify_commands("nextjs")
        assert defaults["test_command"] == "npm test"
        assert defaults["lint_command"] == "npx next lint"
        assert defaults["format_command"] == "npx prettier --check ."
        assert defaults["typecheck_command"] == "npx next build"

    def test_deps_command(self) -> None:
        """Check Next.js default deps command."""
        assert get_default_deps_command("nextjs") == "npm install"

    def test_ignored_paths(self) -> None:
        """Check Next.js ignored paths include framework-specific dirs."""
        paths = get_ignored_paths("nextjs")
        assert "node_modules/" in paths
        assert ".next/" in paths
        assert ".vercel/" in paths
        assert "out/" in paths
        assert ".prd_runner/" in paths  # common path
        assert ".git/" in paths  # common path

    def test_verify_profile(self) -> None:
        """Check Next.js verify profile."""
        assert get_verify_profile_for_language("nextjs") == "nextjs"


class TestNextJsDetectors:
    """Tests for Next.js-specific detector functions."""

    def test_detect_test_framework_npm_test(self) -> None:
        """detect_test_framework infers Jest for nextjs with npm test."""
        assert detect_test_framework("npm test", "nextjs") == "jest"

    def test_detect_test_framework_jest(self) -> None:
        """detect_test_framework detects jest for nextjs."""
        assert detect_test_framework("jest --coverage", "nextjs") == "jest"

    def test_detect_test_framework_vitest(self) -> None:
        """detect_test_framework detects vitest for nextjs."""
        assert detect_test_framework("vitest run", "nextjs") == "vitest"

    def test_detect_linter_next_lint(self) -> None:
        """detect_linter detects 'next lint' as eslint."""
        assert detect_linter("npx next lint", "nextjs") == "eslint"

    def test_detect_linter_eslint(self) -> None:
        """detect_linter still detects eslint for nextjs."""
        assert detect_linter("npx eslint .", "nextjs") == "eslint"

    def test_detect_typechecker_next_build(self) -> None:
        """detect_typechecker detects 'next build' as tsc."""
        assert detect_typechecker("npx next build", "nextjs") == "tsc"

    def test_detect_typechecker_tsc(self) -> None:
        """detect_typechecker still detects tsc for nextjs."""
        assert detect_typechecker("npx tsc --noEmit", "nextjs") == "tsc"


class TestGetVerifyProfileForLanguage:
    """Tests for get_verify_profile_for_language function."""

    def test_returns_correct_profiles(self) -> None:
        """Check correct verify profiles are returned."""
        assert get_verify_profile_for_language("python") == "python"
        assert get_verify_profile_for_language("typescript") == "typescript"
        assert get_verify_profile_for_language("javascript") == "javascript"
        assert get_verify_profile_for_language("nextjs") == "nextjs"
        assert get_verify_profile_for_language("go") == "go"
        assert get_verify_profile_for_language("unknown") == "none"
