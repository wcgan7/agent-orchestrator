"""Tests for constants module."""

import pytest

from feature_prd_runner.constants import (
    IGNORED_PATHS_BY_LANGUAGE,
    get_ignored_review_paths,
)


class TestGetIgnoredReviewPaths:
    """Tests for get_ignored_review_paths function."""

    def test_python_includes_python_specific_paths(self):
        paths = get_ignored_review_paths("python")
        assert "__pycache__/" in paths
        assert ".pytest_cache/" in paths
        assert ".mypy_cache/" in paths

    def test_typescript_includes_node_specific_paths(self):
        paths = get_ignored_review_paths("typescript")
        assert "node_modules/" in paths
        assert "dist/" in paths
        assert ".tsbuildinfo" in paths

    def test_javascript_includes_node_modules(self):
        paths = get_ignored_review_paths("javascript")
        assert "node_modules/" in paths
        assert "dist/" in paths

    def test_go_includes_vendor(self):
        paths = get_ignored_review_paths("go")
        assert "vendor/" in paths

    def test_rust_includes_target(self):
        paths = get_ignored_review_paths("rust")
        assert "target/" in paths

    def test_all_languages_include_common_paths(self):
        for language in ["python", "typescript", "javascript", "go", "rust"]:
            paths = get_ignored_review_paths(language)
            assert ".prd_runner/" in paths
            assert ".git/" in paths

    def test_unknown_language_returns_common_only(self):
        paths = get_ignored_review_paths("unknown")
        assert ".prd_runner/" in paths
        assert ".git/" in paths
        # Should not have language-specific paths
        assert "__pycache__/" not in paths
        assert "node_modules/" not in paths

    def test_default_is_python(self):
        default_paths = get_ignored_review_paths()
        python_paths = get_ignored_review_paths("python")
        assert default_paths == python_paths
