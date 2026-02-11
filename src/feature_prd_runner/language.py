"""Language detection and configuration for multi-language project support.

This module provides utilities to detect the programming language of a project
based on manifest files (package.json, pyproject.toml, go.mod, etc.) and returns
appropriate default configurations for verification commands, dependency installation,
and artifact ignores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

# Supported language identifiers
Language = Literal["python", "typescript", "javascript", "nextjs", "go", "rust", "unknown"]

# Supported test framework identifiers
TestFramework = Literal[
    "pytest", "unittest",  # Python
    "jest", "vitest", "mocha",  # JavaScript/TypeScript
    "go",  # Go
    "cargo",  # Rust
    "unknown",
]

# Supported linter identifiers
Linter = Literal[
    "ruff", "flake8", "pylint",  # Python
    "eslint", "biome",  # JavaScript/TypeScript
    "golangci-lint",  # Go
    "clippy",  # Rust
    "unknown",
]

# Supported type checker identifiers
TypeChecker = Literal[
    "mypy", "pyright",  # Python
    "tsc",  # TypeScript
    "unknown",
]


def detect_language(project_dir: Path) -> Language:
    """Detect the primary programming language of a project.

    Detection is based on the presence of language-specific manifest files.
    The first matching manifest determines the language.

    Detection order (first match wins):
    1. package.json with typescript in devDependencies → typescript
    2. package.json without typescript → javascript
    3. pyproject.toml or setup.py → python
    4. go.mod → go
    5. Cargo.toml → rust
    6. Default → unknown

    Args:
        project_dir: Path to the project root directory.

    Returns:
        The detected language identifier.
    """
    project_dir = Path(project_dir).resolve()

    # Check for Node.js/TypeScript/Next.js projects
    package_json = project_dir / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            dev_deps = data.get("devDependencies", {})
            deps = data.get("dependencies", {})
            all_deps = {**deps, **dev_deps}

            # Check for Next.js (before TypeScript — Next.js projects also have typescript)
            if "next" in all_deps:
                return "nextjs"

            # Check for next.config.* as additional Next.js indicator
            if any((project_dir / f"next.config.{ext}").exists() for ext in ("js", "ts", "mjs")):
                return "nextjs"

            # Check for TypeScript
            if "typescript" in all_deps:
                return "typescript"

            # Check for tsconfig.json as additional TypeScript indicator
            if (project_dir / "tsconfig.json").exists():
                return "typescript"

            return "javascript"
        except (json.JSONDecodeError, OSError):
            # Malformed package.json — check for next.config.* before defaulting to JS
            if any((project_dir / f"next.config.{ext}").exists() for ext in ("js", "ts", "mjs")):
                return "nextjs"
            return "javascript"

    # Check for Python projects
    if (project_dir / "pyproject.toml").exists():
        return "python"
    if (project_dir / "setup.py").exists():
        return "python"
    if (project_dir / "setup.cfg").exists():
        return "python"
    if (project_dir / "requirements.txt").exists():
        return "python"

    # Check for Go projects
    if (project_dir / "go.mod").exists():
        return "go"

    # Check for Rust projects
    if (project_dir / "Cargo.toml").exists():
        return "rust"

    return "unknown"


def detect_test_framework(command: str, language: Language) -> TestFramework:
    """Detect the test framework from a test command string.

    Args:
        command: The test command string (e.g., "npm test", "pytest -v").
        language: The project language for context.

    Returns:
        The detected test framework identifier.
    """
    cmd = (command or "").strip().lower()
    if not cmd:
        return "unknown"

    # Python test frameworks
    if "pytest" in cmd:
        return "pytest"
    if "unittest" in cmd or "python -m unittest" in cmd:
        return "unittest"

    # JavaScript/TypeScript test frameworks
    if "vitest" in cmd:
        return "vitest"
    if "jest" in cmd:
        return "jest"
    if "mocha" in cmd:
        return "mocha"

    # Check for npm/yarn/pnpm test which might run jest/vitest/mocha
    if any(runner in cmd for runner in ["npm test", "yarn test", "pnpm test", "npm run test"]):
        # Infer from language
        if language in ("typescript", "javascript", "nextjs"):
            return "jest"  # Most common default

    # Rust test (check before Go since "cargo test" contains "go test" substring)
    if "cargo test" in cmd:
        return "cargo"

    # Go test
    if "go test" in cmd:
        return "go"

    return "unknown"


def detect_linter(command: str, language: Language) -> Linter:
    """Detect the linter from a lint command string.

    Args:
        command: The lint command string (e.g., "ruff check .", "eslint .").
        language: The project language for context.

    Returns:
        The detected linter identifier.
    """
    cmd = (command or "").strip().lower()
    if not cmd:
        return "unknown"

    # Python linters
    if "ruff" in cmd:
        return "ruff"
    if "flake8" in cmd:
        return "flake8"
    if "pylint" in cmd:
        return "pylint"

    # JavaScript/TypeScript/Next.js linters
    if "next lint" in cmd:
        return "eslint"  # next lint wraps ESLint
    if "eslint" in cmd:
        return "eslint"
    if "biome" in cmd:
        return "biome"

    # Go linters
    if "golangci-lint" in cmd:
        return "golangci-lint"

    # Rust linters
    if "clippy" in cmd:
        return "clippy"

    return "unknown"


def detect_typechecker(command: str, language: Language) -> TypeChecker:
    """Detect the type checker from a typecheck command string.

    Args:
        command: The typecheck command string (e.g., "mypy .", "tsc --noEmit").
        language: The project language for context.

    Returns:
        The detected type checker identifier.
    """
    cmd = (command or "").strip().lower()
    if not cmd:
        return "unknown"

    # Python type checkers
    if "mypy" in cmd:
        return "mypy"
    if "pyright" in cmd:
        return "pyright"

    # TypeScript/Next.js type checker
    if "next build" in cmd:
        return "tsc"  # next build includes type checking
    if "tsc" in cmd:
        return "tsc"

    return "unknown"


def get_default_verify_commands(language: Language) -> dict[str, str | None]:
    """Return default verification commands for a language.

    Args:
        language: The project language.

    Returns:
        A dict with keys: test_command, lint_command, format_command, typecheck_command.
        Values may be None if no default is appropriate.
    """
    defaults: dict[Language, dict[str, str | None]] = {
        "python": {
            "test_command": "pytest -v",
            "lint_command": "ruff check .",
            "format_command": "ruff format --check .",
            "typecheck_command": None,  # mypy requires project-specific config
        },
        "typescript": {
            "test_command": "npm test",
            "lint_command": "npx eslint .",
            "format_command": "npx prettier --check .",
            "typecheck_command": "npx tsc --noEmit",
        },
        "nextjs": {
            "test_command": "npm test",
            "lint_command": "npx next lint",
            "format_command": "npx prettier --check .",
            "typecheck_command": "npx next build",
        },
        "javascript": {
            "test_command": "npm test",
            "lint_command": "npx eslint .",
            "format_command": "npx prettier --check .",
            "typecheck_command": None,
        },
        "go": {
            "test_command": "go test ./... -v",
            "lint_command": "golangci-lint run",
            "format_command": "gofmt -l .",
            "typecheck_command": None,  # Go is compiled
        },
        "rust": {
            "test_command": "cargo test",
            "lint_command": "cargo clippy -- -D warnings",
            "format_command": "cargo fmt --check",
            "typecheck_command": None,  # Rust is compiled
        },
        "unknown": {
            "test_command": None,
            "lint_command": None,
            "format_command": None,
            "typecheck_command": None,
        },
    }
    return defaults.get(language, defaults["unknown"])


def get_default_deps_command(language: Language) -> str | None:
    """Return the default dependency installation command for a language.

    Args:
        language: The project language.

    Returns:
        The default install command, or None if not applicable.
    """
    commands: dict[Language, str | None] = {
        "python": 'python -m pip install -e ".[test]"',
        "typescript": "npm install",
        "nextjs": "npm install",
        "javascript": "npm install",
        "go": "go mod download",
        "rust": "cargo build",
        "unknown": None,
    }
    return commands.get(language)


def get_ignored_paths(language: Language) -> list[str]:
    """Return paths to ignore for code review based on language.

    These paths typically contain build artifacts, caches, and dependencies
    that should not be included in code reviews.

    Args:
        language: The project language.

    Returns:
        A list of path patterns to ignore.
    """
    # Common ignores for all languages
    common = [
        ".prd_runner/",
        ".git/",
        ".DS_Store",
        "*.log",
    ]

    language_specific: dict[Language, list[str]] = {
        "python": [
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            ".tox/",
            ".venv/",
            "venv/",
            ".eggs/",
            "htmlcov/",
            "*.egg-info/",
            "*.egg-info/*",
            ".coverage",
            "*.pyc",
            "*.pyo",
            "dist/",
            "build/",
        ],
        "typescript": [
            "node_modules/",
            "dist/",
            "build/",
            ".next/",
            ".nuxt/",
            "coverage/",
            ".jest_cache/",
            ".turbo/",
            "*.js.map",
            "*.d.ts.map",
            ".tsbuildinfo",
            "*.tsbuildinfo",
        ],
        "nextjs": [
            "node_modules/",
            "dist/",
            "build/",
            ".next/",
            ".nuxt/",
            ".vercel/",
            "out/",
            "coverage/",
            ".jest_cache/",
            ".turbo/",
            "*.js.map",
            "*.d.ts.map",
            ".tsbuildinfo",
            "*.tsbuildinfo",
        ],
        "javascript": [
            "node_modules/",
            "dist/",
            "build/",
            "coverage/",
            ".jest_cache/",
        ],
        "go": [
            "vendor/",
            "bin/",
        ],
        "rust": [
            "target/",
        ],
        "unknown": [],
    }

    return common + language_specific.get(language, [])


def get_verify_profile_for_language(language: Language) -> str:
    """Return the appropriate verify profile name for a language.

    Args:
        language: The project language.

    Returns:
        The verify profile name to use (e.g., "python", "typescript", "none").
    """
    profiles: dict[Language, str] = {
        "python": "python",
        "typescript": "typescript",
        "nextjs": "nextjs",
        "javascript": "javascript",
        "go": "go",
        "rust": "rust",
        "unknown": "none",
    }
    return profiles.get(language, "none")
