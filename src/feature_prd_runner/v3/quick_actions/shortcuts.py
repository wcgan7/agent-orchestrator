"""Shortcut matching engine for quick actions.

Maps well-known prompts (e.g. "run tests", "git status") to concrete shell
commands so they can execute instantly without agent overhead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ShortcutRule:
    name: str
    patterns: list[str]
    command: str  # shell string or "auto:test" / "auto:lint" / "auto:typecheck" / "auto:format"


@dataclass
class ShortcutMatch:
    matched: bool
    shortcut_name: str = ""
    command: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Built-in shortcut table
# ---------------------------------------------------------------------------

_BUILTINS: list[ShortcutRule] = [
    ShortcutRule(
        name="run_tests",
        patterns=[r"^(run\s+)?tests?$", r"^pytest$", r"^npm\s+test$"],
        command="auto:test",
    ),
    ShortcutRule(
        name="lint",
        patterns=[r"^lint$", r"^run\s+lint(er)?$"],
        command="auto:lint",
    ),
    ShortcutRule(
        name="typecheck",
        patterns=[r"^type\s*check$", r"^mypy$", r"^tsc$"],
        command="auto:typecheck",
    ),
    ShortcutRule(
        name="git_status",
        patterns=[r"^git\s+status$"],
        command="git status",
    ),
    ShortcutRule(
        name="git_diff",
        patterns=[r"^git\s+diff$"],
        command="git diff",
    ),
    ShortcutRule(
        name="git_log",
        patterns=[r"^git\s+log$"],
        command="git log --oneline -20",
    ),
    ShortcutRule(
        name="format",
        patterns=[r"^format$", r"^fmt$"],
        command="auto:format",
    ),
]

# ---------------------------------------------------------------------------
# Auto-detection helpers (mirrors bug_scan.py patterns)
# ---------------------------------------------------------------------------


def _detect_test_command(project_dir: Path) -> Optional[str]:
    if (project_dir / "pyproject.toml").exists() or (project_dir / "pytest.ini").exists():
        return "python -m pytest --tb=short -q"
    if (project_dir / "package.json").exists():
        return "npm test"
    if (project_dir / "go.mod").exists():
        return "go test ./..."
    if (project_dir / "Cargo.toml").exists():
        return "cargo test"
    return None


def _detect_lint_command(project_dir: Path) -> Optional[str]:
    if (project_dir / "pyproject.toml").exists():
        return "python -m ruff check ."
    if (project_dir / "package.json").exists():
        return "npx eslint ."
    return None


def _detect_typecheck_command(project_dir: Path) -> Optional[str]:
    if (project_dir / "pyproject.toml").exists():
        return "python -m mypy . --ignore-missing-imports"
    if (project_dir / "tsconfig.json").exists():
        return "npx tsc --noEmit"
    return None


def _detect_format_command(project_dir: Path) -> Optional[str]:
    if (project_dir / "pyproject.toml").exists():
        return "python -m black ."
    if (project_dir / "package.json").exists():
        return "npx prettier --write ."
    return None


_AUTO_RESOLVERS: dict[str, callable] = {
    "auto:test": _detect_test_command,
    "auto:lint": _detect_lint_command,
    "auto:typecheck": _detect_typecheck_command,
    "auto:format": _detect_format_command,
}


def _resolve_command(command: str, project_dir: Path) -> Optional[str]:
    """Resolve an auto:* command to a concrete shell string, or return as-is."""
    resolver = _AUTO_RESOLVERS.get(command)
    if resolver:
        return resolver(project_dir)
    return command


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_shortcuts(project_dir: Path) -> list[ShortcutRule]:
    """Load built-in shortcuts, then merge user overrides from config."""
    rules = list(_BUILTINS)

    user_file = project_dir / ".prd_runner" / "quick_shortcuts.yaml"
    if user_file.exists():
        try:
            import yaml
            with open(user_file) as f:
                user_data = yaml.safe_load(f)
            if isinstance(user_data, list):
                builtin_names = {r.name for r in rules}
                for entry in user_data:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name", "")
                    patterns = entry.get("patterns", [])
                    command = entry.get("command", "")
                    if not name or not command:
                        continue
                    if name in builtin_names:
                        # Override: replace the built-in
                        rules = [r for r in rules if r.name != name]
                    rules.append(ShortcutRule(name=name, patterns=patterns, command=command))
        except Exception:
            pass  # Skip silently on any YAML error

    return rules


def match_prompt(prompt: str, rules: list[ShortcutRule], project_dir: Path) -> ShortcutMatch:
    """Match a user prompt against shortcut rules.

    Returns ShortcutMatch with matched=True and a resolved command on hit,
    or ShortcutMatch(matched=False) on miss.
    """
    normalized = prompt.strip().lower()
    for rule in rules:
        for pattern in rule.patterns:
            if re.fullmatch(pattern, normalized, re.IGNORECASE):
                resolved = _resolve_command(rule.command, project_dir)
                if resolved is None:
                    # auto:* could not detect a command for this project
                    return ShortcutMatch(matched=False)
                return ShortcutMatch(
                    matched=True,
                    shortcut_name=rule.name,
                    command=resolved,
                    confidence=1.0,
                )
    return ShortcutMatch(matched=False)
