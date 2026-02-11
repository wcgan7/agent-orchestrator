"""Security audit generator — scans for vulnerabilities and creates fix tasks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from ..model import Task, TaskPriority, TaskSource, TaskType


class SecurityAuditGenerator:
    """Scan dependencies and code for security issues, create fix tasks."""

    name = "security_audit"
    description = "Scan for dependency vulnerabilities, hardcoded secrets, and security anti-patterns"

    def generate(
        self,
        project_dir: Path,
        *,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> list[Task]:
        tasks: list[Task] = []
        _progress = on_progress or (lambda msg, frac: None)

        # 1. Dependency vulnerability scan
        _progress("Scanning dependencies for vulnerabilities...", 0.1)
        tasks.extend(self._scan_dependencies(project_dir))

        # 2. Hardcoded secrets scan
        _progress("Scanning for hardcoded secrets...", 0.4)
        tasks.extend(self._scan_secrets(project_dir))

        # 3. Common security anti-patterns
        _progress("Scanning for security anti-patterns...", 0.7)
        tasks.extend(self._scan_antipatterns(project_dir))

        _progress("Security audit complete", 1.0)
        return tasks

    def _scan_dependencies(self, project_dir: Path) -> list[Task]:
        """Run dependency audit tools (pip-audit, npm audit, etc.)."""
        tasks: list[Task] = []

        # Python: pip-audit
        if (project_dir / "pyproject.toml").exists() or (project_dir / "requirements.txt").exists():
            result = self._run_command(project_dir, "python -m pip_audit --format=json", timeout=120)
            if result and result.returncode != 0:
                tasks.append(Task(
                    title="Fix Python dependency vulnerabilities",
                    description=(
                        "pip-audit found vulnerable dependencies.\n\n"
                        f"```\n{self._truncate(result.stdout + result.stderr)}\n```"
                    ),
                    task_type=TaskType.SECURITY,
                    priority=TaskPriority.P1,
                    source=TaskSource.SECURITY_AUDIT,
                    labels=["security", "dependencies", "automated"],
                ))

        # Node: npm audit
        if (project_dir / "package.json").exists():
            result = self._run_command(project_dir, "npm audit --json", timeout=120)
            if result and result.returncode != 0:
                tasks.append(Task(
                    title="Fix Node.js dependency vulnerabilities",
                    description=(
                        "npm audit found vulnerable packages.\n\n"
                        f"```\n{self._truncate(result.stdout + result.stderr)}\n```"
                    ),
                    task_type=TaskType.SECURITY,
                    priority=TaskPriority.P1,
                    source=TaskSource.SECURITY_AUDIT,
                    labels=["security", "dependencies", "automated"],
                ))

        return tasks

    def _scan_secrets(self, project_dir: Path) -> list[Task]:
        """Look for hardcoded secrets patterns in source files."""
        import re

        patterns = [
            (r'(?i)(api[_-]?key|secret[_-]?key|password|token)\s*=\s*["\'][^"\']{8,}["\']', "hardcoded secret"),
            (r'(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', "private key in source"),
            (r'(?i)(aws_access_key_id|aws_secret_access_key)\s*=\s*["\'][A-Z0-9/+=]+["\']', "AWS credential"),
        ]

        findings: list[str] = []
        source_exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".yaml", ".yml", ".json", ".toml"}

        for path in project_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in source_exts:
                continue
            if any(part.startswith(".") or part == "node_modules" for part in path.parts):
                continue
            try:
                content = path.read_text(errors="ignore")
                for pattern, desc in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        rel = path.relative_to(project_dir)
                        findings.append(f"- {rel}: {desc} ({len(matches)} occurrence(s))")
            except (OSError, UnicodeDecodeError):
                continue

        if findings:
            return [Task(
                title="Remove hardcoded secrets from source code",
                description=(
                    "Found potential secrets or credentials in source files:\n\n"
                    + "\n".join(findings[:30])
                    + ("\n\n... and more" if len(findings) > 30 else "")
                    + "\n\nMove these to environment variables or a secrets manager."
                ),
                task_type=TaskType.SECURITY,
                priority=TaskPriority.P0,
                source=TaskSource.SECURITY_AUDIT,
                labels=["security", "secrets", "automated"],
            )]
        return []

    def _scan_antipatterns(self, project_dir: Path) -> list[Task]:
        """Look for common security anti-patterns."""
        import re

        checks = [
            (r'eval\s*\(', ".py", "Use of eval() — potential code injection"),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', ".py", "subprocess with shell=True — command injection risk"),
            (r'innerHTML\s*=', (".js", ".ts", ".tsx", ".jsx"), "innerHTML assignment — potential XSS"),
            (r'dangerouslySetInnerHTML', (".js", ".ts", ".tsx", ".jsx"), "dangerouslySetInnerHTML — potential XSS"),
            (r'execute\s*\(\s*f["\']', ".py", "f-string in SQL execute — SQL injection risk"),
        ]

        findings: list[str] = []
        for path in project_dir.rglob("*"):
            if not path.is_file():
                continue
            if any(part.startswith(".") or part == "node_modules" for part in path.parts):
                continue
            try:
                content = path.read_text(errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            for pattern, exts, desc in checks:
                ext_tuple = exts if isinstance(exts, tuple) else (exts,)
                if path.suffix in ext_tuple:
                    matches = re.findall(pattern, content)
                    if matches:
                        rel = path.relative_to(project_dir)
                        findings.append(f"- {rel}: {desc}")

        if findings:
            return [Task(
                title="Fix security anti-patterns in codebase",
                description=(
                    "Found potential security issues:\n\n"
                    + "\n".join(findings[:30])
                    + ("\n\n... and more" if len(findings) > 30 else "")
                ),
                task_type=TaskType.SECURITY,
                priority=TaskPriority.P1,
                source=TaskSource.SECURITY_AUDIT,
                labels=["security", "anti-pattern", "automated"],
            )]
        return []

    def _run_command(
        self, project_dir: Path, command: str, timeout: int = 120,
    ) -> Optional[subprocess.CompletedProcess]:
        try:
            return subprocess.run(
                command, shell=True, cwd=str(project_dir),
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _truncate(self, text: str, max_len: int = 2000) -> str:
        text = text.strip()
        if len(text) > max_len:
            return text[:max_len] + "\n... (truncated)"
        return text
