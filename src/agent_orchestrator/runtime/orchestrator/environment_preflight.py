"""Environment preflight and auto-remediation helpers for worker steps."""

from __future__ import annotations

import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentIssue:
    """One environment preflight issue."""

    code: str
    summary: str
    capability: str | None = None
    recoverable: bool = True


@dataclass(frozen=True)
class EnvironmentPreflightResult:
    """Outcome of preflight + optional remediation pass."""

    ok: bool
    issues: tuple[EnvironmentIssue, ...]
    required_capabilities: tuple[str, ...] = ()
    attempted_remediation: bool = False
    remediation_log: str = ""


def workers_environment_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return normalized worker environment preflight settings."""
    workers = cfg.get("workers") if isinstance(cfg, dict) else {}
    workers = workers if isinstance(workers, dict) else {}
    raw_env = workers.get("environment")
    env: dict[str, Any] = raw_env if isinstance(raw_env, dict) else {}
    required_raw = env.get("required_capabilities_by_step")

    required: dict[str, list[str]] = {}
    if isinstance(required_raw, dict):
        for raw_step, raw_caps in required_raw.items():
            step = str(raw_step or "").strip()
            if not step or not isinstance(raw_caps, list):
                continue
            caps: list[str] = []
            for item in raw_caps:
                cap = str(item or "").strip().lower()
                if cap and cap not in caps:
                    caps.append(cap)
            if caps:
                required[step] = caps

    auto_prepare = bool(env.get("auto_prepare", True))
    capability_fallback = bool(env.get("capability_fallback", True))
    try:
        max_auto_retries = int(env.get("max_auto_retries", 3))
    except Exception:
        max_auto_retries = 3
    if max_auto_retries < 0:
        max_auto_retries = 0
    if max_auto_retries > 20:
        max_auto_retries = 20

    return {
        "auto_prepare": auto_prepare,
        "capability_fallback": capability_fallback,
        "max_auto_retries": max_auto_retries,
        "required_capabilities_by_step": required,
    }


def required_capabilities_for_step(*, step: str, project_dir: Path, cfg: dict[str, Any]) -> tuple[str, ...]:
    """Resolve required environment capabilities for the current step."""
    env_cfg = workers_environment_config(cfg)
    required = list(env_cfg["required_capabilities_by_step"].get(step, []))

    # Built-in heuristic: Prisma verify needs docker + local node deps + registry DNS.
    is_verify_step = step in {"verify", "benchmark"}
    has_prisma_schema = (project_dir / "prisma" / "schema.prisma").exists()
    if is_verify_step and has_prisma_schema:
        for cap in ("docker", "node_deps", "prisma_cli", "network"):
            if cap not in required:
                required.append(cap)

    return tuple(required)


def provider_has_capabilities(*, provider_capabilities: tuple[str, ...], required_capabilities: tuple[str, ...]) -> bool:
    """Return whether provider capability tags satisfy required capabilities."""
    if not required_capabilities:
        return True
    if not provider_capabilities:
        return False
    have = {cap.strip().lower() for cap in provider_capabilities if cap.strip()}
    needed = {cap.strip().lower() for cap in required_capabilities if cap.strip()}
    return needed.issubset(have)


def run_environment_preflight(
    *,
    step: str,
    project_dir: Path,
    cfg: dict[str, Any],
) -> EnvironmentPreflightResult:
    """Run deterministic preflight checks and best-effort auto-remediation."""
    env_cfg = workers_environment_config(cfg)
    required = required_capabilities_for_step(step=step, project_dir=project_dir, cfg=cfg)
    issues = _collect_issues(project_dir=project_dir, required_capabilities=required)
    if not issues:
        return EnvironmentPreflightResult(ok=True, issues=(), required_capabilities=required)

    attempted = False
    remediation_log = ""
    if env_cfg["auto_prepare"] and _can_auto_remediate(issues):
        attempted = True
        remediation_log = _attempt_auto_remediation(project_dir=project_dir)
        issues = _collect_issues(project_dir=project_dir, required_capabilities=required)

    return EnvironmentPreflightResult(
        ok=not issues,
        issues=tuple(issues),
        required_capabilities=required,
        attempted_remediation=attempted,
        remediation_log=remediation_log,
    )


def _collect_issues(*, project_dir: Path, required_capabilities: tuple[str, ...]) -> list[EnvironmentIssue]:
    issues: list[EnvironmentIssue] = []

    has_package_json = (project_dir / "package.json").exists()
    has_node_modules = (project_dir / "node_modules").is_dir()
    has_prisma_schema = (project_dir / "prisma" / "schema.prisma").exists()
    prisma_bin = project_dir / "node_modules" / ".bin" / "prisma"

    for cap in required_capabilities:
        normalized = cap.strip().lower()
        if not normalized:
            continue

        if normalized == "node_deps":
            if has_package_json and not has_node_modules:
                issues.append(
                    EnvironmentIssue(
                        code="node_deps_missing",
                        capability=normalized,
                        summary=(
                            "Node dependencies are missing in this task worktree "
                            "(`node_modules` not found)."
                        ),
                    )
                )
            continue

        if normalized == "prisma_cli":
            if has_prisma_schema and not prisma_bin.exists():
                issues.append(
                    EnvironmentIssue(
                        code="prisma_cli_missing",
                        capability=normalized,
                        summary=(
                            "Local Prisma CLI is missing for Prisma verification "
                            f"(expected `{prisma_bin}`)."
                        ),
                    )
                )
            continue

        if normalized == "docker":
            if not _docker_available():
                issues.append(
                    EnvironmentIssue(
                        code="docker_unavailable",
                        capability=normalized,
                        summary="Docker daemon/socket is unavailable in the worker execution environment.",
                        recoverable=False,
                    )
                )
            continue

        if normalized == "network":
            if not _dns_resolves("registry.npmjs.org"):
                issues.append(
                    EnvironmentIssue(
                        code="network_unavailable",
                        capability=normalized,
                        summary="Network DNS probe failed for `registry.npmjs.org`.",
                        recoverable=False,
                    )
                )
            continue

        if normalized.startswith("tool:"):
            tool = normalized.split(":", 1)[1].strip()
            if tool and shutil.which(tool) is None:
                issues.append(
                    EnvironmentIssue(
                        code="tool_missing",
                        capability=normalized,
                        summary=f"Required tool `{tool}` is not available on PATH.",
                        recoverable=False,
                    )
                )
            continue

    return issues


def _can_auto_remediate(issues: list[EnvironmentIssue]) -> bool:
    recoverable_codes = {"node_deps_missing", "prisma_cli_missing"}
    return any(issue.code in recoverable_codes for issue in issues)


def _attempt_auto_remediation(*, project_dir: Path) -> str:
    package_json = project_dir / "package.json"
    if not package_json.exists():
        return "Skipped auto-remediation: package.json not found."

    commands: list[list[str]] = []
    if (project_dir / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
        commands.append(["pnpm", "install", "--frozen-lockfile"])
    elif (project_dir / "yarn.lock").exists() and shutil.which("yarn"):
        commands.append(["yarn", "install", "--frozen-lockfile"])
    elif (project_dir / "package-lock.json").exists() and shutil.which("npm"):
        commands.append(["npm", "ci", "--no-audit", "--no-fund"])
    elif shutil.which("npm"):
        commands.append(["npm", "install", "--no-audit", "--no-fund"])
    else:
        return "Skipped auto-remediation: no Node package manager available."

    logs: list[str] = []
    for cmd in commands:
        try:
            completed = subprocess.run(
                cmd,
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
        except Exception as exc:
            logs.append(f"{' '.join(cmd)} -> exception: {exc}")
            continue

        if completed.returncode == 0:
            logs.append(f"{' '.join(cmd)} -> ok")
            return "\n".join(logs)

        stderr_tail = (completed.stderr or "").strip()[-400:]
        if stderr_tail:
            logs.append(f"{' '.join(cmd)} -> failed ({completed.returncode}): {stderr_tail}")
        else:
            logs.append(f"{' '.join(cmd)} -> failed ({completed.returncode})")

    return "\n".join(logs) if logs else "Auto-remediation attempted but produced no logs."


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.5)
            sock.connect("/var/run/docker.sock")
    except Exception:
        return False
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _dns_resolves(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, None)
        return True
    except Exception:
        return False
