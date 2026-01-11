from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from ..git_utils import _path_is_allowed
from ..io_utils import _save_data
from ..logging_utils import summarize_pytest_failures
from ..models import VerificationResult
from ..signals import build_allowed_files, extract_failing_paths_from_pytest_log, extract_failures_section, extract_traceback_repo_paths
from ..tasks import _resolve_test_command, _tests_log_path
from ..utils import _now_iso
from ..worker import _capture_test_result_snapshot, _run_command



def run_verify_action(
    *,
    project_dir: Path,
    artifacts_dir: Path,
    run_dir: Path,
    phase: Optional[dict],
    task: dict,
    run_id: str,
    plan_data: dict,
    default_test_command: Optional[str],
    timeout_seconds: int,
) -> VerificationResult:
    phase_id = str(phase.get("id") if phase else task.get("phase_id") or task.get("id"))
    test_command = _resolve_test_command(phase, task, default_test_command)
    allowed_files = build_allowed_files(plan_data)

    logger.info(
        "Running verification: command={} phase={}",
        test_command,
        phase_id,
    )

    # Add robust pytest flags if pytest is the test command
    if test_command:
        if test_command.strip().startswith("pytest") and "--tb=" not in test_command:
            test_command += " --tb=long"
        if test_command.strip().startswith("pytest") and "--disable-warnings" not in test_command:
            test_command += " --disable-warnings"
        if test_command.strip().startswith("pytest") and "-q" not in test_command.split():
            test_command += " -q"

    if not test_command:
        logger.info("Running verification: no test command configured (phase={})", phase_id)
        return VerificationResult(
            run_id=run_id,
            passed=True,
            command=None,
            exit_code=0,
            log_path=None,
            log_tail="No test command configured",
            captured_at=_now_iso(),
            failing_paths=[],
            needs_allowlist_expansion=False,
            error_type=None,
        )
    else:
        logger.info("Running verification: command={} phase={}", test_command, phase_id)

    log_path = _tests_log_path(artifacts_dir, phase_id)
    result = _run_command(
        test_command,
        project_dir,
        log_path,
        timeout_seconds=timeout_seconds,
    )
    snapshot = _capture_test_result_snapshot(
        command=test_command,
        exit_code=result["exit_code"],
        log_path=Path(result["log_path"]),
    )
    log_tail = snapshot.get("log_tail", "")

    # Read full log and extract failures section for robust path extraction
    full_log_text = Path(result["log_path"]).read_text(errors="replace")
    fail_text = extract_failures_section(full_log_text)

    # Write failures excerpt to dedicated file
    fail_path = run_dir / "pytest_failures.txt"
    fail_path.write_text(fail_text)

    # Use fail_text (not log_tail) for failure extraction
    failed_tests = extract_failing_paths_from_pytest_log(fail_text, project_dir)
    trace_paths = extract_traceback_repo_paths(fail_text, project_dir)
    failing_repo_paths = sorted(set(failed_tests) | set(trace_paths))
    EXCLUDE_PREFIXES = (
        ".git/", ".prd_runner/", ".venv/", "venv/", "__pycache__/",
        ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", ".tox/", ".nox/",
    )
    failing_repo_paths = [p for p in failing_repo_paths if not p.startswith(EXCLUDE_PREFIXES)]
    failing_paths = sorted(failing_repo_paths)

    meaningful_allowlist = [p for p in allowed_files if p and p != "README.md"]
    if not meaningful_allowlist:
        outside = []
        needs_expansion = False
    else:
        outside = [p for p in failing_paths if not _path_is_allowed(project_dir, p, allowed_files)]
        needs_expansion = bool(outside)
    
    expansion_paths = outside if needs_expansion else []

    logger.debug("Verify failing_repo_paths: {}", failing_paths)
    logger.debug("Verify expansion_paths: {}", expansion_paths)
    logger.debug(
        "Verify allowlist_used={}",
        allowed_files,
    )
    
    timed_out = bool(result.get("timed_out"))
    passed = result["exit_code"] == 0 and not timed_out
    error_type = "test_timeout" if timed_out else None
    
    if not passed:
        summary = summarize_pytest_failures(log_tail, max_failed=5)
        failed = summary.get("failed") or []
        first_error = summary.get("first_error")

        if failed:
            logger.info("Tests failed: {}", ", ".join(failed))
        elif failing_paths:
            logger.info("Failing files: {}", ", ".join(failing_paths[:5]))

        if first_error:
            logger.info("First error: {}", first_error)

    if needs_expansion:
        logger.info("Verification failed; switching to PLAN_IMPL (expand allowlist)")
        logger.info("Expansion paths: {}", expansion_paths)
    elif not passed:
        logger.info("Verification failed; retrying IMPLEMENT (fix tests)")
    else:
        logger.info("Verification passed")

    event = VerificationResult(
        run_id=run_id,
        passed=passed,
        command=test_command,
        exit_code=int(result["exit_code"]),
        log_path=result["log_path"],
        log_tail=log_tail,
        captured_at=snapshot.get("captured_at", _now_iso()),
        failing_paths=expansion_paths,
        needs_allowlist_expansion=needs_expansion,
        error_type=error_type,
    )
    
    verify_manifest = {
        "run_id": run_id,
        "task_id": str(task.get("id")),
        "phase_id": str(task.get("phase_id") or task.get("id") or ""),
        "step": "verify",
        "command": event.command,
        "exit_code": int(event.exit_code),
        "passed": bool(event.passed),
        "error_type": getattr(event, "error_type", None) or None,
        "timed_out": timed_out,
        "log_path": str(event.log_path),
        "log_tail": str(event.log_tail or ""),
        "captured_at": str(event.captured_at),
        "needs_allowlist_expansion": bool(getattr(event, "needs_allowlist_expansion", False)),
        "failing_paths": list(getattr(event, "failing_paths", []) or []),
        "allowlist_used": allowed_files,
        "failing_repo_paths": failing_paths,
        "expansion_paths": expansion_paths,
        "failures_excerpt_path": str(fail_path),
    }
    _save_data(run_dir / "verify_manifest.json", verify_manifest)
    return event
