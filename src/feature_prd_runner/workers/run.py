"""Execute a worker provider and capture logs/artifacts."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ..utils import _now_iso
from ..worker import _run_codex_worker
from .config import WorkerProviderSpec


@dataclass(frozen=True)
class WorkerRunResult:
    provider: str
    prompt_path: str
    stdout_path: str
    stderr_path: str
    start_time: str
    end_time: str
    runtime_seconds: int
    exit_code: int
    timed_out: bool
    no_heartbeat: bool
    response_text: str = ""


def _run_ollama_generate(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    run_dir: Path,
    timeout_seconds: int,
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
) -> WorkerRunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = run_dir / "prompt.txt"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    prompt_path.write_text(prompt)

    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    options: dict[str, object] = {}
    if temperature is not None:
        options["temperature"] = float(temperature)
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)
    if options:
        payload["options"] = options

    url = endpoint.rstrip("/") + "/api/generate"
    start_iso = _now_iso()
    start = time.monotonic()
    timed_out = False
    response_text_parts: list[str] = []

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Use a small socket timeout so we can enforce overall timeout_seconds.
    socket_timeout_seconds = 15

    try:
        with urllib.request.urlopen(request, timeout=socket_timeout_seconds) as resp, open(
            stdout_path, "w", encoding="utf-8"
        ) as out, open(stderr_path, "w", encoding="utf-8") as err:
            while True:
                if time.monotonic() - start > timeout_seconds:
                    timed_out = True
                    err.write(f"[runner] Ollama timed out after {timeout_seconds}s\n")
                    break

                try:
                    line = resp.readline()
                except socket.timeout:
                    continue

                if not line:
                    break
                try:
                    obj = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    # Best-effort: log raw line and continue.
                    chunk = line.decode("utf-8", errors="replace")
                    err.write(chunk)
                    err.flush()
                    continue

                chunk = str(obj.get("response") or "")
                if chunk:
                    response_text_parts.append(chunk)
                    out.write(chunk)
                    out.flush()

                if bool(obj.get("done")):
                    break
    except urllib.error.HTTPError as exc:
        stderr_path.write_text(f"[runner] Ollama HTTP error: {exc.code} {exc.reason}\n")
        timed_out = False
    except urllib.error.URLError as exc:
        stderr_path.write_text(f"[runner] Ollama URL error: {exc.reason}\n")
        timed_out = False
    except Exception as exc:
        stderr_path.write_text(f"[runner] Ollama error: {exc.__class__.__name__}: {exc}\n")
        timed_out = False

    end_iso = _now_iso()
    runtime_seconds = int(time.monotonic() - start)
    response_text = "".join(response_text_parts)

    exit_code = 0
    if timed_out:
        exit_code = 124
    else:
        # Treat any stderr output as an error.
        try:
            if stderr_path.exists() and stderr_path.read_text(encoding="utf-8").strip():
                exit_code = 1
        except Exception:
            exit_code = 1

    return WorkerRunResult(
        provider=f"ollama:{model}",
        prompt_path=str(prompt_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        start_time=start_iso,
        end_time=end_iso,
        runtime_seconds=runtime_seconds,
        exit_code=exit_code,
        timed_out=timed_out,
        no_heartbeat=False,
        response_text=response_text,
    )


def run_worker(
    *,
    spec: WorkerProviderSpec,
    prompt: str,
    project_dir: Path,
    run_dir: Path,
    timeout_seconds: int,
    heartbeat_seconds: int,
    heartbeat_grace_seconds: int,
    progress_path: Path,
    expected_run_id: Optional[str] = None,
    on_spawn: Optional[Callable[[int], None]] = None,
) -> WorkerRunResult:
    """Run the selected provider and return a normalized run result."""
    if spec.type == "codex":
        logger.info("Starting Codex worker provider='{}' (timeout={}s)", spec.name, timeout_seconds)
        run_result = _run_codex_worker(
            command=str(spec.command),
            prompt=prompt,
            project_dir=project_dir,
            run_dir=run_dir,
            timeout_seconds=timeout_seconds,
            heartbeat_seconds=heartbeat_seconds,
            heartbeat_grace_seconds=heartbeat_grace_seconds,
            progress_path=progress_path,
            expected_run_id=expected_run_id,
            on_spawn=on_spawn,
        )
        return WorkerRunResult(
            provider=spec.name,
            prompt_path=str(run_result.get("prompt_path") or ""),
            stdout_path=str(run_result.get("stdout_path") or ""),
            stderr_path=str(run_result.get("stderr_path") or ""),
            start_time=str(run_result.get("start_time") or _now_iso()),
            end_time=str(run_result.get("end_time") or _now_iso()),
            runtime_seconds=int(run_result.get("runtime_seconds") or 0),
            exit_code=int(run_result.get("exit_code") or 0),
            timed_out=bool(run_result.get("timed_out")),
            no_heartbeat=bool(run_result.get("no_heartbeat")),
            response_text="",
        )

    if spec.type == "ollama":
        logger.info(
            "Starting Ollama worker provider='{}' model='{}' (timeout={}s)",
            spec.name,
            spec.model,
            timeout_seconds,
        )
        return _run_ollama_generate(
            endpoint=str(spec.endpoint),
            model=str(spec.model),
            prompt=prompt,
            run_dir=run_dir,
            timeout_seconds=timeout_seconds,
            temperature=spec.temperature,
            num_ctx=spec.num_ctx,
        )

    raise ValueError(f"Unsupported worker type '{spec.type}'")
