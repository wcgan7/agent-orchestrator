"""Worker diagnostics (list/test)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import WorkerProviderSpec


def test_worker(spec: WorkerProviderSpec) -> tuple[bool, str]:
    if spec.type == "codex":
        from shutil import which

        if not spec.command:
            return False, "Missing command"
        exe = spec.command.split()[0]
        if which(exe):
            return True, f"Found executable in PATH: {exe}"
        return False, f"Executable not found in PATH: {exe}"

    if spec.type == "ollama":
        if not spec.endpoint or not spec.model:
            return False, "Missing endpoint/model"
        url = spec.endpoint.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return False, f"Ollama HTTP error: {exc.code} {exc.reason}"
        except urllib.error.URLError as exc:
            return False, f"Ollama URL error: {exc.reason}"
        try:
            obj: Any = json.loads(raw)
        except json.JSONDecodeError:
            return False, "Ollama /api/tags did not return JSON"
        models = obj.get("models")
        if not isinstance(models, list):
            return False, "Ollama /api/tags JSON missing 'models' list"
        names = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
        if spec.model in names:
            return True, f"Ollama reachable; model available: {spec.model}"
        if names:
            return False, f"Ollama reachable, but model not found: {spec.model} (available: {', '.join(names[:10])})"
        return False, f"Ollama reachable, but no models found (missing pull?)"

    return False, f"Unsupported worker type: {spec.type}"

