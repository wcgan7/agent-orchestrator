"""Helpers for parsing non-agentic worker outputs."""

from __future__ import annotations

import json
import re
from typing import Any


class WorkerOutputError(ValueError):
    pass


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a worker response.

    The worker is instructed to return JSON only, but we defensively handle:
    - Markdown fenced blocks
    - Extra leading/trailing text
    """
    candidate = text.strip()
    if not candidate:
        raise WorkerOutputError("Worker returned empty output")

    m = _JSON_FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # Attempt to salvage by extracting the outermost {...} block.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise WorkerOutputError("Worker output is not valid JSON") from None
        try:
            obj = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise WorkerOutputError(f"Worker output JSON parse error: {exc}") from None

    if not isinstance(obj, dict):
        raise WorkerOutputError("Worker output JSON must be an object")
    return obj


_DIFF_FENCE_RE = re.compile(r"```(?:diff)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def normalize_patch_text(patch_text: str) -> str:
    text = (patch_text or "").strip()
    if not text:
        return ""
    m = _DIFF_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    return text + ("\n" if not text.endswith("\n") else "")


def extract_paths_from_unified_diff(patch_text: str) -> list[str]:
    """Best-effort path extraction from a unified diff.

    Supports:
    - `diff --git a/x b/x` lines
    - `+++ b/x` lines (new path)
    """
    text = patch_text or ""
    paths: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        line = line.rstrip("\n")
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_path = parts[2]
                b_path = parts[3]
                for token in (a_path, b_path):
                    if token.startswith("a/") or token.startswith("b/"):
                        token = token[2:]
                    token = token.strip()
                    if token and token != "/dev/null" and token not in seen:
                        seen.add(token)
                        paths.append(token)
            continue
        if line.startswith("+++ "):
            token = line[4:].strip()
            if token.startswith("b/"):
                token = token[2:]
            if token and token != "/dev/null" and token not in seen:
                seen.add(token)
                paths.append(token)

    return paths

