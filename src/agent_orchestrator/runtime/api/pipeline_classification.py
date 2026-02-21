"""Pipeline classification helper utilities for API routes."""

from __future__ import annotations

import json
from typing import Literal

from ...pipelines.registry import PipelineRegistry
from .schemas import PipelineClassificationResponse


def _canonical_task_type_for_pipeline(registry: PipelineRegistry, pipeline_id: str) -> str:
    template = registry.get(pipeline_id)
    if template.task_types:
        return str(template.task_types[0])
    return "feature"


def _normalize_pipeline_classification_output(
    *,
    summary: str | None,
    allowed_pipelines: list[str],
    registry: PipelineRegistry,
) -> PipelineClassificationResponse:
    allowed_set = set(allowed_pipelines)
    pipeline_id = "feature"
    confidence: Literal["high", "low"] = "low"
    reason = "Pipeline auto-classification was inconclusive."

    if isinstance(summary, str) and summary.strip():
        try:
            payload = json.loads(summary)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            raw_pipeline = str(payload.get("pipeline_id") or "").strip()
            raw_confidence = str(payload.get("confidence") or "").strip().lower()
            raw_reason = str(payload.get("reason") or "").strip()
            if raw_pipeline in allowed_set:
                pipeline_id = raw_pipeline
            if raw_confidence == "high":
                confidence = "high"
            elif raw_confidence == "low":
                confidence = "low"
            if raw_reason:
                reason = raw_reason[:300]

    if pipeline_id not in allowed_set:
        pipeline_id = "feature"
        confidence = "low"
        reason = "Classifier returned an unknown pipeline."
    task_type = _canonical_task_type_for_pipeline(registry, pipeline_id)
    return PipelineClassificationResponse(
        pipeline_id=pipeline_id,
        task_type=task_type,
        confidence=confidence,
        reason=reason,
        allowed_pipelines=allowed_pipelines,
    )
