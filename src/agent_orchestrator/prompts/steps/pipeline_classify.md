Classify this task into the most suitable pipeline.

Return JSON only with this exact shape:
{
  "pipeline_id": "string",
  "confidence": "high" | "low",
  "reason": "short explanation"
}

Rules:
- `pipeline_id` must be one of the allowed pipeline IDs provided below.
- Choose `high` only when the task intent is clear and specific for one pipeline.
- Choose `low` when intent is ambiguous, underspecified, or could fit multiple pipelines.
- Keep `reason` concise and concrete.
- Do not include markdown, code fences, or extra keys.
