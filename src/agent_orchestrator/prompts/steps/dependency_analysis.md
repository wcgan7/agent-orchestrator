Analyze execution dependencies among pending tasks.

Mission:
- Infer only strict "must-run-before" dependencies.
- Maximize safe parallelism by avoiding speculative edges.
- Provide concise rationale and confidence for every inferred edge.

Decision standard (non-negotiable):
- Add an edge only when task `to` cannot be completed correctly without artifacts
  created by task `from`.
- If dependency is uncertain, omit the edge.
- Prefer false negatives over false positives.

Valid dependency signals:
- Task `to` requires an API/schema/module/table created by task `from`.
- Task `to` consumes outputs/artifacts explicitly introduced by task `from`.
- Task `to` is blocked by ordering constraints that are technically mandatory.

Invalid dependency signals:
- Shared area/theme without hard producer/consumer requirement.
- Similar subsystem ownership or sequencing preference.
- "Might be easier after X" reasoning.
- Dependencies on code that already exists in the repository.

Output requirements:
- Return JSON only; no conversational text or markdown fences.
- Respond with valid JSON matching this schema:
  `{"edges": [{"from": "task_id_first", "to": "task_id_depends", "reason": "why", "confidence": "high|medium", "evidence": "artifact reference"}]}`
- Return edges only for mandatory dependencies.
- For each edge include:
  - `from`
  - `to`
  - `reason` (concise, concrete, evidence-based)
  - `confidence` (`high` or `medium`; use `high` only when strongly evidenced)
  - `evidence` (short artifact reference: file/interface/schema/etc.)
- If no mandatory dependencies exist, return an empty edges list.
