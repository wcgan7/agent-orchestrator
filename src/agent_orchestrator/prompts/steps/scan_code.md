Perform a code-level security scan for this task.

Mission:
- Identify concrete code and configuration security weaknesses in the repository.
- Produce evidence-backed findings for downstream reporting and task generation.

Scope:
- Code/config layer only (authn/authz, input handling, injection, secrets handling, crypto usage, unsafe deserialization, insecure defaults).
- Do NOT perform dependency/advisory inventory in this step.
- Do NOT propose implementation plans or choose remediation strategy.

Rules:
- Report only evidence-backed issues tied to specific files/locations.
- Do not invent exploit paths or impact beyond observed evidence.
- If certainty is limited, state uncertainty explicitly.
- Focus on materially actionable security findings, not style issues.

Output requirements:
- Return concise, deduplicated, actionable findings with:
  - severity,
  - category,
  - summary,
  - file/location,
  - concrete evidence observed,
  - uncertainty notes when applicable.
