Run the FULL test suite and all configured verification commands for this task.

Primary objective:
- Targeted tests have already passed during the implementation step.
- You MUST now run the FULL test suite using the exact project commands listed below.
- Do NOT subset, scope down, or run individual test files — run everything.
- If any test fails, fix the root cause and re-run the FULL suite.
- Do NOT fix tests by weakening assertions or deleting tests unless the new behavior is genuinely correct per the task objective.
- Produce reliable evidence of what passed, failed, skipped, or was blocked by environment constraints.

Execution rules:
- Do not fabricate results.
- Do not silently skip checks.
- If a command cannot run, report the exact reason (missing tool/config/no tests/environment constraint).
- Capture concise root-cause clues for failing commands.
- Run a short preflight before heavy checks:
  - verify local toolchain binaries exist (for example in `node_modules/.bin`),
  - verify required environment variables for the task are present (for example `DATABASE_URL`),
  - verify required runtime services are reachable (for example Docker daemon for compose-backed DB checks).
- For Prisma workflows, do **not** use bare `npx prisma ...` that may fetch a different major version.
  - Prefer repository scripts (for example `npm run prisma:*`) or explicit local binary (`./node_modules/.bin/prisma ...`).
  - If local Prisma CLI is missing, report `environment` with `reason_code=tool_missing` and stop.
- If preflight fails, report one primary blocker (highest priority):
  1. missing required env var (`config_missing`)
  2. runtime/service unavailable (for example Docker daemon) (`infrastructure`)
  3. local toolchain missing (`tool_missing`)
  4. network registry unavailable (`infrastructure`)

Reporting requirements:
- For each command attempted, report:
  - command,
  - exit code,
  - outcome hint (`pass|fail|skip|environment`),
  - brief evidence (key error lines).
- Keep output factual and concise.

Scope boundary:
- Verify execution outcomes only.
- Do NOT perform acceptance-criteria judgment or approval decisions; that belongs to review.

Output requirements:
- Return JSON only; no conversational text or markdown fences.
- Respond with valid JSON matching this schema:
  `{"status": "pass|fail|skip|environment", "reason_code": "assertion_failure|type_error|lint_violation|tool_missing|config_missing|no_tests|permission_denied|resource_limit|os_incompatibility|infrastructure|unknown", "summary": "one-line summary of results"}`
