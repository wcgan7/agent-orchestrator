Review the implementation against the task objective and acceptance criteria, then produce actionable findings.

Primary objective:
- Use `.workdoc.md` as the source of truth for intent.
- Read `## Plan` to understand the implementation objective, scope, and acceptance criteria.
- Determine whether the delivered changes fully satisfy those criteria.

Review goals:
- Confirm objective coverage: required behavior is implemented end-to-end.
- Confirm acceptance criteria coverage: each criterion is met or a finding is raised.
- Confirm verification evidence supports correctness.
- Confirm changed user-facing/API/config/setup behavior is documented.

Scope rules:
- In scope: issues introduced by, or directly affected by, this task's changes.
- In scope: missing or incomplete implementation of planned objective/acceptance criteria.
- In scope: correctness, reliability, security, performance regressions, contract/API breakage, and stale/missing docs for changed behavior.
- Out of scope: pre-existing issues in unchanged code unless this task made them worse.
- Out of scope: missing project-wide tooling/config that did not exist before, unless explicitly required by the task.

Severity calibration logic:
- critical:
  - Fails core objective in a way that risks data loss, security exposure, or severe correctness failure.
  - A required acceptance criterion is violated with high user/system impact.
- high:
  - Core objective is materially incomplete or functionally broken.
  - A required acceptance criterion is unmet and blocks expected task outcome.
- medium:
  - Objective is mostly met, but there is a meaningful reliability/quality/doc gap that should be fixed before approval.
  - Required docs for changed behavior are stale/missing.
- low:
  - Minor, non-blocking issue with clear improvement value.
  - Do not add low-severity noise in later cycles unless tied to newly changed code.

Consistency and convergence:
- If prior review cycles are provided, do not contradict previously accepted decisions unless new critical evidence appears.
- Do not re-raise findings already resolved.
- After the first cycle, do not introduce new cosmetic/low-value findings unless they stem from newly changed code.

Documentation check:
- If changed code affects user-facing behavior, CLI, configuration, setup, or API surface, verify documentation updates (including `README.md`).
- Raise at least a medium-severity finding when required documentation is missing or stale.

Output requirements:
- Return findings only; no conversational text.
- If no material issues remain and objective/acceptance criteria are fully satisfied, return zero findings.
- Each finding must include:
  - severity (`critical|high|medium|low`)
  - category
  - summary
  - file
  - line (if known)
  - suggested_fix
- Findings must be specific, evidence-based, and reproducible.
- Do not speculate. Do not suppress or down-rank valid findings.
