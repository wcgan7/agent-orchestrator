Resolve merge conflicts for this task while preserving intended behavior from all conflicting tasks.

Mission:
- Produce a correct merged result for each conflicted file.
- Preserve both sides' intended, non-conflicting changes.
- Avoid regressions, dropped logic, or accidental behavior changes.

Scope:
- Resolve only the provided merge conflicts.
- Do NOT perform unrelated refactors, renames, formatting churn, or cleanup.
- Do NOT rewrite conflict resolution as a fresh implementation unless strictly required.

Conflict-resolution rules:
- Use the conflict context and task objective summaries for all sides.
- Prefer minimal, targeted edits that remove conflict markers cleanly.
- Keep APIs/contracts stable unless conflict data explicitly requires change.
- Preserve relevant comments/TODOs from both sides when still applicable.
- If both sides changed the same behavior, integrate intent explicitly rather than choosing one side blindly.

Validation requirements:
- Ensure no conflict markers remain (`<<<<<<<`, `=======`, `>>>>>>>`).
- Ensure resulting code is syntactically valid in touched files.
- If available, run targeted checks/tests for touched areas.
- If checks cannot run, state exact blocker.

Output requirements:
- Return only a concise merge-resolution summary:
  - conflicted files resolved,
  - key reconciliation decisions made,
  - validations run and outcomes,
  - residual risks or blocked checks.
