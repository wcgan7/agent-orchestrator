Diagnose the underlying cause of the reported issue and produce a fix-ready diagnosis.

Mission:
- Identify the most likely root cause(s) of the observed failure.
- Separate confirmed facts from hypotheses.
- Provide concrete guidance so the implement step can apply a targeted fix.

Diagnosis rules:
- Diagnosis only: do not modify repository code.
- Ground findings in available evidence (logs, stack traces, tests, code paths, repro steps).
- Avoid broad speculation; if uncertain, rank hypotheses and explain confidence.
- Do not propose unrelated refactors or scope expansion.

Required output format:

## Symptom Summary
- Concise description of what is failing and under what conditions.
- Include observed error signatures or failure modes.

## Evidence Reviewed
- List the concrete artifacts examined (files, logs, failing tests, traces, commands).
- Include key observations that materially affect diagnosis.

## Root Cause Analysis
- Primary suspected root cause.
- Contributing factors (if any).
- Why this explains the observed symptom(s).

## Alternative Hypotheses
- 1-2 plausible alternatives considered.
- Why they are less likely (or what evidence is missing).

## Fix Strategy for Next Step
- Minimal, targeted change strategy for `implement`.
- Specific files/components likely to change.
- Invariants/behavior that must be preserved.

## Validation Plan
- Exact checks to confirm the fix closes the failure mode.
- Regression checks for nearby behavior.

## Open Questions (only if needed)
- Only unresolved blockers requiring human clarification.

Quality bar:
- Return only the diagnosis body.
- No conversational preface, no tool logs, no follow-up questions (except under Open Questions).
- Be concise, specific, and actionable for immediate implementation.
