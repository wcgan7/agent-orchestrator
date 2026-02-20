Perform a focused analysis of the task context and produce implementation-ready guidance.

Mission:
- Clarify the problem and current state before planning or implementation.
- Identify key constraints, risks, and decision points.
- Produce analysis that enables the next step to execute with minimal ambiguity.

Analysis rules:
- Analysis only: do not modify repository code.
- Do not assume infrastructure, permissions, credentials, or services not explicitly available.
- Ground conclusions in observable repo/task context; avoid speculation.
- Be specific and actionable; avoid generic statements.
- If information is missing, state exactly what is missing and its impact.

Output format (use these exact section headings):

## Objective Interpreted
- Restate the task objective in concrete technical terms.
- Define what successful completion would look like.

## Current State
- Summarize relevant existing code, architecture, and behavior.
- List concrete files/components inspected (or expected to be inspected).

## Constraints and Assumptions
- List hard constraints affecting approach.
- List assumptions being made; keep them minimal and explicit.

## Gaps and Risks
- Identify implementation gaps between current and target state.
- Identify key risks (correctness, reliability, performance, security, maintainability).

## Options Considered
- Provide 1-3 feasible approaches.
- For each option: tradeoffs, complexity, and risk profile.

## Recommended Direction
- Choose one option and justify why it is best for this task.
- Describe the intended technical approach at a high level.

## Execution Inputs for Next Step
- Provide concrete inputs for the next step (typically `plan` or `implement`):
  - likely files to change,
  - key interfaces/contracts to preserve or update,
  - validation/check strategy to prioritize.

## Open Questions (only if needed)
- Include only unresolved blockers requiring human clarification.

Quality bar:
- Return only the final analysis body.
- No conversational preface, tool logs, or follow-up questions (except under Open Questions).
- Keep it concise but complete enough that the next step can proceed immediately.
