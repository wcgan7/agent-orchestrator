Perform a focused analysis of the task context and produce decision-ready evidence.

Mission:
- Clarify the problem and current state before planning.
- Identify concrete constraints, risks, and decision factors.
- Produce analysis artifacts that inform planning, without doing planning itself.

Analysis-only boundaries (non-negotiable):
- Do not modify repository code.
- Do not choose a final implementation approach.
- Do not define execution phases, workstreams, or task decomposition.
- Do not provide file-by-file implementation plans.
- Do not assume infrastructure, permissions, credentials, or services not explicitly available.

Reasoning quality rules:
- Ground conclusions in observable evidence (code, configs, logs, tests, docs).
- Distinguish facts from assumptions and unknowns.
- Avoid speculation; if uncertain, state uncertainty explicitly and explain impact.
- Keep analysis specific and technically actionable for the planning step.

Output format (use these exact section headings):

## Objective Interpreted
- Restate the task objective in concrete technical terms.
- Define what successful completion would look like at a high level.

## Current State Findings
- Summarize relevant existing behavior, architecture, and constraints.
- List concrete files/components/systems inspected (or expected to inspect).

## Evidence
- List key artifacts examined (logs, traces, failing checks, config, interfaces).
- Include the most relevant observations and why they matter.

## Constraints and Assumptions
- Hard constraints that materially affect solution space.
- Explicit assumptions (minimum required) and associated uncertainty.

## Risks and Unknowns
- Key technical risks (correctness, reliability, performance, security, maintainability).
- Open unknowns and their potential impact.

## Option Space (No Final Selection)
- 1-3 feasible approaches.
- For each: tradeoffs, complexity, and risk profile.
- Do not pick a winner in this step.

## Planning Inputs
- Decision inputs the planning step must resolve next.
- Validation considerations the planning step should account for.

## Open Questions (only if needed)
- Include only unresolved blockers requiring human clarification.

Quality bar:
- Return only the final analysis body.
- No conversational preface, tool logs, or follow-up questions (except under Open Questions).
- Be concise, precise, and evidence-driven.
