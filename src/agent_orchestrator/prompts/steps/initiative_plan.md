Create an initiative-level delivery plan for a broad body of work.

Mission:
- Define a coherent strategy for delivering a broad initiative
  (e.g. major feature program, large refactor, migration, platform extension).
- Produce a plan suitable for decomposition into multiple executable tasks.
- Planning only: do not implement code in this step.

Non-negotiable rules:
- Planning does not modify repository code.
- Do not assume infrastructure, permissions, credentials, teams, or services
  that are not explicitly available.
- Prefer incremental delivery and risk reduction over big-bang rollout.
- Keep the plan actionable and decomposition-ready.

Output format (use these exact section headings):

## Initiative Objective
- Restate business/technical objective and target outcomes.
- Define measurable success criteria.

## Current State and Context
- Summarize relevant architecture, constraints, and known pain points.
- Identify existing systems/interfaces that shape delivery choices.

## Scope and Boundaries
- In scope.
- Out of scope.

## Delivery Strategy
- High-level approach and rationale.
- Key tradeoffs and guiding decisions.

## Workstreams
- Organize work into clear streams/phases as needed for this initiative.
- For each stream/phase, include objective, key deliverables, and expected handoff.
- For small initiatives, a single stream/phase is valid.

## Sequencing and Dependencies
- Describe execution order and key dependencies across streams/phases.
- Highlight critical path and blocking relationships.

## Risk Register
- Key technical/delivery risks.
- Mitigation and fallback for each risk.

## Validation and Quality Strategy
- Define how each stream/phase will be validated.
- Include required checks where relevant (tests, lint, typecheck, migration safety,
  performance/security checks).

## Rollout and Backout Strategy
- Describe rollout approach where relevant.
- Describe safe rollback/backout strategy for high-risk phases.

## Task Decomposition Guidance
- Explain how this initiative plan should be converted into executable tasks.
- Provide guidance on task granularity, dependency encoding, and acceptance criteria.

## Open Questions (only if needed)
- Include only unresolved blockers requiring human clarification.

## Future Considerations (Optional)
- Explicitly out-of-scope follow-up opportunities.

Quality bar:
- Return only the final initiative plan body.
- No conversational preface, tool logs, or follow-up questions outside Open Questions.
- Be specific enough that `generate_tasks` can produce high-quality, dependency-aware subtasks.
