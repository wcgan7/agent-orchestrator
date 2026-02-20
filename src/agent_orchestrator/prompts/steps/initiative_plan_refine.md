Refine the existing initiative plan based on feedback and return a full rewritten initiative plan.

Mission:
- Incorporate feedback into a coherent, initiative-level delivery strategy.
- Preserve valid parts of the base initiative plan unless feedback explicitly changes them.
- Produce a standalone initiative plan ready for task decomposition.

Refinement-only rules:
- Planning only: do not modify repository code.
- Do not assume infrastructure, permissions, credentials, teams, or services not explicitly available.
- Keep strategy and sequencing explicit, but avoid implementation-level detail.
- Resolve feedback precisely; avoid unnecessary rewrites outside requested scope.
- If feedback conflicts with constraints, document the conflict in `## Open Questions`.

Scope boundaries:
- This step refines initiative strategy, sequencing, and decomposition guidance.
- Do not produce executable code plans for individual tasks.
- Do not generate tasks directly in this step.

Output format (use these exact section headings):

## Initiative Objective
- Restate objective and measurable success criteria.

## Current State and Context
- Update relevant context needed to justify strategy changes.

## Scope and Boundaries
- In scope.
- Out of scope.

## Delivery Strategy
- Refined initiative-level approach and rationale.
- Key tradeoffs and decisions.

## Workstreams
- Refined workstreams/phases appropriate to initiative complexity.
- For each stream/phase: objective, key deliverables, expected handoff.

## Sequencing and Dependencies
- Refined execution order, key dependencies, and critical path.

## Risk Register
- Updated major risks and mitigations.

## Validation and Quality Strategy
- How progress and quality are validated per stream/phase.

## Rollout and Backout Strategy
- Refined rollout and fallback/backout plan where relevant.

## Task Decomposition Guidance
- Clear instructions for converting this initiative plan into executable tasks.
- Guidance on granularity, dependency encoding, and acceptance criteria.

## Feedback Mapping
- Map each feedback item to where it is addressed in this rewritten plan.
- If any item is intentionally not applied, explain why.

## Open Questions (only if needed)
- Include only unresolved blockers requiring human clarification.

## Future Considerations (Optional)
- Explicitly out-of-scope follow-up opportunities.

Quality bar:
- Return only the final rewritten initiative plan body.
- No conversational preface, no tool logs, no follow-up questions outside Open Questions.
- Return the full rewritten initiative plan, not a delta summary.
