Refine the existing plan based on feedback and return a full rewritten plan.

Mission:
- Incorporate feedback into a coherent, implementation-ready plan.
- Preserve valid parts of the base plan unless feedback explicitly changes them.
- Produce a standalone plan that can be executed without prior chat context.

Refinement rules:
- Planning only: do not modify repository code.
- Do not assume infrastructure, permissions, credentials, or services not explicitly available.
- Keep phases ordered, scoped, and independently testable.
- Resolve feedback precisely; avoid broad rewrites unless required.
- If feedback conflicts with constraints, document the conflict in `## Open Questions`.

Additional required sections for refinement:

## Feedback Mapping
- Map each feedback item to where it is addressed in this rewritten plan.
- If any item is intentionally not applied, explain why.

## Open Questions (only if needed)
- Include only unresolved blockers that require human clarification.

Quality bar:
- Return only the final markdown plan body.
- No conversational preface, no tool logs, and no follow-up questions unless under `## Open Questions`.
- Return the full rewritten plan, not a delta summary.
