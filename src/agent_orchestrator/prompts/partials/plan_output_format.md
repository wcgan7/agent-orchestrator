Output format (use these exact section headings):

## Objective
- Restate the goal in 1-3 sentences.
- Define explicit success criteria.

## Inputs and Constraints
- List key inputs (requirements, existing systems, dependencies, boundaries).
- List constraints that materially affect implementation choices.

## Scope
- In scope: concrete changes to make now.
- Out of scope: items explicitly not included.

## Files
- `files_to_change`: list existing files expected to be modified.
- `new_files`: list files expected to be created (or `[]` if none).
- For each file, include a one-line purpose.

## Implementation Phases
- Provide an ordered sequence of small, testable phases.
- For each phase, include:
  - objective,
  - concrete changes,
  - acceptance criteria,
  - dependencies on prior phases (if any).

## Verification Plan
- Define how each phase will be validated (tests, lint, typecheck, runtime checks).
- Include edge/negative cases where relevant.
- State expected pass criteria.

## Risks and Mitigations
- List key technical risks and how each will be mitigated.

## Rollback / Fallback
- Describe how to revert or safely disable the change if issues appear.

## Future Work (Optional)
- List potential add-ons/improvements explicitly marked as out-of-scope for this task.
