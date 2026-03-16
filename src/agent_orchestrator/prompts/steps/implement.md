Implement the task completely and safely using the working document as the source of truth.

Core rules:
- Complete the step end-to-end; do not leave partial work.
- Do not leave stubs, TODOs, placeholder comments, empty bodies, or pass-through no-ops.
- Solve from first principles; avoid shortcut, band-aid, or hacky patch fixes.
- Preserve existing behavior unless the task explicitly requires changing it.
- Keep changes minimal and targeted; avoid unrelated refactors.
- Never undo or overwrite unrelated changes in the repository.

Execution requirements:
- Read `.workdoc.md` and implement against the `## Plan` scope.
- Update `## Implementation Log` with completed work, key decisions, and justified deviations.
- Keep repository state coherent and runnable throughout the step.

Style:
- Follow the style guidelines and language-specific fallback defaults provided below in this prompt.

Validation (targeted scope only):
- Run tests directly related to the files you changed. Do NOT run the full test suite.
- Run lint and typecheck on changed files only.
- Fix any failures introduced by your changes before finishing.
- The full test suite will be run in a separate verification step after implementation.

Documentation:
- If behavior, API, CLI, configuration, or setup changes, update relevant documentation in the same step, including `README.md`.
- If the repository maintains a changelog, update it for user-visible behavior changes.

Completion output:
- Return only a concise implementation summary:
  - what changed,
  - why it changed,
  - what validation was run and results.
- Do not include conversational prefaces or follow-up questions.
