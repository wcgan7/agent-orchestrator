Review the implementation and list findings.
Each finding must include a severity (critical / high / medium / low).
Evaluate every acceptance criterion explicitly. Provide concrete
evidence tied to files and diffs — do not speculate. Do not down-rank
findings.
SCOPE: Only flag issues introduced or directly affected by THIS task's
changes. Pre-existing issues in unchanged code are out of scope.
Missing project-wide tooling (e.g. linter configs not present before
this task) is out of scope unless the task description requires it.
CONSISTENCY: If previous review cycles are shown below, do not
contradict earlier decisions. Do not reverse an approach you
previously accepted unless new evidence of a critical defect emerged.
Focus on remaining open issues.
CONVERGENCE: Each cycle must move closer to approval. Do not raise
new low-severity or cosmetic findings after the first cycle unless
they concern code changed since the last review.
If the change affects user-facing behavior, CLI usage, configuration,
or API surface, verify that README.md and relevant documentation were
updated. Raise a medium-severity finding if documentation is stale or
missing.