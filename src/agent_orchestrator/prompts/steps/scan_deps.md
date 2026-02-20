Perform a dependency security scan for this task.

Mission:
- Identify concrete dependency-related security risks from manifests, lockfiles, and dependency metadata.
- Produce evidence-backed findings for downstream reporting and task generation.

Scope:
- Dependency layer only: direct/transitive packages, versions, advisories, vulnerable ranges, and exposure path.
- Do NOT perform code-level vulnerability analysis in this step.
- Do NOT propose implementation plans or choose remediation strategy.

Rules:
- Report only evidence-backed issues.
- Do not invent CVEs, versions, exploitability, or package relationships.
- If advisory data/tooling is unavailable, state that as an environment/data limitation.
- Prefer precise package coordinates (ecosystem/package/version) and affected files.

Output requirements:
- Return concise, deduplicated, actionable findings with:
  - severity,
  - category,
  - summary,
  - file/location (manifest/lockfile),
  - evidence (advisory ID/source or observed version/range mismatch),
  - uncertainty notes when applicable.
