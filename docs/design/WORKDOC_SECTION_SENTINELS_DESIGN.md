# Workdoc Section Sentinel Design

## Purpose
Define a deterministic, machine-readable section format for workdoc sync so worker-authored markdown cannot corrupt, clip, or mis-bound section merges.

## Problem
Current section sync relies on markdown heading detection (`## ...`) to find section bounds. This is brittle when worker output includes additional headings and can lead to dropped or truncated context.

## Goals
- Make section detection deterministic and independent of markdown heading style.
- Preserve append-only retry context across all attempts.
- Keep existing workdocs readable by humans.
- Maintain backward compatibility with workdocs that predate sentinels.

## Non-Goals
- No immediate migration of all historical workdocs.
- No major template redesign beyond adding sentinel comments.

## Sentinel Format
Use paired HTML comments around each orchestrator-managed section body.

Example:

```md
## Plan
<!-- WORKDOC:SECTION plan START -->
_Pending: will be populated by the plan step._
<!-- WORKDOC:SECTION plan END -->
```

Rules:
- Sentinel IDs are lowercase snake_case and stable.
- IDs map 1:1 to canonical step sections (`plan`, `implement`, `verify`, `benchmark`, `review_findings`, etc.).
- `START`/`END` pairs must be unique per section ID.
- Sentinels are structural and must not be removed by orchestrator writes.
- Add lightweight schema marker at init: `<!-- WORKDOC:SCHEMA v1 -->` (observability/migration hint only, not a hard gate).

## Parsing Strategy
Primary strategy (new docs):
- Locate `START` and `END` markers for the target section ID.
- Replace only content between those markers.
- Never rely on `##` heading scanning when sentinel pair exists.

Fallback strategy (legacy docs):
- If sentinel pair does not exist, use existing heading-based logic.
- If heading-based bounds are ambiguous/missing:
  - If step summary exists: append attempt-scoped summary fallback.
  - If step summary is empty: raise explicit diagnostic error (block task), never silently no-op.

## Sync Decision Matrix
For each post-step sync, classify target section structure in canonical/worktree as:
- `VALID_SENTINEL_PAIR`
- `MISSING_SECTION`
- `MALFORMED`

Required outcomes:
- `canonical=VALID_SENTINEL_PAIR`, `worktree=VALID_SENTINEL_PAIR`:
  - Perform sentinel-bounded section merge.
- `canonical=VALID_SENTINEL_PAIR`, `worktree=VALID_SENTINEL_PAIR` but sentinel IDs do not match target section:
  - Treat as `MALFORMED`; if summary exists, append fallback summary; otherwise block with explicit diagnostic.
- `canonical=VALID_SENTINEL_PAIR`, `worktree=MISSING_SECTION|MALFORMED`:
  - If summary exists: append attempt-scoped fallback summary.
  - If summary missing: block task with explicit diagnostic.
- `canonical=MISSING_SECTION`, `worktree=VALID_SENTINEL_PAIR`:
  - Treat as legacy transition path: use legacy heading fallback rules.
  - On ambiguous legacy bounds: fallback append if summary exists, else block.
- `canonical=MALFORMED` (any worktree state):
  - Block task with explicit diagnostic (source-of-truth is invalid).
- Any parser/runtime exception:
  - Block task explicitly; never silently skip.

## Write Semantics
Worker-changed section path:
- If both canonical and worktree contain matching sentinel pair for section ID, merge section body by sentinel bounds.
- Preserve all content outside sentinel bounds verbatim.

Fallback append path:
- Continue attempt-scoped append format (`### Attempt N`, `### Fix Cycle N`).
- Emit `workdoc.updated` with an explicit `sync_mode` (e.g., `sentinel_merge`, `fallback_append`, `fallback_unstructured`).

## Repeatable Step Numbering
For any step that can execute multiple times in one task lifecycle, append numbered sub-blocks within the same section:
- Normal repeatable steps: `### Attempt N`
- Fix loop step (`implement_fix`): `### Fix Cycle N`
- Review loop: `### Review Cycle N`

Do not overwrite prior entries for repeatable steps.

## Retry Behavior Compatibility
- Retry attempt markers remain append-only and outside section sentinels.
- Existing retry invariants are unchanged: no reinitialize on retry, no clearing historical context.
- Review cycle entries remain append-only outside sentinel-managed section replacement in v1.

## Normative Step to Section Mapping
This table is authoritative for section resolution and sentinel IDs.

- `plan` -> `plan` -> `## Plan`
- `initiative_plan` -> `plan` -> `## Plan`
- `analyze` -> `analysis` -> `## Analysis`
- `diagnose` -> `analysis` -> `## Analysis`
- `scan_deps` -> `dependency_scan_findings` -> `## Dependency Scan Findings`
- `scan_code` -> `code_scan_findings` -> `## Code Scan Findings`
- `generate_tasks` -> `generated_tasks` -> `## Generated Tasks`
- `profile` -> `profiling_baseline` -> `## Profiling Baseline`
- `implement` -> `implementation_log` -> `## Implementation Log`
- `prototype` -> `implementation_log` -> `## Implementation Log`
- `implement_fix` -> `fix_log` -> `## Fix Log`
- `verify` -> `verification_results` -> `## Verification Results`
- `benchmark` -> `verification_results` -> `## Verification Results`
- `reproduce` -> `verification_results` -> `## Verification Results`
- `report` -> `final_report` -> `## Final Report`
- `review` -> `review_findings` -> `## Review Findings`

Explicit non-sync skip steps:
- `plan_refine`
- `initiative_plan_refine`

V1 policy:
- Keep existing headings unchanged for compatibility.
- `verify`/`benchmark`/`reproduce` intentionally share one section in v1.

Rules:
- `workdoc_section_for_step` must resolve through this mapping (no free-form IDs).
- Sentinel IDs are stable API and cannot be renamed without migration.
- Retry markers are lifecycle blocks and remain outside section mappings.

## Template Changes
Add sentinels to all section blocks in the initialized workdoc templates.

Minimum required in Phase A:
- Sections currently used by `workdoc_section_for_step`.
- Review findings section block.

## Backward Compatibility
- Legacy workdocs without sentinels continue to function.
- Sync automatically chooses sentinel strategy when available.
- Optional later migration can inject sentinels into legacy docs, but is not required for initial rollout.

## Error Handling
Treat these as explicit errors (block + diagnostic):
- Missing `END` for an existing `START` marker.
- Duplicate section IDs in canonical.
- Mismatched or malformed marker format.

Do not silently continue for structural sentinel errors.

Malformed worktree handling defaults:
- `canonical=VALID`, `worktree=MALFORMED|MISSING_SECTION`:
  - If step summary is non-empty: continue via fallback append.
  - If step summary is empty: block with explicit diagnostic.
- `canonical=VALID`, `worktree=VALID` but wrong sentinel section ID:
  - Treat as malformed (same fallback/block rule above).
- `canonical=MALFORMED`:
  - Always block.

Required diagnostic metadata when fallback or block occurs:
- `workdoc_sync_error_type` (for example: `worktree_malformed`, `worktree_missing_section`, `canonical_malformed`, `section_id_mismatch`)
- `workdoc_sync_mode` (`fallback_append` or `blocked_invalid_structure`)
- `workdoc_sync_step`
- `workdoc_sync_attempt`

Event behavior:
- Fallback append should emit `workdoc.updated` with `sync_mode` and `reason`.
- Blocking should continue to emit `task.blocked` with explicit error text.

## Testing Plan
Add/extend tests for:
- Sentinel-based merge preserves full content when worker includes internal `##` headings.
- Legacy heading fallback still works for old docs.
- Ambiguous legacy bounds route to fallback append (or explicit error when no summary).
- Structural sentinel corruption triggers explicit block diagnostic.
- Shared verification section behavior stays correct for `verify`/`benchmark`/`reproduce` (single section, append-only history).
- Event payload includes `sync_mode` and remains backward compatible.

## Rollout Phases
1. Introduce sentinel parser + writer path behind auto-detect (no flag required).
2. Update template generation to include sentinels.
3. Add fallback hardening to eliminate silent no-op behavior.
4. Add observability fields and regression tests.
5. Optional: add migration utility for legacy canonical workdocs.

## Open Decisions
- Whether to move review-cycle blocks into dedicated sentinel-managed regions in a future phase.
- Whether to make schema marker presence mandatory in a future major version.
