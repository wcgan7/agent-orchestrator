# Workdoc Retry Implementation Phases

## Objective
Track execution of the retry/workdoc behavior change from specification to production-safe implementation.

## Status Legend
- `[ ]` Not started
- `[~]` In progress
- `[x]` Completed

## Phase 0: Baseline and Scope Lock
- [x] Confirm current behavior in code paths (`retry` + run init).
- [x] Define intended behavior and invariants.
- [x] Record design intent in `WORKDOC_RETRY_INTENT_AND_BEHAVIOR.md`.

## Phase 1: Tests First (Expected to Fail Initially)
- [x] Add API tests for retry preserving workdoc content.
- [x] Add test for retry appending attempt marker metadata/content.
- [x] Add test for missing canonical workdoc on retry causing explicit block.
- [x] Add orchestrator tests for `retry_from_step` preserving historical workdoc sections.
- [x] Add preserved-branch retry test ensuring append-only continuity.
- [x] Add workdoc manager tests that first run initializes once and retries do not reinitialize.

## Phase 2: Execution Path Changes
- [x] Update task executor to initialize workdoc only when canonical is absent.
- [x] Otherwise refresh existing canonical workdoc into worktree before step execution.
- [x] Ensure retry path never calls template reset for already-initialized tasks.

## Phase 3: Append-Only Workdoc Enhancements
- [x] Add retry-attempt append hook (attempt number, timestamp, guidance).
- [x] Ensure step sync appends attempt-scoped entries instead of replacing historical entries.
- [x] Keep orchestrator-managed step behavior compatible while preserving history.

## Phase 4: Guardrails and Error Handling
- [x] Add strict guard: missing canonical workdoc on retry blocks with explicit error.
- [x] Verify first-run bootstrap behavior remains unchanged.
- [x] Add diagnostics for malformed/unreadable workdoc cases.

## Phase 5: Events and Observability
- [x] Emit retry lifecycle payload (`attempt`, `start_from_step`, guidance presence).
- [x] Preserve backward compatibility for existing task/workdoc event consumers.

## Phase 6: Validation and Regression
- [x] Run targeted tests for retry, workdoc, worktree, and review loops.
- [x] Run full backend test suite.
- [x] Confirm no regressions in first-run workdoc creation and commit flows.

## Phase 7: Section Sentinel Hardening
- [x] Draft machine-readable sentinel design for deterministic section sync.
- [x] Define sync decision matrix for canonical/worktree mismatch states.
- [x] Define normative step-to-section sentinel ID mapping (including `plan_refine`/`initiative_plan_refine` skip-step rule).
- [x] Define repeatable-step numbering policy (`Attempt N`, `Fix Cycle N`, `Review Cycle N`).

### Phase 7A: Parser + Matrix Enforcement
- [x] Implement sentinel-aware section parsing and target-section merge path.
- [x] Enforce sync decision matrix outcomes for valid/missing/malformed/id-mismatch states.
- [x] Ensure no silent no-op sync path remains.

### Phase 7B: Template Sentinelization
- [x] Add sentinel pairs to initialized workdoc templates for all mapped sections.
- [x] Preserve existing human-readable headings (compatibility-first).
- [x] Preserve v1 shared-section policy for `verify`/`benchmark`.

### Phase 7C: Diagnostics + Observability
- [x] Persist sync diagnostic metadata on fallback/block paths (`workdoc_sync_error_type`, `workdoc_sync_mode`, `workdoc_sync_step`, `workdoc_sync_attempt`).
- [x] Emit additive `workdoc.updated` payload fields (`sync_mode`, `reason`) for fallback/merge observability.
- [x] Preserve backward compatibility of existing task/workdoc event fields.

### Phase 7D: Regression + Sign-off
- [x] Add regression tests for internal `##` headings, malformed sentinels, and mismatch-state combinations.
- [x] Add regression tests for legacy non-sentinel fallback behavior.
- [x] Add regression tests for shared verification section behavior (`verify`/`benchmark`) and append-only numbering.
- [x] Run full backend suite in `.venv` with no new regressions.
- [x] Update docs to match implemented behavior and mark Phase 7 complete.

### Phase 7 Definition of Done
- [x] 7A-7D checklist items are complete.
- [x] Targeted tests pass for parser/matrix/fallback/shared-section/numbering behavior.
- [x] Full backend suite passes in `.venv`.
- [x] Event payload compatibility is verified (additive-only changes).
- [x] Operational safety checks pass (explicit diagnostics, no silent no-op, explicit block errors).
- [x] Documentation parity is confirmed between design and implementation.

## Risks and Notes
- Risk: accidental reset if any alternate run path still calls unconditional init.
- Risk: append logic may duplicate headings if insertion points are incorrect.
- Risk: preserving more metadata could affect prompt size; monitor carefully.

## Rollout Order
1. Tests (red)
2. Executor/workdoc behavior change
3. Append semantics
4. Guards and diagnostics
5. Events
6. Full validation (green)

## Progress Log
- 2026-02-24: Created behavior spec and implementation phase tracker.
- 2026-02-24: Added Phase 1 retry/workdoc tests for context preservation and missing-workdoc retry guard; current failures are expected until Phase 2-4 implementation.
- 2026-02-24: Completed Phase 2 executor changes (init-once, refresh-on-retry) and added retry missing-workdoc strict block behavior; targeted regression suite passes.
- 2026-02-24: Implemented retry-attempt workdoc markers (with guidance/start-step context) and validated retry/workdoc regression coverage (10 targeted tests passing).
- 2026-02-24: Completed attempt-scoped step sync entries for retry continuity and validated full workdoc suite (44 tests passing).
- 2026-02-24: Completed malformed/unreadable workdoc diagnostics across run + API paths, with targeted retry/workdoc regression coverage (13 tests passing).
- 2026-02-24: Completed Phase 5 observability payloads for `task.retry`, retry `task.started`, and retry `workdoc.updated` with compatibility-preserving fields and targeted coverage (11 tests passing).
- 2026-02-24: Completed Phase 6 full-suite validation using `.venv` (`409 passed, 4 skipped`); stabilized one timing-summary test by preventing unintended background execution.
- 2026-02-24: Added sentinel-based section-sync design in `WORKDOC_SECTION_SENTINELS_DESIGN.md` and opened Phase 7 for deterministic section merge hardening.
- 2026-02-24: Phase 7 design hardened with explicit sync decision matrix, normative step-section mapping, and repeatable-step numbering policy.
- 2026-02-24: Split Phase 7 into 7A-7D (parser, templates, diagnostics, validation) and tightened completion criteria for lower-risk rollout.
- 2026-02-24: Completed Phase 7A parser/matrix enforcement in `workdoc_manager.sync_workdoc`, added targeted sentinel/mismatch regression tests, and revalidated full suite in `.venv` (`417 passed, 4 skipped`).
- 2026-02-24: Completed Phase 7B sentinelization for initialized workdocs (schema marker + section markers), added init-template coverage, stabilized one board-ordering test race, and revalidated full suite in `.venv` (`417 passed, 4 skipped`).
- 2026-02-24: Completed Phase 7C diagnostics/observability hardening (persisted sync diagnostic metadata, additive `workdoc.updated` fields, retry cleanup for sync markers), expanded regression coverage, and revalidated full suite in `.venv` (`418 passed, 4 skipped`).
- 2026-02-24: Completed Phase 7D regression/sign-off (added legacy/non-sentinel, malformed-sentinel, and shared-verification coverage) and closed Phase 7 DoD with full-suite validation in `.venv` (`421 passed, 4 skipped`).
