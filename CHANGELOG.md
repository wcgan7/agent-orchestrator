# Changelog

## [0.2.0] - 2026-03-16

### Added
- LLM-generated recommended actions for blocked tasks with concrete recovery suggestions
- Task-level worker provider override for retrying blocked tasks with a different provider
- Merge and pull request review pipelines
- LLM-generated summaries at gate junctures (review cycles, completion)
- Initiative intent preservation through task generation
- Worker environment variable configuration with 4-layer resolution (auto/process/config/task)
- Virtual environment auto-detection for Python projects
- Default project commands configuration in Settings

### Changed
- Consolidated Workers panel into Settings with 3-tab layout (Providers, Execution, Advanced)
- Hybrid save UX: auto-save for toggles/dropdowns, dirty-state save buttons for text/numeric fields
- Replaced HITL toggle buttons with compact dropdown selector
- Default HITL mode changed to supervised
- Improved worker stall detection with defense-in-depth recovery
- Faster worker cancellation on task stop
- Simplified post-merge health check (alert instead of revert)
- Collapsed file changes by default in task detail view

### Fixed
- Report step not capturing summary after review cycle
- `implement_fix` retry bypassing verify when pipeline phase is missing
- Supervised gate skipped after `request_changes` on commit review tasks
- Partially completed pipelines incorrectly marked as done
- Default task timeout rejecting 0 (no timeout)
- Task detail modal refreshing unnecessarily
- Plan tab not rendering for some pipeline types

### Removed
- Standalone Workers route (consolidated into Settings)
- Diagnostics section from Settings

## [0.1.0] - 2026-02-28
### Added
- Initial public release.
