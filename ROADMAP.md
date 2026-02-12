# Feature PRD Runner Roadmap (v3)

> **Status:** Active roadmap tracker for the orchestrator-first revamp.
> **Canonical plan:** [`docs/ORCHESTRATOR_FIRST_REVAMP_PLAN.md`](docs/ORCHESTRATOR_FIRST_REVAMP_PLAN.md)
> **Docs index:** [`docs/README.md`](docs/README.md)

## Scope

This roadmap tracks execution progress for the v3 orchestrator-first transformation.
It intentionally avoids restating the full blueprint and focuses on milestone status.

## Milestones

| Milestone | Focus | Status |
|---|---|---|
| M1 | v3 backend foundations (domain + storage + API shell) | Not Started |
| M2 | Orchestrator runtime (scheduler, claim locks, review loop) | Not Started |
| M3 | UI rebuild (board-first IA, create-work flows, review queue) | Not Started |
| M4 | Cutover + stabilization (hard reset flow, docs, release) | Not Started |

## Near-Term Execution Order

1. Build `src/feature_prd_runner/v3/` module boundaries.
2. Implement v3 task/quick-action/import/review APIs.
3. Wire orchestrator runtime and queue/agent execution.
4. Replatform the web UI onto v3 contracts only.
5. Ship cutover with archived legacy state and docs cleanup.

## Archived Roadmaps

Earlier long-form roadmap documents are archived for historical context:

- [`docs/archive/historical/ROADMAP.md`](docs/archive/historical/ROADMAP.md)
- [`docs/archive/historical/TRANSFORMATION_ROADMAP.md`](docs/archive/historical/TRANSFORMATION_ROADMAP.md)

## Update Policy

- Update this file for milestone-level progress only.
- Update implementation detail in the canonical revamp plan as decisions change.
