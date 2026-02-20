# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Orchestrator is a local orchestration control center for AI-assisted software delivery. It provides a task board, execution controls, review gates, and agent management. The backend is Python/FastAPI; the frontend is React/TypeScript with Vite.

## Development Commands

### Backend (Python)

```bash
# Activate venv
source .venv/bin/activate
# Install (editable, with server deps)
python -m pip install -e ".[server]"

# Run backend server
agent-orchestrator server --project-dir /path/to/repo

# Run all unit tests
pytest

# Run a single test file
pytest tests/test_event_hub.py

# Run a single test
pytest tests/test_event_hub.py::test_name -x

# Integration tests (skipped by default)
AGENT_ORCHESTRATOR_RUN_INTEGRATION=1 pytest tests/test_integration_claude_provider.py

# Type checking
mypy

# Linting
ruff check src/
```

### Frontend (web/)

All frontend commands use `npm --prefix web` from the repo root, or run from `web/`.

```bash
npm --prefix web install
npm --prefix web run dev          # Dev server on :3000 (proxies /api to :8080)
npm --prefix web run build        # Production build (runs typecheck + contract check first)
npm --prefix web run lint         # TypeScript typecheck (tsc --noEmit)
npm --prefix web run test         # Vitest unit tests
npm --prefix web run test -- --reporter=verbose web/src/App.logs.test.tsx  # Single test file
npm --prefix web run check        # lint + test + build (full CI check)
npm --prefix web run e2e          # Playwright e2e (starts backend on :19080 + frontend on :19030)
npm --prefix web run e2e:smoke    # Smoke subset only
```

The build script enforces API contract checks (`check:mounted-api-contracts`) before compiling.

## Architecture

### Backend (`src/agent_orchestrator/`)

- **`runtime/orchestrator/service.py`** — Core `OrchestratorService`. Manages task queue, worker dispatch via `ThreadPoolExecutor`, state transitions, and review cycles. This is the central coordination point.
- **`runtime/api/router.py`** — FastAPI router. All REST endpoints (`/api/*`) and request/response schemas are defined here.
- **`runtime/domain/models.py`** — Dataclasses: `Task`, `ReviewCycle`, `RunRecord`, `PlanRevision`, `AgentRecord`, `TerminalSession`.
- **`runtime/storage/`** — YAML-file repositories with file-locking. Each entity type has its own `.yaml` file in `.agent_orchestrator/`.
- **`runtime/events/`** — `EventBus` + `WebSocketHub` for real-time pub/sub across channels (`tasks`, `queue`, `agents`, `review`, `terminal`).
- **`server/api.py`** — FastAPI app factory with lifespan management.
- **`workers/`** — Worker provider adapters (claude, codex, ollama). `WorkerAdapter` is the abstraction layer.
- **`pipelines/`** — Pipeline template registry for task execution workflows.
- **`collaboration/`** — HITL (Human-In-The-Loop) mode configs: autopilot, supervised, collaborative, review_only.
- **`cli.py`** — CLI entry point (`agent-orchestrator` command).

### Frontend (`web/src/`)

- **`App.tsx`** — Main component handling routing, WebSocket connection, and global state. This is a large monolithic file (~5k lines).
- **`api.ts`** — HTTP client with auth token handling and base URL construction.
- **`components/AppPanels/`** — Panel components (ImportJobPanel, TerminalPanel, TaskExplorerPanel).
- **`components/HITLModeSelector/`** — HITL mode selection UI.
- **`types/`** — Shared TypeScript type definitions.
- **`styles/`** — CSS with variables, base styles, and orchestrator-specific styles.

### Communication

- REST: Frontend calls `/api/*` endpoints, proxied by Vite dev server to backend on `:8080`.
- WebSocket: Frontend connects to `/ws` for real-time event streaming. Events are channel-based with sequence counters for ordering.

### Data Storage

All runtime state lives in `.agent_orchestrator/` within the target project directory (YAML files + JSONL event log). No external database.

### Task Lifecycle

`backlog` → `queued` → `in_progress` → `in_review` → `done`

Tasks support dependency graphs (validated for cycles), parallel execution with configurable concurrency, and review cycles with severity-based findings.

## Key Conventions

- Backend source lives under `src/` (setuptools `package-dir` mapping).
- Python 3.10+ required. Strict mypy enabled.
- Frontend uses TypeScript strict mode. Tests use Vitest + React Testing Library; e2e uses Playwright.
- Proxy env vars: `VITE_API_PROXY_TARGET` (default `http://localhost:8080`), `VITE_WS_PROXY_TARGET`, `VITE_PORT` (default `3000`).
