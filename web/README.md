# Agent Orchestrator - Web Dashboard

> **Docs index:** [`../docs/README.md`](../docs/README.md)
> **User guide:** [`../docs/USER_GUIDE.md`](../docs/USER_GUIDE.md)

Web dashboard for monitoring and controlling Agent Orchestrator.

## Screenshot

<!-- Regenerate with: npm run screenshot:homepage -->
![Agent Orchestrator homepage](./public/homepage-screenshot.png)

## Setup

### Prerequisites

- Node.js 18+ and npm
- Backend server running (see main README)

### Installation

```bash
cd web
npm install
```

### Development

Start the development server:

```bash
npm run dev
```

The dashboard will be available at http://localhost:3000.
The backend API should be running at http://localhost:8080.

Proxy env vars:
- `VITE_API_PROXY_TARGET` (default `http://localhost:8080`)
- `VITE_WS_PROXY_TARGET` (default follows API target)
- `VITE_PORT` (default `3000`)

### Tests

Run unit tests (Vitest):

```bash
npm test
```

The test script includes a mounted-surface guard (`check:mounted-api-contracts`) that fails if `main.tsx`-reachable files import `legacyApi`.

Run real-server browser smoke tests (Playwright):

```bash
npx playwright install chromium
npm run e2e:smoke
```

`e2e:smoke` starts both backend and frontend automatically on ephemeral ports.

Regenerate the homepage screenshot with seeded tasks across board stages:

```bash
npm run screenshot:homepage
```

### Production Build

```bash
npm run build     # Runs typecheck + contract check, then Vite build
npm run preview   # Preview the production build
```

## Features

### Board
Kanban columns (`backlog` → `queued` → `in_progress` → `in_review` → `blocked` → `done` → `cancelled`) with task cards, compact mode toggle, and queue/worker summary.

### Planning
Plan creation and iterative refinement per task. View revision history, provide feedback, and commit a final plan before generating implementation tasks.

### Execution
Orchestrator status indicator (running/paused/stopped/draining), control buttons (pause/resume/drain/stop), execution pipeline visualization with batched "waves" of parallel tasks, and runtime metrics (API calls, wall time, tokens, cost, files changed).

### Workers
Provider health checks for Codex, Claude, and Ollama. Step-to-provider routing table. Active task monitoring and default provider display.

### Settings
Project selector with pin/unpin and directory browser. Concurrency, auto-deps, max review attempts. Role routing, worker provider configuration, per-language project commands, and quality gate thresholds.

### Task Detail Modal
Accessible from any task card. Tabs: **Overview** (description, execution summary, error details, pending gates, HITL blocking issues), **Logs** (live stdout/stderr with step selector), **Activity** (event timeline), **Dependencies** (graph visualization), **Configuration** (approval mode, HITL mode, pipeline, worker model), **Changes** (git diff viewer).

### Create Work
- **Create Task** — manual task creation with all fields
- **Import PRD** — preview/commit + import job detail panel
- **Terminal** — embedded interactive shell session for direct project commands

### Task Explorer
Filtered task list with search, blocked-only toggle, and pagination. Accessible via the board sidebar.

### Real-time Updates
WebSocket (`/ws`) events trigger automatic refresh of board, execution, and task detail surfaces.

## Architecture

### Frontend

- **Framework**: React 18 with TypeScript (strict mode)
- **Build Tool**: Vite
- **Styling**: Plain CSS with CSS variables
- **State Management**: React hooks (`useState`, `useEffect`, `useRef`)
- **Testing**: Vitest + React Testing Library (unit), Playwright (e2e)

### API Integration

The frontend connects to the FastAPI backend:

- REST API (`/api/*`) for data operations
- WebSocket (`/ws`) for real-time event streaming

See `vite.config.ts` for proxy configuration.

## Key Files

| File | Purpose |
|---|---|
| `src/App.tsx` | Main component: routing, WebSocket, global state, all views |
| `src/api.ts` | HTTP client with auth token handling and base URL construction |
| `src/components/AppPanels/` | Panel components (ImportJobPanel, TerminalPanel, TaskExplorerPanel) |
| `src/components/HITLModeSelector/` | HITL mode selection UI |
| `src/types/` | Shared TypeScript type definitions |
| `src/styles/orchestrator.css` | Main stylesheet with CSS variables |
| `src/ui/labels.ts` | Label humanization helpers |
