# Feature PRD Runner - Web Dashboard

> **Status (2026-02-11):** Current frontend setup/development guide.
> **Docs index:** [`../docs/README.md`](../docs/README.md)
> **User guide:** [`../docs/USER_GUIDE.md`](../docs/USER_GUIDE.md)

Modern web dashboard for monitoring and controlling Feature PRD Runner.

## Homepage

![Feature PRD Runner homepage](./public/homepage-screenshot.png)

## Setup

### Prerequisites

- Node.js 18+ and npm
- Backend server running (see main README)

For the backend, if you prefer `uv`, you can install the server extra with:

```bash
uv pip install 'feature-prd-runner[server]'
```

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

The dashboard will be available at http://localhost:3000

The backend API should be running at http://localhost:8080

### Tests

Run unit/integration tests (Vitest):

```bash
npm test
```

This now includes a mounted-surface guard (`check:mounted-api-contracts`) that fails if `main.tsx`-reachable files import `legacyApi`.

Optional:

```bash
npm run test:ui
npm run test:coverage
```

Run real-server browser smoke tests (Playwright):

```bash
npx playwright install chromium
npm run e2e:smoke
```

`e2e:smoke` starts both backend (`:8080`) and frontend (`:3000`) automatically.

Regenerate the homepage screenshot with seeded tasks across board stages:

```bash
npm run screenshot:homepage
```

### Production Build

Build for production:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

## Features

- **Board**: Kanban columns (`backlog` to `done`) with inline task detail/edit/actions.
- **Execution**: Orchestrator status, queue depth, execution batches, pause/resume/drain/stop controls.
- **Review Queue**: Approve or request changes with optional guidance.
- **Agents**: Spawn/pause/resume/terminate agents.
- **Create Work**:
  - Create Task
  - Import PRD (preview/commit + import job detail)
  - Quick Action (detail + promote to task)
- **Task Explorer**: Filtered task list with search, blocked-only toggle, and pagination.
- **Project management**: Active repo selector, pin/unpin project paths, and directory browser.
- **Realtime refresh**: `/ws` events trigger mounted surface refresh.

## Architecture

### Frontend

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Plain CSS with modern features
- **State Management**: React hooks (useState, useEffect)

### API Integration

The frontend connects to the FastAPI backend:

- REST API (`/api/v3/*`) for data fetching
- WebSocket (`/ws`) for real-time updates

See `vite.config.ts` for proxy configuration.

## Components

### App

`web/src/App.tsx` is the mounted UI entry and source of truth for active surfaces.

### App Panels

- `TaskExplorerPanel`
- `ImportJobPanel`
- `QuickActionDetailPanel`

### Legacy Components

Unmounted cockpit components were pruned from `src` to keep the runtime surface focused on mounted routes only.

## Development Notes

### Error Handling

All API calls include error handling and display user-friendly error messages when the backend is unavailable.

### Responsive Design

The dashboard is responsive and works on mobile devices, tablets, and desktops.

## Future Enhancements

- [ ] Dark theme
- [ ] Export reports (PDF, CSV)
- [ ] Email/Slack notifications
- [ ] Dedicated typed API client for mounted routes
- [ ] More hardening + E2E tests
- [ ] WebSocket for all realtime panels (status/tasks/phases)
- [ ] Expand E2E scenarios beyond smoke coverage
