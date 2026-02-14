# Documentation

This project now keeps only user-facing documentation.

## Start Here

- `../README.md`: product overview and quick start.
- `USER_GUIDE.md`: complete usage guide for board, execution, review, agents, and settings.
- `API_REFERENCE.md`: REST and WebSocket reference for `/api`.
- `CLI_REFERENCE.md`: command-line reference for `agent-orchestrator`.
- `../web/README.md`: frontend setup, testing, and UI-specific workflows.
- `../example/README.md`: example assets and local sandbox walkthrough.

## Runtime Data

Runtime state is stored under `.agent_orchestrator/` in your selected project directory.
On first startup with an incompatible legacy state, `.agent_orchestrator` is archived to `.agent_orchestrator_legacy_<timestamp>/`.

## Support Endpoints

- `GET /healthz`
- `GET /readyz`
- `GET /`

Use these for local health and project-target diagnostics.
