# Feature PRD Runner Example

This example is intentionally small and deterministic so you can validate the
v3 task + PRD workflows locally. It includes a tiny Python module, a PRD sample,
and a small unittest suite.

## What's Included

- `feature_prd.md`: sample PRD content you can paste into Import PRD.
- `project/`: Minimal Python package with tests.

## Run With Web UI

```bash
feature-prd-runner server --project-dir ./example/project
npm --prefix web run dev
```

In the dashboard, use:
- `Create Work` -> `Create Task` for manual tasks.
- `Create Work` -> `Import PRD` and paste `example/feature_prd.md`.

## Run With CLI

```bash
feature-prd-runner --project-dir ./example/project task create "Smoke test example project"
feature-prd-runner --project-dir ./example/project task list
```

After running flows, inspect `.prd_runner/v3/` inside `example/project/` for state and events.
