# Feature PRD Runner Example

This example is intentionally small and deterministic so you can verify that the
runner creates phases, runs them, and commits changes. It includes a tiny Python
module, a PRD with two phases, and a simple unittest suite.

## What's Included

- `feature_prd.md`: PRD describing two phases.
- `project/`: Minimal Python package with tests.

## Run

```bash
cd feature_prd_runner/example
python -m feature_prd_runner.runner --project-dir ./project --prd-file ./feature_prd.md --test-command "python -m unittest -q"
```

After the run, inspect `.prd_runner/` inside `project/` for state, logs, and progress.
