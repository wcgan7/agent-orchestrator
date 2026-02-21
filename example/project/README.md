# Example Project

Small Python module used by orchestrator test flows.

## Current State

`math_utils.py` intentionally contains stubs:
- `add_numbers(a, b)`
- `multiply_numbers(a, b)`

Both functions currently raise `NotImplementedError` so the example can be used as a predictable implementation target.

## Run Tests

```bash
python -m unittest -q
```
