"""Prompt template loader with caching."""

from pathlib import Path

_DIR = Path(__file__).parent
_cache: dict[str, str] = {}


def load(name: str) -> str:
    """Load a prompt template by relative path (e.g. 'steps/plan.md')."""
    if name not in _cache:
        _cache[name] = (_DIR / name).read_text().strip()
    return _cache[name]
