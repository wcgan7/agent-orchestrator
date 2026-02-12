from __future__ import annotations

import re
import sys
from pathlib import Path


def test_python_runtime_is_at_least_3_10() -> None:
    assert sys.version_info >= (3, 10)


def test_ui_runtime_uses_v3_api_paths_only() -> None:
    root = Path(__file__).resolve().parents[1]
    ui_root = root / 'web' / 'src'
    ui_files = sorted(
        path for path in ui_root.rglob('*')
        if path.suffix in {'.ts', '.tsx'}
        and '.test.' not in path.name
        and 'src/test/' not in str(path).replace('\\', '/')
    )
    legacy_hits: list[str] = []

    for path in ui_files:
        text = path.read_text(encoding='utf-8')
        for idx, line in enumerate(text.splitlines(), start=1):
            for match in re.finditer(r"/api/[^\"'`\s]*", line):
                token = match.group(0)
                if token.startswith('/api/v3/'):
                    continue
                legacy_hits.append(f"{path}:{idx}: {token}")

    assert not legacy_hits, 'Legacy UI API endpoints detected:\n' + '\n'.join(legacy_hits)
