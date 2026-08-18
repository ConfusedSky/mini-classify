"""Root conftest: puts the repo root on sys.path explicitly, so `from src
import pose` and the top-level tools (`classify_stls`, `migrate_cache_keys`)
resolve under any pytest invocation — `pytest tests/`, `cd tests && pytest`,
or an absolute path from elsewhere. pytest.ini pins rootdir here so this file
is always collected.

The root is needed for the *tools* now, not for `src`: `instrument` and
`naming` moved into the package (2026-08-18), so nothing under `src/` reaches
outside it any more."""
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)
