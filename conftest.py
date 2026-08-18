"""Root conftest: puts the repo root on sys.path explicitly, so `from src
import pose`, `import naming` and `import migrate_cache_keys` resolve under
any pytest invocation — `pytest tests/`, `cd tests && pytest`, or an absolute
path from elsewhere. pytest.ini pins rootdir here so this file is always
collected."""
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)
