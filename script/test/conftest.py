
import os
import sys
from pathlib import Path

import pytest

@pytest.fixture(scope="session", autouse=True)
def _ensure_repo_on_path():
    """
    Make sure `import src.*` works when running pytest from repo root.
    - If tests are placed under V_2/tests, running `pytest` at V_2 root should already work.
    - This fixture adds the nearest parent that contains "src" onto sys.path as a fallback.
    """
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src").exists():
            sys.path.insert(0, str(p))
            break
    yield
