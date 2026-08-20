"""
tests/conftest.py — shared pytest setup for the smoke-test suite.

Two jobs only:
  1. Force matplotlib onto the non-interactive "Agg" backend so importing any
     engine/test module can never pop a window in CI or on a dev machine.
  2. Make the project root importable (``import mga_engine``) regardless of the
     directory pytest is invoked from, and expose the ONE read-only fixture that
     is shared between tests.

Nothing here writes to the repository. The preparation-state fixture opens
data/preparation_state.nc in READ-ONLY mode and never regenerates it.
"""

import os
import sys

import matplotlib

# Must happen before any test (or engine module) imports pyplot.
matplotlib.use("Agg")

import pytest  # noqa: E402

# Project root = parent of tests/. Prepended so `import mga_engine` works even
# when pytest is run from inside tests/ or with a different rootdir.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#: The real preparation state produced by the preparation phase. READ ONLY.
PREPARATION_STATE_PATH = os.path.join(PROJECT_ROOT, "data", "preparation_state.nc")


@pytest.fixture(scope="session")
def prepared_state():
    """Load data/preparation_state.nc READ-ONLY, or skip the test if absent.

    Deliberately never writes to or regenerates the file: `load()` opens the
    netCDF dataset with mode="r" and the returned dict holds plain numpy copies,
    so nothing a test does to it can touch the file on disk.
    """
    if not os.path.exists(PREPARATION_STATE_PATH):
        pytest.skip(
            f"{PREPARATION_STATE_PATH} not found. This test reads the real "
            "preparation state read-only and will NOT regenerate it — run the "
            "preparation phase yourself if you want this test to execute."
        )

    from mga_engine.preparation_state import load

    return load(PREPARATION_STATE_PATH)
