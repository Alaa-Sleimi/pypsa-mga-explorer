# pypsa-mga-explorer

An implementation of Sina Hajikazemi's MGA-Compass method for interactive
near-optimal solution exploration on PyPSA energy system models: instead of a
single cost-optimal solution, it samples and lets you navigate the whole
near-optimal space of alternatives. Includes `mga_exploration.ipynb`, a
reference notebook that runs the full pipeline end to end.

Bachelor's thesis - Alaa Sleimi - TU Darmstadt, FG EINS, 2026.

## What it does

The pipeline has two phases:

- **Preparation** (heavy, run once per network): solve the network for its cost
  optimum, sample the boundary and interior of the near-optimal space under a
  cost slack, and price every sample.
- **Exploration** (interactive, from the saved samples): state a preference over
  which quantities to raise, lower, or hold, navigate to the nearest solution
  satisfying it, and traverse the path to it -- all from the saved samples, with
  no further network solves except where the notebook explicitly says so.

See `docs/workflow.md` for the full walkthrough of both phases, mapped to the
notebook cells that run them.

## Requirements

- Python 3.10 or newer (the codebase uses `X | None` type hints); developed and
  tested on Python 3.13.5.
- HiGHS as the LP/MILP solver, used throughout via `linopy` (and directly via
  `highspy` in one module); no separate solver installation is needed, it ships
  as part of the pinned Python packages.
- Internet access on the first run of the notebook's `model_energy` example
  (Section 1.1), which downloads and caches that network the first time it is
  used.

Exact package versions are pinned in `requirements.txt`; see that file rather
than this one for the full list.

## Installation

**Linux / macOS**

```bash
git clone https://github.com/Alaa-Sleimi/pypsa-mga-explorer.git
cd pypsa-mga-explorer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Alaa-Sleimi/pypsa-mga-explorer.git
cd pypsa-mga-explorer
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quickstart

Open `mga_exploration.ipynb` (with the repo root as the kernel's working
directory) and run its cells in order. For a full walkthrough on the built-in
`model_energy` toy network -- no external data files needed -- follow
`docs/example_model_energy.md`.

## Repository layout

| Entry | Purpose |
|---|---|
| `mga_engine/` | The engine: sampling, pricing, navigation, traverse, projection, and preparation-state persistence. |
| `mga_exploration.ipynb` | The reference notebook: the only intended way to run the pipeline end to end. |
| `tests/` | Automated tests plus a few standalone check scripts; see "Running the tests" below. |
| `docs/` | Reference documentation for settings, the API, the pipeline, and the toy-network walkthrough. |
| `data/` | Preparation-state caches and other generated `.nc`/`.npz` files, written here by the notebook at runtime. The folder is kept in Git via `.gitkeep`; its contents are not committed (see `.gitignore`). |
| `figures/` | PNGs exported by the notebook's figure-export cell; not committed. |

## Documentation

- `docs/configuration.md` -- every notebook setting: what it does, its default, and whether it invalidates the cached preparation state.
- `docs/api.md` -- a map of `mga_engine`'s modules and public functions, with pointers into their docstrings.
- `docs/workflow.md` -- the end-to-end pipeline, preparation phase to exploration phase, mapped to the notebook.
- `docs/example_model_energy.md` -- a full walkthrough of running the notebook on the built-in `model_energy` toy network.

## Running the tests

```bash
pytest tests/test_smoke.py
```

This is the only automated suite in `tests/`. The other four files there
(`test_navigation.py`, `test_traverse.py`, `test_surrogate.py`,
`test_alpha_projection.py`) are standalone scripts, not pytest suites: they run
checks at import time against a cached `data/preparation_state.nc` rather than
defining `test_*` functions, and are meant to be run directly (e.g.
`python -m tests.test_navigation`). Since there is no pytest configuration
narrowing collection, running plain `pytest` or `pytest tests/` will also try to
collect and execute these scripts; run `pytest tests/test_smoke.py` specifically
to avoid that.

## Known limitations

- The cached preparation state's fingerprint covers a whitelist of network
  inputs, not everything that can affect the true optimum -- some changes are
  invisible to it and can lead to a stale cache being reused silently. See
  `docs/configuration.md`'s "Cache invalidation" section for exactly what is and
  is not covered.
- The surrogate module is included in the workflow but was not fully validated.
  Its formulation and results were not verified within the scope of this thesis
  due to time constraints, and it should be treated as experimental.

## References

- Hajikazemi, S. *Interactive Exploration of Near-Optimal Solutions of Energy
  Planning Models*. EEM 2026.
