# Configuration reference for mga_exploration.ipynb

This file does not define any settings itself. Every setting a user configures lives
in the notebook cells themselves, in the order the user meets them, usually under a
`USER CONFIGURATION` banner comment. This file is a reference to look them up: what
each one does, what it defaults to, and whether it invalidates the cached preparation
state. When the notebook and this file disagree, the notebook is correct; open an
issue (or just fix this file).

Cell numbers below are 0-indexed, counting markdown and code cells together, matching
`DOC_INVENTORY.md`.

## Section 1 -- Setup (cell 2)

Imports, quiets linopy/pypsa logging, creates `figures/`, and picks which of the two
network-providing cells below runs.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `network_source` | 2 | `"load_file"` | str | Which of cells 4/5 executes: `"fresh"` (in-notebook demo) or `"load_file"` (load a network) | No (the resulting network does; see below) |

Notes:
- Must be exactly `"fresh"` or `"load_file"`; anything else fails an `assert` in the
  same cell.
- The non-selected cell still runs, but only prints a note and does nothing.

## Section 1.1 -- Provide Your Network: fresh branch (cell 4)

Runs only when `network_source == "fresh"`. Builds a 5-bus demo network with plain
PyPSA code, solves it once for the cost optimum, and defines `build_network()` for the
preparation phase to call repeatedly.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `epsilon` | 4 | `0.05` | float | Near-optimality slack: cost <= `(1 + epsilon) * opt_cost` | Yes (hash) |
| `N_SNAPSHOTS` | 4 | `72` | int | Number of snapshots (weighted to a full year) in the demo network | Yes (count + weightings digest) |
| `build_network()` body | 4 | 5 buses, 6 lines, 13 generators (solar/wind/hydro/biomass/gas), 4000 MW load | code | The network itself | Yes, but only the whitelisted attributes -- see "Cache invalidation" |
| `optimize(...)` keyword arguments | 4 | `solver_name="highs", include_objective_constant=False` | kwargs | The base solve that produces `opt_cost` | No |

Notes:
- `epsilon` has no enforced upper bound; the only check anywhere is
  `0 < epsilon_min < epsilon` in cell 9 and in
  `interior_sampling.derive_epsilon_levels`.
- To explore your own network, edit `build_network()`'s body directly (keep
  `p_nom_extendable=True` and meaningful `carrier` names -- Section 2's PoIs are built
  from carriers) and update `poi_definitions` in Section 2 to match, then set
  `rerun_preparation = True` in Section 3.
- Loads are in MW in this cell (PyPSA's native unit); PoIs later report capacities in
  GW (`mga_engine.poi.MW_TO_GW` divides by 1000).

## Section 1.1 -- Provide Your Network: load_file branch (cell 5)

Runs only when `network_source == "load_file"`. Loads PyPSA's bundled `model_energy`
example, thins its snapshots, and solves it once.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `epsilon` | 5 | `0.05` | float | Same meaning as the cell-4 row; a second, independent definition | Yes (hash) |
| network call `pypsa.examples.model_energy()` | 5 | (hardcoded) | code | Which bundled example is loaded | Yes, whitelisted attributes |
| `STRIDE` | 5 | `12` | int or None | Keep every `STRIDE`-th of the source's snapshots, reweighted to a full year; `None` disables thinning | Yes (count + weightings digest) |
| `optimize(...)` keyword arguments | 5 | `solver_name="highs", include_objective_constant=False, solver_options={"output_flag": False}` | kwargs | The base solve that produces `opt_cost` | No |

Notes:
- Despite what cell 3's markdown says, there is **no `network_path` variable** in this
  branch. To point at a different file, edit the `pypsa.examples.model_energy()` call
  in cell 5 directly.
- `STRIDE=12` over the source's 2920 three-hourly snapshots keeps 244 of them (the
  cell's own comment says "~243"; the actual count, confirmed against the cached
  state, is 244). `STRIDE=0` raises a `ValueError` from the slice; a negative value
  reverses snapshot order instead of thinning it.
- The `model_energy` example downloads and caches under the OS user cache directory
  (`platformdirs`) the first time it is used; the notebook does not print this path.

## Section 2 -- Define Properties of Interest (cell 7)

Declares the PoIs (installed capacity, in GW, of one or more carriers) as a list of
dicts, then builds `poi_specs` from them.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `poi_definitions` (the list) | 7 | 5 entries: wind, solar, battery, electrolysis, turbine | list of dict | The PoI dimensions and their order (= row order of `P_all`) | Yes (hash, order-sensitive) |
| `"component"` | 7 | `"generator"`, `"generator"`, `"storage_unit"`, `"link"`, `"link"` | str | Which PyPSA component the capacity is read from | Yes |
| `"carriers"` | 7 | one-item lists, e.g. `["wind"]` | list of str | Which carriers are summed into this one PoI | Yes |
| `"name"` | 7 | e.g. `"wind_cap_gw"` | str | Label used in the `request` dict (Section 4) and in plots | Yes |
| `"unit"` | 7 | omitted -> `"GW"` | str | Display label only; values are always GW | Yes (hashed even though display-only) |

Notes:
- Every carrier named must exist in the network (`network.generators.carrier.unique()`
  etc. -- names are case- and spacing-sensitive), and at least one matching component
  must have `p_nom_extendable=True`, or `make_poi_specs` raises `ValueError`.
- For a `"link"` PoI, the capacity is the `bus0`-side INPUT capacity of `p_nom`, not
  the link's electrical output -- e.g. a hydrogen turbine link is measured in GW of
  hydrogen in.
- The set and order here must match what the loaded/cached preparation state was
  built with; cell 9 asserts this and raises if they diverge.

## Section 3 -- Preparation Phase (cell 9)

Runs the heavy sampling pipeline (`run_preparation`), or loads and verifies a cached
`.nc` state instead. Total samples = `n_vertices + n_interior_levels * n_samples_per_level`.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `preparation_state_path` | 9 | `"data/preparation_state_modelenergy.nc"` | str (path) | Which `.nc` file is loaded (cell 9) and written (cell 11) | No |
| `rerun_preparation` | 9 | `False` | bool | Force `run_preparation` even if the cache file exists | No (it is the override) |
| `n_vertices` | 9 | `40` | int | Vertex (boundary) MP(r) samples at the outer `epsilon` | Yes, manually (cell 9's own comparison; `check_state` also accepts it as `requested_n_vertices` but the notebook does not pass it) |
| `n_interior_levels` | 9 | `5` | int | Number of interior epsilon levels between `epsilon` and `epsilon_min` | Yes, manually (no `check_state` parameter exists for it) |
| `n_samples_per_level` | 9 | `15` | int | MP(r) samples drawn at each interior level | Yes, manually (same mechanism as `n_vertices`) |
| `epsilon_min` | 9 | `0.005` | float | Deepest (tightest) interior sampling level | Yes, manually (exact float equality) |
| `seed` | 9 | `42` | int | RNG seed for vertex sampling AND the per-level interior seeds (derived via `SeedSequence.spawn`) | Yes, via `check_state(requested_seed=seed)` -- a stored-attribute comparison, not part of the hash |

Notes:
- `n_vertices`, `n_interior_levels`, `n_samples_per_level` must each be `>= 1`
  (asserted in this cell); nothing in the engine itself enforces a minimum.
- `epsilon_min` must satisfy `0 < epsilon_min < epsilon` (asserted here and again
  inside `derive_epsilon_levels`). Levels are only strictly decreasing when
  `epsilon_min < epsilon / n_interior_levels`; this is not checked anywhere.
- The notebook's shipped defaults (`n_vertices=40`, `n_samples_per_level=15`) are
  larger than `run_preparation`'s own function defaults (30 and 10) -- if you read the
  engine docstring for defaults, use the notebook's values instead, since cell 9
  always passes them explicitly.
- A larger sample budget costs one HiGHS solve per sample (roughly linear runtime)
  and narrows the gaps `solve_ccp`/`solve_surrogate_costs` can hit. A budget too small
  can leave parts of the near-optimal space unsampled; if Section 4.1 or Section 7
  later fails with "point lies outside the sampled convex hull", raise `n_vertices`
  and/or `n_samples_per_level` and set `rerun_preparation = True`.
- `preparation_state_path`'s parent directory (`data/`) is created automatically by
  `save`; loading a nonexistent path with `rerun_preparation = False` simply falls
  through to running the preparation phase fresh.

### Save cell (cell 11)

No new settings. Writes `P_all`, `v`, `Gamma`, `solutions`, `p_star`, `epsilon`,
`poi_names`, the fingerprint from `make_fingerprint(network, poi_definitions,
epsilon)`, and the six sampling parameters above to `preparation_state_path`,
atomically and overwriting any existing file at that path. Nothing is saved
automatically after a fresh preparation run -- skipping this cell loses the result
when the kernel stops.

## Section 3.2 -- Near-Optimal Space, 2D Projection (cell 15)

Lets you pick two PoIs and plots the true 2D boundary of the near-optimal space
between them, solved on request via `project_hull` and cached in memory.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `_HULL_CACHE_MAX` | 15 | `32` | int | Max entries in the in-memory LRU cache of `project_hull` results | No |
| `x_sel` / `y_sel` initial values | 15 | `poi_names[0]` / `poi_names[-1]` | Dropdown | Which two PoIs are plotted first | No |

Notes:
- The cache is defined inside this cell, so re-running the cell clears it -- the
  intended way to force fresh projections after changing the network, PoIs, or
  `epsilon` without restarting the kernel.
- The dropdowns must select two different PoIs; selecting the same one twice prints a
  message instead of plotting.

## Section 4 -- Navigation (cell 16, intro only)

No settings of its own. Explains the `request` dict format used by cell 18 and the
APoI index convention (`0` = cost, `1..m` = PoIs, a `+1` offset from `poi_specs`'
0-based order) that the conversion in cell 18 has to apply correctly.

## Section 4.1 -- Navigation Request & Result (cell 18)

Converts your preference into `navigate()`'s inputs and shows the resulting target
solution next to the current one.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `request` | 18 | `{"wind_cap_gw": ("increase", 3.0), "solar_cap_gw": ("decrease", 3.0)}` | dict of str to (str, float) | Navigation preference; dict order is priority, PoIs left out are unconstrained | No |

Notes:
- Each value is `(direction, delta)`: `direction` in `{"increase", "decrease",
  "keep"}`; `delta` is the desired magnitude of change in that PoI's own unit
  (ignored for `"keep"`). An unrecognised name or direction raises `ValueError`.
- `delta`'s sign and magnitude are not validated beyond that check -- a request the
  sample space cannot satisfy is capped or flipped to "keep" by `navigate`'s Loop 1,
  not rejected.
- A PoI left out of `request` is fully free: its value is decided only by the final
  cost-minimisation step.

## Section 4.2 -- Restricted Region after the First Navigation Loop (cell 20)

Plots the sampled space (the convex hull `navigate` actually searches) and, inside it,
the region left after Loop 1's `loop1_bounds` are applied.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `x_sel_r` / `y_sel_r` initial values | 20 | `1` / `2` (APoI indices; `0` = cost) | Dropdown | Which two APoIs are plotted first | No |

Notes:
- `y_sel_r`'s default is actually `2 if len(apoi_options) > 2 else 0`, not the plain
  `2` a first read of the value suggests; with 5 PoIs (6 APoI options including cost)
  it evaluates to `2`, but a network with only one PoI would default to `0`.
- Requires Section 4.1 to have run first; an assertion checks for `alpha_t` and
  `loop1_bounds` in scope.

## Section 4.3 -- Radar Chart, Current to Target (cell 22)

Interactive radar comparing the current and a beta-interpolated solution, normalised
to each PoI's global min-max range from Section 3.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `beta_radar` | 22 | `0.0`, range `[0, 1]`, step `0.01` | FloatSlider | Interpolation position shown on the radar | No |

## Section 5 -- Traverse (cell 24)

Walks the path from the current to the target solution, plotting cost vs. beta
(with the engine's breakpoints and subgradient lower bound) and each PoI's value at
the current beta.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `epsilon_I` | 24 | `1e6` (EUR/yr) | float | Breakpoint improvement tolerance for `extract_breakpoints` | No |
| `beta_slider` | 24 | `0.0`, range `[0, 1]`, step `0.01` | FloatSlider | Position along the path | No |
| envelope resolution `np.linspace(0, 1, 200)` | 24 | `200` (hardcoded) | int | How many points the subgradient envelope line is drawn at | No |

Notes:
- `epsilon_I` is an ABSOLUTE cost tolerance in EUR/yr, not a fraction, so it needs
  rescaling for a network of a different size. Lower it (the cell's own comment
  suggests `2e5`) for more breakpoints and a tighter piecewise curve; nothing in the
  code stops you setting it `<= 0`, which risks unbounded recursion in
  `extract_breakpoints`.
- The 200-point envelope resolution only affects how smooth the plotted red curve
  looks; the underlying subgradient functions themselves are exact at every beta.

## Section 6 -- Save Results (cell 26)

Bookmarks the request, the resulting weights, points and costs from Sections 4-5 with
`np.savez` -- a lightweight, single-path snapshot, unrelated to
`preparation_state.save()`.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `save_path` | 26 | `"data/navigation_path.npz"` | str (path) | Output file for the bookmark | No |

Note: the parent directory must already exist -- this cell does not create it (unlike
the preparation save cell, which does).

## Section 7 -- Surrogate Costs (cell 27)

Recovers a cost vector under which the Section 4.1 target is optimal, via
`solve_surrogate_costs`.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `solve_surrogate_costs` keyword arguments | 27 | not passed (`tol_interior=None`, `normalize=True`) | -- | Interior-point tolerance and L1-normalisation of the recovered `r` | No |

Note: these two engine parameters are not exposed as notebook variables; to change
them, edit the call in cell 27 directly.

## Figure export (cell 28, no heading)

Re-seeds every interactive figure to a fixed state and writes all 11 plots to
`figures/` as PNG. Meant to be run last, after a clean top-to-bottom pass.

| Setting | Cell | Default | Type | What it controls | Affects cache? |
|---|---|---|---|---|---|
| `OUT` | 28 | `Path("figures")` | Path | PNG output directory | No |
| `SAVE` | 28 | `dict(dpi=300, bbox_inches="tight")` | dict | `savefig` keyword arguments | No |
| `PAIR_X`, `PAIR_Y` | 28 | `"wind_cap_gw"`, `"solar_cap_gw"` | str | The PoI pair used for both the 3.2 and 4.2 projection exports | No |
| `BETAS` | 28 | `((0.0, "b0"), (0.5, "b50"), (1.0, "b100"))` | tuple | Beta positions exported for the radar chart | No |

Note: `PAIR_X`/`PAIR_Y` must both be in `poi_names` (asserted). The traverse export
loop below `BETAS` duplicates the same three `(beta, suffix)` pairs as a separate
literal instead of reusing `BETAS` -- editing one without the other desyncs the radar
and traverse exports.

## Cache invalidation

`check_state` (called from cell 9) decides whether the loaded `.nc` file is reused or
the preparation phase must be re-run. It works in two independent layers:

1. **The fingerprint (a sha256 hash).** Built from `epsilon`, `poi_definitions` (in
   order), the snapshot count and a digest of the snapshot weightings, and, for a
   fixed whitelist of components and columns (Bus, Generator, Line, Load,
   StorageUnit, Store, Link, GlobalConstraint -- see `_STATIC_COLS`/`_TS_ATTRS` in
   `mga_engine/preparation_state.py`), their static input columns and a few
   time-varying input series. If the recomputed hash differs from the one stored in
   the file, `check_state` returns `"MISMATCH"` and cell 9 raises a `RuntimeError`.
2. **The sampling parameters (`n_vertices`, `n_interior_levels`, `n_samples_per_level`,
   `epsilon_min`, `seed`).** Stored as plain attributes and compared by cell 9's own
   code (for the first four) plus `check_state`'s `requested_seed` parameter (for
   `seed`). Any mismatch also raises `RuntimeError`. A file saved before these
   attributes existed is reported `"UNVERIFIED"` instead and is used as-is.

**Forces a rerun (mismatch or unverified-then-manual-check):** changing `epsilon`,
`poi_definitions` (contents or order), the snapshot count/weightings, any whitelisted
static column above, `n_vertices`, `n_interior_levels`, `n_samples_per_level`,
`epsilon_min`, or `seed`. Setting `rerun_preparation = True` always reruns regardless
of what `check_state` would say.

**Does NOT force a rerun -- and is INVISIBLE to the fingerprint entirely,** so a
changed network can silently reuse a stale cached state:
- A `Link` gaining a second output (`bus2` + `efficiency2`), i.e. any bus beyond
  `bus0`/`bus1`.
- `Carrier.co2_emissions` (the whole `Carrier` component is not fingerprinted).
- `StorageUnit.cyclic_state_of_charge` and `Store.e_cyclic`.
- A time-varying `standing_loss` on a `StorageUnit` or `Store`.
- A time-varying `marginal_cost` on any component (only the STATIC `marginal_cost`
  and `efficiency` columns are hashed) -- this is the sharpest gap, since it changes
  the true optimum while `check_state` still reports `"PASS"`.

(Verified in Step 3c by mutating an in-memory network and recomputing
`network_fingerprint`, with a working control mutation to confirm the test itself
detects a change at all.)

**To force a rerun regardless of the fingerprint:** set `rerun_preparation = True` in
cell 9, or delete/rename the file at `preparation_state_path` before running the cell.

## Settings you are most likely to change first

- **`epsilon`** (cell 4 or 5) -- widens or narrows the near-optimal space you explore.
- **`n_vertices` / `n_samples_per_level` / `n_interior_levels` / `epsilon_min`**
  (cell 9) -- the sampling budget; raise these first if a point falls outside the
  sampled convex hull.
- **`poi_definitions`** (cell 7) -- which capacities you explore, and in what units
  they are compared.
- **`request`** (cell 18) -- what you actually ask navigation for: priorities,
  directions, magnitudes.
- **`preparation_state_path`** and **`save_path`** (cells 9, 26) -- where results are
  read from and written to; change these to keep multiple states side by side.
- **`network_source`** and `build_network()`'s body (cells 2, 4) -- swap in your own
  network.
- **The `optimize(...)` keyword arguments** (cells 4, 5) -- the solver is hardcoded to
  HiGHS throughout the engine; this is the only place to pass it different options.
- **`epsilon_I`** (cell 24) -- how finely the traverse cost curve tracks the true cost
  versus a straight interpolation.

## Internal / derived variables (not intended for users)

Not settings -- computed from the ones above, or plumbing. Listed so they are not
mistaken for configuration: `opt_cost`, `poi_specs`, `poi_units`, `poi_min`,
`poi_max`, `n_total`, `state`/`status`/`message`/`sampling_params`/`mismatches`
(cell 9), `fingerprint` (cell 11), every `fig_*` handle, `alpha_s`/`alpha_t` and
their derived `p_s`/`p_t`/`cost_s`/`cost_t`, `tau`/`SG`/`SL`/`SE`/`delta` and their
`*_out` results, `loop1_bounds`, `P_tilde`/`apoi_labels`/`apoi_scale`, `SI`/`SS`/
`node_betas`/`node_costs`/`env_betas`/`env_costs`, and the widget `Output` handles
(`out_proj`, `out_restricted`, `out_radar`, `out_traverse`).

Also not exposed by the notebook at all -- engine-level constants a user would have to
edit source to change: `project_hull`/`compute_alpha_hull_2d`'s
`max_iterations=50, min_improvement=1e-6`; `surrogate._solve_milp`'s
`big_bound=1e3`; `navigation.LOCK_IN_TOL=1e-6`; `surrogate.MILP_RANDOM_SEED=0`; and
the HiGHS solver choice used throughout the engine.
