# mga_engine API overview

This file is a map of the `mga_engine` package: what each module is for, and what
each public function or class does in one line. It is not a reference -- full
parameter, return, and edge-case detail lives in each function's own NumPy-style
docstring (`help(mga_engine.<module>.<name>)`, or just read the source; every
function in the package has one). Private helpers (leading underscore) are omitted
below unless a user of the notebook would plausibly need to know one exists.

## network

Small example PyPSA capacity-expansion network for demos and tests; not on the
notebook's own path (the notebook defines its own `build_network()` in-cell).

| Function/class | One-line purpose | Called by |
|---|---|---|
| `build_network` | Build the example 3-bus network, unsolved. | `tests/test_smoke.py`'s fixture (as a guard, no solve). Not imported by the notebook, which defines its own same-named function. |

## poi

Defines Properties of Interest (installed capacity, in GW, of one or more carriers)
in their two equivalent forms: an evaluator over a solved network, and a linopy
expression to optimise over.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `PoiSpec` | One PoI in its two equivalent representations (`evaluate`, `linopy_expr`). | Returned by `capacity_poi`; consumed throughout the engine wherever a `PoiSpec` is expected (`gp_solver`, `vertex_sampling`, `projection`, `evaluate_all`). |
| `capacity_poi` | Build a PoI for the total installed capacity of one or more carriers. | `make_poi_specs`. |
| `make_poi_specs` | Build the ordered list of PoI specs from declarative definitions. | Notebook cells 7, 9, 15; `tests/test_smoke.py`'s fixture. |
| `evaluate_all` | Evaluate every PoI on a solved network. | `preparation_state.run_preparation` (for `p_star`); `vertex_sampling._solve_mp`. |

## ccp_solver

Solves CCP(p), the real-time LP cost surrogate that prices any point inside the
sampled convex hull in milliseconds, from the stored samples alone.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `solve_ccp` | Solve CCP(p) for a single target point. | Notebook cells 18, 24; `traverse.extract_breakpoints`; tests. |

## gp_solver

Third step of the preparation phase: solves GP(p^i) for every sampled point,
pricing it and recovering its PoI duals via a fresh linopy model per sample.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `solve_all_gp` | Solve GP(p^i) for every sampled point and return costs, duals and solutions. | `preparation_state.run_preparation`. |

Note: `_solve_single_gp` (private) is where the PoI duals' sign convention is
settled and documented -- see its Notes section if you need the exact convention
(un-negated, d(cost)/d(p_i) in EUR/yr per GW).

## vertex_sampling

Second step of the preparation phase: samples the extreme points (vertices) of the
near-optimal PoI space by solving MP(r) for random directions under a cost slack.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `sample_vertices` | Sample `n_samples` vertices of the near-optimal space by solving MP(r) for random directions. | `preparation_state.run_preparation`; `interior_sampling.sample_interior`. |

Note: `_build_mga_model` and `_solve_mp` (private) are reused directly by
`projection.py`, which imports them to reuse the same MP(r) machinery for its
on-demand 2-D projections.

## interior_sampling

Runs the same MP(r) sampling at successively tighter cost slacks, so the
near-optimal space is covered radially and not just along its outer boundary.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `derive_epsilon_levels` | Derive the interior-sampling epsilon levels from the outer epsilon. | `preparation_state.run_preparation`; tests. |
| `sample_interior` | Sample interior points by running MP(r) once per descending epsilon level. | `preparation_state.run_preparation`. |

## traverse

Third step of the exploration phase: walks the path from the current to the target
solution, extracting cost breakpoints (Algorithm 2) and a subgradient lower bound
(Algorithm 3) along the way.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `extract_breakpoints` | Algorithm 2: extract the interpolation breakpoints between alpha_s and alpha_t. | Notebook cell 24; tests. |
| `generate_subgradients` | Algorithm 3: build the subgradient lower bound of the cost along the path. | Notebook cell 24; tests. |
| `interpolate` | Evaluate the interpolated point and its linear cost estimate at a given beta. | Notebook cell 24; tests. |

## navigation

Second step of the exploration phase (Algorithm 1): turns a priority-ordered user
preference over PoIs into the weights of a target solution, via a sequence of LPs
over the sample simplex.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `navigate` | Compute target solution weights from the current ones and a user preference. | Notebook cell 18; tests. |

## projection

On-demand plotting support: computes the TRUE 2-D boundary of the near-optimal
space between a pair of PoIs by solving MP(r) on a live network, avoiding the
2-D-projection artefacts of plotting raw sample points.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `project_hull` | Compute the 2-D boundary of the near-optimal space projected onto PoIs (i, j). | Notebook cells 15 (`_hull_for_pair`) and 28 (figure export). |

## alpha_projection

The cheap counterpart to `projection`: computes the 2-D boundary of the SAMPLED
space (the convex hull of `P_all`, optionally intersected with navigation's
loop-1 bounds) via small SciPy LPs over the weights, with no PyPSA model involved.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `compute_alpha_hull_2d` | Compute the 2-D boundary of the sampled space projected onto APoIs (i, j). | Notebook cells 20 (`update_restricted`) and 28 (figure export); tests. |

## surrogate

The surrogate module is included in the workflow but was not fully validated.
Its formulation and results were not verified within the scope of this thesis
due to time constraints, and it should be treated as experimental.

Recovers a surrogate cost vector r under which a chosen PoI-space point is optimal
over the sampled space, formulated as a big-M MILP solved directly through highspy.

| Function/class | One-line purpose | Called by |
|---|---|---|
| `SurrogateResult` | Outcome of one surrogate cost recovery (success or failure). | Returned by `solve_surrogate_costs`; notebook cell 27 reads its fields. |
| `signed_boundary_distance` | Signed distance from a point to the boundary of the sampled convex hull. | `solve_surrogate_costs`; tests. |
| `project_to_boundary` | Project a point onto the nearest point of the hull's surface. | `solve_surrogate_costs`; tests. |
| `solve_surrogate_costs` | Recover a surrogate cost vector r that makes a point optimal over the samples. | Notebook cell 27; tests. |

## preparation_state

Orchestrates the whole preparation phase (`run_preparation`), persists its output
to one netCDF4 file (`save`/`load`), and fingerprints the network setup so a
cached state can be checked against the current one before reuse (`check_state`).

| Function/class | One-line purpose | Called by |
|---|---|---|
| `network_fingerprint` | Return a sha256 fingerprint of a whitelisted part of the network setup. | `make_fingerprint`; `check_state`. |
| `fingerprint_summary` | Build the small human-readable summary stored alongside the fingerprint. | `make_fingerprint`; `check_state`. |
| `make_fingerprint` | Bundle the three fingerprint pieces to hand to `save(fingerprint=...)`. | Notebook cell 11. |
| `check_state` | Compare a loaded state's fingerprint against the current network/PoIs/epsilon (and, manually in the notebook, the sampling parameters). | Notebook cell 9. |
| `save` | Save the preparation phase output to one .nc file, written directly with netCDF4. | Notebook cell 11 (as `save_state`). |
| `load` | Load a preparation state from a .nc file written by `save`. | Notebook cell 9 (as `load_state`); `tests/conftest.py`. |
| `run_preparation` | Run the whole preparation phase (base solve, vertex sampling, interior sampling, GP pricing) and return everything `save` needs. | Notebook cell 9. |

See `docs/configuration.md` for exactly what `network_fingerprint` does and does not
cover, and the consequences for cache reuse.

## Pipeline order

How the modules are typically called, preparation phase first, then exploration:

1. **poi** -- define the Properties of Interest for the network.
2. **vertex_sampling** -- sample the outer boundary of the near-optimal space.
3. **interior_sampling** -- sample its interior at tightening cost slacks (calls `vertex_sampling` internally).
4. **gp_solver** -- price every sampled point and recover its PoI duals.
5. **preparation_state** -- orchestrate steps 1-4 (`run_preparation`) and persist/verify the result (`save`/`load`/`check_state`).
6. **ccp_solver** -- price any point inside the sampled hull in milliseconds, from the saved samples.
7. **navigation** -- turn a user preference into a target solution's weights.
8. **traverse** -- walk the path from the current to the target solution.
9. **projection** / **alpha_projection** -- plot the near-optimal space (true and sampled boundaries) for the notebook's interactive figures.
10. **surrogate** -- (experimental) recover a cost vector under which a chosen point is optimal.
11. **network** -- a standalone example network, not part of the pipeline itself.
