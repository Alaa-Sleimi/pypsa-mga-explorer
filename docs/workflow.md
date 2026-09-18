# Workflow: preparation phase to exploration phase

`mga_exploration.ipynb` runs the MGA-Compass method in two phases: a heavy
**preparation phase**, run once per network, that samples and prices the
near-optimal space; and an interactive **exploration phase** that navigates and
traverses within it using only the saved samples, with no further solver calls
except where noted. This file walks through both, in the order the notebook runs
them. For what each setting does, see `docs/configuration.md`; for what each
function does, see `docs/api.md`.

## Preparation phase

**Build or load the network.** Section 1 (cells 2-5) either builds a small demo
network in the notebook (`network_source = "fresh"`, cell 4) or loads PyPSA's
bundled `model_energy` example (`network_source = "load_file"`, cell 5). The
`model_energy` path gets its own walkthrough in `docs/example_model_energy.md`.

**Solve the original optimization problem (OP).** Still in cells 4/5: the network
is solved once with plain PyPSA (`network.optimize(...)`, HiGHS), giving the true
cost optimum `opt_cost` and, once PoIs are defined in Section 2, the optimal PoI
vector `p_star`. This is a direct PyPSA call, not an `mga_engine` function.

**Add the sub-optimality (slack) constraint.** This is not a separate notebook
step: it happens automatically, once per fresh network build, inside
`vertex_sampling` (see `docs/api.md#vertex_sampling`) whenever it samples a
direction. It adds one constraint pinning total system cost to at most
`(1 + epsilon) * opt_cost`, which is what "near-optimal" means throughout the rest
of the pipeline.

**Vertex sampling.** `vertex_sampling.sample_vertices` solves the slack-constrained
model for a batch of random directions, giving points at the outer boundary of the
near-optimal PoI space. Run from Section 3 (cell 9) via
`preparation_state.run_preparation`.

**Interior sampling.** `interior_sampling.sample_interior` repeats vertex sampling
at a sequence of progressively tighter slacks (derived from `epsilon` down to
`epsilon_min`), so the interior of the space is covered as well as its outer
boundary. Also run from cell 9 via `run_preparation`.

**Pricing every sample.** Once the vertex and interior samples are combined into
one matrix, `gp_solver.solve_all_gp` re-solves the network once per sample with
its PoI values pinned, recording the minimum cost and the PoI duals at that point.
This is what lets the exploration phase price and navigate the space later without
touching the network again. Also run from cell 9, as the last step inside
`run_preparation`.

**What gets cached, and by `check_state`.** Cell 9 either runs all of the above
fresh or loads a previously saved `.nc` file, after checking with
`preparation_state.check_state` that the file actually matches the current
network, PoIs, epsilon, and sampling parameters; cell 11 writes the result (via
`preparation_state.save`) if you want to keep it. Exactly what enters that check,
what does not, and how to force a rerun is covered in full in
`docs/configuration.md`, under "Cache invalidation" -- it is not repeated here.

## Exploration phase

Everything below works from the samples produced above (`P_all`, `v`, `Gamma`)
and needs no further network solve, with the one exception noted under Projection.

**Navigation (Section 4, cells 16-18).** You state a preference: for each PoI you
care about, whether it should go up, go down, or stay put, by how much, and in
what order of priority (the order you list them in). `navigation.navigate` starts
from the current cost-optimal solution and works through your preferences one at
a time, honoring the higher-priority ones first and doing the best it can with the
rest, then settles on the cheapest solution consistent with everything it managed
to satisfy. What comes back is a target solution, plus a note on which of your
requests had to be scaled back or couldn't be met at all. Cell 20 (Section 4.2)
shows the region navigation actually searched after your first-priority requests
were locked in; cell 22 (Section 4.3) shows a radar chart comparing the current
and target solutions.

**Traverse (Section 5, cell 24).** Once you have a target, traverse interpolates
between the current solution and it, and plots how the true system cost behaves
along that path. A straight-line interpolation of cost would understate how the
cost actually changes partway along the path, so traverse also computes a
subgradient lower bound: a guaranteed floor on the true cost at every point along
the path, reusing the PoI duals already computed during preparation rather than
solving anything new. The plot shows both the (piecewise) true cost and this lower
bound, alongside how each PoI's value moves as you slide from the current solution
to the target.

**Projection (Sections 3.2 and 4.2).** Two different views of the near-optimal
space, for a pair of PoIs you choose:

- Cell 15 (Section 3.2) shows the TRUE boundary of the near-optimal space, computed
  on request by solving the network for the selected pair -- this is the one place
  in the exploration phase that does call the solver again, and why that plot has a
  short delay the first time a pair is selected.
- Cell 20 (Section 4.2) instead shows the boundary of the SAMPLED space (the convex
  hull of the points from preparation), which is cheap to compute and is what
  navigation actually searches, intersected with the bounds your first-priority
  requests fixed.

The rest of the notebook builds on these two phases without adding a third: Section
6 (cell 26) bookmarks a navigation session's request and result to a small file;
Section 7 (cell 27) additionally tries to recover a cost vector under which your
target solution would be optimal. The surrogate module is included in the workflow
but was not fully validated. Its formulation and results were not verified within
the scope of this thesis due to time constraints, and it should be treated as
experimental. Cell 28 exports every figure above to `figures/` for the thesis.

## Where the notebook and this file might drift

This file describes the pipeline as the notebook and engine stood at the time of
this documentation pass. If a future change to the notebook or engine is not
reflected here, trust the notebook's actual cell code over its own markdown
commentary, and trust this file's cross-references to `docs/api.md` and
`docs/configuration.md` over any equations or step order repeated from memory
elsewhere -- markdown cells and standalone docs are both secondary to what the code
actually does.
