# Code Review — `mga_engine/`

*Read-only review, 2026-08-18. No code was modified. Scope: `mga_engine/*.py`, `tests/*.py`, and the notebook's import surface (the notebook itself is out of scope for edits).*

**Legend:** [notebook-facing] = a change here could affect what `mga_exploration.ipynb` calls; [safe-internal] = not reachable from the notebook.

---

## 1. Module Inventory

The notebook imports, per its cells: `poi.make_poi_specs`, `preparation_state.{run_preparation, check_state, load, save, make_fingerprint}`, `projection.project_hull`, `ccp_solver.solve_ccp`, `navigation.navigate`, `alpha_projection.compute_alpha_hull_2d`, `traverse.{interpolate, extract_breakpoints, generate_subgradients}`, and `surrogate.solve_surrogate_costs`. Notably, the notebook does **not** import `mga_engine.network` — it defines its own 5-bus `build_network()` and passes it into `run_preparation`.

| Module | Status |
|---|---|
| `__init__.py` | One comment line, no exports. The notebook imports submodules directly, so this is fine as-is. |
| `network.py` | Example 3-bus capacity-expansion network. Exports `build_network()` plus cost constants. **Not on the notebook path** (the notebook has its own network factory). Used as the default network by `gp_solver.solve_all_gp` (when `build_network_fn` is omitted), by every module's `__main__` demo block, and by `tests/test_ray_sampling.py` / `tests/test_projection_plot.py`. Not dead, but it is demo/test infrastructure, not production path. |
| `poi.py` | PoI definitions. Exports `PoiSpec`, `capacity_poi`, `POI_DEFINITIONS`, `make_poi_specs`, `evaluate_all`, `MW_TO_GW`. Imported by `vertex_sampling`, `gp_solver`, `projection`, and the notebook (`make_poi_specs`; the notebook's markdown also references `capacity_poi`). Central and alive. |
| `vertex_sampling.py` | MP(r) vertex sampling of F^P. Exports `sample_vertices`, `sample_boundary_rays`; private `_build_mga_model`, `_solve_mp` are also imported by `projection.py`. `sample_vertices` is used by `interior_sampling` and `preparation_state.run_preparation`. **`sample_boundary_rays` is used only by `tests/test_ray_sampling.py` and `tests/test_projection_plot.py`** — it is not called by the notebook or by `run_preparation`. It looks like an alternative sampler kept for comparison; flag for a deliberate keep/drop decision (open question, not asserted dead). |
| `interior_sampling.py` | Sweeps `sample_vertices` over decreasing epsilon levels. Exports `derive_epsilon_levels`, `sample_interior`, plus legacy constants `EPSILON_LEVELS`, `SAMPLES_PER_LEVEL`. Called by `preparation_state.run_preparation`; not imported by the notebook directly. |
| `gp_solver.py` | GP(pⁱ) solves: min cost with PoIs pinned; returns costs `v`, duals `Gamma`, and full solution dicts. Exports `solve_all_gp` (private `_solve_single_gp`, `_extract_solution`). Called by `run_preparation` and `tests/test_ray_sampling.py`. |
| `ccp_solver.py` | **Alive — not a leftover.** Exports `solve_ccp` (convex-combination LP over the samples). Imported by the notebook (Sections on navigation and traverse), by `traverse.py` (Algorithm 2 midpoint solves), and by `tests/test_navigation.py` / `tests/test_traverse.py`. |
| `alpha_projection.py` | **Alive — not a leftover.** Exports `compute_alpha_hull_2d`: 2D hull of the *sampled* space conv(P̃ columns) via scipy `linprog`, with optional loop-1 bounds from `navigate()`. Imported by the notebook (Section 4.x plotting) and `tests/test_alpha_projection.py`. It complements `projection.py` (true LP boundary on the live network) rather than replacing it — both are used by the notebook. |
| `projection.py` | True 2D projection boundary of the near-optimal LP space via adaptive edge-normal MP(r) solves. Exports `project_hull`. Imported by the notebook and `tests/test_projection_plot.py`. Reuses `vertex_sampling`'s private helpers (see §5). |
| `navigation.py` | Algorithm 1: exports `navigate` (priority-ordered feasibility + improvement loops over LPs, returns `alpha_t`, updated direction sets/deltas, and loop-1 bounds). Imported by the notebook and two tests. |
| `traverse.py` | Algorithms 2 and 3: exports `extract_breakpoints`, `generate_subgradients`, `interpolate`. Imports `solve_ccp`. Imported by the notebook and `tests/test_traverse.py`. |
| `preparation_state.py` | The largest module: netCDF4 save/load of the preparation output, the network fingerprint (`network_fingerprint`, `make_fingerprint`, `check_state`), and the orchestrator `run_preparation`. Heavily notebook-facing (5 symbols imported). |
| `surrogate.py` | SP(p) MILP via raw highspy: exports `solve_surrogate_costs`, `SurrogateResult`, plus geometry helpers `signed_boundary_distance`, `project_to_boundary`. Imported by the notebook (Section 4.1) and `tests/test_surrogate.py`. Untracked in git (new file). |

**No module is fully dead.** The two suspects (`alpha_projection.py`, `ccp_solver.py`) are both actively imported by the notebook. The only "leftover-shaped" symbol is `sample_boundary_rays` (tests-only).

---

## 2. Import & Dependency Map

Top-level (non-`__main__`) imports within `mga_engine/`:

```
poi                ← (nothing internal)
network            ← (nothing internal)
vertex_sampling    ← poi
interior_sampling  ← vertex_sampling
projection         ← poi, vertex_sampling (_build_mga_model, _solve_mp — PRIVATE names)
gp_solver          ← network, poi
ccp_solver         ← (nothing internal)
traverse           ← ccp_solver
navigation         ← (nothing internal)
alpha_projection   ← (nothing internal)
preparation_state  ← poi, vertex_sampling, interior_sampling, gp_solver
                     (all lazily, inside run_preparation — avoids import cost at load())
surrogate          ← (nothing internal; highspy imported lazily inside _solve_milp)
```

Notebook → engine: `poi`, `preparation_state`, `projection`, `ccp_solver`, `navigation`, `alpha_projection`, `traverse`, `surrogate`.
Tests → engine: additionally `network`, `vertex_sampling`, `gp_solver`.

**Findings:**

- **No circular imports.** The graph is a clean DAG; `preparation_state`'s lazy imports also prevent any future cycle through the orchestrator.
- **No genuinely unused imports.** Two borderline cases: `gp_solver.py:16-17` imports `logging`/`warnings` at module top but uses them only in `__main__` (harmless); `vertex_sampling.py:15` imports `pypsa` only for type annotations (fine).
- **Cross-module use of private names:** `projection.py:32` imports `_build_mga_model` and `_solve_mp` from `vertex_sampling`. Not a bug, but underscore names imported across modules are a refactoring hazard (see §5).
- **Pinned-API sensitivity** (no version *mismatch* found, but these are the fragile spots if anything ever shifts):
  - `gp_solver.py:86` — `network.model.solver_model.getObjectiveValue()` reaches through linopy into the raw HiGHS object (see §3, C7).
  - `surrogate.py:_solve_milp` — raw highspy `addVars`/`changeColsIntegrality`/`addRows`; the module docstring correctly confines highspy to this one function and says so.
  - `preparation_state._fingerprint_object` / `gp_solver._extract_solution` — iterate `network.components` with `.static`/`.dynamic`, the PyPSA 1.x component API; consistent with the pinned PyPSA 1.1.2.
  - `network.py:102` — `n.sanitize()` is a PyPSA 1.x call; fine at the pin, but it's the kind of call to re-check if the pin ever moves (it must not now).

---

## 3. Correctness Issues

Ordered roughly by severity. None were fixed; all are described only.

**C1. `traverse.py:56-79` — Algorithm 2 recursion drifts off the linear interpolation path.** *(severity: high — as an open question, not an asserted bug)*
`mid_interpolate` recurses with `(alpha_l, alpha_star)` and `(alpha_star, alpha_u)`, where `alpha_star` is the **CCP-optimal** weight vector at the midpoint, not the linear-interpolation weights `alpha_m`. From the second recursion level on, `alpha_m = (alpha_l + alpha_u)/2` is therefore no longer `(1-β_m)·alpha_s + β_m·alpha_t`, yet `beta_m = (beta_l + beta_u)/2` is still recorded as if it were. Consequences: (a) the returned pairs `(beta, alpha*)` associate a path coordinate with a point that need not lie at that coordinate of the path; (b) `generate_subgradients` then evaluates `g(beta)` on the *pure* linear path (`traverse.py:141`), so the two algorithms use inconsistent geometry for the same `SI`. The module docstring ("breakpoints along the linear interpolation path in F^P") matches the pure-path reading, not the code. This may be exactly what Sina's paper specifies (interpolating between CCP solutions) — **do not resolve unilaterally; confirm against the paper.** If the paper means the pure path, the recursion should pass linear-interpolation alphas down (a small change, [safe-internal] to the engine, and signature-preserving so not notebook-breaking).

**C2. `surrogate.py:417-450, 474-476` — points *outside* conv(P) are mishandled/mislabeled.** *(severity: medium)*
`signed_boundary_distance` can return a negative value (exterior point), but `solve_surrogate_costs` only branches on `dist > tol_interior`. An exterior `p` falls through to the direct MILP solve, which is infeasible (constraint P·a = p, a in the simplex), then hits the "numerically eps-interior … nudged onto the hull" retry, and — if the projection heuristic happens to land inside — returns a success whose message claims the point was eps-interior. For a far-exterior `p` (e.g. a stale `P` passed with a fresh navigation target) this silently solves at a possibly distant projected point with a misleading message. `projection_distance` is recorded, which mitigates it, but an explicit `dist < -tol` branch (distinct status or at least an honest message) would prevent silent misuse. [safe-internal] — return dataclass shape unchanged.

**C3. `traverse.py:129` — active-set detection `alpha[i] > 0` has no tolerance.** *(severity: medium)*
`linprog`/HiGHS routinely returns weights like `1e-13` instead of exact zeros. Every such phantom weight adds a subgradient function to `SS`, which is wrong-ish mathematically (those samples are not truly in the optimal basis) and inflates the number of callables evaluated per β. A tolerance (e.g. `alpha[i] > 1e-9`) is the standard fix. [safe-internal].

**C4. `navigation.py:200-201` — absolute lock-in tolerance `1e-6` applied across APoIs with wildly different units.** *(severity: medium)*
Loop 2 locks each improved APoI value with `±1e-6`. APoI rows mix GW (order 1) and €/yr (order 1e8). For cost, ±1e-6 on a 1e8-scale number is far below solver feasibility tolerance (HiGHS default primal tol 1e-7 *relative* to row activity), so the lock can be numerically over-tight for cost rows and effectively an equality — the LP still solves because the locked point is feasible by construction, but it leaves zero slack for the final `solve_ncp(v)` and is the very offset the surrogate module then has to "nudge" around (its docstring explicitly documents working around this constant). A per-row relative tolerance (or at minimum a named constant) would make the interaction explicit. Note the surrogate docstring and `tol_interior` default were sized to absorb this — if the constant changes, revisit `surrogate.py` accordingly. [safe-internal] (signature unchanged), but flag: the *value* is load-bearing for `surrogate.py`.

**C5. `traverse.py:56-79` — unbounded recursion depth in `extract_breakpoints`.** *(severity: low)*
Each recursion level requires a ≥ `epsilon_I` improvement, and costs are bounded below, so termination is expected in practice — but there is no depth cap, and with a small `epsilon_I` on a large model a deep/degenerate recursion (each level = one CCP solve) or even Python's ~1000-frame limit is reachable. A `max_depth` guard would be cheap. [safe-internal] (new keyword with default — notebook calls unaffected).

**C6. `vertex_sampling.py:83-110` — `_solve_ray` leaks its per-ray constraints on an unexpected exception.** *(severity: low)*
The `ray_eq_{i}` constraints are removed after the solve, but if `assign_solution()` or `evaluate_all()` raises (not just a non-"ok" status), the removal is skipped and every subsequent ray on the reused model carries stale equality constraints. A `try/finally` around the solve would close the gap. Only reachable from tests today (`sample_boundary_rays` is not on the notebook path). [safe-internal].

**C7. `gp_solver.py:86` — objective read through `solver_model.getObjectiveValue()` instead of linopy.** *(severity: low)*
Bypassing linopy's `m.objective.value` couples the code to the raw HiGHS handle that linopy happens to keep, and to HiGHS's notion of the objective (which could differ from linopy's if an objective constant were ever involved — here `include_objective_constant=False` keeps them equal). Works at the pinned versions; fragile by construction. [safe-internal].

**C8. `interior_sampling.py:43` — module-level list as a default argument.** *(severity: low)*
`epsilon_levels: list = EPSILON_LEVELS` shares one mutable list across all calls. It is never mutated inside the function, so there is no live bug, but any future caller doing `epsilon_levels.append(...)` would corrupt the module constant. The classic `None`-default pattern avoids the trap. [safe-internal] (`run_preparation` always passes it explicitly; the constant must stay importable since `gp_solver.__main__` imports it).

**C9. `preparation_state.py:398-409` — `save()` assumes homogeneous solution dicts.** *(severity: low)*
Keys are taken from `solutions[0]` and shapes from `np.stack`; a sample whose dict is missing a key, or whose array has a different shape (possible if a GP solve ever left a component result empty), raises a raw `KeyError`/`ValueError` deep inside the write. The atomic-write wrapper cleans up the temp file correctly, so nothing corrupts — the failure is just undiagnosable. A pre-flight key/shape consistency check with a clear message would help. [safe-internal].

**C10. `run_preparation` seeding: interior levels ignore the `seed` parameter.** *(severity: low, reproducibility-relevant)*
`interior_sampling.sample_interior` (line 75) hardcodes `seed=i * 100` per level. Everything is still deterministic (good for benchmarking), but the `seed` argument of `run_preparation` only affects the vertex phase — two runs with different `seed` values share identical interior samples, which could mislead a variance study during benchmarking. Deriving level seeds from the caller's seed (e.g. `seed + i*100`) would fix it; **note this changes the sampled points, so any cached `preparation_state.nc` comparisons across code versions would differ** (fingerprint does not cover seeds — see §7). [safe-internal] signature-wise.

**C11. Resource handling — netCDF4: no leaks found.** `save()` closes on all paths (try/except/else + temp-file cleanup + `os.replace`), `load()` closes in `finally`. Explicitly checked; this is done right.

**C12. Non-determinism audit.** All RNG use is seeded (`default_rng(seed)`); scipy `linprog(method="highs")` and linopy/HiGHS LP solves are deterministic for a fixed model. Two caveats for benchmarking: (a) HiGHS MILP (`surrogate._solve_milp`) does not pin `threads`, and parallel MIP search can return different (equally optimal) `r` vectors run-to-run; (b) degenerate LPs can return different optimal vertices across HiGHS versions (not across runs). See §7.

**C13. `_lp_boundary_distance` (surrogate.py:134-177) — verified sound, with its stated caveat.** The all-weights-≥-t test is a correct *relative-interior* proxy (any relative-interior point of a hull admits strictly positive weights), and the docstring honestly labels it "conservative … adequate only as a fallback". No action; noted here so it isn't re-litigated later.

---

## 4. Comments & Docstrings

**(a) Stale / contradicting the code**

- `interior_sampling.py:111` — `__main__` comment says "interior (6 epsilon levels…)" but the default `EPSILON_LEVELS` has 5 (the 6-entry `colors` list at line 130 is a matching remnant; the extra color is silently unused).
- `traverse.py` module docstring says breakpoints are found "along the linear interpolation path", which the recursion does not strictly honor (see C1) — whichever way C1 resolves, one of code/docstring must change.
- `ccp_solver.py:4` — "Given a target point p in F^P": the feasible region of CCP is conv(samples) ⊊ F^P; the error message at lines 65-70 gets this right, the docstring oversimplifies.
- `poi.py:23-24` — `MW_TO_GW = 1e3` reads as "multiply by 1e3 to get GW" but is used as a divisor; the constant is really "MW per GW". Minor, but it invites a sign/direction slip in future PoIs.

**(b) Redundant (restate the obvious)** — rare, which is good. Examples: `traverse.py:81` ("Sort breakpoints by beta" above `SI.sort(key=...)`), `vertex_sampling.py:95` ("maximize l == minimize -l" — borderline, arguably earns its place), `navigation.py:52` ("Row 0 is cost…" duplicates the module docstring two paragraphs up).

**(c) Commented-out dead code**

- `tests/test_navigation.py:159-195` — Scenarios A and B are entirely commented out (including their `navigate` calls and plotting); only Scenario C runs. `plot_scenario_a` (lines 104-151) is now only referenced from the commented block, making it and the `matplotlib.patches` import effectively dead.
- `tests/test_traverse.py:81-87` — a duplicated block: `g_values`/`envelope` are computed twice back-to-back (lines 82-83 and 86-87), with the first comment claiming "Plot individual subgradient lines" and no plotting between them. The first pair is dead computation.

**(d) TODO/FIXME/HACK markers** — none found (grep-clean). The nearest thing is `traverse.py:71` "Paper condition (corrected)", which flags a deliberate deviation from the paper — worth keeping, but it should say *what* the uncorrected condition was so a reader can check it against the paper (relates to C1).

**Missing docstrings where one would help**

- `navigation.py:38` — `navigate()` itself has **no docstring**; the module docstring covers it, but `help(navigate)`/IDE hover shows nothing for the single most notebook-visible function in the engine. Moving/duplicating the parameter table into the function docstring is zero-risk. [notebook-facing surface, but docstring-only ⇒ safe]
- `poi.py:27-32` — `PoiSpec` has no class docstring describing the `evaluate`/`linopy_expr` contract (it lives only in the module docstring).
- `vertex_sampling.py:20, 35` — `_build_mga_model` and `_solve_mp` are undocumented yet imported by `projection.py`; they are de-facto shared API and deserve at least a one-liner each.
- `navigation.py` inner helpers (`solve_ncp`, `add_ineq`) have one-liners; adequate.

---

## 5. Code Organization & Style

- **Duplicated hull machinery** — `projection.py` and `alpha_projection.py` each carry near-identical `_ccw_hull_vertices`, `_outward_edge_normals`, `_already_tried`, `_dedupe`, and the same seed-then-refine loop skeleton. The two rightly differ in the solve step (live LP vs. weights LP) and in unit scaling (only `alpha_projection` normalizes axes), but the four geometry helpers could live in one private shared module. [safe-internal] — both public entry points keep their signatures. Keep it minimal: extract the helpers only, do not merge the loops.
- **Private imports across modules** — `projection.py:32` importing `_build_mga_model`/`_solve_mp`. Minimal fix: drop the leading underscores (or add public aliases) in `vertex_sampling` and update the one import. [safe-internal].
- **Long functions** — `navigate()` (~170 lines with two loops and three inner functions), `preparation_state.save()` (~95), `surrogate._solve_milp` (~150), `compute_alpha_hull_2d` (~100). All are readable top-to-bottom and per the no-rewrites constraint should stay; only `navigate()` would clearly benefit from extracting loop 1 and loop 2 into named helpers *if* it is ever touched for C4 anyway. [safe-internal].
- **Magic numbers that should be named constants** *(all [safe-internal])*:
  - `navigation.py:200-201` — the `1e-6` lock-in (twice; also load-bearing for `surrogate`, see C4).
  - `projection.py:214` / `alpha_projection.py:94` — `angle_tol=1e-3` duplicated in both `_already_tried`s.
  - `surrogate.py:220` — `big_bound=1e3` (and derived `M`); fine as a keyword default but worth a module constant with a comment on how it bounds `r`.
  - `traverse.py:35` — `epsilon_I=1e6` default is documented with units in the docstring; acceptable as-is.
- **Print-heavy library code** — every phase prints per-iteration lines (`[vertex]`, `[gp]`, `[interior]`, `[proj]`, `[state]`, `[nav]`). For a thesis notebook this is arguably a feature, but it (a) conflicts with the user's stated minimal-prints preference, (b) adds I/O time inside benchmark loops, and (c) cannot be silenced without redirecting stdout. A module-level `VERBOSE` flag or `logging` would be the minimal-scope fix. [notebook-facing in *output* only — no signatures change, but notebook cell outputs would look different; mark accordingly when acting on it.]
- **Inconsistent naming, minor** — `m` is used both for the PoI count (`m, n = P.shape`) and for the linopy model (`m = network.model`) — within different modules, so tolerable, but `gp_solver` uses `m_dim` alongside a model `m` in the same file; worth harmonizing on `m_dim` for counts if touched. Stray leading space in the comment at `navigation.py:183`. [safe-internal].
- **`__init__.py` exports nothing** — the notebook does 10+ submodule imports. A curated `__all__`/re-export block would shorten notebook imports, but adding it is optional; **doing it must not remove submodule importability** (the notebook imports `from mga_engine.poi import ...` etc.). Recommend leaving as-is until after the thesis crunch. [notebook-facing if done wrong; no-op if done as pure additions.]

---

## 6. Test Assessment

**File-by-file classification:**

| File | Type | Details |
|---|---|---|
| `test_ray_sampling.py` | **Assertion-based**, custom runner | Real `assert`s with messages, PASS/FAIL printout, non-zero exit on failure. Setup (2 full network solves + GP solves per check via `solve_all_gp`) runs in `__main__`. Slow (solver-bound). |
| `test_projection_plot.py` | **Script/visual demo** | No assertions at all; solves, plots, saves `projection_test.png`, and calls `plt.show()` (blocks headless CI). The docstring openly says the plot "doubles as a correctness check" — i.e., a human is the assertion. |
| `test_navigation.py` | **Script with 1 assertion** | Module-level execution on import; needs `data/preparation_state.nc`; scenarios A/B commented out (§4c); one real `assert` (loop-1 bounds coverage) plus rich prints; plotting code present but dead. |
| `test_traverse.py` | **Script/demo** | Module-level execution; needs the `.nc`; zero assertions; plots + `plt.show()`. |
| `test_alpha_projection.py` | **Assertion-style**, custom `check()` | Four genuine correctness checks (containment, monotone shrinking, cost-axis range, degeneracy) recorded as booleans with exit code — but not `assert` statements, and module-level execution requiring the `.nc`. |
| `test_surrogate.py` | **Assertion-style**, custom `check()` | Best of the set: fast closed-form geometry checks on a synthetic unit square, plus one gated integration check that self-skips when the `.nc` is absent. Module-level execution. |

**Coverage gaps — modules with NO test coverage at all:** `poi.py` (nothing checks `capacity_poi`'s evaluate/linopy_expr agree, or the MW→GW conversion), `interior_sampling.py` (in particular `derive_epsilon_levels`, which is pure and trivially testable), `preparation_state.py` (its round-trip check lives only in its own `__main__`; the fingerprint PASS/MISMATCH/UNVERIFIED logic — pure and high-value — has no test), `gp_solver.py` (exercised incidentally by `test_ray_sampling`, never asserted directly: no dual-sign check, no v ≥ opt_cost check), `network.py` (only implicitly). `navigation.py` has exactly one asserted property; `traverse.py` has zero assertions despite C1 being precisely the kind of thing a test would catch.

**Pytest adoption plan (recommendation only — no test code written):**

1. **Make files collectable.** The blockers are (a) module-level execution in 4 of 6 files — wrap the setup + checks into `test_*` functions (the existing `check(label, ok)` calls convert naturally to `assert ok, label`); (b) `plt.show()` calls — drop them or gate behind `if __name__ == "__main__"`, and use the `Agg` backend via a `matplotlib.use("Agg")` in a `conftest.py`; (c) the custom runners (`run_all_checks`, `results` lists, `sys.exit`) become redundant — pytest is the runner. `test_ray_sampling.py` converts most cleanly: its `CHECKS` list becomes parametrized tests over a session-scoped fixture holding `setup`/`P_rays`.
2. **Fixtures for the expensive state.** A `conftest.py` with: a session-scoped `tiny_network` fixture (the existing 3-bus `mga_engine.network.build_network` is already the right size); a session-scoped `prep_state` fixture that either loads `data/preparation_state.nc` if present or runs a *miniature* `run_preparation` (e.g. `n_vertices=6`, `n_interior_levels=2`, `n_samples_per_level=3`) once per session — this removes the hard dependency of `test_navigation`/`test_traverse`/`test_alpha_projection` on a gitignored file; and fixed seeds everywhere (already the convention).
3. **Fast/slow separation.** Register two markers in `pyproject.toml`/`pytest.ini`: `@pytest.mark.solver` for anything touching HiGHS through PyPSA/linopy (ray sampling, projection, GP) and `@pytest.mark.milp` for `surrogate` integration. Default CI/benchmark runs use `-m "not solver and not milp"`, which would still execute: all of `test_surrogate`'s synthetic-square checks, `derive_epsilon_levels`, fingerprint logic, `ccp_solver`/`navigation`/`traverse`/`alpha_projection` against a tiny hand-built `P_all`/`v` (a 4-6 column matrix like `test_surrogate`'s unit square — no solver needed, since those four modules are pure scipy). That last point is the big win: **the entire exploration-phase stack is testable in milliseconds without PyPSA.**
4. **Highest-value new unit tests, in order:** (i) `navigate` invariants on a synthetic P̃ — target satisfies loop-1 bounds, flipped sets are consistent, cost row minimized last; (ii) `extract_breakpoints` on a synthetic path with a known kink (would settle C1's observable behavior); (iii) `solve_ccp` recovers a vertex exactly and raises the documented RuntimeError outside the hull; (iv) `poi.capacity_poi`: `evaluate` equals the linopy expression's value on a solved tiny network (one `solver` -marked test); (v) `preparation_state` save→load round-trip into `tmp_path` plus all three `check_state` outcomes; (vi) `derive_epsilon_levels` edge cases (`n_levels=1`, assertion triggers).
5. **Reproducibility plumbing.** Seeds as fixture constants; `tmp_path` for every artifact (the current tests write PNGs into the repo/tests dir); assert on numbers, never on images — keep the two plot scripts as demos (rename to `scripts/` or `demo_*.py` so pytest doesn't collect them) rather than forcing them into tests.

---

## 7. Benchmarking Readiness

| Item | Status | Notes |
|---|---|---|
| Seeded randomness everywhere | ✅ mostly | All `default_rng` calls seeded. Caveat C10: interior seeds are hardcoded per level, so `seed` only varies the vertex phase. |
| LP determinism | ✅ | Fixed models + HiGHS LP are run-to-run deterministic. |
| MILP determinism | ⚠️ | `surrogate._solve_milp` does not set `threads=1` (or `random_seed`); parallel MIP can return different optimal `r` on the solution ray. Set threads explicitly before benchmarking surrogate calls. |
| Timing hooks | ❌ | No timing anywhere in the engine; benchmarks would have to wrap calls externally (fine) but per-solve timing (e.g. count + wall time per MP(r)/GP/CCP) would need adding. |
| No accidental global state | ✅ mostly | No module mutates shared state across calls; models are rebuilt fresh per phase; `_solve_ray` constraint cleanup has the exception-path gap (C6); `POI_DEFINITIONS`/`EPSILON_LEVELS` are read-only module globals (C8 pattern risk). |
| Solver setup/teardown | ✅ | Fresh `create_model` per phase; netCDF handles closed on all paths (C11); atomic `.nc` writes. |
| Cache validity guard | ✅ | Fingerprint (`check_state`) protects against benchmarking on a stale preparation state. Gap: the fingerprint covers network/PoIs/epsilon but **not** the sampling parameters or seeds — two states with different `n_vertices`/seeds both PASS. The sampling params are stored as attributes, so a caller *can* check them; the notebook should (open question whether it does). |
| Quiet inner loops | ❌ | Per-iteration `print`s in every sampling/GP loop add I/O inside timed regions and flood benchmark logs (§5). |
| Recursion safety under benchmark-scale inputs | ⚠️ | C5: `extract_breakpoints` depth is uncapped; benchmark-scale models with small `epsilon_I` are where it would bite. |

---

## 8. Prioritized Action List

Ranked by severity × effort (highest leverage first). "Effort" is for the minimal-scope fix described above.

| # | Action | File(s) | Severity | Effort | Tag |
|---|--------|---------|----------|--------|-----|
| 1 | Resolve C1 with Sina: does Algorithm 2 recurse on CCP alphas or on the linear path? Then align code or docstring | `traverse.py:56-79` | high (open question) | low (ask) / med (fix) | [safe-internal]* |
| 2 | Add exterior-point branch (`dist < -tol`) with honest status/message | `surrogate.py:417-476` | med | low | [safe-internal] |
| 3 | Tolerance on active-set test (`alpha[i] > tol`) | `traverse.py:129` | med | low | [safe-internal] |
| 4 | Name + rescale the `1e-6` lock-in per APoI row (coordinate with surrogate's `tol_interior`) | `navigation.py:200-201` | med | med | [safe-internal], value is load-bearing |
| 5 | Pytest adoption step 1-3 (collectable files, conftest fixtures, `solver`/`milp` markers) | `tests/*` | med | med | [safe-internal] |
| 6 | Unit tests for the pure exploration stack (navigate/ccp/traverse/alpha_projection on synthetic P̃) | `tests/` (new) | med | med | [safe-internal] |
| 7 | Set `threads=1` (and optionally `random_seed`) in the surrogate MILP for benchmark determinism | `surrogate.py:_solve_milp` | med | low | [safe-internal] |
| 8 | Verbosity flag / logging instead of unconditional prints in sampling/GP/projection loops | `vertex_sampling`, `gp_solver`, `interior_sampling`, `projection`, `preparation_state` | med | med | [notebook-facing] (cell *output* changes only; no signatures) |
| 9 | `max_depth` guard on `extract_breakpoints` (new kwarg, default generous) | `traverse.py` | low-med | low | [safe-internal] |
| 10 | `try/finally` around ray-constraint removal | `vertex_sampling.py:83-110` | low | low | [safe-internal] |
| 11 | Decide fate of `sample_boundary_rays` (keep as comparison sampler for the thesis, or move to demos) | `vertex_sampling.py` | low (clarity) | low | [safe-internal] (tests import it) |
| 12 | Delete commented-out Scenarios A/B + dead `plot_scenario_a`, and the duplicated block in test_traverse | `tests/test_navigation.py:104-195`, `tests/test_traverse.py:81-87` | low | low | [safe-internal] |
| 13 | Docstring for `navigate()` (and `PoiSpec`, `_build_mga_model`, `_solve_mp`) | `navigation.py`, `poi.py`, `vertex_sampling.py` | low | low | [safe-internal] (docs only) |
| 14 | Read objective via `m.objective.value` instead of `solver_model.getObjectiveValue()` | `gp_solver.py:86` | low | low | [safe-internal] |
| 15 | Fix stale comments: "6 epsilon levels", `MW_TO_GW` naming note, ccp docstring "F^P" → conv(samples), traverse path wording (with #1) | `interior_sampling.py:111`, `poi.py:23`, `ccp_solver.py:4` | low | low | [safe-internal] |
| 16 | Pre-flight key/shape consistency check in `save()` | `preparation_state.py:398-409` | low | low | [safe-internal] |
| 17 | Derive interior-level seeds from caller's `seed` (changes sampled points — re-run preparation after) | `interior_sampling.py:75` | low | low | [safe-internal], invalidates cached `.nc` comparisons |
| 18 | `None`-default for `epsilon_levels` | `interior_sampling.py:43` | low | low | [safe-internal] |
| 19 | Consolidate the 4 duplicated hull helpers into one shared private module | `projection.py`, `alpha_projection.py` | low | med | [safe-internal] |
| 20 | De-underscore `_build_mga_model`/`_solve_mp` (shared API in practice) | `vertex_sampling.py`, `projection.py` | low | low | [safe-internal] |

\* #1's code change is engine-internal and signature-preserving, but it can change the *numbers* the notebook's traverse section displays — re-run that section after any fix.

**Explicit open questions (do not resolve without Sina / the paper):**
1. C1 — recursion geometry of Algorithm 2 (and whether the "(corrected)" condition at `traverse.py:71` matches the paper).
2. Should the subgradient set in Algorithm 3 include contributions from the path endpoints (`alpha_s`, `alpha_t`), or only breakpoint-active samples plus the constant `opt_cost` (current behavior)?
3. Sign convention of linopy duals in `Gamma` vs. the paper's γ (a wrong sign would flip the "lower bound" into nonsense; the test_traverse plot is currently the only check).
4. Is `sample_boundary_rays` still part of the thesis narrative (comparison of samplers) or superseded?
5. Should the notebook's `check_state` PASS also require matching sampling parameters (`n_vertices`, seeds), which the fingerprint deliberately excludes?
