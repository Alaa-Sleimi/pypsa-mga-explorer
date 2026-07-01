"""
test_ray_sampling.py — Standalone tests for sample_boundary_rays.

No pytest required. Run directly:

    python test_ray_sampling.py

Runs all checks, prints PASS/FAIL per check, and exits non-zero if any fail.

Setup (build network, solve, run the sampler) happens once in __main__ and is
passed into each check. If your project gets a network differently in tests
(e.g. loading data/preparation_state.nc), adjust the setup block in __main__;
the check functions stay the same.
"""

import sys
import numpy as np

from mga_engine.network import build_network
from mga_engine.poi import make_poi_specs, evaluate_all
from mga_engine.vertex_sampling import sample_boundary_rays
from mga_engine.gp_solver import solve_all_gp


EPSILON = 0.05
N_SAMPLES = 10
SEED = 42
COST_TOL = 1e-5  # relative tolerance for cost-slack assertions


# ─────────────────────────────────────────────────────────────────
# Individual checks. Each takes the shared setup dict + the sampler
# output, and raises AssertionError with a clear message on failure.
# ─────────────────────────────────────────────────────────────────

def check_shape(setup, P_rays):
    """Returns an (m, n_points) matrix; n_points up to 2*ceil(n_samples/2)."""
    m_dim = setup["m_dim"]
    n_directions = -(-N_SAMPLES // 2)  # ceil
    assert P_rays.ndim == 2, "P_rays must be 2D"
    assert P_rays.shape[0] == m_dim, (
        f"expected {m_dim} rows (PoIs), got {P_rays.shape[0]}"
    )
    assert P_rays.shape[1] >= 1, "no points collected"
    assert P_rays.shape[1] <= 2 * n_directions, (
        f"got {P_rays.shape[1]} points, but at most {2 * n_directions} are possible"
    )


def check_points_are_near_optimal(setup, P_rays):
    """
    Re-solve cost at every boundary point via GP(p) and assert each is genuinely
    near-optimal:

        opt_cost * (1 - tol)  <=  v_j  <=  (1 + epsilon) * opt_cost * (1 + tol)

    Lower bound: no feasible point can cost less than the optimum.
    Upper bound: every point must lie within the epsilon cost slack.

    We assert <= the slack bound, NOT == it. A ray from p* stops at the boundary
    of F^P, which may be a physical capacity limit reached BEFORE the cost slack
    binds; such points have v_j strictly below the slack bound and are correct.
    """
    opt_cost = setup["opt_cost"]
    eps = setup["epsilon"]

    v, _Gamma, _solutions = solve_all_gp(P_rays)

    upper = (1.0 + eps) * opt_cost * (1.0 + COST_TOL)
    lower = opt_cost * (1.0 - COST_TOL)

    assert np.all(v <= upper), (
        f"some boundary points exceed the cost slack: "
        f"max v = {v.max():,.2f} > allowed {upper:,.2f}"
    )
    assert np.all(v >= lower), (
        f"some boundary points cost less than the optimum: "
        f"min v = {v.min():,.2f} < allowed {lower:,.2f}"
    )


def check_at_least_one_point_on_the_cost_boundary(setup, P_rays):
    """
    At least one ray should exit through the cost face — i.e. some boundary point
    should reach the slack bound (1+epsilon)*opt_cost within tolerance. Confirms
    the rays actually probe the near-optimal boundary rather than all stopping
    early on unrelated physical limits.
    """
    opt_cost = setup["opt_cost"]
    eps = setup["epsilon"]

    v, _Gamma, _solutions = solve_all_gp(P_rays)

    slack_bound = (1.0 + eps) * opt_cost

    if eps == 0:
        # Degenerate: with no slack, every near-optimal point sits at the cost
        # optimum, so "on the cost face" is trivially the whole set.
        assert np.all(v <= slack_bound * (1.0 + COST_TOL))
    else:
        on_cost_face = np.isclose(v, slack_bound, rtol=COST_TOL)
        assert on_cost_face.any(), (
            "no boundary point reached the cost slack bound — "
            "rays may not be probing the cost face as expected"
        )


def check_points_differ_from_p_star(setup, P_rays):
    """At least some boundary points should move away from p_star (rays travel)."""
    p_star = setup["p_star"]
    dists = np.linalg.norm(P_rays - p_star[:, None], axis=0)
    assert np.any(dists > 1e-6), "all boundary points coincide with p_star"


# ─────────────────────────────────────────────────────────────────
# Runner: run all checks, print PASS/FAIL per check, exit non-zero
# if any failed.
# ─────────────────────────────────────────────────────────────────

CHECKS = [
    ("shape", check_shape),
    ("points_are_near_optimal", check_points_are_near_optimal),
    ("at_least_one_point_on_the_cost_boundary",
     check_at_least_one_point_on_the_cost_boundary),
    ("points_differ_from_p_star", check_points_differ_from_p_star),
]


def run_all_checks(setup, P_rays):
    print("\n" + "=" * 60)
    print("Running ray-sampling checks")
    print("=" * 60)

    n_failed = 0
    for name, fn in CHECKS:
        try:
            fn(setup, P_rays)
            print(f"  PASS  {name}")
        except AssertionError as e:
            n_failed += 1
            print(f"  FAIL  {name}")
            print(f"        {e}")
        except Exception as e:  # unexpected error (not an assertion)
            n_failed += 1
            print(f"  ERROR {name}")
            print(f"        {type(e).__name__}: {e}")

    print("-" * 60)
    total = len(CHECKS)
    passed = total - n_failed
    print(f"  {passed}/{total} checks passed")
    print("=" * 60)
    return n_failed


if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    # --- setup (runs once) ---
    network = build_network()
    network.optimize(
        solver_name="highs",
        include_objective_constant=False,
        solver_options={"output_flag": False},
    )
    opt_cost = float(network.objective)
    poi_specs = make_poi_specs(network)
    p_star = evaluate_all(poi_specs, network)

    setup = {
        "opt_cost": opt_cost,
        "poi_specs": poi_specs,
        "p_star": p_star,
        "m_dim": len(poi_specs),
        "epsilon": EPSILON,
    }

    # --- run the sampler once ---
    network_rays = build_network()
    poi_specs_rays = make_poi_specs(network_rays)
    P_rays = sample_boundary_rays(
        network_rays,
        poi_specs_rays,
        opt_cost,
        p_star,
        epsilon=EPSILON,
        n_samples=N_SAMPLES,
        seed=SEED,
    )

    # --- run all checks and exit accordingly ---
    n_failed = run_all_checks(setup, P_rays)
    sys.exit(1 if n_failed else 0)
