"""
ccp_solver.py — Convex Combination Problem (CCP) solver.

Given a target point p in F^P, solve:

    CCP(p): min  α'v
            s.t. Pα = p       (m equality constraints)
                 α'1 = 1      (convex combination)
                 α ≥ 0

Returns the optimal weights α* and the cost approximation α*'v.
This is the real-time cost surrogate used in the exploration phase.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple


def solve_ccp(
    p: np.ndarray,
    P: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Solve CCP(p) for a single target point.

    Parameters
    ----------
    p : np.ndarray, shape (m,)
        Target point in F^P.
    P : np.ndarray, shape (m, n)
        Matrix of all sample points.
    v : np.ndarray, shape (n,)
        Minimum cost at each sample point from GP solves.

    Returns
    -------
    alpha : np.ndarray, shape (n,)
        Optimal convex combination weights.
    cost_approx : float
        Approximate minimum cost at p: alpha'v.
    """
    m, n = P.shape

    # Equality constraints: Pα = p and α'1 = 1
    # Stack into one system: A_eq α = b_eq
    A_eq = np.vstack([P, np.ones((1, n))])       # shape (m+1, n)
    b_eq = np.hstack([p, 1.0])                   # shape (m+1,)

    # All weights must be non-negative
    bounds = [(0.0, None)] * n

    result = linprog(
        c=v,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.status != 0:
        raise RuntimeError(
            f"CCP solve failed at p={np.round(p, 4)} — "
            f"status {result.status}: {result.message}"
        )

    alpha = result.x
    cost_approx = float(alpha @ v)

    return alpha, cost_approx


if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, evaluate_all
    from mga_engine.vertex_sampling import sample_vertices
    from mga_engine.interior_sampling import sample_interior
    from mga_engine.gp_solver import solve_all_gp

    # Solve OP
    network = build_network()
    network.optimize(solver_name="highs", include_objective_constant=False,
                     solver_options={"output_flag": False})
    opt_cost = network.objective
    poi_specs = make_poi_specs(network)
    p_star = evaluate_all(poi_specs, network)

    # Sample vertices and interior
    network_mga = build_network()
    poi_specs_mga = make_poi_specs(network_mga)
    P_vertices = sample_vertices(
        network_mga, poi_specs_mga, opt_cost,
        epsilon=0.05, n_samples=20, seed=42,
    )
    alpha = 0.1
    P_interior = sample_interior(P_vertices, n_samples=100, seed=0, alpha=alpha)
    P_all = np.hstack([P_vertices, P_interior])

    # GP phase
    v, Gamma = solve_all_gp(P_all)

    # Test CCP at p_star and at a few sample points
    print("\n[ccp] Testing CCP at p_star:")
    alpha_star, cost_star = solve_ccp(p_star, P_all, v)
    print(f"  p_star          = {np.round(p_star, 3)}")
    print(f"  cost approx     = {cost_star:,.0f}")
    print(f"  true opt_cost   = {opt_cost:,.0f}")
    print(f"  gap             = {cost_star - opt_cost:,.0f}")
    print(f"  nonzero weights = {(alpha_star > 1e-6).sum()}")



    p_target = np.array([1.3, 3.25])
    alpha_target, cost_target = solve_ccp(p_target, P_all, v)
    print(f"\n[ccp] Cost at p={p_target}:")
    print(f"  cost_approx = {cost_target:,.0f} €/yr")

    