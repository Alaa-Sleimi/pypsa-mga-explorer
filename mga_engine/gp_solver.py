"""
gp_solver.py — Goal Programming solver GP(p^i) for each sample point.

For each sample point p^i in P_all, solve:

    GP(p^i): min  c'x
             s.t. Ax ≤ b          (existing network constraints)
                  Dx = p^i        (dual: γ^i  — pins PoI values)

Outputs:
    v     : (n,)    minimum cost achievable at each p^i
    Gamma : (m, n)  dual variables for the PoI equality constraints
"""

import logging
import warnings
import numpy as np
import pypsa
from typing import List, Tuple
from mga_engine.network import build_network
from mga_engine.poi import PoiSpec, make_poi_specs


def _solve_single_gp(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    p_i: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Solve GP(p^i) for a single sample point.

    Returns
    -------
    v_i     : float           — minimum cost at p^i
    gamma_i : np.ndarray (m,) — duals for the PoI equality constraints
    """
    network.optimize.create_model(include_objective_constant=False)
    m = network.model

    for j, spec in enumerate(poi_specs):
        m.add_constraints(
            spec.linopy_expr(m) == float(p_i[j]),
            name=f"poi_fix_{j}",
        )

    m.solve(solver_name="highs", output_flag=False)

    status = m.status
    if status != "ok":
        raise RuntimeError(
            f"GP solve failed at p={np.round(p_i, 4)} — status: {status}"
        )

    v_i = network.model.solver_model.getObjectiveValue()

    gamma_i = np.array([
        float(m.dual[f"poi_fix_{j}"].values.flat[0])
        for j in range(len(poi_specs))
    ])

    return v_i, gamma_i


def solve_all_gp(
    P_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve GP(p^i) for all n sample points in P_all.

    Parameters
    ----------
    P_all : np.ndarray, shape (m, n)

    Returns
    -------
    v     : np.ndarray, shape (n,)
    Gamma : np.ndarray, shape (m, n)
    """
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    m_dim, n = P_all.shape

    network_gp = build_network()
    poi_specs_gp = make_poi_specs(network_gp)

    v     = np.zeros(n)
    Gamma = np.zeros((m_dim, n))

    print(f"[gp] Solving GP for {n} sample points ...")
    for i in range(n):
        v_i, gamma_i = _solve_single_gp(network_gp, poi_specs_gp, P_all[:, i])
        v[i]         = v_i
        Gamma[:, i]  = gamma_i
        print(f"  [{i+1}/{n}]  v={v_i:,.0f}  γ={np.round(gamma_i, 3)}")

    print(f"[gp] Done. Cost range: {v.min():,.0f} — {v.max():,.0f}")
    return v, Gamma


if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, evaluate_all
    from mga_engine.vertex_sampling import sample_vertices
    from mga_engine.interior_sampling import sample_interior

    # Solve OP
    network = build_network()
    network.optimize(solver_name="highs", include_objective_constant=False,
                 solver_options={"output_flag": False})
    opt_cost = network.objective
    poi_specs = make_poi_specs(network)
    p_star = evaluate_all(poi_specs, network)

    # Sample vertices
    network_mga = build_network()
    poi_specs_mga = make_poi_specs(network_mga)
    P_vertices = sample_vertices(
        network_mga, poi_specs_mga, opt_cost,
        epsilon=0.05, n_samples=20, seed=42,
    )

    # Sample interior
    alpha = 0.1
    P_interior = sample_interior(P_vertices, n_samples=100, seed=0, alpha=alpha)

    # Combine
    P_all = np.hstack([P_vertices, P_interior])
    print(f"[main] P_all shape: {P_all.shape}  ({P_all.shape[1]} total sample points)")

    # Solve GP for all points
    v, Gamma = solve_all_gp(P_all)


    print (P_all[:, :5])  # print first 5 sample points
    print(f"\n[main] v     shape: {v.shape}")
    print(f"[main] Gamma shape: {Gamma.shape}")
    print(f"[main] opt_cost = {opt_cost:,.0f}")
    print(f"[main] min v   = {v.min():,.0f}  (should be >= opt_cost)")
    print(f"[main] max v   = {v.max():,.0f}  (should be <= (1+ε)*opt_cost)")