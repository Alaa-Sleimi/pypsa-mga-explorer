"""
gp_solver.py — Goal Programming solver GP(p^i) for each sample point.

For each sample point p^i in P_all, solve:

    GP(p^i): min  c'x
             s.t. Ax ≤ b          (existing network constraints)
                  Dx = p^i        (dual: γ^i  — pins PoI values)

Outputs:
    v         : (n,)      minimum cost achievable at each p^i
    Gamma     : (m, n)    dual variables for the PoI equality constraints
    solutions : list of dicts, one per sample — full time-varying arrays
"""

import logging
import warnings
import numpy as np
import pypsa
from typing import List, Tuple
from mga_engine.network import build_network
from mga_engine.poi import PoiSpec, make_poi_specs


def _extract_solution(network) -> dict:
    """
    Generically extract decision variables from a solved PyPSA network.
    
    - Static (planning): only columns ending in '_opt' e.g. p_nom_opt, s_nom_opt
    - Dynamic (operational): everything PyPSA populated after solving e.g. Generator_p
    
    Returns a dict like:
        {"Generator_p_nom_opt": array(n_generators,),
         "Generator_p": array(n_snapshots, n_generators),
         "Line_s_nom_opt": array(n_lines,),
         "Line_p0": array(n_snapshots, n_lines), ...}
    """
    solution = {}

    for component in network.components:
        # --- planning variables: only optimised values ---
        for attr, series in component.static.items():
            if attr.endswith("_opt") and series.notna().any():
                key = f"{component.name}_{attr}"
                solution[key] = series.values.copy()

        # --- operational variables: all populated dynamic results ---
        for attr, df in component.dynamic.items():
            if not df.empty:
                key = f"{component.name}_{attr}"
                solution[key] = df.values.copy()

    return solution


def _solve_single_gp(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    p_i: np.ndarray,
) -> Tuple[float, np.ndarray, dict]:
    """
    Solve GP(p^i) for a single sample point.

    Returns
    -------
    v_i      : float  — minimum cost at p^i
    gamma_i  : (m,)   — duals for PoI equality constraints
    solution : dict   — full time-varying solution arrays
    """
    network.optimize.create_model(include_objective_constant=False)
    m = network.model

    for j, spec in enumerate(poi_specs):
        m.add_constraints(
            spec.linopy_expr(m) == float(p_i[j]),
            name=f"poi_fix_{j}",
        )

    m.solve(solver_name="highs", output_flag=False)

    if m.status != "ok":
        raise RuntimeError(f"GP solve failed at p={np.round(p_i, 4)}")

    network.optimize.assign_solution()

    v_i = network.model.solver_model.getObjectiveValue()

    gamma_i = np.array([
        float(m.dual[f"poi_fix_{j}"].values.flat[0])
        for j in range(len(poi_specs))
    ])

    solution = _extract_solution(network)

    return v_i, gamma_i, solution


def solve_all_gp(
    P_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Solve GP(p^i) for all n sample points.

    Returns
    -------
    v         : (n,)      minimum cost per sample
    Gamma     : (m, n)    dual variables per sample
    solutions : list of dicts, one per sample — full time-varying arrays
    """
    m_dim, n = P_all.shape

    network_gp = build_network()
    poi_specs_gp = make_poi_specs(network_gp)

    v         = np.zeros(n)
    Gamma     = np.zeros((m_dim, n))
    solutions = []

    print(f"[gp] Solving GP for {n} sample points ...")
    for i in range(n):
        v_i, gamma_i, sol = _solve_single_gp(network_gp, poi_specs_gp, P_all[:, i])
        v[i]        = v_i
        Gamma[:, i] = gamma_i
        solutions.append(sol)
        print(f"  [{i+1}/{n}]  v={v_i:,.0f}  γ={np.round(gamma_i, 3)}")

    print(f"[gp] Done. Cost range: {v.min():,.0f} — {v.max():,.0f}")
    return v, Gamma, solutions


if __name__ == "__main__":
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, evaluate_all
    from mga_engine.vertex_sampling import sample_vertices
    from mga_engine.interior_sampling import sample_interior, EPSILON_LEVELS, SAMPLES_PER_LEVEL

    # --- base solve ---
    network = build_network()
    network.optimize(solver_name="highs", include_objective_constant=False,
                     solver_options={"output_flag": False})
    opt_cost = network.objective
    poi_specs = make_poi_specs(network)
    p_star = evaluate_all(poi_specs, network)

    # --- vertices ---
    network_v = build_network()
    poi_specs_v = make_poi_specs(network_v)
    P_vertices = sample_vertices(
        network_v, poi_specs_v, opt_cost,
        epsilon=0.05, n_samples=20, seed=42,
    )

    # --- interior ---
    P_interior = sample_interior(
        build_network_fn=build_network,
        make_poi_specs_fn=make_poi_specs,
        opt_cost=opt_cost,
    )

    # --- combine ---
    P_all = np.hstack([P_vertices, P_interior])
    print(f"[main] P_all shape: {P_all.shape}  ({P_all.shape[1]} total sample points)")

    # --- GP phase ---
    v, Gamma, solutions = solve_all_gp(P_all)

    # --- sanity checks ---
    print(f"\n[main] v shape     : {v.shape}")
    print(f"[main] Gamma shape : {Gamma.shape}")
    print(f"[main] solutions   : {len(solutions)} dicts")
    print(f"[main] keys in first solution: {list(solutions[0].keys())}")
    print(f"[main] opt_cost    : {opt_cost:,.0f}")
    print(f"[main] min v       : {v.min():,.0f}  (should be >= opt_cost)")
    print(f"[main] max v       : {v.max():,.0f}  (should be <= (1+ε)*opt_cost)")