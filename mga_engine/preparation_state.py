"""
preparation_state.py — Save and load the full preparation phase output.

Saves everything needed for the exploration phase into a single .npz file:
    - P_all   : (m, n)     all sample points
    - v       : (n,)       minimum cost at each sample point
    - Gamma   : (m, n)     dual variables at each sample point
    - solutions: list of dicts  full time-varying arrays per sample

Usage:
    save("data/preparation_state.npz", P_all, v, Gamma, solutions)
    state = load("data/preparation_state.npz")
    state["P_all"], state["v"], state["Gamma"], state["solutions"]
"""

import os
import numpy as np


def save(path: str, P_all, v, Gamma, solutions):
    """
    Save preparation phase output to a .npz file.

    Parameters
    ----------
    path      : file path, e.g. "data/preparation_state.npz"
    P_all     : np.ndarray (m, n)
    v         : np.ndarray (n,)
    Gamma     : np.ndarray (m, n)
    solutions : list of dicts — one dict per sample, each dict maps
                key (str) -> np.ndarray of time-varying values
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Flatten solutions list into individual arrays with prefixed keys
    # e.g. solutions[3]["Generator_p"] -> "sol_3_Generator_p"
    flat = {}
    for i, sol in enumerate(solutions):
        for key, arr in sol.items():
            flat[f"sol_{i}_{key}"] = arr

    np.savez(
        path,
        P_all=P_all,
        v=v,
        Gamma=Gamma,
        n_samples=np.array(len(solutions)),
        **flat,
    )
    print(f"[state] Saved preparation state to {path}  ({len(solutions)} samples)")


def load(path: str) -> dict:
    """
    Load preparation phase output from a .npz file.

    Returns
    -------
    dict with keys: P_all, v, Gamma, solutions
    """
    data = np.load(path, allow_pickle=False)

    P_all = data["P_all"]
    v     = data["v"]
    Gamma = data["Gamma"]
    n     = int(data["n_samples"])

    # Reconstruct solutions list from flattened keys
    solutions = []
    for i in range(n):
        sol = {}
        prefix = f"sol_{i}_"
        for key in data.files:
            if key.startswith(prefix):
                sol[key[len(prefix):]] = data[key]
        solutions.append(sol)

    print(f"[state] Loaded preparation state from {path}  ({n} samples)")
    return {"P_all": P_all, "v": v, "Gamma": Gamma, "solutions": solutions}


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

    # --- base solve ---
    network = build_network()
    network.optimize(solver_name="highs", include_objective_constant=False,
                     solver_options={"output_flag": False})
    opt_cost = network.objective
    poi_specs = make_poi_specs(network)

    # --- vertices ---
    network_v = build_network()
    poi_specs_v = make_poi_specs(network_v)
    P_vertices = sample_vertices(
        network_v, poi_specs_v, opt_cost,
        epsilon=0.05, n_samples=50, seed=42,
    )

    # --- interior ---
    P_interior = sample_interior(
        build_network_fn=build_network,
        make_poi_specs_fn=make_poi_specs,
        opt_cost=opt_cost,
    )

    P_all = np.hstack([P_vertices, P_interior])

    # --- GP phase ---
    v, Gamma, solutions = solve_all_gp(P_all)

    # --- save ---
    save("data/preparation_state.npz", P_all, v, Gamma, solutions)

    # --- reload and verify ---
    state = load("data/preparation_state.npz")
    assert np.allclose(state["P_all"], P_all), "P_all mismatch!"
    assert np.allclose(state["v"], v),         "v mismatch!"
    assert np.allclose(state["Gamma"], Gamma), "Gamma mismatch!"
    assert len(state["solutions"]) == len(solutions), "solutions length mismatch!"
    print("[state] Verification passed — save/load is consistent.")
    print(f"[state] Keys in first loaded solution: {list(state['solutions'][0].keys())}")