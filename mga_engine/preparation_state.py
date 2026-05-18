"""
preparation_state.py — Save and load the full preparation phase output.

Saves everything needed for the exploration phase into a single .nc file
using xarray. The file contains:
    - P_all   : (poi, sample)            all sample points
    - v       : (sample,)                minimum cost at each sample point
    - Gamma   : (poi, sample)            dual variables at each sample point
    - solution variables stacked across samples:
        static  e.g. Generator_p_nom_opt : (sample, Generator_p_nom_opt_idx)
        dynamic e.g. Generator_p         : (sample, Generator_p_snapshot, Generator_p_idx)

Usage:
    save("data/preparation_state.nc", P_all, v, Gamma, solutions)
    state = load("data/preparation_state.nc")
    state["P_all"], state["v"], state["Gamma"], state["solutions"]
"""

import os
import numpy as np
import xarray as xr


def save(path: str, P_all, v, Gamma, solutions):
    """
    Save preparation phase output to a single .nc file.

    Parameters
    ----------
    path      : file path, e.g. "data/preparation_state.nc"
    P_all     : np.ndarray (m, n)
    v         : np.ndarray (n,)
    Gamma     : np.ndarray (m, n)
    solutions : list of dicts — one dict per sample
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    n_samples = len(solutions)

    # --- core arrays ---
    ds = xr.Dataset({
        "P_all": xr.DataArray(P_all, dims=["poi", "sample"]),
        "v":     xr.DataArray(v,     dims=["sample"]),
        "Gamma": xr.DataArray(Gamma, dims=["poi", "sample"]),
    })

    # --- solution arrays stacked across samples ---
    for key in solutions[0].keys():
        arr = np.stack([sol[key] for sol in solutions], axis=0)
        if arr.ndim == 2:   # static: (n_samples, n_components)
            dims = ["sample", f"{key}_idx"]
        elif arr.ndim == 3: # dynamic: (n_samples, n_snapshots, n_components)
            dims = ["sample", f"{key}_snapshot", f"{key}_idx"]
        else:
            dims = ["sample"] + [f"{key}_dim{k}" for k in range(arr.ndim - 1)]
        ds[key] = xr.DataArray(arr, dims=dims)

    ds.to_netcdf(path, mode="w")
    ds.close()
    print(f"[state] Saved preparation state to {path}  ({n_samples} samples)")


def load(path: str) -> dict:
    """
    Load preparation phase output from a .nc file.

    Returns
    -------
    dict with keys: P_all, v, Gamma, solutions
    """
    with xr.open_dataset(path) as ds:
        P_all = ds["P_all"].values
        v     = ds["v"].values
        Gamma = ds["Gamma"].values
        n     = len(ds["sample"])

        solution_keys = [k for k in ds.data_vars if k not in ("P_all", "v", "Gamma")]
        solutions = []
        for i in range(n):
            sol = {key: ds[key].values[i] for key in solution_keys}
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
        epsilon=0.05, n_samples=30, seed=42,
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
    save("data/preparation_state.nc", P_all, v, Gamma, solutions)

    # --- reload and verify ---
    state = load("data/preparation_state.nc")
    assert np.allclose(state["P_all"], P_all),            "P_all mismatch!"
    assert np.allclose(state["v"], v),                    "v mismatch!"
    assert np.allclose(state["Gamma"], Gamma),            "Gamma mismatch!"
    assert len(state["solutions"]) == len(solutions),     "solutions length mismatch!"
    print("[state] Verification passed — save/load is consistent.")
    print(f"[state] Keys in first loaded solution: {list(state['solutions'][0].keys())}")