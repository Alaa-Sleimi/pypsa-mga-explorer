"""Goal Programming solves GP(p^i) that price every sampled point.

Third step of the preparation phase: after the sampling modules have collected PoI
points, each point is priced by re-solving the network with its PoI values pinned,

    GP(p^i): min  c'x
             s.t. Ax <= b          (existing network constraints)
                  Dx = p^i         (dual: gamma^i — pins the PoI values)

which yields the cost ``v``, the PoI duals ``Gamma`` and the full solution per sample
that :mod:`mga_engine.preparation_state` stores and the exploration phase reads.
"""

import numpy as np
import pypsa
from typing import List, Tuple
from mga_engine.poi import PoiSpec


def _extract_solution(network) -> dict:
    """Collect a solved network's variable frames into flat NumPy arrays.

    Parameters
    ----------
    network : pypsa.Network
        Network that has been solved and had its solution assigned.

    Returns
    -------
    dict
        Maps ``"<Component>_<attribute>"`` to a copy of that frame's values, e.g.
        ``"Generator_p_nom_opt"`` of shape ``(n_generators,)``, ``"Generator_p"`` of
        shape ``(n_snapshots, n_generators)``, ``"Line_s_nom_opt"``, ``"Line_p0"``.

    Notes
    -----
    Static frames contribute only columns ending in ``_opt`` that hold at least one
    non-NaN value. Dynamic frames contribute EVERY non-empty time-varying frame, so
    the dict also carries inputs such as ``Generator_p_max_pu`` and ``Load_p_set``,
    not just results; those inputs are identical across samples but are stored per
    sample by :func:`mga_engine.preparation_state.save`, which inflates the file.
    Values are copied, so the result never aliases the network's frames.
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
    """Solve GP(p^i) for one sample point, with every PoI pinned to ``p_i``.

    Parameters
    ----------
    network : pypsa.Network
        Network to build the model on. A fresh linopy model is created here, so one
        network object can serve every sample point.
    poi_specs : list of PoiSpec
        PoI specs, in the same order as the entries of `p_i`.
    p_i : numpy.ndarray
        Shape ``(m,)``. Target value of each PoI, in that PoI's unit (GW).

    Returns
    -------
    v_i : float
        Minimum objective value attainable at `p_i`.
    gamma_i : numpy.ndarray
        Shape ``(m,)``. Dual of each ``poi_fix_<j>`` equality constraint.
    solution : dict
        Full solution arrays, as returned by :func:`_extract_solution`.

    Raises
    ------
    RuntimeError
        If the solver status is not ``"ok"``.

    Notes
    -----
    Calls HiGHS. Mutates `network`: replaces ``network.model`` and writes the
    solution into its result columns. The objective excludes PyPSA's objective
    constant, matching the cost the rest of the pipeline compares against.

    Sign convention: ``gamma_i`` is un-negated, i.e. d(cost)/d(p_i) in EUR/yr per GW,
    so a positive value means that raising that PoI's target raises the minimum
    system cost and a negative value means it lowers it. Verified for the pinned
    linopy/highspy versions on the HiGHS minimisation path.
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

    v_i = float(m.objective.value)

    gamma_i = np.array([
        float(m.dual[f"poi_fix_{j}"].values.flat[0])
        for j in range(len(poi_specs))
    ])

    solution = _extract_solution(network)

    return v_i, gamma_i, solution


def solve_all_gp(
    P_all: np.ndarray,
    build_network_fn,
    make_poi_specs_fn,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Solve GP(p^i) for every sampled point and return costs, duals and solutions.

    Parameters
    ----------
    P_all : numpy.ndarray
        Shape ``(m, n)``. Sample matrix; each column is one PoI target point.
    build_network_fn : callable
        ``build_network_fn() -> pypsa.Network``. Called once; every sample is then
        solved on that one network, with a fresh model per sample.
    make_poi_specs_fn : callable
        ``make_poi_specs_fn(network) -> list of PoiSpec``, in `P_all` row order.

    Returns
    -------
    v : numpy.ndarray
        Shape ``(n,)``. Minimum cost at each sample.
    Gamma : numpy.ndarray
        Shape ``(m, n)``. Column ``i`` holds the PoI duals at sample ``i``.
    solutions : list of dict
        One :func:`_extract_solution` dict per sample, in column order.

    Raises
    ------
    RuntimeError
        Propagated from the first failed GP solve; no partial result is returned.

    Notes
    -----
    Both callables are required; there is no default network or PoI definition.
    Calls HiGHS once per sample and prints a line per sample plus a cost range.
    """
    m_dim, n = P_all.shape

    network_gp = build_network_fn()
    poi_specs_gp = make_poi_specs_fn(network_gp)

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