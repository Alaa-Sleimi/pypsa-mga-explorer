"""
vertex_sampling.py — Vertex sampling via MP(r).

For each random direction r ∈ R^m we solve:

    MP(r): min  r'p  =  r'Dx
           s.t. x ∈ F^M
                (i.e. Ax ≤ b  AND  c'x ≤ (1 + ε) · c'x*)

Each solve gives one extreme point of F^P in the direction r.
We collect n_samples such points to build the matrix P_vertices (m × n_samples).
"""

import numpy as np
import pypsa
from typing import List
from mga_engine.poi import PoiSpec, evaluate_all


def _build_mga_model(
    network: pypsa.Network,
    opt_cost: float,
    epsilon: float,
) -> None:
    network.optimize.create_model(include_objective_constant=False)
    m = network.model
    mga_rhs = (1.0 + epsilon) * opt_cost
    m.add_constraints(
        m.objective.expression <= mga_rhs,
        name="mga_slack",
    )
    print(f"[vertex] Linopy model ready | MGA slack ≤ {mga_rhs:,.0f}  (ε={epsilon})")


def _solve_mp(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    direction: np.ndarray,
) -> np.ndarray:
    m = network.model

    new_obj = sum(
        float(direction[i]) * poi_specs[i].linopy_expr(m)
        for i in range(len(poi_specs))
    )
    m.objective = new_obj

    network.model.solve(solver_name="highs", output_flag=False)

    if network.model.status != "ok":
        raise RuntimeError(
            f"[vertex] MP(r) solve failed with status: {network.model.status}"
        )

    network.optimize.assign_solution()
    return evaluate_all(poi_specs, network)


def _solve_ray(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    l_var,
    p_start: np.ndarray,
    direction: np.ndarray,
):
    """
    Shoot one ray from p_start along `direction` and return the boundary point.

    Solves:
        max  l
        s.t. poi_i(x) == p_start[i] + l * direction[i]   for every PoI i
             x ∈ F^M                                       (mga_slack already on model)

    `l_var` is the scalar Linopy variable added once by the caller. The per-direction
    equality constraints are added here and removed before returning, so they do not
    accumulate across directions (reused-model pattern).

    Returns
    -------
    p : np.ndarray (m,) boundary point, or None if the solve was not "ok".
    """
    m = network.model
    ray_con_names = []

    # poi_i(x) - direction_i * l == p_start_i
    for i in range(len(poi_specs)):
        cname = f"ray_eq_{i}"
        m.add_constraints(
            poi_specs[i].linopy_expr(m) - float(direction[i]) * l_var
            == float(p_start[i]),
            name=cname,
        )
        ray_con_names.append(cname)

    # maximize l  ==  minimize -l  (model is default-minimize)
    m.objective = -l_var

    network.model.solve(solver_name="highs", output_flag=False)
    status = network.model.status

    p = None
    if status == "ok":
        network.optimize.assign_solution()
        p = evaluate_all(poi_specs, network)

    # remove the per-direction constraints so the next ray starts clean
    for cname in ray_con_names:
        m.remove_constraints(cname)

    return status, p


def sample_vertices(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    opt_cost: float,
    epsilon: float = 0.05,
    n_samples: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample n_samples vertices of F^P by solving MP(r) with random directions.

    Returns
    -------
    P_vertices : np.ndarray of shape (m, n_samples)
    """
    rng = np.random.default_rng(seed)
    m_dim = len(poi_specs)

    _build_mga_model(network, opt_cost, epsilon)

    P_vertices = np.zeros((m_dim, n_samples))

    print(f"[vertex] Sampling {n_samples} vertices ...")
    for k in range(n_samples):
        r = rng.standard_normal(m_dim)
        r /= np.linalg.norm(r)

        p = _solve_mp(network, poi_specs, r)
        P_vertices[:, k] = p
        print(f"  [{k+1}/{n_samples}]  p = {np.round(p, 3)}")

    return P_vertices


def sample_boundary_rays(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    opt_cost: float,
    p_star: np.ndarray,
    epsilon: float = 0.05,
    n_samples: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample boundary points of F^P by ray-shooting from p_start = p*.

    For each random unit direction r we solve, in BOTH the +r and -r senses,

        max  l   s.t.  poi_i(x) == p_start[i] + l * r[i]   for every PoI i,
                       x ∈ F^M

    Each solve returns one boundary point p_start + l*r. Shooting random rays
    from a fixed interior start spreads samples over the boundary surface,
    rather than clustering on the few dominant vertices that random objective
    directions (MP(r)) tend to return.

    n_samples counts TOTAL points. Since each direction yields two points (+r, -r),
    we shoot ceil(n_samples / 2) directions. With an odd n_samples this returns
    one extra point (round up), e.g. n_samples=11 -> 6 directions -> 12 points.

    Directions whose solve is infeasible/unbounded are skipped (logged), so the
    returned matrix may have fewer columns than 2 * n_directions.

    Parameters
    ----------
    network   : PyPSA network (will have a fresh MGA model built on it)
    poi_specs : list of PoiSpec
    opt_cost  : optimal total system cost c'x*
    p_star    : np.ndarray (m,) PoI vector at the optimal point (ray start)
    epsilon   : near-optimality slack
    n_samples : total number of boundary points to aim for
    seed      : RNG seed

    Returns
    -------
    P_rays : np.ndarray of shape (m, n_points_collected)
    """
    rng = np.random.default_rng(seed)
    m_dim = len(poi_specs)
    p_start = np.asarray(p_star, dtype=float)

    n_directions = -(-n_samples // 2)  # ceil division (round up)

    _build_mga_model(network, opt_cost, epsilon)

    # Add the scalar step-length variable ONCE; reused for every ray.
    l_var = network.model.add_variables(name="ray_l")

    collected = []
    n_failed = 0

    print(f"[ray] Shooting {n_directions} directions (±r) "
          f"-> up to {2 * n_directions} points ...")
    for k in range(n_directions):
        r = rng.standard_normal(m_dim)
        r /= np.linalg.norm(r)

        for sign in (+1.0, -1.0):
            direction = sign * r
            status, p = _solve_ray(network, poi_specs, l_var, p_start, direction)
            if status == "ok":
                collected.append(p)
                print(f"  [{len(collected)}]  sign={sign:+.0f}  p = {np.round(p, 3)}")
            else:
                n_failed += 1
                print(f"  [skip] direction {k} sign={sign:+.0f} -> status: {status}")

    if n_failed:
        print(f"[ray] {n_failed} ray solve(s) skipped (infeasible/unbounded).")

    if not collected:
        raise RuntimeError("[ray] No boundary points collected — all ray solves failed.")

    P_rays = np.column_stack(collected)
    print(f"[ray] Collected {P_rays.shape[1]} boundary points.")
    return P_rays


if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=".*experimental.*", category=UserWarning, module="linopy")

    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, evaluate_all

    # --- Baseline solve ---
    network = build_network()
    _, _ = network.optimize(
        solver_name="highs",
        include_objective_constant=False,
    )
    opt_cost = network.objective
    poi_specs = make_poi_specs(network)
    p_star = evaluate_all(poi_specs, network)
    print(f"Optimal cost : {opt_cost:,.0f} €/yr")
    print(f"p*           : {np.round(p_star, 3)}")

    # --- Vertex sampling (epsilon=0 here to verify all points land on boundary) ---
    network_mga = build_network()
    P_vertices = sample_vertices(
        network_mga, poi_specs, opt_cost,
        epsilon=0,
        n_samples=20,
        seed=42,
    )

    print("\nSampled vertices (columns of P_vertices):")
    print(np.round(P_vertices, 3))