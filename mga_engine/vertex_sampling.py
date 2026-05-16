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
    network.optimize.assign_solution()

    return evaluate_all(poi_specs, network)


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


if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, evaluate_all

    network = build_network()
    status, _ = network.optimize(
        solver_name="highs",
        include_objective_constant=False,
    )
    opt_cost = network.objective
    poi_specs = make_poi_specs(network)
    p_star = evaluate_all(poi_specs, network)

    network_mga = build_network()
    poi_specs_mga = make_poi_specs(network_mga)
    P_vertices = sample_vertices(
        network_mga, poi_specs_mga, opt_cost,
        epsilon=0,
        n_samples=20,
        seed=42,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    points = P_vertices.T
    hull = ConvexHull(points)

    ax.scatter(points[:, 0], points[:, 1],
               color="steelblue", s=80, zorder=3, label="vertices")
    ax.scatter(p_star[0], p_star[1],
               color="red", s=150, marker="*", zorder=4, label="optimal x*")

    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1],
                "k-", linewidth=1.2, alpha=0.6)

    for idx, (x, y) in enumerate(points):
        ax.annotate(str(idx), (x, y),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("solar_cap_gw")
    ax.set_ylabel("wind_cap_gw")
    ax.set_title("Near-optimal feasible space F^P (ε=5%)\n2 PoIs — all vertices should be on boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feasible_space_2poi.png", dpi=150)
    plt.show()