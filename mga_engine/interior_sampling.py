"""
interior_sampling.py — Interior point sampling via random convex combinations.

Given the vertex matrix P_vertices (m × n_vertices), each interior sample is:

    p = P_vertices @ α,   α ~ Dirichlet(α_param,...,α_param),  α ≥ 0,  sum(α) = 1

Using α_param < 1 (e.g. 0.3) produces sparse weight vectors that concentrate
on a few vertices per draw, spreading samples near faces and edges rather than
collapsing everything to the centroid (which happens with α_param=1 and many vertices).
"""

import numpy as np


def sample_interior(
    P_vertices: np.ndarray,
    n_samples: int = 50,
    seed: int = 0,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Sample n_samples interior points of F^P as convex combinations of vertices.

    Parameters
    ----------
    P_vertices : np.ndarray, shape (m, n_vertices)
        Vertex matrix from sample_vertices().
    n_samples : int
        Number of interior points to generate.
    seed : int
        RNG seed for reproducibility.
    alpha : float
        Dirichlet concentration parameter. Values < 1 produce sparse weights
        (points near faces/edges). Value of 1 gives uniform simplex sampling
        but collapses to centroid when n_vertices is large.

    Returns
    -------
    P_interior : np.ndarray, shape (m, n_samples)
    """
    rng = np.random.default_rng(seed)
    n_vertices = P_vertices.shape[1]

    alphas = rng.dirichlet(np.ones(n_vertices) * alpha, size=n_samples)  # (n_samples, n_vertices)

    P_interior = P_vertices @ alphas.T  # (m, n_samples)

    print(f"[interior] Sampled {n_samples} interior points from {n_vertices} vertices (alpha={alpha}).")
    return P_interior


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, evaluate_all
    from mga_engine.vertex_sampling import sample_vertices

    # Solve OP
    network = build_network()
    network.optimize(solver_name="highs", include_objective_constant=False)
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
    P_interior = sample_interior(P_vertices, n_samples=100, seed=0, alpha=0.1)

# Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    hull = ConvexHull(P_vertices.T)
    for simplex in hull.simplices:
        ax.plot(P_vertices[0, simplex], P_vertices[1, simplex],
                "k-", linewidth=1.2, alpha=0.5)

    ax.scatter(P_vertices[0], P_vertices[1],
               color="steelblue", s=80, zorder=3, label="vertices")
    ax.scatter(P_interior[0], P_interior[1],
               color="orange", s=30, alpha=0.6, zorder=2, label="interior samples")
    ax.scatter(p_star[0], p_star[1],
               color="red", s=150, marker="*", zorder=4, label="optimal x*")

    ax.set_xlabel("solar_cap_gw")
    ax.set_ylabel("wind_cap_gw")
    ax.set_title(f"F^P — vertices + interior samples (ε=5%, α={0.1})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feasible_space_interior.png", dpi=150)
    plt.show()