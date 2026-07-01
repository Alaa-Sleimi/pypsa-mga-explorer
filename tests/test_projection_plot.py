"""
test_projection_plot.py — Standalone visual test for projection.py.

No pytest. Run directly from the project root:

    python -m tests.test_projection_plot
        or
    python tests/test_projection_plot.py

It builds the network, computes the 2D near-optimal hull for PoIs (0, 1) via
project_hull, samples points with BOTH samplers (sample_vertices and
sample_boundary_rays), and plots:

    - the hull outline (true projected boundary)
    - vertex-sampler points   (one colour)
    - ray-sampler points      (another colour)
    - p* (cost-optimal point) marked

The figure is saved to projection_test.png next to this script AND shown in a
window. The plot doubles as a correctness check: every sampled point should lie
ON or INSIDE the hull. Points clearly outside the hull would indicate a bug
(e.g. in the hull orientation or an epsilon mismatch).
"""

import os
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

from mga_engine.network import build_network
from mga_engine.poi import make_poi_specs, evaluate_all
from mga_engine.vertex_sampling import sample_vertices, sample_boundary_rays
from mga_engine.projection import project_hull


# ── Shared settings (one epsilon everywhere, as agreed) ──────────
EPSILON = 0.05
N_SAMPLES = 20
SEED = 42
I, J = 0, 1  # PoI indices to project onto (solar, wind)

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "projection_test.png")


def _close_polygon(verts: np.ndarray) -> np.ndarray:
    """Append the first vertex to the end so the outline draws closed."""
    return np.vstack([verts, verts[0]])


def main():
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    # --- base solve: opt_cost, poi_specs, p* ---
    network = build_network()
    network.optimize(
        solver_name="highs",
        include_objective_constant=False,
        solver_options={"output_flag": False},
    )
    opt_cost = float(network.objective)
    poi_specs = make_poi_specs(network)
    p_star = evaluate_all(poi_specs, network)
    name_i, name_j = poi_specs[I].name, poi_specs[J].name
    print(f"opt_cost = {opt_cost:,.0f}")
    print(f"p*       = {np.round(p_star, 3)}")

    # --- 2D hull (its own fresh network) ---
    net_hull = build_network()
    specs_hull = make_poi_specs(net_hull)
    hull = project_hull(
        net_hull, specs_hull, opt_cost,
        epsilon=EPSILON, i=I, j=J,
    )

    # --- vertex sampler (its own fresh network) ---
    net_v = build_network()
    specs_v = make_poi_specs(net_v)
    P_vertices = sample_vertices(
        net_v, specs_v, opt_cost,
        epsilon=EPSILON, n_samples=N_SAMPLES, seed=SEED,
    )

    # --- ray sampler (its own fresh network) ---
    net_r = build_network()
    specs_r = make_poi_specs(net_r)
    P_rays = sample_boundary_rays(
        net_r, specs_r, opt_cost, p_star,
        epsilon=EPSILON, n_samples=N_SAMPLES, seed=SEED,
    )

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # hull outline
    closed = _close_polygon(hull)
    ax.plot(closed[:, 0], closed[:, 1],
            color="#333333", lw=2, label="Near-optimal hull", zorder=2)
    ax.fill(closed[:, 0], closed[:, 1],
            color="#333333", alpha=0.05, zorder=1)

    # vertex-sampler points (row I = x, row J = y)
    ax.scatter(P_vertices[I, :], P_vertices[J, :],
               s=45, color="#4C8EDA", edgecolor="white", linewidth=0.5,
               label=f"sample_vertices (n={P_vertices.shape[1]})", zorder=3)

    # ray-sampler points
    ax.scatter(P_rays[I, :], P_rays[J, :],
               s=45, color="#E07B54", edgecolor="white", linewidth=0.5,
               marker="^", label=f"sample_boundary_rays (n={P_rays.shape[1]})",
               zorder=3)

    # p* marker
    ax.scatter([p_star[I]], [p_star[J]],
               s=180, color="#2E7D32", marker="*", edgecolor="white",
               linewidth=0.8, label="p* (cost-optimal)", zorder=4)

    ax.set_xlabel(name_i, fontsize=11)
    ax.set_ylabel(name_j, fontsize=11)
    ax.set_title(f"Near-optimal projection onto ({name_i}, {name_j})  —  ε={EPSILON}",
                 fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    fig.savefig(PNG_PATH, dpi=150)
    print(f"[plot] Saved figure to {PNG_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
