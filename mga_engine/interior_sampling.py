"""
interior_sampling.py — Interior point sampling via MP(r) across decreasing epsilon levels.

We re-run MP(r) at multiple epsilon levels, each tighter than the last.
Higher epsilon → points near the outer boundary of F^P.
Lower epsilon  → points closer to x*, deep in the interior.
Together they give good radial coverage of the whole space.
"""

import numpy as np
from mga_engine.vertex_sampling import sample_vertices


# Kept only as documented defaults for sample_interior's signature. The actual
# levels are derived from user parameters via derive_epsilon_levels(); these
# equal derive_epsilon_levels(epsilon=0.05, n_levels=5, epsilon_min=0.005).
EPSILON_LEVELS = [0.04, 0.03, 0.02, 0.01, 0.005]
SAMPLES_PER_LEVEL = 10


def derive_epsilon_levels(epsilon: float, n_levels: int, epsilon_min: float) -> list:
    """Derive the interior sampling epsilon levels from the outer epsilon.

    Levels descend linearly from epsilon in steps of epsilon / n_levels — the
    top level is the first step below epsilon (strictly below it) — and the
    last level is epsilon_min. With epsilon=0.05, n_levels=5, epsilon_min=0.005
    this reproduces the sequence formerly hardcoded here:
    [0.04, 0.03, 0.02, 0.01, 0.005].
    """
    assert n_levels >= 1, f"n_levels must be >= 1, got {n_levels}"
    assert 0 < epsilon_min < epsilon, (
        f"epsilon_min must be > 0 and < epsilon ({epsilon}), got {epsilon_min}")
    step = epsilon / n_levels
    # round() strips float artifacts (0.04000000000000001 -> 0.04) so the
    # levels print cleanly; numerically irrelevant at 12 decimals.
    return [round(epsilon - k * step, 12) for k in range(1, n_levels)] + [epsilon_min]


def sample_interior(
    build_network_fn,
    make_poi_specs_fn,
    opt_cost: float,
    epsilon_levels: list | None = None,
    samples_per_level: int = SAMPLES_PER_LEVEL,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample interior points by running MP(r) at decreasing epsilon levels.

    Parameters
    ----------
    build_network_fn   : callable that returns a fresh PyPSA network
    make_poi_specs_fn  : callable that takes a network and returns poi_specs
    opt_cost           : c'x* from the base solve
    epsilon_levels     : list of epsilon values to sweep through (typically
                         from derive_epsilon_levels)
    samples_per_level  : number of MP(r) solves per epsilon level
    seed               : caller seed; one independent child seed per epsilon
                         level is derived from it via SeedSequence.spawn()

    Returns
    -------
    P_interior : np.ndarray, shape (m, total_samples)
    """
    if epsilon_levels is None:
        epsilon_levels = EPSILON_LEVELS

    # One independent child seed per epsilon level, derived from the caller's
    # seed: same seed -> identical samples every run, different seed ->
    # genuinely different samples at every level.
    level_seeds = [
        int(child.generate_state(1)[0])
        for child in np.random.SeedSequence(seed).spawn(len(epsilon_levels))
    ]

    all_columns = []

    for i, eps in enumerate(epsilon_levels):
        print(f"[interior] epsilon={eps} — sampling {samples_per_level} points ...")
        network = build_network_fn()
        poi_specs = make_poi_specs_fn(network)

        P_level = sample_vertices(
            network=network,
            poi_specs=poi_specs,
            opt_cost=opt_cost,
            epsilon=eps,
            n_samples=samples_per_level,
            seed=level_seeds[i],   # per-level child seed derived from `seed`
        )
        all_columns.append(P_level)

    P_interior = np.hstack(all_columns)
    print(f"[interior] Done — {P_interior.shape[1]} interior points total.")
    return P_interior


if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, evaluate_all
    from mga_engine.vertex_sampling import sample_vertices

    # --- base solve ---
    network = build_network()
    network.optimize(solver_name="highs", include_objective_constant=False)
    opt_cost = network.objective
    poi_specs = make_poi_specs(network)
    p_star = evaluate_all(poi_specs, network)

    # --- vertices (epsilon=0.05, fresh network) ---
    network_v = build_network()
    poi_specs_v = make_poi_specs(network_v)
    P_vertices = sample_vertices(
        network_v, poi_specs_v, opt_cost,
        epsilon=0.05, n_samples=50, seed=42,
    )

    # --- interior (5 epsilon levels, fresh network per level) ---
    P_interior = sample_interior(
        build_network_fn=build_network,
        make_poi_specs_fn=make_poi_specs,
        opt_cost=opt_cost,
    )

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    hull = ConvexHull(P_vertices.T)
    for simplex in hull.simplices:
        ax.plot(P_vertices[0, simplex], P_vertices[1, simplex],
                "k-", linewidth=1.2, alpha=0.5)

    ax.scatter(P_vertices[0], P_vertices[1],
               color="steelblue", s=80, zorder=3, label="vertices (ε=5%)")

    # colour interior points by epsilon level so we can see the layering
    colors = ["#f4a460", "#e07b39", "#c85a1e", "#a03010", "#781800", "#500000"]
    for i, eps in enumerate(EPSILON_LEVELS):
        start = i * SAMPLES_PER_LEVEL
        end = start + SAMPLES_PER_LEVEL
        ax.scatter(P_interior[0, start:end], P_interior[1, start:end],
                   color=colors[i], s=30, alpha=0.7, zorder=2, label=f"ε={eps}")

    ax.scatter(p_star[0], p_star[1],
               color="red", s=150, marker="*", zorder=4, label="optimal x*")

    ax.set_xlabel("solar_cap_gw")
    ax.set_ylabel("wind_cap_gw")
    ax.set_title("F^P — vertices + interior samples across epsilon levels")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feasible_space_interior.png", dpi=150)
    plt.show()