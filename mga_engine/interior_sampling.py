"""Interior sampling of the near-optimal PoI space via MP(r) at tightening epsilon.

Runs the MP(r) sampling of :mod:`mga_engine.vertex_sampling` once per cost-slack
level, each level tighter than the last: a high epsilon puts points near the outer
boundary of F^P, a low one keeps them close to the cost optimum x*. Together the
levels cover the interior radially instead of only its outer hull, which is what the
CCP surrogate needs to price points away from the boundary.
"""

import numpy as np
from mga_engine.vertex_sampling import sample_vertices


# Kept only as documented defaults for sample_interior's signature. The actual
# levels are derived from user parameters via derive_epsilon_levels(); these
# equal derive_epsilon_levels(epsilon=0.05, n_levels=5, epsilon_min=0.005).
EPSILON_LEVELS = [0.04, 0.03, 0.02, 0.01, 0.005]
SAMPLES_PER_LEVEL = 10


def derive_epsilon_levels(epsilon: float, n_levels: int, epsilon_min: float) -> list:
    """Derive the interior-sampling epsilon levels from the outer epsilon.

    Levels descend linearly from `epsilon` in steps of ``epsilon / n_levels``,
    starting one step below `epsilon` so that the outer level already covered by the
    vertex sampling is not repeated, and the last level is replaced by `epsilon_min`.
    With ``epsilon=0.05, n_levels=5, epsilon_min=0.005`` this reproduces
    ``EPSILON_LEVELS``, the sequence formerly hardcoded here.

    Parameters
    ----------
    epsilon : float
        Outer relative cost slack. Every level stays strictly below it.
    n_levels : int
        Number of levels to return. Must be at least 1.
    epsilon_min : float
        Tightest level, returned as the last entry. Must satisfy
        ``0 < epsilon_min < epsilon``.

    Returns
    -------
    list of float
        Length `n_levels`, rounded to 12 decimals so the values print cleanly.

    Raises
    ------
    AssertionError
        If `n_levels` is below 1, or `epsilon_min` is outside ``(0, epsilon)``.
        These are plain ``assert`` statements, so they vanish under ``python -O``.

    Notes
    -----
    The result is only strictly decreasing when ``epsilon_min < epsilon / n_levels``.
    ``derive_epsilon_levels(0.05, 5, 0.02)`` gives ``[0.04, 0.03, 0.02, 0.01, 0.02]``,
    whose last level is looser than the one before it. Nothing here or in the callers
    rejects that.
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
    """Sample interior points by running MP(r) once per descending epsilon level.

    Parameters
    ----------
    build_network_fn : callable
        ``build_network_fn() -> pypsa.Network``. Called once per level; each level
        needs its own unsolved network because its cost slack differs.
    make_poi_specs_fn : callable
        ``make_poi_specs_fn(network) -> list of PoiSpec``. Must return the same PoIs
        in the same order at every level, since the columns are stacked into one
        matrix.
    opt_cost : float
        Minimum system cost c'x* in EUR/yr from the base solve; see
        :func:`mga_engine.vertex_sampling._build_mga_model`.
    epsilon_levels : list of float, optional
        Relative cost slacks to sweep, normally from :func:`derive_epsilon_levels`.
        ``None`` falls back to the module constant ``EPSILON_LEVELS``.
    samples_per_level : int, default 10
        MP(r) solves per level, i.e. columns contributed by each level.
    seed : int, default 42
        Caller seed. One independent child seed per level is derived from it with
        ``numpy.random.SeedSequence(seed).spawn()``, so the same `seed` reproduces
        every level's directions and a different `seed` changes all of them at once.

    Returns
    -------
    numpy.ndarray
        Shape ``(m, len(epsilon_levels) * samples_per_level)``. The levels' matrices
        side by side, in level order, with PoI values in GW.

    Raises
    ------
    RuntimeError
        Propagated from the first failed MP(r) solve.

    Notes
    -----
    Builds one network and calls HiGHS `samples_per_level` times per level, and
    prints a line per level plus a total. Nothing checks that the levels actually
    descend or stay below the outer epsilon.
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