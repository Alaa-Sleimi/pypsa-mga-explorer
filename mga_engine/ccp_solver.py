"""Convex Combination Problem (CCP): the real-time cost surrogate.

First step of the exploration phase. Given a target point p inside the convex hull of
the sampled points, it finds the cheapest way to write p as a convex combination of
those samples,

    CCP(p): min  a'v
            s.t. Pa = p       (m equality constraints)
                 a'1 = 1      (convex combination)
                 a >= 0

so navigation, traverse and the notebook can price a point in milliseconds, from the
stored samples alone, without touching the network.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple


def solve_ccp(
    p: np.ndarray,
    P: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Solve CCP(p) for a single target point.

    Parameters
    ----------
    p : numpy.ndarray
        Shape ``(m,)``. Target point in PoI space.
    P : numpy.ndarray
        Shape ``(m, n)``. Matrix of all sample points, one per column.
    v : numpy.ndarray
        Shape ``(n,)``. Minimum cost at each sample point, from the GP solves.

    Returns
    -------
    alpha : numpy.ndarray
        Shape ``(n,)``. Weights with ``P @ alpha == p``, sum 1, all non-negative.
    cost_approx : float
        Approximate minimum cost at `p`, equal to ``alpha @ v``.

    Raises
    ------
    RuntimeError
        If the LP is infeasible (SciPy status 2), i.e. `p` lies outside the convex
        hull of the samples and sampling must be made denser; or any other status.

    Notes
    -----
    One SciPy/HiGHS LP; deterministic; no network, no file access. Shapes are not
    validated. The cost comes from the stored samples alone.
    """
    m, n = P.shape

    # Equality constraints: Pα = p and α'1 = 1
    # Stack into one system: A_eq α = b_eq
    A_eq = np.vstack([P, np.ones((1, n))])       # shape (m+1, n)
    b_eq = np.hstack([p, 1.0])                   # shape (m+1,)

    # All weights must be non-negative
    bounds = [(0.0, None)] * n

    result = linprog(
        c=v,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.status == 2:
        # Infeasible: the equality system Pα = p, α'1 = 1, α ≥ 0 has no
        # solution, i.e. p lies outside the convex hull of the sample points.
        raise RuntimeError(
            f"CCP solve failed at p={np.round(p, 4)} — point lies outside the "
            f"sampled convex hull. Increase the sampling density "
            f"(n_vertices / n_samples_per_level) and re-run the preparation. "
            f"(solver status {result.status}: {result.message})"
        )
    if result.status != 0:
        raise RuntimeError(
            f"CCP solve failed at p={np.round(p, 4)} — "
            f"status {result.status}: {result.message}"
        )

    alpha = result.x
    cost_approx = float(alpha @ v)

    return alpha, cost_approx