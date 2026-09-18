"""2-D projection of the SAMPLED near-optimal space, in weight space.

Exploration-phase plotting support. Where :mod:`mga_engine.projection` solves MP(r) on
a live network and returns the true LP boundary, this module stays in alpha space:
the region is the convex hull of the stored samples, which is what
:mod:`mga_engine.navigation` actually searches, optionally intersected with the
per-APoI interval bounds navigate() reports. Every direction solve is one small SciPy
LP over the weights, so a plot costs milliseconds and needs no PyPSA or linopy model.

APoI index convention, as in :mod:`mga_engine.navigation`: index 0 is the cost (row 0
of P_tilde is v), indices 1..m are the PoI values.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError
from typing import List, Optional, Tuple


def _bounds_matrix(P_tilde: np.ndarray, bounds) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Turn ``(apoi_index, lower, upper)`` bounds into linprog ``A_ub`` / ``b_ub`` rows.

    Parameters
    ----------
    P_tilde : numpy.ndarray
        Shape ``(m + 1, n)``. APoI matrix ``[v'; P_all]``; row `k` of a bound entry
        supplies that APoI's coefficients over the weights.
    bounds : list of tuple or None
        Entries ``(apoi_index, lower, upper)`` with ``None`` on an unbounded side,
        the format :func:`mga_engine.navigation.navigate` returns as
        ``loop1_bounds``.

    Returns
    -------
    A_ub : numpy.ndarray or None
        Shape ``(rows, n)``. One row per finite side: ``row @ alpha <= upper`` for an
        upper bound, the negated row for a lower bound.
    b_ub : numpy.ndarray or None
        Shape ``(rows,)``. The matching right-hand sides.

    Notes
    -----
    Returns ``(None, None)`` when `bounds` is ``None`` or empty, and also when every
    entry has ``None`` on both sides, which linprog reads as no inequality system at
    all. Bounds are not validated: a lower above its upper simply yields an
    infeasible LP, which the callers report as a failed direction.
    """
    if not bounds:
        return None, None
    A, b = [], []
    for k, lo, hi in bounds:
        row = P_tilde[k]
        if hi is not None:
            A.append(row)
            b.append(float(hi))
        if lo is not None:
            A.append(-row)
            b.append(-float(lo))
    if not A:
        return None, None
    return np.array(A), np.array(b)


def _solve_direction_2d(
    row_i: np.ndarray,
    row_j: np.ndarray,
    w: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Maximise ``w[0] * row_i . alpha + w[1] * row_j . alpha`` over the simplex.

    Parameters
    ----------
    row_i, row_j : numpy.ndarray
        Shape ``(n,)``. The two APoI rows spanning the plane. Either the raw rows of
        P_tilde or rows already divided by an axis scale; the returned point is in
        whatever units these carry.
    w : numpy.ndarray
        Shape ``(2,)``. In-plane direction to maximise along. linprog minimises, so
        the objective handed to it is the negated combination.
    A_ub, b_ub : numpy.ndarray or None
        Extra inequality rows over the weights from :func:`_bounds_matrix`, or
        ``None`` for the unrestricted hull.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(2,)``. The point ``(row_i . alpha, row_j . alpha)`` at the optimum,
        or ``None`` if the LP did not solve to optimality.

    Notes
    -----
    The weights are constrained to the simplex (non-negative, summing to 1) in
    addition to `A_ub`. Any non-zero SciPy status, infeasible and unbounded alike,
    collapses to ``None``, so the caller cannot tell which occurred.
    One SciPy/HiGHS LP; deterministic; no network access.
    """
    n = row_i.shape[0]
    c = -(float(w[0]) * row_i + float(w[1]) * row_j)
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if result.status != 0:
        return None
    return np.array([row_i @ result.x, row_j @ result.x])


def _ccw_hull_vertices(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, as vertices in counter-clockwise order.

    Parameters
    ----------
    pts : numpy.ndarray
        Shape ``(k, 2)``. Points to hull.

    Returns
    -------
    numpy.ndarray
        Shape ``(h, 2)``. Hull vertices; SciPy reports them CCW for 2-D input.

    Raises
    ------
    scipy.spatial.QhullError
        If fewer than 3 points are given, or they are collinear. Unlike in
        :mod:`mga_engine.projection`, both callers here catch this and fall back to
        a degenerate result.
    """
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def _outward_edge_normals(points_ccw: np.ndarray) -> List[np.ndarray]:
    """Return the outward unit normal of every edge of a CCW polygon.

    Parameters
    ----------
    points_ccw : numpy.ndarray
        Shape ``(k, 2)``. Hull vertices in counter-clockwise order.

    Returns
    -------
    list of numpy.ndarray
        One unit vector of shape ``(2,)`` per edge, in edge order: edge ``(dx, dy)``
        gives ``(dy, -dx)`` normalised. Zero-length edges are skipped.

    Notes
    -----
    Orientation is assumed, not checked: clockwise input yields inward normals.
    Duplicated from :func:`mga_engine.projection._outward_edge_normals`.
    """
    normals = []
    n = len(points_ccw)
    for k in range(n):
        edge = points_ccw[(k + 1) % n] - points_ccw[k]
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normals.append(normal / norm)
    return normals


def _already_tried(w: np.ndarray, tried: List[np.ndarray], angle_tol: float = 1e-3) -> bool:
    """Report whether a direction has effectively been shot already.

    Parameters
    ----------
    w : numpy.ndarray
        Shape ``(2,)``. Candidate unit direction.
    tried : list of numpy.ndarray
        Unit directions already shot.
    angle_tol : float, default 1e-3
        COSINE tolerance despite the name: `w` counts as tried when ``w . t > 1 -
        angle_tol`` for some `t`, about 0.045 rad (2.6 degrees) at the default.

    Returns
    -------
    bool
        True if `w` matches a previously shot direction that closely.

    Notes
    -----
    Duplicated from :func:`mga_engine.projection._already_tried`.
    """
    for t in tried:
        if np.dot(w, t) > 1.0 - angle_tol:
            return True
    return False


def _dedupe(pts: np.ndarray, tol: float) -> np.ndarray:
    """Drop 2-D points that duplicate an earlier one within a Euclidean tolerance.

    Parameters
    ----------
    pts : numpy.ndarray
        Shape ``(k, 2)``. Points in the order they were found.
    tol : float
        Absolute Euclidean tolerance. Required here, unlike
        :func:`mga_engine.projection._dedupe`, because the caller sizes it to the
        data; the points reaching this function are already axis-scaled.

    Returns
    -------
    numpy.ndarray
        Shape ``(h, 2)``. The first occurrence of each distinct point, in input
        order. An empty input yields an array of shape ``(0,)``, not ``(0, 2)``.

    Notes
    -----
    Quadratic in the number of survivors.
    """
    out = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < tol for q in out):
            out.append(p)
    return np.array(out)


def _classify(pts: np.ndarray, scale: float) -> Tuple[np.ndarray, str]:
    """Turn the collected boundary points into the final region, however degenerate.

    Parameters
    ----------
    pts : numpy.ndarray
        Shape ``(k, 2)``. Collected boundary points, in SCALED coordinates.
    scale : float
        Characteristic size of the point cloud, used to size the dedupe tolerance
        as ``max(1e-12, 1e-9 * scale)``.

    Returns
    -------
    points : numpy.ndarray
        Shape ``(h, 2)``: the CCW hull vertices, the two extreme points, or the
        single surviving point, matching `kind`.
    kind : str
        ``"polygon"`` in the generic case, ``"segment"`` when the points are
        (near-)collinear, ``"point"`` when they all coincide.

    Notes
    -----
    Collinearity is decided by an SVD of the centred points: the region is a segment
    when the second singular value is at most ``1e-6`` times the first, and the two
    endpoints are then the extremes along the first principal direction.
    Never raises ``QhullError``: a hull failure falls back to the same segment, so a
    genuinely 1-D or 0-D region is reported rather than crashing the caller.
    """
    distinct = _dedupe(pts, tol=max(1e-12, 1e-9 * scale))
    if len(distinct) == 1:
        return distinct, "point"

    centered = distinct - distinct.mean(axis=0)
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    if s[1] <= 1e-6 * s[0]:
        # (near-)collinear: segment between the extremes along the main axis
        t = centered @ Vt[0]
        return np.array([distinct[np.argmin(t)], distinct[np.argmax(t)]]), "segment"

    try:
        return _ccw_hull_vertices(distinct), "polygon"
    except QhullError:
        t = centered @ Vt[0]
        return np.array([distinct[np.argmin(t)], distinct[np.argmax(t)]]), "segment"


def compute_alpha_hull_2d(
    P_tilde: np.ndarray,
    i: int,
    j: int,
    bounds=None,
    max_iterations: int = 50,
    min_improvement: float = 1e-6,
) -> Tuple[np.ndarray, str]:
    """Compute the 2-D boundary of the sampled space projected onto APoIs (i, j).

    The region is the convex hull of the columns of `P_tilde`, optionally intersected
    with `bounds`. Adaptive edge-normal refinement, as in
    :func:`mga_engine.projection.project_hull`: four axis-aligned seeds, then each
    hull edge's outward normal is shot once and the point kept when it pushes the
    hull out by more than the relative threshold.

    Parameters
    ----------
    P_tilde : numpy.ndarray
        Shape ``(m + 1, n)``. APoI matrix ``[v'; P_all]``: row 0 holds the cost of
        each sample in EUR/yr, rows 1..m the PoI values in GW.
    i, j : int
        APoI indices of the two axes. Must differ; index 0 (cost) is allowed.
    bounds : list of tuple, optional
        Entries ``(apoi_index, lower, upper)`` with ``None`` on an unbounded side,
        the ``loop1_bounds`` format of :func:`mga_engine.navigation.navigate`.
        ``None`` or empty gives the unrestricted outer hull.
    max_iterations : int, default 50
        Cap on refinement rounds, as a safety net rather than a normal stop.
    min_improvement : float, default 1e-6
        Relative acceptance threshold for a pushed-out point, applied in scaled
        coordinates.

    Returns
    -------
    points : numpy.ndarray
        Shape ``(k, 2)``, in the ORIGINAL units of axes `i` and `j`: CCW hull
        vertices for ``"polygon"``, the two endpoints for ``"segment"``, or a single
        point of shape ``(1, 2)`` for ``"point"``.
    kind : str
        ``"polygon"``, ``"segment"`` or ``"point"``; see :func:`_classify`.

    Raises
    ------
    ValueError
        If ``i == j``.
    RuntimeError
        If all four seed solves fail, which means `bounds` does not intersect the
        sampled space at all.

    Notes
    -----
    The two axes are internally divided by their span before refinement, because cost
    in EUR/yr and PoIs in GW differ by roughly nine orders of magnitude and a single
    relative threshold could not serve both; the result is multiplied back, so
    callers always see original units. An axis whose span is pure solver noise keeps
    its magnitude as the scale instead, which leaves the region correctly thin.
    A failed direction solve is skipped silently and not counted, and a ``QhullError``
    during refinement ends refinement early and leaves the rest to :func:`_classify`,
    so a degenerate region returns a ``"segment"`` or ``"point"`` rather than raising.
    SciPy LPs only: no network, no file access, no prints, and deterministic.
    """
    if i == j:
        raise ValueError("compute_alpha_hull_2d needs two DIFFERENT APoI indices (i != j).")

    P_tilde = np.asarray(P_tilde, dtype=float)
    A_ub, b_ub = _bounds_matrix(P_tilde, bounds)

    # --- 1. seed: 4 axis-aligned directions (max/min of each axis) ---
    seed_dirs = [
        np.array([+1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, +1.0]),
        np.array([0.0, -1.0]),
    ]
    points = []
    for w in seed_dirs:
        pt = _solve_direction_2d(P_tilde[i], P_tilde[j], w, A_ub, b_ub)
        if pt is not None:
            points.append(pt)
    if not points:
        raise RuntimeError(
            "[aproj] All seed solves failed — the bounds do not intersect the sampled space."
        )

    # Normalize each axis by its span (the seeds hit each axis's min/max) so
    # refinement and degeneracy checks are unit-independent - cost (EUR/yr) and
    # PoIs (GW) differ by ~9 orders of magnitude. An axis whose span is pure
    # solver noise keeps its magnitude as scale, so it stays thin when scaled.
    pts_arr = np.array(points)
    axis_scale = np.empty(2)
    for a in range(2):
        lo, hi = pts_arr[:, a].min(), pts_arr[:, a].max()
        span, mag = hi - lo, max(1.0, abs(lo), abs(hi))
        axis_scale[a] = span if span > 1e-8 * mag else mag
    row_i_s = P_tilde[i] / axis_scale[0]
    row_j_s = P_tilde[j] / axis_scale[1]
    points = [pt / axis_scale for pt in points]

    scale = np.linalg.norm(np.ptp(np.array(points), axis=0))
    if scale == 0:
        scale = 1.0
    abs_improve = min_improvement * scale

    # --- 2-4. adaptive edge-normal refinement (in scaled coordinates) ---
    tried_normals = []
    for _ in range(max_iterations):
        try:
            hull_pts = _ccw_hull_vertices(np.array(points))
        except QhullError:
            break  # degenerate region - _classify below handles it

        added_this_round = False
        for w in _outward_edge_normals(hull_pts):
            if _already_tried(w, tried_normals):
                continue
            tried_normals.append(w)

            pt = _solve_direction_2d(row_i_s, row_j_s, w, A_ub, b_ub)
            if pt is None:
                continue
            if pt @ w - np.max(hull_pts @ w) > abs_improve:
                points.append(pt)
                added_this_round = True

        if not added_this_round:
            break

    result_pts, kind = _classify(np.array(points), scale)
    return result_pts * axis_scale, kind
