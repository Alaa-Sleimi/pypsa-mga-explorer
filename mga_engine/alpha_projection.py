"""
alpha_projection.py — Adaptive 2D projection of the SAMPLED near-optimal space.

Unlike projection.py (which solves MP(r) on the live network and yields the
true LP boundary), this module works entirely in alpha space: the region is
conv(columns of P_tilde) — the convex hull of the stored samples, i.e. the
space the navigation algorithm (navigation.py) actually searches — optionally
intersected with per-APoI interval bounds (the loop-1 bounds navigate()
returns). Each direction solve is a small scipy linprog over the weights
alpha; no PyPSA/Linopy model is involved.

Directions are chosen with the same adaptive edge-normal refinement scheme as
projection.py: seed with 4 axis-aligned directions, build the 2D convex hull,
push outward along each edge normal, and stop when no push improves the hull
by more than a relative threshold (or the iteration cap is reached).

APoI index convention (as in navigation.py):
    0        = cost (row 0 of P_tilde = v)
    1, 2, .. = PoI values
"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError
from typing import List, Optional, Tuple


def _bounds_matrix(P_tilde: np.ndarray, bounds) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Turn (apoi_index, lower, upper) bounds into linprog A_ub / b_ub rows."""
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
    """
    Maximize  w[0] * row_i·alpha + w[1] * row_j·alpha  over alpha in the simplex
    (intersected with the extra A_ub/b_ub bounds), i.e. minimize the negation.

    Returns the 2D point (row_i·alpha, row_j·alpha), or None if the solve failed.
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
    """Convex hull vertices of 2D points in CCW order (scipy is CCW for 2D)."""
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def _outward_edge_normals(points_ccw: np.ndarray) -> List[np.ndarray]:
    """Outward unit normal of each edge of a CCW polygon: (dx,dy) -> (dy,-dx)."""
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
    """True if w is within angle_tol (cosine) of a normal already shot."""
    for t in tried:
        if np.dot(w, t) > 1.0 - angle_tol:
            return True
    return False


def _dedupe(pts: np.ndarray, tol: float) -> np.ndarray:
    """Remove duplicate 2D points (within Euclidean tol)."""
    out = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < tol for q in out):
            out.append(p)
    return np.array(out)


def _classify(pts: np.ndarray, scale: float) -> Tuple[np.ndarray, str]:
    """
    Turn the collected boundary points into the final result:
      ("polygon", CCW hull vertices)  — the generic case
      ("segment", 2 endpoints)        — points (near-)collinear
      ("point",   1 point)            — everything coincides
    Never raises QhullError: degenerate regions fall back gracefully.
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
    """
    Compute the 2D boundary of the sampled space conv(columns of P_tilde),
    optionally intersected with per-APoI interval bounds, projected onto
    APoIs (i, j).

    Parameters
    ----------
    P_tilde        : np.ndarray (m+1, n) — APoI matrix [v'; P] (row 0 = cost)
    i, j           : APoI indices of the two axes (0 = cost allowed)
    bounds         : optional list of (apoi_index, lower, upper), None where
                     unbounded — the loop-1 format from navigate(). None or
                     empty gives the unrestricted (outer) hull.
    max_iterations : cap on edge-normal refinement rounds (safety net)
    min_improvement: relative threshold for accepting a pushed-out point

    Returns
    -------
    (points, kind) :
        kind "polygon" — points are CCW hull vertices, shape (k, 2), k >= 3
        kind "segment" — points are the 2 endpoints (degenerate, 1D region)
        kind "point"   — points is a single point, shape (1, 2)
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
    # refinement and degeneracy checks are unit-independent — cost (€/yr) and
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
            break  # degenerate region — _classify below handles it

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
