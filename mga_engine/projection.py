"""On-demand 2-D projection of the true near-optimal space onto a pair of PoIs.

Exploration-phase plotting support. For a chosen PoI pair it computes the boundary of
the projection of F^P onto that plane by solving MP(r) on a live network for
directions confined to those two coordinates, which avoids the artefact that plotted
sample points can look interior in 2-D merely because of the other dimensions.
Reuses the MP(r) machinery of :mod:`mga_engine.vertex_sampling`; the cheap
sample-space counterpart, which needs no network, is
:mod:`mga_engine.alpha_projection`.
"""

import numpy as np
import pypsa
from typing import List
from scipy.spatial import ConvexHull

from mga_engine.poi import PoiSpec
from mga_engine.vertex_sampling import _build_mga_model, _solve_mp


def _solve_direction_2d(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    i: int,
    j: int,
    w: np.ndarray,
):
    """Maximise ``w[0] * PoI_i + w[1] * PoI_j`` over the near-optimal set.

    Parameters
    ----------
    network : pypsa.Network
        Network whose model was prepared by
        :func:`mga_engine.vertex_sampling._build_mga_model`.
    poi_specs : list of PoiSpec
        All PoI specs of the network; only entries `i` and `j` enter the objective,
        but the full list is needed to read the solution back.
    i, j : int
        Indices into `poi_specs` of the two PoIs spanning the plane.
    w : numpy.ndarray
        Shape ``(2,)``. In-plane direction to maximise along.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(2,)``. The point ``(PoI_i, PoI_j)`` in GW at the optimum, or
        ``None`` if the solve failed.

    Notes
    -----
    :func:`mga_engine.vertex_sampling._solve_mp` MINIMISES ``r'p``, so maximising
    along `w` means minimising along ``-w``: the full-length direction is zero
    everywhere except ``[i] = -w[0]`` and ``[j] = -w[1]``.
    A ``RuntimeError`` from the solve is swallowed and reported as ``None``, so the
    caller cannot distinguish an infeasible direction from any other solver failure.
    Calls HiGHS and mutates `network`.
    """
    m_dim = len(poi_specs)
    direction = np.zeros(m_dim)
    direction[i] = -float(w[0])
    direction[j] = -float(w[1])

    try:
        p_full = _solve_mp(network, poi_specs, direction)
    except RuntimeError:
        return None

    return np.array([p_full[i], p_full[j]])


def _outward_edge_normals(points_ccw: np.ndarray) -> List[np.ndarray]:
    """Return the outward unit normal of every edge of a CCW polygon.

    Parameters
    ----------
    points_ccw : numpy.ndarray
        Shape ``(k, 2)``. Hull vertices in counter-clockwise order.

    Returns
    -------
    list of numpy.ndarray
        One unit vector of shape ``(2,)`` per edge ``v_k -> v_{k+1}``, in edge order.
        For a CCW polygon the outward normal of edge ``(dx, dy)`` is ``(dy, -dx)``
        normalised. Zero-length edges are skipped, so the list can be shorter than
        `points_ccw`.

    Notes
    -----
    Orientation is assumed, not checked: given clockwise input every normal points
    inward instead.
    """
    normals = []
    n = len(points_ccw)
    for k in range(n):
        v0 = points_ccw[k]
        v1 = points_ccw[(k + 1) % n]
        edge = v1 - v0
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normals.append(normal / norm)
    return normals


def _ccw_hull_vertices(pts: np.ndarray) -> np.ndarray:
    """Convex hull of a set of 2-D points, as vertices in counter-clockwise order.

    Parameters
    ----------
    pts : numpy.ndarray
        Shape ``(k, 2)``. Points to hull.

    Returns
    -------
    numpy.ndarray
        Shape ``(h, 2)``. Hull vertices. SciPy reports ``ConvexHull.vertices`` in CCW
        order for 2-D input, so no reordering is needed here.

    Raises
    ------
    scipy.spatial.QhullError
        If fewer than 3 points are given, or they are collinear. :func:`project_hull`
        does NOT catch this, unlike the equivalent helper in
        :mod:`mga_engine.alpha_projection`.
    """
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def project_hull(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    opt_cost: float,
    epsilon: float,
    i: int,
    j: int,
    max_iterations: int = 50,
    min_improvement: float = 1e-6,
) -> np.ndarray:
    """Compute the 2-D boundary of the near-optimal space projected onto PoIs (i, j).

    Adaptive edge-normal refinement: seed with the four axis-aligned directions
    (max and min of each of the two PoIs), hull the points found so far, then shoot
    each hull edge's outward normal and keep the returned point when it lies outside
    the current hull by more than `min_improvement` relative to the hull's size.
    Repeat until a round adds nothing or `max_iterations` rounds have passed.

    Parameters
    ----------
    network : pypsa.Network
        Unsolved network. A fresh MGA model is built on it here and reused for every
        direction, so the same object serves the whole projection.
    poi_specs : list of PoiSpec
        PoI specs for `network`, in the row order `i` and `j` index into.
    opt_cost : float
        Minimum system cost c'x* in EUR/yr; see
        :func:`mga_engine.vertex_sampling._build_mga_model`.
    epsilon : float
        Relative cost slack defining the near-optimal set.
    i, j : int
        Indices into `poi_specs` of the two PoIs to project onto. Must differ.
    max_iterations : int, default 50
        Cap on refinement rounds, as a safety net rather than a normal stop.
    min_improvement : float, default 1e-6
        Relative acceptance threshold. A new point counts only if its support along
        the shot normal beats the current hull's by more than
        ``min_improvement * scale``, where ``scale`` is the norm of the seed points'
        bounding-box diagonal. This is what stops solver noise from looping forever.

    Returns
    -------
    numpy.ndarray
        Shape ``(k, 2)``. Hull vertices in counter-clockwise order, in
        ``(PoI_i, PoI_j)`` coordinates, in GW.

    Raises
    ------
    ValueError
        If ``i == j``.
    RuntimeError
        If fewer than 3 distinct seed points survive, which usually means the
        projection is degenerate (a segment or a point).
    scipy.spatial.QhullError
        Propagated from :func:`_ccw_hull_vertices` when the points are collinear.
        Unlike :func:`mga_engine.alpha_projection.compute_alpha_hull_2d`, this
        function has no degenerate fallback.

    Notes
    -----
    Calls HiGHS once per direction: 4 seeds plus one per accepted normal, on `network`
    itself, whose model and result columns are overwritten throughout.
    Individual failed direction solves are silently skipped and only counted, so a
    hull can be built from fewer directions than were tried; the count is printed.
    Reaching `max_iterations` is reported on stdout and leaves an APPROXIMATE hull,
    inscribed in the true projection since every vertex is an attained point.
    Each returned normal is shot at most once, within a cosine tolerance of 1e-3.
    """
    if i == j:
        raise ValueError("project_hull needs two DIFFERENT PoI indices (i != j).")

    _build_mga_model(network, opt_cost, epsilon)

    name_i = poi_specs[i].name
    name_j = poi_specs[j].name
    print(f"[proj] Projecting near-optimal space onto ({name_i}, {name_j}) ...")

    # --- 1. seed: 4 axis-aligned directions (max/min of each PoI) ---
    seed_dirs = [
        np.array([+1.0, 0.0]),  # max PoI_i
        np.array([-1.0, 0.0]),  # min PoI_i
        np.array([0.0, +1.0]),  # max PoI_j
        np.array([0.0, -1.0]),  # min PoI_j
    ]

    points = []
    n_failed = 0
    for w in seed_dirs:
        pt = _solve_direction_2d(network, poi_specs, i, j, w)
        if pt is not None:
            points.append(pt)
        else:
            n_failed += 1

    points = _dedupe(points)
    if len(points) < 3:
        raise RuntimeError(
            f"[proj] Only {len(points)} distinct seed points found; "
            "cannot form a 2D hull (the projection may be degenerate / 1D)."
        )

    # characteristic scale for the relative improvement threshold
    pts_arr = np.array(points)
    scale = np.linalg.norm(pts_arr.max(axis=0) - pts_arr.min(axis=0))
    if scale == 0:
        scale = 1.0
    abs_improve = min_improvement * scale

    # --- 2-4. adaptive edge-normal refinement ---
    tried_normals = []  # avoid re-shooting essentially the same normal
    for iteration in range(max_iterations):
        hull_pts = _ccw_hull_vertices(np.array(points))
        normals = _outward_edge_normals(hull_pts)

        added_this_round = False
        for w in normals:
            # skip a normal we've already shot in (within a small angle)
            if _already_tried(w, tried_normals):
                continue
            tried_normals.append(w)

            pt = _solve_direction_2d(network, poi_specs, i, j, w)
            if pt is None:
                n_failed += 1
                continue

            # how far outside the current hull is this point, along w?
            # support of current hull in direction w vs. the new point's projection
            current_support = np.max(hull_pts @ w)
            new_support = pt @ w
            if new_support - current_support > abs_improve:
                points.append(pt)
                added_this_round = True

        if not added_this_round:
            print(f"[proj] Converged after {iteration + 1} refinement round(s).")
            break
    else:
        print(f"[proj] Reached max_iterations={max_iterations} (may be approximate).")

    if n_failed:
        print(f"[proj] {n_failed} direction solve(s) skipped (infeasible/unbounded).")

    hull_xy = _ccw_hull_vertices(np.array(points))
    print(f"[proj] Hull has {len(hull_xy)} vertices.")
    return hull_xy


def _dedupe(points: List[np.ndarray], tol: float = 1e-9) -> List[np.ndarray]:
    """Drop points that duplicate an earlier one within an absolute tolerance.

    Parameters
    ----------
    points : list of numpy.ndarray
        Points of shape ``(2,)``, in the order they were found.
    tol : float, default 1e-9
        ABSOLUTE Euclidean tolerance, not scaled to the data. On PoI values in GW
        this is effectively exact-duplicate removal.

    Returns
    -------
    list of numpy.ndarray
        The first occurrence of each distinct point, in input order.

    Notes
    -----
    Compares every candidate against every kept point, so the cost is quadratic in
    the number of survivors; fine for the handful of points a projection produces.
    """
    out = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in out):
            out.append(p)
    return out


def _already_tried(w: np.ndarray, tried: List[np.ndarray], angle_tol: float = 1e-3) -> bool:
    """Report whether a direction has effectively been shot already.

    Parameters
    ----------
    w : numpy.ndarray
        Shape ``(2,)``. Candidate unit direction.
    tried : list of numpy.ndarray
        Unit directions already shot.
    angle_tol : float, default 1e-3
        COSINE tolerance despite the name: `w` counts as already tried when
        ``w . t > 1 - angle_tol`` for some `t`, i.e. within about 0.045 rad (2.6
        degrees) of it at the default value.

    Returns
    -------
    bool
        True if `w` matches a previously shot direction that closely.
    """
    for t in tried:
        if np.dot(w, t) > 1.0 - angle_tol:
            return True
    return False
