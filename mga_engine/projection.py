"""
projection.py — Adaptive 2D projection of the near-optimal feasible space.

For a chosen pair of PoIs (i, j) we compute the TRUE boundary of the projection
of F^P onto the (PoI_i, PoI_j) plane, by solving MP(r) for directions that are
zero in all PoI coordinates except i and j.

Directions are chosen adaptively (edge-normal refinement):
  1. Seed with 4 axis-aligned directions: max/min PoI_i, max/min PoI_j.
  2. Build the 2D convex hull of the points found so far.
  3. For each hull edge, take its OUTWARD normal as the next direction and
     maximize PoI along it (via MP(r) with the negated normal, since _solve_mp
     minimizes r'p).
  4. If the new point lies outside the current hull by more than a relative
     threshold, add it and repeat. Stop when no edge can be pushed further or
     the iteration cap is reached.

The result is the convex hull of the 2D projection — which avoids the problem
that plotted sample points can appear interior in 2D because of the other
dimensions.

Computed on request per PoI pair, on a live network. Reuses the MP(r) machinery
from vertex_sampling (_build_mga_model, _solve_mp).
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
    """
    Maximize  w_x * PoI_i + w_y * PoI_j  over the near-optimal set.

    _solve_mp minimizes r'p, so to MAXIMIZE along w we minimize along -w:
    we build a full-length direction that is zero everywhere except
    index i = -w_x and index j = -w_y.

    Returns the 2D point (PoI_i, PoI_j), or None if the solve failed.
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
    """
    Given hull vertices in counter-clockwise order, return the OUTWARD unit
    normal for each edge (v_k -> v_{k+1}).

    For a CCW polygon, the outward normal of edge (dx, dy) is (dy, -dx)
    normalized.
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
    """
    Convex hull of a set of 2D points, returned as vertices in CCW order.
    scipy's ConvexHull.vertices are already CCW for 2D input.
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
    """
    Compute the 2D convex-hull boundary of the near-optimal space projected
    onto PoIs (i, j).

    Parameters
    ----------
    network        : PyPSA network (a fresh MGA model is built on it here)
    poi_specs      : list of PoiSpec
    opt_cost       : optimal total system cost c'x*
    epsilon        : near-optimality slack
    i, j           : indices (into poi_specs) of the two PoIs to project onto
    max_iterations : cap on edge-normal refinement directions (safety net)
    min_improvement: relative threshold; a new point is accepted only if it lies
                     outside the current hull by more than this (scaled to hull
                     size), which guards against solver-noise non-termination

    Returns
    -------
    hull_xy : np.ndarray of shape (k, 2)
        Ordered (CCW) hull vertices in (PoI_i, PoI_j) coordinates.
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
    """Remove duplicate points (within tol)."""
    out = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in out):
            out.append(p)
    return out


def _already_tried(w: np.ndarray, tried: List[np.ndarray], angle_tol: float = 1e-3) -> bool:
    """True if w is within angle_tol (cosine) of a normal we've already shot."""
    for t in tried:
        if np.dot(w, t) > 1.0 - angle_tol:
            return True
    return False


if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, evaluate_all

    network = build_network()
    network.optimize(solver_name="highs", include_objective_constant=False,
                     solver_options={"output_flag": False})
    opt_cost = network.objective
    poi_specs = make_poi_specs(network)

    network_proj = build_network()
    poi_specs_proj = make_poi_specs(network_proj)

    hull = project_hull(
        network_proj, poi_specs_proj, opt_cost,
        epsilon=0.05, i=0, j=1,
    )

    print("\nHull vertices (PoI_0, PoI_1):")
    print(np.round(hull, 3))
