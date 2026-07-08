"""
tests/test_alpha_projection.py — Test the sampled-space 2D hull (alpha_projection).

Loads P_all, v from data/preparation_state.nc and checks compute_alpha_hull_2d:
  1. the outer hull (no bounds) contains all projected sample points
  2. bounding a third APoI shrinks (never grows) the region, strictly when
     the bound cuts into the observed range
  3. cost (APoI 0) works as an axis, with cost values inside the sampled
     cost range [min(v), max(v)] (a subset of [opt_cost, (1+eps)*opt_cost])
  4. a zero-width band on a plotted axis degrades to segment/point, no crash

Run with:
    python -m tests.test_alpha_projection
"""

import sys
import numpy as np
from scipy.spatial import ConvexHull

from mga_engine.preparation_state import load
from mga_engine.alpha_projection import compute_alpha_hull_2d

# --- Load preparation state ---
state = load("data/preparation_state.nc")
P_all = state["P_all"]
v     = state["v"]

# APoI matrix: row 0 = cost, rows 1..m = PoIs (navigation.py convention)
P_tilde = np.vstack([v[np.newaxis, :], P_all])

results = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    results.append(ok)


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --- 1. Outer hull contains all projected sample points (PoI pair 1, 2) ---
i, j = 1, 2
outer_pts, outer_kind = compute_alpha_hull_2d(P_tilde, i, j)
if outer_kind == "polygon":
    samples = P_tilde[[i, j]].T                     # (n, 2) projected samples
    scale = np.linalg.norm(outer_pts.max(axis=0) - outer_pts.min(axis=0))
    eqs = ConvexHull(outer_pts).equations           # A @ p + b <= 0 inside
    max_violation = float((samples @ eqs[:, :2].T + eqs[:, 2]).max())
    ok = max_violation <= 1e-6 * scale
    detail = f"max violation = {max_violation:.3e}, tol = {1e-6 * scale:.3e}"
else:
    ok, detail = False, f"expected polygon, got kind={outer_kind!r}"
check("1. outer hull contains all projected sample points", ok, detail)

# --- 2. Bound on a third APoI (cost) shrinks the region ---
# Cap cost inside its observed range: only the cheap part of the space remains.
cost_cap = float(v.min() + 0.2 * (v.max() - v.min()))
inner_pts, inner_kind = compute_alpha_hull_2d(P_tilde, i, j, bounds=[(0, None, cost_cap)])
area_out = polygon_area(outer_pts) if outer_kind == "polygon" else 0.0
area_in = polygon_area(inner_pts) if inner_kind == "polygon" else 0.0
ok = (area_in <= area_out * (1 + 1e-9)) and (area_in < area_out * 0.999)
check("2. bounding a third APoI (cost cap) strictly shrinks the region", ok,
      f"outer area = {area_out:.4f}, inner area = {area_in:.4f} (kind={inner_kind})")

# --- 3. Cost as an axis: pair (0, 1) runs, costs inside the sampled range ---
try:
    cost_pts, cost_kind = compute_alpha_hull_2d(P_tilde, 0, 1)
    cost_tol = 1e-6 * float(v.max() - v.min())
    lo_ok = cost_pts[:, 0].min() >= v.min() - cost_tol
    hi_ok = cost_pts[:, 0].max() <= v.max() + cost_tol
    ok = lo_ok and hi_ok
    detail = (f"kind={cost_kind}, cost in [{cost_pts[:, 0].min():,.0f}, "
              f"{cost_pts[:, 0].max():,.0f}] vs sampled [{v.min():,.0f}, {v.max():,.0f}]")
except Exception as e:
    ok, detail = False, f"raised {type(e).__name__}: {e}"
check("3. cost axis works and stays inside the sampled cost range", ok, detail)

# --- 4. Degenerate case: zero-width band on a plotted axis -> segment/point ---
mid = float(0.5 * (P_tilde[i].min() + P_tilde[i].max()))
try:
    deg_pts, deg_kind = compute_alpha_hull_2d(P_tilde, i, j, bounds=[(i, mid, mid)])
    ok = deg_kind in ("segment", "point")
    detail = f"kind={deg_kind}, {len(deg_pts)} point(s)"
except Exception as e:
    ok, detail = False, f"raised {type(e).__name__}: {e}"
check("4. zero-width band degrades to segment/point without raising", ok, detail)

# --- Summary ---
n_fail = results.count(False)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
sys.exit(1 if n_fail else 0)
