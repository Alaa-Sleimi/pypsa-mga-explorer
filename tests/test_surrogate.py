"""
tests/test_surrogate.py — Test surrogate cost recovery (SP(p) MILP).

Geometry checks (boundary distance, projection) run on a small synthetic
hull (the unit square) where every answer is known in closed form. One
integration check runs a single real MILP solve on the saved preparation
state, and is skipped when data/preparation_state.nc is absent (the file
is gitignored and not available in CI).

Run with:
    python -m tests.test_surrogate
"""

import os
import sys

import numpy as np

from mga_engine.surrogate import (
    project_to_boundary,
    signed_boundary_distance,
    solve_surrogate_costs,
)

results = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    results.append(ok)


# --- Synthetic hull: unit square, corners as sample columns (m=2, n=4) ---
P_sq = np.array([[0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 1.0]])

# --- 1. Signed boundary distance: interior / boundary / exterior signs ---
d_in = signed_boundary_distance(np.array([0.5, 0.5]), P_sq)
check("interior point has positive boundary distance",
      abs(d_in - 0.5) < 1e-9, f"d = {d_in:.6f}, expected 0.5")

d_on = signed_boundary_distance(np.array([1.0, 0.5]), P_sq)
check("boundary point has ~zero boundary distance",
      abs(d_on) < 1e-9, f"d = {d_on:.2e}")

d_out = signed_boundary_distance(np.array([1.5, 0.5]), P_sq)
check("exterior point has negative boundary distance",
      abs(d_out + 0.5) < 1e-9, f"d = {d_out:.6f}, expected -0.5")

# --- 2. Projection lands on the hull surface ---
p_hat, d_proj = project_to_boundary(np.array([0.6, 0.5]), P_sq)
check("projection returns a point", p_hat is not None)
if p_hat is not None:
    d_hat = signed_boundary_distance(p_hat, P_sq)
    check("projected point lies on the hull surface",
          abs(d_hat) < 1e-9, f"|d| = {abs(d_hat):.2e}")
    check("projection distance is the distance to the nearest facet",
          abs(d_proj - 0.4) < 1e-9, f"d = {d_proj:.6f}, expected 0.4")

# --- 3. Boundary solve: feasible, normalized, correct supporting normal ---
res_b = solve_surrogate_costs(np.array([1.0, 0.5]), P_sq)
check("boundary point solves (status='boundary')",
      res_b.status == "boundary" and res_b.r is not None,
      f"status = {res_b.status}")
if res_b.r is not None:
    l1 = float(np.sum(np.abs(res_b.r)))
    check("recovered r is L1-normalized", abs(l1 - 1.0) < 1e-8,
          f"||r||_1 = {l1:.8f}")
    # mid-edge of x=1: the only supporting normals are (-a, 0), a > 0
    check("r is the correct supporting normal (-1, 0)",
          np.allclose(res_b.r, [-1.0, 0.0], atol=1e-6),
          f"r = {res_b.r}")

# --- 4. Interior point triggers projection before the solve ---
res_i = solve_surrogate_costs(np.array([0.5, 0.4]), P_sq)
check("interior point is projected (projected=True)",
      res_i.projected and res_i.status == "projected",
      f"status = {res_i.status}, projected = {res_i.projected}")
check("interior projection distance is recorded",
      abs(res_i.projection_distance - 0.4) < 1e-9,
      f"d = {res_i.projection_distance:.6f}, expected 0.4")

# --- 5. Shape-mismatch guard ---
try:
    solve_surrogate_costs(np.zeros(3), P_sq)
    check("p/P shape mismatch raises ValueError", False, "no exception raised")
except ValueError:
    check("p/P shape mismatch raises ValueError", True)

# --- 6. Integration: one real MILP solve on the saved preparation state ---
STATE_PATH = "data/preparation_state.nc"
if os.path.exists(STATE_PATH):
    from scipy.spatial import ConvexHull

    from mga_engine.preparation_state import load

    P_all = load(STATE_PATH)["P_all"]
    vtx = ConvexHull(P_all.T).vertices[0]        # a true hull vertex
    res = solve_surrogate_costs(P_all[:, vtx], P_all)
    check("real-data hull vertex yields a surrogate cost vector",
          res.status != "failed" and res.r is not None,
          f"status = {res.status}, message = {res.message}")
    if res.r is not None:
        l1 = float(np.sum(np.abs(res.r)))
        check("real-data r is L1-normalized", abs(l1 - 1.0) < 1e-8,
              f"||r||_1 = {l1:.8f}")
else:
    print(f"[SKIP] integration check: {STATE_PATH} not found")

# --- Summary ---
n_fail = results.count(False)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
