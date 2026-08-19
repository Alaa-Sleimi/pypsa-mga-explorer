"""Surrogate cost recovery for near-optimal exploration (SP(p)).

Given a point ``p`` in PoI space and the sample matrix ``P``, recover a
surrogate cost vector ``r`` such that ``p`` is an optimal solution of the
weighted convex combination problem CCP-W(r).

This implements the MILP formulation of SP(p):

    min   w
    s.t.  P a       = p                         (1.1)
          1'a       = 1,  a >= 0                 (1.2)
          lambda*1 <= r'P                        (1.3)
          r'p       = lambda                     (1.4, linearized: Pa -> p)
          r_i >= 1 - M (1 - z_plus_i)            (1.5, big-M)
          r_i <= -1 + M (1 - z_minus_i)          (1.6, big-M)
          z_plus, z_minus in {0,1}^m             (1.7)
          z_plus_i + z_minus_i <= 1              (1.7b)
          1'z_plus + 1'z_minus >= 1              (1.8)
          -w*1 <= r <= w*1                       (1.9)

The binaries (1.5-1.8) force at least one |r_i| >= 1, which excludes the
trivial solution r = 0 without assuming a sign for r and without an
origin-facet blind spot. (1.5)/(1.6) are big-M disjunctions: with
M = big_bound + 1 (the tightest valid M for the |r_i| <= big_bound
variable bounds) the constraint is vacuous when the binary is 0 and
enforces |r_i| >= 1 when it is 1; (1.7b) keeps the two signs mutually
exclusive per coordinate. 1.9 + min w bound the solution and pick the
minimum-infinity-norm representative on the solution ray.

Interior points of conv(P) admit no supporting hyperplane, so SP(p) is
infeasible there. Following Sina's suggestion, an interior p is first
projected onto the boundary of conv(P) and SP is solved at the projected
point. Projection is a heuristic and is expected to fail on a minority of
cases (facets through the origin, low-dimensional faces); those return
status="failed" rather than raising.

Points classified as boundary (within ``tol_interior``) can still be
numerically eps-interior -- navigation targets carry a ~1e-6 lock-in
offset -- where the MILP is correctly infeasible in exact arithmetic. If
the direct solve at such a point is infeasible, it is projected onto the
hull surface (a negligible nudge) and retried once.

Contract
--------
- ``p``: 1-D array of length m (the PoI vector for the point of interest).
- ``P``: 2-D array of shape (m, n), the sample matrix; columns are samples.
Callers holding a navigation result as alpha weights should pass
``p = P @ alpha`` and the same ``P`` used elsewhere in the notebook.

NOTE ON highspy: the MILP is assembled through highspy's Highs() API using
addVars / addConstrs-style row addition via the low-level model interface.
The exact call names in highspy have shifted across releases; the single
place that touches highspy is ``_solve_milp``. If your highspy version
rejects a call there, that function is the only thing to adjust -- the rest
of the module is pure numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

try:  # QhullError location differs across scipy versions
    from scipy.spatial import QhullError as _QhullError
except ImportError:  # pragma: no cover
    try:
        from scipy.spatial.qhull import QhullError as _QhullError  # type: ignore
    except ImportError:  # pragma: no cover
        _QhullError = Exception  # fallback: catch broadly if unavailable


# HiGHS seed for the SP(p) MILP solve — fixed (together with threads=1) so
# tie-breaking in the MIP search is run-to-run deterministic.
MILP_RANDOM_SEED = 0


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SurrogateResult:
    """Outcome of a surrogate cost recovery."""

    r: Optional[np.ndarray]          # recovered cost vector (m,), normalized; None on failure
    status: str                      # "boundary" | "projected" | "failed"
    projected: bool                  # was p moved to the boundary before solving
    p_used: Optional[np.ndarray]     # the point actually solved at
    projection_distance: float       # how far p was moved (0.0 if not projected)
    w: Optional[float]               # infinity-norm value from the MILP (diagnostic)
    message: str                     # human-readable explanation
    extras: dict = field(default_factory=dict)  # alpha, lambda, raw r, etc.


# --------------------------------------------------------------------------- #
# Geometry: boundary distance and projection
# --------------------------------------------------------------------------- #
def _hull_facets(P: np.ndarray):
    """Return (A, b) with rows a_k, b_k for facets a_k . x <= b_k of conv(P).

    ``P`` is (m, n); points are the n columns. Returns None if the hull is
    degenerate / Qhull fails (caller falls back to an LP test).
    """
    pts = np.asarray(P, dtype=float).T  # (n_points, dim)
    try:
        hull = ConvexHull(pts)
    except (_QhullError, ValueError):
        return None
    # hull.equations rows are [a_1 ... a_d, b0] with a . x + b0 <= 0,
    # i.e. a . x <= -b0.
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]
    return A, b


def signed_boundary_distance(p: np.ndarray, P: np.ndarray) -> float:
    """Signed distance from p to the boundary of conv(P).

    Positive  -> strictly inside (min slack to any facet).
    ~zero     -> on the boundary.
    Negative  -> outside the hull.

    Falls back to an LP-based interiority test when Qhull is unavailable
    (high dimension). The fallback returns a *signed* value of the same
    sign convention but not a true Euclidean distance.
    """
    p = np.asarray(p, dtype=float).ravel()
    facets = _hull_facets(P)
    if facets is not None:
        A, b = facets
        # facet normals from Qhull are unit-norm, so (b - A p) is Euclidean slack
        slack = b - A @ p
        return float(np.min(slack))
    return _lp_boundary_distance(p, P)


def _lp_boundary_distance(p: np.ndarray, P: np.ndarray) -> float:
    """LP fallback interiority test for high-dimensional conv(P).

    Solves the Chebyshev-style expansion: how far can we push p in the
    "most interior" sense while staying representable as a convex
    combination? We use the max-slack LP

        max  t
        s.t. P a = p,  1'a = 1,  a >= t,  a <= 1

    If the optimal t* > 0, every sample carries strictly positive weight,
    which is a sufficient condition for p to be interior; we return t* as a
    positive (interior) signal. If infeasible or t* <= 0, we return a small
    non-positive value flagging boundary/exterior. This is a conservative
    proxy, not a Euclidean distance -- adequate only as a fallback.
    """
    from scipy.optimize import linprog

    P = np.asarray(P, dtype=float)
    m, n = P.shape
    # variables: [a (n), t (1)]
    c = np.zeros(n + 1)
    c[-1] = -1.0  # maximize t

    # equality: P a = p  and  1'a = 1
    A_eq = np.zeros((m + 1, n + 1))
    A_eq[:m, :n] = P
    A_eq[m, :n] = 1.0
    b_eq = np.concatenate([p, [1.0]])

    # inequality: a >= t  ->  t - a_i <= 0
    A_ub = np.zeros((n, n + 1))
    for i in range(n):
        A_ub[i, i] = -1.0
        A_ub[i, -1] = 1.0
    b_ub = np.zeros(n)

    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return -1.0  # cannot represent p as convex combo -> treat as exterior
    t_star = res.x[-1]
    return float(t_star)


def project_to_boundary(p: np.ndarray, P: np.ndarray):
    """Project an interior p onto the surface of conv(P).

    Reading (ii): nearest point on the hull *boundary*. For each facet, we
    project p onto the facet's hyperplane; among the projections that land
    inside the hull (all facet constraints satisfied up to tolerance), we
    take the closest. Returns (p_hat, distance). If no valid projection is
    found (e.g. Qhull unavailable), returns (None, inf).
    """
    p = np.asarray(p, dtype=float).ravel()
    facets = _hull_facets(P)
    if facets is None:
        return None, float("inf")
    A, b = facets

    best_p = None
    best_d = float("inf")
    tol = 1e-9
    for k in range(A.shape[0]):
        a_k = A[k]
        b_k = b[k]
        denom = a_k @ a_k
        if denom <= 0:
            continue
        # projection of p onto hyperplane a_k . x = b_k
        p_hat = p + a_k * (b_k - a_k @ p) / denom
        # must remain inside the hull (all other facets satisfied)
        if np.all(A @ p_hat <= b + tol):
            d = float(np.linalg.norm(p_hat - p))
            if d < best_d:
                best_d = d
                best_p = p_hat
    if best_p is None:
        return None, float("inf")
    return best_p, best_d


# --------------------------------------------------------------------------- #
# The MILP
# --------------------------------------------------------------------------- #
def _solve_milp(p_solve: np.ndarray, P: np.ndarray, big_bound: float = 1e3):
    """Solve the SP(p) MILP at ``p_solve`` using highspy.

    Returns a dict {feasible, r, alpha, lam, w} where r/alpha/lam/w are None
    if infeasible.

    Variable order in the flat vector x:
        alpha[0..n-1], r[0..m-1], lam, z_plus[0..m-1], z_minus[0..m-1], w
    """
    import highspy

    P = np.asarray(P, dtype=float)
    p_solve = np.asarray(p_solve, dtype=float).ravel()
    m, n = P.shape

    # index offsets in the flat variable vector
    o_alpha = 0
    o_r = o_alpha + n
    o_lam = o_r + m
    o_zp = o_lam + 1
    o_zm = o_zp + m
    o_w = o_zm + m
    N = o_w + 1  # total number of variables

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    # Deterministic MILP: single thread + fixed seed remove run-to-run
    # nondeterminism in tie-breaking (which optimal r is returned).
    h.setOptionValue("threads", 1)
    h.setOptionValue("random_seed", MILP_RANDOM_SEED)

    inf = highspy.kHighsInf

    # ---- add variables with bounds and integrality ----
    lb = np.empty(N)
    ub = np.empty(N)
    # alpha >= 0
    lb[o_alpha:o_alpha + n] = 0.0
    ub[o_alpha:o_alpha + n] = inf
    # r free (bounded by w through 1.9, and by big_bound for numerics)
    lb[o_r:o_r + m] = -big_bound
    ub[o_r:o_r + m] = big_bound
    # lambda free
    lb[o_lam] = -big_bound
    ub[o_lam] = big_bound
    # z_plus, z_minus binary
    lb[o_zp:o_zp + m] = 0.0
    ub[o_zp:o_zp + m] = 1.0
    lb[o_zm:o_zm + m] = 0.0
    ub[o_zm:o_zm + m] = 1.0
    # w >= 0
    lb[o_w] = 0.0
    ub[o_w] = big_bound

    # objective: min w
    cost = np.zeros(N)
    cost[o_w] = 1.0

    h.addVars(N, lb, ub)
    # integrality: 1 = integer, 0 = continuous
    integrality = np.zeros(N, dtype=np.int32)
    integrality[o_zp:o_zp + m] = 1
    integrality[o_zm:o_zm + m] = 1
    h.changeColsIntegrality(N, np.arange(N, dtype=np.int32), integrality)
    h.changeColsCost(N, np.arange(N, dtype=np.int32), cost)

    # ---- constraints, added as sparse rows ----
    # helper to push a single row a . x  (op)  rhs
    rows_start = []
    rows_index = []
    rows_value = []
    rows_lower = []
    rows_upper = []

    def add_row(coeffs: dict, lo: float, hi: float):
        rows_start.append(len(rows_index))
        for j, v in coeffs.items():
            if v != 0.0:
                rows_index.append(int(j))
                rows_value.append(float(v))
        rows_lower.append(lo)
        rows_upper.append(hi)

    # (1.1) P alpha = p    -> m equality rows
    for i in range(m):
        coeffs = {o_alpha + j: P[i, j] for j in range(n)}
        add_row(coeffs, p_solve[i], p_solve[i])

    # (1.2a) 1' alpha = 1
    add_row({o_alpha + j: 1.0 for j in range(n)}, 1.0, 1.0)
    # (1.2b) alpha >= 0 already in bounds

    # (1.3) lambda*1 <= r'P   ->  for each sample j:  sum_i r_i P[i,j] - lambda >= 0
    for j in range(n):
        coeffs = {o_r + i: P[i, j] for i in range(m)}
        coeffs[o_lam] = -1.0
        add_row(coeffs, 0.0, inf)

    # (1.4) r'p = lambda   ->  sum_i r_i p_i - lambda = 0
    coeffs = {o_r + i: p_solve[i] for i in range(m)}
    coeffs[o_lam] = -1.0
    add_row(coeffs, 0.0, 0.0)

    # Big-M for the sign disjunction: tightest valid M given |r_i| <= big_bound
    M = big_bound + 1.0

    # (1.5) r_i >= 1 - M(1 - z_plus_i)  ->  r_i - M z_plus_i >= 1 - M
    for i in range(m):
        add_row({o_r + i: 1.0, o_zp + i: -M}, 1.0 - M, inf)

    # (1.6) r_i <= -1 + M(1 - z_minus_i)  ->  r_i + M z_minus_i <= M - 1
    for i in range(m):
        add_row({o_r + i: 1.0, o_zm + i: M}, -inf, M - 1.0)

    # (1.7b) z_plus_i + z_minus_i <= 1  (a coordinate cannot demand both signs)
    for i in range(m):
        add_row({o_zp + i: 1.0, o_zm + i: 1.0}, -inf, 1.0)

    # (1.8) 1'z_plus + 1'z_minus >= 1
    coeffs = {}
    for i in range(m):
        coeffs[o_zp + i] = 1.0
        coeffs[o_zm + i] = 1.0
    add_row(coeffs, 1.0, inf)

    # (1.9) -w <= r_i <= w   ->  r_i - w <= 0  and  r_i + w >= 0
    for i in range(m):
        add_row({o_r + i: 1.0, o_w: -1.0}, -inf, 0.0)
        add_row({o_r + i: 1.0, o_w: 1.0}, 0.0, inf)

    num_rows = len(rows_lower)
    rows_start_arr = np.array(rows_start, dtype=np.int32)
    rows_index_arr = np.array(rows_index, dtype=np.int32)
    rows_value_arr = np.array(rows_value, dtype=float)
    rows_lower_arr = np.array(rows_lower, dtype=float)
    rows_upper_arr = np.array(rows_upper, dtype=float)

    h.addRows(num_rows, rows_lower_arr, rows_upper_arr,
              len(rows_index_arr), rows_start_arr, rows_index_arr, rows_value_arr)

    h.run()

    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        return {"feasible": False, "r": None, "alpha": None, "lam": None, "w": None,
                "model_status": h.modelStatusToString(status)}

    sol = h.getSolution()
    x = np.array(sol.col_value)
    alpha = x[o_alpha:o_alpha + n]
    r = x[o_r:o_r + m]
    lam = float(x[o_lam])
    w = float(x[o_w])
    return {"feasible": True, "r": r, "alpha": alpha, "lam": lam, "w": w,
            "model_status": "Optimal"}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def solve_surrogate_costs(
    p: np.ndarray,
    P: np.ndarray,
    *,
    tol_interior: Optional[float] = None,
    normalize: bool = True,
) -> SurrogateResult:
    """Recover a surrogate cost vector r for the point p over samples P.

    Parameters
    ----------
    p : (m,) array
        The PoI vector for the point of interest (e.g. a navigation target).
    P : (m, n) array
        Sample matrix; columns are samples in PoI space.
    tol_interior : float, optional
        Points with signed boundary distance above this are treated as
        interior and projected. Defaults to 1e-4 * (smallest PoI range),
        which also absorbs the ~1e-6 navigation lock-in offset.
    normalize : bool
        If True (default), return r rescaled to ||r||_1 = 1. The MILP
        natively produces the ||r||_inf = 1 representative; both lie on the
        same solution ray, so rescaling is exact.

    Returns
    -------
    SurrogateResult
    """
    p = np.asarray(p, dtype=float).ravel()
    P = np.asarray(P, dtype=float)
    m, n = P.shape

    if p.shape[0] != m:
        raise ValueError(f"p has length {p.shape[0]} but P has {m} rows")

    # default interior tolerance: relative to smallest PoI extent
    if tol_interior is None:
        ranges = P.max(axis=1) - P.min(axis=1)
        smallest = float(np.min(ranges[ranges > 0])) if np.any(ranges > 0) else 1.0
        tol_interior = 1e-4 * smallest

    dist = signed_boundary_distance(p, P)

    projected = False
    proj_dist = 0.0
    p_solve = p

    if dist > tol_interior:
        p_hat, proj_dist = project_to_boundary(p, P)
        if p_hat is None:
            return SurrogateResult(
                r=None, status="failed", projected=False, p_used=None,
                projection_distance=0.0, w=None,
                message=(f"Point is interior (distance {dist:.3g}) and could "
                         f"not be projected onto the boundary."),
            )
        projected = True
        p_solve = p_hat

    result = _solve_milp(p_solve, P)
    nudged = False

    if not result["feasible"] and not projected:
        # Boundary-classified point with no exact supporting hyperplane:
        # typical for numerically eps-interior points (the ~1e-6 navigation
        # lock-in offset). Nudge onto the hull surface and retry once.
        p_hat, d_hat = project_to_boundary(p, P)
        if p_hat is not None:
            retry = _solve_milp(p_hat, P)
            if retry["feasible"]:
                projected = True
                nudged = True
                proj_dist = d_hat
                p_solve = p_hat
                result = retry

    if not result["feasible"]:
        hs = result.get("model_status", "unknown")
        if projected:
            msg = (f"MILP infeasible at the projected boundary point "
                   f"(HiGHS status: {hs}). This is an expected minority "
                   f"outcome (e.g. a supporting hyperplane through the "
                   f"origin, or a low-dimensional face).")
        else:
            msg = (f"Boundary point but MILP infeasible (HiGHS status: {hs}; "
                   f"unexpected -- check for an origin facet or a "
                   f"low-dimensional face).")
        return SurrogateResult(
            r=None, status="failed", projected=projected, p_used=p_solve,
            projection_distance=proj_dist, w=None, message=msg,
        )

    r_raw = result["r"]
    n1 = float(np.sum(np.abs(r_raw)))
    r_out = r_raw / n1 if (normalize and n1 > 0) else r_raw

    status = "projected" if projected else "boundary"
    if nudged:
        msg = (f"Point was numerically eps-interior (boundary distance "
               f"{dist:.3g}, e.g. navigation lock-in offset); nudged "
               f"{proj_dist:.3g} onto the hull and solved.")
    elif projected:
        msg = (f"Solved at projected boundary point "
               f"(moved {proj_dist:.3g} from the interior point).")
    else:
        msg = "Solved at the point (on the boundary within tolerance)."

    return SurrogateResult(
        r=r_out, status=status, projected=projected, p_used=p_solve,
        projection_distance=proj_dist, w=result["w"], message=msg,
        extras={"r_raw": r_raw, "alpha": result["alpha"], "lambda": result["lam"],
                "l1_norm_raw": n1, "boundary_distance": dist},
    )