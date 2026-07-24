"""
navigation.py — Navigation Algorithm (Algorithm 1 from Sina's paper).

Given the current solution weights alpha_s and user preferences
(priority order tau, direction sets SE/SG/SL, magnitude delta),
finds the target weights alpha_t representing the new solution.

APoI index convention:
    0        = cost (objective function value)
    1, 2, .. = PoI values (e.g. solar=1, wind=2)

Inputs:
    P_all   : np.ndarray, shape (m, n)   — PoI sample matrix
    v       : np.ndarray, shape (n,)     — min cost at each sample
    alpha_s : np.ndarray, shape (n,)     — current solution weights
    tau     : list[int]                  — priority order over APoI indices
    SG      : set[int]                   — APoI indices to increase
    SL      : set[int]                   — APoI indices to decrease
    SE      : set[int]                   — APoI indices to keep equal
    delta   : np.ndarray, shape (m+1,)  — desired magnitude per APoI index

Outputs:
    alpha_t : np.ndarray, shape (n,)     — target solution weights
    SG_out  : set[int]                   — updated direction sets (feedback)
    SL_out  : set[int]
    SE_out  : set[int]
    delta_out : np.ndarray               — updated magnitudes (feedback)
    loop1_bounds : list[tuple]           — per-APoI interval bounds fixed by the
                   first (feasibility) loop, as (apoi_index, lower, upper) with
                   None where unbounded; APoIs in no direction set get no entry
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple


def navigate(
    P_all: np.ndarray,
    v: np.ndarray,
    alpha_s: np.ndarray,
    tau: list,
    SG: set,
    SL: set,
    SE: set,
    delta: np.ndarray,
) -> Tuple[np.ndarray, set, set, set, np.ndarray, list]:

    m, n = P_all.shape

    # --- Step 1: Build P_tilde = [v' ; P], shape (m+1, n)
    # Row 0 is cost, rows 1..m are PoI values
    P_tilde = np.vstack([v[np.newaxis, :], P_all])   # shape (m+1, n)

    # --- Step 2: Current APoI values p_tilde_s
    p_tilde_s = P_tilde @ alpha_s                     # shape (m+1,)

    # Work on mutable copies of direction sets and delta
    SG = set(SG)
    SL = set(SL)
    SE = set(SE)
    delta = delta.copy()

    # --- NCP state: accumulates equality constraints as we go
    # NCP: min 0  s.t.  alpha >= 0, sum(alpha) = 1, [accumulated constraints]
    # We implement NCP as a linprog problem that we rebuild at each solve.
    # A_ub @ alpha <= b_ub holds the inequality constraints added by AddConst.
    A_ub_rows = []
    b_ub_rows = []

    # Base equality: sum(alpha) = 1
    # linprog A_eq alpha = b_eq
    A_eq_base = np.ones((1, n))
    b_eq_base = np.array([1.0])

    def solve_ncp(c_obj):
        """Solve NCP with current constraints and objective c_obj."""
        A_ub = np.array(A_ub_rows).reshape(-1, n) if A_ub_rows else None
        b_ub = np.array(b_ub_rows) if b_ub_rows else None
        bounds = [(0.0, None)] * n
        result = linprog(
            c=c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq_base,
            b_eq=b_eq_base,
            bounds=bounds,
            method="highs",
        )
        if result.status != 0:
            raise RuntimeError(f"NCP solve failed: {result.message}")
        return result.x

    def add_ineq(row, rhs):
        """AddConst: add one row to the inequality system."""
        A_ub_rows.append(row)
        b_ub_rows.append(rhs)

    def solve_ncp_absdev(row, ref):
        """Solve NCP with objective min |row @ alpha - ref| (paper Algorithm 1,
        SE case), linearized with one auxiliary variable t >= 0 and
        t >= row @ alpha - ref, t >= ref - row @ alpha.
        Returns (alpha, achieved deviation t)."""
        n_rows = len(A_ub_rows)
        A_ub_alpha = np.array(A_ub_rows).reshape(n_rows, n)
        A_ub = np.vstack([
            np.hstack([A_ub_alpha, np.zeros((n_rows, 1))]),
            np.append(row, -1.0),    #  row @ alpha - t <= ref
            np.append(-row, -1.0),   # -row @ alpha - t <= -ref
        ])
        b_ub = np.array([*b_ub_rows, ref, -ref])
        A_eq = np.hstack([A_eq_base, np.zeros((1, 1))])
        c = np.zeros(n + 1)
        c[-1] = 1.0
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq_base,
            bounds=[(0.0, None)] * (n + 1),
            method="highs",
        )
        if result.status != 0:
            raise RuntimeError(f"NCP solve failed: {result.message}")
        return result.x[:n], float(result.x[n])

    # --- Loop 1: Feasibility check and delta/direction update
    for i in tau:
        row_i = P_tilde[i]          # shape (n,), the i-th APoI row
        ps_i  = p_tilde_s[i]        # scalar current value

        if i in SE:
            # Minimize |P_tilde[i] @ alpha - ps_i|
            _, dev = solve_ncp_absdev(row_i, ps_i)
            delta[i] = max(dev, delta[i])

        elif i in SL:
            alpha_star = solve_ncp(row_i)         # minimise i-th APoI
            val = P_tilde[i] @ alpha_star
            if ps_i > val:
                # Feasible decrease exists
                delta[i] = min(ps_i - val, delta[i])
            else:
                # Cannot decrease — flip to SE
                delta[i] = val - ps_i
                SL.discard(i)
                SE.add(i)

        elif i in SG:
            alpha_star = solve_ncp(-row_i)        # maximise i-th APoI
            val = P_tilde[i] @ alpha_star
            if val > ps_i:
                # Feasible increase exists
                delta[i] = min(val - ps_i, delta[i])
            else:
                # Cannot increase — flip to SE
                delta[i] = ps_i - val
                SG.discard(i)
                SE.add(i)

        # AddConst based on (possibly updated) direction
        if i in SE:
            add_ineq( row_i, ps_i + delta[i])    #  P_tilde[i] alpha <= ps_i + delta
            add_ineq(-row_i, -(ps_i - delta[i])) # -P_tilde[i] alpha <= -(ps_i - delta)
        elif i in SL:
            add_ineq(row_i, ps_i - delta[i])     #  P_tilde[i] alpha <= ps_i - delta
        elif i in SG:
            add_ineq(-row_i, -(ps_i + delta[i])) # -P_tilde[i] alpha <= -(ps_i + delta)

    # --- Snapshot loop-1 bounds: the per-APoI intervals AddConst fixed above,
    # in (apoi_index, lower, upper) form with None where unbounded
    loop1_bounds = []
    for i in tau:
        ps_i = float(p_tilde_s[i])
        if i in SE:
            loop1_bounds.append((i, ps_i - float(delta[i]), ps_i + float(delta[i])))
        elif i in SL:
            loop1_bounds.append((i, None, ps_i - float(delta[i])))
        elif i in SG:
            loop1_bounds.append((i, ps_i + float(delta[i]), None))

     # --- Loop 2: Improvement — push each priority as far as possible
    for i in tau:
        row_i = P_tilde[i]
        ps_i  = p_tilde_s[i]

        try:
            if i in SE:
                alpha_star, _ = solve_ncp_absdev(row_i, ps_i)  # minimise |value - ps_i|
            elif i in SL:
                alpha_star = solve_ncp(row_i)      # minimise value = maximise decrease
            elif i in SG:
                alpha_star = solve_ncp(-row_i)     # maximise value = maximise increase
            else:
                continue

            val_t = P_tilde[i] @ alpha_star
            # Lock in this APoI value with a small tolerance
            add_ineq( row_i, val_t + 1e-6)
            add_ineq(-row_i, -(val_t - 1e-6))

        except RuntimeError:
            # If pushing this priority further is infeasible, skip it
            print(f"[nav] Warning: improvement for APoI {i} infeasible, skipping.")
            continue

    # --- Final solve: minimise cost subject to all accumulated constraints
    alpha_t = solve_ncp(v)

    return alpha_t, SG, SL, SE, delta, loop1_bounds


