"""Navigation: turn a user's preference into the weights of a target solution.

Second step of the exploration phase (Algorithm 1). Given the current solution's
weights alpha_s and a preference over the APoIs - a priority order, which ones to
increase, decrease or hold, and by how much - it returns the weights alpha_t of the
solution to move to, which :mod:`mga_engine.traverse` then walks towards. Everything
happens as small LPs over the stored samples; the network is never touched.

APoI index convention, also used by :mod:`mga_engine.alpha_projection`: index 0 is
the cost, indices 1..m are the PoI values in P_all row order.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple

# Absolute lock-in window for improved PoIs (Loop 2).
# LOAD-BEARING: surrogate.py's tol_interior is sized to absorb this offset
# (navigation leaves locked boundary points ~1e-6 inside the hull; the
# surrogate's project-and-retry compensates). Do not change this value
# without re-validating surrogate.py together.
LOCK_IN_TOL = 1e-6


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
    """Compute target solution weights from the current ones and a user preference.

    Algorithm 1, in three stages over the priority order `tau`. Loop 1 checks each
    APoI's requested direction against what the sample set actually allows, shrinks
    `delta` to the achievable amount (or flips the APoI into SE when its requested
    direction is impossible), and constrains it to the resulting interval. Loop 2
    then pushes each APoI as far in its direction as the constraints accumulated so
    far permit and locks the achieved value in. A final LP minimises cost subject to
    all of them, so a preference that leaves room is spent on the cheapest solution
    satisfying it.

    Parameters
    ----------
    P_all : numpy.ndarray
        Shape ``(m, n)``. Sample matrix, one PoI point in GW per column.
    v : numpy.ndarray
        Shape ``(n,)``. Minimum cost at each sample, in EUR/yr; APoI row 0.
    alpha_s : numpy.ndarray
        Shape ``(n,)``. Weights of the current solution. Expected to be on the
        simplex (non-negative, summing to 1); this is not validated.
    tau : list of int
        APoI indices in priority order, most important first. An index in none of
        `SG`, `SL`, `SE` is skipped by both loops.
    SG, SL, SE : set of int
        APoI indices to increase, to decrease, and to hold at their current value.
        Expected to be disjoint; this is not validated.
    delta : numpy.ndarray
        Shape ``(m + 1,)``. Desired change per APoI index, as an absolute amount in
        that APoI's own unit: EUR/yr for index 0, GW for the PoIs. Must be an array,
        since ``.copy()`` is called on it.

    Returns
    -------
    alpha_t : numpy.ndarray
        Shape ``(n,)``. Weights of the target solution.
    SG_out, SL_out, SE_out : set of int
        The direction sets after Loop 1's feasibility flips, so the caller can show
        which requests survived.
    delta_out : numpy.ndarray
        Shape ``(m + 1,)``. `delta` reduced to the magnitudes actually achievable.
    loop1_bounds : list of tuple
        One ``(apoi_index, lower, upper)`` per `tau` entry that ended up in a
        direction set, with ``None`` on an unbounded side: the interval Loop 1 fixed
        for that APoI. Accepted directly as the ``bounds`` argument of
        :func:`mga_engine.alpha_projection.compute_alpha_hull_2d`.

    Raises
    ------
    RuntimeError
        If an LP in Loop 1 or the final solve fails, e.g. because the requested
        preferences together are infeasible over the samples.

    Notes
    -----
    `SG`, `SL`, `SE` and `delta` are copied before use, so the caller's objects are
    never mutated; the returned ones are the updated copies.
    A Loop-2 LP that fails is caught, reported with a printed warning and skipped:
    that APoI then keeps the interval Loop 1 gave it instead of a locked value, and
    the call still returns normally.
    Loop 2 locks each achieved value into a window of ``LOCK_IN_TOL``, which leaves
    `alpha_t` marginally off the exact hull boundary; see that constant's comment,
    since :mod:`mga_engine.surrogate` is calibrated against it.
    Roughly ``2 * len(tau) + 1`` SciPy LPs, more where SE is involved. No network and
    no file access.
    """
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
        """Solve the NCP with the constraints accumulated so far.

        Parameters
        ----------
        c_obj : numpy.ndarray
            Shape ``(n,)``. Linear objective over the weights. Pass a negated
            P_tilde row to maximise that APoI instead of minimising it.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``. Optimal weights: non-negative, summing to 1, and
            satisfying every row added by `add_ineq` so far.

        Raises
        ------
        RuntimeError
            If SciPy's status is not 0, i.e. infeasible, unbounded, or a solver
            failure.
        """
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
        """AddConst: append one row ``row @ alpha <= rhs`` to the NCP.

        Parameters
        ----------
        row : numpy.ndarray
            Shape ``(n,)``. Coefficients over the weights. Negate both `row` and
            `rhs` to express a lower bound.
        rhs : float
            Right-hand side, in the unit of the APoI the row came from.

        Returns
        -------
        None
            Appends to the enclosing ``A_ub_rows`` and ``b_ub_rows``, so every later
            solve in this call carries the new row.
        """
        A_ub_rows.append(row)
        b_ub_rows.append(rhs)

    def solve_ncp_absdev(row, ref):
        """Solve the NCP minimising the absolute deviation ``|row @ alpha - ref|``.

        The SE case of Algorithm 1, linearised with one auxiliary variable ``t >= 0``
        bounded by ``t >= row @ alpha - ref`` and ``t >= ref - row @ alpha``, so that
        minimising t minimises the deviation.

        Parameters
        ----------
        row : numpy.ndarray
            Shape ``(n,)``. The APoI row to hold near `ref`.
        ref : float
            Value to stay as close to as possible, in that APoI's unit.

        Returns
        -------
        alpha : numpy.ndarray
            Shape ``(n,)``. Optimal weights, with the auxiliary variable dropped.
        deviation : float
            The achieved ``t``, i.e. the smallest reachable
            ``|row @ alpha - ref|``. Zero when the value can be held exactly.

        Raises
        ------
        RuntimeError
            If SciPy's status is not 0.
        """
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
                # Cannot decrease - flip to SE
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
                # Cannot increase - flip to SE
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

     # --- Loop 2: Improvement - push each priority as far as possible
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
            add_ineq( row_i, val_t + LOCK_IN_TOL)
            add_ineq(-row_i, -(val_t - LOCK_IN_TOL))

        except RuntimeError:
            # If pushing this priority further is infeasible, skip it
            print(f"[nav] Warning: improvement for APoI {i} infeasible, skipping.")
            continue

    # --- Final solve: minimise cost subject to all accumulated constraints
    alpha_t = solve_ncp(v)

    return alpha_t, SG, SL, SE, delta, loop1_bounds


