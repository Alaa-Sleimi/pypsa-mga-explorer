"""Traverse algorithms: the cost profile along the path to a chosen solution.

Third step of the exploration phase, once :mod:`mga_engine.navigation` has turned a
preference into the target weights alpha_t.

Algorithm 2 (:func:`extract_breakpoints`) walks the interpolation path from alpha_s
to alpha_t, which is linear in the PoI coordinates, and keeps the positions where
CCP prices the path point well below the linear interpolation; a breakpoint's cost is
that CCP-minimised value, not the interpolated one.

Algorithm 3 (:func:`generate_subgradients`) turns those breakpoints into subgradient
functions of the true cost, a piecewise-linear LOWER bound along the same path. The
notebook plots both against the linear estimate from :func:`interpolate`.
"""

import numpy as np
from typing import List, Tuple
from mga_engine.ccp_solver import solve_ccp


# ---------------------------------------------------------------------------
# Algorithm 2: Interpolation Points Extraction
# ---------------------------------------------------------------------------

def extract_breakpoints(
    P_all: np.ndarray,
    v: np.ndarray,
    alpha_s: np.ndarray,
    alpha_t: np.ndarray,
    epsilon_I: float = 1e6,
) -> List[Tuple[float, np.ndarray]]:
    """Algorithm 2: extract the interpolation breakpoints between alpha_s and alpha_t.

    Bisects the path recursively. At an interval's midpoint the linearly interpolated
    weights are repriced with CCP; if that is cheaper than the interpolation by at
    least `epsilon_I`, the midpoint becomes a breakpoint and both halves are bisected
    in turn, with the CCP weights as their shared endpoint.

    Parameters
    ----------
    P_all : numpy.ndarray
        Shape ``(m, n)``. Sample matrix, one PoI point in GW per column.
    v : numpy.ndarray
        Shape ``(n,)``. Minimum cost at each sample, in EUR/yr.
    alpha_s : numpy.ndarray
        Shape ``(n,)``. Weights of the current solution, the start of the path.
    alpha_t : numpy.ndarray
        Shape ``(n,)``. Weights of the target solution, from
        :func:`mga_engine.navigation.navigate`.
    epsilon_I : float, default 1e6
        Improvement tolerance, in ABSOLUTE cost units (EUR/yr) rather than a
        fraction, so it has to be rescaled for a network of a different size.

    Returns
    -------
    list of (float, numpy.ndarray)
        The breakpoints ``(beta, alpha_star)``, sorted by ``beta``: the position in
        ``(0, 1)`` along the path, and the CCP weights of shape ``(n,)`` there. Empty
        when the linear interpolation is never beaten by `epsilon_I`.

    Notes
    -----
    Recursion is bounded only by `epsilon_I`. The midpoint weights are always
    CCP-feasible, so with ``epsilon_I <= 0`` the condition holds at essentially every
    midpoint and the bisection can run until Python's recursion limit; this is not
    checked.
    A CCP failure inside an interval is caught and that interval is silently dropped,
    so a truncated breakpoint list looks exactly like a legitimately empty one.
    One SciPy LP per visited interval; no network and no file access.
    """
    SI = []

    def mid_interpolate(beta_l, beta_u, alpha_l, alpha_u):
        """Test one interval's midpoint and recurse into both halves if it is kept.

        Parameters
        ----------
        beta_l, beta_u : float
            Lower and upper position of the interval along the path.
        alpha_l, alpha_u : numpy.ndarray
            Shape ``(n,)``. Weights at `beta_l` and `beta_u`.

        Returns
        -------
        None
            Appends the accepted breakpoint to the enclosing ``SI`` list. Returns
            early, keeping nothing, if CCP fails at the midpoint.
        """
        # Midpoint weights via linear interpolation
        alpha_m = (alpha_l + alpha_u) / 2.0
        p_m     = P_all @ alpha_m

        # Solve CCP at midpoint to get tighter cost estimate
        try:
            alpha_star, _ = solve_ccp(p_m, P_all, v)
        except RuntimeError:
            # If CCP fails at this midpoint, skip this interval
            return

        cost_ccp    = float(v @ alpha_star)   # v'alpha*  - tighter estimate
        cost_linear = float(v @ alpha_m)      # v'alpha_m - linear estimate

        # Paper condition (corrected): v'alpha* <= v'alpha_m - epsilon_I
        if cost_ccp <= cost_linear - epsilon_I:
            beta_m = (beta_l + beta_u) / 2.0
            SI.append((beta_m, alpha_star))
            # Recurse into both subintervals
            mid_interpolate(beta_l, beta_m, alpha_l, alpha_star)
            mid_interpolate(beta_m, beta_u, alpha_star, alpha_u)

    mid_interpolate(0.0, 1.0, alpha_s, alpha_t)

    # Sort breakpoints by beta
    SI.sort(key=lambda x: x[0])
    return SI


# ---------------------------------------------------------------------------
# Algorithm 3: Subgradient Generation
# ---------------------------------------------------------------------------

def generate_subgradients(
    Gamma: np.ndarray,
    P_all: np.ndarray,
    v: np.ndarray,
    opt_cost: float,
    alpha_s: np.ndarray,
    alpha_t: np.ndarray,
    SI: List[Tuple[float, np.ndarray]],
) -> list:
    """Algorithm 3: build the subgradient lower bound of the cost along the path.

    Every sample that carries a non-zero weight at some breakpoint contributes one
    affine function of the path parameter,

        g(beta) = v_i + (P((1 - beta) alpha_s + beta alpha_t) - p_i) . gamma_i,

    and the constant ``g(beta) = opt_cost`` is appended. Their pointwise maximum is
    the lower envelope of the true cost that the notebook plots.

    Parameters
    ----------
    Gamma : numpy.ndarray
        Shape ``(m, n)``. Column i holds the PoI duals gamma_i at sample i, in EUR/yr
        per GW, as returned by :func:`mga_engine.gp_solver.solve_all_gp`.
    P_all : numpy.ndarray
        Shape ``(m, n)``. Sample matrix, one PoI point in GW per column.
    v : numpy.ndarray
        Shape ``(n,)``. Minimum cost at each sample, in EUR/yr.
    opt_cost : float
        Unconstrained minimum system cost c'x* in EUR/yr, which bounds the cost
        everywhere along the path from below.
    alpha_s, alpha_t : numpy.ndarray
        Shape ``(n,)``. Start and target weights defining the path.
    SI : list of (float, numpy.ndarray)
        Breakpoints from :func:`extract_breakpoints`. Only which weights are non-zero
        is used here; the beta values are ignored.

    Returns
    -------
    list of callable
        Each entry maps ``beta -> float``. The active samples come first, in set
        iteration order, and the constant bound last. With `SI` empty the list holds
        only that constant, i.e. a flat bound at `opt_cost`.

    Notes
    -----
    Sign convention: gamma_i, the dual of the constraint pinning PoI i in the GP, is
    un-negated, i.e. d(cost)/d(p_i) in EUR/yr per GW, so a positive value means that
    raising that PoI's target raises the minimum system cost. The GP value function
    v(p) is convex, so gamma_i is a subgradient of it at p_i and
    ``v(p) >= v_i + (p - p_i) . gamma_i`` holds for every p. Each g above is exactly
    that inequality evaluated along the path, which is why it is a valid LOWER bound
    on the true cost. Verified for the pinned linopy/highspy versions on the HiGHS
    minimisation path.

    A sample counts as active when ``alpha[i] > 0`` strictly, so numerically tiny
    weights contribute too. The returned closures hold `P_all`, `alpha_s` and
    `alpha_t` by reference, so mutating those arrays in place afterwards changes what
    the functions return.
    """
    SS = []
    I  = set()

    # Collect all sample indices with nonzero weight across all breakpoints
    for (beta, alpha) in SI:
        for i in range(len(alpha)):
            if alpha[i] > 0:
                I.add(i)

    # Generate one subgradient function per active sample index
    for i in I:
        p_i     = P_all[:, i]       # shape (m,)
        gamma_i = Gamma[:, i]       # shape (m,)
        v_i     = v[i]              # scalar

        # Capture i by value using default argument
        def make_g(p_i=p_i, gamma_i=gamma_i, v_i=v_i):
            """Bind one sample's ``(p_i, gamma_i, v_i)`` by value and return its g.

            The three defaults are the capture mechanism, not real parameters: they
            freeze the current loop values so every closure keeps its own sample.

            Returns
            -------
            callable
                The subgradient ``g(beta) -> float`` for that sample.
            """
            def g(beta):
                """Evaluate this sample's lower bound at path position `beta`."""
                p_beta = P_all @ ((1 - beta) * alpha_s + beta * alpha_t)
                return v_i + float((p_beta - p_i) @ gamma_i)
            return g

        SS.append(make_g())

    # Add the constant subgradient g(beta) = c'x* = opt_cost
    SS.append(lambda beta: opt_cost)

    return SS


# ---------------------------------------------------------------------------
# Convenience: evaluate path at a given beta
# ---------------------------------------------------------------------------

def interpolate(
    P_all: np.ndarray,
    v: np.ndarray,
    alpha_s: np.ndarray,
    alpha_t: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, float]:
    """Evaluate the interpolated point and its linear cost estimate at `beta`.

    Parameters
    ----------
    P_all : numpy.ndarray
        Shape ``(m, n)``. Sample matrix, one PoI point in GW per column.
    v : numpy.ndarray
        Shape ``(n,)``. Minimum cost at each sample, in EUR/yr.
    alpha_s, alpha_t : numpy.ndarray
        Shape ``(n,)``. Start and target weights defining the path.
    beta : float
        Position along the path. Normally in ``[0, 1]``; the range is not checked,
        and outside it the result extrapolates past the two endpoints.

    Returns
    -------
    p_beta : numpy.ndarray
        Shape ``(m,)``. PoI values in GW at `beta`.
    cost_beta : float
        Linear cost estimate ``v . alpha(beta)`` in EUR/yr. Being a convex
        combination of sample costs, it is an UPPER bound on the true cost;
        :func:`extract_breakpoints` prices the same point more tightly with CCP.

    Notes
    -----
    Pure NumPy arithmetic: no solve, no side effects.
    """
    alpha_beta = (1 - beta) * alpha_s + beta * alpha_t
    p_beta     = P_all @ alpha_beta
    cost_beta  = float(v @ alpha_beta)
    return p_beta, cost_beta