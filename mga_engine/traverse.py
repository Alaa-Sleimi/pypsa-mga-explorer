"""
traverse.py — Traverse Algorithms (Algorithms 2 and 3 from Sina's paper).

Algorithm 2: Interpolation Points Extraction
    Given alpha_s and alpha_t, finds breakpoints SI along the linear
    interpolation path in F^P where the cost approximation improves
    significantly over the simple linear interpolation.

Algorithm 3: Subgradient Generation
    Given SI, generates subgradient functions of phi(p) at the
    breakpoints, providing a piecewise-linear lower bound on the
    true cost along the interpolation path.

APoI convention matches navigation.py:
    index 0 = cost, index 1..m = PoI values

Run with:
    python -m mga_engine.traverse
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
    """
    Algorithm 2: Interpolation Points Extraction.

    Parameters
    ----------
    P_all     : np.ndarray, shape (m, n)  — PoI sample matrix
    v         : np.ndarray, shape (n,)    — min cost at each sample
    alpha_s   : np.ndarray, shape (n,)    — start weights
    alpha_t   : np.ndarray, shape (n,)    — target weights
    epsilon_I : float                     — improvement tolerance (€/yr)

    Returns
    -------
    SI : list of (beta, alpha*) tuples, sorted by beta
         beta  : float in (0, 1)  — position along interpolation path
         alpha* : np.ndarray      — CCP optimal weights at that point
    """
    SI = []

    def mid_interpolate(beta_l, beta_u, alpha_l, alpha_u):
        # Midpoint weights via linear interpolation
        alpha_m = (alpha_l + alpha_u) / 2.0
        p_m     = P_all @ alpha_m

        # Solve CCP at midpoint to get tighter cost estimate
        try:
            alpha_star, _ = solve_ccp(p_m, P_all, v)
        except RuntimeError:
            # If CCP fails at this midpoint, skip this interval
            return

        cost_ccp    = float(v @ alpha_star)   # v'α*  — tighter estimate
        cost_linear = float(v @ alpha_m)      # v'αm  — linear estimate

        # Paper condition (corrected): v'α* ≤ v'αm - εI
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
    """
    Algorithm 3: Subgradient Generation.

    For each breakpoint in SI, identifies which sample points have
    nonzero weights (alpha_i > 0) and generates a subgradient function
    g(beta) = v_i + (P((1-beta)*alpha_s + beta*alpha_t) - p_i)' * gamma_i

    Also adds the constant subgradient g(beta) = c'x* = opt_cost.

    Parameters
    ----------
    Gamma    : np.ndarray, shape (m, n)  — dual variables from GP solves
    P_all    : np.ndarray, shape (m, n)  — PoI sample matrix
    v        : np.ndarray, shape (n,)    — min cost at each sample
    opt_cost : float                     — optimal cost c'x*
    alpha_s  : np.ndarray, shape (n,)   — start weights
    alpha_t  : np.ndarray, shape (n,)   — target weights
    SI       : list of (beta, alpha*)   — breakpoints from Algorithm 2

    Returns
    -------
    SS : list of callables, each g(beta) -> float
         These are the subgradient functions evaluated along the path.
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
            def g(beta):
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
    """
    Evaluate the interpolated point and linear cost estimate at beta.

    Returns
    -------
    p_beta    : np.ndarray, shape (m,)  — PoI values at beta
    cost_beta : float                   — linear cost estimate alpha(beta)'v
    """
    alpha_beta = (1 - beta) * alpha_s + beta * alpha_t
    p_beta     = P_all @ alpha_beta
    cost_beta  = float(v @ alpha_beta)
    return p_beta, cost_beta