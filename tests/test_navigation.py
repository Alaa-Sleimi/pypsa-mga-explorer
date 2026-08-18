"""
tests/test_navigation.py — Test the navigation algorithm using saved preparation state.

Loads P_all, v, p_star from data/preparation_state.nc and runs three
navigation scenarios without re-running the preparation phase.

Scenarios:
    A — solar↑ 0.3 GW, wind↓ 0.3 GW  (both feasible)
    B — solar↑ 3.0 GW (infeasible → capped), wind↓ 0.3 GW
    C — solar↑ 0.3 GW, wind↓ 5.0 GW  (infeasible → capped)

Run with:
    python -m tests.test_navigation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from mga_engine.preparation_state import load
from mga_engine.ccp_solver import solve_ccp
from mga_engine.navigation import navigate

# --- Load preparation state ---
state  = load("data/preparation_state.nc")
P_all  = state["P_all"]
v      = state["v"]
p_star = state["p_star"]

# --- Starting point: CCP at p_star ---
alpha_s, cost_s = solve_ccp(p_star, P_all, v)
p_s = P_all @ alpha_s

# --- APoI metadata ---
APOI_NAMES = {0: "cost", 1: "solar", 2: "wind"}
APOI_UNITS = {0: "€/yr", 1: "GW", 2: "GW"}


def print_nav_result(label, tau, SG_in, SL_in, SE_in, delta_in,
                     SG_out, SL_out, SE_out, delta_out, p_s, p_t, cost_s, cost_t):
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {label}")
    print(f"{'='*60}")
    print(f"\n  Starting point:")
    print(f"    solar = {p_s[0]:.3f} GW")
    print(f"    wind  = {p_s[1]:.3f} GW")
    print(f"    cost  = {cost_s:,.0f} €/yr")

    actual_change = {1: p_t[0] - p_s[0], 2: p_t[1] - p_s[1]}

    print(f"\n  Navigation result per APoI (in priority order):")
    for i in tau:
        name  = APOI_NAMES[i]
        unit  = APOI_UNITS[i]
        d_in  = delta_in[i]
        d_out = delta_out[i]

        if i == 0:
            print(f"    APoI {i} ({name}): handled implicitly by final cost minimisation")
            continue

        actual = actual_change[i]

        if i in SG_in and i in SG_out:
            if actual >= d_in - 1e-6:
                status = (f"requested INCREASE by ≥{d_in:.3f} {unit} "
                          f"→ actually increased by {actual:+.3f} {unit} ✓")
            else:
                status = (f"requested INCREASE by ≥{d_in:.3f} {unit} "
                          f"→ CAPPED at maximum feasible: {actual:+.3f} {unit} "
                          f"(delta updated to {d_out:.3f} {unit})")
        elif i in SG_in and i in SE_out:
            status = (f"requested INCREASE by {d_in:.3f} {unit} "
                      f"→ INFEASIBLE, flipped to EQUAL "
                      f"(max feasible deviation: {d_out:.3f} {unit})")
        elif i in SL_in and i in SL_out:
            if abs(actual) >= d_in - 1e-6:
                status = (f"requested DECREASE by ≥{d_in:.3f} {unit} "
                          f"→ actually decreased by {actual:+.3f} {unit} ✓")
            else:
                status = (f"requested DECREASE by ≥{d_in:.3f} {unit} "
                          f"→ CAPPED at maximum feasible: {actual:+.3f} {unit} "
                          f"(delta updated to {d_out:.3f} {unit})")
        elif i in SL_in and i in SE_out:
            status = (f"requested DECREASE by {d_in:.3f} {unit} "
                      f"→ INFEASIBLE, flipped to EQUAL "
                      f"(max feasible deviation: {d_out:.3f} {unit})")
        elif i in SE_in:
            status = (f"requested EQUAL within {d_in:.3f} {unit} "
                      f"→ kept within {actual:+.3f} {unit} ✓")
        else:
            status = f"delta={d_out:.3f} {unit}"

        print(f"    APoI {i} ({name}): {status}")

    print(f"\n  Target point:")
    print(f"    solar = {p_t[0]:.3f} GW  (Δ={p_t[0]-p_s[0]:+.3f} GW)")
    print(f"    wind  = {p_t[1]:.3f} GW  (Δ={p_t[1]-p_s[1]:+.3f} GW)")
    print(f"    cost  = {cost_t:,.0f} €/yr  (Δ={cost_t-cost_s:+,.0f} €/yr)")
    print(f"{'='*60}\n")


# ==========================================================
# SCENARIO C — wind decrease too large → capped
# solar first priority: INCREASE 0.3 GW
# wind second priority: DECREASE 5.0 GW (infeasible)
# ==========================================================
tau_c   = [1, 2, 0]
SG_c    = {1}
SL_c    = {2}
SE_c    = set()
delta_c = np.array([0.0, 0.3, 5.0])

alpha_t_c, SG_co, SL_co, SE_co, delta_co, loop1_bounds_c = navigate(
    P_all, v, alpha_s, tau_c, SG_c, SL_c, SE_c, delta_c)
# loop-1 bounds cover exactly the APoIs that ended up in a direction set
assert {b[0] for b in loop1_bounds_c} == (SG_co | SL_co | SE_co)
p_t_c    = P_all @ alpha_t_c
cost_t_c = float(alpha_t_c @ v)

print_nav_result("C — wind↓ 5GW (infeasible → capped)",
                 tau_c, SG_c, SL_c, SE_c, delta_c,
                 SG_co, SL_co, SE_co, delta_co,
                 p_s, p_t_c, cost_s, cost_t_c)