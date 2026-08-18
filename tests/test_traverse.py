"""
tests/test_traverse.py — Test traverse algorithms using saved preparation state.

Loads preparation state, runs navigation to get alpha_s and alpha_t,
then runs Algorithm 2 (breakpoints) and Algorithm 3 (subgradients),
and plots the cost approximation along the interpolation path.

Run with:
    python -m tests.test_traverse
"""

import numpy as np
import matplotlib.pyplot as plt

from mga_engine.preparation_state import load
from mga_engine.ccp_solver import solve_ccp
from mga_engine.navigation import navigate
from mga_engine.traverse import extract_breakpoints, generate_subgradients, interpolate

# --- Load preparation state ---
state    = load("data/preparation_state.nc")
P_all    = state["P_all"]
v        = state["v"]
Gamma    = state["Gamma"]
p_star   = state["p_star"]
opt_cost = float(v.min())   # c'x* approximated as min(v)

# --- Starting point ---
alpha_s, cost_s = solve_ccp(p_star, P_all, v)
p_s = P_all @ alpha_s

# --- Navigation: solar↑ 0.3 GW, wind↓ 0.3 GW ---
tau   = [1, 2, 0]
SG    = {1}
SL    = {2}
SE    = set()
delta = np.array([0.0, 0.3, 0.3])

alpha_t, _, _, _, _, _ = navigate(P_all, v, alpha_s, tau, SG, SL, SE, delta)
p_t    = P_all @ alpha_t
cost_t = float(alpha_t @ v)

print(f"\n[traverse] Start:  solar={p_s[0]:.3f} GW, wind={p_s[1]:.3f} GW, cost={cost_s:,.0f} €/yr")
print(f"[traverse] Target: solar={p_t[0]:.3f} GW, wind={p_t[1]:.3f} GW, cost={cost_t:,.0f} €/yr")

# --- Algorithm 2: Extract breakpoints ---
epsilon_I = 1e6   # 1M €/yr tolerance
SI = extract_breakpoints(P_all, v, alpha_s, alpha_t, epsilon_I=epsilon_I)

print(f"\n[traverse] Breakpoints found: {len(SI)}")
for beta, alpha in SI:
    p_bp   = P_all @ alpha
    cost_bp = float(v @ alpha)
    print(f"  β={beta:.3f}  solar={p_bp[0]:.3f} GW  wind={p_bp[1]:.3f} GW  cost={cost_bp:,.0f} €/yr")

# --- Algorithm 3: Generate subgradients ---
SS = generate_subgradients(Gamma, P_all, v, opt_cost, alpha_s, alpha_t, SI)
print(f"\n[traverse] Subgradient functions generated: {len(SS)}")

# --- Plot: cost along interpolation path ---
betas = np.linspace(0, 1, 200)

# Linear interpolation cost
cost_linear = [interpolate(P_all, v, alpha_s, alpha_t, b)[1] for b in betas]

# CCP upper bound at each beta
cost_ccp = []
for b in betas:
    p_b = interpolate(P_all, v, alpha_s, alpha_t, b)[0]
    try:
        _, c = solve_ccp(p_b, P_all, v)
        cost_ccp.append(c)
    except RuntimeError:
        cost_ccp.append(np.nan)

# Subgradient lower bound: max over all subgradients at each beta
cost_lower = [max(g(b) for g in SS) for b in betas]

fig, ax = plt.subplots(figsize=(9, 5))

# --- Plot individual subgradient lines (only those that contribute to envelope) ---
g_values = np.array([[g(b) for b in betas] for g in SS])  # shape (n_subgrads, n_betas)
envelope = g_values.max(axis=0)                             # pointwise maximum

for g_vals in g_values:
    is_max = np.isclose(g_vals, envelope, atol=1e3)
    # Only plot lines that are at the maximum at least once
    if not is_max.any():
        continue
    for j in range(len(betas) - 1):
        style = "-" if is_max[j] else "--"
        alpha_val = 1.0 if is_max[j] else 0.4
        ax.plot(betas[j:j+2], g_vals[j:j+2],
                color="lightcoral", lw=1.2,
                linestyle=style, alpha=alpha_val)
# --- Main curves ---
ax.plot(betas, cost_linear, "b--", lw=1.5, label="linear interpolation α(β)'v")
ax.plot(betas, cost_ccp,    "g-",  lw=1.5, label="CCP upper bound α*'v")
ax.plot(betas, envelope,    "r-",  lw=2.0, label="subgradient lower bound (envelope)")

# --- Breakpoints ---
for beta, alpha in SI:
    cost_bp = float(v @ alpha)
    ax.axvline(beta, color="gray", lw=0.8, linestyle=":")
    ax.scatter(beta, cost_bp, color="purple", s=60, zorder=5)

ax.set_xlabel("β  (0 = start, 1 = target)")
ax.set_ylabel("Cost approximation (€/yr)")
ax.set_title("Traverse — cost along interpolation path")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("traverse_cost_path.png", dpi=150)
plt.show()
print("\n[traverse] Plot saved to traverse_cost_path.png")