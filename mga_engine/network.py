"""
network.py — Example 3-bus PyPSA capacity-expansion network.

Topology:
    north -- south -- east

Generators:
    solar_north, solar_east   (extendable, zero marginal cost)
    wind_north                (extendable, zero marginal cost)
    gas_south                 (extendable, expensive marginal cost)

Loads:
    fixed demand at each bus

Snapshot weighting:
    72 snapshots (3 representative days) scaled so weights sum to 8760 h/yr.
"""

import numpy as np
import pypsa

# Annualised capital costs [€/MW/yr]
SOLAR_CAPEX = 60_000
WIND_CAPEX  = 90_000
GAS_CAPEX   = 30_000

# Marginal costs [€/MWh]
GAS_MC   = 60.0
SOLAR_MC = 0.0
WIND_MC  = 0.0

# Fixed line capacity [MW]
LINE_S_NOM = 2000.0

# Snapshot setup
N_SNAPSHOTS = 72
WEIGHT      = 8760 / N_SNAPSHOTS


def build_network() -> pypsa.Network:
    """Build and return the example network (not yet optimised)."""

    n = pypsa.Network()
    n.set_snapshots(range(N_SNAPSHOTS))
    n.snapshot_weightings[:] = WEIGHT

    # Buses
    n.add("Bus", "north", carrier="AC")
    n.add("Bus", "south", carrier="AC")
    n.add("Bus", "east",  carrier="AC")

    # Lines (small resistance added to avoid zero-r warning)
    n.add("Line", "north-south", bus0="north", bus1="south",
          s_nom=LINE_S_NOM, x=0.01, r=0.001, carrier="AC")
    n.add("Line", "south-east",  bus0="south", bus1="east",
          s_nom=LINE_S_NOM, x=0.01, r=0.001, carrier="AC")

    # Capacity factors
    hours = np.arange(N_SNAPSHOTS)
    solar_cf = np.clip(np.sin(np.pi * (hours % 24 - 6) / 12), 0, 1)
    wind_cf  = 0.3 + 0.15 * np.sin(2 * np.pi * hours / 24 + 1.0)
    wind_cf += 0.05 * np.sin(2 * np.pi * hours / (24 * 3))

    # Solar generators
    for bus, name in [("north", "solar_north"), ("east", "solar_east")]:
        n.add(
            "Generator", name,
            bus=bus,
            carrier="solar",
            p_nom=0.0,
            p_nom_extendable=True,
            p_nom_min=0.0,
            capital_cost=SOLAR_CAPEX,
            marginal_cost=SOLAR_MC,
            p_max_pu=solar_cf,
        )

    # Wind generator
    n.add(
        "Generator", "wind_north",
        bus="north",
        carrier="wind",
        p_nom=0.0,
        p_nom_extendable=True,
        p_nom_min=0.0,
        capital_cost=WIND_CAPEX,
        marginal_cost=WIND_MC,
        p_max_pu=wind_cf,
    )

    # Gas backup
    n.add(
        "Generator", "gas_south",
        bus="south",
        carrier="gas",
        p_nom=0.0,
        p_nom_extendable=True,
        p_nom_min=0.0,
        capital_cost=GAS_CAPEX,
        marginal_cost=GAS_MC,
    )

    # Loads [MW]
    for bus, load_mw in [("north", 500), ("south", 800), ("east", 400)]:
        n.add("Load", f"load_{bus}", bus=bus, p_set=load_mw)

    n.sanitize()
    return n


if __name__ == "__main__":
    from mga_engine.poi import make_poi_specs, evaluate_all

    network = build_network()
    status, _ = network.optimize(
        solver_name="highs",
        include_objective_constant=False,
    )

    print(f"Status       : {status}")
    print(f"Optimal cost : {network.objective:,.0f} €/yr")
    print()
    print("Optimal capacities [MW]:")
    print(network.generators[["carrier", "p_nom_opt"]])
    print()

    specs = make_poi_specs(network)        
    p_star = evaluate_all(specs, network)
    print("PoI vector p* at optimal solution:")
    for spec, val in zip(specs, p_star):
        print(f"  {spec.name:<20s}  {val:.4f}")