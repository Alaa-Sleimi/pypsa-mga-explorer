"""
poi.py — Properties of Interest (PoI) definitions.

Each PoiSpec bundles two representations of the same quantity:
  - evaluate(network)  : compute the scalar value from a solved network
  - linopy_expr(model) : return a Linopy LinearExpression of the same
                         quantity, for use as an objective or constraint

Both representations are defined together in make_poi_specs().
No name-matching logic is needed anywhere outside this file.
"""

from dataclasses import dataclass
from typing import Callable, List
import numpy as np
import pypsa
import linopy

# Unit conversion: all PoI values are expressed in GW, not MW
MW_TO_GW = 1e3


@dataclass
class PoiSpec:
    name: str
    evaluate: Callable[[pypsa.Network], float]
    linopy_expr: Callable[["linopy.Model"], "linopy.LinearExpression"]


def make_poi_specs(network: pypsa.Network) -> List[PoiSpec]:
    # Only generator carrier names are needed here — network does not need to be solved yet
    solar_gens = network.generators.index[
        network.generators.carrier == "solar"
    ].tolist()

    wind_gens = network.generators.index[
        network.generators.carrier == "wind"
    ].tolist()

    assert len(solar_gens) > 0, "No solar generators found — check carrier names in network"
    assert len(wind_gens)  > 0, "No wind generators found — check carrier names in network"

    def solar_cap_eval(net):
        return float(net.generators.loc[solar_gens, "p_nom_opt"].sum()) / MW_TO_GW

    def wind_cap_eval(net):
        return float(net.generators.loc[wind_gens, "p_nom_opt"].sum()) / MW_TO_GW

    def solar_cap_expr(m):
        return m.variables["Generator-p_nom"].sel(name=solar_gens).sum() / MW_TO_GW

    def wind_cap_expr(m):
        return m.variables["Generator-p_nom"].sel(name=wind_gens).sum() / MW_TO_GW

    return [
        PoiSpec("solar_cap_gw", solar_cap_eval, solar_cap_expr),
        PoiSpec("wind_cap_gw",  wind_cap_eval,  wind_cap_expr),
    ]


def evaluate_all(specs: List[PoiSpec], network: pypsa.Network) -> np.ndarray:
    """Return the full PoI vector p = [poi_0, ..., poi_{m-1}] for a solved network."""
    return np.array([s.evaluate(network) for s in specs], dtype=float)


if __name__ == "__main__":
    from mga_engine.network import build_network

    network = build_network()
    network.optimize(
        solver_name="highs",
        include_objective_constant=False,  # keeps objective as pure total system cost
    )

    specs = make_poi_specs(network)
    p_star = evaluate_all(specs, network)

    print("PoI vector p* at optimal solution:")
    for spec, val in zip(specs, p_star):
        print(f"  {spec.name:<20s}  {val:.4f} GW")