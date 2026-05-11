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


@dataclass
class PoiSpec:
    name: str
    evaluate: Callable[[pypsa.Network], float]
    linopy_expr: Callable  # (linopy.Model) -> linopy.LinearExpression


def make_poi_specs(network: pypsa.Network) -> List[PoiSpec]:
    solar_gens = network.generators.index[
        network.generators.carrier == "solar"
    ].tolist()

    wind_gens = network.generators.index[
        network.generators.carrier == "wind"
    ].tolist()

    def solar_cap_eval(net):
        return float(net.generators.loc[solar_gens, "p_nom_opt"].sum()) / 1e3

    def wind_cap_eval(net):
        return float(net.generators.loc[wind_gens, "p_nom_opt"].sum()) / 1e3

    def solar_cap_expr(m):
        return m.variables["Generator-p_nom"].sel(name=solar_gens).sum() / 1e3

    def wind_cap_expr(m):
        return m.variables["Generator-p_nom"].sel(name=wind_gens).sum() / 1e3

    return [
        PoiSpec("solar_cap_gw", solar_cap_eval, solar_cap_expr),
        PoiSpec("wind_cap_gw",  wind_cap_eval,  wind_cap_expr),
    ]


def evaluate_all(specs: List[PoiSpec], network: pypsa.Network) -> np.ndarray:
    """Return the full PoI vector p = [poi_0, ..., poi_{m-1}] for a solved network."""
    return np.array([s.evaluate(network) for s in specs], dtype=float)