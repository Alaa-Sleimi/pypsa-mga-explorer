"""
poi.py — Properties of Interest (PoI) definitions.

Each PoiSpec bundles two representations of the same quantity:
  - evaluate(network)  : compute the scalar value from a solved network
  - linopy_expr(model) : return a Linopy LinearExpression of the same
                         quantity, for use as an objective or constraint

The technical part (turning a PoI into those two representations, which needs
PyPSA's internal variable names) lives in builder functions here. Which PoIs
actually exist is data-driven: a declarative list of definitions — see
POI_DEFINITIONS — is turned into PoiSpecs by make_poi_specs(). Callers (and the
notebook) only pass that declarative list; no name-matching logic is needed
outside this file.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
import pypsa
import linopy

# Unit conversion: all PoI values are expressed in GW, not MW
MW_TO_GW = 1e3


@dataclass
class PoiSpec:
    name: str
    unit: str
    evaluate: Callable[[pypsa.Network], float]
    linopy_expr: Callable[["linopy.Model"], "linopy.LinearExpression"]


def capacity_poi(
    network: pypsa.Network,
    carrier: str,
    name: Optional[str] = None,
    unit: str = "GW",
) -> PoiSpec:
    """Build a PoiSpec for the total installed capacity of all generators of one carrier.

    Both representations sum the generators' nominal power and convert MW -> GW:
      - evaluate(net) : sum of p_nom_opt   (from a solved network)
      - linopy_expr(m): sum of Generator-p_nom   (the optimisation variable)

    The value is always in GW; `unit` is the display label and should stay "GW".
    The network need not be solved — only carrier membership is read here.
    """
    gens = network.generators.index[network.generators.carrier == carrier].tolist()
    assert len(gens) > 0, f"No generators with carrier {carrier!r} found — check the network"

    if name is None:
        name = f"{carrier}_cap_gw"

    def _eval(net):
        return float(net.generators.loc[gens, "p_nom_opt"].sum()) / MW_TO_GW

    def _expr(m):
        return m.variables["Generator-p_nom"].sel(name=gens).sum() / MW_TO_GW

    return PoiSpec(name, unit, _eval, _expr)


# Declarative default: which PoIs this project explores. Each entry is passed as
# keyword arguments to capacity_poi(). Edit this list (or pass your own to
# make_poi_specs) to change the PoIs — but the set/order must match whatever the
# preparation phase was run with, since P_all's rows are these PoIs in order.
POI_DEFINITIONS = [
    {"carrier": "solar", "name": "solar_cap_gw", "unit": "GW"},
    {"carrier": "wind",  "name": "wind_cap_gw",  "unit": "GW"},
]


def make_poi_specs(
    network: pypsa.Network,
    definitions: Optional[List[dict]] = None,
) -> List[PoiSpec]:
    """Build the list of PoiSpecs from a declarative list of definitions.

    Each definition is a dict of keyword arguments for capacity_poi
    (e.g. {"carrier": "solar", "name": "solar_cap_gw", "unit": "GW"}).
    Defaults to POI_DEFINITIONS when none is given.
    """
    if definitions is None:
        definitions = POI_DEFINITIONS
    return [capacity_poi(network, **d) for d in definitions]


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
        print(f"  {spec.name:<20s}  {val:.4f} {spec.unit}")