"""
poi.py — Properties of Interest (PoI) definitions.

Each PoiSpec bundles two representations of the same quantity:
  - evaluate(network)  : compute the scalar value from a solved network
  - linopy_expr(model) : return a Linopy LinearExpression of the same
                         quantity, for use as an objective or constraint

The technical part (turning a PoI into those two representations, which needs
PyPSA's internal variable names) lives in builder functions here. Which PoIs
actually exist is data-driven: the caller passes a declarative list of
definitions to make_poi_specs(), which turns it into PoiSpecs. Nothing about any
particular network (carrier names, technologies, counts) lives in this file —
the caller supplies all of it.

Each definition is a dict:

    {"component": "generator" | "link" | "storage_unit",  # which PyPSA component holds capacity
     "carriers":  ["solar", ...],                         # one or more carriers, summed into one PoI
     "name":      "solar_cap_gw",                         # label for plots/outputs
     "unit":      "GW"}                                   # optional display label, defaults to "GW"
"""

from dataclasses import dataclass
from typing import Callable, List, Sequence
import numpy as np
import pypsa
import linopy

# Unit conversion: MW per GW — used as a DIVISOR (MW values / MW_TO_GW = GW),
# so all PoI values are expressed in GW, not MW
MW_TO_GW = 1e3

# Which PyPSA components a capacity PoI can be built on. Maps the caller-facing
# component name to (the network's static frame, the linopy capacity variable).
# All three components carry their capacity in p_nom / p_nom_opt, so the rest of
# the builder is shared.
_CAPACITY_COMPONENTS = {
    "generator":    ("generators",    "Generator-p_nom"),
    "link":         ("links",         "Link-p_nom"),
    "storage_unit": ("storage_units", "StorageUnit-p_nom"),
}


@dataclass
class PoiSpec:
    """Bundles the two representations of one PoI: an `evaluate` callable and a
    `linopy_expr` callable for the same quantity.

    Attributes
    ----------
    name        : str — PoI identifier (row label in P_all)
    unit        : str — display unit label
    evaluate    : Callable[[pypsa.Network], float] — value from a solved network
    linopy_expr : Callable[[linopy.Model], linopy.LinearExpression] — the same
                  quantity as an optimisation expression
    """

    name: str
    unit: str
    evaluate: Callable[[pypsa.Network], float]
    linopy_expr: Callable[["linopy.Model"], "linopy.LinearExpression"]


def capacity_poi(
    network: pypsa.Network,
    carriers: Sequence[str],
    component: str,
    name: str,
    unit: str = "GW",
) -> PoiSpec:
    """Build a PoiSpec for the total installed capacity of one or more carriers.

    Works for any PyPSA network: `component` selects which component holds the
    capacity ("generator" -> Generator-p_nom, "link" -> Link-p_nom,
    "storage_unit" -> StorageUnit-p_nom) and
    `carriers` lists the carriers to sum into this single PoI (e.g. offshore
    wind split across two carriers). Both representations sum nominal power over
    the matching components and convert MW -> GW:
      - evaluate(net) : sum of p_nom_opt      (from a solved network)
      - linopy_expr(m): sum of <Component>-p_nom  (the optimisation variable)

    PyPSA only creates the -p_nom variable for EXTENDABLE components, so
    linopy_expr sums the variable over the extendable matches and adds the fixed
    capacity of the non-extendable ones as a constant. That keeps it numerically
    identical to evaluate(), which reads p_nom_opt for every match.

    The value is always in GW; `unit` is the display label and should stay "GW".
    The network need not be solved — only carrier membership is read here.

    Raises
    ------
    ValueError
        If `component` is not "generator"/"link"/"storage_unit", if `carriers`
        is not a non-empty list, if any carrier matches no component in this
        network, or if no match is extendable (the PoI would be a constant,
        leaving nothing for the MGA search to vary).
    """
    if component not in _CAPACITY_COMPONENTS:
        raise ValueError(
            f"PoI {name!r}: component must be one of "
            f"{sorted(_CAPACITY_COMPONENTS)}, got {component!r}"
        )
    # A bare string is iterable, so it would silently match per character.
    if isinstance(carriers, str) or not isinstance(carriers, (list, tuple)):
        raise ValueError(
            f"PoI {name!r}: carriers must be a list of carrier names "
            f'(e.g. ["solar"]), got {carriers!r}'
        )
    if len(carriers) == 0:
        raise ValueError(f"PoI {name!r}: carriers must name at least one carrier")

    static_attr, var_name = _CAPACITY_COMPONENTS[component]
    static = getattr(network, static_attr)

    # --- resolve the component names ONCE, here, not on every call ---
    present = set(static["carrier"]) if len(static) else set()
    missing = [c for c in carriers if c not in present]
    if missing:
        raise ValueError(
            f"PoI {name!r}: carrier(s) {missing!r} match no {component} in this "
            f"network. Available {component} carriers: {sorted(present)!r}"
        )

    matched = static.loc[static["carrier"].isin(list(carriers))]
    extendable = matched["p_nom_extendable"].astype(bool)
    matched_names = matched.index.tolist()
    ext_names = matched.index[extendable].tolist()
    # Non-extendable capacity is fixed data, so it enters linopy_expr as a constant.
    fixed_mw = float(matched.loc[~extendable, "p_nom"].sum())

    if not ext_names:
        raise ValueError(
            f"PoI {name!r}: none of the {len(matched_names)} {component}(s) with "
            f"carrier(s) {list(carriers)!r} are extendable, so this PoI is a "
            f"constant ({fixed_mw / MW_TO_GW:.4g} {unit}) and cannot be explored. "
            f"Set p_nom_extendable=True on them, or choose another carrier."
        )

    def _eval(net):
        return float(getattr(net, static_attr).loc[matched_names, "p_nom_opt"].sum()) / MW_TO_GW

    def _expr(m):
        expr = m.variables[var_name].sel(name=ext_names).sum()
        if fixed_mw:
            expr = expr + fixed_mw
        return expr / MW_TO_GW

    return PoiSpec(name, unit, _eval, _expr)


# Keys accepted in a PoI definition dict. Anything else is a typo and is
# rejected rather than silently ignored.
_REQUIRED_KEYS = {"component", "carriers", "name"}
_OPTIONAL_KEYS = {"unit"}


def make_poi_specs(
    network: pypsa.Network,
    definitions: List[dict],
) -> List[PoiSpec]:
    """Build the list of PoiSpecs from a declarative list of definitions.

    Each definition is a dict of keyword arguments for capacity_poi, e.g.
    {"component": "generator", "carriers": ["solar"], "name": "solar_cap_gw"}.
    The order is significant: it fixes the PoI row order of P_all, so it must
    match whatever the preparation phase was run with.
    """
    if definitions is None:
        raise ValueError(
            "make_poi_specs needs an explicit list of PoI definitions, e.g. "
            '[{"component": "generator", "carriers": ["solar"], "name": "solar_cap_gw"}]'
        )

    specs = []
    for i, d in enumerate(definitions):
        if not isinstance(d, dict):
            raise ValueError(f"poi_definitions[{i}] must be a dict, got {type(d).__name__}")

        missing = _REQUIRED_KEYS - d.keys()
        if missing:
            raise ValueError(
                f"poi_definitions[{i}] is missing required key(s) {sorted(missing)}; "
                f"got {sorted(d)}"
            )
        unknown = d.keys() - _REQUIRED_KEYS - _OPTIONAL_KEYS
        if unknown:
            hint = ' (did you mean "carriers"?)' if "carrier" in unknown else ""
            raise ValueError(
                f"poi_definitions[{i}] has unknown key(s) {sorted(unknown)}{hint}; "
                f"expected {sorted(_REQUIRED_KEYS)} plus optional {sorted(_OPTIONAL_KEYS)}"
            )

        specs.append(capacity_poi(network, **d))

    return specs


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

    # One single-carrier generator PoI per carrier the demo network happens to
    # have — derived from the network, so no carrier name is hardcoded here.
    definitions = [
        {"component": "generator", "carriers": [c], "name": f"{c}_cap_gw"}
        for c in sorted(network.generators.carrier.unique())
    ]

    specs = make_poi_specs(network, definitions)
    p_star = evaluate_all(specs, network)

    print("PoI vector p* at optimal solution:")
    for spec, val in zip(specs, p_star):
        print(f"  {spec.name:<20s}  {val:.4f} {spec.unit}")