"""Properties of Interest (PoI): the solution metrics the MGA search explores.

A PoI is one scalar summary of a solution — here the total installed capacity, in GW,
of one or more carriers. Every other module works in PoI space, so this module is
where a network's components are turned into the two representations the pipeline
needs, and where the PoI ORDER is fixed that later becomes the row order of ``P_all``.

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
    """One PoI in its two equivalent representations.

    Attributes
    ----------
    name : str
        PoI identifier; also this PoI's row label in ``P_all``.
    unit : str
        Display label only. The value is always in GW.
    evaluate : callable
        ``evaluate(network) -> float``: the value read from a SOLVED network.
    linopy_expr : callable
        ``linopy_expr(model) -> linopy.LinearExpression``: the same quantity as an
        optimisation expression, usable as an objective or a constraint.

    Notes
    -----
    Both callables close over the component names resolved when the spec was built,
    not over the network object, so a spec built on one network can be evaluated on
    another network that has the same component names.
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
    """Build a PoI for the total installed capacity of one or more carriers.

    Parameters
    ----------
    network : pypsa.Network
        Network to resolve the carriers against; need not be solved.
    carriers : list of str or tuple of str
        Carriers summed into this one PoI. A bare string is rejected.
    component : {"generator", "link", "storage_unit"}
        Component holding the capacity, i.e. ``<Component>-p_nom``.
    name : str
        PoI identifier; also this PoI's row label in ``P_all``.
    unit : str, default "GW"
        Display label only; the value is always GW.

    Returns
    -------
    PoiSpec
        ``evaluate`` sums ``p_nom_opt`` over every match; ``linopy_expr`` sums the
        ``-p_nom`` variable over the extendable matches (the only ones PyPSA creates
        a variable for) plus the fixed capacity of the rest, so the two agree. Both
        convert MW to GW.

    Raises
    ------
    ValueError
        Unsupported `component`; `carriers` not a non-empty list or tuple; a carrier
        matching nothing here; or no extendable match, which would make this PoI a
        constant with nothing for the MGA search to vary.

    Notes
    -----
    For links ``p_nom`` is the ``bus0`` INPUT capacity: a hydrogen turbine link is
    measured in GW of hydrogen in, not electricity out.
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
        """Sum p_nom_opt [MW] over the matched components of `net` and return GW."""
        return float(getattr(net, static_attr).loc[matched_names, "p_nom_opt"].sum()) / MW_TO_GW

    def _expr(m):
        """Return the same capacity in GW as a linopy expression over model `m`."""
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
    """Build the ordered list of PoI specs from declarative definitions.

    Parameters
    ----------
    network : pypsa.Network
        Network the definitions are resolved against; need not be solved.
    definitions : list of dict
        One dict per PoI, passed as keyword arguments to :func:`capacity_poi`, e.g.
        ``{"component": "generator", "carriers": ["solar"], "name": "solar_cap_gw"}``.
        Required keys: ``component``, ``carriers``, ``name``; optional: ``unit``.

    Returns
    -------
    list of PoiSpec
        Specs in definition order.

    Raises
    ------
    ValueError
        `definitions` is None, an entry is not a dict, a required key is missing, or
        an unknown key is present (``"carrier"`` is hinted towards ``"carriers"``);
        plus anything :func:`capacity_poi` raises.

    Notes
    -----
    The order is significant: it fixes the PoI row order of ``P_all`` and must match
    what the preparation phase was run with.
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
    """Evaluate every PoI on a solved network.

    Parameters
    ----------
    specs : list of PoiSpec
        Specs to evaluate, in the intended PoI order.
    network : pypsa.Network
        Solved network; each ``evaluate`` reads ``p_nom_opt``.

    Returns
    -------
    numpy.ndarray
        Shape ``(m,)``. The PoI vector ``p = [poi_0, ..., poi_{m-1}]`` in GW,
        ordered like `specs`.

    Notes
    -----
    On an UNSOLVED network PyPSA's ``p_nom_opt`` column exists and is all zeros, so
    this silently returns zeros instead of raising. Make sure the network has been
    optimised, or the caller will read a valid-looking all-zero PoI vector.
    """
    return np.array([s.evaluate(network) for s in specs], dtype=float)