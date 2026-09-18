"""Run the preparation phase, and persist and verify its output.

Last module of the preparation phase and the first thing the exploration phase
touches. :func:`run_preparation` orchestrates the whole heavy pipeline (base solve,
vertex and interior sampling, GP pricing) and :func:`save` / :func:`load` move the
result through a single netCDF4 file, so the many solves happen once per network
rather than once per session. :func:`network_fingerprint` and :func:`check_state`
then confirm that a cached file really belongs to the network, PoI definitions and
epsilon in front of it.

netCDF4 is used directly rather than through xarray, which is faster on the large
per-sample solution arrays that bigger models produce.

The file contains:
    - P_all   : (poi, sample)            all sample points
    - v       : (sample,)                minimum cost at each sample point
    - Gamma   : (poi, sample)            dual variables at each sample point
    - p_star  : (poi,)                   PoI vector at the optimal solution
    - epsilon : scalar                   near-optimality threshold used in preparation
    - n_vertices, n_interior_levels, n_samples_per_level, epsilon_min :
                                         sampling parameters used (global attributes)
    - poi_names : list[str]              PoI name per row of P_all, in order
    - solution variables stacked across samples:
        static  e.g. Generator_p_nom_opt : (sample, <key>__dim0)
        dynamic e.g. Generator_p         : (sample, <key>__dim0, <key>__dim1)

Optionally (added later, backward-compatible) the file also carries a
*network fingerprint* as three global attributes:
    - fingerprint         : sha256 hex of the network structure + PoI defs + epsilon
    - fingerprint_version : int schema version of the fingerprint algorithm
    - fingerprint_summary : small JSON summary (counts, carriers, poi_names, epsilon)
These let the exploration phase confirm a cached state actually belongs to the
current network before using it (see network_fingerprint / check_state). Files
written before fingerprinting simply lack these attributes and load fine, being
reported as UNVERIFIED rather than crashing.

The PoI rows of P_all/Gamma/p_star are ordered, and poi_names records what each
row is. The exploration phase loads this to confirm its PoI definitions match the
ones the preparation phase was run with.

Usage:
    save("data/preparation_state.nc", P_all, v, Gamma, solutions, p_star, epsilon, poi_names)
    state = load("data/preparation_state.nc")
    state["P_all"], state["v"], state["Gamma"], state["solutions"]
    state["p_star"], state["epsilon"], state["poi_names"]
"""

import os
import json
import hashlib
from math import floor, log10

import numpy as np
from netCDF4 import Dataset


# ---------------------------------------------------------------------------
# Network fingerprint - a cheap, solver-free structural hash used to detect a
# preparation state loaded against the wrong network / PoIs / epsilon.
# ---------------------------------------------------------------------------

# Bump when the fingerprint contents/algorithm change so old fingerprints are
# treated as UNVERIFIED rather than compared against a different scheme.
FINGERPRINT_VERSION = 1

# Per-component INPUT columns that define the model. Deliberately excludes any
# solved/result columns (e.g. *_opt) so the fingerprint is solve-invariant and
# platform/solver independent. Only components in this map are fingerprinted;
# others (Carrier, SubNetwork, ...) are ignored.
_STATIC_COLS = {
    "Bus":         ["carrier"],
    "Generator":   ["bus", "carrier", "p_nom", "p_nom_extendable", "p_nom_min",
                    "p_nom_max", "capital_cost", "marginal_cost", "efficiency",
                    "p_max_pu", "p_min_pu", "sign"],
    "Line":        ["bus0", "bus1", "s_nom", "s_nom_extendable", "s_nom_min",
                    "s_nom_max", "x", "r", "capital_cost"],
    "Load":        ["bus", "p_set", "sign"],
    "StorageUnit": ["bus", "carrier", "p_nom", "p_nom_extendable", "p_nom_min",
                    "p_nom_max", "capital_cost", "marginal_cost", "max_hours",
                    "efficiency_store", "efficiency_dispatch"],
    "Store":       ["bus", "carrier", "e_nom", "e_nom_extendable", "e_nom_min",
                    "e_nom_max", "capital_cost", "marginal_cost"],
    "Link":        ["bus0", "bus1", "p_nom", "p_nom_extendable", "p_nom_min",
                    "p_nom_max", "capital_cost", "marginal_cost", "efficiency"],
    "GlobalConstraint": ["type", "carrier_attribute", "sense", "constant"],
}

# Per-component time-varying INPUT attributes to hash (again, no result frames
# like Generator_p or Bus_marginal_price).
_TS_ATTRS = {
    "Generator":   ["p_max_pu", "p_min_pu", "p_set"],
    "Load":        ["p_set"],
    "StorageUnit": ["inflow", "p_max_pu", "p_min_pu"],
    "Store":       ["e_max_pu", "e_min_pu"],
    "Link":        ["p_max_pu", "p_min_pu"],
}


def _num(x, sig: int = 6):
    """Round a float to `sig` significant figures; non-finite values to a sentinel.

    Parameters
    ----------
    x : object
        Value to canonicalise. Anything that does not convert to float is returned
        as ``str(x)``.
    sig : int, default 6
        Significant figures to keep.

    Returns
    -------
    float or str
        The rounded float, ``0.0`` for zero with ``-0.0`` normalised away, or one of
        the strings ``"nan"``, ``"inf"``, ``"-inf"``, ``str(x)``.

    Notes
    -----
    Significant-figure rounding, rather than decimal rounding, keeps the fingerprint
    stable against float-repr noise while still catching real changes in capacities
    and costs at any magnitude. :func:`_timeseries_digest` instead rounds to 6
    DECIMALS, so the two paths do not have the same sensitivity.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0.0:
        return 0.0
    ndigits = -int(floor(log10(abs(x)))) + (sig - 1)
    return round(x, ndigits) + 0.0   # + 0.0 normalises -0.0 -> 0.0


def _cell(x):
    """Canonicalise one static-table cell to a JSON-stable primitive.

    Parameters
    ----------
    x : object
        A single cell of a component's static frame.

    Returns
    -------
    bool or str or float
        Booleans (including ``numpy.bool_``) as ``bool`` and strings unchanged, so
        that ``True`` and ``1.0`` cannot collide; anything else through
        :func:`_num`.
    """
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x
    return _num(x)


def _timeseries_digest(df):
    """Stable sha256 of a rounded time-varying frame, or ``None`` if there is none.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series or None
        A time-varying frame. A Series is widened to a one-column frame first.

    Returns
    -------
    str or None
        The hex digest over the sorted column names, the shape and the rounded
        values; ``None`` for ``None``, for an empty frame, and for anything without
        an ``empty`` attribute.

    Raises
    ------
    ValueError
        From the ``f8`` conversion if the frame holds non-numeric values.

    Notes
    -----
    Columns are sorted before hashing so column ORDER cannot change the digest, and
    values are rounded to 6 DECIMALS with ``-0.0`` normalised so float noise cannot
    either. Decimal rounding means an absolute change below 5e-7 is invisible, which
    differs from :func:`_num`'s significant-figure rounding; on a per-unit series
    like ``p_max_pu`` that is a fine tolerance, on a large-magnitude series it is a
    much looser one.
    Columns are reindexed by their stringified names, so a frame with non-string
    column labels would hash all-NaN columns instead of its values.
    """
    if df is None:
        return None
    if not hasattr(df, "columns"):        # a Series -> single-column frame
        df = df.to_frame()
    try:
        if df.empty:
            return None
    except AttributeError:
        return None

    cols = sorted(map(str, df.columns.tolist()))
    arr = np.ascontiguousarray(df.reindex(columns=cols).to_numpy(dtype="f8"))
    arr = np.round(arr, 6) + 0.0

    h = hashlib.sha256()
    h.update("|".join(cols).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _fingerprint_object(network, poi_definitions, epsilon) -> dict:
    """Build the canonical, JSON-serialisable structure that gets hashed.

    Parameters
    ----------
    network : pypsa.Network
        Network to describe. Solved or unsolved: only input data is read.
    poi_definitions : list of dict
        PoI definitions in their intended order, as given to
        :func:`mga_engine.poi.make_poi_specs`.
    epsilon : float
        Near-optimality slack the preparation runs at.

    Returns
    -------
    dict
        Keys ``"version"``, ``"epsilon"``, ``"n_snapshots"``,
        ``"snapshot_weightings"`` (a digest), ``"components"`` (per component name,
        ``{"cols", "rows"}`` plus ``"ts"`` when any time series is non-empty) and
        ``"poi_definitions"`` (``[component, carriers, name, unit]`` per PoI).

    Notes
    -----
    What enters, exactly: for each component whose name is a key of ``_STATIC_COLS``
    and whose static frame is non-empty, the rows of the WHITELISTED columns only,
    each row prefixed by the component's index name and the rows sorted so row order
    cannot change the hash; plus a digest of each attribute in ``_TS_ATTRS`` for that
    component. Everything else is omitted, including all ``*_opt`` result columns,
    which is what makes the fingerprint solve-invariant.
    Snapshots enter only as their COUNT plus a digest of the weightings, so replacing
    the snapshot timestamps while keeping the count and weightings is invisible.
    PoI order is preserved deliberately, since it fixes the row order of P_all, and
    each PoI's carrier list is hashed in order too; ``json.dumps(sort_keys=True)``
    sorts dict keys but never list entries. A PoI definition without ``"unit"``
    hashes as ``"GW"``, matching the default in
    :func:`mga_engine.poi.make_poi_specs`.
    See :func:`network_fingerprint` for the consequences of what is left out.
    """
    components = {}
    for component in network.components:
        cname = getattr(component, "name", None)
        if cname not in _STATIC_COLS:
            continue
        static = component.static
        if static is None or len(static) == 0:
            continue

        cols = [c for c in _STATIC_COLS[cname] if c in static.columns]
        rows = []
        for idx in static.index:
            rows.append([str(idx)] + [_cell(static.at[idx, c]) for c in cols])
        rows.sort()
        entry = {"cols": cols, "rows": rows}

        dynamic = {a: d for a, d in component.dynamic.items()}
        ts = {}
        for attr in _TS_ATTRS.get(cname, []):
            digest = _timeseries_digest(dynamic.get(attr))
            if digest is not None:
                ts[attr] = digest
        if ts:
            entry["ts"] = ts

        components[cname] = entry

    return {
        "version": FINGERPRINT_VERSION,
        "epsilon": _num(epsilon),
        "n_snapshots": int(len(network.snapshots)),
        "snapshot_weightings": _timeseries_digest(network.snapshot_weightings),
        "components": components,
        # PoI order matters (it fixes P_all's rows), so keep it as given. The
        # carriers list is hashed in order too, since it is part of the PoI's
        # identity (json.dumps preserves list order; sort_keys only sorts dicts).
        "poi_definitions": [
            [d.get("component"), d.get("carriers"), d.get("name"), d.get("unit", "GW")]
            for d in poi_definitions
        ],
    }


def network_fingerprint(network, poi_definitions, epsilon) -> str:
    """Return a sha256 fingerprint of a WHITELISTED part of the network setup.

    Parameters
    ----------
    network : pypsa.Network
        Network to fingerprint. Solved or unsolved: only input data is read.
    poi_definitions : list of dict
        PoI definitions in their intended order, as given to
        :func:`mga_engine.poi.make_poi_specs`.
    epsilon : float
        Near-optimality slack; a different epsilon gives a different fingerprint.

    Returns
    -------
    str
        64-character sha256 hex digest of :func:`_fingerprint_object`, serialised
        with sorted dict keys and no whitespace.

    Notes
    -----
    Pure and cheap: no ``optimize`` call and no ``*_opt`` column is read, so the
    digest is deterministic, solve-invariant and independent of platform and solver.

    COVERED: epsilon; the snapshot count and a digest of the snapshot weightings; the
    ordered PoI definitions; and, for Bus, Generator, Line, Load, StorageUnit, Store,
    Link and GlobalConstraint only, the columns listed in ``_STATIC_COLS`` and the
    time series listed in ``_TS_ATTRS``.

    NOT COVERED, so a change to any of these leaves the digest identical and a stale
    cached state is silently reused (each of these was checked against the code):

    - Whole components: ``Carrier`` (so ``co2_emissions`` is invisible),
      ``Transformer``, ``ShuntImpedance``, and every other component absent from
      ``_STATIC_COLS``.
    - StorageUnit: ``cyclic_state_of_charge``, ``state_of_charge_initial``,
      ``sign``, and the time-varying ``standing_loss`` and ``marginal_cost``.
    - Store: ``e_cyclic``, ``e_initial``, ``sign``, and the time-varying
      ``standing_loss`` and ``marginal_cost``.
    - Link: any bus beyond ``bus0``/``bus1``, i.e. the extra legs and efficiencies
      (``bus2``, ``efficiency2``, ...) of a multi-output link, plus its time-varying
      ``efficiency`` and ``marginal_cost``.
    - Generator: ``committable``, the ramp limits, ``start_up_cost`` /
      ``shut_down_cost``, ``e_sum_min`` / ``e_sum_max``, ``build_year`` /
      ``lifetime`` / ``active``, and the time-varying ``efficiency`` and
      ``marginal_cost``. Only the STATIC ``marginal_cost`` and ``efficiency``
      columns are hashed.
    - Line: the time-varying ``s_max_pu``.
    - The snapshot timestamps themselves, as noted in :func:`_fingerprint_object`.

    Time-varying ``marginal_cost`` is the sharpest of these: switching a generator
    from a static cost to a time series, or editing that series, changes the optimum
    while :func:`check_state` still reports PASS.
    """
    obj = _fingerprint_object(network, poi_definitions, epsilon)
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_summary(network, poi_definitions, epsilon) -> dict:
    """Build the small human-readable summary stored alongside the fingerprint.

    Parameters
    ----------
    network : pypsa.Network
        Network to summarise.
    poi_definitions : list of dict
        PoI definitions in their intended order.
    epsilon : float
        Near-optimality slack.

    Returns
    -------
    dict
        Keys ``n_buses``, ``n_generators``, ``n_lines``, ``n_loads``,
        ``n_snapshots``, ``carriers`` (the sorted distinct GENERATOR carriers),
        ``poi_names`` and ``epsilon``.

    Notes
    -----
    Diagnostic only: the hash is the source of truth, and this exists so that
    :func:`_summary_diff` can say WHAT differs on a mismatch. It is coarser than the
    fingerprint in both directions: it counts no links, storage units or stores, and
    lists carriers only for generators, so for a network whose difference lies
    elsewhere the diff falls back to its generic "coarse summary matches" line.
    """
    gens = network.generators
    carriers = sorted(map(str, gens["carrier"].unique().tolist())) if len(gens) else []
    return {
        "n_buses": int(len(network.buses)),
        "n_generators": int(len(network.generators)),
        "n_lines": int(len(network.lines)),
        "n_loads": int(len(network.loads)),
        "n_snapshots": int(len(network.snapshots)),
        "carriers": carriers,
        "poi_names": [d.get("name") for d in poi_definitions],
        "epsilon": float(epsilon),
    }


def make_fingerprint(network, poi_definitions, epsilon) -> dict:
    """Bundle the three fingerprint pieces to hand to ``save(fingerprint=...)``.

    Parameters
    ----------
    network : pypsa.Network
        Network the preparation was, or will be, run on.
    poi_definitions : list of dict
        PoI definitions in their intended order.
    epsilon : float
        Near-optimality slack the preparation runs at.

    Returns
    -------
    dict
        Keys ``"fingerprint"`` (the hex digest), ``"fingerprint_version"`` (the
        current ``FINGERPRINT_VERSION``) and ``"fingerprint_summary"`` (the dict from
        :func:`fingerprint_summary`).

    Notes
    -----
    Must be called with the same network, PoI definitions and epsilon the samples
    were actually produced with; nothing cross-checks that, so fingerprinting a
    different setup would certify the wrong thing.
    """
    return {
        "fingerprint": network_fingerprint(network, poi_definitions, epsilon),
        "fingerprint_version": FINGERPRINT_VERSION,
        "fingerprint_summary": fingerprint_summary(network, poi_definitions, epsilon),
    }


def _write_fingerprint_attrs(ds, fingerprint):
    """Write the fingerprint bundle onto an open dataset as GLOBAL ATTRIBUTES.

    Parameters
    ----------
    ds : netCDF4.Dataset
        Dataset open for writing.
    fingerprint : dict or str
        The bundle from :func:`make_fingerprint`, or a bare hash string, which is
        wrapped into a bundle carrying only the digest.

    Returns
    -------
    None
        Sets ``ds.fingerprint``, ``ds.fingerprint_version`` and, when a summary is
        present, ``ds.fingerprint_summary`` as JSON with sorted keys. Returns
        without writing anything if the bundle carries no digest.

    Notes
    -----
    Global attributes, never variables: :func:`load` treats every variable other than
    the four reserved ones as a per-sample solution key, so a fingerprint stored as a
    variable would be reconstructed as bogus solution data.
    A missing ``fingerprint_version`` in the bundle defaults to the CURRENT
    ``FINGERPRINT_VERSION``, which would mislabel a digest computed under an older
    scheme; :func:`make_fingerprint` always supplies it.
    """
    if isinstance(fingerprint, str):
        fingerprint = {"fingerprint": fingerprint}
    digest = fingerprint.get("fingerprint")
    if digest is None:
        return
    ds.fingerprint = str(digest)
    ds.fingerprint_version = int(fingerprint.get("fingerprint_version", FINGERPRINT_VERSION))
    summary = fingerprint.get("fingerprint_summary")
    if summary is not None:
        ds.fingerprint_summary = json.dumps(summary, sort_keys=True)


def _summary_diff(old: dict, new: dict) -> list:
    """Describe how two fingerprint summaries differ, one line per key.

    Parameters
    ----------
    old : dict
        The summary stored in the loaded state. A missing key reads as
        ``"<missing>"``.
    new : dict
        The summary recomputed from the current setup.

    Returns
    -------
    list of str
        One indented line per differing key, over the fixed key list. When no key
        differs, a single line saying the coarse summary matches while the detailed
        fingerprint does not, which points at capacities, costs or time series.
    """
    keys = ["n_buses", "n_generators", "n_lines", "n_loads", "n_snapshots",
            "carriers", "poi_names", "epsilon"]
    lines = []
    for k in keys:
        o = old.get(k, "<missing>")
        c = new.get(k)
        if o != c:
            lines.append(f"  - {k}: saved={o!r}  →  current={c!r}")
    if not lines:
        lines.append("  - coarse summary matches, but the detailed fingerprint differs "
                     "(likely capacities, costs, or time-series such as p_max_pu changed).")
    return lines


def check_state(state, network, poi_definitions, requested_epsilon,
                requested_n_vertices=None, requested_seed=None,
                requested_n_samples_per_level=None):
    """Compare a loaded state's fingerprint against the current setup.

    The network / PoI / epsilon fingerprint is always checked. The sampling
    parameters are compared as well, but only the ones the caller actually asks
    about: each ``requested_*`` argument defaults to ``None``, meaning "do not check
    this one".

    Parameters
    ----------
    state : dict
        A state as returned by :func:`load`. Only its metadata keys are read, never
        the arrays.
    network : pypsa.Network
        The network the exploration is about to run against. Solved or unsolved:
        only input data is read.
    poi_definitions : list of dict
        The current PoI definitions, in order, exactly as handed to
        :func:`mga_engine.poi.make_poi_specs`. Order is part of the fingerprint.
    requested_epsilon : float
        The epsilon the current run intends to use. This also covers the
        vertex-phase epsilon, since it enters the network fingerprint.
    requested_n_vertices : int, optional
        Vertex-sample count to compare against the stored one. ``None`` skips it.
    requested_seed : int, optional
        RNG seed to compare against the stored one. ``None`` skips it.
    requested_n_samples_per_level : int, optional
        Interior samples per level to compare against the stored one. ``None``
        skips it.

    Returns
    -------
    status : str
        ``"PASS"`` when the fingerprint matches this network, these PoIs and this
        epsilon, and every requested sampling parameter that could be compared
        matches too. ``"MISMATCH"`` on a real difference, in the fingerprint or in a
        supplied sampling parameter. ``"UNVERIFIED"`` when there is nothing
        comparable: no stored fingerprint, a different ``FINGERPRINT_VERSION``, or a
        requested sampling parameter the file predates.
    message : str
        Human-readable explanation. On a fingerprint mismatch it embeds the
        :func:`_summary_diff` lines.

    Notes
    -----
    Pure and cheap: recomputes the fingerprint (no solver call) and compares strings
    and values. Nothing is raised on a mismatch and nothing is deleted; the caller
    decides what a given status means, and the notebook turns ``"MISMATCH"`` into a
    ``RuntimeError``.
    The messages name the notebook's own controls ("set rerun_preparation = True and
    re-run Section 3"), so they read oddly outside that notebook.
    ``"PASS"`` is only as strong as the fingerprint: a network change outside the
    whitelist in :func:`network_fingerprint` still passes, and a stale cached state
    is then reused silently. ``n_interior_levels`` and ``epsilon_min`` are stored by
    :func:`save` but cannot be checked here at all, so a caller that varies them has
    to compare them itself.
    A ``"MISMATCH"`` on the fingerprint short-circuits: the sampling parameters are
    not examined, so its message names only the network-level difference.
    """
    stored = state.get("fingerprint")
    if stored is None:
        return ("UNVERIFIED",
                "This preparation state was saved before fingerprinting was added, so it "
                "cannot be checked against the current network. Proceeding unverified — if "
                "results look wrong, set rerun_preparation = True to rebuild it.")

    stored_version = state.get("fingerprint_version")
    if stored_version != FINGERPRINT_VERSION:
        return ("UNVERIFIED",
                f"Preparation state fingerprint version ({stored_version}) differs from this "
                f"code's ({FINGERPRINT_VERSION}); cannot compare reliably. Proceeding unverified.")

    current = network_fingerprint(network, poi_definitions, requested_epsilon)
    if current != stored:
        old_summary = state.get("fingerprint_summary") or {}
        new_summary = fingerprint_summary(network, poi_definitions, requested_epsilon)
        message = ("Preparation state does NOT match the current setup:\n"
                   + "\n".join(_summary_diff(old_summary, new_summary))
                   + "\n\nThe cached samples were built for a different network / PoIs / epsilon, "
                     "so loading them would silently corrupt everything downstream. "
                     "Set rerun_preparation = True and re-run Section 3 to rebuild the state.")
        return ("MISMATCH", message)

    # --- sampling parameters: compared only when the caller supplies a
    # requested_* value; a stored attribute may be None on an older file ---
    requested = {
        "n_vertices": requested_n_vertices,
        "seed": requested_seed,
        "n_samples_per_level": requested_n_samples_per_level,
    }
    mismatches = []
    unverifiable = []
    for key, want in requested.items():
        if want is None:
            continue  # caller didn't ask to check this one
        have = state.get(key)
        if have is None:
            unverifiable.append(key)
        elif have != want:
            mismatches.append(f"  - {key}: saved={have!r}  →  current={want!r}")

    if mismatches:
        message = ("Preparation state does NOT match the current sampling parameters:\n"
                   + "\n".join(mismatches)
                   + "\n\nThe cached samples were drawn with different sampling parameters, "
                     "so loading them would silently corrupt everything downstream. "
                     "Set rerun_preparation = True and re-run Section 3 to rebuild the state.")
        return ("MISMATCH", message)

    if unverifiable:
        message = ("Preparation state matches the current network, PoI definitions, and "
                    "epsilon, but the following sampling-parameter attribute(s) are missing "
                    f"from this file (older file predates them): {', '.join(unverifiable)}. "
                    "Cannot verify those against the current sampling parameters. Proceeding "
                    "unverified for them — if results look wrong, set rerun_preparation = True "
                    "to rebuild it.")
        return ("UNVERIFIED", message)

    return ("PASS",
            "Preparation state matches the current network, PoI definitions, and epsilon.")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _dim_name(key: str, axis: int) -> str:
    """Build the netCDF dimension name for one axis of one solution key.

    Parameters
    ----------
    key : str
        Solution key, e.g. ``"Generator_p_nom_opt"``.
    axis : int
        Index of the trailing axis, counted after the leading sample axis.

    Returns
    -------
    str
        ``"<key>__dim<axis>"``, with spaces in `key` replaced by underscores. Only
        spaces are replaced; any other character a solution key may carry is passed
        through to netCDF unchanged.
    """
    safe = key.replace(" ", "_")
    return f"{safe}__dim{axis}"


def _safe_remove(path: str):
    """Delete a file if possible, ignoring the failure if it is not.

    Parameters
    ----------
    path : str
        File to remove.

    Returns
    -------
    None

    Notes
    -----
    Every ``OSError`` is swallowed, a missing file and a permission error alike.
    Used only on :func:`save`'s error path, where the original exception is about to
    be re-raised and must not be masked by a cleanup failure.
    """
    try:
        os.remove(path)
    except OSError:
        pass


def save(path: str, P_all, v, Gamma, solutions, p_star, epsilon, poi_names,
         fingerprint=None, n_vertices=None, n_interior_levels=None,
         n_samples_per_level=None, epsilon_min=None, seed=None):
    """Save the preparation phase output to one .nc file, written with netCDF4.

    Parameters
    ----------
    path : str
        Destination, e.g. ``"data/preparation_state.nc"``. Missing parent
        directories are created.
    P_all : numpy.ndarray
        Shape ``(m, n)``. Sample matrix, one PoI point in GW per column.
    v : numpy.ndarray
        Shape ``(n,)``. Minimum cost at each sample, in EUR/yr.
    Gamma : numpy.ndarray
        Shape ``(m, n)``. PoI duals per sample, in EUR/yr per GW.
    solutions : list of dict
        One :func:`mga_engine.gp_solver._extract_solution` dict per sample, in the
        column order of `P_all`. All of them must share keys and per-key shapes.
    p_star : numpy.ndarray
        Shape ``(m,)``. PoI vector in GW at the cost optimum.
    epsilon : float
        Near-optimality slack the preparation ran at.
    poi_names : list of str
        PoI name per row of `P_all`, in order; length must equal ``m``.
    fingerprint : dict or str, optional
        The bundle from :func:`make_fingerprint`, or a bare hash string. Stored as
        global attributes when given; omitting it simply writes an unverifiable file.
    n_vertices, n_interior_levels, n_samples_per_level, seed : int, optional
        Sampling parameters the preparation was run with.
    epsilon_min : float, optional
        Tightest interior level the preparation used.

    Returns
    -------
    None
        Writes the file and prints a confirmation line.

    Raises
    ------
    AssertionError
        If ``len(poi_names)`` does not equal `P_all`'s row count. A plain ``assert``,
        so it vanishes under ``python -O``.
    ValueError
        If the solution dicts disagree on keys or on any key's shape, naming the
        offending sample; also from the shape unpacking if `P_all` is not 2-D.
    IndexError
        If `solutions` is empty.

    Notes
    -----
    Each optional sampling parameter is stored only if it is not ``None``, and
    :func:`load` returns ``None`` for whatever is absent; that is also how
    :func:`check_state` tells "not stored" from "does not match".
    The write is atomic: the file is built as ``path + ".tmp"`` and only
    ``os.replace``-d into place after a clean close, so an interrupted save can never
    truncate the existing good .nc. It does OVERWRITE any file already at `path`.
    Everything is written as f8 and uncompressed, so integer and boolean solution
    values come back as floats and the files are large: roughly 90 MiB for 230
    samples of a 24-bus network.
    ``poi_names`` round-trips as one newline-joined attribute, so a PoI name
    containing a newline would split into two names on load.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    P_all  = np.asarray(P_all,  dtype="f8")
    v      = np.asarray(v,      dtype="f8")
    Gamma  = np.asarray(Gamma,  dtype="f8")
    p_star = np.asarray(p_star, dtype="f8")

    m_dim, n_samples = P_all.shape

    assert len(poi_names) == m_dim, (
        f"poi_names has {len(poi_names)} entries but P_all has {m_dim} PoI rows"
    )

    # Pre-flight: every solution dict must have the same keys and per-key array
    # shapes as solutions[0]; otherwise the np.stack below fails inscrutably.
    ref_keys = set(solutions[0].keys())
    ref_shapes = {k: np.asarray(solutions[0][k]).shape for k in ref_keys}
    for s_idx, sol in enumerate(solutions[1:], start=1):
        if set(sol.keys()) != ref_keys:
            missing = sorted(ref_keys - set(sol.keys()))
            extra = sorted(set(sol.keys()) - ref_keys)
            raise ValueError(
                f"solutions[{s_idx}] keys differ from solutions[0]: "
                f"missing={missing}, extra={extra}"
            )
        for k in ref_keys:
            shape = np.asarray(sol[k]).shape
            if shape != ref_shapes[k]:
                raise ValueError(
                    f"solutions[{s_idx}][{k!r}] has shape {shape}, but "
                    f"solutions[0][{k!r}] has shape {ref_shapes[k]}"
                )

    tmp = path + ".tmp"
    ds = Dataset(tmp, mode="w", format="NETCDF4")
    try:
        # --- core dimensions ---
        ds.createDimension("poi", m_dim)
        ds.createDimension("sample", n_samples)

        # --- scalar / small metadata (global attributes) ---
        ds.epsilon = float(epsilon)
        if n_vertices is not None:
            ds.n_vertices = int(n_vertices)
        if n_interior_levels is not None:
            ds.n_interior_levels = int(n_interior_levels)
        if n_samples_per_level is not None:
            ds.n_samples_per_level = int(n_samples_per_level)
        if epsilon_min is not None:
            ds.epsilon_min = float(epsilon_min)
        if seed is not None:
            ds.seed = int(seed)
        # newline-joined so it round-trips as a single string attribute
        ds.poi_names = "\n".join(poi_names)
        if fingerprint is not None:
            _write_fingerprint_attrs(ds, fingerprint)

        # --- core arrays ---
        ds.createVariable("P_all", "f8", ("poi", "sample"))[:, :] = P_all
        ds.createVariable("v",     "f8", ("sample",))[:]          = v
        ds.createVariable("Gamma", "f8", ("poi", "sample"))[:, :] = Gamma
        ds.createVariable("p_star", "f8", ("poi",))[:]            = p_star

        # --- solution arrays stacked across samples ---
        # Each key -> array of shape (n_samples, *per_sample_shape).
        # One named dimension is created per trailing axis, sized to that axis.
        for key in solutions[0].keys():
            arr = np.stack([np.asarray(sol[key]) for sol in solutions], axis=0)
            arr = np.asarray(arr, dtype="f8")

            dim_names = ["sample"]
            for axis, size in enumerate(arr.shape[1:]):
                dname = _dim_name(key, axis)
                if dname not in ds.dimensions:
                    ds.createDimension(dname, size)
                dim_names.append(dname)

            ds.createVariable(key, "f8", tuple(dim_names))[...] = arr
    except BaseException:
        ds.close()
        _safe_remove(tmp)
        raise
    else:
        ds.close()

    os.replace(tmp, path)   # atomic swap into place after a clean write
    print(f"[state] Saved preparation state to {path}  ({n_samples} samples)")


def load(path: str) -> dict:
    """Load a preparation state from a .nc file written by :func:`save`.

    Parameters
    ----------
    path : str
        File to read.

    Returns
    -------
    dict
        ``"P_all"`` ``(m, n)``, ``"v"`` ``(n,)``, ``"Gamma"`` ``(m, n)`` and
        ``"p_star"`` ``(m,)`` as float arrays; ``"epsilon"``, ``"n_vertices"``,
        ``"n_interior_levels"``, ``"n_samples_per_level"``, ``"epsilon_min"``,
        ``"seed"``, ``"poi_names"``, ``"fingerprint"``, ``"fingerprint_version"``
        and ``"fingerprint_summary"`` from the global attributes, each ``None`` when
        the file predates it; and ``"solutions"``, a list of n dicts rebuilt one
        sample at a time.

    Raises
    ------
    OSError
        From netCDF4 if `path` is missing or unreadable.
    KeyError
        If one of the four core variables, or the ``sample`` dimension, is absent.

    Notes
    -----
    Every variable other than ``P_all``, ``v``, ``Gamma`` and ``p_star`` is treated
    as a per-sample solution key, which is why :func:`_write_fingerprint_attrs`
    stores the fingerprint as attributes and not as variables.
    A ``fingerprint_summary`` that is not valid JSON becomes ``None`` silently, which
    leaves :func:`check_state` able to compare the hash but unable to say what
    differs.
    All samples are read into memory at once, so peak memory is the whole file.
    Prints a confirmation line.
    """
    ds = Dataset(path, mode="r")
    try:
        P_all  = np.asarray(ds.variables["P_all"][:],  dtype="f8")
        v      = np.asarray(ds.variables["v"][:],      dtype="f8")
        Gamma  = np.asarray(ds.variables["Gamma"][:],  dtype="f8")
        p_star = np.asarray(ds.variables["p_star"][:], dtype="f8")
        n      = ds.dimensions["sample"].size

        epsilon   = float(ds.epsilon) if hasattr(ds, "epsilon") else None
        poi_names = ds.poi_names.split("\n") if hasattr(ds, "poi_names") else None

        # Sampling parameters (absent in files saved before they were added,
        # including files from the short-lived n_samples attribute era).
        n_vertices = int(ds.n_vertices) if hasattr(ds, "n_vertices") else None
        n_interior_levels = (int(ds.n_interior_levels)
                             if hasattr(ds, "n_interior_levels") else None)
        n_samples_per_level = (int(ds.n_samples_per_level)
                               if hasattr(ds, "n_samples_per_level") else None)
        epsilon_min = float(ds.epsilon_min) if hasattr(ds, "epsilon_min") else None
        seed = int(ds.seed) if hasattr(ds, "seed") else None

        fingerprint = str(ds.fingerprint) if hasattr(ds, "fingerprint") else None
        fingerprint_version = (int(ds.fingerprint_version)
                               if hasattr(ds, "fingerprint_version") else None)
        if hasattr(ds, "fingerprint_summary"):
            try:
                fingerprint_summary_val = json.loads(ds.fingerprint_summary)
            except (ValueError, TypeError):
                fingerprint_summary_val = None
        else:
            fingerprint_summary_val = None

        reserved = {"P_all", "v", "Gamma", "p_star"}
        solution_keys = [k for k in ds.variables if k not in reserved]

        solutions = []
        for i in range(n):
            sol = {key: np.asarray(ds.variables[key][i], dtype="f8")
                   for key in solution_keys}
            solutions.append(sol)
    finally:
        ds.close()

    print(f"[state] Loaded preparation state from {path}  ({n} samples)")
    return {
        "P_all": P_all,
        "v": v,
        "Gamma": Gamma,
        "p_star": p_star,
        "epsilon": epsilon,
        "n_vertices": n_vertices,
        "n_interior_levels": n_interior_levels,
        "n_samples_per_level": n_samples_per_level,
        "epsilon_min": epsilon_min,
        "seed": seed,
        "poi_names": poi_names,
        "fingerprint": fingerprint,
        "fingerprint_version": fingerprint_version,
        "fingerprint_summary": fingerprint_summary_val,
        "solutions": solutions,
    }


def run_preparation(
    build_network_fn,
    make_poi_specs_fn,
    epsilon: float = 0.05,
    n_vertices: int = 30,
    seed: int = 42,
    n_interior_levels: int = 5,
    n_samples_per_level: int = 10,
    epsilon_min: float = 0.005,
):
    """Run the whole preparation phase and return everything :func:`save` needs.

    The heavy step: a base solve for c'x* and p*, vertex sampling at `epsilon`,
    interior sampling down the derived epsilon levels, then one GP solve per sample.
    The exploration notebook calls this directly, so a user never has to run a
    separate backend script.

    Parameters
    ----------
    build_network_fn : callable
        ``build_network_fn() -> pypsa.Network``, returning a FRESH network each time.
        Called ``2 + n_interior_levels + 1`` times, because each sampling stage
        builds its own optimisation model.
    make_poi_specs_fn : callable
        ``make_poi_specs_fn(network) -> list of PoiSpec``. Must produce the same
        PoIs, in the same order, on every call; that order fixes the rows of `P_all`.
    epsilon : float, default 0.05
        Relative cost slack for the outer (vertex) boundary.
    n_vertices : int, default 30
        Number of vertex MP(r) samples.
    seed : int, default 42
        RNG seed. Used directly for the vertex sampling, and the per-level interior
        seeds are derived from it with ``SeedSequence.spawn()``, so one seed makes
        the whole sample set reproducible.
    n_interior_levels : int, default 5
        Number of interior epsilon levels, derived from `epsilon` down to
        `epsilon_min` by :func:`mga_engine.interior_sampling.derive_epsilon_levels`.
    n_samples_per_level : int, default 10
        Interior MP(r) samples per level.
    epsilon_min : float, default 0.005
        Tightest interior level. Must satisfy ``0 < epsilon_min < epsilon``.

    Returns
    -------
    P_all : numpy.ndarray
        Shape ``(m, n_vertices + n_interior_levels * n_samples_per_level)``. The
        vertex samples first, then the interior ones, in level order.
    v : numpy.ndarray
        Shape ``(n,)``. Minimum cost at each sample, in EUR/yr.
    Gamma : numpy.ndarray
        Shape ``(m, n)``. PoI duals per sample, in EUR/yr per GW.
    solutions : list of dict
        One full solution dict per sample, in column order.
    p_star : numpy.ndarray
        Shape ``(m,)``. PoI vector in GW at the cost optimum.
    poi_names : list of str
        PoI name per row of `P_all`, from the specs built on the base network.

    Raises
    ------
    AssertionError
        From :func:`mga_engine.interior_sampling.derive_epsilon_levels` if the
        epsilon levels are not well formed.
    RuntimeError
        From the first failed MP(r) or GP solve. The run aborts with no partial
        result, after however many solves already succeeded.
    ValueError
        From :func:`mga_engine.poi.make_poi_specs` if a PoI definition does not match
        the network.

    Notes
    -----
    The six return values feed straight into :func:`save`; ``opt_cost`` is NOT among
    them and is not saved, so a caller that needs c'x* has to recompute it.
    The base ``optimize`` call's status is not checked, so an infeasible or
    unbounded base network yields a meaningless `opt_cost` and the sampling proceeds
    from it rather than failing here.
    One base solve plus ``n_vertices + n_interior_levels * n_samples_per_level``
    MP(r) solves and the same number of GP solves; for the defaults that is
    1 + 80 + 80. Many prints, no file I/O.
    The four pipeline modules, :func:`mga_engine.gp_solver.solve_all_gp` among them,
    are imported inside this function rather than at module scope. The measured
    effect is that importing this module pulls in neither pypsa nor linopy: the
    state-loading path, :func:`load` and :func:`check_state`, stays free of the
    sampling stack. There is no circular import between these modules that would
    force it.
    TODO(alaa): was avoiding the pypsa/linopy import cost on the load-only path the
    actual reason for the lazy imports, or was there another one worth recording?
    """
    from mga_engine.poi import evaluate_all
    from mga_engine.vertex_sampling import sample_vertices
    from mga_engine.interior_sampling import sample_interior, derive_epsilon_levels
    from mga_engine.gp_solver import solve_all_gp

    # --- base solve: optimal cost and p* ---
    network = build_network_fn()
    network.optimize(solver_name="highs", include_objective_constant=False,
                     solver_options={"output_flag": False})
    opt_cost = network.objective
    poi_specs = make_poi_specs_fn(network)
    p_star = evaluate_all(poi_specs, network)

    # --- vertex samples (outer boundary at epsilon) ---
    net_v = build_network_fn()
    P_vertices = sample_vertices(net_v, make_poi_specs_fn(net_v), opt_cost,
                                 epsilon=epsilon, n_samples=n_vertices, seed=seed)

    # --- interior samples (decreasing epsilon levels derived from epsilon) ---
    epsilon_levels = derive_epsilon_levels(epsilon, n_interior_levels, epsilon_min)
    P_interior = sample_interior(build_network_fn, make_poi_specs_fn, opt_cost,
                                 epsilon_levels=epsilon_levels,
                                 samples_per_level=n_samples_per_level,
                                 seed=seed)

    P_all = np.hstack([P_vertices, P_interior])

    # --- GP solve per sample (cost + duals + full solution) ---
    v, Gamma, solutions = solve_all_gp(P_all, build_network_fn, make_poi_specs_fn)

    poi_names = [s.name for s in poi_specs]
    return P_all, v, Gamma, solutions, p_star, poi_names
