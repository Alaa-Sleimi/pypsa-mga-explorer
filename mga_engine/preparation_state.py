"""
preparation_state.py — Save and load the full preparation phase output.

Saves everything needed for the exploration phase into a single .nc file
using the netCDF4 library directly (no xarray). Writing the netCDF file
without the xarray layer is faster on the large solution arrays produced
by bigger models, which is what we benchmark on.

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
# Network fingerprint — a cheap, solver-free structural hash used to detect a
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
    """Round a float to `sig` significant figures; non-finite -> stable sentinel.

    Significant-figure rounding keeps the fingerprint stable across float-repr
    noise while still catching real changes in capacities/costs at any scale.
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
    """Canonicalise a single static-table cell to a JSON-stable primitive."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x
    return _num(x)


def _timeseries_digest(df):
    """Stable sha256 hex of a (rounded) time-varying frame, or None if empty.

    Columns are sorted so column ordering can't change the digest; values are
    rounded to 6 decimals and -0.0 normalised so float noise doesn't either.
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
    """Build the canonical (JSON-serialisable) structure that gets hashed."""
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
        # PoI order matters (it fixes P_all's rows), so keep it as given.
        "poi_definitions": [
            [d.get("carrier"), d.get("name"), d.get("unit", "GW")]
            for d in poi_definitions
        ],
    }


def network_fingerprint(network, poi_definitions, epsilon) -> str:
    """Return a sha256 hex fingerprint of the network structure, PoI definitions
    (in order) and epsilon.

    Pure and cheap: reads only pre-solve metadata (no ``optimize`` call, no
    ``*_opt`` result columns), so it is deterministic and solve-invariant.
    """
    obj = _fingerprint_object(network, poi_definitions, epsilon)
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_summary(network, poi_definitions, epsilon) -> dict:
    """Small, human-readable summary stored alongside the fingerprint, used to
    explain *what* differs on a mismatch. Not the source of truth (the hash is)."""
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
    """Bundle the three fingerprint pieces to hand to save(fingerprint=...)."""
    return {
        "fingerprint": network_fingerprint(network, poi_definitions, epsilon),
        "fingerprint_version": FINGERPRINT_VERSION,
        "fingerprint_summary": fingerprint_summary(network, poi_definitions, epsilon),
    }


def _write_fingerprint_attrs(ds, fingerprint):
    """Write the fingerprint bundle as GLOBAL ATTRIBUTES only.

    Never as variables: load() treats every non-reserved variable as a per-sample
    solution key, so a fingerprint variable would corrupt solution reconstruction.
    Accepts either the bundle dict from make_fingerprint() or a bare hash string.
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
    """Human-readable lines describing how two fingerprint summaries differ."""
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


def check_state(state, network, poi_definitions, requested_epsilon):
    """Compare a loaded state's fingerprint against the current setup.

    Returns
    -------
    (status, message) where status is one of:
        "PASS"       — fingerprint matches this network / PoIs / epsilon
        "MISMATCH"   — a real mismatch; message diffs the summaries
        "UNVERIFIED" — no/incomparable fingerprint (old file or version bump)

    Pure and cheap — recomputes the current fingerprint (no solver calls) and
    compares strings.
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
    if current == stored:
        return ("PASS",
                "Preparation state matches the current network, PoI definitions, and epsilon.")

    old_summary = state.get("fingerprint_summary") or {}
    new_summary = fingerprint_summary(network, poi_definitions, requested_epsilon)
    message = ("Preparation state does NOT match the current setup:\n"
               + "\n".join(_summary_diff(old_summary, new_summary))
               + "\n\nThe cached samples were built for a different network / PoIs / epsilon, "
                 "so loading them would silently corrupt everything downstream. "
                 "Set rerun_preparation = True and re-run Section 3 to rebuild the state.")
    return ("MISMATCH", message)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _dim_name(key: str, axis: int) -> str:
    """Per-key, per-axis dimension name. Sanitised to avoid odd characters."""
    safe = key.replace(" ", "_")
    return f"{safe}__dim{axis}"


def _safe_remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def save(path: str, P_all, v, Gamma, solutions, p_star, epsilon, poi_names,
         fingerprint=None, n_vertices=None, n_interior_levels=None,
         n_samples_per_level=None, epsilon_min=None):
    """
    Save preparation phase output to a single .nc file using netCDF4 directly.

    Parameters
    ----------
    path        : file path, e.g. "data/preparation_state.nc"
    P_all       : np.ndarray (m, n)
    v           : np.ndarray (n,)
    Gamma       : np.ndarray (m, n)
    solutions   : list of dicts — one dict per sample
    p_star      : np.ndarray (m,)  PoI vector at the optimal point
    epsilon     : float            near-optimality threshold used in preparation
    poi_names   : list[str]        PoI name per row of P_all, in order (length m)
    fingerprint : optional network fingerprint bundle from make_fingerprint()
                  (or a bare hash string). When given, it is stored as global
                  attributes; omitting it keeps existing callers unchanged.
    n_vertices, n_interior_levels, n_samples_per_level, epsilon_min :
                  optional sampling parameters the preparation was run with.
                  Each one given is stored as a global attribute so load()
                  can hand it back for checking; omitted ones are not stored.

    The write is atomic: the file is built as ``path + ".tmp"`` and only
    ``os.replace``-d into place after a clean close, so an interrupted save can
    never truncate or corrupt the existing good .nc.
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
    """
    Load preparation phase output from a .nc file written by save().

    Returns
    -------
    dict with keys: P_all, v, Gamma, p_star, epsilon, n_vertices,
    n_interior_levels, n_samples_per_level, epsilon_min, poi_names,
    solutions, fingerprint, fingerprint_version, fingerprint_summary

    Notes
    -----
    epsilon, poi_names and the three fingerprint fields are read from global
    attributes. Files written before any of them were added will not have them;
    in that case they are returned as None (backward compatible).
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
    """
    Run the full preparation phase on an arbitrary network and return everything
    save() needs. This is the heavy step (many solves) that produces a
    preparation_state .nc — the exploration notebook calls this directly, so a
    user never has to run a separate backend script.

    Parameters
    ----------
    build_network_fn  : callable returning a FRESH network. Called several times,
                        since the sampling builds a new optimisation model each
                        time it runs.
    make_poi_specs_fn : callable taking a network -> list[PoiSpec]. Must produce
                        the same PoIs, in the same order, you intend to explore.
    epsilon           : near-optimality slack for the outer (vertex) boundary.
    n_vertices        : number of vertex MP(r) samples.
    seed              : RNG seed for vertex sampling.
    n_interior_levels : number of interior epsilon levels (derived from epsilon
                        down to epsilon_min via derive_epsilon_levels).
    n_samples_per_level : interior MP(r) samples per epsilon level.
    epsilon_min       : smallest interior level; must be > 0 and < epsilon.

    Returns
    -------
    (P_all, v, Gamma, solutions, p_star, poi_names) — feed straight into save().
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
                                 samples_per_level=n_samples_per_level)

    P_all = np.hstack([P_vertices, P_interior])

    # --- GP solve per sample (cost + duals + full solution) ---
    v, Gamma, solutions = solve_all_gp(P_all, build_network_fn, make_poi_specs_fn)

    poi_names = [s.name for s in poi_specs]
    return P_all, v, Gamma, solutions, p_star, poi_names


if __name__ == "__main__":
    import logging, warnings
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("pypsa").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="linopy")

    from mga_engine.network import build_network
    from mga_engine.poi import make_poi_specs, POI_DEFINITIONS

    EPSILON = 0.05
    N_VERTICES = 30
    N_INTERIOR_LEVELS = 5
    N_SAMPLES_PER_LEVEL = 10
    EPSILON_MIN = 0.005

    # Full preparation pipeline on the example network (same call the notebook makes)
    P_all, v, Gamma, solutions, p_star, poi_names = run_preparation(
        build_network, make_poi_specs, epsilon=EPSILON, n_vertices=N_VERTICES,
        n_interior_levels=N_INTERIOR_LEVELS, n_samples_per_level=N_SAMPLES_PER_LEVEL,
        epsilon_min=EPSILON_MIN,
    )

    # --- fingerprint (cheap, no solve): a fresh unsolved network is enough ---
    fingerprint = make_fingerprint(build_network(), POI_DEFINITIONS, EPSILON)

    # --- save ---
    save("data/preparation_state.nc", P_all, v, Gamma, solutions, p_star, EPSILON,
         poi_names, fingerprint=fingerprint, n_vertices=N_VERTICES,
         n_interior_levels=N_INTERIOR_LEVELS, n_samples_per_level=N_SAMPLES_PER_LEVEL,
         epsilon_min=EPSILON_MIN)

    # --- reload and verify ---
    state = load("data/preparation_state.nc")
    assert np.allclose(state["P_all"], P_all),         "P_all mismatch!"
    assert np.allclose(state["v"], v),                 "v mismatch!"
    assert np.allclose(state["Gamma"], Gamma),         "Gamma mismatch!"
    assert np.allclose(state["p_star"], p_star),       "p_star mismatch!"
    assert state["poi_names"] == poi_names,            "poi_names mismatch!"
    assert state["epsilon"] == EPSILON,                "epsilon mismatch!"
    assert len(state["solutions"]) == len(solutions),  "solutions length mismatch!"
    assert state["fingerprint"] == fingerprint["fingerprint"], "fingerprint mismatch!"

    status, message = check_state(state, build_network(), POI_DEFINITIONS, EPSILON)
    assert status == "PASS", f"check_state expected PASS, got {status}: {message}"

    print("[state] Verification passed — save/load is consistent.")
    print(f"[state] epsilon = {state['epsilon']}")
    print(f"[state] poi_names = {state['poi_names']}")
    print(f"[state] fingerprint = {state['fingerprint'][:16]}…  (v{state['fingerprint_version']})")
    print(f"[state] check_state: {status} — {message}")
    print(f"[state] p_star = {p_star}")
    print(f"[state] Keys in first loaded solution: {list(state['solutions'][0].keys())}")
