"""
tests/test_smoke.py — fast, deterministic smoke tests for the MGA engine.

Scope and safety rules this file obeys:
  * Nothing here writes to data/. Every test that saves a preparation state
    writes into pytest's `tmp_path` and nowhere else.
  * The one test that touches data/preparation_state.nc (test_navigate_*) reads
    it READ-ONLY via the `prepared_state` fixture in conftest.py, and skips if
    the file is missing rather than regenerating it.
  * Only the navigate test is "slow" (it loads the real .nc). Everything else is
    synthetic: no PyPSA optimize, no solver, no network solve. The one PyPSA
    object built (for check_state's fingerprint) is never optimised.
  * All randomness is avoided outright — every array below is written by hand.

Run with:
    pytest tests/test_smoke.py -v
"""

import numpy as np
import pytest

from mga_engine.ccp_solver import solve_ccp
from mga_engine.interior_sampling import derive_epsilon_levels
from mga_engine.navigation import navigate
from mga_engine.preparation_state import check_state, load, make_fingerprint, save


# ===========================================================================
# Shared synthetic fixtures (hand-built, no solver, no randomness)
# ===========================================================================

# Unit square in 2 PoI dimensions, one column per sample point.
# Every column is a vertex of the hull, so each has a UNIQUE representation as a
# convex combination -> solve_ccp's alpha at a column is pinned down exactly.
#            col0     col1     col2     col3
#          (0, 0)   (1, 0)   (1, 1)   (0, 1)
SQUARE_P = np.array(
    [[0.0, 1.0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 1.0]]
)
SQUARE_V = np.array([10.0, 20.0, 40.0, 30.0])


def _synthetic_state(m=2, n=3, static_dim=4, dynamic_dims=(5, 4)):
    """Build a tiny, fully hand-written preparation state.

    Shapes mirror what save()/load() expect:
        P_all  (m, n) | v (n,) | Gamma (m, n) | p_star (m,)
        solutions: list of n dicts, identical keys and per-key shapes
    """
    P_all = np.arange(m * n, dtype="f8").reshape(m, n) + 1.0
    v = np.array([100.0, 200.0, 300.0])[:n]
    Gamma = -(np.arange(m * n, dtype="f8").reshape(m, n) + 0.5)
    p_star = np.array([1.25, 3.5])[:m]
    poi_names = ["solar_cap_gw", "wind_cap_gw"][:m]

    solutions = []
    for k in range(n):
        solutions.append({
            # static result frame, e.g. Generator_p_nom_opt : (n_generators,)
            "Generator_p_nom_opt": np.full(static_dim, float(k), dtype="f8"),
            # dynamic result frame, e.g. Generator_p : (n_snapshots, n_generators)
            "Generator_p": np.full(dynamic_dims, float(k) * 0.5, dtype="f8"),
        })

    return {
        "P_all": P_all,
        "v": v,
        "Gamma": Gamma,
        "solutions": solutions,
        "p_star": p_star,
        "poi_names": poi_names,
    }


@pytest.fixture
def synthetic_state():
    return _synthetic_state()


# ===========================================================================
# A. solve_ccp on a synthetic hull (no solver, no network)
# ===========================================================================

@pytest.mark.parametrize("col", [0, 1, 2, 3])
def test_solve_ccp_recovers_a_known_vertex(col):
    """A target equal to column `col` of P must be recovered as alpha == e_col.

    Because every column of SQUARE_P is a hull vertex, the equality system
    P alpha = p, alpha'1 = 1, alpha >= 0 has exactly one solution, so this is a
    genuine determinism check and not just "some feasible alpha".
    """
    p = SQUARE_P[:, col]

    alpha, cost_approx = solve_ccp(p, SQUARE_P, SQUARE_V)

    expected = np.zeros(SQUARE_P.shape[1])
    expected[col] = 1.0

    assert alpha.shape == (SQUARE_P.shape[1],)
    assert np.allclose(alpha, expected, atol=1e-9)
    assert np.isclose(cost_approx, SQUARE_V[col], atol=1e-9)
    # cost_approx must be exactly the reported alpha'v, not an independent value
    assert np.isclose(cost_approx, float(alpha @ SQUARE_V), atol=1e-12)


def test_solve_ccp_weights_are_a_convex_combination():
    """For an interior target, alpha must still be non-negative and sum to 1,
    and P @ alpha must reproduce the target."""
    p = np.array([0.5, 0.25])   # strictly inside the unit square

    alpha, cost_approx = solve_ccp(p, SQUARE_P, SQUARE_V)

    assert np.all(alpha >= -1e-12), f"negative weight in alpha={alpha}"
    assert np.isclose(alpha.sum(), 1.0, atol=1e-9)
    assert np.allclose(SQUARE_P @ alpha, p, atol=1e-9)
    # The LP minimises alpha'v, so the surrogate cost can never exceed the most
    # expensive sample nor fall below the cheapest.
    assert SQUARE_V.min() - 1e-9 <= cost_approx <= SQUARE_V.max() + 1e-9


def test_solve_ccp_raises_outside_the_hull():
    """A target far outside the sampled hull must raise RuntimeError with the
    'outside the sampled convex hull' guidance (ccp_solver.py status == 2 path)."""
    p_outside = np.array([5.0, 5.0])

    with pytest.raises(RuntimeError, match="outside the sampled convex hull"):
        solve_ccp(p_outside, SQUARE_P, SQUARE_V)


# ===========================================================================
# B. preparation_state save/load round-trip (tmp_path only — never data/)
# ===========================================================================

def test_save_load_round_trip(tmp_path, synthetic_state):
    """save() then load() must return bit-identical arrays and round-trip the
    scalar metadata, including the `seed` sampling parameter."""
    path = str(tmp_path / "preparation_state.nc")
    epsilon = 0.05
    seed = 1234

    save(
        path,
        synthetic_state["P_all"],
        synthetic_state["v"],
        synthetic_state["Gamma"],
        synthetic_state["solutions"],
        synthetic_state["p_star"],
        epsilon,
        synthetic_state["poi_names"],
        n_vertices=7,
        n_interior_levels=3,
        n_samples_per_level=2,
        epsilon_min=0.005,
        seed=seed,
    )

    loaded = load(path)

    np.testing.assert_array_equal(loaded["P_all"], synthetic_state["P_all"])
    np.testing.assert_array_equal(loaded["v"], synthetic_state["v"])
    np.testing.assert_array_equal(loaded["Gamma"], synthetic_state["Gamma"])
    np.testing.assert_array_equal(loaded["p_star"], synthetic_state["p_star"])

    assert loaded["poi_names"] == synthetic_state["poi_names"]
    assert np.isclose(loaded["epsilon"], epsilon)

    # sampling parameters, incl. the seed this suite specifically locks in
    assert loaded["seed"] == seed
    assert loaded["n_vertices"] == 7
    assert loaded["n_interior_levels"] == 3
    assert loaded["n_samples_per_level"] == 2
    assert np.isclose(loaded["epsilon_min"], 0.005)

    # solutions: same count, same keys, same values per sample
    assert len(loaded["solutions"]) == len(synthetic_state["solutions"])
    for got, want in zip(loaded["solutions"], synthetic_state["solutions"]):
        assert set(got.keys()) == set(want.keys())
        for key in want:
            np.testing.assert_allclose(got[key], want[key])


def test_save_omits_absent_optional_metadata(tmp_path, synthetic_state):
    """Optional sampling parameters are only stored when supplied; a file saved
    without them loads them back as None (the backward-compatible path)."""
    path = str(tmp_path / "no_optionals.nc")

    save(
        path,
        synthetic_state["P_all"],
        synthetic_state["v"],
        synthetic_state["Gamma"],
        synthetic_state["solutions"],
        synthetic_state["p_star"],
        0.05,
        synthetic_state["poi_names"],
    )

    loaded = load(path)
    assert loaded["seed"] is None
    assert loaded["n_vertices"] is None
    assert loaded["fingerprint"] is None
    assert loaded["fingerprint_version"] is None
    assert loaded["fingerprint_summary"] is None


# ===========================================================================
# C. check_state seed logic (tri-state PASS / MISMATCH / UNVERIFIED)
# ===========================================================================

EPSILON_FOR_FINGERPRINT = 0.05


@pytest.fixture(scope="module")
def fingerprint_network():
    """An UNSOLVED example network + its PoI definitions.

    network_fingerprint()/check_state() read only pre-solve metadata, so no
    optimize() call is needed anywhere in this fixture — no solver runs.
    """
    from mga_engine.network import build_network
    from mga_engine.poi import POI_DEFINITIONS

    return build_network(), POI_DEFINITIONS


def _save_state_with_fingerprint(path, state, network, poi_definitions, seed):
    """Save `state` to `path` with a real fingerprint for `network`, and `seed`."""
    fingerprint = make_fingerprint(network, poi_definitions, EPSILON_FOR_FINGERPRINT)
    save(
        path,
        state["P_all"], state["v"], state["Gamma"], state["solutions"],
        state["p_star"], EPSILON_FOR_FINGERPRINT, state["poi_names"],
        fingerprint=fingerprint,
        seed=seed,
    )
    return load(path)


def test_check_state_matching_seed_passes(tmp_path, synthetic_state, fingerprint_network):
    """Same network / PoIs / epsilon AND the same seed -> PASS."""
    network, poi_definitions = fingerprint_network
    saved_seed = 42

    loaded = _save_state_with_fingerprint(
        str(tmp_path / "seeded.nc"), synthetic_state,
        network, poi_definitions, saved_seed,
    )
    assert loaded["seed"] == saved_seed

    status, message = check_state(
        loaded, network, poi_definitions, EPSILON_FOR_FINGERPRINT,
        requested_seed=saved_seed,
    )
    assert status == "PASS", message


def test_check_state_different_seed_mismatches(tmp_path, synthetic_state, fingerprint_network):
    """Same network / PoIs / epsilon but a DIFFERENT seed -> MISMATCH.

    This is the staleness detection the cache relies on: samples drawn with
    another seed are a different sample set even though the model is identical.
    """
    network, poi_definitions = fingerprint_network
    saved_seed = 42

    loaded = _save_state_with_fingerprint(
        str(tmp_path / "seeded.nc"), synthetic_state,
        network, poi_definitions, saved_seed,
    )

    status, message = check_state(
        loaded, network, poi_definitions, EPSILON_FOR_FINGERPRINT,
        requested_seed=saved_seed + 1,
    )
    assert status == "MISMATCH", message
    assert "seed" in message


def test_check_state_seed_not_requested_still_passes(tmp_path, synthetic_state,
                                                    fingerprint_network):
    """requested_seed=None means 'don't check this one' -> PASS regardless."""
    network, poi_definitions = fingerprint_network

    loaded = _save_state_with_fingerprint(
        str(tmp_path / "seeded.nc"), synthetic_state,
        network, poi_definitions, 42,
    )

    status, message = check_state(
        loaded, network, poi_definitions, EPSILON_FOR_FINGERPRINT,
    )
    assert status == "PASS", message


def test_check_state_seed_missing_from_file_is_unverified(tmp_path, synthetic_state,
                                                          fingerprint_network):
    """A seed was requested but the file predates the attribute -> UNVERIFIED
    (the third leg of the tri-state, distinct from MISMATCH)."""
    network, poi_definitions = fingerprint_network
    path = str(tmp_path / "no_seed.nc")

    fingerprint = make_fingerprint(network, poi_definitions, EPSILON_FOR_FINGERPRINT)
    save(
        path,
        synthetic_state["P_all"], synthetic_state["v"], synthetic_state["Gamma"],
        synthetic_state["solutions"], synthetic_state["p_star"],
        EPSILON_FOR_FINGERPRINT, synthetic_state["poi_names"],
        fingerprint=fingerprint,
        # deliberately no seed=
    )
    loaded = load(path)
    assert loaded["seed"] is None

    status, message = check_state(
        loaded, network, poi_definitions, EPSILON_FOR_FINGERPRINT,
        requested_seed=42,
    )
    assert status == "UNVERIFIED", message
    assert "seed" in message


def test_check_state_no_fingerprint_is_unverified(tmp_path, synthetic_state,
                                                  fingerprint_network):
    """A file saved before fingerprinting existed loads fine and reports
    UNVERIFIED rather than crashing."""
    network, poi_definitions = fingerprint_network
    path = str(tmp_path / "no_fingerprint.nc")

    save(
        path,
        synthetic_state["P_all"], synthetic_state["v"], synthetic_state["Gamma"],
        synthetic_state["solutions"], synthetic_state["p_star"],
        EPSILON_FOR_FINGERPRINT, synthetic_state["poi_names"],
        seed=42,
    )
    loaded = load(path)

    status, _ = check_state(
        loaded, network, poi_definitions, EPSILON_FOR_FINGERPRINT,
        requested_seed=42,
    )
    assert status == "UNVERIFIED"


def test_check_state_wrong_epsilon_mismatches(tmp_path, synthetic_state,
                                              fingerprint_network):
    """epsilon is part of the fingerprint, so a different one -> MISMATCH."""
    network, poi_definitions = fingerprint_network

    loaded = _save_state_with_fingerprint(
        str(tmp_path / "seeded.nc"), synthetic_state,
        network, poi_definitions, 42,
    )

    status, _ = check_state(loaded, network, poi_definitions, 0.10, requested_seed=42)
    assert status == "MISMATCH"


# ===========================================================================
# D. derive_epsilon_levels (pure: no network, no solver)
# ===========================================================================

def test_derive_epsilon_levels_reproduces_the_documented_default():
    """The docstring promises (0.05, 5, 0.005) -> the formerly hardcoded list."""
    assert derive_epsilon_levels(0.05, 5, 0.005) == [0.04, 0.03, 0.02, 0.01, 0.005]


def test_derive_epsilon_levels_single_level():
    """n_levels=1 collapses to just epsilon_min (range(1, 1) is empty)."""
    assert derive_epsilon_levels(0.05, 1, 0.005) == [0.005]


@pytest.mark.parametrize(
    "epsilon, n_levels, epsilon_min",
    [
        (0.05, 5, 0.005),
        (0.10, 3, 0.010),
        (0.20, 4, 0.001),
        (0.05, 2, 0.010),
    ],
)
def test_derive_epsilon_levels_invariants(epsilon, n_levels, epsilon_min):
    """Length == n_levels, strictly decreasing, all within (0, epsilon), and the
    last level is exactly epsilon_min.

    NOTE on the parametrisation: strict monotonicity only holds while
    epsilon_min < epsilon / n_levels (the size of one linear step). All cases
    above satisfy that; see the suite summary for the regime where it does not.
    """
    levels = derive_epsilon_levels(epsilon, n_levels, epsilon_min)

    assert len(levels) == n_levels
    assert levels[-1] == epsilon_min
    assert all(0.0 < lvl < epsilon for lvl in levels)
    assert all(a > b for a, b in zip(levels, levels[1:])), levels


@pytest.mark.parametrize(
    "epsilon, n_levels, epsilon_min",
    [
        (0.05, 0, 0.005),     # n_levels must be >= 1
        (0.05, 5, 0.0),       # epsilon_min must be > 0
        (0.05, 5, 0.05),      # epsilon_min must be < epsilon
        (0.05, 5, 0.06),      # epsilon_min must be < epsilon
    ],
)
def test_derive_epsilon_levels_rejects_bad_input(epsilon, n_levels, epsilon_min):
    """The two module-level asserts guard the contract."""
    with pytest.raises(AssertionError):
        derive_epsilon_levels(epsilon, n_levels, epsilon_min)


# ===========================================================================
# E. navigate loop-1 bounds invariant (reads data/preparation_state.nc READ-ONLY)
# ===========================================================================

def test_navigate_loop1_bounds_cover_the_direction_sets(prepared_state):
    """Mirror of the single real assertion in tests/test_navigation.py:

        loop-1 bounds cover exactly the APoIs that ended up in a direction set

    Same scenario C as the existing test — solar (APoI 1) INCREASE by 0.3 GW,
    wind (APoI 2) DECREASE by 5.0 GW (infeasible -> capped/flipped), cost (0)
    last with no direction. No new correctness claim is made here.

    The state is loaded read-only by the `prepared_state` fixture; this test
    never writes to data/.
    """
    P_all = prepared_state["P_all"]
    v = prepared_state["v"]
    p_star = prepared_state["p_star"]

    # Starting point: CCP at p_star, exactly as test_navigation.py does it.
    alpha_s, _cost_s = solve_ccp(p_star, P_all, v)

    tau = [1, 2, 0]
    SG = {1}
    SL = {2}
    SE = set()
    delta = np.array([0.0, 0.3, 5.0])

    alpha_t, SG_out, SL_out, SE_out, _delta_out, loop1_bounds = navigate(
        P_all, v, alpha_s, tau, SG, SL, SE, delta
    )

    # THE invariant reused from tests/test_navigation.py:117
    assert {b[0] for b in loop1_bounds} == (SG_out | SL_out | SE_out)

    # Structural sanity only (not a new correctness claim about the algorithm):
    # navigate must return weights of the right shape forming a convex combination.
    assert alpha_t.shape == alpha_s.shape
    assert np.all(alpha_t >= -1e-9)
    assert np.isclose(alpha_t.sum(), 1.0, atol=1e-6)
