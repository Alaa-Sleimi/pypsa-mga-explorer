"""Vertex sampling of the near-optimal PoI space via MP(r).

Second step of the preparation phase. Every random direction r in R^m is minimised
over the near-optimal feasible set,

    MP(r): min  r'p = r'Dx
           s.t. x in F^M   (i.e. Ax <= b  AND  c'x <= (1 + epsilon) * c'x*)

so each solve returns one extreme point of the PoI-space projection F^P. The
n_samples points collected here outline the near-optimal space; together with the
points from :mod:`mga_engine.interior_sampling` they are the samples that
:mod:`mga_engine.gp_solver` prices.
"""

import numpy as np
import pypsa
from typing import List
from mga_engine.poi import PoiSpec, evaluate_all


def _build_mga_model(
    network: pypsa.Network,
    opt_cost: float,
    epsilon: float,
) -> None:
    """Build a fresh linopy model on `network` and add the MGA cost-slack constraint.

    Parameters
    ----------
    network : pypsa.Network
        Network to build the model on. Any existing ``network.model`` is replaced.
    opt_cost : float
        Minimum system cost c'x* in EUR/yr. Must come from a solve with
        ``include_objective_constant=False``, as used here, or the slack is off by
        PyPSA's objective constant.
    epsilon : float
        Relative cost slack. The constraint added is
        ``objective <= (1 + epsilon) * opt_cost``.

    Returns
    -------
    None
        The model is left on ``network.model``, carrying the slack row named
        ``"mga_slack"``.

    Notes
    -----
    Prints the slack right-hand side. A negative `opt_cost` would turn the slack into
    a constraint tighter than the optimum; this is not checked.
    """
    network.optimize.create_model(include_objective_constant=False)
    m = network.model
    mga_rhs = (1.0 + epsilon) * opt_cost
    m.add_constraints(
        m.objective.expression <= mga_rhs,
        name="mga_slack",
    )
    print(f"[vertex] Linopy model ready | MGA slack ≤ {mga_rhs:,.0f}  (ε={epsilon})")


def _solve_mp(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    direction: np.ndarray,
) -> np.ndarray:
    """Solve MP(r) on the already-built model and return the PoI vector it reaches.

    Parameters
    ----------
    network : pypsa.Network
        Network whose ``network.model`` was prepared by :func:`_build_mga_model`;
        the ``"mga_slack"`` constraint must already be on it.
    poi_specs : list of PoiSpec
        PoI specs defining the coordinates, in the intended row order.
    direction : numpy.ndarray
        Shape ``(m,)``. The direction r. The objective becomes
        ``sum_i direction[i] * poi_i`` and is MINIMISED, so r points away from the
        face being sampled.

    Returns
    -------
    numpy.ndarray
        Shape ``(m,)``. PoI values at the optimum, in GW, from
        :func:`mga_engine.poi.evaluate_all`.

    Raises
    ------
    RuntimeError
        If the solver status is not ``"ok"``.

    Notes
    -----
    Calls HiGHS. Mutates `network`: the model objective is overwritten and the
    solution is written into the result columns. The cost objective is gone
    afterwards (the cost slack survives as a constraint), so the model can only be
    reused for further MP(r) solves.
    """
    m = network.model

    new_obj = sum(
        float(direction[i]) * poi_specs[i].linopy_expr(m)
        for i in range(len(poi_specs))
    )
    m.objective = new_obj

    network.model.solve(solver_name="highs", output_flag=False)

    if network.model.status != "ok":
        raise RuntimeError(
            f"[vertex] MP(r) solve failed with status: {network.model.status}"
        )

    network.optimize.assign_solution()
    return evaluate_all(poi_specs, network)


def sample_vertices(
    network: pypsa.Network,
    poi_specs: List[PoiSpec],
    opt_cost: float,
    epsilon: float = 0.05,
    n_samples: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Sample `n_samples` vertices of F^P by solving MP(r) for random directions.

    Parameters
    ----------
    network : pypsa.Network
        Unsolved network. Its model is rebuilt here and then reused for every
        direction, so one network object serves the whole sweep.
    poi_specs : list of PoiSpec
        PoI specs for `network`, in the intended row order of the result.
    opt_cost : float
        Minimum system cost c'x* in EUR/yr; see :func:`_build_mga_model`.
    epsilon : float, default 0.05
        Relative cost slack defining the near-optimal set.
    n_samples : int, default 10
        Number of random directions, i.e. of columns in the result.
    seed : int, default 42
        Seed of the ``numpy.random.default_rng`` that draws the directions.

    Returns
    -------
    numpy.ndarray
        Shape ``(m, n_samples)``. Column k holds the PoI vector in GW returned by
        MP(r) for direction k.

    Raises
    ------
    RuntimeError
        Propagated from the first failed MP(r) solve; the sweep aborts and no
        partial matrix is returned.

    Notes
    -----
    Directions are standard normal vectors normalised to unit length, so the same
    `seed` reproduces exactly the same directions; whether identical directions also
    give identical vertices depends on the solver's tie-breaking, since a degenerate
    LP can have a whole optimal face. For the same reason a returned point need not
    be a vertex, and "vertex" here is nominal.
    Calls HiGHS `n_samples` times and prints one line per sample.
    """
    rng = np.random.default_rng(seed)
    m_dim = len(poi_specs)

    _build_mga_model(network, opt_cost, epsilon)

    P_vertices = np.zeros((m_dim, n_samples))

    print(f"[vertex] Sampling {n_samples} vertices ...")
    for k in range(n_samples):
        r = rng.standard_normal(m_dim)
        r /= np.linalg.norm(r)

        p = _solve_mp(network, poi_specs, r)
        P_vertices[:, k] = p
        print(f"  [{k+1}/{n_samples}]  p = {np.round(p, 3)}")

    return P_vertices