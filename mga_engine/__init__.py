# mga_engine — interactive near-optimal solution exploration for PyPSA
"""MGA-Compass near-optimal solution exploration for PyPSA networks.

Preparation phase (heavy, run once per network): :mod:`~mga_engine.poi` defines the
Properties of Interest, :mod:`~mga_engine.vertex_sampling` and
:mod:`~mga_engine.interior_sampling` sample the near-optimal PoI space,
:mod:`~mga_engine.gp_solver` prices every sample, and
:mod:`~mga_engine.preparation_state` saves and verifies the result.

Exploration phase (interactive, from the saved samples): :mod:`~mga_engine.ccp_solver`
prices a point, :mod:`~mga_engine.navigation` moves to a preferred solution,
:mod:`~mga_engine.traverse` walks the path to it, and
:mod:`~mga_engine.surrogate` recovers the cost vector that makes a point optimal.
:mod:`~mga_engine.projection` and :mod:`~mga_engine.alpha_projection` produce the 2-D
regions the notebook plots.
"""
