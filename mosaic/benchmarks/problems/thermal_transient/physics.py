# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input factory for thermal-transient: the thermal-mesh inputs plus the time loop."""

from __future__ import annotations

from typing import Any

import numpy as np

from mosaic.benchmarks.core.config import SolverSpec
from mosaic.benchmarks.problems.thermal_mesh.physics import (
    make_inputs as _thermal_inputs,
)

from .references import default_t_end, eigenmode


def make_inputs(
    spec: SolverSpec,
    ic: np.ndarray,
    *,
    steps: int = 100,
    t_end: float | None = None,
    Lx: float = 2.0,
    rho_0: float = 0.5,
    Q_total: float = 0.0,
    adjoint_checkpoints: int | str | None = None,
    **phys: Any,
) -> dict:
    """Thermal-mesh inputs (no heat flux by default) plus the eigenmode, dt and steps.

    The end time is fixed (lam * t_end = 1 for the uniform density rho_0 unless
    ``t_end`` is given) and dt = t_end / steps, so a sweep over ``steps`` refines the
    time step to one physical time: the closed form stays the same and float32
    outputs do not underflow at long horizons. The IC is the density field, as in
    thermal-mesh; ``rho_0`` only sets the end time here unless passed on as a
    uniform density by the experiment.

    ``adjoint_checkpoints`` (a count, or ``"sqrt"`` for ceil(sqrt(steps))) is passed on
    only when given, since only solvers with a checkpointing option accept the field.
    """
    inputs = _thermal_inputs(spec, ic, Lx=Lx, Q_total=Q_total, **phys)
    points = np.asarray(inputs["hex_mesh"]["points"])[: inputs["hex_mesh"]["n_points"]]
    T_end = default_t_end(Lx, rho_0) if t_end is None else float(t_end)
    out = {
        **inputs,
        "initial_temperature": eigenmode(points[:, 0], Lx),
        "dt": np.array([T_end / steps], dtype=np.float32),
        "steps": int(steps),
    }
    if adjoint_checkpoints is not None:
        out["adjoint_checkpoints"] = (
            int(np.ceil(np.sqrt(steps)))
            if adjoint_checkpoints == "sqrt"
            else int(adjoint_checkpoints)
        )
    return out
