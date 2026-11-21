# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-solver exclusions for thermal-transient.

Firedrake and FEniCS implement the time loop (``steps > 0`` in their thermal-mesh
tesseracts); FEniCS's legacy dolfin-adjoint has no checkpointing, so it runs only the
full-tape side of the horizon sweep. torch-fem implements it too (an iterative
backend: warm-started CG to ``cg_rtol``, implicit or unrolled adjoint) and has no
checkpointing option either. JAX-FEM rejects the transient fields, so it
is listed here as not implemented rather than left to fail; deal.II's gradient
experiments are categorically out, as in thermal-mesh, since it has no AD path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.benchmarks.core.config import Exclusion, ExclusionCategory
from mosaic.benchmarks.problems.thermal_mesh.exclusions import DEALII_NO_VJP

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem

NO_TIME_LOOP = Exclusion(
    ExclusionCategory.NOT_IMPLEMENTED,
    "This thermal-mesh tesseract has no time loop yet: it does not accept the "
    "thermal-transient fields (initial_temperature, dt, steps).",
)
NO_CHECKPOINTING = Exclusion(
    ExclusionCategory.NOT_IMPLEMENTED,
    "This tesseract has no adjoint checkpointing option, so only the full-tape run applies.",
)

_UNPORTED = ("jax_fem",)


def register(problem: Problem) -> None:
    """Attach per-solver exclusions; called once from ``config.py``."""
    for suite in ("forward", "gradient", "cost"):
        problem.exclude(suite, {k: NO_TIME_LOOP for k in _UNPORTED})
    problem.exclude("forward", {"dealii_heat": NO_TIME_LOOP})
    problem.exclude("cost/temporal_cost", {"dealii_heat": NO_TIME_LOOP})
    problem.exclude("cost/vjp_cost", {"dealii_heat": DEALII_NO_VJP})
    problem.exclude("gradient", {"dealii_heat": DEALII_NO_VJP})
    for key in (
        "gradient/horizon_sweep_limits/checkpoint",
        "gradient/analytic_check/checkpoint",
    ):
        problem.exclude(
            key,
            {"fenics_heat": NO_CHECKPOINTING, "torch_fem_thermal": NO_CHECKPOINTING},
        )
