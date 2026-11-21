# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical schemas for thermal-transient: the thermal-mesh inputs plus a time loop.

thermal-transient reuses the thermal-mesh tesseracts. A solver that supports it adds
:class:`TransientFields` to its thermal-mesh input schema::

    from mosaic_shared.problems.thermal_mesh import InputSchema as _Base
    from mosaic_shared.problems.thermal_transient import TransientFields


    class InputSchema(
        make_differentiable(_Base, ["rho", "source"]), TransientFields
    ): ...

(``make_differentiable`` only wraps fields declared on the class it is given, so the
transient fields come in as a mixin rather than a subclass of the canonical schema.)
With ``steps == 0`` (the default) the solver solves the steady problem exactly as for
thermal-mesh; with ``steps > 0`` it starts from ``initial_temperature`` and takes
``steps`` backward Euler steps of size ``dt`` of

    dT/dt = div(k(rho) grad T) + source    (unit heat capacity)

with the thermal-mesh boundary conditions, and reports the outputs for the final
temperature. Tesseracts without these fields reject them (extra fields are forbidden),
so an unsupported solver fails loudly instead of returning the steady answer.
"""

import numpy as np
from mosaic_shared.problems.thermal_mesh import InputSchema as _ThermalInputSchema
from mosaic_shared.problems.thermal_mesh import OutputSchema
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Float32


class TransientFields(BaseModel):
    """The time-loop inputs a thermal-mesh solver adds to support thermal-transient."""

    initial_temperature: Array[(None,), Float32] = Field(
        default_factory=lambda: np.zeros(1, dtype=np.float32),
        description=(
            "Nodal temperature at t = 0, shape (n_nodes,), in the input node order. "
            "Dirichlet nodes take their prescribed value regardless. Unused when steps == 0."
        ),
    )
    dt: Array[(1,), Float32] = Field(
        default_factory=lambda: np.zeros(1, dtype=np.float32),
        description="Time step of backward Euler. Unused when steps == 0.",
    )
    steps: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of backward Euler steps; total time = steps * dt. 0 solves the "
            "steady problem (thermal-mesh)."
        ),
    )

    @model_validator(mode="after")
    def _check_time_step(self) -> "TransientFields":
        # Tesseract's abstract_eval validates this schema with ShapeDType placeholders,
        # which carry no value, so only a concrete dt is checked.
        if isinstance(self.dt, np.ndarray) and self.steps > 0 and not self.dt[0] > 0:
            raise ValueError(f"steps = {self.steps} needs a positive dt, got {self.dt}")
        return self


class InputSchema(_ThermalInputSchema, TransientFields):
    """Canonical (non-differentiable) thermal-transient inputs, for docs and tests."""


__all__ = ["InputSchema", "OutputSchema", "TransientFields"]
