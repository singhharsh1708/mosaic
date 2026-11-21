# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plain-JAX thermal-mesh tesseract against the thermal-transient references.

Runs the tesseract API in-process (no Docker), so it needs JAX and ``mosaic_shared``
installed as a top-level package, as in the tesseract's own environment
(``mosaic/tesseracts/thermal-mesh/jax-heat/tesseract_requirements.txt`` with CPU
JAX); it is skipped elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
try:
    import mosaic_shared.schema_types  # noqa: F401
    from tesseract_core import Tesseract
except ImportError:
    pytest.skip(
        "mosaic_shared / tesseract_core not installed as top-level packages",
        allow_module_level=True,
    )

from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.thermal_mesh.ics import _uniform
from mosaic.benchmarks.problems.thermal_transient.references import (
    default_t_end,
    discrete_dloss_drho_uniform,
    discrete_output,
)

_API = (
    Path(__file__).parents[1]
    / "mosaic/tesseracts/thermal-mesh/jax-heat/tesseract_api.py"
)
_SLAB = {"nx": 8, "ny": None, "nz": 1, "Lx": 2.0, "Ly": 1.0, "rho_0": 0.5}
_REF = {k: v for k, v in _SLAB.items() if k != "Ly"}


@pytest.fixture(scope="module")
def tess():
    return Tesseract.from_tesseract_api(_API)


def _inputs(steps: int, **extra) -> dict:
    ic = _uniform(**_SLAB)
    return get_config("thermal-transient").make_inputs(
        "JAX", ic, **_SLAB, steps=steps, **extra
    )


def _grad_uniform(tess, inputs: dict, J: float) -> float:
    """d(J^2)/d(eps) for rho + eps, as analytic_param_check computes it."""
    g = tess.vector_jacobian_product(
        inputs=inputs,
        vjp_inputs=["rho"],
        vjp_outputs=["identification_error"],
        cotangent_vector={"identification_error": np.float32(2.0 * J)},
    )["rho"]
    return float(np.sum(np.asarray(g, dtype=np.float64)))


@pytest.mark.parametrize("steps", [10, 100])
def test_forward_and_uniform_gradient_match_discrete_reference(tess, steps):
    t_end = default_t_end(_SLAB["Lx"], _SLAB["rho_0"])
    inputs = _inputs(steps)
    J = float(tess.apply(inputs)["identification_error"])
    J_ref = float(discrete_output(None, t_end, **_REF, steps=steps))
    assert abs(J - J_ref) / J_ref < 1e-6

    g_ref = discrete_dloss_drho_uniform(None, t_end, **_REF, steps=steps)
    for extra in ({}, {"adjoint_checkpoints": "sqrt"}, {"adjoint_checkpoints": 3}):
        g = _grad_uniform(tess, _inputs(steps, **extra), J)
        assert abs(g - g_ref) / abs(g_ref) < 1e-5, extra


def test_tape_and_checkpoint_give_the_same_gradient(tess):
    rho = np.random.default_rng(0).uniform(0.2, 0.9, 32).astype(np.float32)

    def grad(**extra):
        inputs = {**_inputs(100, **extra), "rho": rho}
        return np.asarray(
            tess.vector_jacobian_product(
                inputs=inputs,
                vjp_inputs=["rho"],
                vjp_outputs=["identification_error"],
                cotangent_vector={"identification_error": np.float32(1.0)},
            )["rho"]
        )

    tape = grad()
    for checkpoints in (10, 7):  # 7 leaves a shorter last chunk
        np.testing.assert_allclose(
            grad(adjoint_checkpoints=checkpoints), tape, rtol=1e-6, atol=0
        )


def test_steady_solve_is_linear_conduction(tess):
    """Uniform density, flux q on the right face: T = q x / k, exact for HEX8."""
    from mosaic.benchmarks.problems.thermal_transient.references import conductivity

    inputs = get_config("thermal-mesh").make_inputs(
        "JAX", _uniform(**_SLAB), **_SLAB, Q_total=1.0
    )
    q = float(np.asarray(inputs["boundary_conditions"]["neumann"]["values"])[0, 0])
    k = conductivity(0.5)
    C = float(tess.apply(inputs)["thermal_compliance"])
    # C = q * int_right T = q * (q Lx / k) * (area of the right face = Q_total / q)
    C_ref = (q * 2.0 / k) * 1.0
    assert abs(C - C_ref) / C_ref < 1e-6


def _abstract(x):
    """Arrays (and NumPy scalars) as the shape/dtype placeholders abstract_eval takes."""
    if isinstance(x, dict):
        return {k: _abstract(v) for k, v in x.items()}
    if isinstance(x, np.ndarray | np.generic):
        return {"shape": list(np.shape(x)), "dtype": str(np.asarray(x).dtype)}
    return x


def test_abstract_eval_with_steps(tess):
    out = tess.abstract_eval(_abstract(_inputs(1000, adjoint_checkpoints="sqrt")))
    assert tuple(out["identification_error"]["shape"]) == ()
    assert out["thermal_compliance"]["dtype"] == "float32"


def test_xla_memory_reports_the_tape_growing_with_steps():
    import importlib.util

    spec = importlib.util.spec_from_file_location("jax_heat_api", _API)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    tape = [api.xla_memory(_inputs(s))["temp"] for s in (100, 1000)]
    ckpt = api.xla_memory(_inputs(1000, adjoint_checkpoints="sqrt"))["temp"]
    assert tape[1] > 5 * tape[0]
    assert ckpt < tape[1] / 5
