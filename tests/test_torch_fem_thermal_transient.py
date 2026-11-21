# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch-fem's thermal-transient path (``steps > 0``) against the exact discrete scheme.

The tesseract module is imported in-process, without Docker. Needs ``torch`` and
``torch-fem`` (not harness dependencies, so the module is skipped where they are
missing); a CPU torch wheel is enough.

At a tight CG tolerance both adjoint modes reproduce the closed-form discrete
reference. At a loose one, on a short step, they differ: the unrolled gradient is
the derivative of the executed CG iterations, and central differences of the served
forward at the same tolerance agree with it rather than with the implicit adjoint.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torchfem")

from mosaic.benchmarks.problems.thermal_mesh.ics import _uniform
from mosaic.benchmarks.problems.thermal_transient.config import (
    _SOLVERS,
    problem,
)
from mosaic.benchmarks.problems.thermal_transient.references import (
    default_t_end,
    discrete_dloss_drho_uniform,
    discrete_output,
)

_API = (
    Path(__file__).parents[1]
    / "mosaic/tesseracts/thermal-mesh/torch-fem-thermal/tesseract_api.py"
)
_SLAB = {"ny": None, "nz": 1, "Lx": 2.0, "Ly": 1.0, "rho_0": 0.5}


@pytest.fixture(scope="module")
def api():
    """Load the torch-fem thermal tesseract module straight from source."""
    spec = importlib.util.spec_from_file_location("torch_fem_thermal_api", _API)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(rho: np.ndarray, **physics) -> dict:
    name = _SOLVERS["torch_fem_thermal"].name
    return problem.make_inputs(name, rho, **{**_SLAB, **physics})


def _J(api, inputs: dict) -> float:
    return float(
        api.apply(api.InputSchema.model_validate(inputs))["identification_error"]
    )


def _grad(api, inputs: dict, cotangent: float = 1.0) -> np.ndarray:
    out = api.vector_jacobian_product(
        api.InputSchema.model_validate(inputs),
        {"rho"},
        {"identification_error"},
        {"identification_error": np.float32(cotangent)},
    )
    return np.asarray(out["rho"], dtype=np.float64)


@pytest.mark.parametrize("mode", ["implicit", "unrolled"])
@pytest.mark.parametrize(("nx", "steps"), [(8, 1), (8, 10), (8, 50), (16, 10)])
def test_tight_tolerance_matches_the_discrete_reference(api, mode, nx, steps):
    """J and the uniform-density gradient of J^2 agree to float32 round-off."""
    inputs = {
        **_inputs(_uniform(nx=nx), nx=nx, steps=steps),
        "cg_rtol": 1e-12,
        "adjoint_mode": mode,
    }
    t_end = default_t_end(2.0, 0.5)
    J_ref = float(discrete_output(None, t_end, nx=nx, steps=steps))
    dloss_ref = discrete_dloss_drho_uniform(None, t_end, nx=nx, steps=steps)

    J = _J(api, inputs)
    # The harness differentiates sum(J^2): cotangent 2 J, summed over the cells.
    dloss = float(np.sum(_grad(api, inputs, cotangent=2.0 * J)))
    assert J == pytest.approx(J_ref, rel=1e-6)
    assert dloss == pytest.approx(dloss_ref, rel=1e-6)


def test_loose_tolerance_fd_follows_the_executed_iterations(api):
    """cg_rtol = 1e-3, one step of t_end/100: FD of the served forward picks unrolled.

    The mechanism of the fd_heldout study, through the tesseract endpoints: the
    unrolled gradient is far from the converged one, the implicit adjoint is close
    to it, and same-tolerance central differences match the unrolled gradient.
    """
    rng = np.random.default_rng(1)
    rho = np.clip(0.5 * np.exp(0.3 * rng.standard_normal(32)), 0.05, 0.95)
    base = _inputs(rho.astype(np.float32), nx=8, steps=1, t_end=default_t_end() / 100)
    loose = {**base, "cg_rtol": 1e-3}

    g_unrolled = _grad(api, {**loose, "adjoint_mode": "unrolled"})
    g_implicit = _grad(api, {**loose, "adjoint_mode": "implicit"})
    g_converged = _grad(api, {**base, "cg_rtol": 1e-12})

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    assert rel(g_unrolled, g_converged) > 0.1
    assert rel(g_implicit, g_converged) < 0.03
    assert rel(g_unrolled, g_implicit) > 0.1

    # Central differences of the float32 forward at the same tolerance, against the
    # exact float32 perturbation, so input rounding does not enter the comparison.
    dirs = np.random.default_rng(0).standard_normal((6, rho.size))
    rho32 = loose["rho"]
    err = {"unrolled": [], "implicit": []}
    for d in dirs:
        r_plus = (rho32 + 1e-2 * d / np.linalg.norm(d)).astype(np.float32)
        r_minus = (rho32 - 1e-2 * d / np.linalg.norm(d)).astype(np.float32)
        fd = _J(api, {**loose, "rho": r_plus}) - _J(api, {**loose, "rho": r_minus})
        delta = r_plus.astype(np.float64) - r_minus.astype(np.float64)
        for key, g in (("unrolled", g_unrolled), ("implicit", g_implicit)):
            pred = g @ delta
            err[key].append(abs(fd - pred) / max(abs(fd), abs(pred)))
    assert np.median(err["unrolled"]) < 5e-3
    assert np.median(err["implicit"]) > 2e-2


def test_abstract_eval_accepts_transient_inputs():
    """abstract_eval validates ShapeDType placeholders, so no validator reads values."""
    from tesseract_core import Tesseract

    def placeholder(v):
        if isinstance(v, dict):
            return {k: placeholder(x) for k, x in v.items()}
        if isinstance(v, np.ndarray | np.generic):
            return {"shape": list(np.shape(v)), "dtype": str(v.dtype)}
        return v

    inputs = {
        **_inputs(_uniform(nx=8), nx=8, steps=50),
        "cg_rtol": 1e-3,
        "adjoint_mode": "unrolled",
    }
    out = Tesseract.from_tesseract_api(_API).abstract_eval(placeholder(inputs))
    assert out["identification_error"]["shape"] == ()
    assert out["thermal_compliance"]["dtype"] == "float32"


def test_unrolled_gradient_is_zero_when_cg_runs_no_iterations(api):
    # Zero initial temperature, source and flux: every step's right-hand side is zero,
    # so CG returns at once and the unrolled solve does not depend on rho.
    rho = np.full(32, 0.5, dtype=np.float32)
    inputs = _inputs(rho, nx=8, steps=1)
    inputs["adjoint_mode"] = "unrolled"  # make_inputs does not pass solver options on
    inputs["initial_temperature"] = np.zeros_like(inputs["initial_temperature"])
    assert _J(api, inputs) == 0.0
    np.testing.assert_array_equal(_grad(api, inputs), 0.0)
