# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heat conduction on a hexahedral mesh in plain JAX, steady or backward Euler.

Trilinear (HEX8) finite elements with SIMP conductivity, a per-cell heat source,
Dirichlet temperatures and Neumann heat fluxes, written directly in JAX so that a
whole solve, and its gradient, is one jit-compiled XLA program. The element
matrices depend only on the mesh and are computed once on the host; the
conductivity scales them per cell, so the operator is a traceable function of rho.

With ``steps == 0`` the steady problem is solved (thermal-mesh). With ``steps > 0``
(thermal-transient) the solver takes ``steps`` backward Euler steps

    (M + dt K(rho)) T_{n+1} = M T_n + dt F(source)

in a ``jax.lax.scan``. Each step solves by conjugate gradients (Jacobi
preconditioner); ``jax.scipy.sparse.linalg.cg`` is a ``custom_linear_solve``, so
reverse mode transposes the solve instead of taping the CG iterations. The
gradient is plain reverse mode through the scan: every step's state is kept (a full
tape), or, with ``adjoint_checkpoints``, the steps are split into chunks wrapped in
``jax.checkpoint`` so only the chunk boundaries are kept and each chunk is
recomputed once on the way back.

Because the gradient is a single XLA program, XLA's own memory analysis of the
compiled program predicts its device memory; :func:`xla_memory` reports it for
given inputs without running the gradient.
"""

import hashlib
import math
from collections.abc import Callable
from functools import partial, wraps
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.thermal_transient import TransientFields
from mosaic_shared.schema_types import make_differentiable
from pydantic import Field
from tesseract_core.runtime import ShapeDType

# Relative residual at which each conjugate-gradient solve stops (float64).
CG_TOL = 1e-12


class InputSchema(
    make_differentiable(_CanonicalInputSchema, ["rho", "source"]), TransientFields
):
    """Inputs for the JAX heat solver, extended with material parameters.

    With ``steps > 0`` (thermal-transient) the solver takes ``steps`` backward Euler
    steps from ``initial_temperature`` instead of the steady solve.
    """

    adjoint_checkpoints: int | None = Field(
        default=None,
        ge=1,
        description=(
            "thermal-transient only: split the steps into this many chunks, each "
            "wrapped in jax.checkpoint, so the gradient keeps the state at the chunk "
            "boundaries and recomputes each chunk once on the way back (about "
            "adjoint_checkpoints + steps / adjoint_checkpoints states). None keeps "
            "every step on the tape."
        ),
    )
    k_max: float = Field(
        default=1.0,
        description="Maximum thermal conductivity (fully solid/conducting material).",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p (k(ρ) = k_min + (k_max−k_min)·ρ^p).",
    )


class OutputSchema(
    make_differentiable(
        _CanonicalOutputSchema, ["thermal_compliance", "identification_error"]
    )
):
    """JAX heat solver output schema."""


def _x64(fn: Callable) -> Callable:
    """Run ``fn`` with 64-bit JAX types, without changing the process-wide setting."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with jax.enable_x64(True):
            return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Host-side setup: element matrices and boundary data (depends on mesh and BCs only)
# ---------------------------------------------------------------------------

# Reference coordinates of the HEX8 nodes (VTK / HexMesh order: the z-min face
# counterclockwise, then the z-max face).
_HEX_REF = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)
# The six faces of a hex, each with its nodes in cyclic order.
_HEX_FACES = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
)
_GAUSS = np.array([-1.0, 1.0]) / np.sqrt(3.0)  # two-point rule, unit weights


def _element_matrices(
    pts: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-conductivity stiffness, mass and load (int N_a) of every cell.

    2x2x2 Gauss quadrature of the trilinear map, exact for the mass and for the
    stiffness of parallelepiped cells. Returns arrays of shape (E, 8, 8), (E, 8, 8)
    and (E, 8).
    """
    X = pts[cells]  # (E, 8, 3)
    n_cells = len(cells)
    ke = np.zeros((n_cells, 8, 8))
    me = np.zeros((n_cells, 8, 8))
    fe = np.zeros((n_cells, 8))
    for xi in _GAUSS:
        for eta in _GAUSS:
            for zeta in _GAUSS:
                q = np.array([xi, eta, zeta])
                factors = 1.0 + _HEX_REF * q  # (8, 3)
                N = factors.prod(axis=1) / 8.0
                dN = np.empty((8, 3))
                for d in range(3):
                    others = [o for o in range(3) if o != d]
                    dN[:, d] = (
                        _HEX_REF[:, d] * factors[:, others[0]] * factors[:, others[1]]
                    ) / 8.0
                J = np.einsum("eai,aj->eij", X, dN)  # dX_i / dxi_j
                detJ = np.linalg.det(J)
                if np.any(detJ <= 0):
                    raise ValueError(
                        "hex cells must be non-degenerate and positively oriented"
                    )
                grad = np.einsum("aj,eji->eai", dN, np.linalg.inv(J))
                ke += np.einsum("eai,ebi->eab", grad, grad) * detJ[:, None, None]
                me += np.outer(N, N)[None] * detJ[:, None, None]
                fe += N[None] * detJ[:, None]
    return ke, me, fe


def _neumann_load(
    pts: np.ndarray, cells: np.ndarray, mask: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Nodal load sum_g q_g int_{Gamma_g} N_a dGamma of the Neumann fluxes.

    A boundary face (one that belongs to a single cell) is in Neumann group g when
    all four of its nodes are, as in the Firedrake solver. The same vector gives the
    thermal compliance, C = sum_g q_g int_{Gamma_g} T = load . T.
    """
    load = np.zeros(len(pts))
    if values.size == 0 or not np.any(mask > 0):
        return load
    faces = cells[:, _HEX_FACES].reshape(-1, 4)
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    faces = faces[counts[inverse.ravel()] == 1]
    groups = mask[faces]
    on_group = (groups[:, 0] > 0) & np.all(groups == groups[:, :1], axis=1)
    faces, g = faces[on_group], groups[on_group, 0]
    q = values[g - 1, 0]
    ref = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    X = pts[faces]  # (F, 4, 3)
    for s in _GAUSS:
        for t in _GAUSS:
            factors = 1.0 + ref * np.array([s, t])
            N = factors.prod(axis=1) / 4.0
            dNs = ref[:, 0] * factors[:, 1] / 4.0
            dNt = ref[:, 1] * factors[:, 0] / 4.0
            area = np.linalg.norm(
                np.cross(
                    np.einsum("fai,a->fi", X, dNs), np.einsum("fai,a->fi", X, dNt)
                ),
                axis=1,
            )
            np.add.at(load, faces, (q * area)[:, None] * N[None])
    return load


def _padded(arr: Any, n: int, fill: float = 0.0) -> np.ndarray:
    """``arr`` trimmed or padded with ``fill`` to length n."""
    out = np.full(n, fill, dtype=np.float64)
    a = np.asarray(arr if arr is not None else [], dtype=np.float64).ravel()[:n]
    out[: len(a)] = a
    return out


def _setup(pts: np.ndarray, cells: np.ndarray, bc: dict) -> dict[str, np.ndarray]:
    """Element matrices and boundary data for a mesh and its boundary conditions."""
    n = len(pts)
    ke, me, fe = _element_matrices(pts, cells)
    d_bc = bc.get("dirichlet") or {}
    n_bc = bc.get("neumann") or {}
    d_mask = _padded(d_bc.get("mask"), n).astype(np.int64)
    d_vals = d_bc.get("values")
    n_groups = int(d_mask.max(initial=0))
    d_vals = (
        np.zeros((n_groups, 1))
        if d_vals is None
        else np.asarray(d_vals, dtype=np.float64).reshape(-1, 1)
    )
    T_dirichlet = np.zeros(n)
    fixed = d_mask > 0
    T_dirichlet[fixed] = d_vals[d_mask[fixed] - 1, 0]
    n_mask = _padded(n_bc.get("mask"), n).astype(np.int64)
    n_vals = np.asarray(
        n_bc.get("values") if n_bc.get("values") is not None else np.zeros((0, 1)),
        dtype=np.float64,
    ).reshape(-1, 1)
    return {
        "cells": cells.astype(np.int32),
        "ke": ke,
        "me": me,
        "fe": fe,
        "free": (~fixed).astype(np.float64),
        "T_dirichlet": T_dirichlet,
        "neumann_load": _neumann_load(pts, cells, n_mask, n_vals),
    }


# The setup arrays of the last (mesh, BC) seen, already on the device. Only the
# setup is cached; every endpoint call runs its own solve.
_SETUP_CACHE: dict[str, dict[str, jax.Array]] = {}


def _cached_setup(inputs: InputSchema) -> dict[str, jax.Array]:
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    bc = inputs.boundary_conditions.model_dump()
    h = hashlib.sha256()
    for arr in (
        pts,
        cells,
        *(
            (bc.get(kind) or {}).get(key)
            for kind in ("dirichlet", "neumann")
            for key in ("mask", "values")
        ),
    ):
        if arr is None:
            h.update(b"none")
            continue
        arr = np.ascontiguousarray(arr)
        h.update(f"{arr.dtype}{arr.shape}".encode())
        h.update(arr.tobytes())
    key = h.hexdigest()
    if key not in _SETUP_CACHE:
        _SETUP_CACHE.clear()
        _SETUP_CACHE[key] = {
            k: jnp.asarray(v) for k, v in _setup(pts, cells, bc).items()
        }
    return _SETUP_CACHE[key]


# ---------------------------------------------------------------------------
# The solver, traceable end to end
# ---------------------------------------------------------------------------


def _scatter(m: dict, local: jax.Array) -> jax.Array:
    """Sum per-cell nodal values (E, 8) into a nodal vector."""
    n = m["free"].shape[0]
    return jnp.zeros(n, local.dtype).at[m["cells"]].add(local)


def _matvec(m: dict, A_e: jax.Array, v: jax.Array) -> jax.Array:
    """Assembled operator sum_e A_e applied to a nodal vector, cell by cell."""
    return _scatter(m, jnp.einsum("eab,eb->ea", A_e, v[m["cells"]]))


def _solve(m: dict, A_e: jax.Array, b: jax.Array, x0: jax.Array) -> jax.Array:
    """Solve the operator on the free nodes (zero on the Dirichlet nodes) by CG.

    The operator is masked on both sides, so it stays symmetric and positive
    definite, and its transpose solve (reverse mode) is the same solve.
    """
    free = m["free"]
    diag = free * _scatter(m, jnp.diagonal(A_e, axis1=1, axis2=2)) + (1.0 - free)
    dinv = jax.lax.stop_gradient(1.0 / diag)

    def A(v: jax.Array) -> jax.Array:
        return free * _matvec(m, A_e, free * v) + (1.0 - free) * v

    x, _ = jax.scipy.sparse.linalg.cg(
        A, free * b, x0=x0, tol=CG_TOL, M=lambda r: dinv * r
    )
    return x


def _chunked_scan(
    body: Callable, T: jax.Array, steps: int, checkpoints: int | None
) -> jax.Array:
    """``steps`` applications of ``body``; with checkpoints, in jax.checkpoint chunks.

    Chunks have length ceil(steps / checkpoints), plus one shorter chunk for the
    remainder, so the gradient keeps one state per chunk and the states of one
    chunk while it is recomputed.
    """
    if checkpoints is None:
        return jax.lax.scan(body, T, None, length=steps)[0]
    length = math.ceil(steps / min(checkpoints, steps))
    n_full, rem = divmod(steps, length)

    @jax.checkpoint
    def chunk(T: jax.Array, _: None) -> tuple[jax.Array, None]:
        return jax.lax.scan(body, T, None, length=length)[0], None

    T = jax.lax.scan(chunk, T, None, length=n_full)[0]
    if rem:
        T = jax.checkpoint(lambda T: jax.lax.scan(body, T, None, length=rem)[0])(T)
    return T


def _temperature(
    m: dict,
    rho: jax.Array,
    source: jax.Array,
    T0: jax.Array,
    dt: jax.Array,
    k_max: float,
    p_exp: float,
    steps: int,
    checkpoints: int | None,
) -> jax.Array:
    """Final nodal temperature: steady solve (steps == 0) or backward Euler."""
    n_cells, n = m["ke"].shape[0], m["free"].shape[0]
    # Clip the value but not the derivative, as the other backends do (they clip before
    # rho becomes a control): at rho = 1 jnp.clip's derivative would be one half.
    rho = rho[:n_cells].astype(jnp.float64)
    rho = rho + jax.lax.stop_gradient(jnp.clip(rho, 0.0, 1.0) - rho)
    k_min = 1e-3 * k_max
    k = k_min + (k_max - k_min) * rho**p_exp
    src = (
        jnp.zeros(n_cells)
        .at[: min(n_cells, source.shape[0])]
        .set(source[:n_cells].astype(jnp.float64))
    )
    F = _scatter(m, src[:, None] * m["fe"]) + m["neumann_load"]
    TD = m["T_dirichlet"]

    if steps == 0:
        K_e = k[:, None, None] * m["ke"]
        return _solve(m, K_e, F - _matvec(m, K_e, TD), jnp.zeros(n)) + TD

    dt = dt[0].astype(jnp.float64)
    A_e = m["me"] + dt * k[:, None, None] * m["ke"]
    rhs_const = dt * F - _matvec(m, A_e, TD)  # Dirichlet lift, the same every step
    free = m["free"]
    T = jnp.zeros(n).at[: min(n, T0.shape[0])].set(T0[:n].astype(jnp.float64))
    T = free * T + TD

    def step(T: jax.Array, _: None) -> tuple[jax.Array, None]:
        u = _solve(m, A_e, _matvec(m, m["me"], T) + rhs_const, free * (T - TD))
        return u + TD, None

    return _chunked_scan(step, T, steps, checkpoints)


def _outputs(m: dict, x: dict, opts: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
    """(thermal_compliance, identification_error) of the final temperature.

    ``x`` holds the input arrays (rho, source, T0, dt, target) and ``opts`` the
    static options (k_max, p_exp, steps, checkpoints).
    """
    T = _temperature(m, x["rho"], x["source"], x["T0"], x["dt"], **opts)
    C = m["neumann_load"] @ T
    n_t = min(T.shape[0], x["target"].shape[0])
    misfit = T[:n_t] - x["target"][:n_t]
    return C, jnp.sum(misfit**2)


def _opts(*, k_max: float, p_exp: float, steps: int, checkpoints: int | None) -> dict:
    return {"k_max": k_max, "p_exp": p_exp, "steps": steps, "checkpoints": checkpoints}


_STATIC = ("k_max", "p_exp", "steps", "checkpoints")


@partial(jax.jit, static_argnames=_STATIC)
def _forward(m: dict, x: dict, **opts: Any) -> tuple[jax.Array, jax.Array]:
    return _outputs(m, x, _opts(**opts))


@partial(jax.jit, static_argnames=_STATIC)
def _gradient(
    m: dict, x: dict, cot_C: jax.Array, cot_I: jax.Array, **opts: Any
) -> tuple[jax.Array, jax.Array]:
    """d(cot_C C + cot_I I)/d(rho, source), shaped like the rho and source inputs."""

    def loss(rho: jax.Array, source: jax.Array) -> jax.Array:
        C, err = _outputs(m, {**x, "rho": rho, "source": source}, _opts(**opts))
        return cot_C * C + cot_I * err

    return jax.grad(loss, argnums=(0, 1))(x["rho"], x["source"])


def _args(inputs: InputSchema) -> tuple[tuple, dict]:
    """Positional array arguments and static options of the jitted functions."""
    m = _cached_setup(inputs)
    x = {
        "rho": inputs.rho,
        "source": inputs.source,
        "T0": inputs.initial_temperature,
        "dt": inputs.dt,
        "target": inputs.target_temperature,
    }
    arrays = (m, {k: jnp.asarray(v, dtype=jnp.float64) for k, v in x.items()})
    static = {
        "k_max": float(inputs.k_max),
        "p_exp": float(inputs.p_exp),
        "steps": int(inputs.steps),
        "checkpoints": inputs.adjoint_checkpoints if inputs.steps > 0 else None,
    }
    return arrays, static


def _cotangents(cotangent_vector: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.asarray(cotangent_vector.get("thermal_compliance", 0.0), jnp.float64),
        jnp.asarray(cotangent_vector.get("identification_error", 0.0), jnp.float64),
    )


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


@_x64
def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: thermal compliance and identification error of the final temperature."""
    arrays, static = _args(inputs)
    C, err = _forward(*arrays, **static)
    return OutputSchema(
        thermal_compliance=np.float32(C), identification_error=np.float32(err)
    )


@_x64
def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Reverse mode through the whole solve as one compiled program.

    Supports rho and source against thermal_compliance and identification_error;
    the gradient re-runs the forward pass inside the same program.
    """
    assert vjp_inputs <= {"rho", "source"}
    assert vjp_outputs <= {"thermal_compliance", "identification_error"}
    arrays, static = _args(inputs)
    g_rho, g_source = _gradient(*arrays, *_cotangents(cotangent_vector), **static)
    grads = {"rho": g_rho, "source": g_source}
    return {k: np.asarray(grads[k], dtype=np.float32) for k in vjp_inputs}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }


# ---------------------------------------------------------------------------
# XLA's memory analysis of the compiled gradient (not a Tesseract endpoint)
# ---------------------------------------------------------------------------


@_x64
def xla_memory(inputs: InputSchema | dict) -> dict[str, Any]:
    """XLA's memory analysis of the compiled VJP for these inputs, without running it.

    Compiles the program ``vector_jacobian_product`` runs (gradient of
    identification_error with respect to rho, the thermal-transient setting) for
    the default backend and returns the sizes XLA plans for it, in bytes:
    ``temp`` (scratch, which holds the tape), ``argument``, ``output``, ``alias``
    and ``generated_code``, plus ``peak_estimate = temp + argument + output -
    alias``, the device memory the program needs on top of what the process
    already holds. Compiling takes about as long as for a real call; nothing is
    executed.
    """
    if isinstance(inputs, dict):
        inputs = InputSchema.model_validate(inputs)
    arrays, static = _args(inputs)
    cot = _cotangents({"identification_error": 1.0})
    compiled = _gradient.lower(*arrays, *cot, **static).compile()
    ma = compiled.memory_analysis()
    out: dict[str, Any] = {
        "backend": jax.default_backend(),
        "device": jax.devices()[0].device_kind,
        "steps": static["steps"],
        "adjoint_checkpoints": static["checkpoints"],
        "n_nodes": int(arrays[0]["free"].shape[0]),
        "n_cells": int(arrays[0]["ke"].shape[0]),
    }
    for name in ("temp", "argument", "output", "alias", "generated_code"):
        out[name] = getattr(ma, f"{name}_size_in_bytes", None)
    out["peak_estimate"] = (
        out["temp"] + out["argument"] + out["output"] - (out["alias"] or 0)
        if None not in (out["temp"], out["argument"], out["output"])
        else None
    )
    return out
