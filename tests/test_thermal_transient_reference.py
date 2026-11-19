# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thermal-transient references.

The discrete references reduce the 3D scheme to a 1D eigenproblem by
separation of variables. These tests check that reduction against a direct
solve of the 3D scheme: trilinear hexahedra assembled by Gauss quadrature,
T = 0 at x = 0, backward Euler with dense solves.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.benchmarks.problems.thermal_transient.references import (
    conductivity,
    continuum_output,
    default_t_end,
    discrete_dloss_drho_uniform,
    discrete_output,
    eigenmode,
)

_LX, _LY = 2.0, 1.0


def _hex8_matrices(nx: int, ny: int, nz: int, Lz: float):
    """Global stiffness (unit conductivity) and mass of a structured Q1 hex mesh."""
    h = np.array([_LX / nx, _LY / ny, Lz / nz])
    g = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    s = 2.0 * corners - 1.0  # reference corners in {-1, 1}^3
    Ke, Me = np.zeros((8, 8)), np.zeros((8, 8))
    for xi in g:
        for eta in g:
            for zeta in g:
                q = np.array([xi, eta, zeta])
                N = np.prod(1.0 + s * q, axis=1) / 8.0
                dN = np.empty((8, 3))
                for d in range(3):
                    other = [e for e in range(3) if e != d]
                    dN[:, d] = (
                        s[:, d] * np.prod(1.0 + s[:, other] * q[other], axis=1) / 8.0
                    )
                dN = dN * (2.0 / h)  # d/dx = (2/h) d/dxi
                w = np.prod(h) / 8.0  # Jacobian determinant, unit Gauss weights
                Ke += w * dN @ dN.T
                Me += w * np.outer(N, N)

    def nid(ix, iy, iz):
        return iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix

    n = (nx + 1) * (ny + 1) * (nz + 1)
    K, M = np.zeros((n, n)), np.zeros((n, n))
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                dofs = [
                    nid(ix + int(c[0]), iy + int(c[1]), iz + int(c[2])) for c in corners
                ]
                K[np.ix_(dofs, dofs)] += Ke
                M[np.ix_(dofs, dofs)] += Me
    x = np.tile(np.linspace(0.0, _LX, nx + 1), (ny + 1) * (nz + 1))
    return K, M, x


def _direct_J(nx, ny, nz, rho_0, steps, t_end):
    K, M, x = _hex8_matrices(nx, ny, nz, Lz=_LX / nx)
    free = x > 1e-12
    A = M + (t_end / steps) * conductivity(rho_0) * K
    Aff, Mff = A[np.ix_(free, free)], M[np.ix_(free, free)]
    T = eigenmode(x, _LX).astype(np.float64)[free]
    for _ in range(steps):
        T = np.linalg.solve(Aff, Mff @ T)
    return float(T @ T)  # nodal sum of T^2; the x = 0 nodes are zero


@pytest.mark.parametrize("nx,ny,nz", [(4, 2, 1), (8, 3, 2)], ids=str)
@pytest.mark.parametrize("steps", [1, 7, 40])
def test_discrete_output_matches_a_direct_3d_solve(nx, ny, nz, steps):
    t = default_t_end(_LX, 0.5)
    J = discrete_output(None, t, nx=nx, ny=ny, nz=nz, Lx=_LX, rho_0=0.5, steps=steps)
    assert float(J) == pytest.approx(_direct_J(nx, ny, nz, 0.5, steps, t), rel=1e-12)


@pytest.mark.parametrize("steps", [1, 7, 40])
def test_dloss_drho_matches_central_differences_of_the_direct_solve(steps):
    nx, ny, nz, rho_0, eps = 4, 2, 1, 0.5, 1e-5
    t = default_t_end(_LX, rho_0)  # held fixed: only the conductivity moves

    def loss(r):
        return _direct_J(nx, ny, nz, r, steps, t) ** 2

    fd = (loss(rho_0 + eps) - loss(rho_0 - eps)) / (2 * eps)
    exact = discrete_dloss_drho_uniform(
        None, t, nx=nx, ny=ny, nz=nz, Lx=_LX, rho_0=rho_0, steps=steps
    )
    assert exact == pytest.approx(fd, rel=1e-7)


def test_discrete_output_converges_to_the_continuum_one():
    t = default_t_end(_LX, 0.5)
    errs = []
    for nx in (8, 16, 32):
        d = float(discrete_output(None, t, nx=nx, Lx=_LX, rho_0=0.5, steps=4000))
        c = float(continuum_output(None, t, nx=nx, Lx=_LX, rho_0=0.5))
        errs.append(abs(d / c - 1))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-3
