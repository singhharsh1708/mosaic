# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed forms for the transient eigenmode: the continuum and the exact discrete solution.

Setup (see :mod:`.config`): uniform density rho_0, SIMP conductivity
k = k_min + (k_max - k_min) rho_0^p, unit heat capacity, T = 0 at x = 0 and
insulated elsewhere, initial temperature sin(pi x / (2 Lx)) at the nodes. That is
a single eigenmode, so

    T(x, t) = sin(pi x / (2 Lx)) exp(-lam t),    lam = k (pi / (2 Lx))^2.

The output is J = sum over nodes of T(t)^2 (``identification_error`` with a zero
target), and the harness differentiates sum(output^2) = J^2.

The trilinear (Q1) mass and stiffness matrices of the structured hex mesh are
tensor products, K = k (Kx (x) My (x) Mz + Mx (x) Ky (x) Mz + Mx (x) My (x) Kz), and
Ky, Kz annihilate constants, so backward Euler applied to an initial field that
depends on x alone stays of the form f (x) 1 (x) 1 and reduces exactly to

    (Mx + dt k Kx) f_{n+1} = Mx f_n,    J_h = (ny + 1)(nz + 1) |f_N|^2

(the nodal sum, not a mass-weighted norm). With Kx v_i = mu_i Mx v_i (M-orthonormal),
f_0 = V c, f_N = V (c q^-N) with q = 1 + dt k mu, which gives J_h and its derivative in
closed form. Comparing a solver's gradient with the discrete form measures its
differentiation error alone; the continuum form adds the discretisation error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

K_MAX = 1.0
P_EXP = 3.0
K_MIN_RATIO = 1e-3


def conductivity(rho_0: float, k_max: float = K_MAX, p_exp: float = P_EXP) -> float:
    """SIMP conductivity at a uniform density (rho clipped to [0, 1], as the solvers do)."""
    k_min = K_MIN_RATIO * k_max
    return k_min + (k_max - k_min) * float(np.clip(rho_0, 0.0, 1.0)) ** p_exp


def dconductivity(rho_0: float, k_max: float = K_MAX, p_exp: float = P_EXP) -> float:
    """dk/drho at rho_0."""
    k_min = K_MIN_RATIO * k_max
    return p_exp * (k_max - k_min) * float(rho_0) ** (p_exp - 1)


def default_t_end(Lx: float = 2.0, rho_0: float = 0.5) -> float:
    """The end time with lam * t_end = 1, so the closed form decays by e^-2."""
    return 1.0 / (conductivity(rho_0) * (np.pi / (2 * Lx)) ** 2)


def eigenmode(points_x: np.ndarray, Lx: float) -> np.ndarray:
    """The initial temperature at the given node x-coordinates, rounded to float32."""
    return np.sin(np.pi * np.asarray(points_x, dtype=np.float64) / (2 * Lx)).astype(
        np.float32
    )


def _x_matrices(nx: int, Lx: float):
    h, n = Lx / nx, nx + 1
    Kx, Mx = np.zeros((n, n)), np.zeros((n, n))
    for e in range(nx):
        i = [e, e + 1]
        Kx[np.ix_(i, i)] += np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        Mx[np.ix_(i, i)] += h / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])
    return Kx[1:, 1:], Mx[1:, 1:]  # x = 0 is held at zero


def discrete_J_and_dJdk(
    nx: int, ny: int, nz: int, Lx: float, rho_0: float, steps: int, t_end: float
) -> tuple[float, float]:
    """Exact J_h = sum_nodes T_N^2 of the discrete scheme, and dJ_h/dk."""
    from scipy.linalg import eigh

    Kx, Mx = _x_matrices(nx, Lx)
    x = np.linspace(0.0, Lx, nx + 1)[1:]
    f0 = eigenmode(x, Lx).astype(np.float64)
    mu, V = eigh(Kx, Mx)  # V' Mx V = I
    c = V.T @ (Mx @ f0)
    k = conductivity(rho_0)
    dt = t_end / steps
    # (1 + dt k mu)^-N as exp(-N log1p(dt k mu)): a power of the rounded 1 + dt k mu amplifies its rounding by N.
    log_q = np.log1p(dt * k * mu)
    a = c * np.exp(-float(steps) * log_q)
    da_dk = c * (-float(steps)) * np.exp((-float(steps) - 1) * log_q) * dt * mu
    fN, dfN = V @ a, V @ da_dk
    w = (ny + 1) * (nz + 1)
    return w * float(fN @ fN), w * 2.0 * float(fN @ dfN)


def continuum_J(
    nx: int, ny: int, nz: int, Lx: float, rho_0: float, t_end: float
) -> float:
    """sum_nodes of the continuum T(t_end)^2 (the nodal sum of the exact solution)."""
    x = np.linspace(0.0, Lx, nx + 1)
    f0 = eigenmode(x, Lx).astype(np.float64)
    lam = conductivity(rho_0) * (np.pi / (2 * Lx)) ** 2
    return (ny + 1) * (nz + 1) * float(np.sum(f0**2)) * np.exp(-2 * lam * t_end)


def _mesh_args(nx: int, ny: int | None, nz: int, N: int | None):
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)
    return nx, ny, nz


def discrete_output(
    ic: Any,
    t: float,
    L: float = 2.0,
    nx: int = 16,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    Lx: float = 2.0,
    rho_0: float = 0.5,
    steps: int = 100,
) -> np.ndarray:
    """``forward/agreement`` reference: the exact discrete output J_h at t = t_end."""
    nx, ny, nz = _mesh_args(nx, ny, nz, N)
    return np.asarray(discrete_J_and_dJdk(nx, ny, nz, Lx, rho_0, steps, t)[0])


def continuum_output(
    ic: Any,
    t: float,
    L: float = 2.0,
    nx: int = 16,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    Lx: float = 2.0,
    rho_0: float = 0.5,
) -> np.ndarray:
    """``forward/baseline`` reference: the nodal sum of the continuum solution at t."""
    nx, ny, nz = _mesh_args(nx, ny, nz, N)
    return np.asarray(continuum_J(nx, ny, nz, Lx, rho_0, t))


def discrete_dloss_drho_uniform(
    ic: Any,
    t: float,
    nx: int = 16,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    Lx: float = 2.0,
    rho_0: float = 0.5,
    steps: int = 100,
) -> float:
    """``analytic_param_check`` reference with wrt="rho": d(J^2)/d(eps) for rho + eps.

    That is what the harness gets by summing the VJP over the cells. Exact for the
    discrete scheme: d(J^2)/d(eps) = 2 J_h dJ_h/dk dk/drho.
    """
    nx, ny, nz = _mesh_args(nx, ny, nz, N)
    J, dJdk = discrete_J_and_dJdk(nx, ny, nz, Lx, rho_0, steps, t)
    return 2.0 * J * dJdk * dconductivity(rho_0)
