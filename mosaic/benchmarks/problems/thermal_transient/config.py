# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quasi-2D transient heat conduction with a closed-form reference (mosaic#375 idea 4).

The thermal-mesh slab, solvers and SIMP material, with a time loop: the density
is uniform, there is no heat flux or source, and the initial temperature is the
eigenmode sin(pi x / (2 Lx)) of the cold-left-face, insulated-elsewhere slab, so the
temperature decays as exp(-lam t) in closed form (:mod:`.references`). The output
is J = sum over nodes of T(t_end)^2 (``identification_error`` with a zero target).

The point of the problem is the time axis that thermal-mesh does not have: a
sweep over the number of steps to a fixed end time exposes how each solver's
reverse mode stores the forward pass (a full tape grows with the step count,
checkpointing does not), with references that do not depend on the peers:

- the exact discrete solution of the scheme every solver implements here (trilinear
  hexahedra, backward Euler), for the forward value and for the gradient along a
  uniform change in density, so their error is the solver's alone;
- the continuum solution, for the forward value against mesh refinement.

Meshes use nx a power of two so that the node coordinates are exact in float32;
otherwise the discrete reference is exact only to about 1e-7.
"""

from __future__ import annotations

from mosaic.benchmarks.core.config import Problem, SolverSpec, discover_solvers
from mosaic.benchmarks.core.status_checks import (
    max_error,
    max_peer_k,
    max_rel_err,
    median_k,
    min_cosine,
    rel_err_peer_outlier,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.problems.shared.cost import temporal_cost, vjp_cost
from mosaic.benchmarks.problems.shared.forward import agreement
from mosaic.benchmarks.problems.shared.gradient import (
    analytic_param_check,
    fd_check,
    horizon_sweep_limits,
)
from mosaic.benchmarks.problems.shared.plots.cost import plot_cost
from mosaic.benchmarks.problems.shared.plots.forward import plot_agreement
from mosaic.benchmarks.problems.shared.plots.gradient import (
    plot_analytic_check,
    plot_fd_check,
    plot_horizon_sweep_limits,
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles
from mosaic.benchmarks.problems.thermal_mesh.ics import _random, _uniform

from .exclusions import register as _register_exclusions
from .physics import make_inputs
from .references import (
    K_MAX,
    P_EXP,
    continuum_output,
    discrete_dloss_drho_uniform,
    discrete_output,
)

_TESSERACT_SLUG = "thermal-mesh"

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)
apply_styles(_SOLVERS)
for _key in ("fenics_heat", "dealii_heat", "firedrake_heat", "torch_fem_thermal"):
    if _key in _SOLVERS:
        _SOLVERS[_key].input_overrides = {"k_max": K_MAX, "p_exp": P_EXP}

problem = Problem(
    name="thermal-transient",
    category_label="Heat transfer",
    description=(
        "**How does reverse mode cope with a long time horizon?** A slab cools from a "
        "single eigenmode, so the temperature and its sensitivity to the conductivity "
        "are known in closed form at every time. Sweeping the number of backward Euler "
        "steps to a fixed end time shows how each solver's reverse mode stores the "
        "forward pass: a full tape grows with the number of steps, checkpointing trades "
        "that memory for recomputation. Diffusion is contractive, so the gradient stays "
        "well conditioned at any horizon, and a failure here comes from memory or the "
        "differentiation strategy rather than from the physics."
    ),
    bc_description=(
        "The thermal-mesh slab $[0,2]\\times[0,1]$ (one HEX8 layer, $L_z = L_x/n_x$), "
        "uniform density $\\rho_0 = 0.5$ with SIMP conductivity $k(\\rho_0)$, unit heat "
        "capacity. Dirichlet $T=0$ at $x=0$, all other faces insulated, no heat flux or "
        "source. Initial temperature $T_0 = \\sin(\\pi x / 2L_x)$, so "
        "$T = T_0\\,e^{-\\lambda t}$ with $\\lambda = k (\\pi/2L_x)^2$; the end time is "
        "$t_\\text{end} = 1/\\lambda$ and $dt = t_\\text{end}/\\text{steps}$. Output: "
        "$J = \\sum_\\text{nodes} T(t_\\text{end})^2$."
    ),
    tesseract_dir=_TESSERACT_SLUG,
    output_key="identification_error",
    ic_key="rho",
    solvers=list(_SOLVERS.values()),
    make_inputs=make_inputs,
    error_fn=l2_error_rel,
    domain_extent=2.0,
    resolution_key="nx",
    status_checks={
        "forward": [median_k(3.0, floor=1e-5)],
        "cost": [max_peer_k(20.0)],
        # Peer checks cannot single out one of two solvers, so each check also has
        # an absolute bound (measured: 1.4e-5 median at the best step size).
        "gradient/fd_check": [
            min_cosine(0.99),
            rel_err_peer_outlier(50.0),
            max_rel_err(1e-2),
        ],
    },
)

problem.add_ic(
    "uniform",
    fn=_uniform,
    description="Uniform density $\\rho_0$, the setting in which the closed forms hold.",
    plot_params={"rho_0": 0.5, "nx": 16, "ny": 8, "nz": 1},
    plot=plot_ic,
)
problem.add_ic(
    "random",
    fn=_random,
    description=(
        "Gaussian-noise density around $\\rho_0 = 0.5$, for the finite-difference check "
        "(per-cell sensitivities that are not all equal)."
    ),
    plot_params={"rho_0": 0.5, "noise": 0.3, "nx": 16, "ny": 8, "nz": 1, "seed": 0},
    plot=plot_ic,
)

_SLAB = {"ny": None, "nz": 1, "Lx": 2.0, "Ly": 1.0, "rho_0": 0.5}

# Forward
problem.add_experiment(
    "forward/agreement",
    agreement,
    plot_description=(
        "$J$ against the number of steps to a fixed end time, scored against the exact "
        "solution of the discrete scheme (trilinear hexahedra, backward Euler)."
    ),
    ic={"name": "uniform", "seed": 0},
    physics={**_SLAB, "nx": 16, "steps": [10, 100, 1000]},
    reference=discrete_output,
    plot=plot_agreement,
    # Against the exact discrete solution the error is float32 round-off (measured
    # 1e-8 to 2e-8), so an absolute bound singles out a wrong solver even with two.
    status_check=[median_k(3.0, floor=1e-5), max_error(1e-6)],
)
problem.add_experiment(
    "forward/baseline",
    agreement,
    plot_description=(
        "$J$ against mesh resolution $N$ at 1000 steps, scored against the continuum "
        "solution (its error is the scheme's spatial and temporal error)."
    ),
    ic={"name": "uniform", "seed": 0},
    physics={**_SLAB, "N": [4, 8, 16, 32], "steps": 1000},
    reference=continuum_output,
    plot=plot_agreement,
)

# Gradient
problem.add_experiment(
    "gradient/fd_check",
    fd_check,
    plot_description=(
        "FD gradient error vs step size, AD/FD direction cosine and gradient fields for "
        "a random density, 50 steps."
    ),
    ic={"name": "random", "seed": 0},
    physics={**_SLAB, "nx": 8, "steps": 50},
    fd={"eps_values": [1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 6},
    plot=plot_fd_check,
)
problem.add_experiment(
    # Named analytic_check so the status classifier reads its per-step errors
    # (core/status.py routes only analytic_check and analytic_check_nu).
    "gradient/analytic_check",
    analytic_param_check,
    plot_description=(
        "Relative error of the gradient of $J^2$ summed over the cells (the derivative "
        "along a uniform change in density) against its exact value for the discrete "
        "scheme, over the number of steps, with the full tape and with checkpointing: "
        "the differentiation error alone."
    ),
    ic={"name": "uniform", "seed": 0},
    runs=[
        {
            "name": "tape",
            "physics": {**_SLAB, "nx": 16, "steps": [10, 100, 1000, 10000]},
        },
        {
            "name": "checkpoint",
            "physics": {
                **_SLAB,
                "nx": 16,
                "steps": [10, 100, 1000, 10000],
                "adjoint_checkpoints": "sqrt",
            },
        },
    ],
    wrt="rho",
    reference=discrete_dloss_drho_uniform,
    plot=plot_analytic_check,
    # Provisional, measured on Firedrake and FEniCS (2e-9 to 3e-8, float32 round-off).
    # A continuous adjoint is O(dt) off, about 10% at 10 steps, so the bound at each
    # step count follows dt: an O(dt) adjoint passes and a wrong gradient does not,
    # whichever points are present.
    status_check=[
        median_k(3.0, floor=1e-5),
        max_error(0.2, at=10),
        max_error(0.02, at=100),
        max_error(2e-3, at=1000),
        max_error(1e-3, at=10000),
    ],
)
problem.add_experiment(
    "gradient/horizon_sweep_limits",
    horizon_sweep_limits,
    plot_description=(
        "Per-solver rollout limits: the step count at first failure (out of memory, "
        "timeout), and wall time and peak memory per successful VJP, from 10 to "
        "$10^5$ steps to a fixed end time."
    ),
    ic={"name": "uniform", "seed": 0},
    runs=[
        {
            "name": "tape",
            "physics": {**_SLAB, "nx": 32, "steps": [10, 100, 1000, 10000, 100000]},
        },
        {
            "name": "checkpoint",
            "physics": {
                **_SLAB,
                "nx": 32,
                "steps": [10, 100, 1000, 10000, 100000],
                "adjoint_checkpoints": "sqrt",
            },
        },
    ],
    plot=plot_horizon_sweep_limits,
)

# Cost
problem.add_experiment(
    "cost/temporal_cost",
    temporal_cost,
    plot_description="Forward wall-clock time against the number of steps.",
    physics={**_SLAB, "nx": 32, "steps": [10, 100, 1000, 10000]},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/vjp_cost",
    vjp_cost,
    plot_description="VJP wall-clock time against mesh size and against the number of steps.",
    runs=[
        {
            "name": "by_N",
            "physics": {**_SLAB, "nx": [8, 16, 32, 64], "steps": 100},
            "cost": {"n_trials": 3},
        },
        {
            "name": "by_steps",
            "physics": {**_SLAB, "nx": 32, "steps": [10, 100, 1000, 10000]},
            "cost": {"n_trials": 3},
        },
    ],
    plot=plot_cost,
)

_register_exclusions(problem)

__all__ = ["problem"]
