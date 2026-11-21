# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thermal topology optimisation on a structured hexahedral mesh via torch-fem.

Uses Meyer-Nils/torch-fem `SolidHeat` (HEX8 Galerkin) with PyTorch autograd
for SIMP topology-optimisation sensitivities.  This is the first PyTorch
backend on thermal-mesh and the first GPU-native thermal VJP path in Mosaic.

SIMP conductivity:
    k(ρ) = k_min + (k_max − k_min) · ρ^p    (k_min = 1e-3 · k_max)

Objective:
    C = ∮_Γ_N q_n · T dΓ    (implemented as Σ f_ext · T_nodal)

Source-identification objective:
    E = Σ_i (T_i − T_target_i)^2

Gradients (rho and source) flow through `torch.autograd` over the entire
FE solve thanks to `model.solve(differentiable_parameters=...)` in torch-fem.

With ``steps > 0`` (thermal-transient) the solver takes ``steps`` backward Euler
steps from ``initial_temperature`` as an iterative backend: torch-fem assembles
the element conductivity and mass matrices, and every step is a Jacobi-PCG solve
to ``cg_rtol``, warm-started from the previous step (see ``_transient_torchfem``).
"""

from typing import Any, Literal

import numpy as np
import torch
from mosaic_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.thermal_transient import TransientFields
from mosaic_shared.schema_types import make_differentiable
from pydantic import Field
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from tesseract_core.runtime import ShapeDType
from torchfem import SolidHeat
from torchfem.materials import IsotropicConductivity3D


class InputSchema(
    make_differentiable(_CanonicalInputSchema, ["rho", "source"]), TransientFields
):
    """Inputs for torch-fem thermal solver (canonical + SIMP material params).

    With ``steps > 0`` (thermal-transient) the solver takes ``steps`` backward Euler
    steps from ``initial_temperature`` instead of the steady solve.
    """

    k_max: float = Field(
        default=1.0,
        description="Maximum thermal conductivity (fully solid material).",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p (k(ρ) = k_min + (k_max−k_min)·ρ^p).",
    )
    cg_rtol: float = Field(
        default=1e-6,
        gt=0.0,
        lt=1.0,
        description=(
            "thermal-transient only: relative residual tolerance of every per-step "
            "Jacobi-PCG solve, ||b - A x|| <= cg_rtol ||b||, warm-started from the "
            "previous step."
        ),
    )
    adjoint_mode: Literal["implicit", "unrolled"] = Field(
        default="implicit",
        description=(
            "thermal-transient only: 'implicit' solves the transposed step system "
            "exactly (sparse LU) at each returned state, the discrete adjoint along "
            "the computed trajectory; 'unrolled' differentiates through the CG "
            "iterations the forward executed."
        ),
    )


class OutputSchema(
    make_differentiable(
        _CanonicalOutputSchema, ["thermal_compliance", "identification_error"]
    )
):
    """Torch-FEM thermal solver output schema."""


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

# Auto-detect CUDA.  torch-fem internally allocates many auxiliary index /
# buffer tensors via ``torch.arange(...)`` / ``torch.zeros(...)`` without an
# explicit ``device=`` kwarg, so we set ``torch.set_default_device(_DEVICE)``
# below to ensure those allocations land on the chosen device instead of
# silently reverting to CPU (which would trigger device mismatches once our
# mesh / material / force tensors are on CUDA).
#
# Sparse linear solve: on GPU we use Jacobi-preconditioned CG in torch, so the
# forward and adjoint solves stay on the device.  On CPU-only hosts torch-fem
# picks the solver automatically (SciPy).
#
# The CG stops at a relative residual of 1e-6 instead of torch-fem's 1e-10. No
# check changes between 1e-12 and 1e-6; on the small forward meshes CG reaches
# round-off before its first residual test either way.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SOLVE_KWARGS: dict[str, Any] = (
    {"method": "cg", "preconditioner": "jacobi", "stol": 1e-6}
    if _DEVICE.type == "cuda"
    else {}
)
_DTYPE = torch.float64  # torch-fem defaults to float64 for solver stability

# torch-fem internals assume torch.get_default_dtype() matches the tensors we
# pass in (see sparse.py:eval_residual where the NR solver allocates a DU in
# default dtype and assigns into it from our float64 BC tensors).  Align the
# global default so the Newton iterate, BC buffer, and residual are all
# float64 end-to-end.  Similarly, align the default device so torch-fem's
# internal ``torch.zeros/arange(...)`` calls land on the same device as the
# tensors we construct explicitly with ``device=_DEVICE``.
torch.set_default_dtype(_DTYPE)
torch.set_default_device(_DEVICE)
torch.use_deterministic_algorithms(True, warn_only=True)


# SIMP parameters (baked into the InputSchema defaults but kept here as fallbacks)
_K_MIN_RATIO = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mesh(
    points_np: np.ndarray, cells_np: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy mesh arrays to torch tensors on the chosen device."""
    nodes = torch.as_tensor(points_np, dtype=_DTYPE, device=_DEVICE)
    elements = torch.as_tensor(cells_np, dtype=torch.long, device=_DEVICE)
    return nodes, elements


def _build_bc_tensors(
    model: SolidHeat, n_nodes: int, bc_dict: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build constraint / temperature / Neumann heat-flux tensors (all shape (n_nodes, 1)).

    - Dirichlet nodes: constraints[i]=True, temperatures[i]=value.
    - Neumann face flux: q_n integrated over the boundary faces whose nodes
      are all marked by the mask (the right face, or a strip of it).

    All boundary-condition tensors are allocated on _DEVICE with the
    appropriate dtype (bool for the constraints mask, _DTYPE for the nodal
    temperature/heat-flux values) so they can be assigned directly to the
    corresponding ``model.*`` properties without a device hop.
    """
    constraints = torch.zeros(n_nodes, 1, dtype=torch.bool, device=_DEVICE)
    temperatures = torch.zeros(n_nodes, 1, dtype=_DTYPE, device=_DEVICE)
    heat_flux = torch.zeros(n_nodes, 1, dtype=_DTYPE, device=_DEVICE)

    # --- Dirichlet ---
    d_bc = bc_dict.get("dirichlet") or {}
    d_mask = np.asarray(d_bc.get("mask", []), dtype=np.int32)
    d_vals_raw = d_bc.get("values")
    d_vals = (
        np.asarray(d_vals_raw, dtype=np.float64)
        if d_vals_raw is not None
        else np.zeros((0, 1), dtype=np.float64)
    )
    if d_mask.size > 0:
        mask = np.zeros(n_nodes, dtype=np.int32)
        mask[: min(n_nodes, d_mask.size)] = d_mask[:n_nodes]
        constrained = np.where(mask > 0)[0]
        if constrained.size > 0:
            groups = mask[constrained] - 1
            prescribed = (
                d_vals[groups, 0]
                if d_vals.size > 0
                else np.zeros(constrained.size, dtype=np.float64)
            )
            c_idx = torch.as_tensor(constrained, dtype=torch.long, device=_DEVICE)
            constraints[c_idx, 0] = True
            temperatures[c_idx, 0] = torch.as_tensor(
                prescribed, dtype=_DTYPE, device=_DEVICE
            )

    # --- Neumann flux (uniform q_n) on the marked boundary faces ---
    n_bc = bc_dict.get("neumann") or {}
    n_mask = np.asarray(n_bc.get("mask", []), dtype=np.int32)
    n_vals_raw = n_bc.get("values")
    n_vals = (
        np.asarray(n_vals_raw, dtype=np.float64)
        if n_vals_raw is not None
        else np.zeros((0, 1), dtype=np.float64)
    )
    if n_mask.size > 0 and n_vals.size > 0:
        mask = np.zeros(n_nodes, dtype=bool)
        mask[: min(n_nodes, n_mask.size)] = n_mask[:n_nodes] > 0
        if mask.any():
            heat_flux = model.integrate_surface_load(
                torch.as_tensor(mask, device=_DEVICE),
                torch.tensor([float(n_vals[0, 0])], dtype=_DTYPE, device=_DEVICE),
            )

    return constraints, temperatures, heat_flux


def _forward_torchfem(
    points_np: np.ndarray,
    cells_np: np.ndarray,
    bc_dict: dict,
    rho_t: torch.Tensor,
    source_t: torch.Tensor,
    k_max: float,
    p_exp: float,
    differentiable_params: tuple | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a torch-fem SolidHeat forward solve and return (T_nodal, f_internal).

    Notes:
        - ``rho_t`` and ``source_t`` are (n_cells,) tensors on ``_DEVICE``.  If
          ``differentiable_params`` is not None they must be leaves with
          ``requires_grad=True`` (or share a graph with one).
        - SIMP is implemented by scaling ``material.KAPPA`` by
          ``k_min_ratio + (1 - k_min_ratio) * rho^p`` after ``.vectorize``;
          the base material is created at ``kappa = k_max`` so the final
          per-element conductivity is ``k = k_max · (k_min_ratio + (1-k_min_ratio)·ρ^p)
          = k_min + (k_max-k_min)·ρ^p``.
        - The source contributes to ``model.heat_flux`` through ``source_t``; if
          ``source_t`` is a live autograd leaf its grad flows back too.
    """
    n_nodes = points_np.shape[0]

    nodes, elements = _build_mesh(points_np, cells_np)

    # Base material at k_max; we scale per-element below via SIMP.
    material = IsotropicConductivity3D(kappa=float(k_max))
    model = SolidHeat(nodes, elements, material)

    # Vectorise the material across elements (creates KAPPA of shape (n_elem, 3, 3))
    model.material = material.vectorize(model.n_elem)

    # SIMP scaling: k = k_min + (k_max - k_min) * rho^p
    #             = k_max · (k_min_ratio + (1 - k_min_ratio) · rho^p)
    simp_scale = _K_MIN_RATIO + (1.0 - _K_MIN_RATIO) * rho_t**p_exp  # (n_cells,)
    # model.material.KAPPA shape (n_cells, 3, 3); multiply elementwise along elem dim
    model.material.KAPPA = simp_scale[:, None, None] * model.material.KAPPA

    # Boundary conditions and Neumann heat flux (source added separately below)
    constraints, temperatures, heat_flux_base = _build_bc_tensors(
        model, n_nodes, bc_dict
    )

    # Add the volumetric source as a body load.  We always build it so
    # torch-fem's NewtonRaphsonAdjoint sees source_t as a live dependency of
    # ``model.heat_flux`` (otherwise ``eval_residual`` has no path back to source
    # and ``torch.autograd.grad`` returns None for that parameter).
    src_nodal = model.integrate_body_load(source_t[:, None])  # (n_nodes, 1)
    heat_flux_final = heat_flux_base + src_nodal  # (n_nodes, 1)

    model.constraints = constraints
    model.temperatures = temperatures
    model.heat_flux = heat_flux_final

    # Linear solve (solver choice: see ``_SOLVE_KWARGS`` above).  The returned
    # tensor path goes through SolidHeat's custom autograd so gradients flow
    # back to differentiable_parameters on _DEVICE.
    u_k, f_k, *_ = model.solve(
        differentiable_parameters=differentiable_params,
        **_SOLVE_KWARGS,
    )
    # Return u (nodal T), f_k (full internal force = f_ext at equilibrium),
    # and the Neumann-only sub-force needed to compute the boundary integral
    # ∮_Γ_N q_n T dΓ = f_Neumann^T · T.  Without this split, compliance and
    # source-gradient would pick up the volumetric source contribution too,
    # which disagrees with the peer solvers that integrate over Γ_N only.
    return u_k, f_k, heat_flux_base


# ---------------------------------------------------------------------------
# Transient solve (thermal-transient: steps > 0)
# ---------------------------------------------------------------------------
#
# torch-fem's own transient heat solver (``SolidHeat.integrate``) is Crank-Nicolson
# inside a Newton loop and starts every linear solve from zero, so the time loop is
# written here: backward Euler, as the other thermal-transient backends, with the
# conductivity and mass matrices from torch-fem and one warm-started Jacobi-PCG solve
# per step to ``cg_rtol``. The step matrix A = M + dt K(rho) acts on the free nodes
# element by element; Dirichlet nodes are held at their prescribed values.


class _StepOperator:
    """A = M + dt sum_e w_e k_e restricted to the free nodes, element by element."""

    def __init__(
        self,
        elements: torch.Tensor,
        m_e: torch.Tensor,
        k_e: torch.Tensor,
        dt: float,
        free: torch.Tensor,
    ) -> None:
        self.elements = elements
        self.m_e = m_e
        self.k_e = k_e
        self.dt = dt
        self.free = free  # (n_nodes,) float mask, 0 at Dirichlet nodes
        self.n = free.shape[0]
        self._lu = None

    def _scatter(self, ye: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.n, dtype=ye.dtype, device=ye.device)
        return out.index_add(0, self.elements.reshape(-1), ye.reshape(-1))

    def mass(self, x: torch.Tensor) -> torch.Tensor:
        """M x over all nodes."""
        return self._scatter(torch.einsum("eij,ej->ei", self.m_e, x[self.elements]))

    def full(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """(M + dt K(w)) x over all nodes."""
        xe = x[self.elements]
        ye = torch.einsum("eij,ej->ei", self.m_e, xe) + self.dt * w[:, None] * (
            torch.einsum("eij,ej->ei", self.k_e, xe)
        )
        return self._scatter(ye)

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.free * self.full(self.free * x, w)

    def diagonal(self, w: torch.Tensor) -> torch.Tensor:
        de = torch.diagonal(self.m_e, dim1=1, dim2=2) + self.dt * w[:, None] * (
            torch.diagonal(self.k_e, dim1=1, dim2=2)
        )
        return self._scatter(de)

    def exact_solve(self, rhs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """A^-1 rhs on the free nodes by sparse LU (factorised once per call)."""
        free_idx = torch.nonzero(self.free > 0).reshape(-1).cpu().numpy()
        if self._lu is None:
            el = self.elements.cpu().numpy()
            vals = (
                (self.m_e + self.dt * w.detach()[:, None, None] * self.k_e)
                .detach()
                .cpu()
                .numpy()
            )
            rows = np.repeat(el, el.shape[1], axis=1).reshape(-1)
            cols = np.tile(el, (1, el.shape[1])).reshape(-1)
            A = coo_matrix((vals.reshape(-1), (rows, cols)), shape=(self.n, self.n))
            self._lu = splu(A.tocsr()[free_idx][:, free_idx].tocsc())
        out = np.zeros(self.n)
        out[free_idx] = self._lu.solve(rhs.detach().cpu().numpy()[free_idx])
        return torch.as_tensor(out, dtype=rhs.dtype, device=rhs.device)


def _pcg(
    op: _StepOperator,
    w: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    rtol: float,
) -> torch.Tensor:
    """Jacobi-PCG from x0 until ||b - A x|| <= rtol ||b|| (tested every iteration).

    Written out of place, so with autograd enabled the graph holds exactly the
    iterations executed (the stopping test itself is not differentiated).
    """
    b_norm = float(torch.linalg.norm(b))
    if b_norm == 0.0:
        return torch.zeros_like(b)
    tol = rtol * b_norm
    d = op.diagonal(w)
    d_inv = op.free / torch.where(op.free > 0, d, torch.ones_like(d))
    x = x0
    r = b - op(x, w)
    if float(torch.linalg.norm(r)) <= tol:
        return x
    z = d_inv * r
    p = z
    rz = torch.dot(r, z)
    for _ in range(10 * op.n):
        Ap = op(p, w)
        alpha = rz / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if float(torch.linalg.norm(r)) <= tol:
            return x
        z = d_inv * r
        rz_next = torch.dot(r, z)
        p = z + (rz_next / rz) * p
        rz = rz_next
    raise RuntimeError(
        f"CG did not reach cg_rtol={rtol:g} within {10 * op.n} iterations"
    )


class _ImplicitStep(torch.autograd.Function):
    """One backward Euler solve; the backward pass is the exact transposed solve.

    The forward stops CG at ``rtol``; the backward treats the returned state x as the
    solution of A(w) x = b and returns lambda = A^-T g_x for b and
    -dt lambda_e^T k_e x_e for w, so the chain over the steps is the discrete
    adjoint along the computed trajectory.
    """

    @staticmethod
    def forward(
        ctx: Any,
        w: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor,
        op: _StepOperator,
        rtol: float,
    ) -> torch.Tensor:
        x = _pcg(op, w.detach(), b.detach(), x0.detach(), rtol)
        ctx.save_for_backward(w, x)
        ctx.op = op
        return x

    @staticmethod
    def backward(ctx: Any, g_x: torch.Tensor) -> tuple:
        w, x = ctx.saved_tensors
        op = ctx.op
        lam = op.exact_solve(op.free * g_x, w)  # A is symmetric
        g_w = -op.dt * torch.einsum(
            "ei,eij,ej->e", lam[op.elements], op.k_e, x[op.elements]
        )
        return g_w, lam, None, None, None


def _transient_torchfem(
    points_np: np.ndarray,
    cells_np: np.ndarray,
    bc_dict: dict,
    rho_t: torch.Tensor,
    source_t: torch.Tensor,
    k_max: float,
    p_exp: float,
    inputs_dict: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward Euler from the initial temperature; returns (T_final, f_Neumann).

    Each step solves (M + dt K(rho)) T_{n+1} = M T_n + dt (f_source + f_Neumann)
    (unit heat capacity, SIMP conductivity as in the steady path) on the free nodes
    by Jacobi-PCG to ``cg_rtol`` from T_n. ``adjoint_mode`` picks how the gradient
    passes through each solve: the exact transposed solve at the returned state
    (``implicit``) or autograd through the executed iterations (``unrolled``).
    """
    n_nodes = points_np.shape[0]
    steps = int(inputs_dict["steps"])
    dt = float(np.asarray(inputs_dict["dt"]).reshape(-1)[0])
    rtol = float(inputs_dict.get("cg_rtol", 1e-6))
    unrolled = inputs_dict.get("adjoint_mode", "implicit") == "unrolled"

    nodes, elements = _build_mesh(points_np, cells_np)
    # Unit density, so the torch-fem mass matrix is that of unit heat capacity.
    material = IsotropicConductivity3D(kappa=float(k_max))
    model = SolidHeat(nodes, elements, material)
    model.material = material.vectorize(model.n_elem)
    k_e = model.k0().detach()  # element conductivity at k_max
    m_e = model.integrate_mass().detach()

    constraints, temperatures, heat_flux_base = _build_bc_tensors(
        model, n_nodes, bc_dict
    )
    src_nodal = model.integrate_body_load(source_t[:, None])
    load = dt * (src_nodal + heat_flux_base).reshape(-1)

    con = constraints.reshape(-1)
    free = (~con).to(_DTYPE)
    g = torch.where(con, temperatures.reshape(-1), torch.zeros_like(free))
    op = _StepOperator(elements, m_e, k_e, dt, free)

    # SIMP: k = k_max (k_min_ratio + (1 - k_min_ratio) rho^p), as in the steady path.
    w = _K_MIN_RATIO + (1.0 - _K_MIN_RATIO) * rho_t**p_exp

    T0_np = np.zeros(n_nodes)
    T0_in = np.asarray(inputs_dict["initial_temperature"], dtype=np.float64).reshape(-1)
    T0_np[: min(n_nodes, T0_in.size)] = T0_in[:n_nodes]
    T = torch.where(con, g, torch.as_tensor(T0_np, dtype=_DTYPE, device=_DEVICE))

    for _ in range(steps):
        b = free * (op.mass(T) + load - op.full(g, w))
        x0 = free * T
        if unrolled:
            x = _pcg(op, w, b, x0, rtol)
        else:
            x = _ImplicitStep.apply(w, b, x0, op, rtol)
        T = g + x
    return T, heat_flux_base


def _apply_core(inputs_dict: dict, want_grad: bool) -> dict:
    """Shared forward solver used by both ``apply`` and ``vector_jacobian_product``.

    Args:
        inputs_dict: Raw dict form of the InputSchema.
        want_grad: If True, input tensors are set up as live autograd leaves.
    """
    hm = inputs_dict["hex_mesh"]
    n_cells = int(hm["n_faces"])
    n_nodes = int(hm["n_points"])
    points_np = np.asarray(hm["points"][:n_nodes], dtype=np.float64)
    cells_np = np.asarray(hm["faces"][:n_cells], dtype=np.int64)

    # Clip before rho becomes an autograd leaf; torch.clamp would zero the
    # gradient at rho = 0 and rho = 1.
    rho_np = np.clip(np.asarray(inputs_dict["rho"][:n_cells], dtype=np.float64), 0, 1)
    source_np = np.asarray(inputs_dict["source"][:n_cells], dtype=np.float64)
    target_np = np.asarray(inputs_dict.get("target_temperature", []), dtype=np.float64)

    k_max = float(inputs_dict.get("k_max", 1.0))
    p_exp = float(inputs_dict.get("p_exp", 3.0))

    rho_t = torch.as_tensor(rho_np, dtype=_DTYPE, device=_DEVICE)
    source_t = torch.as_tensor(source_np, dtype=_DTYPE, device=_DEVICE)
    if want_grad:
        rho_t.requires_grad_(True)
        source_t.requires_grad_(True)
        diff_params = (rho_t, source_t)
    else:
        diff_params = None

    if int(inputs_dict.get("steps", 0)) > 0:
        u_k, f_neumann = _transient_torchfem(
            points_np,
            cells_np,
            inputs_dict["boundary_conditions"],
            rho_t,
            source_t,
            k_max,
            p_exp,
            inputs_dict,
        )
    else:
        u_k, _f_k, f_neumann = _forward_torchfem(
            points_np,
            cells_np,
            inputs_dict["boundary_conditions"],
            rho_t,
            source_t,
            k_max,
            p_exp,
            diff_params,
        )

    # Compliance C = ∮_Γ_N q_n · T dΓ = f_Neumann^T · T (boundary work only,
    # not the total internal work).  Using f_k (full internal force) would add
    # the volumetric source contribution and break agreement with the peer
    # solvers' surface-integral compliance.
    thermal_compliance = torch.inner(f_neumann.reshape(-1), u_k.reshape(-1))

    # Identification error: Σ (T_nodal - T_target)^2
    T_nodal = u_k.reshape(-1)
    if target_np.size > 0:
        target_t = torch.as_tensor(target_np, dtype=_DTYPE, device=_DEVICE)
        n = min(T_nodal.shape[0], target_t.shape[0])
        diff = T_nodal[:n] - target_t[:n]
        identification_error = torch.sum(diff * diff)
    else:
        identification_error = torch.zeros((), dtype=_DTYPE, device=_DEVICE)

    return {
        "thermal_compliance": thermal_compliance,
        "identification_error": identification_error,
        "rho_t": rho_t,
        "source_t": source_t,
    }


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve heat conduction and return compliance + identification error.

    Uses torch-fem SolidHeat with SIMP-scaled IsotropicConductivity3D material.
    """
    out = _apply_core(inputs.model_dump(), want_grad=False)
    tc = out["thermal_compliance"].detach().cpu().numpy().astype(np.float32)
    ide = out["identification_error"].detach().cpu().numpy().astype(np.float32)
    return {
        "thermal_compliance": np.float32(tc),
        "identification_error": np.float32(ide),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via torch.autograd.grad through the torch-fem SolidHeat solve.

    Supports gradients w.r.t. ``rho`` and ``source``; both flow through
    ``differentiable_parameters=(rho, source)`` — torch-fem's custom autograd
    implements the implicit-function adjoint for the linear FE system.
    """
    assert vjp_inputs <= {"rho", "source"}
    assert vjp_outputs <= {"thermal_compliance", "identification_error"}

    if not vjp_inputs:
        return {}

    inputs_dict = inputs.model_dump()
    out = _apply_core(inputs_dict, want_grad=True)

    rho_t = out["rho_t"]
    source_t = out["source_t"]

    # Build the scalar loss by summing each output against its cotangent so a
    # single backward pass yields the full VJP over whatever outputs we're asked
    # about.  Missing / zero cotangents contribute nothing.
    loss_terms: list[torch.Tensor] = []
    for key, tensor in (
        ("thermal_compliance", out["thermal_compliance"]),
        ("identification_error", out["identification_error"]),
    ):
        if key not in vjp_outputs:
            continue
        if key not in cotangent_vector:
            continue
        ct_val = cotangent_vector[key]
        ct_t = torch.as_tensor(
            np.asarray(ct_val, dtype=np.float64), dtype=_DTYPE, device=_DEVICE
        )
        if ct_t.ndim == 0 and tensor.ndim == 0:
            loss_terms.append(tensor * ct_t)
        else:
            # Flatten both sides and align sizes
            t_flat = tensor.reshape(-1)
            c_flat = ct_t.reshape(-1)
            n = min(t_flat.shape[0], c_flat.shape[0])
            loss_terms.append(torch.sum(t_flat[:n] * c_flat[:n]))

    n_rho = int(np.asarray(inputs_dict["rho"]).shape[0])
    n_source = int(np.asarray(inputs_dict["source"]).shape[0])
    n_cells_active = int(inputs_dict["hex_mesh"]["n_faces"])

    result: dict[str, Any] = {}

    if not loss_terms:
        # No cotangent path — return zeros for every requested input.
        if "rho" in vjp_inputs:
            result["rho"] = np.zeros(n_rho, dtype=np.float32)
        if "source" in vjp_inputs:
            result["source"] = np.zeros(n_source, dtype=np.float32)
        return result

    loss = sum(loss_terms)

    targets: list[torch.Tensor] = []
    keys: list[str] = []
    if "rho" in vjp_inputs:
        targets.append(rho_t)
        keys.append("rho")
    if "source" in vjp_inputs:
        targets.append(source_t)
        keys.append("source")

    # An unrolled solve whose CG ran no iterations (zero right-hand side, or a warm
    # start already within tolerance) does not depend on the inputs: zero gradient.
    if not loss.requires_grad:
        grads = [None] * len(targets)
    else:
        grads = torch.autograd.grad(loss, targets, allow_unused=True)

    for key, g in zip(keys, grads, strict=False):
        n_full = n_rho if key == "rho" else n_source
        arr = np.zeros(n_full, dtype=np.float32)
        if g is not None:
            g_np = g.detach().cpu().numpy().astype(np.float32)
            n_copy = min(arr.shape[0], g_np.shape[0], n_cells_active)
            arr[:n_copy] = g_np[:n_copy]
        result[key] = arr

    return result


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
