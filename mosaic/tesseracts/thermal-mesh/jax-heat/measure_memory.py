# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""XLA's predicted device memory of the jax-heat gradient against the measured peak.

Runs the tesseract API in-process (no Docker): needs JAX, tesseract-core and
mosaic_shared only. One case per process, so the allocator's peak belongs to it:

    python measure_memory.py --nx 64 --steps 10 100 1000 10000 100000 --out results.jsonl
    python measure_memory.py --nx 64 --steps 100000 --checkpoints sqrt --out results.jsonl
    python measure_memory.py --nx 256 --steps 100000 --compile-only   # prediction only

Each case is the thermal-transient setup (the thermal-mesh slab, one HEX8 layer,
ny = nx / 2, uniform rho = 0.5, eigenmode start, t_end with lam t_end = 1) and
appends one JSON line: XLA's memory analysis of the compiled VJP (bytes), and,
unless --compile-only, the device's bytes in use before the VJP and its peak after
(``memory_stats``, None on CPU), the wall time, and J and the uniform gradient.
A strategy stops at its first failure (e.g. out of memory), since larger step
counts would fail too.
"""

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

_API = Path(__file__).with_name("tesseract_api.py")


def _inputs(nx: int, steps: int, checkpoints: str | None) -> dict:
    import numpy as np
    from mosaic_shared.problems.thermal_mesh import make_default_inputs

    Lx, Ly, rho_0 = 2.0, 1.0, 0.5
    inputs = make_default_inputs(
        nx=nx,
        ny=max(1, nx // 2),
        nz=1,
        Lx=Lx,
        Ly=Ly,
        Lz=Lx / nx,
        rho_0=rho_0,
        Q_total=0.0,
    )
    k = 1e-3 + (1.0 - 1e-3) * rho_0**3
    t_end = 1.0 / (k * (np.pi / (2 * Lx)) ** 2)
    x = np.asarray(inputs["hex_mesh"]["points"], dtype=np.float64)[:, 0]
    inputs["initial_temperature"] = np.sin(np.pi * x / (2 * Lx)).astype(np.float32)
    inputs["dt"] = np.array([t_end / steps], dtype=np.float32)
    inputs["steps"] = steps
    if checkpoints is not None:
        inputs["adjoint_checkpoints"] = (
            math.isqrt(steps - 1) + 1 if checkpoints == "sqrt" else int(checkpoints)
        )
    return inputs


def _device_stat(key: str) -> int | None:
    import jax

    stats = jax.devices()[0].memory_stats()
    return stats.get(key) if stats else None


def run_case(nx: int, steps: int, checkpoints: str | None, compile_only: bool) -> dict:
    """Run one case in this process and return its record."""
    import jax
    import numpy as np

    spec = importlib.util.spec_from_file_location("jax_heat_api", _API)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)

    inputs = api.InputSchema.model_validate(_inputs(nx, steps, checkpoints))
    res: dict = {"nx": nx, "steps": steps, "checkpoints": checkpoints}
    t0 = time.perf_counter()
    res["xla"] = api.xla_memory(inputs)
    res["compile_s"] = time.perf_counter() - t0
    res["device_limit"] = _device_stat("bytes_limit")
    if compile_only:
        res["ok"] = True
        return res
    J = float(api.apply(inputs).identification_error)
    res["bytes_in_use_before"] = _device_stat("bytes_in_use")
    res["peak_before"] = _device_stat("peak_bytes_in_use")
    for key in ("first_vjp_s", "warm_vjp_s"):  # the first may include a compile
        t0 = time.perf_counter()
        g = api.vector_jacobian_product(
            inputs, {"rho"}, {"identification_error"}, {"identification_error": 2.0 * J}
        )["rho"]
        res[key] = time.perf_counter() - t0
        if key == "first_vjp_s":
            res["peak_after"] = _device_stat("peak_bytes_in_use")
    res["J"] = J
    res["dJ2_uniform"] = float(np.sum(np.asarray(g, dtype=np.float64)))
    res["device"] = jax.devices()[0].device_kind
    res["ok"] = True
    return res


def main() -> None:
    """Run each case in a child process (or, with --case, be that child)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, nargs="+", default=[64])
    ap.add_argument(
        "--steps", type=int, nargs="+", default=[10, 100, 1000, 10000, 100000]
    )
    ap.add_argument("--checkpoints", default=None, help="a chunk count, or sqrt")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=None, help="append one JSON line per case")
    ap.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.case:  # child process: one case, result as the last stdout line
        try:
            res = run_case(a.nx[0], a.steps[0], a.checkpoints, a.compile_only)
        except Exception as e:  # record the failure (e.g. out of memory)
            res = {
                "nx": a.nx[0],
                "steps": a.steps[0],
                "checkpoints": a.checkpoints,
                "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:500]}",
                "traceback": traceback.format_exc()[-2000:],
            }
        print(json.dumps(res))
        return

    for nx in a.nx:
        for steps in a.steps:
            cmd = [
                sys.executable,
                __file__,
                "--case",
                "--nx",
                str(nx),
                "--steps",
                str(steps),
            ]
            if a.checkpoints:
                cmd += ["--checkpoints", a.checkpoints]
            if a.compile_only:
                cmd.append("--compile-only")
            try:
                r = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=a.timeout
                )
                lines = r.stdout.strip().splitlines()
                res = (
                    json.loads(lines[-1])
                    if lines
                    else {
                        "ok": False,
                        "error": f"exit {r.returncode}: {r.stderr[-1000:]}",
                    }
                )
            except subprocess.TimeoutExpired:
                res = {"ok": False, "error": f"timeout after {a.timeout} s"}
            res.setdefault("nx", nx)
            res.setdefault("steps", steps)
            print(json.dumps(res), flush=True)
            if a.out:
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(res) + "\n")
            if not res.get("ok"):
                break


if __name__ == "__main__":
    main()
