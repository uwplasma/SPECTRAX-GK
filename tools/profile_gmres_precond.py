"""Profile GMRES iteration counts with and without the drift preconditioner."""

from __future__ import annotations

import argparse
import numpy as np

import jax
import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    linear_rhs_cached,
    _build_implicit_operator,
    _x64_enabled,
)


def gmres_iterations(matvec, b: np.ndarray, precond=None, tol: float = 1.0e-6, maxiter: int = 40) -> int:
    """Basic GMRES iteration count for a left-preconditioned system."""

    def apply_precond(v):
        return v if precond is None else precond(v)

    r0 = apply_precond(b)
    beta = np.linalg.norm(r0)
    if beta == 0.0:
        return 0
    n = b.size
    V = np.zeros((n, maxiter + 1), dtype=np.complex128)
    H = np.zeros((maxiter + 1, maxiter), dtype=np.complex128)
    V[:, 0] = r0 / beta
    g = np.zeros(maxiter + 1, dtype=np.complex128)
    g[0] = beta
    for k in range(maxiter):
        w = apply_precond(matvec(V[:, k]))
        for j in range(k + 1):
            H[j, k] = np.vdot(V[:, j], w)
            w = w - H[j, k] * V[:, j]
        H[k + 1, k] = np.linalg.norm(w)
        if H[k + 1, k] != 0.0 and k + 1 < n:
            V[:, k + 1] = w / H[k + 1, k]
        y, *_ = np.linalg.lstsq(H[: k + 2, : k + 1], g[: k + 2], rcond=None)
        res = np.linalg.norm(g[: k + 2] - H[: k + 2, : k + 1] @ y)
        if res < tol:
            return k + 1
    return maxiter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile GMRES preconditioners.")
    parser.add_argument("--Nx", type=int, default=1)
    parser.add_argument("--Ny", type=int, default=4)
    parser.add_argument("--Nz", type=int, default=8)
    parser.add_argument("--Lx", type=float, default=62.8)
    parser.add_argument("--Ly", type=float, default=62.8)
    parser.add_argument("--y0", type=float, default=20.0)
    parser.add_argument("--ntheta", type=int, default=16)
    parser.add_argument("--nperiod", type=int, default=1)
    parser.add_argument("--boundary", choices=["periodic", "linked"], default="periodic")
    parser.add_argument("--Nl", type=int, default=2)
    parser.add_argument("--Nm", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--maxiter", type=int, default=40)
    return parser.parse_args()


def _precond_from_operator(
    G0: jnp.ndarray,
    cache,
    params,
    dt: float,
    terms: LinearTerms,
    precond_key: str | None,
) -> callable | None:
    _, _shape, _size, _dt_val, precond_op, _matvec, _squeeze = _build_implicit_operator(
        G0, cache, params, dt, terms, precond_key
    )
    if precond_op is None:
        return None

    def apply(v: np.ndarray) -> np.ndarray:
        return np.asarray(precond_op(jnp.asarray(v)))

    return apply


def main() -> int:
    args = _parse_args()
    grid = GridConfig(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        Lx=args.Lx,
        Ly=args.Ly,
        boundary=args.boundary,
        y0=args.y0,
        ntheta=args.ntheta,
        nperiod=args.nperiod,
    )
    cfg = CycloneBaseCase(grid=grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        omega_d_scale=0.2,
        omega_star_scale=0.55,
        rho_star=0.9,
        kpar_scale=float(geom.gradpar()),
    )
    Nl, Nm = int(args.Nl), int(args.Nm)
    grid_spec = build_spectral_grid(cfg.grid)
    cache = build_linear_cache(grid_spec, geom, params, Nl, Nm)
    shape = (Nl, Nm, grid_spec.ky.size, grid_spec.kx.size, grid_spec.z.size)
    size = int(np.prod(shape))
    dt = float(args.dt)
    terms = LinearTerms()
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    G0 = jnp.zeros(shape, dtype=base_dtype)

    @jax.jit
    def matvec_jax(v: jnp.ndarray) -> jnp.ndarray:
        G = v.reshape(shape)
        dG, _ = linear_rhs_cached(G, cache, params, terms=terms)
        return (G - dt * dG).reshape(size)

    def matvec_np(v: np.ndarray) -> np.ndarray:
        return np.asarray(matvec_jax(jnp.asarray(v)))

    rng = np.random.default_rng(0)
    b = rng.normal(size=size) + 1j * rng.normal(size=size)
    precond_keys = {
        "plain": None,
        "diag": "diag",
        "pas": "pas",
        "pas_coarse": "pas-coarse",
        "hermite_line": "hermite-line",
        "hermite_line_coarse": "hermite-line-coarse",
    }

    for label, key in precond_keys.items():
        precond = _precond_from_operator(G0, cache, params, dt, terms, key)
        iters = gmres_iterations(
            matvec_np,
            b,
            precond=precond,
            tol=float(args.tol),
            maxiter=int(args.maxiter),
        )
        print(f"iters_{label}={iters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
