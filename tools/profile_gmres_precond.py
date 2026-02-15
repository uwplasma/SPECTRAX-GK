"""Profile GMRES iteration counts with and without the drift preconditioner."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache, linear_rhs_cached


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


def main() -> int:
    grid = GridConfig(
        Nx=1,
        Ny=4,
        Nz=8,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=16,
        nperiod=1,
    )
    cfg = CycloneBaseCase(grid=grid)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        omega_d_scale=0.2,
        omega_star_scale=0.55,
        rho_star=0.9,
    )
    Nl, Nm = 2, 3
    grid_spec = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    cache = build_linear_cache(grid_spec, geom, params, Nl, Nm)
    shape = (Nl, Nm, grid_spec.ky.size, grid_spec.kx.size, grid_spec.z.size)
    size = int(np.prod(shape))
    dt = 0.02

    @jax.jit
    def matvec_jax(v: jnp.ndarray) -> jnp.ndarray:
        G = v.reshape(shape)
        dG, _ = linear_rhs_cached(G, cache, params, operator="full")
        return (G - dt * dG).reshape(size)

    def matvec_np(v: np.ndarray) -> np.ndarray:
        return np.asarray(matvec_jax(jnp.asarray(v)))

    damping = params.nu * cache.lb_lam + params.nu_hyper * cache.hyper_ratio
    l = jnp.arange(Nl, dtype=jnp.float32)[:, None, None, None, None]
    m = jnp.arange(Nm, dtype=jnp.float32)[None, :, None, None, None]
    diag = -1j * params.tz * params.omega_d_scale * (
        cache.cv_d[None, None, ...] * (2.0 * m + 1.0)
        + cache.gb_d[None, None, ...] * (2.0 * l + 1.0)
    )
    bgrad = cache.bgrad[None, None, None, None, :]
    mirror_diag = params.vth * params.omega_d_scale * (2.0 * l + 1.0) * (2.0 * m + 1.0)
    mirror_weight = 0.2
    diag = diag - mirror_weight * bgrad * mirror_diag
    precond = 1.0 / (1.0 + dt * damping - dt * diag)

    def precond_np(v: np.ndarray) -> np.ndarray:
        v_jax = jnp.asarray(v).reshape(shape)
        return np.asarray((v_jax * precond).reshape(size))

    rng = np.random.default_rng(0)
    b = rng.normal(size=size) + 1j * rng.normal(size=size)

    iters_plain = gmres_iterations(matvec_np, b, precond=None, tol=1.0e-6, maxiter=40)
    iters_prec = gmres_iterations(matvec_np, b, precond=precond_np, tol=1.0e-6, maxiter=40)

    print(f"iters_plain={iters_plain}")
    print(f"iters_precond={iters_prec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
