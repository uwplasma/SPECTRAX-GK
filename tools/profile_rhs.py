"""Profile the linear RHS at XLA level for performance tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.profiler as jprof

from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    _apply_gx_hypercollisions,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, linear_rhs_cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/spectrax_profile_rhs"))
    parser.add_argument("--hlo", type=Path, default=Path("/tmp/spectrax_rhs_hlo.txt"))
    parser.add_argument("--mem", type=Path, default=Path("/tmp/spectrax_rhs_memory.prof"))
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    cfg = CycloneBaseCase()
    grid_cfg = GridConfig(
        Nx=1,
        Ny=8,
        Nz=32,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=16,
        nperiod=1,
    )
    grid = build_spectral_grid(grid_cfg)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    params = _apply_gx_hypercollisions(params)
    terms = LinearTerms()

    Nl, Nm = 4, 8
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    G0 = jnp.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    G0 = G0.at[0, 0, 1, 0, :].set(1e-3 + 0.0j)

    rhs_fn = lambda G: linear_rhs_cached(G, cache, params, terms=terms)[0]
    jit_rhs = jax.jit(rhs_fn)
    jit_rhs(G0).block_until_ready()

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    jprof.start_trace(str(args.trace_dir))
    for _ in range(args.steps):
        jit_rhs(G0).block_until_ready()
    jprof.stop_trace()

    hlo_text = jit_rhs.lower(G0).compiler_ir(dialect="hlo").as_hlo_text()
    args.hlo.write_text(hlo_text)
    jprof.save_device_memory_profile(str(args.mem))

    print(f"Wrote trace to {args.trace_dir}")
    print(f"Wrote HLO to {args.hlo}")
    print(f"Wrote memory profile to {args.mem}")


if __name__ == "__main__":
    main()
