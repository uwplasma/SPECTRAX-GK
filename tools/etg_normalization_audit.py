#!/usr/bin/env python3
"""Focused ETG normalization audit against GX reference values."""

from __future__ import annotations

from dataclasses import asdict
from pprint import pformat

import numpy as np

from spectraxgk.benchmarks import (
    ETGBaseCase,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    load_etg_reference,
    run_etg_linear,
)
from spectraxgk.linear_krylov import KrylovConfig


def _rho_e_over_rho_i(mass_ratio: float, te_over_ti: float) -> float:
    return np.sqrt(te_over_ti / mass_ratio)


def main() -> int:
    cfg = ETGBaseCase()
    model = cfg.model
    ref = load_etg_reference()

    kcfg_default = KrylovConfig(
        method="propagator",
        krylov_dim=16,
        restarts=1,
        omega_min_factor=0.0,
        omega_target_factor=0.5,
        omega_cap_factor=0.5,
        omega_sign=-1,
        power_iters=80,
        power_dt=0.002,
    )
    kcfg_low = KrylovConfig(
        method="propagator",
        krylov_dim=16,
        restarts=1,
        omega_min_factor=0.0,
        omega_target_factor=0.0,
        omega_cap_factor=2.0,
        omega_sign=-1,
        power_iters=80,
        power_dt=0.002,
    )

    print("=== ETG normalization audit ===")
    print(
        "GX frequency diagnostic: omega is computed from phi(t)/phi(t-dt) at midplane "
        "(no diamagnetic/frame shift; lab frame, normalized to a/vti)."
    )
    print("Config:")
    print(pformat(asdict(cfg), width=120, sort_dicts=False))
    print(
        f"omega_star_scale={ETG_OMEGA_STAR_SCALE} omega_d_scale={ETG_OMEGA_D_SCALE} rho_star={ETG_RHO_STAR}"
    )
    rho_e = _rho_e_over_rho_i(float(model.mass_ratio), float(model.Te_over_Ti))
    print(f"rho_e/rho_i = {rho_e:.6f} (sqrt(Te/Ti / mass_ratio))")

    for ky in (5.0, 25.0):
        ky_rhoe = ky * rho_e
        ref_idx = int(np.argmin(np.abs(ref.ky - ky)))
        ref_gamma = float(ref.gamma[ref_idx])
        ref_omega = float(ref.omega[ref_idx])
        kcfg = kcfg_low if ky < 10.0 else kcfg_default
        res = run_etg_linear(
            ky_target=ky,
            cfg=cfg,
            Nl=48,
            Nm=16,
            solver="krylov",
            krylov_cfg=kcfg,
            mode_method="z_index",
            fit_signal="phi",
        )
        rel_g = (res.gamma - ref_gamma) / ref_gamma if ref_gamma != 0.0 else np.nan
        rel_w = (res.omega - ref_omega) / ref_omega if ref_omega != 0.0 else np.nan
        print("\n--- ky audit ---")
        print(f"ky(rho_i)={ky:.3f} ky(rho_e)={ky_rhoe:.4f}")
        print(f"ref gamma={ref_gamma:.6f} omega={ref_omega:.6f}")
        print(f"calc gamma={res.gamma:.6f} omega={res.omega:.6f}")
        print(f"rel gamma={rel_g:+.2%} rel omega={rel_w:+.2%}")
        if abs(ky - 25.0) < 1.0e-6:
            eig_ref = ref_gamma - 1j * ref_omega
            eig_calc = res.gamma - 1j * res.omega
            print(f"GX eigenvalue (ref) = {eig_ref.real:+.6f} {eig_ref.imag:+.6f}j")
            print(f"SPECTRAX eigenvalue = {eig_calc.real:+.6f} {eig_calc.imag:+.6f}j")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
