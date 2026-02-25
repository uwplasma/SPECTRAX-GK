#!/usr/bin/env python3
"""Focused ETG/KBM normalization audit against GS2 reference values."""

from __future__ import annotations

from dataclasses import asdict
from pprint import pformat

import numpy as np

from spectraxgk.benchmarks import (
    ETGBaseCase,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    KBMBaseCase,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    load_etg_reference_gs2,
    load_kbm_reference_gs2,
    run_etg_linear,
    run_kbm_beta_scan,
)
from spectraxgk.linear_krylov import KrylovConfig


def _rho_e_over_rho_i(mass_ratio: float, te_over_ti: float) -> float:
    return np.sqrt(te_over_ti / mass_ratio)


def main() -> int:
    cfg = ETGBaseCase()
    model = cfg.model
    ref = load_etg_reference_gs2()

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

    print("=== ETG normalization audit (GS2 reference) ===", flush=True)
    print(
        "Frequency diagnostic uses SPECTRAX growth extraction in the same reporting "
        "normalization used for GS2 comparison tables."
    , flush=True)
    print("Config:", flush=True)
    print(pformat(asdict(cfg), width=120, sort_dicts=False), flush=True)
    print(
        f"omega_star_scale={ETG_OMEGA_STAR_SCALE} omega_d_scale={ETG_OMEGA_D_SCALE} rho_star={ETG_RHO_STAR}"
    , flush=True)
    rho_e = _rho_e_over_rho_i(float(model.mass_ratio), float(model.Te_over_Ti))
    print(f"rho_e/rho_i = {rho_e:.6f} (sqrt(Te/Ti / mass_ratio))", flush=True)

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
        print("\n--- ky audit ---", flush=True)
        print(f"ky(rho_i)={ky:.3f} ky(rho_e)={ky_rhoe:.4f}", flush=True)
        print(f"ref gamma={ref_gamma:.6f} omega={ref_omega:.6f}", flush=True)
        print(f"calc gamma={res.gamma:.6f} omega={res.omega:.6f}", flush=True)
        print(f"rel gamma={rel_g:+.2%} rel omega={rel_w:+.2%}", flush=True)
        if abs(ky - 25.0) < 1.0e-6:
            eig_ref = ref_gamma - 1j * ref_omega
            eig_calc = res.gamma - 1j * res.omega
            print(f"GS2 eigenvalue (ref) = {eig_ref.real:+.6f} {eig_ref.imag:+.6f}j", flush=True)
            print(f"SPECTRAX eigenvalue = {eig_calc.real:+.6f} {eig_calc.imag:+.6f}j", flush=True)

    print("\n=== KBM normalization audit (GS2 reference) ===", flush=True)
    kbm_ref = load_kbm_reference_gs2()
    cfg_kbm = KBMBaseCase()
    betas = np.asarray([0.2, 0.3, 0.4], dtype=float)
    scan = run_kbm_beta_scan(
        betas,
        cfg=cfg_kbm,
        ky_target=0.3,
        Nl=24,
        Nm=12,
        dt=5.0e-4,
        steps=3000,
        method="imex2",
        solver="time",
        fit_signal="phi",
        mode_method="z_index",
        auto_window=True,
    )
    for beta, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(kbm_ref.ky - beta)))
        ref_gamma = float(kbm_ref.gamma[idx])
        ref_omega = float(kbm_ref.omega[idx])
        rel_g = (float(gamma) - ref_gamma) / ref_gamma if ref_gamma != 0.0 else np.nan
        rel_w = (float(omega) - ref_omega) / ref_omega if ref_omega != 0.0 else np.nan
        print(
            f"beta={float(beta):.3f} gamma={float(gamma):.6f} ref={ref_gamma:.6f} rel={rel_g:+.2%} | "
            f"omega={float(omega):.6f} ref={ref_omega:.6f} rel={rel_w:+.2%}"
        , flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
