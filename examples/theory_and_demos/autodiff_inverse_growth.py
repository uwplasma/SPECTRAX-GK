"""Autodiff inverse/sensitivity demo for a linear ITG growth proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, integrate_linear, build_linear_cache
from spectraxgk.plotting import set_plot_style
from spectraxgk.species import Species, build_linear_params


def _estimate_growth(phi_t: jnp.ndarray, t: jnp.ndarray, start_idx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi_win = phi_t[start_idx:]
    t_win = t[start_idx:]
    amp = jnp.abs(phi_win) + 1.0e-12
    log_amp = jnp.log(amp)
    phase = jnp.unwrap(jnp.angle(phi_win))
    t_centered = t_win - jnp.mean(t_win)
    log_centered = log_amp - jnp.mean(log_amp)
    phase_centered = phase - jnp.mean(phase)
    denom = jnp.sum(t_centered**2)
    gamma = jnp.sum(t_centered * log_centered) / denom
    omega = -jnp.sum(t_centered * phase_centered) / denom
    return gamma, omega


def _growth_from_params(
    params_vec: jnp.ndarray,
    *,
    grid,
    geom,
    cache,
    G0,
    dt: float,
    steps: int,
    ky_index: int,
    kx_index: int,
    z_index: int,
    start_idx: int,
) -> jnp.ndarray:
    tprim = params_vec[0]
    fprim = params_vec[1]
    params = LinearParams(
        charge_sign=jnp.asarray([1.0]),
        density=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        temp=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        rho=jnp.asarray([1.0]),
        tz=jnp.asarray([1.0]),
        R_over_LTi=jnp.asarray([tprim]),
        R_over_Ln=jnp.asarray([fprim]),
        tau_e=1.0,
    )
    _, phi_t = integrate_linear(G0, grid, geom, params, dt=dt, steps=steps, cache=cache)
    t = jnp.arange(steps) * dt
    phi_mode = phi_t[:, ky_index, kx_index, z_index]
    gamma, omega = _estimate_growth(phi_mode, t, start_idx)
    return jnp.asarray([gamma, omega])


def run_demo(
    *,
    outdir: Path,
    steps: int,
    dt: float,
    ky_index: int,
    kx_index: int,
    z_index: int,
    tprim_true: float,
    fprim_true: float,
    tprim_init: float,
    fprim_init: float,
    gd_steps: int,
    gd_lr: float,
    plot: bool = True,
    write_files: bool = True,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    grid_cfg = GridConfig(Nx=1, Ny=8, Nz=32, Lx=6.28, Ly=6.28)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    Nl, Nm = 2, 2
    G0 = jnp.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    G0 = G0.at[0, 0, ky_index, kx_index, :].set(1.0e-3 + 0.0j)

    cache = build_linear_cache(
        grid,
        geom,
        build_linear_params([Species(1.0, 1.0, 1.0, 1.0, 2.0, 0.8)], tau_e=1.0),
        Nl,
        Nm,
    )
    start_idx = max(2, steps // 2)

    growth_fn = jax.jit(
        lambda params_vec: _growth_from_params(
            params_vec,
            grid=grid,
            geom=geom,
            cache=cache,
            G0=G0,
            dt=dt,
            steps=steps,
            ky_index=ky_index,
            kx_index=kx_index,
            z_index=z_index,
            start_idx=start_idx,
        )
    )

    target = np.asarray(growth_fn(jnp.asarray([tprim_true, fprim_true])))

    def loss_fn(params_vec):
        gamma = growth_fn(params_vec)
        return jnp.sum((gamma - target) ** 2)

    grad_fn = jax.grad(loss_fn)

    tprim_hist = []
    fprim_hist = []
    loss_hist = []
    params_vec = jnp.asarray([tprim_init, fprim_init])
    for _ in range(gd_steps):
        loss_val = loss_fn(params_vec)
        grad_val = grad_fn(params_vec)
        tprim_hist.append(float(params_vec[0]))
        fprim_hist.append(float(params_vec[1]))
        loss_hist.append(float(loss_val))
        params_vec = params_vec - gd_lr * grad_val

    tprim_hist = np.asarray(tprim_hist)
    fprim_hist = np.asarray(fprim_hist)
    loss_hist = np.asarray(loss_hist)

    sweep_tprim = np.linspace(1.2, 3.8, 16)
    sweep_tprim_vals = np.asarray(
        jax.vmap(lambda val: growth_fn(jnp.asarray([val, fprim_true])))(jnp.asarray(sweep_tprim))
    )
    sweep_fprim = np.linspace(0.4, 1.6, 16)
    sweep_fprim_vals = np.asarray(
        jax.vmap(lambda val: growth_fn(jnp.asarray([tprim_true, val])))(jnp.asarray(sweep_fprim))
    )

    tprim_check = 2.2
    fprim_check = 0.9
    eps = 1.0e-3
    params_center = jnp.asarray([tprim_check, fprim_check])
    jac_ad = np.asarray(jax.jacobian(growth_fn)(params_center))
    jac_fd = np.zeros_like(jac_ad)
    for idx in range(2):
        shift = np.zeros(2, dtype=float)
        shift[idx] = eps
        gamma_plus = np.asarray(growth_fn(jnp.asarray(params_center + shift)))
        gamma_minus = np.asarray(growth_fn(jnp.asarray(params_center - shift)))
        jac_fd[:, idx] = (gamma_plus - gamma_minus) / (2.0 * eps)
    rel_err_cols = np.linalg.norm(jac_ad - jac_fd, axis=0) / (np.linalg.norm(jac_fd, axis=0) + 1.0e-12)

    obs_final = np.asarray(growth_fn(params_vec))
    residual = obs_final - target
    sigma2 = float(np.mean(residual**2) + 1.0e-12)
    jtj = jac_ad.T @ jac_ad + 1.0e-9 * np.eye(2)
    cov = sigma2 * np.linalg.inv(jtj)

    summary = {
        "target_observables": target.tolist(),
        "tprim_init": float(tprim_init),
        "fprim_init": float(fprim_init),
        "tprim_final": float(tprim_hist[-1]) if tprim_hist.size else float(tprim_init),
        "fprim_final": float(fprim_hist[-1]) if fprim_hist.size else float(fprim_init),
        "loss_final": float(loss_hist[-1])
        if loss_hist.size
        else float(loss_fn(jnp.asarray([tprim_init, fprim_init]))),
        "jac_autodiff": jac_ad.tolist(),
        "jac_finite_diff": jac_fd.tolist(),
        "jac_rel_error": rel_err_cols.tolist(),
        "covariance": cov.tolist(),
        "sigma2": sigma2,
    }

    if write_files:
        data_path = outdir / "autodiff_inverse_growth_tprim_sweep.csv"
        np.savetxt(
            data_path,
            np.column_stack([sweep_tprim, sweep_tprim_vals]),
            delimiter=",",
            header="R_over_LTi,gamma,omega",
            comments="",
        )
        data_path_f = outdir / "autodiff_inverse_growth_fprim_sweep.csv"
        np.savetxt(
            data_path_f,
            np.column_stack([sweep_fprim, sweep_fprim_vals]),
            delimiter=",",
            header="R_over_Ln,gamma,omega",
            comments="",
        )
        summary_path = outdir / "autodiff_inverse_growth_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

    if plot:
        set_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))

        ax0 = axes[0, 0]
        ax0.plot(sweep_tprim, sweep_tprim_vals[:, 0], marker="o", color="#1f77b4", label=r"$\gamma$")
        ax0.plot(sweep_tprim, sweep_tprim_vals[:, 1], marker="s", color="#ff7f0e", label=r"$\omega$")
        ax0.axhline(target[0], color="#1f77b4", linestyle="--", alpha=0.6)
        ax0.axhline(target[1], color="#ff7f0e", linestyle="--", alpha=0.6)
        ax0.set_xlabel(r"$R/L_{Ti}$")
        ax0.set_ylabel(r"$\gamma$")
        ax0.set_title("Sensitivity vs $R/L_{Ti}$")
        ax0.legend(loc="best")

        ax1 = axes[0, 1]
        ax1.plot(sweep_fprim, sweep_fprim_vals[:, 0], marker="o", color="#1f77b4", label=r"$\gamma$")
        ax1.plot(sweep_fprim, sweep_fprim_vals[:, 1], marker="s", color="#ff7f0e", label=r"$\omega$")
        ax1.axhline(target[0], color="#1f77b4", linestyle="--", alpha=0.6)
        ax1.axhline(target[1], color="#ff7f0e", linestyle="--", alpha=0.6)
        ax1.set_xlabel(r"$R/L_{n}$")
        ax1.set_ylabel(r"$\gamma$")
        ax1.set_title("Sensitivity vs $R/L_{n}$")
        ax1.legend(loc="best")

        ax2 = axes[1, 0]
        ax2.plot(tprim_hist, fprim_hist, marker="o", color="#2ca02c")
        ax2.scatter([tprim_true], [fprim_true], color="#d62728", marker="x", s=60, label="target")
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2.0 * np.sqrt(np.maximum(vals, 0.0))
        from matplotlib.patches import Ellipse
        ellipse = Ellipse((summary["tprim_final"], summary["fprim_final"]), width, height, angle=angle, fill=False, color="#9467bd")
        ax2.add_patch(ellipse)
        ax2.set_xlabel(r"$R/L_{Ti}$")
        ax2.set_ylabel(r"$R/L_{n}$")
        ax2.set_title("Inverse solve + 1σ ellipse")
        ax2.legend(loc="best")

        ax3 = axes[1, 1]
        ax3.bar(["$R/L_{Ti}$", "$R/L_{n}$"], rel_err_cols, color=["#9467bd", "#8c564b"])
        ax3.set_ylabel("Jacobian rel. error")
        ax3.set_title("Autodiff vs finite diff")

        fig.suptitle("Autodiff inverse/sensitivity demo (two-parameter ITG proxy)")
        fig.tight_layout()

        fig_path = outdir / "autodiff_inverse_growth.png"
        fig.savefig(fig_path, dpi=200)
        fig.savefig(outdir / "autodiff_inverse_growth.pdf")
        print(f"Wrote {fig_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Autodiff inverse growth demo.")
    parser.add_argument("--outdir", type=Path, default=Path("docs/_static"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--ky-index", type=int, default=1)
    parser.add_argument("--kx-index", type=int, default=0)
    parser.add_argument("--z-index", type=int, default=0)
    parser.add_argument("--tprim-true", type=float, default=2.8)
    parser.add_argument("--fprim-true", type=float, default=0.8)
    parser.add_argument("--tprim-init", type=float, default=1.6)
    parser.add_argument("--fprim-init", type=float, default=1.1)
    parser.add_argument("--gd-steps", type=int, default=18)
    parser.add_argument("--gd-lr", type=float, default=0.7)
    args = parser.parse_args()

    summary = run_demo(
        outdir=args.outdir,
        steps=args.steps,
        dt=args.dt,
        ky_index=args.ky_index,
        kx_index=args.kx_index,
        z_index=args.z_index,
        tprim_true=args.tprim_true,
        fprim_true=args.fprim_true,
        tprim_init=args.tprim_init,
        fprim_init=args.fprim_init,
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
        plot=True,
        write_files=True,
    )
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
