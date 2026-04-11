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


def _estimate_gamma(phi_t: jnp.ndarray, t: jnp.ndarray, start_idx: int) -> jnp.ndarray:
    phi_win = phi_t[start_idx:]
    t_win = t[start_idx:]
    amp = jnp.abs(phi_win) + 1.0e-12
    log_amp = jnp.log(amp)
    t_centered = t_win - jnp.mean(t_win)
    log_centered = log_amp - jnp.mean(log_amp)
    slope = jnp.sum(t_centered * log_centered) / jnp.sum(t_centered**2)
    return slope


def _growth_from_tprim(
    tprim: jnp.ndarray,
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
    fprim: float,
    start_idx: int,
) -> jnp.ndarray:
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
    phi_mode = phi_t[:, ky_index, kx_index, z_index]
    t = jnp.arange(steps) * dt
    return _estimate_gamma(phi_mode, t, start_idx)


def run_demo(
    *,
    outdir: Path,
    steps: int,
    dt: float,
    ky_index: int,
    kx_index: int,
    z_index: int,
    tprim_true: float,
    tprim_init: float,
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
        lambda tprim: _growth_from_tprim(
            tprim,
            grid=grid,
            geom=geom,
            cache=cache,
            G0=G0,
            dt=dt,
            steps=steps,
            ky_index=ky_index,
            kx_index=kx_index,
            z_index=z_index,
            fprim=cfg.model.R_over_Ln,
            start_idx=start_idx,
        )
    )

    target_gamma = float(growth_fn(jnp.asarray(tprim_true)))

    def loss_fn(tprim):
        gamma = growth_fn(tprim)
        return (gamma - target_gamma) ** 2

    grad_fn = jax.grad(loss_fn)

    tprim_hist = []
    loss_hist = []
    tprim = jnp.asarray(tprim_init)
    for _ in range(gd_steps):
        loss_val = loss_fn(tprim)
        grad_val = grad_fn(tprim)
        tprim_hist.append(float(tprim))
        loss_hist.append(float(loss_val))
        tprim = tprim - gd_lr * grad_val

    tprim_hist = np.asarray(tprim_hist)
    loss_hist = np.asarray(loss_hist)

    sweep = np.linspace(1.2, 3.8, 16)
    gamma_sweep = np.asarray(jax.vmap(growth_fn)(jnp.asarray(sweep)))

    tprim_check = 2.2
    eps = 1.0e-3
    gamma_plus = float(growth_fn(jnp.asarray(tprim_check + eps)))
    gamma_minus = float(growth_fn(jnp.asarray(tprim_check - eps)))
    grad_fd = (gamma_plus - gamma_minus) / (2.0 * eps)
    grad_ad = float(jax.grad(growth_fn)(jnp.asarray(tprim_check)))
    grad_rel = float(abs(grad_ad - grad_fd) / (abs(grad_fd) + 1.0e-12))

    summary = {
        "target_gamma": target_gamma,
        "tprim_init": float(tprim_init),
        "tprim_final": float(tprim_hist[-1]) if tprim_hist.size else float(tprim_init),
        "loss_final": float(loss_hist[-1]) if loss_hist.size else float(loss_fn(tprim_init)),
        "grad_autodiff": grad_ad,
        "grad_finite_diff": grad_fd,
        "grad_rel_error": grad_rel,
    }

    if write_files:
        data_path = outdir / "autodiff_inverse_growth.csv"
        np.savetxt(
            data_path,
            np.column_stack([sweep, gamma_sweep]),
            delimiter=",",
            header="R_over_LTi,gamma",
            comments="",
        )
        summary_path = outdir / "autodiff_inverse_growth_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

    if plot:
        set_plot_style()
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

        ax0 = axes[0]
        ax0.plot(sweep, gamma_sweep, marker="o", color="#1f77b4")
        ax0.axhline(target_gamma, color="#d62728", linestyle="--", label="target")
        ax0.set_xlabel(r"$R/L_{Ti}$")
        ax0.set_ylabel(r"$\gamma$")
        ax0.set_title("Sensitivity sweep")
        ax0.legend(loc="best")

        ax1 = axes[1]
        ax1.plot(np.arange(1, len(loss_hist) + 1), loss_hist, marker="o", color="#2ca02c")
        ax1.set_xlabel("Gradient step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Inverse solve")

        ax2 = axes[2]
        ax2.bar(["autodiff", "finite diff"], [grad_ad, grad_fd], color=["#9467bd", "#8c564b"])
        ax2.set_ylabel(r"$d\gamma/d(R/L_{Ti})$")
        ax2.set_title("Gradient check")

        fig.suptitle("Autodiff inverse/sensitivity demo (Cyclone ITG proxy)")
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
    parser.add_argument("--tprim-init", type=float, default=1.6)
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
        tprim_init=args.tprim_init,
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
        plot=True,
        write_files=True,
    )
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
