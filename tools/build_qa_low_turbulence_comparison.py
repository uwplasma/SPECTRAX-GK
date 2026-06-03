#!/usr/bin/env python3
"""Build the aspect-6 QA low-turbulence optimization comparison panel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.qa_low_turbulence import (  # noqa: E402
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
    qa_low_turbulence_comparison_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "qa_low_turbulence_comparison.png"
COLORS = {
    "qa_constraints": "#244c66",
    "qa_plus_nonlinear_heat_flux": "#b45f2a",
}
LABELS = {
    "qa_constraints": "Reduced QA constraints",
    "qa_plus_nonlinear_heat_flux": "Reduced QA + NL envelope",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="PNG output path")
    parser.add_argument("--pdf", action="store_true", help="also write a PDF companion")
    parser.add_argument("--workers", type=int, default=1, help="finite-difference worker count")
    parser.add_argument("--steps", type=int, default=60, help="Adam steps for each reduced optimization")
    parser.add_argument("--nonlinear-dt", type=float, default=0.20, help="RK2 time step in the reduced nonlinear envelope")
    parser.add_argument("--nonlinear-steps", type=int, default=2000, help="RK2 steps in the reduced nonlinear envelope")
    parser.add_argument("--nonlinear-weight", type=float, default=8.0, help="transport residual weight")
    parser.add_argument("--iota-operating-floor", type=float, default=0.70, help="operating iota floor above the formal minimum")
    return parser.parse_args()


def _artifact_base(path: Path) -> Path:
    return path.with_suffix("") if path.suffix else path


def _obs_map(result: dict[str, Any]) -> dict[str, float]:
    return dict(
        zip(
            QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
            map(float, result["final_observables"]),
            strict=True,
        )
    )


def _set_equal_3d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    xy_limits = np.array(
        [
            [np.nanmin(x), np.nanmax(x)],
            [np.nanmin(y), np.nanmax(y)],
        ],
        dtype=float,
    )
    xy_center = np.mean(xy_limits, axis=1)
    xy_radius = 0.52 * float(np.max(xy_limits[:, 1] - xy_limits[:, 0]))
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    z_center = 0.5 * (z_min + z_max)
    z_radius = max(0.06, 0.62 * (z_max - z_min))
    ax.set_xlim(xy_center[0] - xy_radius, xy_center[0] + xy_radius)
    ax.set_ylim(xy_center[1] - xy_radius, xy_center[1] + xy_radius)
    ax.set_zlim(z_center - z_radius, z_center + z_radius)
    ax.set_box_aspect((1.0, 1.0, 0.45))


def _write_scan_csv(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for design in payload["designs"]:
        scan = design["density_gradient_scan"]
        for aln, q, cv, trend, gamma in zip(
            scan["density_gradient_axis"],
            scan["heat_flux_mean"],
            scan["heat_flux_cv"],
            scan["heat_flux_trend"],
            scan["growth_rate"],
            strict=True,
        ):
            rows.append(
                {
                    "design_name": design["design_name"],
                    "a_over_Ln": aln,
                    "a_over_LTi": scan["fixed_temperature_gradient"],
                    "heat_flux_mean": q,
                    "heat_flux_cv": cv,
                    "heat_flux_trend": trend,
                    "growth_rate": gamma,
                    "linear_slope_dQ_d_a_over_Ln": scan["linear_slope_dQ_d_a_over_Ln"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for result in payload["results"]:
        obs = _obs_map(result)
        rows.append(
            {
                "design_name": result["design_name"],
                "includes_nonlinear_heat_flux": result["includes_nonlinear_heat_flux"],
                "aspect": obs["aspect"],
                "mean_iota": obs["mean_iota"],
                "iota_operating_floor_violation": obs["iota_operating_floor_violation"],
                "qa_residual": obs["qa_residual"],
                "helical_ripple_amplitude": result["final_params"][2],
                "growth_rate": obs["growth_rate"],
                "quasilinear_heat_flux": obs["quasilinear_heat_flux"],
                "nonlinear_heat_flux_mean": obs["nonlinear_heat_flux_mean"],
                "scalar_gradient_gate_passed": result["scalar_gradient_gate"]["passed"],
                "residual_gradient_gate_passed": result["residual_gradient_gate"]["passed"],
                "observable_gradient_gate_passed": result["observable_gradient_gate"]["passed"],
                "final_objective": result["final_objective"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_payload(payload: dict[str, Any], out: Path) -> None:
    """Render the publication-style comparison panel."""

    set_plot_style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig = plt.figure(figsize=(17.2, 13.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, hspace=0.12, wspace=0.08)
    ax_scan = fig.add_subplot(grid[0, 0])
    ax_trace = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[0, 2])
    ax_surf0 = fig.add_subplot(grid[1, 0], projection="3d")
    ax_surf1 = fig.add_subplot(grid[1, 1], projection="3d")
    ax_final = fig.add_subplot(grid[1, 2])
    ax_b0 = fig.add_subplot(grid[2, 0], projection="3d")
    ax_b1 = fig.add_subplot(grid[2, 1], projection="3d")
    ax_window = fig.add_subplot(grid[2, 2])

    b_values = [np.asarray(design["lcfs_bmag"]["bmag"], dtype=float) for design in payload["designs"]]
    bmin = min(float(np.nanmin(item)) for item in b_values)
    bmax = max(float(np.nanmax(item)) for item in b_values)
    bnorm = colors.Normalize(vmin=bmin, vmax=bmax)
    bcmap = plt.colormaps["jet"]

    for design in payload["designs"]:
        name = design["design_name"]
        color = COLORS[name]
        label = LABELS[name]
        scan = design["density_gradient_scan"]
        x = np.asarray(scan["density_gradient_axis"], dtype=float)
        y = np.asarray(scan["heat_flux_mean"], dtype=float)
        ax_scan.plot(x, y, marker="o", lw=2.3, color=color, label=label)
        slope = float(scan["linear_slope_dQ_d_a_over_Ln"])
        ax_scan.text(x[-1], y[-1], f"  slope={slope:.2e}", color=color, va="center", fontsize=8)

        trace = design["fixed_gradient_trace"]
        time = np.asarray(trace["times"], dtype=float)
        q = np.asarray(trace["heat_flux"], dtype=float)
        start = int(trace["window"]["start_index"])
        running = np.cumsum(q[start:]) / np.arange(1, q[start:].size + 1, dtype=float)
        ax_trace.plot(time, q, lw=2.1, color=color, label=label)
        ax_trace.plot(time[start:], running, lw=1.4, color=color, ls=":", alpha=0.85)
        ax_trace.axhline(float(trace["window"]["mean"]), color=color, ls="--", lw=1.1, alpha=0.75)
        ax_trace.axvspan(time[start], time[-1], color=color, alpha=0.08)

    ax_scan.set_xlabel(r"density gradient $a/L_n$")
    ax_scan.set_ylabel(r"late-window $\langle Q_{\rm env}\rangle$")
    ax_scan.set_title(r"Gradient scan at fixed $a/L_{Ti}=6$")
    ax_scan.legend(frameon=False, fontsize=8)
    ax_scan.grid(alpha=0.25)

    ax_trace.set_xlabel(r"$t v_{ti}/a$")
    ax_trace.set_ylabel(r"$Q_{\rm env}(t)$")
    ax_trace.set_title(
        rf"Reduced envelope: $a/L_n={payload['fixed_density_gradient']:.1f}$, "
        rf"$a/L_{{Ti}}={payload['fixed_temperature_gradient']:.1f}$"
    )
    ax_trace.legend(frameon=False, fontsize=8)
    ax_trace.grid(alpha=0.25)

    for result in payload["results"]:
        name = result["design_name"]
        hist = np.asarray([row["objective"] for row in result["history"]], dtype=float)
        steps = np.asarray([row["step"] for row in result["history"]], dtype=float)
        ax_hist.semilogy(steps, hist, lw=2.2, color=COLORS[name], label=LABELS[name])
    ax_hist.set_xlabel("Adam step")
    ax_hist.set_ylabel(r"$||r||^2$")
    ax_hist.set_title("Constrained objectives")
    ax_hist.legend(frameon=False, fontsize=8)
    ax_hist.grid(alpha=0.25)

    surface_axes = [ax_surf0, ax_surf1]
    b_axes = [ax_b0, ax_b1]
    for ax, design in zip(surface_axes, payload["designs"], strict=True):
        name = design["design_name"]
        surface = design["surface"]
        x = np.asarray(surface["x"], dtype=float)
        y = np.asarray(surface["y"], dtype=float)
        z = np.asarray(surface["z"], dtype=float)
        bmag = np.asarray(design["lcfs_bmag"]["bmag"], dtype=float)
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=bcmap(bnorm(bmag)),
            alpha=0.95,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        ax.plot_wireframe(
            x,
            y,
            z,
            rstride=max(1, x.shape[0] // 10),
            cstride=max(1, x.shape[1] // 10),
            color="white",
            linewidth=0.28,
            alpha=0.42,
        )
        _set_equal_3d(ax, x, y, z)
        ax.view_init(elev=26, azim=48)
        ax.set_title(rf"Reduced LCFS $|B|$: {LABELS[name]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(False)
        mappable = plt.cm.ScalarMappable(norm=bnorm, cmap=bcmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, fraction=0.040, pad=0.02, label=r"$|B|/B_0$")

    for ax, design in zip(b_axes, payload["designs"], strict=True):
        name = design["design_name"]
        bmap = design["lcfs_bmag"]
        theta = np.asarray(bmap["theta"], dtype=float) / np.pi
        zeta = np.asarray(bmap["zeta"], dtype=float) / np.pi
        zz, tt = np.meshgrid(zeta, theta, indexing="xy")
        bmag = np.asarray(bmap["bmag"], dtype=float)
        zpad = 0.08 * max(bmax - bmin, 1.0e-6)
        ax.plot_surface(zz, tt, bmag, cmap=bcmap, norm=bnorm, linewidth=0, antialiased=True, alpha=0.96)
        ax.contourf(
            zz,
            tt,
            bmag,
            zdir="z",
            offset=bmin - zpad,
            levels=18,
            cmap=bcmap,
            norm=bnorm,
            alpha=0.78,
        )
        ax.contour(zz, tt, bmag, levels=9, colors="white", linewidths=0.35, alpha=0.72)
        ax.set_xlabel(r"$\phi_B/\pi$")
        ax.set_ylabel(r"$\theta_B/\pi$")
        ax.set_zlabel("")
        ax.set_zlim(bmin - zpad, bmax + zpad)
        ax.view_init(elev=30, azim=-52)
        ax.set_title(rf"Boozer LCFS $|B|$: {LABELS[name]}")
        mappable = plt.cm.ScalarMappable(norm=bnorm, cmap=bcmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, fraction=0.040, pad=0.02, label=r"$|B|/B_0$")

    result_by_name = {result["design_name"]: result for result in payload["results"]}
    metric_names = (r"$A/A_0$", r"$\iota$", r"$10^4 R_{QA}$", r"$10^2\langle Q_{\rm env}\rangle$")
    x = np.arange(len(metric_names), dtype=float)
    width = 0.36
    for offset, design in zip((-0.5 * width, 0.5 * width), payload["designs"], strict=True):
        name = design["design_name"]
        obs = _obs_map(result_by_name[name])
        values = [
            obs["aspect"] / payload["target_aspect"],
            obs["mean_iota"],
            1.0e4 * obs["qa_residual"],
            1.0e2 * obs["nonlinear_heat_flux_mean"],
        ]
        ax_final.bar(x + offset, values, width=width, color=COLORS[name], label=LABELS[name])
    ax_final.axhline(1.0, color="black", ls=":", lw=1.0, alpha=0.5)
    ax_final.axhline(payload["operating_iota_floor"], color="#7c2d12", ls=":", lw=1.0, alpha=0.55)
    ax_final.set_xticks(x)
    ax_final.set_xticklabels(metric_names, rotation=18, ha="right")
    ax_final.set_title("Final reduced observables")
    ax_final.grid(axis="y", alpha=0.25)
    ax_final.legend(frameon=False, fontsize=8)

    conv_names = ("CV", "trend", "half-window drift")
    x = np.arange(len(conv_names), dtype=float)
    for offset, design in zip((-0.5 * width, 0.5 * width), payload["designs"], strict=True):
        name = design["design_name"]
        trace = design["fixed_gradient_trace"]
        window = trace["window"]
        gate = trace["long_window_convergence"]
        values = [
            float(window["cv"]),
            float(window["trend"]),
            float(gate["half_window_relative_mean_change"]),
        ]
        ax_window.bar(x + offset, values, width=width, color=COLORS[name], label=LABELS[name])
    ax_window.axhline(payload["config"]["long_window_max_cv"], color="black", ls=":", lw=1.0, alpha=0.45)
    ax_window.set_yscale("log")
    ax_window.set_xticks(x)
    ax_window.set_xticklabels(conv_names, rotation=18, ha="right")
    ax_window.set_ylabel("relative metric")
    ax_window.set_title("Reduced-envelope late-window convergence")
    ax_window.grid(axis="y", alpha=0.25)
    ax_window.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Aspect-6 reduced QA low-turbulence optimizer: constraints-only vs transport-aware design",
        fontsize=15,
        fontweight="bold",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=185, bbox_inches="tight")
    plt.close(fig)


def write_artifacts(payload: dict[str, Any], out: Path, *, write_pdf: bool = True) -> dict[str, str]:
    base = _artifact_base(out)
    paths = {
        "png": str(base.with_suffix(".png")),
        "json": str(base.with_suffix(".json")),
        "summary_csv": str(base.with_suffix(".summary.csv")),
        "scan_csv": str(base.with_suffix(".scan.csv")),
    }
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    _write_summary_csv(payload, base.with_suffix(".summary.csv"))
    _write_scan_csv(payload, base.with_suffix(".scan.csv"))
    plot_payload(payload, base.with_suffix(".png"))
    if write_pdf:
        plot_payload(payload, base.with_suffix(".pdf"))
        paths["pdf"] = str(base.with_suffix(".pdf"))
    return paths


def main() -> int:
    args = _parse_args()
    cfg = QALowTurbulenceConfig(
        steps=int(args.steps),
        nonlinear_dt=float(args.nonlinear_dt),
        nonlinear_steps=int(args.nonlinear_steps),
        nonlinear_weight=float(args.nonlinear_weight),
        iota_operating_floor=float(args.iota_operating_floor),
    )
    payload = qa_low_turbulence_comparison_payload(cfg, finite_difference_workers=int(args.workers))
    paths = write_artifacts(payload, args.out, write_pdf=bool(args.pdf))
    print(json.dumps({"passed": payload["comparison_metrics"]["passed"], "paths": paths}, indent=2))
    return 0 if payload["comparison_metrics"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
