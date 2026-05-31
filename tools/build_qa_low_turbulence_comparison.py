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
    "qa_constraints": "QA constraints",
    "qa_plus_nonlinear_heat_flux": "QA + reduced NL Q",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="PNG output path")
    parser.add_argument("--pdf", action="store_true", help="also write a PDF companion")
    parser.add_argument("--workers", type=int, default=1, help="finite-difference worker count")
    parser.add_argument("--steps", type=int, default=40, help="Adam steps for each reduced optimization")
    parser.add_argument("--nonlinear-steps", type=int, default=540, help="RK2 steps in the reduced nonlinear envelope")
    parser.add_argument("--nonlinear-weight", type=float, default=8.0, help="transport residual weight")
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
    limits = np.array(
        [
            [np.nanmin(x), np.nanmax(x)],
            [np.nanmin(y), np.nanmax(y)],
            [np.nanmin(z), np.nanmax(z)],
        ],
        dtype=float,
    )
    center = np.mean(limits, axis=1)
    radius = 0.52 * float(np.max(limits[:, 1] - limits[:, 0]))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


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
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
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
                "qa_residual": obs["qa_residual"],
                "growth_rate": obs["growth_rate"],
                "quasilinear_heat_flux": obs["quasilinear_heat_flux"],
                "nonlinear_heat_flux_mean": obs["nonlinear_heat_flux_mean"],
                "scalar_gradient_gate_passed": result["scalar_gradient_gate"]["passed"],
                "residual_gradient_gate_passed": result["residual_gradient_gate"]["passed"],
                "final_objective": result["final_objective"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_payload(payload: dict[str, Any], out: Path) -> None:
    """Render the publication-style comparison panel."""

    set_plot_style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig = plt.figure(figsize=(16.5, 13.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, hspace=0.12, wspace=0.08)
    ax_scan = fig.add_subplot(grid[0, 0])
    ax_trace = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[0, 2])
    ax_surf0 = fig.add_subplot(grid[1, 0], projection="3d")
    ax_surf1 = fig.add_subplot(grid[1, 1], projection="3d")
    ax_metrics = fig.add_subplot(grid[1, 2])
    ax_b0 = fig.add_subplot(grid[2, 0])
    ax_b1 = fig.add_subplot(grid[2, 1])
    ax_scope = fig.add_subplot(grid[2, 2])

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
        ax_trace.plot(time, q, lw=2.1, color=color, label=label)
        ax_trace.axhline(float(trace["window"]["mean"]), color=color, ls="--", lw=1.1, alpha=0.75)
        ax_trace.axvspan(time[start], time[-1], color=color, alpha=0.08)

    ax_scan.set_xlabel(r"density gradient $a/L_n$")
    ax_scan.set_ylabel(r"late-window $\langle Q_i \rangle$ (reduced NL)")
    ax_scan.set_title(r"Gradient scan at fixed $a/L_{Ti}=6$")
    ax_scan.legend(frameon=False, fontsize=8)
    ax_scan.grid(alpha=0.25)

    ax_trace.set_xlabel(r"$t v_{ti}/a$")
    ax_trace.set_ylabel(r"$Q_i(t)$")
    ax_trace.set_title(
        rf"Fixed-gradient trace: $a/L_n={payload['fixed_density_gradient']:.1f}$, "
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
        ax.plot_surface(x, y, z, color=COLORS[name], alpha=0.85, linewidth=0, antialiased=True)
        _set_equal_3d(ax, x, y, z)
        ax.view_init(elev=22, azim=35)
        ax.set_title(f"Reduced LCFS: {LABELS[name]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(False)

    for ax, design in zip(b_axes, payload["designs"], strict=True):
        name = design["design_name"]
        bmap = design["lcfs_bmag"]
        theta = np.asarray(bmap["theta"], dtype=float) / np.pi
        zeta = np.asarray(bmap["zeta"], dtype=float) / np.pi
        bmag = np.asarray(bmap["bmag"], dtype=float)
        c = ax.contourf(zeta, theta, bmag, levels=18, cmap="cividis")
        ax.contour(zeta, theta, bmag, levels=8, colors="white", linewidths=0.35, alpha=0.7)
        ax.set_xlabel(r"$\zeta/\pi$")
        ax.set_ylabel(r"$\theta/\pi$")
        ax.set_title(rf"LCFS $|B|$: {LABELS[name]}")
        fig.colorbar(c, ax=ax, fraction=0.047, pad=0.02, label=r"$|B|/B_0$")

    ax_metrics.axis("off")
    metrics = payload["comparison_metrics"]
    lines = [
        "Gate summary",
        f"Aspect target: {payload['target_aspect']:.1f}",
        f"Minimum iota: {payload['minimum_iota']:.2f}",
        f"AD/FD gates passed: {metrics['ad_fd_gates_passed']}",
        f"Constraint gates passed: {metrics['constraints_passed']}",
        f"Fixed-gradient Q reduction: {100.0 * metrics['relative_heat_flux_reduction_at_fixed_gradients']:.1f}%",
        "",
        "Final values",
    ]
    for result in payload["results"]:
        obs = _obs_map(result)
        lines.extend(
            [
                LABELS[result["design_name"]],
                f"  A={obs['aspect']:.3f}, iota={obs['mean_iota']:.3f}",
                f"  QA residual={obs['qa_residual']:.2e}",
                f"  <Q_i>={obs['nonlinear_heat_flux_mean']:.3e}",
            ]
        )
    ax_metrics.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    ax_scope.axis("off")
    scope = [
        "Scope and method",
        "- Reduced max-mode-1 QA controls",
        "- Aspect-ratio and iota-floor constraints",
        "- Differentiable RK2 nonlinear ITG envelope",
        "- Central finite-difference gradient gates",
        "- Production nonlinear claims still need long-window GK audits",
        "",
        "Equations used in this panel",
        r"$J=||r_{A}, r_{\iota}, r_{QA}, r_{reg}, r_Q||^2$",
        r"$dE/dt=2\gamma E-\alpha E^2$",
        r"$Q_i=W_iE$",
    ]
    ax_scope.text(
        0.02,
        0.98,
        "\n".join(scope),
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff7ed", "edgecolor": "#fed7aa"},
    )

    fig.suptitle(
        "Aspect-6 QA low-turbulence optimization: constraints-only vs transport-aware design",
        fontsize=15,
        fontweight="bold",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
        nonlinear_steps=int(args.nonlinear_steps),
        nonlinear_weight=float(args.nonlinear_weight),
    )
    payload = qa_low_turbulence_comparison_payload(cfg, finite_difference_workers=int(args.workers))
    paths = write_artifacts(payload, args.out, write_pdf=bool(args.pdf))
    print(json.dumps({"passed": payload["comparison_metrics"]["passed"], "paths": paths}, indent=2))
    return 0 if payload["comparison_metrics"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
