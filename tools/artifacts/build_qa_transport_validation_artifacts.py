#!/usr/bin/env python3
"""Build reduced QA transport comparison and time-horizon validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.objectives.qa_low_turbulence_artifacts import (  # noqa: E402
    qa_low_turbulence_comparison_payload,
)
from spectraxgk.objectives.qa_low_turbulence_contracts import (  # noqa: E402
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
)
from spectraxgk.objectives.qa_low_turbulence_model import (  # noqa: E402
    qa_low_turbulence_heat_flux_trace,
    qa_low_turbulence_window_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_OUT = ROOT / "docs" / "_static" / "qa_low_turbulence_comparison.png"
COMPARISON_COLORS = {
    "qa_constraints": "#244c66",
    "qa_plus_nonlinear_heat_flux": "#b45f2a",
}
COMPARISON_LABELS = {
    "qa_constraints": "Reduced QA constraints",
    "qa_plus_nonlinear_heat_flux": "Reduced QA + NL envelope",
}


def _parse_comparison_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_COMPARISON_OUT, help="PNG output path")
    parser.add_argument("--pdf", action="store_true", help="also write a PDF companion")
    parser.add_argument(
        "--workers", type=int, default=1, help="finite-difference worker count"
    )
    parser.add_argument(
        "--steps", type=int, default=60, help="Adam steps for each reduced optimization"
    )
    parser.add_argument(
        "--nonlinear-dt",
        type=float,
        default=0.20,
        help="RK2 time step in the reduced nonlinear envelope",
    )
    parser.add_argument(
        "--nonlinear-steps",
        type=int,
        default=2000,
        help="RK2 steps in the reduced nonlinear envelope",
    )
    parser.add_argument(
        "--nonlinear-weight", type=float, default=8.0, help="transport residual weight"
    )
    parser.add_argument(
        "--iota-operating-floor",
        type=float,
        default=0.70,
        help="operating iota floor above the formal minimum",
    )
    return parser.parse_args(argv)


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
                "residual_gradient_gate_passed": result["residual_gradient_gate"][
                    "passed"
                ],
                "observable_gradient_gate_passed": result["observable_gradient_gate"][
                    "passed"
                ],
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

    b_values = [
        np.asarray(design["lcfs_bmag"]["bmag"], dtype=float)
        for design in payload["designs"]
    ]
    bmin = min(float(np.nanmin(item)) for item in b_values)
    bmax = max(float(np.nanmax(item)) for item in b_values)
    bnorm = colors.Normalize(vmin=bmin, vmax=bmax)
    bcmap = plt.colormaps["jet"]

    for design in payload["designs"]:
        name = design["design_name"]
        color = COMPARISON_COLORS[name]
        label = COMPARISON_LABELS[name]
        scan = design["density_gradient_scan"]
        x = np.asarray(scan["density_gradient_axis"], dtype=float)
        y = np.asarray(scan["heat_flux_mean"], dtype=float)
        ax_scan.plot(x, y, marker="o", lw=2.3, color=color, label=label)
        slope = float(scan["linear_slope_dQ_d_a_over_Ln"])
        ax_scan.text(
            x[-1], y[-1], f"  slope={slope:.2e}", color=color, va="center", fontsize=8
        )

        trace = design["fixed_gradient_trace"]
        time = np.asarray(trace["times"], dtype=float)
        q = np.asarray(trace["heat_flux"], dtype=float)
        start = int(trace["window"]["start_index"])
        running = np.cumsum(q[start:]) / np.arange(1, q[start:].size + 1, dtype=float)
        ax_trace.plot(time, q, lw=2.1, color=color, label=label)
        ax_trace.plot(time[start:], running, lw=1.4, color=color, ls=":", alpha=0.85)
        ax_trace.axhline(
            float(trace["window"]["mean"]), color=color, ls="--", lw=1.1, alpha=0.75
        )
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
        ax_hist.semilogy(steps, hist, lw=2.2, color=COMPARISON_COLORS[name], label=COMPARISON_LABELS[name])
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
        ax.set_title(rf"Reduced LCFS $|B|$: {COMPARISON_LABELS[name]}")
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
        ax.plot_surface(
            zz,
            tt,
            bmag,
            cmap=bcmap,
            norm=bnorm,
            linewidth=0,
            antialiased=True,
            alpha=0.96,
        )
        ax.contour(
            zz,
            tt,
            bmag,
            zdir="z",
            offset=bmin - zpad,
            levels=18,
            cmap="jet",
            linewidths=0.55,
        )
        ax.contour(zz, tt, bmag, levels=9, colors="black", linewidths=0.35, alpha=0.55)
        ax.set_xlabel(r"$\phi_B/\pi$")
        ax.set_ylabel(r"$\theta_B/\pi$")
        ax.set_zlabel("")
        ax.set_zlim(bmin - zpad, bmax + zpad)
        ax.view_init(elev=30, azim=-52)
        ax.set_title(rf"Boozer LCFS $|B|$: {COMPARISON_LABELS[name]}")
        mappable = plt.cm.ScalarMappable(norm=bnorm, cmap=bcmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, fraction=0.040, pad=0.02, label=r"$|B|/B_0$")

    result_by_name = {result["design_name"]: result for result in payload["results"]}
    metric_names = (
        r"$A/A_0$",
        r"$\iota$",
        r"$10^4 R_{QA}$",
        r"$10^2\langle Q_{\rm env}\rangle$",
    )
    x = np.arange(len(metric_names), dtype=float)
    width = 0.36
    for offset, design in zip(
        (-0.5 * width, 0.5 * width), payload["designs"], strict=True
    ):
        name = design["design_name"]
        obs = _obs_map(result_by_name[name])
        values = [
            obs["aspect"] / payload["target_aspect"],
            obs["mean_iota"],
            1.0e4 * obs["qa_residual"],
            1.0e2 * obs["nonlinear_heat_flux_mean"],
        ]
        ax_final.bar(
            x + offset, values, width=width, color=COMPARISON_COLORS[name], label=COMPARISON_LABELS[name]
        )
    ax_final.axhline(1.0, color="black", ls=":", lw=1.0, alpha=0.5)
    ax_final.axhline(
        payload["operating_iota_floor"], color="#7c2d12", ls=":", lw=1.0, alpha=0.55
    )
    ax_final.set_xticks(x)
    ax_final.set_xticklabels(metric_names, rotation=18, ha="right")
    ax_final.set_title("Final reduced observables")
    ax_final.grid(axis="y", alpha=0.25)
    ax_final.legend(frameon=False, fontsize=8)

    conv_names = ("CV", "trend", "half-window drift")
    x = np.arange(len(conv_names), dtype=float)
    for offset, design in zip(
        (-0.5 * width, 0.5 * width), payload["designs"], strict=True
    ):
        name = design["design_name"]
        trace = design["fixed_gradient_trace"]
        window = trace["window"]
        gate = trace["long_window_convergence"]
        values = [
            float(window["cv"]),
            float(window["trend"]),
            float(gate["half_window_relative_mean_change"]),
        ]
        ax_window.bar(
            x + offset, values, width=width, color=COMPARISON_COLORS[name], label=COMPARISON_LABELS[name]
        )
    ax_window.axhline(
        payload["config"]["long_window_max_cv"],
        color="black",
        ls=":",
        lw=1.0,
        alpha=0.45,
    )
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


def write_comparison_artifacts(
    payload: dict[str, Any], out: Path, *, write_pdf: bool = True
) -> dict[str, str]:
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




DEFAULT_COMPARISON = ROOT / "docs" / "_static" / "qa_low_turbulence_comparison.json"
DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "qa_low_turbulence_time_horizon_audit"
HORIZON_COLORS = {
    "qa_constraints": "#244c66",
    "qa_plus_nonlinear_heat_flux": "#b45f2a",
}
HORIZON_LABELS = {
    "qa_constraints": "QA constraints",
    "qa_plus_nonlinear_heat_flux": "QA + reduced NL Q",
}


def _parse_horizon_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=[150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0],
        help="time horizons t v_ti/a to audit; longest is the reference",
    )
    parser.add_argument("--nonlinear-dt", type=float, default=0.20)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--cv-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--trend-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--half-window-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--pdf", action="store_true", help="write a PDF companion")
    return parser.parse_args(argv)


def _load_designs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    designs: list[dict[str, Any]] = []
    for result in payload["results"]:
        designs.append(
            {
                "design_name": str(result["design_name"]),
                "final_params": [float(x) for x in result["final_params"]],
            }
        )
    return designs


def _window_row(
    *,
    design_name: str,
    params: Sequence[float],
    horizon: float,
    nonlinear_dt: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray | int]]:
    steps = int(round(float(horizon) / float(nonlinear_dt)))
    cfg = QALowTurbulenceConfig(nonlinear_dt=float(nonlinear_dt), nonlinear_steps=steps)
    times, heat_flux = qa_low_turbulence_heat_flux_trace(
        params,
        cfg,
        density_gradient=cfg.fixed_density_gradient,
        temperature_gradient=cfg.fixed_temperature_gradient,
    )
    window = qa_low_turbulence_window_metrics(
        times,
        heat_flux,
        tail_fraction=cfg.nonlinear_tail_fraction,
    )
    times_np = np.asarray(times, dtype=float)
    heat_flux_np = np.asarray(heat_flux, dtype=float)
    start = int(window["start_index"])
    tail = heat_flux_np[start:]
    split = max(1, tail.size // 2)
    mean = float(window["mean"])
    half_change = abs(
        float(np.mean(tail[split:])) - float(np.mean(tail[:split]))
    ) / max(
        abs(mean),
        1.0e-14,
    )
    return (
        {
            "design_name": design_name,
            "t_end": float(times_np[-1]),
            "window_start": float(times_np[start]),
            "mean": mean,
            "cv": float(window["cv"]),
            "trend": float(window["trend"]),
            "half_window_relative_mean_change": float(half_change),
        },
        {"times": times_np, "heat_flux": heat_flux_np, "start_index": start},
    )


def build_time_horizon_payload(
    comparison_json: Path = DEFAULT_COMPARISON,
    *,
    horizons: Sequence[float] = (150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0),
    nonlinear_dt: float = 0.20,
    relative_tolerance: float = 1.0e-3,
    cv_tolerance: float = 1.0e-3,
    trend_tolerance: float = 1.0e-3,
    half_window_tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    """Return a JSON-ready time-horizon audit payload."""

    sorted_horizons = sorted(float(h) for h in horizons)
    if len(sorted_horizons) < 2:
        raise ValueError("at least two horizons are required")
    designs = _load_designs(comparison_json)
    rows: list[dict[str, Any]] = []
    traces: dict[str, list[dict[str, np.ndarray | int]]] = {}
    for design in designs:
        design_rows: list[dict[str, Any]] = []
        design_traces: list[dict[str, np.ndarray | int]] = []
        for horizon in sorted_horizons:
            row, trace = _window_row(
                design_name=design["design_name"],
                params=design["final_params"],
                horizon=horizon,
                nonlinear_dt=nonlinear_dt,
            )
            design_rows.append(row)
            design_traces.append(trace)
        reference_mean = float(design_rows[-1]["mean"])
        t400_row = min(design_rows, key=lambda row: abs(float(row["t_end"]) - 400.0))
        t400_mean = float(t400_row["mean"])
        for row in design_rows:
            row["relative_change_vs_t400"] = (float(row["mean"]) - t400_mean) / max(
                abs(t400_mean),
                1.0e-14,
            )
            row["relative_change_vs_reference"] = (
                float(row["mean"]) - reference_mean
            ) / max(
                abs(reference_mean),
                1.0e-14,
            )
            row["reference_t_end"] = float(design_rows[-1]["t_end"])
            rows.append(row)
        traces[design["design_name"]] = design_traces

    metrics: dict[str, Any] = {}
    for design in designs:
        name = design["design_name"]
        design_rows = [row for row in rows if row["design_name"] == name]
        t400_row = min(design_rows, key=lambda row: abs(float(row["t_end"]) - 400.0))
        passed = bool(
            abs(float(t400_row["relative_change_vs_reference"]))
            <= float(relative_tolerance)
            and float(t400_row["cv"]) <= float(cv_tolerance)
            and float(t400_row["trend"]) <= float(trend_tolerance)
            and float(t400_row["half_window_relative_mean_change"])
            <= float(half_window_tolerance)
        )
        metrics[name] = {
            "t400_relative_change_vs_reference": float(
                t400_row["relative_change_vs_reference"]
            ),
            "t400_cv": float(t400_row["cv"]),
            "t400_trend": float(t400_row["trend"]),
            "t400_half_window_relative_mean_change": float(
                t400_row["half_window_relative_mean_change"]
            ),
            "relative_tolerance": float(relative_tolerance),
            "cv_tolerance": float(cv_tolerance),
            "trend_tolerance": float(trend_tolerance),
            "half_window_tolerance": float(half_window_tolerance),
            "passed": passed,
            "recommendation": (
                "t=400 is sufficient for the reduced envelope"
                if passed
                else "extend the reduced envelope beyond t=400"
            ),
        }
    return {
        "kind": "qa_low_turbulence_time_horizon_audit",
        "comparison_json": str(comparison_json),
        "horizons": sorted_horizons,
        "reference_t_end": sorted_horizons[-1],
        "rows": rows,
        "metrics": metrics,
        "passed": all(bool(metric["passed"]) for metric in metrics.values()),
        "claim_level": "reduced nonlinear-envelope time-horizon audit, not a full-GK convergence claim",
        "scope_notes": [
            "The audit reuses the tracked optimized reduced QA designs and varies only the reduced nonlinear-envelope horizon.",
            "Production nonlinear turbulent-flux claims still require long post-transient replicated SPECTRAX-GK audits.",
        ],
        "_plot_traces": traces,
    }


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(payload: dict[str, Any], path: Path) -> None:
    set_plot_style()
    traces = payload["_plot_traces"]
    rows = payload["rows"]
    horizons = payload["horizons"]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    ax = axs[0, 0]
    for name, series in traces.items():
        trace = series[-1]
        times = np.asarray(trace["times"], dtype=float)
        q = np.asarray(trace["heat_flux"], dtype=float)
        start = int(trace["start_index"])
        ax.plot(times, q, color=HORIZON_COLORS[name], lw=2.0, label=HORIZON_LABELS[name])
        ax.axvline(400.0, color=HORIZON_COLORS[name], ls="--", alpha=0.65)
        ax.axvspan(times[start], times[-1], color=HORIZON_COLORS[name], alpha=0.06)
    ax.set_title("Fixed-gradient reduced-envelope traces to reference horizon")
    ax.set_xlabel(r"$t v_{ti}/a$")
    ax.set_ylabel(r"$Q_{\rm env}(t)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axs[0, 1]
    for name in traces:
        design_rows = [row for row in rows if row["design_name"] == name]
        ax.plot(
            [row["t_end"] for row in design_rows],
            [row["mean"] for row in design_rows],
            marker="o",
            color=HORIZON_COLORS[name],
            label=HORIZON_LABELS[name],
        )
    ax.set_title("Late-window mean vs horizon")
    ax.set_xlabel(r"$t_{end} v_{ti}/a$")
    ax.set_ylabel(r"late-window $\langle Q_{\rm env}\rangle$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axs[1, 0]
    for name in traces:
        design_rows = [row for row in rows if row["design_name"] == name]
        ax.semilogy(
            [row["t_end"] for row in design_rows],
            [abs(row["relative_change_vs_reference"]) for row in design_rows],
            marker="o",
            color=HORIZON_COLORS[name],
            label=HORIZON_LABELS[name],
        )
    ax.axhline(1.0e-3, color="black", ls=":", lw=1.0, label="0.1%")
    ax.set_title("Horizon error relative to reference mean")
    ax.set_xlabel(r"$t_{end} v_{ti}/a$")
    ax.set_ylabel("relative mean difference")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)

    ax = axs[1, 1]
    x = np.arange(len(horizons))
    width = 0.18
    for i, name in enumerate(traces):
        design_rows = [row for row in rows if row["design_name"] == name]
        ax.bar(
            x + (i - 0.5) * width,
            [row["half_window_relative_mean_change"] for row in design_rows],
            width=width,
            color=HORIZON_COLORS[name],
            alpha=0.85,
            label=HORIZON_LABELS[name],
        )
    ax.axhline(0.02, color="black", ls="--", lw=1.0, label="long-window gate 2%")
    ax.set_xticks(x, [f"{h:g}" for h in horizons])
    ax.set_title("Half-window drift gate")
    ax.set_xlabel(r"$t_{end} v_{ti}/a$")
    ax.set_ylabel("relative first/second-half mean change")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "QA low-turbulence reduced nonlinear time-horizon audit", fontweight="bold"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_horizon_artifacts(
    payload: dict[str, Any], out_prefix: Path, *, write_pdf: bool = False
) -> dict[str, str]:
    payload_to_write = dict(payload)
    payload_to_write.pop("_plot_traces", None)
    paths = {
        "json": str(out_prefix.with_suffix(".json")),
        "csv": str(out_prefix.with_suffix(".csv")),
        "png": str(out_prefix.with_suffix(".png")),
    }
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(payload_to_write, separators=(",", ":")),
        encoding="utf-8",
    )
    _write_csv(payload["rows"], out_prefix.with_suffix(".csv"))
    _plot(payload, out_prefix.with_suffix(".png"))
    if write_pdf:
        _plot(payload, out_prefix.with_suffix(".pdf"))
        paths["pdf"] = str(out_prefix.with_suffix(".pdf"))
    return paths




def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"comparison", "horizon-audit"}:
        raise SystemExit("mode must be 'comparison' or 'horizon-audit'")
    mode, mode_args = args[0], args[1:]
    if mode == "comparison":
        parsed = _parse_comparison_args(mode_args)
        cfg = QALowTurbulenceConfig(
            steps=int(parsed.steps),
            nonlinear_dt=float(parsed.nonlinear_dt),
            nonlinear_steps=int(parsed.nonlinear_steps),
            nonlinear_weight=float(parsed.nonlinear_weight),
            iota_operating_floor=float(parsed.iota_operating_floor),
        )
        payload = qa_low_turbulence_comparison_payload(
            cfg, finite_difference_workers=int(parsed.workers)
        )
        paths = write_comparison_artifacts(
            payload, parsed.out, write_pdf=bool(parsed.pdf)
        )
        passed = bool(payload["comparison_metrics"]["passed"])
        print(json.dumps({"passed": passed, "paths": paths}, indent=2))
        return 0 if passed else 1

    parsed = _parse_horizon_args(mode_args)
    payload = build_time_horizon_payload(
        parsed.comparison_json,
        horizons=parsed.horizons,
        nonlinear_dt=float(parsed.nonlinear_dt),
        relative_tolerance=float(parsed.relative_tolerance),
        cv_tolerance=float(parsed.cv_tolerance),
        trend_tolerance=float(parsed.trend_tolerance),
        half_window_tolerance=float(parsed.half_window_tolerance),
    )
    paths = write_horizon_artifacts(
        payload, parsed.out_prefix, write_pdf=bool(parsed.pdf)
    )
    print(
        json.dumps(
            {"passed": payload["passed"], "metrics": payload["metrics"], "paths": paths},
            indent=2,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
