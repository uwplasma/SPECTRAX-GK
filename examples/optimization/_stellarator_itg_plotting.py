"""Plotting helpers for differentiable stellarator ITG optimization examples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import numpy as np

from spectraxgk import (
    StellaratorITGOptimizationConfig,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
    qa_max_mode1_observables,
    stellarator_itg_density_gradient_scan,
)


COLORS = {
    "growth": "#1b6f8f",
    "quasilinear_flux": "#b55a30",
    "nonlinear_heat_flux": "#386641",
}


def write_result_artifacts(result: Any, out_base: Path, *, title: str) -> None:
    """Write JSON/CSV plus a publication-style summary figure."""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize_artifact_payload(_augment_result_payload(result.to_dict()))
    out_base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_history_csv(payload, out_base.with_suffix(".history.csv"))
    _plot_result(payload, out_base.with_suffix(".png"), title=title)
    _plot_result(payload, out_base.with_suffix(".pdf"), title=title)


def write_comparison_artifacts(payload: dict[str, Any], out_base: Path) -> None:
    """Write the three-objective comparison payload and plot."""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize_artifact_payload(payload)
    out_base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_comparison(payload, out_base.with_suffix(".png"))
    _plot_comparison(payload, out_base.with_suffix(".pdf"))


def write_portfolio_gate_artifacts(payload: dict[str, Any], out_base: Path) -> None:
    """Write a reduced multi-surface/alpha/ky portfolio-gate payload and plot."""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize_artifact_payload(payload)
    out_base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_portfolio_gate(payload, out_base.with_suffix(".png"))
    _plot_portfolio_gate(payload, out_base.with_suffix(".pdf"))


def _sanitize_artifact_payload(value: Any) -> Any:
    """Drop machine-local optional-package path metadata from public sidecars."""

    if isinstance(value, dict):
        return {
            key: _sanitize_artifact_payload(item)
            for key, item in value.items()
            if key not in {"vmec_jax_paths", "booz_xform_jax_paths"}
        }
    if isinstance(value, list):
        return [_sanitize_artifact_payload(item) for item in value]
    return value


def _config_from_payload(payload: dict[str, Any]) -> StellaratorITGOptimizationConfig:
    return StellaratorITGOptimizationConfig(**payload["config"])


def _obs_for_params(params: list[float], cfg: StellaratorITGOptimizationConfig) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in qa_max_mode1_observables(params, cfg).items()
        if np.ndim(np.asarray(value)) == 0
    }


def _trace_payload(params: list[float], cfg: StellaratorITGOptimizationConfig) -> dict[str, Any]:
    times, heat_flux = nonlinear_heat_flux_trace(params, cfg)
    window = nonlinear_heat_flux_window_metrics(
        times,
        heat_flux,
        tail_fraction=cfg.nonlinear_tail_fraction,
    )
    return {
        "trace_kind": "smooth_reduced_nonlinear_envelope_not_full_turbulent_gk",
        "trace_equation": "dE/dt = 2 gamma E - alpha E^2; Q_env = W_i E",
        "density_gradient": float(cfg.reference_density_gradient),
        "temperature_gradient": float(cfg.reference_temperature_gradient),
        "times": np.asarray(times, dtype=float).tolist(),
        "heat_flux": np.asarray(heat_flux, dtype=float).tolist(),
        "window": {
            "mean": float(window["mean"]),
            "std": float(window["std"]),
            "cv": float(window["cv"]),
            "trend": float(window["trend"]),
            "slope": float(window["slope"]),
            "start_index": int(window["start_index"]),
        },
    }


def _reduced_surface(params: list[float], cfg: StellaratorITGOptimizationConfig) -> dict[str, Any]:
    obs = _obs_for_params(params, cfg)
    minor_shift, elong_shift, ripple, shear_shift = [float(value) for value in params]
    theta = np.linspace(0.0, 2.0 * np.pi, 44, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 44, endpoint=False)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    aspect = max(float(obs["aspect"]), 1.0e-6)
    major_radius = 1.0
    minor_radius = major_radius / aspect
    nfp = 2
    elongation = 1.0 + 0.22 * elong_shift
    axis_radius = major_radius * (1.0 + 0.18 * ripple * np.cos(nfp * zz))
    axis_height = minor_radius * 0.95 * ripple * np.sin(nfp * zz)
    radius = axis_radius + minor_radius * (
        np.cos(tt)
        + 1.10 * ripple * np.cos(tt - nfp * zz)
        + 0.035 * shear_shift * np.cos(2.0 * tt)
        + 0.050 * minor_shift * np.cos(tt + nfp * zz)
    )
    height = axis_height + minor_radius * (
        elongation * np.sin(tt)
        + 0.90 * ripple * np.sin(tt - nfp * zz)
        + 0.035 * shear_shift * np.sin(2.0 * tt)
    )
    return {
        "theta": theta.tolist(),
        "zeta": zeta.tolist(),
        "x": (radius * np.cos(zz)).tolist(),
        "y": (radius * np.sin(zz)).tolist(),
        "z": height.tolist(),
        "scope": "reduced_max_mode1_visualization_not_solved_vmec_equilibrium",
    }


def _reduced_lcfs_bmag(params: list[float], cfg: StellaratorITGOptimizationConfig) -> dict[str, Any]:
    obs = _obs_for_params(params, cfg)
    _, elong_shift, ripple, _ = [float(value) for value in params]
    theta = np.linspace(0.0, 2.0 * np.pi, 44, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 44, endpoint=False)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    nfp = 2
    qa_amp = float(obs["qa_residual"])
    bmag = (
        1.0
        + 0.052 * np.cos(tt)
        + 0.016 * elong_shift * np.cos(2.0 * tt)
        + 0.190 * ripple * np.cos(tt - nfp * zz)
        + 0.026 * qa_amp * np.cos(2.0 * tt - nfp * zz)
    )
    return {
        "theta": theta.tolist(),
        "zeta": zeta.tolist(),
        "bmag": bmag.tolist(),
        "scope": "synthetic_reduced_lcfs_bmag_not_booz_xform_jax_output",
    }


def _augment_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = _config_from_payload(payload)
    initial_params = [float(value) for value in payload["initial_params"]]
    final_params = [float(value) for value in payload["final_params"]]
    payload["reduced_diagnostics"] = {
        "claim_level": "reduced_max_mode1_diagnostics_not_solved_vmec_or_full_nonlinear_scan",
        "initial": {
            "density_gradient_scan": stellarator_itg_density_gradient_scan(initial_params, cfg),
            "fixed_gradient_trace": _trace_payload(initial_params, cfg),
            "surface": _reduced_surface(initial_params, cfg),
            "lcfs_bmag": _reduced_lcfs_bmag(initial_params, cfg),
        },
        "final": {
            "density_gradient_scan": stellarator_itg_density_gradient_scan(final_params, cfg),
            "fixed_gradient_trace": _trace_payload(final_params, cfg),
            "surface": _reduced_surface(final_params, cfg),
            "lcfs_bmag": _reduced_lcfs_bmag(final_params, cfg),
        },
    }
    return payload


def _write_history_csv(payload: dict[str, Any], path: Path) -> None:
    names = list(payload["observable_names"])
    params = list(payload["parameter_names"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["step", "objective", "gradient_norm"] + params + names
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in payload["history"]:
            out = {
                "step": row["step"],
                "objective": row["objective"],
                "gradient_norm": row["gradient_norm"],
            }
            out.update(dict(zip(params, row["params"], strict=True)))
            out.update(dict(zip(names, row["observables"], strict=True)))
            writer.writerow(out)


def _set_equal_3d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    limits = np.asarray(
        [
            [np.nanmin(x), np.nanmax(x)],
            [np.nanmin(y), np.nanmax(y)],
        ],
        dtype=float,
    )
    center = np.mean(limits, axis=1)
    radius = 0.52 * float(np.max(limits[:, 1] - limits[:, 0]))
    z_center = 0.5 * float(np.nanmin(z) + np.nanmax(z))
    z_radius = max(0.045, 0.62 * float(np.nanmax(z) - np.nanmin(z)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(z_center - z_radius, z_center + z_radius)
    ax.set_box_aspect((1.0, 1.0, 0.45))


def _plot_reduced_surface(ax: plt.Axes, fig: plt.Figure, diagnostics: dict[str, Any], *, title: str) -> None:
    surface = diagnostics["surface"]
    bmap = diagnostics["lcfs_bmag"]
    x = np.asarray(surface["x"], dtype=float)
    y = np.asarray(surface["y"], dtype=float)
    z = np.asarray(surface["z"], dtype=float)
    bmag = np.asarray(bmap["bmag"], dtype=float)
    norm = mpl_colors.Normalize(vmin=float(np.nanmin(bmag)), vmax=float(np.nanmax(bmag)))
    cmap = plt.colormaps["cividis"]
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=cmap(norm(bmag)),
        alpha=0.95,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=max(1, x.shape[0] // 9),
        cstride=max(1, x.shape[1] // 9),
        color="white",
        linewidth=0.25,
        alpha=0.40,
    )
    _set_equal_3d(ax, x, y, z)
    ax.view_init(elev=25, azim=48)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, label=r"$|B|/B_0$")


def _plot_reduced_boozer_bmag(ax: plt.Axes, fig: plt.Figure, diagnostics: dict[str, Any], *, title: str) -> None:
    bmap = diagnostics["lcfs_bmag"]
    theta = np.asarray(bmap["theta"], dtype=float) / np.pi
    zeta = np.asarray(bmap["zeta"], dtype=float) / np.pi
    zz, tt = np.meshgrid(zeta, theta, indexing="xy")
    bmag = np.asarray(bmap["bmag"], dtype=float)
    bmin = float(np.nanmin(bmag))
    bmax = float(np.nanmax(bmag))
    zpad = 0.08 * max(bmax - bmin, 1.0e-6)
    norm = mpl_colors.Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.colormaps["cividis"]
    ax.plot_surface(zz, tt, bmag, cmap=cmap, norm=norm, linewidth=0, antialiased=True, alpha=0.96)
    ax.contourf(zz, tt, bmag, zdir="z", offset=bmin - zpad, levels=18, cmap=cmap, norm=norm, alpha=0.76)
    ax.contour(zz, tt, bmag, levels=9, colors="white", linewidths=0.32, alpha=0.7)
    ax.set_xlabel(r"$\phi_B/\pi$")
    ax.set_ylabel(r"$\theta_B/\pi$")
    ax.set_zlabel("")
    ax.set_zlim(bmin - zpad, bmax + zpad)
    ax.view_init(elev=30, azim=-52)
    ax.set_title(title)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, label=r"$|B|/B_0$")


def _plot_result(payload: dict[str, Any], path: Path, *, title: str) -> None:
    history = payload["history"]
    names = list(payload["observable_names"])
    params = list(payload["parameter_names"])
    steps = np.asarray([row["step"] for row in history], dtype=float)
    objective = np.asarray([row["objective"] for row in history], dtype=float)
    obs = np.asarray([row["observables"] for row in history], dtype=float)
    par = np.asarray([row["params"] for row in history], dtype=float)
    idx = {name: names.index(name) for name in names}
    color = COLORS.get(payload["objective_kind"], "#3f3f46")
    diagnostics = payload["reduced_diagnostics"]
    initial_diag = diagnostics["initial"]
    final_diag = diagnostics["final"]

    fig = plt.figure(figsize=(17.0, 13.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, hspace=0.12, wspace=0.10)
    ax_obj = fig.add_subplot(grid[0, 0])
    ax_obs = fig.add_subplot(grid[0, 1])
    ax_constraints = fig.add_subplot(grid[0, 2])
    ax_controls = fig.add_subplot(grid[1, 0])
    ax_scan = fig.add_subplot(grid[1, 1])
    ax_trace = fig.add_subplot(grid[1, 2])
    ax_surface = fig.add_subplot(grid[2, 0], projection="3d")
    ax_boozer = fig.add_subplot(grid[2, 1], projection="3d")
    ax_metrics = fig.add_subplot(grid[2, 2])
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax_obj.semilogy(steps, objective, color=color, lw=2.4)
    ax_obj.scatter([steps[0], steps[-1]], [objective[0], objective[-1]], color=color, s=34, zorder=3)
    ax_obj.set_xlabel("optimizer step")
    ax_obj.set_ylabel("constrained objective")
    ax_obj.set_title("Objective reduction")
    ax_obj.grid(alpha=0.25)

    ax_obs.plot(steps, obs[:, idx["growth_rate"]], lw=2.0, label=r"$\gamma$")
    ax_obs.plot(steps, obs[:, idx["quasilinear_heat_flux"]], lw=2.0, label=r"$Q_i^{QL}$")
    ax_obs.plot(
        steps,
        obs[:, idx["nonlinear_heat_flux_mean"]],
        lw=2.0,
        label=r"$Q_{\rm env}$",
    )
    ax_obs.set_xlabel("optimizer step")
    ax_obs.set_ylabel("ITG observable")
    ax_obs.set_title("Transport observables")
    ax_obs.legend(frameon=False, fontsize=9)
    ax_obs.grid(alpha=0.25)

    ax_constraints.plot(steps, obs[:, idx["aspect"]], color="#0f766e", lw=2.0, label="aspect")
    ax_constraints.axhline(payload["config"]["target_aspect"], color="#0f766e", ls=":", lw=1.4)
    ax_iota = ax_constraints.twinx()
    ax_iota.plot(steps, obs[:, idx["mean_iota"]], color="#7c2d12", lw=2.0, label=r"$\iota$")
    ax_iota.axhline(payload["config"]["target_iota"], color="#7c2d12", ls=":", lw=1.4)
    ax_constraints.set_xlabel("optimizer step")
    ax_constraints.set_ylabel("aspect")
    ax_iota.set_ylabel(r"mean $\iota$")
    ax_constraints.set_title("QA constraints")
    ax_constraints.grid(alpha=0.25)

    for col, name in enumerate(params):
        ax_controls.plot(steps, par[:, col], lw=1.8, label=_short_param(name))
    ax_controls.set_xlabel("optimizer step")
    ax_controls.set_ylabel("control value")
    ax_controls.set_title("Max-mode-1 controls")
    ax_controls.legend(frameon=False, fontsize=8, ncol=2)
    ax_controls.grid(alpha=0.25)

    for label, diag, plot_color, linestyle in (
        ("initial", initial_diag, "#9ca3af", "--"),
        ("optimized", final_diag, color, "-"),
    ):
        scan = diag["density_gradient_scan"]
        ax_scan.plot(
            scan["density_gradient_axis"],
            scan["heat_flux_mean"],
            marker="o",
            lw=2.1,
            color=plot_color,
            ls=linestyle,
            label=f"{label} " + r"$Q_{\rm env}$",
        )
    ax_scan.set_xlabel(r"density gradient $a/L_n$")
    ax_scan.set_ylabel(r"late-window $\langle Q_{\rm env}\rangle$")
    ax_scan.set_title(r"Reduced response at fixed $a/L_{Ti}=6$")
    ax_scan.legend(frameon=False, fontsize=8)
    ax_scan.grid(alpha=0.25)

    for label, diag, plot_color, linestyle in (
        ("initial", initial_diag, "#9ca3af", "--"),
        ("optimized", final_diag, color, "-"),
    ):
        trace = diag["fixed_gradient_trace"]
        time = np.asarray(trace["times"], dtype=float)
        heat_flux = np.asarray(trace["heat_flux"], dtype=float)
        start = int(trace["window"]["start_index"])
        ax_trace.plot(time, heat_flux, color=plot_color, ls=linestyle, lw=2.0, label=label)
        if label == "optimized":
            ax_trace.axvspan(time[start], time[-1], color=color, alpha=0.10)
            ax_trace.axhline(float(trace["window"]["mean"]), color=color, ls=":", lw=1.2)
    ax_trace.set_xlabel(r"$t v_{ti}/a$")
    ax_trace.set_ylabel(r"$Q_{\rm env}(t)$")
    ax_trace.set_title("Reduced fixed-gradient envelope")
    ax_trace.legend(frameon=False, fontsize=8)
    ax_trace.grid(alpha=0.25)

    _plot_reduced_surface(ax_surface, fig, final_diag, title=r"Optimized reduced LCFS $|B|$")
    _plot_reduced_boozer_bmag(ax_boozer, fig, final_diag, title=r"Optimized Boozer LCFS $|B|$")

    initial_vals = np.asarray(payload["initial_observables"], dtype=float)
    final_vals = np.asarray(payload["final_observables"], dtype=float)
    metric_keys = ("growth_rate", "quasilinear_heat_flux", "nonlinear_heat_flux_mean")
    ratios = [
        final_vals[idx[key]] / max(initial_vals[idx[key]], 1.0e-14)
        for key in metric_keys
    ]
    metric_labels = [r"$\gamma$", r"$Q_i^{QL}$", r"$Q_{\rm env}$"]
    x = np.arange(len(metric_labels), dtype=float)
    ax_metrics.bar(x, ratios, color=color, alpha=0.88, label="final / initial")
    ax_metrics.axhline(1.0, color="black", ls=":", lw=1.0, alpha=0.55)
    ax_metrics.set_xticks(x, metric_labels)
    ax_metrics.set_ylabel("ratio")
    ax_metrics.set_title("Reduced transport reduction")
    ax_metrics.grid(axis="y", alpha=0.25)
    ax_gate = ax_metrics.twinx()
    gate_values = [
        float(payload["gradient_gate"]["max_abs_error"]),
        float(final_diag["fixed_gradient_trace"]["window"]["cv"]),
        float(final_diag["fixed_gradient_trace"]["window"]["trend"]),
    ]
    gate_labels = ("AD-FD", "CV", "trend")
    x_gate = np.arange(len(gate_labels), dtype=float) + len(metric_labels) + 0.65
    ax_gate.bar(x_gate, gate_values, width=0.42, color="#475569", alpha=0.72, label="gates")
    ax_gate.set_yscale("log")
    ax_gate.set_ylabel("gate metric")
    ax_metrics.set_xlim(-0.6, float(x_gate[-1]) + 0.7)
    ax_metrics.set_xticks(np.concatenate([x, x_gate]))
    ax_metrics.set_xticklabels(metric_labels + list(gate_labels), rotation=18, ha="right")

    for ax in (ax_obj, ax_obs, ax_constraints, ax_controls, ax_scan, ax_trace, ax_metrics):
        for spine in ax.spines.values():
            spine.set_alpha(0.35)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(payload: dict[str, Any], path: Path) -> None:
    results = list(payload["results"])
    names = list(payload["observable_names"])
    idx = {name: names.index(name) for name in names}
    labels = [_objective_label(str(r["objective_kind"])) for r in results]
    wrapped_labels = [label.replace("quasilinear flux", "quasilinear\nflux") for label in labels]
    colors = [COLORS.get(r["objective_kind"], "#3f3f46") for r in results]
    final_obs = np.asarray([r["final_observables"] for r in results], dtype=float)
    initial_obs = np.asarray([r["initial_observables"] for r in results], dtype=float)
    x = np.arange(len(results))

    fig, axs = plt.subplots(2, 2, figsize=(11.6, 7.8), constrained_layout=True)
    fig.suptitle("Differentiable QA stellarator ITG optimization comparison", fontsize=13.5, fontweight="bold")

    for i, result in enumerate(results):
        hist = np.asarray([row["objective"] for row in result["history"]], dtype=float)
        axs[0, 0].semilogy(hist, color=colors[i], lw=2.0, label=labels[i])
    axs[0, 0].set_xlabel("optimizer step")
    axs[0, 0].set_ylabel("objective")
    axs[0, 0].set_title("Constrained objective histories")
    axs[0, 0].legend(frameon=False, fontsize=8)
    axs[0, 0].grid(alpha=0.25)

    width = 0.24
    metrics = ["growth_rate", "quasilinear_heat_flux", "nonlinear_heat_flux_mean"]
    metric_labels = [r"$\gamma$", r"$Q_i^{QL}$", r"$Q_{\rm env}$"]
    for j, metric in enumerate(metrics):
        axs[0, 1].bar(x + (j - 1) * width, final_obs[:, idx[metric]], width=width, label=metric_labels[j])
    axs[0, 1].set_xticks(x, wrapped_labels, rotation=0, ha="center")
    axs[0, 1].set_ylabel("final value")
    axs[0, 1].set_title("Final transport observables")
    axs[0, 1].legend(frameon=False, fontsize=8)
    axs[0, 1].grid(axis="y", alpha=0.25)

    reduction = final_obs[:, [idx[m] for m in metrics]] / initial_obs[:, [idx[m] for m in metrics]]
    im = axs[1, 0].imshow(reduction, cmap="viridis_r", vmin=0.0, vmax=max(1.0, float(np.nanmax(reduction))))
    axs[1, 0].set_xticks(np.arange(len(metrics)), metric_labels)
    axs[1, 0].set_yticks(x, wrapped_labels)
    axs[1, 0].set_title("Final / initial transport ratio")
    for i in range(reduction.shape[0]):
        for j in range(reduction.shape[1]):
            axs[1, 0].text(j, i, f"{reduction[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)

    aspect_values = final_obs[:, idx["aspect"]]
    iota_values = final_obs[:, idx["mean_iota"]]
    axs[1, 1].scatter(aspect_values, iota_values, s=90, c=colors)
    for i, label in enumerate(labels):
        offset = (-8, 7) if aspect_values[i] > np.nanmean(aspect_values) else (7, 7)
        ha = "right" if offset[0] < 0 else "left"
        axs[1, 1].annotate(
            label,
            (aspect_values[i], iota_values[i]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha=ha,
        )
    axs[1, 1].axvline(results[0]["config"]["target_aspect"], color="#0f766e", ls=":", lw=1.4)
    axs[1, 1].axhline(results[0]["config"]["target_iota"], color="#7c2d12", ls=":", lw=1.4)
    xpad = max(1.5e-3, 0.15 * float(np.ptp(aspect_values) or 1.0e-3))
    ypad = max(1.0e-4, 0.25 * float(np.ptp(iota_values) or 1.0e-4))
    axs[1, 1].set_xlim(
        min(float(np.min(aspect_values)), results[0]["config"]["target_aspect"]) - xpad,
        max(float(np.max(aspect_values)), results[0]["config"]["target_aspect"]) + xpad,
    )
    axs[1, 1].set_ylim(
        min(float(np.min(iota_values)), results[0]["config"]["target_iota"]) - ypad,
        max(float(np.max(iota_values)), results[0]["config"]["target_iota"]) + ypad,
    )
    axs[1, 1].set_xlabel("aspect")
    axs[1, 1].set_ylabel(r"mean $\iota$")
    axs[1, 1].set_title("Final QA constraint location")
    axs[1, 1].grid(alpha=0.25)

    for ax in axs.ravel():
        for spine in ax.spines.values():
            spine.set_alpha(0.35)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _short_param(name: str) -> str:
    return {
        "minor_radius_log_shift": "minor radius",
        "vertical_elongation_shift": "elongation",
        "helical_ripple_amplitude": "helical ripple",
        "magnetic_shear_shift": "shear",
    }.get(name, name)


def _objective_label(kind: str) -> str:
    return {
        "growth": "growth",
        "quasilinear_flux": "quasilinear flux",
        "nonlinear_heat_flux": "reduced NL window",
    }.get(kind, kind.replace("_", " "))


def _plot_portfolio_gate(payload: dict[str, Any], path: Path) -> None:
    sample = payload["sample_set"]
    surfaces = np.asarray(sample["surfaces"], dtype=float)
    alphas = np.asarray(sample["alphas"], dtype=float)
    ky_values = np.asarray(sample["ky_values"], dtype=float)
    objectives = list(payload["objective_names"])
    n_obj = len(objectives)
    table = _portfolio_tensor_from_payload(payload, surfaces, alphas, ky_values, objectives)
    alpha_mean = np.mean(table, axis=1)

    fig, axs = plt.subplots(1, n_obj + 1, figsize=(5.1 * (n_obj + 1), 4.3), constrained_layout=True)
    if n_obj == 0:
        axs = np.asarray([axs])
    fig.suptitle("Reduced multi-surface/field-line ITG objective portfolio gate", fontsize=13.5, fontweight="bold")

    for idx, objective in enumerate(objectives):
        ax = axs[idx]
        data = alpha_mean[:, :, idx]
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(_objective_label(objective))
        ax.set_xlabel(r"$k_y \rho_i$")
        ax.set_ylabel("normalized toroidal flux")
        ax.set_xticks(np.arange(len(ky_values)), [f"{value:.2f}" for value in ky_values])
        ax.set_yticks(np.arange(len(surfaces)), [f"{value:.2f}" for value in surfaces])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2e}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    report = payload["portfolio_report"]
    conditioning = report["conditioning_gate"]
    scalar_gate = report["scalar_gradient_gate"]
    row_gate = report["row_jacobian_gate"]
    text = (
        f"Gate passed: {payload['passed']}\n"
        f"Samples: {sample['n_samples']} ({len(surfaces)} surfaces, {len(sample['alphas'])} alphas, {len(ky_values)} ky)\n"
        f"Reduction: {sample['reduction']}\n"
        f"Scalar AD/FD: {scalar_gate['passed']}  max |err|={scalar_gate['max_abs_error']:.2e}\n"
        f"Rows AD/FD: {row_gate['passed']}  max |err|={row_gate['max_abs_error']:.2e}\n"
        f"Rank: {conditioning['sensitivity_map_rank']}/{conditioning['min_rank']}\n"
        f"Condition number: {conditioning['jacobian_condition_number']:.2e}\n"
        "Claim: reduced portfolio gate, not a nonlinear transport claim"
    )
    axs[-1].axis("off")
    axs[-1].text(
        0.02,
        0.96,
        text,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    for ax in axs[:-1]:
        for spine in ax.spines.values():
            spine.set_alpha(0.35)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _portfolio_tensor_from_payload(
    payload: dict[str, Any],
    surfaces: np.ndarray,
    alphas: np.ndarray,
    ky_values: np.ndarray,
    objectives: list[str],
) -> np.ndarray:
    """Return ``(surface, alpha, ky, objective)`` rows from old or new payloads."""

    raw_table = payload.get("base_objective_tensor", payload["base_objective_table"])
    table = np.asarray(raw_table, dtype=float)
    expected_shape = (len(surfaces), len(alphas), len(ky_values), len(objectives))
    if table.shape == expected_shape:
        return table
    flat_shape = (int(np.prod(expected_shape[:-1])), expected_shape[-1])
    if table.shape == flat_shape:
        return table.reshape(expected_shape)
    raise ValueError(
        "portfolio gate table must have shape "
        f"{expected_shape} or flattened shape {flat_shape}; got {table.shape}"
    )
