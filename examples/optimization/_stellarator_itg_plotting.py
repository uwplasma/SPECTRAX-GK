"""Plotting helpers for differentiable stellarator ITG optimization examples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "growth": "#1b6f8f",
    "quasilinear_flux": "#b55a30",
    "nonlinear_heat_flux": "#386641",
}


def write_result_artifacts(result: Any, out_base: Path, *, title: str) -> None:
    """Write JSON/CSV plus a publication-style summary figure."""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    out_base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_history_csv(payload, out_base.with_suffix(".history.csv"))
    _plot_result(payload, out_base.with_suffix(".png"), title=title)
    _plot_result(payload, out_base.with_suffix(".pdf"), title=title)


def write_comparison_artifacts(payload: dict[str, Any], out_base: Path) -> None:
    """Write the three-objective comparison payload and plot."""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_comparison(payload, out_base.with_suffix(".png"))
    _plot_comparison(payload, out_base.with_suffix(".pdf"))


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


def _plot_result(payload: dict[str, Any], path: Path, *, title: str) -> None:
    history = payload["history"]
    names = list(payload["observable_names"])
    params = list(payload["parameter_names"])
    steps = np.asarray([row["step"] for row in history])
    objective = np.asarray([row["objective"] for row in history])
    obs = np.asarray([row["observables"] for row in history], dtype=float)
    par = np.asarray([row["params"] for row in history], dtype=float)
    idx = {name: names.index(name) for name in names}
    color = COLORS.get(payload["objective_kind"], "#3f3f46")
    has_trace = payload.get("nonlinear_trace") is not None

    if has_trace:
        fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.2), constrained_layout=True)
        axes = axs.ravel()
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10.6, 7.0), constrained_layout=True)
        axes = axs.ravel()
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes[0].semilogy(steps, objective, color=color, lw=2.4)
    axes[0].scatter([steps[0], steps[-1]], [objective[0], objective[-1]], color=color, s=34, zorder=3)
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("constrained objective")
    axes[0].set_title("Objective reduction")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, obs[:, idx["growth_rate"]], lw=2.0, label=r"$\gamma$")
    axes[1].plot(steps, obs[:, idx["quasilinear_heat_flux"]], lw=2.0, label=r"$Q_i^{QL}$")
    axes[1].plot(steps, obs[:, idx["nonlinear_heat_flux_mean"]], lw=2.0, label=r"$\langle Q_i\rangle_{NL}$")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("ITG observable")
    axes[1].set_title("Transport observables")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.25)

    axes[2].plot(steps, obs[:, idx["aspect"]], color="#0f766e", lw=2.0, label="aspect")
    axes[2].axhline(payload["config"]["target_aspect"], color="#0f766e", ls=":", lw=1.4)
    ax_iota = axes[2].twinx()
    ax_iota.plot(steps, obs[:, idx["mean_iota"]], color="#7c2d12", lw=2.0, label=r"$\iota$")
    ax_iota.axhline(payload["config"]["target_iota"], color="#7c2d12", ls=":", lw=1.4)
    axes[2].set_xlabel("optimizer step")
    axes[2].set_ylabel("aspect")
    ax_iota.set_ylabel(r"mean $\iota$")
    axes[2].set_title("QA constraints")
    axes[2].grid(alpha=0.25)

    for col, name in enumerate(params):
        axes[3].plot(steps, par[:, col], lw=1.8, label=_short_param(name))
    axes[3].set_xlabel("optimizer step")
    axes[3].set_ylabel("control value")
    axes[3].set_title("Max-mode-1 controls")
    axes[3].legend(frameon=False, fontsize=8, ncol=2)
    axes[3].grid(alpha=0.25)

    if has_trace:
        trace = payload["nonlinear_trace"]
        time = np.asarray(trace["times"], dtype=float)
        initial_q = np.asarray(trace["initial_heat_flux"], dtype=float)
        final_q = np.asarray(trace["final_heat_flux"], dtype=float)
        start = int(trace["final_window"]["start_index"])
        axes[4].plot(time, initial_q, color="#9ca3af", lw=1.8, label="initial")
        axes[4].plot(time, final_q, color=color, lw=2.2, label="optimized")
        axes[4].axvspan(time[start], time[-1], color=color, alpha=0.12, label="averaging window")
        axes[4].axhline(trace["final_window"]["mean"], color=color, ls="--", lw=1.4)
        axes[4].set_xlabel(r"$t v_{ti}/a$")
        axes[4].set_ylabel(r"$Q_i$ envelope")
        axes[4].set_title("Nonlinear heat-flux window")
        axes[4].legend(frameon=False, fontsize=8)
        axes[4].grid(alpha=0.25)

        final = payload["final_observables"]
        text = (
            f"AD/FD gate: {payload['gradient_gate']['passed']}\n"
            f"max |AD-FD| = {payload['gradient_gate']['max_abs_error']:.2e}\n"
            f"tail CV = {trace['final_window']['cv']:.3f}\n"
            f"tail trend = {trace['final_window']['trend']:.3f}\n"
            f"final QA residual = {final[idx['qa_residual']]:.2e}"
        )
        axes[5].axis("off")
        axes[5].text(
            0.02,
            0.92,
            text,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_alpha(0.35)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(payload: dict[str, Any], path: Path) -> None:
    results = list(payload["results"])
    names = list(payload["observable_names"])
    idx = {name: names.index(name) for name in names}
    labels = [r["objective_kind"].replace("_", " ") for r in results]
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
    metric_labels = [r"$\gamma$", r"$Q_i^{QL}$", r"$\langle Q_i\rangle_{NL}$"]
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
