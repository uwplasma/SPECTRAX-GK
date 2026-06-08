#!/usr/bin/env python3
"""Summarize how current quasilinear candidates transfer to stellarator heat fluxes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"

CASE_LABELS = {
    "hsx_nonlinear_window": "HSX",
    "w7x_nonlinear_window": "W7-X",
    "cyclone_miller_long_window": "Cyclone Miller",
    "dshape_external_vmec_t250_window": "D-shaped VMEC",
    "updown_asym_external_vmec_t450_window": "Up-down VMEC",
    "circular_external_vmec_t450_window": "Circular VMEC",
    "itermodel_external_vmec_t350_window": "ITERModel VMEC",
    "cyclone_long_window": "Cyclone",
}

MODEL_LABELS = {
    "positive_mixing_length": "positive-growth ML",
    "linear_weight": "linear-weight fit",
    "spectral_envelope_ridge": "spectral-envelope ridge",
}

MODEL_COLORS = {
    "positive_mixing_length": "#7f1d1d",
    "linear_weight": "#6b7280",
    "spectral_envelope_ridge": "#0f766e",
}

MODEL_MARKERS = {
    "positive_mixing_length": "X",
    "linear_weight": "o",
    "spectral_envelope_ridge": "D",
}

STELLARATOR_CASES = ("hsx_nonlinear_window", "w7x_nonlinear_window")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _relative_error(predicted: float | None, observed: float | None, *, floor: float = 1.0e-12) -> float | None:
    if predicted is None or observed is None:
        return None
    if not math.isfinite(predicted) or not math.isfinite(observed):
        return None
    return abs(predicted - observed) / max(abs(observed), floor)


def _candidate_predictions(candidate_report: dict[str, Any], candidate: str) -> dict[str, float]:
    payload = candidate_report["candidates"][candidate]
    return {
        str(row["holdout_case"]): float(row["predicted_heat_flux"])
        for row in payload["rows"]
    }


def _candidate_intervals(candidate_report: dict[str, Any], candidate: str) -> dict[str, tuple[float, float]]:
    payload = candidate_report["candidates"][candidate]
    return {
        str(row["holdout_case"]): (
            float(row["prediction_interval_low"]),
            float(row["prediction_interval_high"]),
        )
        for row in payload["rows"]
    }


def build_report(
    *,
    saturation_report: Path = STATIC / "quasilinear_saturation_rule_sweep.json",
    candidate_report: Path = STATIC / "quasilinear_candidate_uncertainty.json",
    model_selection_status: Path = STATIC / "quasilinear_model_selection_status.json",
    qa_audit: Path = STATIC / "qa_no_ess_to_optimized_nonlinear_audit.json",
    qh_gate: Path = STATIC / "external_vmec_qh_high_grid_convergence_gate.json",
) -> dict[str, Any]:
    """Build a claim-scoped stellarator quasilinear usefulness report."""

    sat = _load_json(saturation_report)
    cand = _load_json(candidate_report)
    status = _load_json(model_selection_status)
    qa = _load_json(qa_audit)
    qh = _load_json(qh_gate)

    simple_predictions = {
        rule: [float(x) for x in sat["rules"][rule]["predicted_heat_flux"]]
        for rule in ("positive_mixing_length", "linear_weight")
    }
    spectral_predictions = _candidate_predictions(cand, "spectral_envelope_ridge")
    spectral_intervals = _candidate_intervals(cand, "spectral_envelope_ridge")

    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(sat["cases"]):
        case_name = str(case["case"])
        observed = float(case["observed_heat_flux"])
        row: dict[str, Any] = {
            "case": case_name,
            "label": CASE_LABELS.get(case_name, case_name.replace("_", " ")),
            "geometry": str(case["geometry"]),
            "split": str(case["split"]),
            "observed_heat_flux": observed,
            "observed_heat_flux_std": case.get("observed_heat_flux_std"),
            "stellarator_family": case_name in STELLARATOR_CASES,
        }
        for model in ("positive_mixing_length", "linear_weight"):
            predicted = simple_predictions[model][idx]
            row[f"{model}_prediction"] = predicted
            row[f"{model}_relative_error"] = _relative_error(predicted, observed)
        predicted = spectral_predictions.get(case_name)
        row["spectral_envelope_ridge_prediction"] = predicted
        row["spectral_envelope_ridge_relative_error"] = _relative_error(predicted, observed)
        interval = spectral_intervals.get(case_name)
        if interval is not None:
            row["spectral_envelope_ridge_interval_low"] = interval[0]
            row["spectral_envelope_ridge_interval_high"] = interval[1]
            row["spectral_envelope_ridge_interval_contains_observed"] = interval[0] <= observed <= interval[1]
        rows.append(row)

    qa_comparison = dict(qa["comparison"])
    qh_gate_report = dict(qh["gate_report"])
    qh_pairwise = None
    for gate in qh_gate_report.get("gates", []):
        if gate.get("metric") == "least_window_pairwise_heat_flux_symmetric_relative_difference":
            qh_pairwise = gate.get("observed")
            break

    metrics = dict(status["metrics"])
    simple_metrics = {
        rule: sat["rules"][rule]["holdout_mean_abs_relative_error"]
        for rule in ("positive_mixing_length", "linear_weight", "absolute_growth_mixing_length")
    }
    best_model = "spectral_envelope_ridge"
    claim = (
        "Simple one-constant quasilinear rules do not transfer as absolute stellarator heat-flux "
        "predictors on the admitted portfolio. The spectral-envelope ridge candidate is the best "
        "current scoped model-selection result, but QA/QH coverage still requires matched, converged "
        "nonlinear holdouts before universal stellarator-flux claims."
    )

    return {
        "kind": "quasilinear_stellarator_usefulness",
        "claim_level": "scoped_model_skill_summary_not_runtime_absolute_flux_predictor",
        "source_artifacts": {
            "saturation_rule_sweep": str(saturation_report.relative_to(ROOT)),
            "candidate_uncertainty": str(candidate_report.relative_to(ROOT)),
            "model_selection_status": str(model_selection_status.relative_to(ROOT)),
            "qa_matched_nonlinear_audit": str(qa_audit.relative_to(ROOT)),
            "qh_high_grid_convergence_gate": str(qh_gate.relative_to(ROOT)),
        },
        "models": {
            "positive_mixing_length": {
                "label": MODEL_LABELS["positive_mixing_length"],
                "holdout_mean_abs_relative_error": simple_metrics["positive_mixing_length"],
                "accepted": False,
                "reason": "fails holdout gate and predicts zero for admitted HSX/W7-X finite nonlinear windows",
            },
            "linear_weight": {
                "label": MODEL_LABELS["linear_weight"],
                "mean_abs_relative_error": metrics["linear_weight_mean_abs_relative_error"],
                "accepted": False,
                "reason": "beaten by spectral-envelope candidate and fails transport gate",
            },
            "spectral_envelope_ridge": {
                "label": MODEL_LABELS["spectral_envelope_ridge"],
                "mean_abs_relative_error": metrics["candidate_mean_abs_relative_error"],
                "prediction_interval_coverage": metrics["candidate_prediction_interval_coverage"],
                "accepted": True,
                "reason": "best current scoped candidate; not exposed as a runtime saturation rule",
            },
        },
        "rows": rows,
        "stellarator_status": {
            "HSX": "admitted nonlinear holdout; simple positive-growth QL predicts zero while nonlinear Q is finite",
            "W7-X": "admitted nonlinear holdout; simple positive-growth QL predicts zero while nonlinear Q is finite",
            "QA": {
                "status": "matched nonlinear audit only; strict QL-vs-nonlinear optimization comparison still staged",
                "baseline_heat_flux": qa_comparison["baseline_mean"],
                "optimized_heat_flux": qa_comparison["optimized_mean"],
                "relative_reduction": qa_comparison["relative_reduction"],
            },
            "QH": {
                "status": "excluded from QL calibration until grid/window convergence passes",
                "high_grid_gate_passed": bool(qh.get("passed", False)),
                "least_window_pairwise_heat_flux_symmetric_relative_difference": qh_pairwise,
            },
        },
        "best_current_model": best_model,
        "readme_sentence": claim,
        "notes": [
            "The plot uses only tracked JSON artifacts and does not refit any quasilinear model.",
            "QA and QH are shown as scope/status evidence, not as accepted quasilinear calibration points.",
            "A universal stellarator absolute-flux proxy needs more converged nonlinear holdouts and richer saturation theory.",
        ],
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    rows = list(report["rows"])
    fields = [
        "case",
        "label",
        "geometry",
        "split",
        "observed_heat_flux",
        "observed_heat_flux_std",
        "positive_mixing_length_prediction",
        "positive_mixing_length_relative_error",
        "linear_weight_prediction",
        "linear_weight_relative_error",
        "spectral_envelope_ridge_prediction",
        "spectral_envelope_ridge_relative_error",
        "spectral_envelope_ridge_interval_low",
        "spectral_envelope_ridge_interval_high",
        "spectral_envelope_ridge_interval_contains_observed",
        "stellarator_family",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_figure(report: dict[str, Any], *, out: Path, title: str, dpi: int = 220) -> dict[str, str]:
    """Write a publication-facing stellarator QL usefulness figure."""

    out.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    rows = list(report["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax0, ax1 = axes

    positive_values: list[float] = []
    for row in rows:
        obs = float(row["observed_heat_flux"])
        if obs > 0.0:
            positive_values.append(obs)
        for model in MODEL_LABELS:
            pred = row.get(f"{model}_prediction")
            if isinstance(pred, (float, int)) and math.isfinite(float(pred)) and float(pred) > 0.0:
                positive_values.append(float(pred))
    lo = min(positive_values) * 0.35
    hi = max(positive_values) * 2.2
    ax0.plot([lo, hi], [lo, hi], color="0.2", linewidth=1.4, linestyle="--", label="1:1")
    ax0.fill_between([lo, hi], [lo / 2.0, hi / 2.0], [lo * 2.0, hi * 2.0], color="0.85", alpha=0.55, label="factor 2")
    ax0.fill_between([lo, hi], [lo / 10.0, hi / 10.0], [lo * 10.0, hi * 10.0], color="0.93", alpha=0.55, label="factor 10")

    for model, label in MODEL_LABELS.items():
        xs = []
        ys = []
        edgecolors = []
        sizes = []
        for row in rows:
            pred = row.get(f"{model}_prediction")
            if pred is None or not math.isfinite(float(pred)):
                continue
            xs.append(float(row["observed_heat_flux"]))
            ys.append(max(float(pred), lo * 0.45))
            edgecolors.append("black" if bool(row.get("stellarator_family")) else "white")
            sizes.append(88 if bool(row.get("stellarator_family")) else 58)
        ax0.scatter(
            xs,
            ys,
            s=sizes,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            edgecolor=edgecolors,
            linewidth=1.0,
            label=label,
            zorder=4,
        )
    label_offsets = {"hsx_nonlinear_window": (10, 12), "w7x_nonlinear_window": (10, -16)}
    for row in rows:
        if row["case"] in STELLARATOR_CASES:
            obs = float(row["observed_heat_flux"])
            spec = float(row["spectral_envelope_ridge_prediction"])
            ax0.annotate(
                row["label"],
                (obs, spec),
                xytext=label_offsets[str(row["case"])],
                textcoords="offset points",
                fontsize=8.5,
                color="0.15",
                arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.7},
            )
    ax0.text(
        0.03,
        0.03,
        "ML predicts zero for HSX/W7-X;\npoints are clipped to log floor",
        transform=ax0.transAxes,
        fontsize=8.7,
        color="#7f1d1d",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#fecaca", "alpha": 0.9},
    )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel(r"Observed nonlinear late-window $\langle Q_i\rangle$")
    ax0.set_ylabel(r"Quasilinear model prediction")
    ax0.set_title("Admitted train/holdout skill")
    ax0.legend(loc="upper left", fontsize=8.2, frameon=True, ncols=1)

    stellarator_rows = [row for row in rows if row["case"] in STELLARATOR_CASES]
    labels = [row["label"] for row in stellarator_rows] + ["QA audit", "QH pilot"]
    x = np.arange(len(labels), dtype=float)
    width = 0.22
    observed = [float(row["observed_heat_flux"]) for row in stellarator_rows]
    spectral = [float(row["spectral_envelope_ridge_prediction"]) for row in stellarator_rows]
    linear = [float(row["linear_weight_prediction"]) for row in stellarator_rows]
    positive_ml = [max(float(row["positive_mixing_length_prediction"]), 0.03) for row in stellarator_rows]
    qa_status = report["stellarator_status"]["QA"]
    observed += [float(qa_status["optimized_heat_flux"]), np.nan]
    spectral += [np.nan, np.nan]
    linear += [np.nan, np.nan]
    positive_ml += [np.nan, np.nan]

    ax1.bar(x - 1.5 * width, observed, width, color="#0f172a", label="observed nonlinear")
    ax1.bar(x - 0.5 * width, spectral, width, color=MODEL_COLORS["spectral_envelope_ridge"], label="spectral-envelope")
    ax1.bar(x + 0.5 * width, linear, width, color=MODEL_COLORS["linear_weight"], label="linear-weight")
    ax1.bar(x + 1.5 * width, positive_ml, width, color=MODEL_COLORS["positive_mixing_length"], label="positive-growth ML")

    baseline = float(qa_status["baseline_heat_flux"])
    optimized = float(qa_status["optimized_heat_flux"])
    qa_x = x[2]
    ax1.scatter([qa_x - 1.5 * width, qa_x - 1.5 * width], [baseline, optimized], color="#0f172a", s=[60, 80], zorder=5)
    ax1.annotate(
        f"QA nonlinear audit\n{qa_status['relative_reduction']:.0%} reduction",
        (qa_x - 1.5 * width, optimized),
        xytext=(-44, 26),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 0.9},
        fontsize=8.4,
        ha="right",
    )
    qh_x = x[3]
    ax1.text(
        qh_x - 0.10,
        max(observed[:2]) * 0.72,
        "QH excluded\nfailed grid gate",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#7f1d1d",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#fecaca", "alpha": 0.9},
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_ylabel(r"Ion heat flux / model units")
    ax1.set_title("Stellarator coverage and scope")
    ax1.legend(loc="upper right", fontsize=7.8, frameon=True)
    ax1.set_xlim(-0.55, len(labels) - 0.35)
    ax1.set_ylim(0.0, max(baseline, max(spectral[:2]), max(linear[:2])) * 1.35)

    fig.suptitle(title, fontsize=13.8, fontweight="semibold")
    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(report, csv_path)
    return {"png": str(png), "pdf": str(pdf), "json": str(json_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STATIC / "quasilinear_stellarator_usefulness.png")
    parser.add_argument("--title", default="Quasilinear models need nonlinear stellarator calibration")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    paths = write_figure(report, out=args.out, title=args.title, dpi=args.dpi)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
