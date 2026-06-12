#!/usr/bin/env python3
"""Score quasilinear models as screening/ranking proxies for nonlinear heat flux."""

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

MODEL_LABELS = {
    "positive_mixing_length": "positive-growth ML",
    "linear_weight": "linear-weight fit",
    "absolute_growth_mixing_length": "absolute-growth ML",
    "spectral_envelope_ridge": "spectral-envelope ridge",
    "linear_state_ridge": "linear-state ridge",
}

MODEL_COLORS = {
    "positive_mixing_length": "#7f1d1d",
    "linear_weight": "#6b7280",
    "absolute_growth_mixing_length": "#b45309",
    "spectral_envelope_ridge": "#0f766e",
    "linear_state_ridge": "#2563eb",
}

CASE_LABELS = {
    "cyclone_long_window": "Cyclone",
    "cyclone_miller_long_window": "Cyclone Miller",
    "hsx_nonlinear_window": "HSX",
    "w7x_nonlinear_window": "W7-X",
    "dshape_external_vmec_t250_window": "D-shaped VMEC",
    "itermodel_external_vmec_t350_window": "ITERModel VMEC",
    "updown_asym_external_vmec_t450_window": "Up-down VMEC",
    "circular_external_vmec_t450_window": "Circular VMEC",
    "cth_like_external_vmec_t700_high_grid_ensemble": "CTH-like VMEC",
    "shaped_tokamak_pressure_external_vmec_t650_high_grid_window": "Shaped-pressure VMEC",
    "qp_diag_nfp2_m4_final_t250": "QP VMEC",
}

DEFAULT_MODELS = (
    "positive_mixing_length",
    "linear_weight",
    "absolute_growth_mixing_length",
    "spectral_envelope_ridge",
    "linear_state_ridge",
)


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


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _pairwise_order_accuracy(predicted: np.ndarray, observed: np.ndarray) -> tuple[float, int, int]:
    correct = 0
    total = 0
    for i in range(observed.size):
        for j in range(i + 1, observed.size):
            observed_sign = np.sign(observed[i] - observed[j])
            predicted_sign = np.sign(predicted[i] - predicted[j])
            if observed_sign == 0.0 or predicted_sign == 0.0:
                continue
            total += 1
            if observed_sign == predicted_sign:
                correct += 1
    if total == 0:
        return float("nan"), correct, total
    return float(correct / total), correct, total


def _relative_errors(predicted: np.ndarray, observed: np.ndarray, floor: float) -> np.ndarray:
    return np.abs(predicted - observed) / np.maximum(np.abs(observed), floor)


def _candidate_rows(candidate_report: dict[str, Any], candidate: str) -> dict[str, dict[str, Any]]:
    payload = candidate_report["candidates"][candidate]
    return {str(row["holdout_case"]): row for row in payload["rows"]}


def build_report(
    *,
    saturation_report: Path = STATIC / "quasilinear_saturation_rule_sweep.json",
    candidate_report: Path = STATIC / "quasilinear_candidate_uncertainty.json",
    observed_floor: float = 1.0e-12,
    log_floor: float = 1.0e-3,
    spearman_gate: float = 0.75,
    pairwise_order_gate: float = 0.75,
    absolute_error_gate: float = 0.35,
) -> dict[str, Any]:
    """Build screening/ranking skill metrics from existing QL/nonlinear reports."""

    sat = _load_json(saturation_report)
    cand = _load_json(candidate_report)
    cases = list(sat["cases"])
    case_names = [str(row["case"]) for row in cases]
    observed = np.asarray([float(row["observed_heat_flux"]) for row in cases], dtype=float)
    holdout_mask = np.asarray([str(row["split"]) == "holdout" for row in cases], dtype=bool)
    predictions: dict[str, np.ndarray] = {
        rule: np.asarray(sat["rules"][rule]["predicted_heat_flux"], dtype=float)
        for rule in ("positive_mixing_length", "linear_weight", "absolute_growth_mixing_length")
    }
    for candidate in ("linear_weight", "spectral_envelope_ridge", "linear_state_ridge"):
        rows = _candidate_rows(cand, candidate)
        predictions[candidate] = np.asarray(
            [float(rows[name]["predicted_heat_flux"]) for name in case_names],
            dtype=float,
        )

    model_rows: list[dict[str, Any]] = []
    for model in DEFAULT_MODELS:
        predicted = predictions[model]
        relative_error = _relative_errors(predicted, observed, observed_floor)
        holdout_relative_error = relative_error[holdout_mask]
        pairwise_accuracy, pairwise_correct, pairwise_total = _pairwise_order_accuracy(predicted, observed)
        holdout_pairwise_accuracy, holdout_correct, holdout_total = _pairwise_order_accuracy(
            predicted[holdout_mask], observed[holdout_mask]
        )
        spearman = _spearman(predicted, observed)
        holdout_spearman = _spearman(predicted[holdout_mask], observed[holdout_mask])
        log_pearson = _pearson(np.log(np.maximum(predicted, log_floor)), np.log(np.maximum(observed, log_floor)))
        holdout_log_pearson = _pearson(
            np.log(np.maximum(predicted[holdout_mask], log_floor)),
            np.log(np.maximum(observed[holdout_mask], log_floor)),
        )
        absolute_gate_passed = bool(float(np.nanmean(relative_error)) <= absolute_error_gate)
        screening_gate_passed = bool(
            math.isfinite(spearman)
            and spearman >= spearman_gate
            and math.isfinite(pairwise_accuracy)
            and pairwise_accuracy >= pairwise_order_gate
        )
        holdout_screening_gate_passed = bool(
            math.isfinite(holdout_spearman)
            and holdout_spearman >= spearman_gate
            and math.isfinite(holdout_pairwise_accuracy)
            and holdout_pairwise_accuracy >= pairwise_order_gate
        )
        model_rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "mean_abs_relative_error": float(np.nanmean(relative_error)),
                "holdout_mean_abs_relative_error": float(np.nanmean(holdout_relative_error)),
                "log_pearson": log_pearson,
                "holdout_log_pearson": holdout_log_pearson,
                "spearman": spearman,
                "holdout_spearman": holdout_spearman,
                "pairwise_order_accuracy": pairwise_accuracy,
                "pairwise_order_correct": pairwise_correct,
                "pairwise_order_total": pairwise_total,
                "holdout_pairwise_order_accuracy": holdout_pairwise_accuracy,
                "holdout_pairwise_order_correct": holdout_correct,
                "holdout_pairwise_order_total": holdout_total,
                "absolute_flux_gate_passed": absolute_gate_passed,
                "screening_gate_passed": screening_gate_passed,
                "holdout_screening_gate_passed": holdout_screening_gate_passed,
            }
        )

    accepted_screening = [row["model"] for row in model_rows if bool(row["screening_gate_passed"])]
    accepted_holdout_screening = [
        row["model"] for row in model_rows if bool(row["holdout_screening_gate_passed"])
    ]
    mean_error_gate_models = [row["model"] for row in model_rows if bool(row["absolute_flux_gate_passed"])]
    best_screening = max(
        model_rows,
        key=lambda row: (
            float("-inf") if row["spearman"] is None else float(row["spearman"]),
            float("-inf") if row["pairwise_order_accuracy"] is None else float(row["pairwise_order_accuracy"]),
            -float(row["mean_abs_relative_error"]),
        ),
    )
    best_holdout_screening = max(
        model_rows,
        key=lambda row: (
            float("-inf") if row["holdout_spearman"] is None else float(row["holdout_spearman"]),
            float("-inf")
            if row["holdout_pairwise_order_accuracy"] is None
            else float(row["holdout_pairwise_order_accuracy"]),
            -float(row["holdout_mean_abs_relative_error"]),
        ),
    )
    case_rows = []
    for idx, case in enumerate(cases):
        row: dict[str, Any] = {
            "case": case_names[idx],
            "label": CASE_LABELS.get(case_names[idx], case_names[idx].replace("_", " ")),
            "split": str(case["split"]),
            "geometry": str(case["geometry"]),
            "observed_heat_flux": float(observed[idx]),
        }
        for model in DEFAULT_MODELS:
            row[f"{model}_prediction"] = float(predictions[model][idx])
        case_rows.append(row)

    return {
        "kind": "quasilinear_screening_skill",
        "claim_level": "screening_correlation_model_development_not_absolute_flux_promotion",
        "source_artifacts": {
            "saturation_rule_sweep": str(saturation_report.relative_to(ROOT)),
            "candidate_uncertainty": str(candidate_report.relative_to(ROOT)),
        },
        "gates": {
            "absolute_error_gate": absolute_error_gate,
            "spearman_gate": spearman_gate,
            "pairwise_order_gate": pairwise_order_gate,
            "mean_error_gate_models": mean_error_gate_models,
            "accepted_absolute_flux_models": [],
            "accepted_screening_models": accepted_screening,
            "accepted_holdout_screening_models": accepted_holdout_screening,
            "absolute_flux_promotion_passed": False,
            "screening_correlation_passed": bool(accepted_screening),
            "holdout_screening_correlation_passed": bool(accepted_holdout_screening),
            "best_screening_model": best_screening["model"],
            "best_holdout_screening_model": best_holdout_screening["model"],
        },
        "models": model_rows,
        "cases": case_rows,
        "notes": [
            "Screening gates are ranking/correlation diagnostics, not absolute heat-flux promotion gates.",
            "Held-out-only screening is reported separately and remains below gate until more independent nonlinear holdouts are admitted.",
            "All metrics are computed from tracked nonlinear-window and quasilinear model-selection artifacts.",
            "A model may pass screening or mean-error gates while still failing universal absolute-flux promotion requirements.",
        ],
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = list(report["models"][0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["models"])


def write_figure(report: dict[str, Any], *, out: Path, title: str, dpi: int = 220) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    model_rows = list(report["models"])
    case_rows = list(report["cases"])
    best = report["gates"]["best_screening_model"]
    observed = np.asarray([row["observed_heat_flux"] for row in case_rows], dtype=float)
    best_pred = np.asarray([row[f"{best}_prediction"] for row in case_rows], dtype=float)
    linear_pred = np.asarray([row["linear_weight_prediction"] for row in case_rows], dtype=float)
    simple_pred = np.asarray([row["positive_mixing_length_prediction"] for row in case_rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), constrained_layout=True)
    ax0, ax1, ax2 = axes
    floor = 1.0e-3
    positive = np.concatenate([observed[observed > 0.0], best_pred[best_pred > 0.0], linear_pred[linear_pred > 0.0], simple_pred[simple_pred > 0.0]])
    lo = max(float(np.min(positive)) * 0.35, floor)
    hi = float(np.max(positive)) * 2.2
    ax0.plot([lo, hi], [lo, hi], linestyle="--", color="0.2", linewidth=1.3, label="1:1")
    ax0.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2], color="0.86", alpha=0.55, label="factor 2")
    ax0.scatter(observed, np.maximum(best_pred, floor), s=75, color=MODEL_COLORS[best], edgecolor="black", label=MODEL_LABELS[best], zorder=4)
    ax0.scatter(observed, np.maximum(linear_pred, floor), s=54, color=MODEL_COLORS["linear_weight"], edgecolor="white", label=MODEL_LABELS["linear_weight"], zorder=3)
    ax0.scatter(observed, np.maximum(simple_pred, floor), s=65, color=MODEL_COLORS["positive_mixing_length"], marker="x", label=MODEL_LABELS["positive_mixing_length"], zorder=3)
    label_offsets = {"hsx_nonlinear_window": (9, 12), "w7x_nonlinear_window": (9, -15)}
    for row, pred in zip(case_rows, best_pred, strict=True):
        if row["case"] in label_offsets:
            ax0.annotate(
                row["label"],
                (row["observed_heat_flux"], max(pred, floor)),
                xytext=label_offsets[str(row["case"])],
                textcoords="offset points",
                fontsize=8.3,
                arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.7},
            )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel(r"Observed nonlinear late-window $\langle Q_i\rangle$")
    ax0.set_ylabel("Model prediction")
    ax0.set_title("Best screened model vs nonlinear windows")
    ax0.legend(loc="upper left", fontsize=8.1, frameon=True)

    labels = [row["label"] for row in model_rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.26
    spearman = np.asarray([row["spearman"] for row in model_rows], dtype=float)
    pairwise = np.asarray([row["pairwise_order_accuracy"] for row in model_rows], dtype=float)
    mare = np.asarray([row["mean_abs_relative_error"] for row in model_rows], dtype=float)
    absolute_skill = 1.0 - np.minimum(mare, 1.5) / 1.5
    ax1.axhline(report["gates"]["spearman_gate"], color="#0f766e", linestyle="--", linewidth=1.1, label="rank gate")
    ax1.bar(x - width, spearman, width, color="#0f766e", label="Spearman")
    ax1.bar(x, pairwise, width, color="#2563eb", label="pairwise order")
    ax1.bar(x + width, absolute_skill, width, color="#9ca3af", label="absolute skill")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylim(-0.55, 1.05)
    ax1.set_ylabel("Screening metric")
    ax1.set_title("Full-portfolio screening")
    ax1.legend(loc="lower right", fontsize=8.0, frameon=True)
    ax1.text(
        0.02,
        0.05,
        f"screening gate: {', '.join(report['gates']['accepted_screening_models']) or 'none'}\n"
        f"mean-error gate: {', '.join(report['gates']['mean_error_gate_models']) or 'none'}\n"
        "absolute promotion: none",
        transform=ax1.transAxes,
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.92},
    )

    holdout_spearman = np.asarray([row["holdout_spearman"] for row in model_rows], dtype=float)
    holdout_pairwise = np.asarray([row["holdout_pairwise_order_accuracy"] for row in model_rows], dtype=float)
    holdout_mare = np.asarray([row["holdout_mean_abs_relative_error"] for row in model_rows], dtype=float)
    holdout_absolute_skill = 1.0 - np.minimum(holdout_mare, 1.5) / 1.5
    ax2.axhline(report["gates"]["spearman_gate"], color="#0f766e", linestyle="--", linewidth=1.1, label="rank gate")
    ax2.bar(x - width, holdout_spearman, width, color="#0f766e", label="holdout Spearman")
    ax2.bar(x, holdout_pairwise, width, color="#2563eb", label="holdout pairwise")
    ax2.bar(x + width, holdout_absolute_skill, width, color="#9ca3af", label="holdout abs. skill")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.set_ylim(-0.65, 1.05)
    ax2.set_ylabel("Held-out metric")
    ax2.set_title("Held-out-only promotion check")
    ax2.legend(loc="lower right", fontsize=8.0, frameon=True)
    ax2.text(
        0.02,
        0.05,
        f"heldout screening gate: {', '.join(report['gates']['accepted_holdout_screening_models']) or 'none'}\n"
        f"best heldout model: {MODEL_LABELS[report['gates']['best_holdout_screening_model']]}\n"
        "needs more nonlinear holdouts",
        transform=ax2.transAxes,
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.92},
    )
    fig.suptitle(title, fontsize=13.6, fontweight="semibold")

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
    parser.add_argument("--out", type=Path, default=STATIC / "quasilinear_screening_skill.png")
    parser.add_argument("--title", default="Quasilinear screening skill needs nonlinear calibration")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    paths = write_figure(report, out=args.out, title=args.title, dpi=args.dpi)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
