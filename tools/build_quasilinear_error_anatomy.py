#!/usr/bin/env python3
"""Build a fail-closed error-anatomy artifact for quasilinear model candidates."""

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
DEFAULT_UNCERTAINTY = ROOT / "docs/_static/quasilinear_candidate_uncertainty.json"
DEFAULT_SCREENING = ROOT / "docs/_static/quasilinear_screening_skill.json"
DEFAULT_SATURATION = ROOT / "docs/_static/quasilinear_saturation_rule_sweep.json"
DEFAULT_OUT = ROOT / "docs/_static/quasilinear_error_anatomy.png"

SHORT_LABELS = {
    "cyclone_long_window": "Cyclone",
    "cyclone_miller_long_window": "Cyclone Miller",
    "hsx_nonlinear_window": "HSX",
    "w7x_nonlinear_window": "W7-X",
    "dshape_external_vmec_t250_window": "D-shaped VMEC",
    "itermodel_external_vmec_t350_window": "ITERModel VMEC",
    "updown_asym_external_vmec_t450_window": "Up-down VMEC",
    "circular_external_vmec_t450_window": "Circular VMEC",
    "cth_like_external_vmec_t700_high_grid_ensemble": "CTH-like VMEC",
    "shaped_tokamak_pressure_external_vmec_t650_high_grid_window": "Shaped-pressure",
}


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _geometry_group(geometry: str) -> str:
    text = geometry.lower()
    if text in {"cyclone", "cyclone_miller"}:
        return "local axisymmetric"
    if text in {"hsx", "w7x"}:
        return "stellarator benchmark"
    if "cth" in text or "qh" in text or "qi" in text:
        return "external stellarator VMEC"
    if text.endswith("external_vmec"):
        return "external axisymmetric VMEC"
    return "other"


def _case_metadata(
    screening: dict[str, Any],
    saturation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for item in saturation.get("cases", []):
        if not isinstance(item, dict):
            continue
        case = str(item.get("case", ""))
        if case:
            metadata[case] = {
                "case": case,
                "geometry": str(item.get("geometry", "")),
                "split": str(item.get("split", "")),
                "observed_heat_flux": _finite_float(item.get("observed_heat_flux")),
                "observed_heat_flux_std": _finite_float(item.get("observed_heat_flux_std")),
                "shape_passed": bool(item.get("shape_passed", False)),
            }
    for item in screening.get("cases", []):
        if not isinstance(item, dict):
            continue
        case = str(item.get("case", ""))
        if case:
            metadata.setdefault(case, {"case": case})
            metadata[case].update(
                {
                    "geometry": str(item.get("geometry", metadata[case].get("geometry", ""))),
                    "split": str(item.get("split", metadata[case].get("split", ""))),
                    "label": str(item.get("label", SHORT_LABELS.get(case, case))),
                }
            )
    return metadata


def _model_metrics(screening: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in screening.get("models", []):
        if not isinstance(model, dict):
            continue
        rows.append(
            {
                "model": str(model.get("model", "")),
                "label": str(model.get("label", model.get("model", ""))),
                "mean_abs_relative_error": _finite_float(model.get("mean_abs_relative_error")),
                "holdout_mean_abs_relative_error": _finite_float(
                    model.get("holdout_mean_abs_relative_error")
                ),
                "spearman": _finite_float(model.get("spearman")),
                "holdout_spearman": _finite_float(model.get("holdout_spearman")),
                "pairwise_order_accuracy": _finite_float(model.get("pairwise_order_accuracy")),
                "holdout_pairwise_order_accuracy": _finite_float(
                    model.get("holdout_pairwise_order_accuracy")
                ),
                "absolute_flux_gate_passed": bool(model.get("absolute_flux_gate_passed", False)),
                "screening_gate_passed": bool(model.get("screening_gate_passed", False)),
                "holdout_screening_gate_passed": bool(
                    model.get("holdout_screening_gate_passed", False)
                ),
            }
        )
    return rows


def build_error_anatomy_report(
    *,
    candidate_uncertainty: str | Path = DEFAULT_UNCERTAINTY,
    screening_skill: str | Path = DEFAULT_SCREENING,
    saturation_sweep: str | Path = DEFAULT_SATURATION,
    candidate: str = "spectral_envelope_ridge",
    transport_gate: float | None = None,
) -> dict[str, Any]:
    """Return per-case residual anatomy for the current QL candidate."""

    uncertainty = _read_json(candidate_uncertainty)
    screening = _read_json(screening_skill)
    saturation = _read_json(saturation_sweep)
    candidates = uncertainty.get("candidates", {})
    if not isinstance(candidates, dict) or candidate not in candidates:
        raise ValueError(f"candidate {candidate!r} not found in {candidate_uncertainty}")
    candidate_payload = candidates[candidate]
    if not isinstance(candidate_payload, dict):
        raise ValueError(f"candidate {candidate!r} payload is not an object")
    rows_in = candidate_payload.get("rows", [])
    if not isinstance(rows_in, list) or not rows_in:
        raise ValueError(f"candidate {candidate!r} has no rows")

    gate = (
        float(transport_gate)
        if transport_gate is not None
        else _finite_float(uncertainty.get("transport_gate"), default=0.35)
    )
    metadata = _case_metadata(screening, saturation)
    rows: list[dict[str, Any]] = []
    for item in rows_in:
        if not isinstance(item, dict):
            continue
        case = str(item.get("holdout_case", ""))
        meta = metadata.get(case, {"case": case})
        observed = _finite_float(item.get("observed_heat_flux"))
        predicted = _finite_float(item.get("predicted_heat_flux"))
        rel_error = _finite_float(item.get("absolute_relative_error"))
        ratio = predicted / observed if observed and math.isfinite(observed) else float("nan")
        geometry = str(meta.get("geometry", ""))
        row = {
            "case": case,
            "label": str(meta.get("label", SHORT_LABELS.get(case, case))),
            "split": str(meta.get("split", "")),
            "geometry": geometry,
            "geometry_group": _geometry_group(geometry),
            "observed_heat_flux": observed,
            "predicted_heat_flux": predicted,
            "prediction_to_observed_ratio": ratio,
            "signed_relative_error": (predicted - observed) / observed
            if observed and math.isfinite(observed)
            else float("nan"),
            "absolute_relative_error": rel_error,
            "above_transport_gate": bool(math.isfinite(rel_error) and rel_error > gate),
            "overpredicts": bool(math.isfinite(ratio) and ratio > 1.0),
            "prediction_interval_contains_observed": bool(
                item.get("prediction_interval_contains_observed", False)
            ),
            "prediction_interval_low": _finite_float(item.get("prediction_interval_low")),
            "prediction_interval_high": _finite_float(item.get("prediction_interval_high")),
        }
        rows.append(row)
    rows.sort(key=lambda row: float(row["absolute_relative_error"]), reverse=True)
    finite_errors = np.asarray(
        [row["absolute_relative_error"] for row in rows if math.isfinite(float(row["absolute_relative_error"]))],
        dtype=float,
    )
    total_error = float(np.sum(finite_errors)) if finite_errors.size else float("nan")
    for row in rows:
        err = float(row["absolute_relative_error"])
        row["error_budget_fraction"] = err / total_error if total_error > 0.0 else float("nan")

    group_rows: list[dict[str, Any]] = []
    for group in sorted({str(row["geometry_group"]) for row in rows}):
        group_errors = np.asarray(
            [
                float(row["absolute_relative_error"])
                for row in rows
                if row["geometry_group"] == group
                and math.isfinite(float(row["absolute_relative_error"]))
            ],
            dtype=float,
        )
        if not group_errors.size:
            continue
        group_rows.append(
            {
                "geometry_group": group,
                "n": int(group_errors.size),
                "mean_abs_relative_error": float(np.mean(group_errors)),
                "max_abs_relative_error": float(np.max(group_errors)),
                "error_budget_fraction": float(np.sum(group_errors) / total_error)
                if total_error > 0.0
                else float("nan"),
            }
        )
    model_rows = _model_metrics(screening)
    model_rows.sort(key=lambda row: float(row["mean_abs_relative_error"]))
    screening_gates = screening.get("gates", {}) if isinstance(screening.get("gates", {}), dict) else {}
    promotion_gate = (
        uncertainty.get("promotion_gate", {})
        if isinstance(uncertainty.get("promotion_gate", {}), dict)
        else {}
    )
    blockers: list[str] = []
    if bool(promotion_gate.get("passed", False)) is False:
        blockers.append("candidate_uncertainty_promotion_gate_failed")
    if not screening_gates.get("screening_correlation_passed", False):
        blockers.append("full_portfolio_screening_correlation_gate_failed")
    if not screening_gates.get("holdout_screening_correlation_passed", False):
        blockers.append("heldout_screening_correlation_gate_failed")
    if any(bool(row["above_transport_gate"]) for row in rows):
        blockers.append("case_residuals_exceed_transport_gate")
    return {
        "kind": "quasilinear_error_anatomy",
        "claim_level": "model_development_residual_anatomy_not_absolute_flux_promotion",
        "candidate": candidate,
        "transport_gate": gate,
        "case_count": len(rows),
        "holdout_count": sum(1 for row in rows if row["split"] == "holdout"),
        "candidate_mean_abs_relative_error": _finite_float(
            candidate_payload.get("mean_abs_relative_error")
        ),
        "candidate_prediction_interval_coverage": _finite_float(
            candidate_payload.get("prediction_interval_coverage")
        ),
        "rows": rows,
        "geometry_group_summary": group_rows,
        "model_summary": model_rows,
        "promotion_gate": {
            "passed": False,
            "blockers": blockers,
            "claim_boundary": (
                "This artifact diagnoses why quasilinear model candidates fail or pass. "
                "It does not promote a runtime/TOML absolute-flux predictor."
            ),
        },
        "source_artifacts": {
            "candidate_uncertainty": str(Path(candidate_uncertainty)),
            "screening_skill": str(Path(screening_skill)),
            "saturation_sweep": str(Path(saturation_sweep)),
        },
    }


def write_error_anatomy_figure(
    report: dict[str, Any],
    *,
    out: str | Path = DEFAULT_OUT,
    title: str = "Quasilinear residual anatomy",
    dpi: int = 190,
    write_pdf: bool = False,
) -> dict[str, str]:
    """Write PNG/CSV/JSON artifacts for a QL error-anatomy report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(report["rows"])
    if not rows:
        raise ValueError("report contains no rows")
    gate = float(report["transport_gate"])

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
    ax0, ax1 = axes

    observed = np.asarray([row["observed_heat_flux"] for row in rows], dtype=float)
    predicted = np.asarray([row["predicted_heat_flux"] for row in rows], dtype=float)
    positive = np.concatenate([observed[observed > 0.0], predicted[predicted > 0.0]])
    lim_lo = float(np.min(positive)) * 0.55
    lim_hi = float(np.max(positive)) * 1.65
    colors = ["#b91c1c" if row["overpredicts"] else "#1d4ed8" for row in rows]
    ax0.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="0.25", linestyle="--", linewidth=1.3)
    ax0.scatter(observed, predicted, c=colors, s=52, edgecolor="white", linewidth=0.7, zorder=3)
    for row in rows[:5]:
        ax0.annotate(
            str(row["label"]),
            (float(row["observed_heat_flux"]), float(row["predicted_heat_flux"])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "0.82", "alpha": 0.82},
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.6},
        )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lim_lo, lim_hi)
    ax0.set_ylim(lim_lo, lim_hi)
    ax0.set_xlabel("observed nonlinear heat flux")
    ax0.set_ylabel("leave-one-out QL prediction")
    ax0.set_title("Prediction residuals")
    ax0.grid(True, which="both", alpha=0.24)
    ax0.text(
        0.03,
        0.97,
        "red: overpredicts\nblue: underpredicts",
        transform=ax0.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.78", "alpha": 0.85},
    )

    labels = [str(row["label"]) for row in rows]
    errors = np.asarray([row["absolute_relative_error"] for row in rows], dtype=float)
    ypos = np.arange(len(rows))
    ax1.barh(ypos, errors, color=colors, alpha=0.88)
    ax1.axvline(gate, color="#111827", linestyle="--", linewidth=1.2, label=f"{gate:.2f} gate")
    ax1.set_yticks(ypos, labels)
    ax1.invert_yaxis()
    ax1.set_xscale("log")
    ax1.set_xlabel("absolute relative error")
    ax1.set_title("Error budget by case")
    ax1.grid(True, axis="x", alpha=0.24)
    for y, row in zip(ypos, rows, strict=True):
        ax1.text(
            float(row["absolute_relative_error"]) * 1.06,
            y,
            f"{100.0 * float(row['error_budget_fraction']):.0f}%",
            va="center",
            fontsize=7.5,
        )
    ax1.legend(frameon=False, loc="lower right")
    subtitle = (
        f"{report['candidate']} | mean error = "
        f"{float(report['candidate_mean_abs_relative_error']):.3f} | "
        "absolute-flux promotion = FAIL"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13.5, fontweight="bold")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    outputs = {"png": str(out_path)}
    if write_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs["pdf"] = str(pdf_path)
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "case",
            "label",
            "split",
            "geometry",
            "geometry_group",
            "observed_heat_flux",
            "predicted_heat_flux",
            "prediction_to_observed_ratio",
            "signed_relative_error",
            "absolute_relative_error",
            "error_budget_fraction",
            "above_transport_gate",
            "prediction_interval_contains_observed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    outputs.update({"json": str(json_path), "csv": str(csv_path)})
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-uncertainty", type=Path, default=DEFAULT_UNCERTAINTY)
    parser.add_argument("--screening-skill", type=Path, default=DEFAULT_SCREENING)
    parser.add_argument("--saturation-sweep", type=Path, default=DEFAULT_SATURATION)
    parser.add_argument("--candidate", default="spectral_envelope_ridge")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default="Quasilinear residual anatomy")
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument("--write-pdf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_error_anatomy_report(
        candidate_uncertainty=args.candidate_uncertainty,
        screening_skill=args.screening_skill,
        saturation_sweep=args.saturation_sweep,
        candidate=str(args.candidate),
    )
    outputs = write_error_anatomy_figure(
        report,
        out=args.out,
        title=str(args.title),
        dpi=int(args.dpi),
        write_pdf=bool(args.write_pdf),
    )
    print(f"saved {outputs['png']}")
    print(f"saved {outputs['json']}")
    print(f"saved {outputs['csv']}")
    print(
        "promotion_passed={passed} blockers={blockers}".format(
            passed=report["promotion_gate"]["passed"],
            blockers=",".join(report["promotion_gate"]["blockers"]) or "none",
        )
    )
    return 0 if report["promotion_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
