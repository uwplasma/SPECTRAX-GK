#!/usr/bin/env python3
"""Compare simple quasilinear saturation/intensity rules across held-out cases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import sys

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.quasilinear_calibration import calibration_point_from_nonlinear_window_summary  # noqa: E402
from spectraxgk.quasilinear import saturation_amplitude2  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SaturationCase:
    """One nonlinear-window target used in the saturation-rule sweep."""

    case: str
    split: str
    geometry: str
    spectrum: Path
    nonlinear_summary: Path
    shape_gate: Path | None = None


DEFAULT_CASES = (
    SaturationCase(
        case="cyclone_long_window",
        split="train",
        geometry="cyclone",
        spectrum=ROOT / "docs/_static/quasilinear_cyclone_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT / "docs/_static/nonlinear_cyclone_gate_summary.json",
        shape_gate=ROOT / "docs/_static/quasilinear_cyclone_spectrum_shape_gate.json",
    ),
    SaturationCase(
        case="cyclone_miller_long_window",
        split="holdout",
        geometry="cyclone_miller",
        spectrum=ROOT / "docs/_static/quasilinear_cyclone_miller_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT / "docs/_static/nonlinear_cyclone_miller_gate_summary.json",
        shape_gate=ROOT / "docs/_static/quasilinear_cyclone_miller_spectrum_shape_gate.json",
    ),
    SaturationCase(
        case="hsx_nonlinear_window",
        split="holdout",
        geometry="hsx",
        spectrum=ROOT / "docs/_static/quasilinear_hsx_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT / "docs/_static/nonlinear_hsx_gate_summary.json",
        shape_gate=ROOT / "docs/_static/quasilinear_hsx_spectrum_shape_gate.json",
    ),
    SaturationCase(
        case="w7x_nonlinear_window",
        split="holdout",
        geometry="w7x",
        spectrum=ROOT / "docs/_static/quasilinear_w7x_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT / "docs/_static/nonlinear_w7x_gate_summary.json",
        shape_gate=ROOT / "docs/_static/quasilinear_w7x_spectrum_shape_gate.json",
    ),
)


RULE_LABELS = {
    "positive_mixing_length": r"$\max(\gamma,0)\,\hat Q/k_\perp^2$",
    "linear_weight": r"$\hat Q$",
    "absolute_growth_mixing_length": r"$|\gamma|\,\hat Q/k_\perp^2$",
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


def _load_table(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    return data


def _required_column(data: np.ndarray, path: Path, column: str) -> np.ndarray:
    names = set(data.dtype.names or ())
    if column not in names:
        raise ValueError(f"{path} is missing required column '{column}'")
    return np.asarray(data[column], dtype=float)


def raw_rule_estimates(spectrum_csv: str | Path, *, floor: float = 1.0e-300) -> dict[str, float]:
    """Return unscaled integrated flux estimates for simple saturation rules."""

    path = Path(spectrum_csv)
    data = _load_table(path)
    gamma = _required_column(data, path, "gamma")
    kperp2 = np.maximum(_required_column(data, path, "kperp_eff2"), float(floor))
    weight = _required_column(data, path, "heat_flux_weight_total")
    finite = np.isfinite(gamma) & np.isfinite(kperp2) & np.isfinite(weight)
    if not np.any(finite):
        raise ValueError(f"{path} contains no finite quasilinear samples")
    gamma = gamma[finite]
    kperp2 = kperp2[finite]
    weight = weight[finite]
    amplitudes = {
        "positive_mixing_length": np.asarray(
            [
                saturation_amplitude2(gamma=float(g), kperp_eff2_value=float(k2), rule="mixing_length")
                for g, k2 in zip(gamma, kperp2, strict=True)
            ],
            dtype=float,
        ),
        "linear_weight": np.asarray(
            [
                saturation_amplitude2(gamma=float(g), kperp_eff2_value=float(k2), rule="linear_weight")
                for g, k2 in zip(gamma, kperp2, strict=True)
            ],
            dtype=float,
        ),
        "absolute_growth_mixing_length": np.asarray(
            [
                saturation_amplitude2(
                    gamma=float(g),
                    kperp_eff2_value=float(k2),
                    rule="absolute_growth_mixing_length",
                )
                for g, k2 in zip(gamma, kperp2, strict=True)
            ],
            dtype=float,
        ),
    }
    return {
        rule: float(np.sum(np.maximum(weight, 0.0) * np.maximum(amp, 0.0)))
        for rule, amp in amplitudes.items()
    }


def _fit_scale(raw: np.ndarray, observed: np.ndarray, mask: np.ndarray, *, floor: float) -> float:
    x = np.asarray(raw[mask], dtype=float)
    y = np.asarray(observed[mask], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > floor)
    if not np.any(finite):
        return float("nan")
    denom = float(np.dot(x[finite], x[finite]))
    if denom <= floor:
        return float("nan")
    return float(np.dot(x[finite], y[finite]) / denom)


def _shape_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "shape_passed": bool(data.get("passed", False)),
        "shape_tv": data.get("total_variation_distance"),
        "shape_cosine": data.get("cosine_similarity"),
        "shape_gate": str(path),
    }


def _summary_gate_passed(payload: dict[str, Any]) -> bool:
    if isinstance(payload.get("gate_report"), dict):
        return bool(payload["gate_report"].get("passed", False))
    if isinstance(payload.get("promotion_gate"), dict):
        return bool(payload["promotion_gate"].get("passed", False))
    if "gate_passed" in payload:
        return bool(payload.get("gate_passed"))
    return False


def nonlinear_input_validation_report(
    cases: tuple[SaturationCase, ...],
    *,
    required_splits: tuple[str, ...] = ("train", "holdout"),
) -> dict[str, Any]:
    """Return gate provenance for nonlinear summaries used by model diagnostics."""

    rows = []
    passed = True
    for case in cases:
        path = Path(case.nonlinear_summary)
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = case.split in required_splits
        gate_passed = _summary_gate_passed(payload)
        row_passed = (not required) or gate_passed
        if not row_passed:
            passed = False
        gate_report = payload.get("gate_report") if isinstance(payload.get("gate_report"), dict) else {}
        rows.append(
            {
                "case": case.case,
                "split": case.split,
                "required": required,
                "nonlinear_summary": str(path),
                "passed": row_passed,
                "gate_passed": gate_passed,
                "gate_case": str(gate_report.get("case", payload.get("case", path.stem))),
                "reason": "matched passed nonlinear summary gate"
                if row_passed and required
                else ("not required split" if row_passed else "nonlinear summary gate is missing or failed"),
            }
        )
    return {
        "kind": "quasilinear_model_input_validation",
        "passed": passed,
        "required_splits": list(required_splits),
        "cases": rows,
    }


def require_validated_nonlinear_inputs(cases: tuple[SaturationCase, ...]) -> dict[str, Any]:
    """Raise if any train/holdout case lacks a passed nonlinear summary gate."""

    report = nonlinear_input_validation_report(cases)
    if not bool(report["passed"]):
        failed = [row["case"] for row in report["cases"] if not bool(row["passed"])]
        raise ValueError(f"unvalidated nonlinear train/holdout input(s): {', '.join(failed)}")
    return report


def build_saturation_rule_sweep(
    cases: tuple[SaturationCase, ...] = DEFAULT_CASES,
    *,
    observed_floor: float = 1.0e-12,
    require_validated_inputs: bool = True,
) -> dict[str, Any]:
    """Fit one scalar per rule on train cases and score all cases."""

    input_validation = (
        require_validated_nonlinear_inputs(cases)
        if require_validated_inputs
        else {"kind": "quasilinear_model_input_validation", "passed": None, "required": False}
    )
    case_rows = []
    rule_names = tuple(RULE_LABELS)
    for case in cases:
        observed_point = calibration_point_from_nonlinear_window_summary(
            case.nonlinear_summary,
            predicted_heat_flux=1.0,
            split=case.split,
            saturation_rule="diagnostic_only",
            geometry=case.geometry,
            electron_model="adiabatic",
            quasilinear_artifact=str(case.spectrum),
        )
        raw = raw_rule_estimates(case.spectrum)
        case_rows.append(
            {
                "case": case.case,
                "split": case.split,
                "geometry": case.geometry,
                "spectrum": str(case.spectrum),
                "nonlinear_summary": str(case.nonlinear_summary),
                "observed_heat_flux": observed_point.observed_heat_flux,
                "observed_heat_flux_std": observed_point.observed_heat_flux_std,
                "raw_estimates": raw,
                **_shape_payload(case.shape_gate),
            }
        )

    observed = np.asarray([row["observed_heat_flux"] for row in case_rows], dtype=float)
    train_mask = np.asarray([row["split"] == "train" for row in case_rows], dtype=bool)
    if not np.any(train_mask):
        raise ValueError("at least one train case is required to fit saturation-rule scales")

    rules = {}
    for rule in rule_names:
        raw = np.asarray([row["raw_estimates"][rule] for row in case_rows], dtype=float)
        scale = _fit_scale(raw, observed, train_mask, floor=observed_floor)
        predicted = raw * scale if np.isfinite(scale) else np.full_like(raw, np.nan)
        rel_error = np.abs(predicted - observed) / np.maximum(np.abs(observed), observed_floor)
        holdout = rel_error[~train_mask]
        rules[rule] = {
            "label": RULE_LABELS[rule],
            "scale": float(scale),
            "predicted_heat_flux": predicted.tolist(),
            "absolute_relative_error": rel_error.tolist(),
            "holdout_mean_abs_relative_error": None if holdout.size == 0 else float(np.nanmean(holdout)),
            "holdout_max_abs_relative_error": None if holdout.size == 0 else float(np.nanmax(holdout)),
        }

    null_predicted = np.full_like(observed, float(np.nanmean(observed[train_mask])))
    null_rel_error = np.abs(null_predicted - observed) / np.maximum(np.abs(observed), observed_floor)
    null_holdout = null_rel_error[~train_mask]
    null_holdout_mean = None if null_holdout.size == 0 else float(np.nanmean(null_holdout))
    transport_gate = 0.35
    accepted_rules = []
    for rule, payload in rules.items():
        mean_error = payload["holdout_mean_abs_relative_error"]
        if mean_error is None:
            continue
        beats_null = null_holdout_mean is None or float(mean_error) < float(null_holdout_mean)
        if float(mean_error) <= transport_gate and beats_null:
            accepted_rules.append(rule)

    return {
        "kind": "quasilinear_saturation_rule_sweep",
        "claim_level": "model_comparison_not_validated_transport",
        "observed_floor": float(observed_floor),
        "train_cases": [row["case"] for row in case_rows if row["split"] == "train"],
        "rules": rules,
        "null_training_mean_baseline": {
            "label": "training-mean null",
            "predicted_heat_flux": null_predicted.tolist(),
            "absolute_relative_error": null_rel_error.tolist(),
            "holdout_mean_abs_relative_error": null_holdout_mean,
            "holdout_max_abs_relative_error": None if null_holdout.size == 0 else float(np.nanmax(null_holdout)),
        },
        "input_validation": input_validation,
        "promotion_gate": {
            "passed": bool(accepted_rules),
            "accepted_rules": accepted_rules,
            "transport_mean_relative_error_gate": transport_gate,
            "requires_beating_training_mean_null": True,
            "null_training_mean_holdout_mean_abs_relative_error": null_holdout_mean,
            "best_rule": min(
                rules,
                key=lambda name: float("inf")
                if rules[name]["holdout_mean_abs_relative_error"] is None
                else float(rules[name]["holdout_mean_abs_relative_error"]),
            ),
            "best_rule_holdout_mean_abs_relative_error": min(
                float("inf")
                if payload["holdout_mean_abs_relative_error"] is None
                else float(payload["holdout_mean_abs_relative_error"])
                for payload in rules.values()
            ),
        },
        "cases": case_rows,
        "notes": (
            "One scalar per rule is fitted on train cases only. The sweep is a diagnostic "
            "for saturation-model development, not a validated absolute-flux claim. Candidate "
            "rules should beat the training-mean null baseline before being promoted."
        ),
    }


def write_saturation_rule_sweep_figure(report: dict[str, Any], *, out: str | Path, title: str) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a saturation-rule sweep report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cases = list(report["cases"])
    rules = dict(report["rules"])
    null_baseline = dict(report["null_training_mean_baseline"])
    labels = [str(row["case"]) for row in cases]
    y = np.arange(len(labels))

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    ax0, ax1 = axes
    colors = {
        "positive_mixing_length": "#0f4c81",
        "linear_weight": "#2a9d8f",
        "absolute_growth_mixing_length": "#d1495b",
    }
    offsets = np.linspace(-0.22, 0.22, len(rules))
    floor = 1.0e-3
    max_err = floor
    for offset, (rule, payload) in zip(offsets, rules.items(), strict=True):
        err = np.asarray(payload["absolute_relative_error"], dtype=float)
        plot_err = np.where((err > 0.0) & np.isfinite(err), err, floor)
        max_err = max(max_err, float(np.nanmax(plot_err)))
        ax0.scatter(
            plot_err,
            y + offset,
            s=65,
            color=colors.get(rule, "0.4"),
            label=payload["label"],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    null_err = np.asarray(null_baseline["absolute_relative_error"], dtype=float)
    null_plot_err = np.where((null_err > 0.0) & np.isfinite(null_err), null_err, floor)
    max_err = max(max_err, float(np.nanmax(null_plot_err)))
    ax0.scatter(
        null_plot_err,
        y + 0.34,
        s=64,
        marker="^",
        color="#b45309",
        label=null_baseline["label"],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax0.axvline(0.35, color="#c2410c", linestyle="--", linewidth=1.5, label="0.35 gate")
    ax0.set_xscale("log")
    ax0.set_xlim(floor * 0.7, max(max_err, 0.35) * 1.5)
    ax0.set_yticks(y, labels)
    ax0.invert_yaxis()
    ax0.set_xlabel("absolute relative error after Cyclone scale fit")
    ax0.set_title("Absolute-flux transfer")
    ax0.grid(True, axis="x", alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8)

    shape_tv = np.asarray([np.nan if row.get("shape_tv") is None else float(row["shape_tv"]) for row in cases])
    shape_cos = np.asarray([np.nan if row.get("shape_cosine") is None else float(row["shape_cosine"]) for row in cases])
    x = np.arange(len(labels))
    width = 0.38
    ax1.bar(x - width / 2.0, shape_tv, width=width, label="TV distance", color="#f97316")
    ax1.bar(x + width / 2.0, 1.0 - shape_cos, width=width, label="1 - cosine", color="#2a9d8f")
    ax1.axhline(0.2, color="#f97316", linestyle=":", linewidth=1.2)
    ax1.axhline(0.05, color="#2a9d8f", linestyle=":", linewidth=1.2)
    ax1.set_xticks(x, labels, rotation=25, ha="right")
    ax1.set_ylabel("shape mismatch")
    ax1.set_title("Normalized spectrum-shape diagnostics")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "docs/_static/quasilinear_saturation_rule_sweep.png"))
    parser.add_argument("--title", default="Quasilinear saturation-rule sweep")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_saturation_rule_sweep()
    paths = write_saturation_rule_sweep_figure(report, out=args.out, title=args.title)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    for rule, payload in report["rules"].items():
        print(
            "{rule}: holdout_mean_abs_relative_error={mean}".format(
                rule=rule,
                mean=payload["holdout_mean_abs_relative_error"],
            )
        )
    print(
        "training_mean_null: holdout_mean_abs_relative_error={mean}".format(
            mean=report["null_training_mean_baseline"]["holdout_mean_abs_relative_error"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
