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

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.parallel import independent_map  # noqa: E402
from spectraxgk.diagnostics.quasilinear_calibration import (
    calibration_point_from_nonlinear_window_summary,
)  # noqa: E402
from spectraxgk.diagnostics.quasilinear_transport import (  # noqa: E402
    saturation_amplitude2,
    shape_aware_power_law_objective,
)


ROOT = Path(__file__).resolve().parents[2]


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
        spectrum=ROOT
        / "docs/_static/quasilinear_cyclone_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT / "docs/_static/nonlinear_cyclone_gate_summary.json",
        shape_gate=ROOT / "docs/_static/quasilinear_cyclone_spectrum_shape_gate.json",
    ),
    SaturationCase(
        case="cyclone_miller_long_window",
        split="holdout",
        geometry="cyclone_miller",
        spectrum=ROOT
        / "docs/_static/quasilinear_cyclone_miller_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/nonlinear_cyclone_miller_gate_summary.json",
        shape_gate=ROOT
        / "docs/_static/quasilinear_cyclone_miller_spectrum_shape_gate.json",
    ),
    SaturationCase(
        case="hsx_nonlinear_window",
        split="holdout",
        geometry="hsx",
        spectrum=ROOT
        / "docs/_static/quasilinear_hsx_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT / "docs/_static/nonlinear_hsx_gate_summary.json",
        shape_gate=ROOT / "docs/_static/quasilinear_hsx_spectrum_shape_gate.json",
    ),
    SaturationCase(
        case="w7x_nonlinear_window",
        split="holdout",
        geometry="w7x",
        spectrum=ROOT
        / "docs/_static/quasilinear_w7x_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT / "docs/_static/nonlinear_w7x_gate_summary.json",
        shape_gate=ROOT / "docs/_static/quasilinear_w7x_spectrum_shape_gate.json",
    ),
    SaturationCase(
        case="dshape_external_vmec_t250_window",
        split="holdout",
        geometry="dshape_external_vmec",
        spectrum=ROOT
        / "docs/_static/quasilinear_vmec_dshape_linear_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/external_vmec_dshape_t250_n64_transport_window.json",
        shape_gate=None,
    ),
    SaturationCase(
        case="itermodel_external_vmec_t350_window",
        split="train",
        geometry="itermodel_external_vmec",
        spectrum=ROOT
        / "docs/_static/quasilinear_vmec_itermodel_linear_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/external_vmec_itermodel_t350_n64_transport_window.json",
        shape_gate=None,
    ),
    SaturationCase(
        case="updown_asym_external_vmec_t450_window",
        split="holdout",
        geometry="updown_asym_external_vmec",
        spectrum=ROOT
        / "docs/_static/quasilinear_vmec_updown_asym_linear_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/external_vmec_updown_asym_t450_n64_transport_window.json",
        shape_gate=None,
    ),
    SaturationCase(
        case="circular_external_vmec_t450_window",
        split="holdout",
        geometry="circular_external_vmec",
        spectrum=ROOT
        / "docs/_static/quasilinear_vmec_circular_linear_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/external_vmec_circular_t450_n64_transport_window.json",
        shape_gate=None,
    ),
    SaturationCase(
        case="cth_like_external_vmec_t700_high_grid_ensemble",
        split="holdout",
        geometry="cth_like_external_vmec",
        spectrum=ROOT
        / "docs/_static/quasilinear_vmec_jax_cth_like_linear_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/external_vmec_cth_like_modified_replicates_t700/replicate_ensemble_gate.json",
        shape_gate=None,
    ),
    SaturationCase(
        case="shaped_tokamak_pressure_external_vmec_t650_high_grid_window",
        split="holdout",
        geometry="shaped_tokamak_external_vmec",
        spectrum=ROOT
        / "docs/_static/quasilinear_vmec_jax_shaped_tokamak_pressure_linear_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/external_vmec_shaped_tokamak_pressure_replicates_t650/replicate_ensemble_gate.json",
        shape_gate=None,
    ),
    SaturationCase(
        case="qp_diag_nfp2_m4_final_t250",
        split="holdout",
        geometry="qp_diag_nfp2_m4",
        spectrum=ROOT
        / "docs/_static/quasilinear_vmec_qp_diag_nfp2_m4_final_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/qp_diag_nfp2_m4_t250_replicate_ensemble_gate.json",
        shape_gate=None,
    ),
    SaturationCase(
        case="solovev_reference_repair_dt002_amp1em5_n48_t250",
        split="holdout",
        geometry="solovev_external_vmec",
        spectrum=ROOT
        / "docs/_static/quasilinear_vmec_solovev_linear_spectrum_scan.quasilinear_spectrum.csv",
        nonlinear_summary=ROOT
        / "docs/_static/external_vmec_holdouts/solovev_reference_repair_dt002_amp1em5_n48_t250/solovev_n48_t250_ensemble_gate.json",
        shape_gate=None,
    ),
)


RULE_LABELS = {
    "positive_mixing_length": r"$\max(\gamma,0)\,\hat Q/k_\perp^2$",
    "linear_weight": r"$\hat Q$",
    "absolute_growth_mixing_length": r"$|\gamma|\,\hat Q/k_\perp^2$",
}

DEFAULT_HOLDOUT_RELATIVE_ERROR_GATE = 0.35

CASE_LABELS = {
    "cyclone_long_window": "Cyclone train",
    "cyclone_miller_long_window": "Cyclone Miller",
    "hsx_nonlinear_window": "HSX",
    "w7x_nonlinear_window": "W7-X",
    "dshape_external_vmec_t250_window": "D-shaped VMEC",
    "itermodel_external_vmec_t350_window": "ITERModel VMEC",
    "updown_asym_external_vmec_t450_window": "Up-Down Asym VMEC",
    "circular_external_vmec_t450_window": "Circular VMEC",
    "cth_like_external_vmec_t700_high_grid_ensemble": "CTH-like VMEC",
    "shaped_tokamak_pressure_external_vmec_t650_high_grid_window": "Shaped-pressure VMEC",
    "qp_diag_nfp2_m4_final_t250": "QP VMEC",
    "solovev_reference_repair_dt002_amp1em5_n48_t250": "Solovev VMEC",
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


def _artifact_path(path: str | Path | None) -> str | None:
    """Return a stable repo-relative artifact path when possible."""

    if path is None:
        return None
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return path_obj.as_posix()


def _load_table(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    return data


def _case_label(case: object) -> str:
    return CASE_LABELS.get(str(case), str(case).replace("_", " "))


def _required_column(data: np.ndarray, path: Path, column: str) -> np.ndarray:
    names = set(data.dtype.names or ())
    if column not in names:
        raise ValueError(f"{path} is missing required column '{column}'")
    return np.asarray(data[column], dtype=float)


def raw_rule_estimates(
    spectrum_csv: str | Path, *, floor: float = 1.0e-300
) -> dict[str, float]:
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
                saturation_amplitude2(
                    gamma=float(g), kperp_eff2_value=float(k2), rule="mixing_length"
                )
                for g, k2 in zip(gamma, kperp2, strict=True)
            ],
            dtype=float,
        ),
        "linear_weight": np.asarray(
            [
                saturation_amplitude2(
                    gamma=float(g), kperp_eff2_value=float(k2), rule="linear_weight"
                )
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


def _fit_scale(
    raw: np.ndarray, observed: np.ndarray, mask: np.ndarray, *, floor: float
) -> float:
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
    if path is None:
        return {
            "shape_gate": None,
            "shape_gate_status": "missing",
            "shape_passed": None,
            "shape_tv": None,
            "shape_cosine": None,
            "shape_tv_gate": None,
            "shape_cosine_gate": None,
            "shape_gate_kind": None,
            "shape_gate_notes": "no shape-gate JSON configured for this case",
        }
    if not path.exists():
        return {
            "shape_gate": _artifact_path(path),
            "shape_gate_status": "missing",
            "shape_passed": None,
            "shape_tv": None,
            "shape_cosine": None,
            "shape_tv_gate": None,
            "shape_cosine_gate": None,
            "shape_gate_kind": None,
            "shape_gate_notes": "configured shape-gate JSON is missing",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    kind = data.get("kind")
    passed = bool(data.get("passed", False))
    status = (
        "invalid_kind"
        if kind != "quasilinear_spectrum_shape_gate"
        else ("passed" if passed else "failed")
    )
    return {
        "shape_gate": _artifact_path(path),
        "shape_gate_kind": kind,
        "shape_gate_status": status,
        "shape_passed": passed,
        "shape_tv": data.get("total_variation_distance"),
        "shape_cosine": data.get("cosine_similarity"),
        "shape_tv_gate": data.get("tv_gate"),
        "shape_cosine_gate": data.get("cosine_gate"),
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
        gate_report = (
            payload.get("gate_report")
            if isinstance(payload.get("gate_report"), dict)
            else {}
        )
        rows.append(
            {
                "case": case.case,
                "split": case.split,
                "required": required,
                "nonlinear_summary": _artifact_path(path),
                "passed": row_passed,
                "gate_passed": gate_passed,
                "gate_case": str(
                    gate_report.get("case", payload.get("case", path.stem))
                ),
                "reason": "matched passed nonlinear summary gate"
                if row_passed and required
                else (
                    "not required split"
                    if row_passed
                    else "nonlinear summary gate is missing or failed"
                ),
            }
        )
    return {
        "kind": "quasilinear_model_input_validation",
        "passed": passed,
        "required_splits": list(required_splits),
        "cases": rows,
    }


def require_validated_nonlinear_inputs(
    cases: tuple[SaturationCase, ...],
) -> dict[str, Any]:
    """Raise if any train/holdout case lacks a passed nonlinear summary gate."""

    report = nonlinear_input_validation_report(cases)
    if not bool(report["passed"]):
        failed = [row["case"] for row in report["cases"] if not bool(row["passed"])]
        raise ValueError(
            f"unvalidated nonlinear train/holdout input(s): {', '.join(failed)}"
        )
    return report


def _saturation_case_row(case: SaturationCase) -> dict[str, Any]:
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
    return {
        "case": case.case,
        "split": case.split,
        "geometry": case.geometry,
        "spectrum": _artifact_path(case.spectrum),
        "nonlinear_summary": _artifact_path(case.nonlinear_summary),
        "observed_heat_flux": observed_point.observed_heat_flux,
        "observed_heat_flux_std": observed_point.observed_heat_flux_std,
        "raw_estimates": raw,
        **_shape_payload(case.shape_gate),
    }


def build_saturation_rule_sweep(
    cases: tuple[SaturationCase, ...] = DEFAULT_CASES,
    *,
    observed_floor: float = 1.0e-12,
    require_validated_inputs: bool = True,
    workers: int = 1,
    parallel_executor: str = "thread",
    holdout_relative_error_gate: float = DEFAULT_HOLDOUT_RELATIVE_ERROR_GATE,
) -> dict[str, Any]:
    """Fit one scalar per rule on train cases and score all cases."""

    if holdout_relative_error_gate <= 0.0:
        raise ValueError("holdout_relative_error_gate must be positive")
    input_validation = (
        require_validated_nonlinear_inputs(cases)
        if require_validated_inputs
        else {
            "kind": "quasilinear_model_input_validation",
            "passed": None,
            "required": False,
        }
    )
    rule_names = tuple(RULE_LABELS)
    case_rows = independent_map(
        _saturation_case_row,
        cases,
        workers=workers,
        executor=parallel_executor,
    )

    observed = np.asarray([row["observed_heat_flux"] for row in case_rows], dtype=float)
    train_mask = np.asarray([row["split"] == "train" for row in case_rows], dtype=bool)
    if not np.any(train_mask):
        raise ValueError(
            "at least one train case is required to fit saturation-rule scales"
        )

    rules = {}
    for rule in rule_names:
        raw = np.asarray([row["raw_estimates"][rule] for row in case_rows], dtype=float)
        scale = _fit_scale(raw, observed, train_mask, floor=observed_floor)
        predicted = raw * scale if np.isfinite(scale) else np.full_like(raw, np.nan)
        rel_error = np.abs(predicted - observed) / np.maximum(
            np.abs(observed), observed_floor
        )
        holdout = rel_error[~train_mask]
        holdout_mean = None if holdout.size == 0 else float(np.nanmean(holdout))
        holdout_max = None if holdout.size == 0 else float(np.nanmax(holdout))
        rules[rule] = {
            "label": RULE_LABELS[rule],
            "scale": float(scale),
            "predicted_heat_flux": predicted.tolist(),
            "absolute_relative_error": rel_error.tolist(),
            "holdout_relative_error_gate": float(holdout_relative_error_gate),
            "holdout_gate_passed": None
            if holdout_mean is None
            else bool(holdout_mean <= holdout_relative_error_gate),
            "holdout_mean_abs_relative_error": holdout_mean,
            "holdout_max_abs_relative_error": holdout_max,
        }

    null_predicted = np.full_like(observed, float(np.nanmean(observed[train_mask])))
    null_rel_error = np.abs(null_predicted - observed) / np.maximum(
        np.abs(observed), observed_floor
    )
    null_holdout = null_rel_error[~train_mask]
    null_holdout_mean = (
        None if null_holdout.size == 0 else float(np.nanmean(null_holdout))
    )
    transport_gate = float(holdout_relative_error_gate)
    accepted_rules = []
    for rule, payload in rules.items():
        mean_error = payload["holdout_mean_abs_relative_error"]
        if mean_error is None:
            continue
        beats_null = null_holdout_mean is None or float(mean_error) < float(
            null_holdout_mean
        )
        if float(mean_error) <= transport_gate and beats_null:
            accepted_rules.append(rule)

    return {
        "kind": "quasilinear_saturation_rule_sweep",
        "claim_level": "model_comparison_not_validated_transport",
        "observed_floor": float(observed_floor),
        "holdout_relative_error_gate": float(holdout_relative_error_gate),
        "any_rule_holdout_gate_passed": any(
            payload["holdout_gate_passed"] is True for payload in rules.values()
        ),
        "best_rule_by_holdout_mean_abs_relative_error": min(
            rules,
            key=lambda name: float("inf")
            if rules[name]["holdout_mean_abs_relative_error"] is None
            else float(rules[name]["holdout_mean_abs_relative_error"]),
        ),
        "train_cases": [row["case"] for row in case_rows if row["split"] == "train"],
        "rules": rules,
        "null_training_mean_baseline": {
            "label": "training-mean null",
            "predicted_heat_flux": null_predicted.tolist(),
            "absolute_relative_error": null_rel_error.tolist(),
            "holdout_mean_abs_relative_error": null_holdout_mean,
            "holdout_max_abs_relative_error": None
            if null_holdout.size == 0
            else float(np.nanmax(null_holdout)),
        },
        "input_validation": input_validation,
        "parallel": {
            "workers": int(workers),
            "executor": str(parallel_executor),
            "identity_contract": "parallel case rows preserve serial case ordering",
        },
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


def write_saturation_rule_sweep_figure(
    report: dict[str, Any], *, out: str | Path, title: str
) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a saturation-rule sweep report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cases = list(report["cases"])
    rules = dict(report["rules"])
    null_baseline = dict(report["null_training_mean_baseline"])
    labels = [_case_label(row["case"]) for row in cases]
    y = np.arange(len(labels))

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    ax0, ax1 = axes
    gate = float(
        report.get("holdout_relative_error_gate", DEFAULT_HOLDOUT_RELATIVE_ERROR_GATE)
    )
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
    ax0.axvline(
        gate, color="#c2410c", linestyle="--", linewidth=1.5, label=f"{gate:.2g} gate"
    )
    ax0.set_xscale("log")
    ax0.set_xlim(floor * 0.7, max(max_err, gate) * 1.5)
    ax0.set_yticks(y, labels)
    ax0.invert_yaxis()
    ax0.set_xlabel("absolute relative error after Cyclone scale fit")
    ax0.set_title("Absolute-flux transfer")
    ax0.grid(True, axis="x", alpha=0.25)
    ax0.legend(loc="lower left", fontsize=8, framealpha=0.92)

    shape_cases = [
        row
        for row in cases
        if row.get("shape_tv") is not None and row.get("shape_cosine") is not None
    ]
    pending_shape_count = len(cases) - len(shape_cases)
    shape_labels = [_case_label(row["case"]) for row in shape_cases]
    shape_tv = np.asarray([float(row["shape_tv"]) for row in shape_cases], dtype=float)
    shape_cos = np.asarray(
        [float(row["shape_cosine"]) for row in shape_cases], dtype=float
    )
    tv_gates = [
        row.get("shape_tv_gate")
        for row in shape_cases
        if row.get("shape_tv_gate") is not None
    ]
    cosine_gates = [
        row.get("shape_cosine_gate")
        for row in shape_cases
        if row.get("shape_cosine_gate") is not None
    ]
    tv_gate = float(tv_gates[0]) if tv_gates else 0.2
    cosine_gate = float(cosine_gates[0]) if cosine_gates else 0.95
    x = np.arange(len(shape_labels))
    width = 0.38
    ax1.bar(
        x - width / 2.0, shape_tv, width=width, label="TV distance", color="#f97316"
    )
    ax1.bar(
        x + width / 2.0,
        1.0 - shape_cos,
        width=width,
        label="1 - cosine",
        color="#2a9d8f",
    )
    if pending_shape_count:
        ax1.text(
            0.99,
            0.03,
            f"{pending_shape_count} external-VMEC shape gates pending",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="0.35",
        )
    ax1.axhline(tv_gate, color="#f97316", linestyle=":", linewidth=1.2)
    ax1.axhline(1.0 - cosine_gate, color="#2a9d8f", linestyle=":", linewidth=1.2)
    ax1.set_xticks(x, shape_labels, rotation=25, ha="right")
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
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


# Shape-aware saturation model-development contracts.
DEFAULT_SHAPE_CASES = tuple(
    case for case in DEFAULT_CASES if case.shape_gate is not None
)


def _shape_load_table(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    return data


def _shape_required_column(data: np.ndarray, path: Path, column: str) -> np.ndarray:
    names = set(data.dtype.names or ())
    if column not in names:
        raise ValueError(f"{path} is missing required column '{column}'")
    return np.asarray(data[column], dtype=float)


def _shape_gate_payload(case: SaturationCase) -> dict[str, Any]:
    if case.shape_gate is None or not Path(case.shape_gate).exists():
        raise ValueError(f"{case.case} is missing a tracked shape-gate JSON")
    data = json.loads(Path(case.shape_gate).read_text(encoding="utf-8"))
    required = (
        "kind",
        "passed",
        "ky",
        "quasilinear_distribution",
        "nonlinear_distribution",
        "total_variation_distance",
        "cosine_similarity",
        "tv_gate",
        "cosine_gate",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"{case.shape_gate} is missing '{key}'")
    if data["kind"] != "quasilinear_spectrum_shape_gate":
        raise ValueError(f"{case.shape_gate} is not a quasilinear spectrum-shape gate")
    return data


def fit_power_law_shape_exponent(
    cases: tuple[SaturationCase, ...],
    *,
    passed_only: bool = False,
    floor: float = 1.0e-300,
) -> dict[str, Any]:
    """Fit ``nonlinear_shape / quasilinear_shape ~ C_case * ky**exponent``.

    Each training case receives its own intercept, while the exponent is shared.
    The fitted exponent is therefore a shape-transfer parameter, not an
    absolute-flux scale.
    """

    xs: list[float] = []
    ys: list[float] = []
    groups: list[int] = []
    used_cases: list[str] = []
    for group, case in enumerate(cases):
        payload = _shape_gate_payload(case)
        if passed_only and not bool(payload.get("passed", False)):
            continue
        ky = np.asarray(payload["ky"], dtype=float)
        ql = np.asarray(payload["quasilinear_distribution"], dtype=float)
        nl = np.asarray(payload["nonlinear_distribution"], dtype=float)
        mask = (
            (ky > 0.0)
            & (ql > floor)
            & (nl > floor)
            & np.isfinite(ky)
            & np.isfinite(ql)
            & np.isfinite(nl)
        )
        if not np.any(mask):
            continue
        ky_use = ky[mask]
        x = np.log(ky_use / np.exp(np.mean(np.log(ky_use))))
        y = np.log(nl[mask] / ql[mask])
        xs.extend(float(v) for v in x)
        ys.extend(float(v) for v in y)
        groups.extend([group] * len(x))
        used_cases.append(case.case)

    if len(used_cases) < 1 or len(xs) < 2:
        raise ValueError("not enough shape samples to fit a power-law envelope")
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    group_arr = np.asarray(groups, dtype=int)
    n_groups = max(groups) + 1
    design = np.zeros((x_arr.size, n_groups + 1), dtype=float)
    design[np.arange(x_arr.size), group_arr] = 1.0
    design[:, -1] = x_arr
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    residual = y_arr - design @ coef
    return {
        "exponent": float(coef[-1]),
        "used_cases": used_cases,
        "n_samples": int(x_arr.size),
        "rms_log_shape_residual": float(np.sqrt(np.mean(residual**2))),
        "passed_only": bool(passed_only),
    }


def shape_aware_raw_estimate(spectrum_csv: str | Path, *, exponent: float) -> float:
    """Return ``sum Qhat(ky) * (ky/geomean(ky))**exponent`` from a spectrum CSV."""

    path = Path(spectrum_csv)
    data = _shape_load_table(path)
    ky = _shape_required_column(data, path, "ky")
    weight = np.maximum(_shape_required_column(data, path, "heat_flux_weight_total"), 0.0)
    finite = (ky > 0.0) & np.isfinite(ky) & np.isfinite(weight)
    if not np.any(finite):
        raise ValueError(f"{path} contains no finite positive ky/weight samples")
    ky = ky[finite]
    weight = weight[finite]
    features = np.stack([np.zeros_like(weight), np.ones_like(weight), weight], axis=-1)
    return float(
        np.sum(
            np.asarray(shape_aware_power_law_objective(features, ky, exponent=exponent))
        )
    )


def _tracked_observed_flux(case: SaturationCase) -> tuple[float, float | None] | None:
    """Return observed flux from tracked calibration sidecars when raw traces are absent.

    Several long nonlinear diagnostics live under ignored ``tools_out`` paths to
    keep the repository small.  CI and source distributions still need to
    replay candidate-model audits from tracked evidence, so the plotting tools
    fall back to the compact calibration-point sidecar instead of requiring the
    raw trace to be present.
    """

    candidates = (
        ROOT / "docs/_static/quasilinear_stellarator_train_holdout_points.json",
        ROOT / "docs/_static/quasilinear_stellarator_train_holdout_report.json",
    )
    aliases = {
        "cth_like_external_vmec_t700_high_grid_ensemble": "cth_like_external_vmec_t700_high_grid_window",
    }
    wanted = {case.case, aliases.get(case.case, case.case)}
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        points = payload if isinstance(payload, list) else payload.get("points", [])
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict) or point.get("case") not in wanted:
                continue
            observed = point.get("observed_heat_flux")
            if observed is None:
                continue
            observed_std = point.get("observed_heat_flux_std")
            return (
                float(observed),
                None if observed_std is None else float(observed_std),
            )
    return None


def _observed_flux(case: SaturationCase) -> tuple[float, float | None]:
    try:
        point = calibration_point_from_nonlinear_window_summary(
            case.nonlinear_summary,
            predicted_heat_flux=1.0,
            split=case.split,
            saturation_rule="shape_aware_power_law",
            geometry=case.geometry,
            electron_model="adiabatic",
            quasilinear_artifact=str(case.spectrum),
        )
    except FileNotFoundError:
        tracked = _tracked_observed_flux(case)
        if tracked is None:
            raise
        return tracked
    return point.observed_heat_flux, point.observed_heat_flux_std


def _shape_fit_scale(raw: np.ndarray, observed: np.ndarray, *, floor: float) -> float:
    finite = np.isfinite(raw) & np.isfinite(observed) & (np.abs(raw) > floor)
    if not np.any(finite):
        return float("nan")
    denom = float(np.dot(raw[finite], raw[finite]))
    if denom <= floor:
        return float("nan")
    return float(np.dot(raw[finite], observed[finite]) / denom)


def build_shape_aware_saturation_report(
    cases: tuple[SaturationCase, ...] = DEFAULT_SHAPE_CASES,
    *,
    observed_floor: float = 1.0e-12,
    passed_shape_only: bool = False,
    require_validated_inputs: bool = True,
    holdout_relative_error_gate: float = DEFAULT_HOLDOUT_RELATIVE_ERROR_GATE,
) -> dict[str, Any]:
    """Run leave-one-case-out validation for the power-law shape-aware model."""

    if holdout_relative_error_gate <= 0.0:
        raise ValueError("holdout_relative_error_gate must be positive")
    input_validation = (
        require_validated_nonlinear_inputs(cases)
        if require_validated_inputs
        else {
            "kind": "quasilinear_model_input_validation",
            "passed": None,
            "required": False,
        }
    )
    case_rows = []
    for case in cases:
        observed, observed_std = _observed_flux(case)
        shape_payload = _shape_gate_payload(case)
        case_rows.append(
            {
                "case": case.case,
                "geometry": case.geometry,
                "spectrum": _artifact_path(case.spectrum),
                "nonlinear_summary": _artifact_path(case.nonlinear_summary),
                "shape_gate": _artifact_path(case.shape_gate),
                "shape_gate_kind": shape_payload.get("kind"),
                "shape_gate_status": "passed"
                if bool(shape_payload.get("passed", False))
                else "failed",
                "shape_passed": bool(shape_payload.get("passed", False)),
                "shape_tv": shape_payload.get("total_variation_distance"),
                "shape_cosine": shape_payload.get("cosine_similarity"),
                "shape_tv_gate": shape_payload.get("tv_gate"),
                "shape_cosine_gate": shape_payload.get("cosine_gate"),
                "observed_heat_flux": float(observed),
                "observed_heat_flux_std": None
                if observed_std is None
                else float(observed_std),
            }
        )

    observed_arr = np.asarray(
        [row["observed_heat_flux"] for row in case_rows], dtype=float
    )
    loo_rows = []
    for holdout_idx, holdout_case in enumerate(cases):
        train_cases = tuple(case for i, case in enumerate(cases) if i != holdout_idx)
        fit = fit_power_law_shape_exponent(train_cases, passed_only=passed_shape_only)
        exponent = float(fit["exponent"])
        train_raw = np.asarray(
            [
                shape_aware_raw_estimate(case.spectrum, exponent=exponent)
                for case in train_cases
            ]
        )
        train_observed = np.asarray(
            [
                case_rows[i]["observed_heat_flux"]
                for i in range(len(cases))
                if i != holdout_idx
            ]
        )
        scale = _shape_fit_scale(train_raw, train_observed, floor=observed_floor)
        holdout_raw = shape_aware_raw_estimate(holdout_case.spectrum, exponent=exponent)
        predicted = float(scale * holdout_raw)
        observed = float(observed_arr[holdout_idx])
        rel_error = abs(predicted - observed) / max(abs(observed), observed_floor)
        baseline_raw_train = np.asarray(
            [
                shape_aware_raw_estimate(case.spectrum, exponent=0.0)
                for case in train_cases
            ]
        )
        baseline_scale = _shape_fit_scale(
            baseline_raw_train, train_observed, floor=observed_floor
        )
        baseline_predicted = float(
            baseline_scale
            * shape_aware_raw_estimate(holdout_case.spectrum, exponent=0.0)
        )
        baseline_rel_error = abs(baseline_predicted - observed) / max(
            abs(observed), observed_floor
        )
        null_predicted = float(np.mean(train_observed))
        null_rel_error = abs(null_predicted - observed) / max(
            abs(observed), observed_floor
        )
        loo_rows.append(
            {
                "holdout_case": holdout_case.case,
                "train_cases": [case.case for case in train_cases],
                "exponent": exponent,
                "scale": scale,
                "predicted_heat_flux": predicted,
                "observed_heat_flux": observed,
                "absolute_relative_error": float(rel_error),
                "baseline_linear_weight_predicted_heat_flux": baseline_predicted,
                "baseline_linear_weight_absolute_relative_error": float(
                    baseline_rel_error
                ),
                "null_training_mean_predicted_heat_flux": null_predicted,
                "null_training_mean_absolute_relative_error": float(null_rel_error),
                "shape_fit": fit,
            }
        )

    shape_errors = np.asarray(
        [row["absolute_relative_error"] for row in loo_rows], dtype=float
    )
    baseline_errors = np.asarray(
        [row["baseline_linear_weight_absolute_relative_error"] for row in loo_rows],
        dtype=float,
    )
    null_errors = np.asarray(
        [row["null_training_mean_absolute_relative_error"] for row in loo_rows],
        dtype=float,
    )
    all_fit = fit_power_law_shape_exponent(cases, passed_only=passed_shape_only)
    shape_mean = float(np.nanmean(shape_errors))
    baseline_mean = float(np.nanmean(baseline_errors))
    null_mean = float(np.nanmean(null_errors))
    transport_gate = float(holdout_relative_error_gate)
    return {
        "kind": "quasilinear_shape_aware_saturation_report",
        "claim_level": "leave_one_geometry_out_model_development",
        "observed_floor": float(observed_floor),
        "holdout_relative_error_gate": float(holdout_relative_error_gate),
        "passed_shape_only": bool(passed_shape_only),
        "input_validation": input_validation,
        "all_case_shape_fit": all_fit,
        "metrics": {
            "shape_aware_mean_abs_relative_error": shape_mean,
            "shape_aware_max_abs_relative_error": float(np.nanmax(shape_errors)),
            "shape_aware_all_case_gate_passed": bool(
                np.all(shape_errors <= transport_gate)
            ),
            "baseline_linear_weight_mean_abs_relative_error": baseline_mean,
            "baseline_linear_weight_max_abs_relative_error": float(
                np.nanmax(baseline_errors)
            ),
            "baseline_linear_weight_all_case_gate_passed": bool(
                np.all(baseline_errors <= transport_gate)
            ),
            "null_training_mean_mean_abs_relative_error": null_mean,
            "null_training_mean_max_abs_relative_error": float(np.nanmax(null_errors)),
            "null_training_mean_all_case_gate_passed": bool(
                np.all(null_errors <= transport_gate)
            ),
        },
        "promotion_gate": {
            "passed": bool(
                shape_mean <= transport_gate
                and shape_mean < baseline_mean
                and shape_mean < null_mean
            ),
            "transport_mean_relative_error_gate": transport_gate,
            "requires_beating_linear_weight_baseline": True,
            "requires_beating_training_mean_null": True,
            "shape_aware_mean_abs_relative_error": shape_mean,
            "baseline_linear_weight_mean_abs_relative_error": baseline_mean,
            "null_training_mean_mean_abs_relative_error": null_mean,
        },
        "cases": case_rows,
        "leave_one_out": loo_rows,
        "notes": (
            "The power-law exponent is fitted from training nonlinear spectrum-shape gates only; "
            "the held-out nonlinear shape is not used for that held-out prediction. This is a "
            "model-development diagnostic, not a validated transport claim."
        ),
    }


def write_shape_aware_saturation_figure(
    report: dict[str, Any], *, out: str | Path, title: str
) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a shape-aware saturation report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(report["leave_one_out"])
    labels = [str(row["holdout_case"]) for row in rows]
    short_labels = [
        label.replace("_long_window", "").replace("_nonlinear_window", "")
        for label in labels
    ]
    observed = np.asarray([row["observed_heat_flux"] for row in rows], dtype=float)
    predicted = np.asarray([row["predicted_heat_flux"] for row in rows], dtype=float)
    baseline = np.asarray(
        [row["baseline_linear_weight_predicted_heat_flux"] for row in rows], dtype=float
    )
    null = np.asarray(
        [row["null_training_mean_predicted_heat_flux"] for row in rows], dtype=float
    )
    shape_err = np.asarray(
        [row["absolute_relative_error"] for row in rows], dtype=float
    )
    baseline_err = np.asarray(
        [row["baseline_linear_weight_absolute_relative_error"] for row in rows],
        dtype=float,
    )
    null_err = np.asarray(
        [row["null_training_mean_absolute_relative_error"] for row in rows], dtype=float
    )

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    ax0, ax1 = axes
    positive = np.concatenate(
        [
            observed[observed > 0.0],
            predicted[predicted > 0.0],
            baseline[baseline > 0.0],
            null[null > 0.0],
        ]
    )
    lo = float(np.min(positive)) * 0.6
    hi = float(np.max(positive)) * 1.7
    ax0.plot(
        [lo, hi], [lo, hi], color="0.25", linestyle="--", linewidth=1.5, label="1:1"
    )
    ax0.scatter(
        observed,
        baseline,
        s=70,
        facecolors="none",
        edgecolors="#6b7280",
        linewidth=1.5,
        label="linear-weight LOO",
    )
    ax0.scatter(
        observed,
        null,
        s=65,
        marker="^",
        color="#b45309",
        edgecolor="white",
        linewidth=0.8,
        label="train-mean null",
    )
    ax0.scatter(
        observed,
        predicted,
        s=75,
        color="#0f4c81",
        edgecolor="white",
        linewidth=0.8,
        label="shape-aware LOO",
    )
    for label, xval, yval in zip(short_labels, observed, predicted, strict=True):
        ax0.annotate(
            label, (xval, yval), xytext=(5, 4), textcoords="offset points", fontsize=7
        )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel("observed nonlinear heat-flux window")
    ax0.set_ylabel("leave-one-out prediction")
    ax0.set_title("Absolute flux")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, which="both", alpha=0.24)

    y = np.arange(len(labels))
    height = 0.25
    ax1.barh(
        y - height,
        baseline_err,
        height=height,
        color="#9ca3af",
        label="linear-weight baseline",
    )
    ax1.barh(y, null_err, height=height, color="#b45309", label="train-mean null")
    ax1.barh(y + height, shape_err, height=height, color="#0f4c81", label="shape-aware")
    gate = float(
        report.get("holdout_relative_error_gate", DEFAULT_HOLDOUT_RELATIVE_ERROR_GATE)
    )
    ax1.axvline(
        gate, color="#c2410c", linestyle="--", linewidth=1.5, label=f"{gate:.2g} gate"
    )
    ax1.set_xscale("log")
    ax1.set_yticks(y, short_labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("absolute relative error")
    ax1.set_title("Leave-one-geometry-out errors")
    ax1.grid(True, axis="x", alpha=0.24)
    ax1.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


def build_saturation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(ROOT / "docs/_static/quasilinear_saturation_rule_sweep.png"),
    )
    parser.add_argument("--title", default="Quasilinear saturation-rule sweep")
    parser.add_argument(
        "--workers", type=int, default=1, help="Independent case-row workers."
    )
    parser.add_argument(
        "--parallel-executor", choices=("thread", "process"), default="thread"
    )
    return parser


def build_shape_aware_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the shape-aware quasilinear saturation diagnostic."
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "docs/_static/quasilinear_shape_aware_saturation.png"),
    )
    parser.add_argument(
        "--title", default="Shape-aware quasilinear saturation diagnostic"
    )
    parser.add_argument(
        "--passed-shape-only",
        action="store_true",
        help="Fit shape exponent using only training cases whose shape gate passed.",
    )
    return parser


def _run_shape_aware(args: argparse.Namespace) -> int:
    report = build_shape_aware_saturation_report(
        passed_shape_only=args.passed_shape_only
    )
    paths = write_shape_aware_saturation_figure(report, out=args.out, title=args.title)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    print(
        "shape_aware_mean_abs_relative_error={shape:.6g} "
        "baseline_mean_abs_relative_error={base:.6g} "
        "null_mean_abs_relative_error={null:.6g}".format(
            shape=report["metrics"]["shape_aware_mean_abs_relative_error"],
            base=report["metrics"]["baseline_linear_weight_mean_abs_relative_error"],
            null=report["metrics"]["null_training_mean_mean_abs_relative_error"],
        )
    )
    return 0


def _run_saturation(args: argparse.Namespace) -> int:
    report = build_saturation_rule_sweep(
        workers=args.workers, parallel_executor=args.parallel_executor
    )
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
            mean=report["null_training_mean_baseline"][
                "holdout_mean_abs_relative_error"
            ],
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv[:1] == ["shape-aware"]:
        return _run_shape_aware(build_shape_aware_parser().parse_args(raw_argv[1:]))
    return _run_saturation(build_saturation_parser().parse_args(raw_argv))


if __name__ == "__main__":
    sys.exit(main())
