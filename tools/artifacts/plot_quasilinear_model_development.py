#!/usr/bin/env python3
"""Render quasilinear model-development claim-boundary artifacts.

This command-family owns the lightweight quasilinear model-development panels
that guard absolute-flux claims: the dataset-sufficiency audit and the
model-selection status panel. Keeping both in one module makes the release
artifact path easier to discover without changing the generated artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.diagnostics.quasilinear_model_selection import (  # noqa: E402
    DEFAULT_REQUIRED_CANDIDATE,
    build_quasilinear_model_selection_status_from_paths,
)

from plot_quasilinear_saturation_rule_sweep import (  # noqa: E402
    DEFAULT_CASES,
    SaturationCase,
    nonlinear_input_validation_report,
)  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_SELECTION_OUT = (
    ROOT / "docs/_static/quasilinear_model_selection_status.png"
)
DEFAULT_DATASET_OUT = ROOT / "docs/_static/quasilinear_dataset_sufficiency.png"
DEFAULT_DATASET = ROOT / "docs/_static/quasilinear_dataset_sufficiency.json"
DEFAULT_CANDIDATE = ROOT / "docs/_static/quasilinear_candidate_uncertainty.json"
DEFAULT_CALIBRATION_REPORTS = tuple(
    sorted((ROOT / "docs/_static").glob("quasilinear_*train_holdout_report.json"))
)


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


GATE_LABELS = {
    "dataset_sufficiency_passed": "dataset\nsufficiency",
    "candidate_uncertainty_passed": "candidate\nuncertainty",
    "required_candidate_accepted": "candidate\naccepted",
    "required_candidate_eligible": "candidate\neligible",
    "required_candidate_transport_error": "transport\nerror",
    "required_candidate_interval_coverage": "interval\ncoverage",
    "required_candidate_beats_training_mean_null": "beats\nnull",
    "required_candidate_beats_linear_weight": "beats\nlinear",
    "absolute_flux_not_promoted": "absolute flux\nnot promoted",
    "calibration_reports_have_holdout_metrics": "holdout\nmetrics",
    "optimized_equilibrium_nonlinear_audit_present": "optimized NL\naudit present",
    "optimized_equilibrium_nonlinear_audit_qualified": "optimized NL\naudit passes",
    "optimized_equilibrium_nonlinear_audit_scope_limited": "optimized NL\nscope limited",
}


def write_model_selection_status_artifacts(
    status: dict[str, Any],
    *,
    out: str | Path,
    title: str,
    dpi: int = 220,
    write_pdf: bool = True,
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for a model-selection status."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    metrics = dict(status.get("metrics", {}))
    labels = ["spectral\nenvelope", "linear\nweight", "train-mean\nnull"]
    values = [
        metrics.get("candidate_mean_abs_relative_error"),
        metrics.get("linear_weight_mean_abs_relative_error"),
        metrics.get("null_training_mean_mean_abs_relative_error"),
    ]
    finite_values = [
        float(v) for v in values if v is not None and math.isfinite(float(v))
    ]
    if not finite_values:
        finite_values = [1.0]
    gate_value = metrics.get("transport_mean_relative_error_gate")
    gates = list(status.get("gate_report", {}).get("gates", []))
    passed = [bool(gate.get("passed", False)) for gate in gates]
    gate_labels = [
        GATE_LABELS.get(
            str(gate.get("metric", "unknown")),
            str(gate.get("metric", "unknown")).replace("_", "\n"),
        )
        for gate in gates
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.05]},
    )
    ax0, ax1 = axes
    colors = ["#0f4c81", "#8d99ae", "#b45309"]
    x = np.arange(len(labels))
    bars = ax0.bar(
        x, [float(v) if v is not None else np.nan for v in values], color=colors
    )
    ax0.set_yscale("log")
    ax0.set_xticks(x, labels)
    ax0.set_ylabel("mean absolute relative error")
    ax0.set_title("Leave-one-geometry-out skill")
    ax0.grid(True, axis="y", alpha=0.25)
    if gate_value is not None and math.isfinite(float(gate_value)):
        ax0.axhline(
            float(gate_value),
            color="#c2410c",
            linestyle="--",
            linewidth=1.5,
            label=f"transport gate {float(gate_value):.2g}",
        )
        ax0.legend(loc="best", fontsize=8)
    ymin = max(min(finite_values) * 0.55, 1.0e-4)
    ymax = max(finite_values) * 1.8
    ax0.set_ylim(ymin, ymax)
    for bar, value in zip(bars, values, strict=True):
        if value is None or not math.isfinite(float(value)):
            continue
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) / 1.08,
            f"{float(value):.3g}",
            ha="center",
            va="top",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    y = np.arange(len(gate_labels))
    gate_colors = ["#2a9d8f" if item else "#d1495b" for item in passed]
    ax1.barh(y, np.ones_like(y, dtype=float), color=gate_colors)
    ax1.set_yticks(y, gate_labels)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xticks([])
    ax1.set_title("Promotion guardrails")
    for ypos, item in zip(y, passed, strict=True):
        ax1.text(
            0.5,
            ypos,
            "pass" if item else "fail",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9,
        )
    ax1.invert_yaxis()

    candidate = str(status.get("required_candidate", DEFAULT_REQUIRED_CANDIDATE))
    candidate_label = candidate.replace("_", " ")
    claim = str(status.get("claim_level", "unknown")).replace("_", " ")
    claim_label = "scoped model-selection result; not a runtime absolute-flux predictor"
    if "incomplete" in claim:
        claim_label = "model-selection or scope gate incomplete"
    fig.suptitle(f"{title}\n{candidate_label}: {claim_label}", fontsize=13)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    paths = {"png": str(out_path)}
    if write_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        paths["pdf"] = str(pdf_path)
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(_json_clean(status), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["json"] = str(json_path)

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("metric", "passed", "detail"),
            lineterminator="\n",
        )
        writer.writeheader()
        for gate in gates:
            writer.writerow(
                {
                    "metric": gate.get("metric", ""),
                    "passed": bool(gate.get("passed", False)),
                    "detail": gate.get("detail", ""),
                }
            )
    paths["csv"] = str(csv_path)
    return paths


def build_model_selection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        action="append",
        default=None,
        help="Train/holdout calibration report. Defaults to tracked QL reports.",
    )
    parser.add_argument(
        "--optimized-equilibrium-nonlinear-audit",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional optimized-equilibrium nonlinear audit JSON. If supplied, "
            "it must be scoped and qualified; it cannot promote universal "
            "absolute-flux claims."
        ),
    )
    parser.add_argument(
        "--require-optimized-equilibrium-nonlinear-audit",
        action="store_true",
        help="Fail closed unless a qualified optimized-equilibrium nonlinear audit is supplied.",
    )
    parser.add_argument("--required-candidate", default=DEFAULT_REQUIRED_CANDIDATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_SELECTION_OUT)
    parser.add_argument("--title", default="Quasilinear model-selection status")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-pdf", action="store_true")
    return parser


def model_selection_main(argv: list[str] | None = None) -> int:
    args = build_model_selection_parser().parse_args(argv)
    reports = (
        tuple(args.calibration_report)
        if args.calibration_report is not None
        else DEFAULT_CALIBRATION_REPORTS
    )
    status = build_quasilinear_model_selection_status_from_paths(
        dataset_sufficiency=args.dataset,
        candidate_uncertainty=args.candidate,
        calibration_reports=reports,
        optimized_equilibrium_nonlinear_audits=tuple(
            args.optimized_equilibrium_nonlinear_audit
        ),
        required_candidate=args.required_candidate,
        require_optimized_equilibrium_nonlinear_audit=(
            args.require_optimized_equilibrium_nonlinear_audit
        ),
    )
    paths = write_model_selection_status_artifacts(
        status,
        out=args.out,
        title=args.title,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )
    for key in ("png", "pdf", "json", "csv"):
        if key in paths:
            print(f"saved {paths[key]}")
    print(
        "promotion_gate_passed={passed} blockers={blockers}".format(
            passed=status["promotion_gate"]["passed"],
            blockers=",".join(status["promotion_gate"]["blockers"]) or "none",
        )
    )
    return 0 if status["promotion_gate"]["passed"] else 1


CANDIDATE_PARAMETER_COUNTS = {
    "linear_weight": 1,
    "shape_power_law": 2,
    "spectral_envelope_ridge": 3,
    "linear_state_ridge": 5,
}

CANDIDATE_LABELS = {
    "linear_weight": "linear weight",
    "shape_power_law": "shape power law",
    "spectral_envelope_ridge": "spectral-envelope ridge",
    "linear_state_ridge": "linear-state ridge",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _summary_gate_case(summary_path: Path) -> str:
    payload = _read_json(summary_path)
    gate_report = payload.get("gate_report")
    if isinstance(gate_report, dict) and gate_report.get("case"):
        return str(gate_report["case"])
    if payload.get("case"):
        return str(payload["case"])
    return summary_path.stem


def _shape_gate_passed(path: Path | None) -> bool | None:
    if path is None or not path.exists():
        return None
    payload = _read_json(path)
    return bool(payload.get("passed", False))


def _load_promotion_gate(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _read_json(path)
    gate = payload.get("promotion_gate")
    if not isinstance(gate, dict):
        return None
    accepted_raw = gate.get("accepted_candidates", gate.get("accepted_rules", []))
    accepted = list(accepted_raw) if isinstance(accepted_raw, list) else []
    return {
        "path": str(path),
        "passed": bool(gate.get("passed", False)),
        "accepted": accepted,
        "kind": payload.get("kind", path.stem),
    }


def _nonlinear_index_rows(
    path: Path | None, used_gate_cases: set[str]
) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = _read_json(path)
    case_names = [str(item) for item in payload.get("cases", [])]
    thresholds = payload.get("case_gate_thresholds", {})
    passed = payload.get("case_gate_passed", {})
    rows = []
    for case in case_names:
        if case in used_gate_cases:
            continue
        reason = "not matched to a quasilinear spectrum"
        if case.startswith("kbm"):
            reason = "electromagnetic nonlinear lane; electrostatic quasilinear channels are not promoted"
        rows.append(
            {
                "case": case,
                "gate_passed": bool(passed.get(case, False)),
                "case_gate_mean_rel": thresholds.get(case),
                "reason": reason,
            }
        )
    return rows


def build_dataset_sufficiency_report(
    cases: tuple[SaturationCase, ...] = DEFAULT_CASES,
    *,
    nonlinear_index: str | Path | None = ROOT
    / "docs/_static/nonlinear_window_statistics.json",
    candidate_gate: str | Path | None = ROOT
    / "docs/_static/quasilinear_candidate_uncertainty.json",
    saturation_gate: str | Path | None = ROOT
    / "docs/_static/quasilinear_saturation_rule_sweep.json",
    min_total_electrostatic_cases: int = 6,
    min_explicit_train_geometries: int = 2,
    min_holdout_geometries: int = 3,
    min_leave_one_out_train_to_parameter_ratio: float = 2.0,
) -> dict[str, Any]:
    """Build a dataset-volume gate for calibrated quasilinear promotion.

    The gate does not fit a model. It audits whether the existing nonlinear
    windows, spectrum-shape checks, and candidate reports are sufficient to
    promote a richer absolute-flux saturation rule.
    """

    input_validation = nonlinear_input_validation_report(cases)
    case_rows = []
    for case in cases:
        summary_path = Path(case.nonlinear_summary)
        case_rows.append(
            {
                "case": case.case,
                "gate_case": _summary_gate_case(summary_path),
                "split": case.split,
                "geometry": case.geometry,
                "spectrum": str(case.spectrum),
                "nonlinear_summary": str(summary_path),
                "shape_gate": None if case.shape_gate is None else str(case.shape_gate),
                "shape_gate_passed": _shape_gate_passed(case.shape_gate),
            }
        )

    train_geometries = sorted(
        {row["geometry"] for row in case_rows if row["split"] == "train"}
    )
    holdout_geometries = sorted(
        {row["geometry"] for row in case_rows if row["split"] == "holdout"}
    )
    all_geometries = sorted({row["geometry"] for row in case_rows})
    used_gate_cases = {str(row["gate_case"]) for row in case_rows}
    excluded_cases = _nonlinear_index_rows(
        Path(nonlinear_index) if nonlinear_index is not None else None, used_gate_cases
    )

    n_cases = len(case_rows)
    loo_train_cases_per_fold = max(n_cases - 1, 0)
    candidate_rows = []
    for name, n_parameters in CANDIDATE_PARAMETER_COUNTS.items():
        ratio = (
            float("inf")
            if n_parameters == 0
            else float(loo_train_cases_per_fold / n_parameters)
        )
        candidate_rows.append(
            {
                "candidate": name,
                "label": CANDIDATE_LABELS[name],
                "n_parameters": int(n_parameters),
                "leave_one_out_train_cases_per_fold": int(loo_train_cases_per_fold),
                "train_to_parameter_ratio": ratio,
                "min_train_to_parameter_ratio": float(
                    min_leave_one_out_train_to_parameter_ratio
                ),
                "data_volume_passed": bool(
                    ratio >= min_leave_one_out_train_to_parameter_ratio
                ),
            }
        )

    requirements = {
        "validated_input_gates": bool(input_validation["passed"]),
        "minimum_total_electrostatic_cases": n_cases >= min_total_electrostatic_cases,
        "minimum_explicit_train_geometries": len(train_geometries)
        >= min_explicit_train_geometries,
        "minimum_holdout_geometries": len(holdout_geometries) >= min_holdout_geometries,
        "candidate_data_volume": any(
            bool(row["data_volume_passed"]) for row in candidate_rows
        ),
    }
    blockers = [name for name, passed in requirements.items() if not passed]

    candidate_gate_payload = _load_promotion_gate(
        Path(candidate_gate) if candidate_gate is not None else None
    )
    saturation_gate_payload = _load_promotion_gate(
        Path(saturation_gate) if saturation_gate is not None else None
    )
    downstream_gates = {
        "candidate_uncertainty": candidate_gate_payload,
        "saturation_rule_sweep": saturation_gate_payload,
    }
    downstream_passed = any(
        gate is not None and bool(gate["passed"]) for gate in downstream_gates.values()
    )
    if not downstream_passed:
        blockers.append("downstream_candidate_skill_gates_not_passed")

    return {
        "kind": "quasilinear_dataset_sufficiency",
        "claim_level": "scoped_low_parameter_candidate_promotion_not_runtime_option",
        "input_validation": input_validation,
        "requirements": {
            "min_total_electrostatic_cases": int(min_total_electrostatic_cases),
            "min_explicit_train_geometries": int(min_explicit_train_geometries),
            "min_holdout_geometries": int(min_holdout_geometries),
            "min_leave_one_out_train_to_parameter_ratio": float(
                min_leave_one_out_train_to_parameter_ratio
            ),
            "current_total_cases": int(n_cases),
            "current_explicit_train_geometries": len(train_geometries),
            "current_holdout_geometries": len(holdout_geometries),
            "current_leave_one_out_train_cases_per_fold": int(loo_train_cases_per_fold),
            "checks": requirements,
        },
        "promotion_gate": {
            "passed": bool(not blockers and downstream_passed),
            "blockers": blockers,
            "requires_passed_nonlinear_input_gates": True,
            "requires_downstream_candidate_skill_gates": True,
        },
        "downstream_gates": downstream_gates,
        "cases": case_rows,
        "train_geometries": train_geometries,
        "holdout_geometries": holdout_geometries,
        "geometries": all_geometries,
        "candidate_requirements": candidate_rows,
        "excluded_validated_nonlinear_cases": excluded_cases,
        "notes": (
            "This report is a promotion guard, not a transport model. It prevents "
            "candidate saturation rules from being documented as absolute-flux "
            "predictors until the nonlinear calibration portfolio is large enough, "
            "electrostatic-compatible, and at least one downstream held-out skill gate passes."
        ),
    }


def _short_case_label(name: str) -> str:
    return (
        name.replace("_nonlinear_window", "")
        .replace("_long_window", "")
        .replace("cyclone_miller", "miller")
        .replace("_", "\n")
    )


def write_dataset_sufficiency_figure(
    report: dict[str, Any],
    *,
    out: str | Path,
    title: str,
    dpi: int = 220,
    write_pdf: bool = True,
) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a quasilinear dataset-sufficiency gate."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()

    case_rows = list(report["cases"])
    y = np.arange(len(case_rows))
    split_colors = {"train": "#0f4c81", "holdout": "#2a9d8f"}
    for idx, row in enumerate(case_rows):
        color = split_colors.get(str(row["split"]), "#6b7280")
        shape_passed = row.get("shape_gate_passed")
        marker = "o" if shape_passed is not False else "X"
        ax0.scatter(
            0.0,
            idx,
            s=130,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
        )
        ax0.text(
            0.08, idx, f"{row['geometry']} / {row['split']}", va="center", fontsize=9
        )
    ax0.set_yticks(y, [_short_case_label(str(row["case"])) for row in case_rows])
    ax0.set_xlim(-0.18, 1.25)
    ax0.set_xticks([])
    ax0.invert_yaxis()
    ax0.set_title("Validated electrostatic calibration cases")
    ax0.spines[["top", "right", "bottom"]].set_visible(False)
    ax0.grid(False)
    ax0.scatter([], [], color=split_colors["train"], label="train")
    ax0.scatter([], [], color=split_colors["holdout"], label="holdout")
    ax0.scatter([], [], color="#6b7280", marker="X", label="shape gate failed")
    ax0.legend(loc="lower right", fontsize=8)

    requirements = report["requirements"]
    metric_labels = ["total\ncases", "train\ngeometries", "holdout\ngeometries"]
    current = np.asarray(
        [
            requirements["current_total_cases"],
            requirements["current_explicit_train_geometries"],
            requirements["current_holdout_geometries"],
        ],
        dtype=float,
    )
    required = np.asarray(
        [
            requirements["min_total_electrostatic_cases"],
            requirements["min_explicit_train_geometries"],
            requirements["min_holdout_geometries"],
        ],
        dtype=float,
    )
    x = np.arange(len(metric_labels))
    ax1.bar(x - 0.18, current, width=0.36, color="#0f4c81", label="current")
    ax1.bar(
        x + 0.18,
        required,
        width=0.36,
        color="#e5e7eb",
        edgecolor="#374151",
        label="required",
    )
    for xpos, cur, req in zip(x, current, required, strict=True):
        ax1.text(xpos - 0.18, cur + 0.08, f"{cur:.0f}", ha="center", fontsize=9)
        ax1.text(xpos + 0.18, req + 0.08, f"{req:.0f}", ha="center", fontsize=9)
    ax1.set_xticks(x, metric_labels)
    ax1.set_ylabel("count")
    ax1.set_title("Dataset volume requirements")
    ax1.set_ylim(0.0, max(float(np.max(required)), float(np.max(current))) + 1.1)
    ax1.grid(True, axis="y", alpha=0.24)
    ax1.legend(loc="upper left", fontsize=8)

    candidate_rows = list(report["candidate_requirements"])
    labels = [str(row["label"]).replace(" ", "\n") for row in candidate_rows]
    ratios = np.asarray(
        [row["train_to_parameter_ratio"] for row in candidate_rows], dtype=float
    )
    ratio_gate = float(requirements["min_leave_one_out_train_to_parameter_ratio"])
    bar_colors = ["#0f4c81" if ratio >= ratio_gate else "#d1495b" for ratio in ratios]
    bars = ax2.bar(np.arange(len(labels)), ratios, color=bar_colors)
    for bar, row in zip(bars, candidate_rows, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(bar.get_height(), 0.05) + 0.05,
            f"{row['leave_one_out_train_cases_per_fold']}/{row['n_parameters']}",
            ha="center",
            fontsize=9,
        )
    ax2.axhline(
        ratio_gate,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=f"{ratio_gate:g}x gate",
    )
    ax2.set_xticks(np.arange(len(labels)), labels)
    ax2.set_ylabel("LOO train cases / parameters")
    ax2.set_title("Candidate data-volume guard")
    ax2.set_ylim(0.0, max(float(np.max(ratios)), ratio_gate) + 0.75)
    ax2.grid(True, axis="y", alpha=0.24)
    ax2.legend(loc="upper right", fontsize=8)

    blockers = list(report["promotion_gate"]["blockers"])
    text_lines = [
        "Promotion gate: "
        + ("PASS" if report["promotion_gate"]["passed"] else "BLOCKED"),
        "",
        "Blockers:",
        *(f"- {item.replace('_', ' ')}" for item in blockers[:5]),
    ]
    excluded = list(report.get("excluded_validated_nonlinear_cases", []))
    if excluded:
        text_lines.extend(["", "Validated but excluded:"])
        for row in excluded[:3]:
            text_lines.append(f"- {row['case']}: {row['reason']}")
    ax3.axis("off")
    ax3.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=9.5,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
        },
    )
    ax3.set_title("Scope and exclusions")

    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    paths = {"png": str(out_path)}
    if write_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        paths["pdf"] = str(pdf_path)
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["json"] = str(json_path)
    return paths


def build_dataset_sufficiency_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--title", default="Quasilinear dataset-sufficiency gate")
    parser.add_argument(
        "--no-pdf", action="store_true", help="Only write PNG and JSON artifacts."
    )
    return parser


def dataset_sufficiency_main(argv: list[str] | None = None) -> int:
    args = build_dataset_sufficiency_parser().parse_args(argv)
    report = build_dataset_sufficiency_report()
    paths = write_dataset_sufficiency_figure(
        report,
        out=args.out,
        title=args.title,
        write_pdf=not args.no_pdf,
    )
    print(f"saved {paths['png']}")
    if "pdf" in paths:
        print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    print(
        "promotion_gate_passed={passed} blockers={blockers}".format(
            passed=report["promotion_gate"]["passed"],
            blockers=",".join(report["promotion_gate"]["blockers"]) or "none",
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dataset = build_dataset_sufficiency_parser()
    dataset.description = "Render the quasilinear dataset-sufficiency gate."
    sub_dataset = sub.add_parser(
        "dataset-sufficiency",
        help="render quasilinear dataset-sufficiency gate artifacts",
        parents=[dataset],
        add_help=False,
    )
    sub_dataset.set_defaults(func=dataset_sufficiency_main)

    model = build_model_selection_parser()
    model.description = "Render the quasilinear model-selection status."
    sub_model = sub.add_parser(
        "model-selection-status",
        help="render quasilinear model-selection status artifacts",
        parents=[model],
        add_help=False,
    )
    sub_model.set_defaults(func=model_selection_main)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        build_parser().parse_args(args)
        return 0
    command, *remaining = args
    if command == "dataset-sufficiency":
        return dataset_sufficiency_main(remaining)
    if command == "model-selection-status":
        return model_selection_main(remaining)
    build_parser().error(f"unknown command {command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
