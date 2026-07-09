#!/usr/bin/env python3
"""Render quasilinear model-development claim-boundary artifacts.

This command-family owns the lightweight quasilinear model-development panels
that guard absolute-flux claims: dataset sufficiency, model-selection
status, screening skill, stellarator usefulness, and residual anatomy. Keeping
the family in one module makes the release artifact path easier to discover
without changing the generated artifacts.
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


# Consolidated screening-skill artifact implementation
_screening_ROOT = Path(__file__).resolve().parents[2]
_screening_STATIC = _screening_ROOT / "docs" / "_static"
_screening_MODEL_LABELS = {
    "positive_mixing_length": "positive-growth ML",
    "linear_weight": "linear-weight fit",
    "absolute_growth_mixing_length": "absolute-growth ML",
    "spectral_envelope_ridge": "spectral-envelope ridge",
    "linear_state_ridge": "linear-state ridge",
}
_screening_MODEL_COLORS = {
    "positive_mixing_length": "#7f1d1d",
    "linear_weight": "#6b7280",
    "absolute_growth_mixing_length": "#b45309",
    "spectral_envelope_ridge": "#0f766e",
    "linear_state_ridge": "#2563eb",
}
_screening_CASE_LABELS = {
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
_screening_DEFAULT_MODELS = (
    "positive_mixing_length",
    "linear_weight",
    "absolute_growth_mixing_length",
    "spectral_envelope_ridge",
    "linear_state_ridge",
)


def _screening__load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _screening__json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _screening__json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_screening__json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _screening__json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _screening__rankdata(values: np.ndarray) -> np.ndarray:
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


def _screening__pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _screening__spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    return _screening__pearson(_screening__rankdata(x), _screening__rankdata(y))


def _screening__pairwise_order_accuracy(
    predicted: np.ndarray, observed: np.ndarray
) -> tuple[float, int, int]:
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
        return (float("nan"), correct, total)
    return (float(correct / total), correct, total)


def _screening__relative_errors(
    predicted: np.ndarray, observed: np.ndarray, floor: float
) -> np.ndarray:
    return np.abs(predicted - observed) / np.maximum(np.abs(observed), floor)


def _screening__candidate_rows(
    candidate_report: dict[str, Any], candidate: str
) -> dict[str, dict[str, Any]]:
    payload = candidate_report["candidates"][candidate]
    return {str(row["holdout_case"]): row for row in payload["rows"]}


def _screening_build_report(
    *,
    saturation_report: Path = _screening_STATIC
    / "quasilinear_saturation_rule_sweep.json",
    candidate_report: Path = _screening_STATIC
    / "quasilinear_candidate_uncertainty.json",
    observed_floor: float = 1e-12,
    log_floor: float = 0.001,
    spearman_gate: float = 0.75,
    pairwise_order_gate: float = 0.75,
    absolute_error_gate: float = 0.35,
) -> dict[str, Any]:
    """Build screening/ranking skill metrics from existing QL/nonlinear reports."""
    sat = _screening__load_json(saturation_report)
    cand = _screening__load_json(candidate_report)
    cases = list(sat["cases"])
    case_names = [str(row["case"]) for row in cases]
    observed = np.asarray(
        [float(row["observed_heat_flux"]) for row in cases], dtype=float
    )
    holdout_mask = np.asarray(
        [str(row["split"]) == "holdout" for row in cases], dtype=bool
    )
    predictions: dict[str, np.ndarray] = {
        rule: np.asarray(sat["rules"][rule]["predicted_heat_flux"], dtype=float)
        for rule in (
            "positive_mixing_length",
            "linear_weight",
            "absolute_growth_mixing_length",
        )
    }
    for candidate in ("linear_weight", "spectral_envelope_ridge", "linear_state_ridge"):
        rows = _screening__candidate_rows(cand, candidate)
        predictions[candidate] = np.asarray(
            [float(rows[name]["predicted_heat_flux"]) for name in case_names],
            dtype=float,
        )
    model_rows: list[dict[str, Any]] = []
    for model in _screening_DEFAULT_MODELS:
        predicted = predictions[model]
        relative_error = _screening__relative_errors(
            predicted, observed, observed_floor
        )
        holdout_relative_error = relative_error[holdout_mask]
        pairwise_accuracy, pairwise_correct, pairwise_total = (
            _screening__pairwise_order_accuracy(predicted, observed)
        )
        holdout_pairwise_accuracy, holdout_correct, holdout_total = (
            _screening__pairwise_order_accuracy(
                predicted[holdout_mask], observed[holdout_mask]
            )
        )
        spearman = _screening__spearman(predicted, observed)
        holdout_spearman = _screening__spearman(
            predicted[holdout_mask], observed[holdout_mask]
        )
        log_pearson = _screening__pearson(
            np.log(np.maximum(predicted, log_floor)),
            np.log(np.maximum(observed, log_floor)),
        )
        holdout_log_pearson = _screening__pearson(
            np.log(np.maximum(predicted[holdout_mask], log_floor)),
            np.log(np.maximum(observed[holdout_mask], log_floor)),
        )
        absolute_gate_passed = bool(
            float(np.nanmean(relative_error)) <= absolute_error_gate
        )
        screening_gate_passed = bool(
            math.isfinite(spearman)
            and spearman >= spearman_gate
            and math.isfinite(pairwise_accuracy)
            and (pairwise_accuracy >= pairwise_order_gate)
        )
        holdout_screening_gate_passed = bool(
            math.isfinite(holdout_spearman)
            and holdout_spearman >= spearman_gate
            and math.isfinite(holdout_pairwise_accuracy)
            and (holdout_pairwise_accuracy >= pairwise_order_gate)
        )
        model_rows.append(
            {
                "model": model,
                "label": _screening_MODEL_LABELS[model],
                "mean_abs_relative_error": float(np.nanmean(relative_error)),
                "holdout_mean_abs_relative_error": float(
                    np.nanmean(holdout_relative_error)
                ),
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
    accepted_screening = [
        row["model"] for row in model_rows if bool(row["screening_gate_passed"])
    ]
    accepted_holdout_screening = [
        row["model"] for row in model_rows if bool(row["holdout_screening_gate_passed"])
    ]
    mean_error_gate_models = [
        row["model"] for row in model_rows if bool(row["absolute_flux_gate_passed"])
    ]
    best_screening = max(
        model_rows,
        key=lambda row: (
            float("-inf") if row["spearman"] is None else float(row["spearman"]),
            float("-inf")
            if row["pairwise_order_accuracy"] is None
            else float(row["pairwise_order_accuracy"]),
            -float(row["mean_abs_relative_error"]),
        ),
    )
    best_holdout_screening = max(
        model_rows,
        key=lambda row: (
            float("-inf")
            if row["holdout_spearman"] is None
            else float(row["holdout_spearman"]),
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
            "label": _screening_CASE_LABELS.get(
                case_names[idx], case_names[idx].replace("_", " ")
            ),
            "split": str(case["split"]),
            "geometry": str(case["geometry"]),
            "observed_heat_flux": float(observed[idx]),
        }
        for model in _screening_DEFAULT_MODELS:
            row[f"{model}_prediction"] = float(predictions[model][idx])
        case_rows.append(row)
    return {
        "kind": "quasilinear_screening_skill",
        "claim_level": "screening_correlation_model_development_not_absolute_flux_promotion",
        "source_artifacts": {
            "saturation_rule_sweep": str(
                saturation_report.relative_to(_screening_ROOT)
            ),
            "candidate_uncertainty": str(candidate_report.relative_to(_screening_ROOT)),
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


def _screening_write_csv(report: dict[str, Any], path: Path) -> None:
    fields = list(report["models"][0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["models"])


def _screening_write_figure(
    report: dict[str, Any], *, out: Path, title: str, dpi: int = 220
) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    model_rows = list(report["models"])
    case_rows = list(report["cases"])
    best = report["gates"]["best_screening_model"]
    observed = np.asarray([row["observed_heat_flux"] for row in case_rows], dtype=float)
    best_pred = np.asarray(
        [row[f"{best}_prediction"] for row in case_rows], dtype=float
    )
    linear_pred = np.asarray(
        [row["linear_weight_prediction"] for row in case_rows], dtype=float
    )
    simple_pred = np.asarray(
        [row["positive_mixing_length_prediction"] for row in case_rows], dtype=float
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), constrained_layout=True)
    ax0, ax1, ax2 = axes
    floor = 0.001
    positive = np.concatenate(
        [
            observed[observed > 0.0],
            best_pred[best_pred > 0.0],
            linear_pred[linear_pred > 0.0],
            simple_pred[simple_pred > 0.0],
        ]
    )
    lo = max(float(np.min(positive)) * 0.35, floor)
    hi = float(np.max(positive)) * 2.2
    ax0.plot(
        [lo, hi], [lo, hi], linestyle="--", color="0.2", linewidth=1.3, label="1:1"
    )
    ax0.fill_between(
        [lo, hi],
        [lo / 2, hi / 2],
        [lo * 2, hi * 2],
        color="0.86",
        alpha=0.55,
        label="factor 2",
    )
    ax0.scatter(
        observed,
        np.maximum(best_pred, floor),
        s=75,
        color=_screening_MODEL_COLORS[best],
        edgecolor="black",
        label=_screening_MODEL_LABELS[best],
        zorder=4,
    )
    ax0.scatter(
        observed,
        np.maximum(linear_pred, floor),
        s=54,
        color=_screening_MODEL_COLORS["linear_weight"],
        edgecolor="white",
        label=_screening_MODEL_LABELS["linear_weight"],
        zorder=3,
    )
    ax0.scatter(
        observed,
        np.maximum(simple_pred, floor),
        s=65,
        color=_screening_MODEL_COLORS["positive_mixing_length"],
        marker="x",
        label=_screening_MODEL_LABELS["positive_mixing_length"],
        zorder=3,
    )
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
    ax0.set_xlabel("Observed nonlinear late-window $\\langle Q_i\\rangle$")
    ax0.set_ylabel("Model prediction")
    ax0.set_title("Best screened model vs nonlinear windows")
    ax0.legend(loc="upper left", fontsize=8.1, frameon=True)
    labels = [row["label"] for row in model_rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.26
    spearman = np.asarray([row["spearman"] for row in model_rows], dtype=float)
    pairwise = np.asarray(
        [row["pairwise_order_accuracy"] for row in model_rows], dtype=float
    )
    mare = np.asarray(
        [row["mean_abs_relative_error"] for row in model_rows], dtype=float
    )
    absolute_skill = 1.0 - np.minimum(mare, 1.5) / 1.5
    ax1.axhline(
        report["gates"]["spearman_gate"],
        color="#0f766e",
        linestyle="--",
        linewidth=1.1,
        label="rank gate",
    )
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
        f"screening gate: {', '.join(report['gates']['accepted_screening_models']) or 'none'}\nmean-error gate: {', '.join(report['gates']['mean_error_gate_models']) or 'none'}\nabsolute promotion: none",
        transform=ax1.transAxes,
        fontsize=8.4,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.92,
        },
    )
    holdout_spearman = np.asarray(
        [row["holdout_spearman"] for row in model_rows], dtype=float
    )
    holdout_pairwise = np.asarray(
        [row["holdout_pairwise_order_accuracy"] for row in model_rows], dtype=float
    )
    holdout_mare = np.asarray(
        [row["holdout_mean_abs_relative_error"] for row in model_rows], dtype=float
    )
    holdout_absolute_skill = 1.0 - np.minimum(holdout_mare, 1.5) / 1.5
    ax2.axhline(
        report["gates"]["spearman_gate"],
        color="#0f766e",
        linestyle="--",
        linewidth=1.1,
        label="rank gate",
    )
    ax2.bar(
        x - width, holdout_spearman, width, color="#0f766e", label="holdout Spearman"
    )
    ax2.bar(x, holdout_pairwise, width, color="#2563eb", label="holdout pairwise")
    ax2.bar(
        x + width,
        holdout_absolute_skill,
        width,
        color="#9ca3af",
        label="holdout abs. skill",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.set_ylim(-0.65, 1.05)
    ax2.set_ylabel("Held-out metric")
    ax2.set_title("Held-out-only promotion check")
    ax2.legend(loc="lower right", fontsize=8.0, frameon=True)
    ax2.text(
        0.02,
        0.05,
        f"heldout screening gate: {', '.join(report['gates']['accepted_holdout_screening_models']) or 'none'}\nbest heldout model: {_screening_MODEL_LABELS[report['gates']['best_holdout_screening_model']]}\nneeds more nonlinear holdouts",
        transform=ax2.transAxes,
        fontsize=8.4,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.92,
        },
    )
    fig.suptitle(title, fontsize=13.6, fontweight="semibold")
    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    json_path.write_text(
        json.dumps(_screening__json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _screening_write_csv(report, csv_path)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def _screening_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_screening_STATIC / "quasilinear_screening_skill.png",
    )
    parser.add_argument(
        "--title", default="Quasilinear screening skill needs nonlinear calibration"
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _screening_main() -> None:
    args = _screening_parse_args()
    report = _screening_build_report()
    paths = _screening_write_figure(
        report, out=args.out, title=args.title, dpi=args.dpi
    )
    print(json.dumps(paths, indent=2))


# Consolidated stellarator-usefulness artifact implementation
_stellarator_ROOT = Path(__file__).resolve().parents[2]
_stellarator_STATIC = _stellarator_ROOT / "docs" / "_static"
_stellarator_CASE_LABELS = {
    "hsx_nonlinear_window": "HSX",
    "w7x_nonlinear_window": "W7-X",
    "cyclone_miller_long_window": "Cyclone Miller",
    "dshape_external_vmec_t250_window": "D-shaped VMEC",
    "updown_asym_external_vmec_t450_window": "Up-down VMEC",
    "circular_external_vmec_t450_window": "Circular VMEC",
    "itermodel_external_vmec_t350_window": "ITERModel VMEC",
    "cyclone_long_window": "Cyclone",
    "cth_like_external_vmec_t700_high_grid_ensemble": "CTH-like VMEC",
    "shaped_tokamak_pressure_external_vmec_t650_high_grid_window": "Shaped-pressure VMEC",
    "qp_diag_nfp2_m4_final_t250": "QP VMEC",
}
_stellarator_MODEL_LABELS = {
    "positive_mixing_length": "positive-growth ML",
    "linear_weight": "linear-weight fit",
    "spectral_envelope_ridge": "spectral-envelope ridge",
}
_stellarator_MODEL_COLORS = {
    "positive_mixing_length": "#7f1d1d",
    "linear_weight": "#6b7280",
    "spectral_envelope_ridge": "#0f766e",
}
_stellarator_MODEL_MARKERS = {
    "positive_mixing_length": "X",
    "linear_weight": "o",
    "spectral_envelope_ridge": "D",
}
_stellarator_STELLARATOR_CASES = ("hsx_nonlinear_window", "w7x_nonlinear_window")


def _stellarator__load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stellarator__json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stellarator__json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stellarator__json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _stellarator__json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _stellarator__relative_error(
    predicted: float | None, observed: float | None, *, floor: float = 1e-12
) -> float | None:
    if predicted is None or observed is None:
        return None
    if not math.isfinite(predicted) or not math.isfinite(observed):
        return None
    return abs(predicted - observed) / max(abs(observed), floor)


def _stellarator__candidate_predictions(
    candidate_report: dict[str, Any], candidate: str
) -> dict[str, float]:
    payload = candidate_report["candidates"][candidate]
    return {
        str(row["holdout_case"]): float(row["predicted_heat_flux"])
        for row in payload["rows"]
    }


def _stellarator__candidate_intervals(
    candidate_report: dict[str, Any], candidate: str
) -> dict[str, tuple[float, float]]:
    payload = candidate_report["candidates"][candidate]
    return {
        str(row["holdout_case"]): (
            float(row["prediction_interval_low"]),
            float(row["prediction_interval_high"]),
        )
        for row in payload["rows"]
    }


def _stellarator_build_report(
    *,
    saturation_report: Path = _stellarator_STATIC
    / "quasilinear_saturation_rule_sweep.json",
    candidate_report: Path = _stellarator_STATIC
    / "quasilinear_candidate_uncertainty.json",
    model_selection_status: Path = _stellarator_STATIC
    / "quasilinear_model_selection_status.json",
    qa_audit: Path = _stellarator_STATIC
    / "qa_no_ess_to_optimized_nonlinear_audit.json",
    qh_gate: Path = _stellarator_STATIC
    / "external_vmec_qh_high_grid_convergence_gate.json",
) -> dict[str, Any]:
    """Build a claim-scoped stellarator quasilinear usefulness report."""
    sat = _stellarator__load_json(saturation_report)
    cand = _stellarator__load_json(candidate_report)
    status = _stellarator__load_json(model_selection_status)
    qa = _stellarator__load_json(qa_audit)
    qh = _stellarator__load_json(qh_gate)
    simple_predictions = {
        rule: [float(x) for x in sat["rules"][rule]["predicted_heat_flux"]]
        for rule in ("positive_mixing_length", "linear_weight")
    }
    spectral_predictions = _stellarator__candidate_predictions(
        cand, "spectral_envelope_ridge"
    )
    spectral_intervals = _stellarator__candidate_intervals(
        cand, "spectral_envelope_ridge"
    )
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(sat["cases"]):
        case_name = str(case["case"])
        observed = float(case["observed_heat_flux"])
        row: dict[str, Any] = {
            "case": case_name,
            "label": _stellarator_CASE_LABELS.get(
                case_name, case_name.replace("_", " ")
            ),
            "geometry": str(case["geometry"]),
            "split": str(case["split"]),
            "observed_heat_flux": observed,
            "observed_heat_flux_std": case.get("observed_heat_flux_std"),
            "stellarator_family": case_name in _stellarator_STELLARATOR_CASES,
        }
        for model in ("positive_mixing_length", "linear_weight"):
            predicted = simple_predictions[model][idx]
            row[f"{model}_prediction"] = predicted
            row[f"{model}_relative_error"] = _stellarator__relative_error(
                predicted, observed
            )
        predicted = spectral_predictions.get(case_name)
        row["spectral_envelope_ridge_prediction"] = predicted
        row["spectral_envelope_ridge_relative_error"] = _stellarator__relative_error(
            predicted, observed
        )
        interval = spectral_intervals.get(case_name)
        if interval is not None:
            row["spectral_envelope_ridge_interval_low"] = interval[0]
            row["spectral_envelope_ridge_interval_high"] = interval[1]
            row["spectral_envelope_ridge_interval_contains_observed"] = (
                interval[0] <= observed <= interval[1]
            )
        rows.append(row)
    qa_comparison = dict(qa["comparison"])
    qh_gate_report = dict(qh["gate_report"])
    qh_pairwise = None
    for gate in qh_gate_report.get("gates", []):
        if (
            gate.get("metric")
            == "least_window_pairwise_heat_flux_symmetric_relative_difference"
        ):
            qh_pairwise = gate.get("observed")
            break
    metrics = dict(status["metrics"])
    simple_metrics = {
        rule: sat["rules"][rule]["holdout_mean_abs_relative_error"]
        for rule in (
            "positive_mixing_length",
            "linear_weight",
            "absolute_growth_mixing_length",
        )
    }
    best_model = "spectral_envelope_ridge"
    accepted_candidates = set(
        cand.get("promotion_gate", {}).get("accepted_candidates", [])
    )
    candidate_accepted = best_model in accepted_candidates and bool(
        status.get("passed", False)
    )
    claim = "Simple one-constant quasilinear rules do not transfer as absolute stellarator heat-flux predictors on the admitted portfolio. The spectral-envelope ridge model is the best current rank-screening candidate, but the expanded uncertainty/model-selection gate is not accepted; the frozen ledger now points to missing saturation physics rather than additional holdout collection before universal stellarator-flux claims."
    return {
        "kind": "quasilinear_stellarator_usefulness",
        "claim_level": "scoped_model_skill_summary_not_runtime_absolute_flux_predictor",
        "source_artifacts": {
            "saturation_rule_sweep": str(
                saturation_report.relative_to(_stellarator_ROOT)
            ),
            "candidate_uncertainty": str(
                candidate_report.relative_to(_stellarator_ROOT)
            ),
            "model_selection_status": str(
                model_selection_status.relative_to(_stellarator_ROOT)
            ),
            "qa_matched_nonlinear_audit": str(qa_audit.relative_to(_stellarator_ROOT)),
            "qh_high_grid_convergence_gate": str(
                qh_gate.relative_to(_stellarator_ROOT)
            ),
        },
        "models": {
            "positive_mixing_length": {
                "label": _stellarator_MODEL_LABELS["positive_mixing_length"],
                "holdout_mean_abs_relative_error": simple_metrics[
                    "positive_mixing_length"
                ],
                "accepted": False,
                "reason": "fails holdout gate and predicts zero for admitted HSX/W7-X finite nonlinear windows",
            },
            "linear_weight": {
                "label": _stellarator_MODEL_LABELS["linear_weight"],
                "mean_abs_relative_error": metrics[
                    "linear_weight_mean_abs_relative_error"
                ],
                "accepted": False,
                "reason": "beaten by spectral-envelope candidate and fails transport gate",
            },
            "spectral_envelope_ridge": {
                "label": _stellarator_MODEL_LABELS["spectral_envelope_ridge"],
                "mean_abs_relative_error": metrics["candidate_mean_abs_relative_error"],
                "prediction_interval_coverage": metrics[
                    "candidate_prediction_interval_coverage"
                ],
                "accepted": candidate_accepted,
                "reason": "best current rank-screening candidate; uncertainty/model-selection gate is not accepted",
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
                "status": "excluded from frozen QL calibration because grid/window convergence has not passed",
                "high_grid_gate_passed": bool(qh.get("passed", False)),
                "least_window_pairwise_heat_flux_symmetric_relative_difference": qh_pairwise,
            },
        },
        "best_current_model": best_model,
        "readme_sentence": claim,
        "notes": [
            "The plot uses only tracked JSON artifacts and does not refit any quasilinear model.",
            "QA and QH are shown as scope/status evidence, not as accepted quasilinear calibration points.",
            "A universal stellarator absolute-flux proxy now needs better saturation theory on the frozen admitted ledger.",
        ],
    }


def _stellarator_write_csv(report: dict[str, Any], path: Path) -> None:
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
        writer = csv.DictWriter(
            fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _stellarator_write_figure(
    report: dict[str, Any], *, out: Path, title: str, dpi: int = 220
) -> dict[str, str]:
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
        for model in _stellarator_MODEL_LABELS:
            pred = row.get(f"{model}_prediction")
            if (
                isinstance(pred, (float, int))
                and math.isfinite(float(pred))
                and (float(pred) > 0.0)
            ):
                positive_values.append(float(pred))
    lo = min(positive_values) * 0.35
    hi = max(positive_values) * 2.2
    ax0.plot(
        [lo, hi], [lo, hi], color="0.2", linewidth=1.4, linestyle="--", label="1:1"
    )
    ax0.fill_between(
        [lo, hi],
        [lo / 2.0, hi / 2.0],
        [lo * 2.0, hi * 2.0],
        color="0.85",
        alpha=0.55,
        label="factor 2",
    )
    ax0.fill_between(
        [lo, hi],
        [lo / 10.0, hi / 10.0],
        [lo * 10.0, hi * 10.0],
        color="0.93",
        alpha=0.55,
        label="factor 10",
    )
    for model, label in _stellarator_MODEL_LABELS.items():
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
            edgecolors.append(
                "black" if bool(row.get("stellarator_family")) else "white"
            )
            sizes.append(88 if bool(row.get("stellarator_family")) else 58)
        ax0.scatter(
            xs,
            ys,
            s=sizes,
            marker=_stellarator_MODEL_MARKERS[model],
            color=_stellarator_MODEL_COLORS[model],
            edgecolor=edgecolors,
            linewidth=1.0,
            label=label,
            zorder=4,
        )
    label_offsets = {
        "hsx_nonlinear_window": (10, 12),
        "w7x_nonlinear_window": (10, -16),
    }
    for row in rows:
        if row["case"] in _stellarator_STELLARATOR_CASES:
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
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#fecaca",
            "alpha": 0.9,
        },
    )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel("Observed nonlinear late-window $\\langle Q_i\\rangle$")
    ax0.set_ylabel("Quasilinear model prediction")
    ax0.set_title("Admitted train/holdout skill")
    ax0.legend(loc="upper left", fontsize=8.2, frameon=True, ncols=1)
    stellarator_rows = [
        row for row in rows if row["case"] in _stellarator_STELLARATOR_CASES
    ]
    labels = [row["label"] for row in stellarator_rows] + ["QA audit", "QH pilot"]
    x = np.arange(len(labels), dtype=float)
    width = 0.22
    observed = [float(row["observed_heat_flux"]) for row in stellarator_rows]
    spectral = [
        float(row["spectral_envelope_ridge_prediction"]) for row in stellarator_rows
    ]
    linear = [float(row["linear_weight_prediction"]) for row in stellarator_rows]
    positive_ml = [
        max(float(row["positive_mixing_length_prediction"]), 0.03)
        for row in stellarator_rows
    ]
    qa_status = report["stellarator_status"]["QA"]
    observed += [float(qa_status["optimized_heat_flux"]), np.nan]
    spectral += [np.nan, np.nan]
    linear += [np.nan, np.nan]
    positive_ml += [np.nan, np.nan]
    ax1.bar(
        x - 1.5 * width, observed, width, color="#0f172a", label="observed nonlinear"
    )
    ax1.bar(
        x - 0.5 * width,
        spectral,
        width,
        color=_stellarator_MODEL_COLORS["spectral_envelope_ridge"],
        label="spectral-envelope",
    )
    ax1.bar(
        x + 0.5 * width,
        linear,
        width,
        color=_stellarator_MODEL_COLORS["linear_weight"],
        label="linear-weight",
    )
    ax1.bar(
        x + 1.5 * width,
        positive_ml,
        width,
        color=_stellarator_MODEL_COLORS["positive_mixing_length"],
        label="positive-growth ML",
    )
    baseline = float(qa_status["baseline_heat_flux"])
    optimized = float(qa_status["optimized_heat_flux"])
    qa_x = x[2]
    ax1.scatter(
        [qa_x - 1.5 * width, qa_x - 1.5 * width],
        [baseline, optimized],
        color="#0f172a",
        s=[60, 80],
        zorder=5,
    )
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
        qh_x - 0.1,
        max(observed[:2]) * 0.72,
        "QH excluded\nfailed grid gate",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#7f1d1d",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#fecaca",
            "alpha": 0.9,
        },
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_ylabel("Ion heat flux / model units")
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
    json_path.write_text(
        json.dumps(_stellarator__json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _stellarator_write_csv(report, csv_path)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def _stellarator_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_stellarator_STATIC / "quasilinear_stellarator_usefulness.png",
    )
    parser.add_argument(
        "--title", default="Quasilinear models need nonlinear stellarator calibration"
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _stellarator_main() -> None:
    args = _stellarator_parse_args()
    report = _stellarator_build_report()
    paths = _stellarator_write_figure(
        report, out=args.out, title=args.title, dpi=args.dpi
    )
    print(json.dumps(paths, indent=2))


# Consolidated error-anatomy artifact implementation
_error_ROOT = Path(__file__).resolve().parents[2]
_error_DEFAULT_UNCERTAINTY = (
    _error_ROOT / "docs/_static/quasilinear_candidate_uncertainty.json"
)
_error_DEFAULT_SCREENING = _error_ROOT / "docs/_static/quasilinear_screening_skill.json"
_error_DEFAULT_SATURATION = (
    _error_ROOT / "docs/_static/quasilinear_saturation_rule_sweep.json"
)
_error_DEFAULT_OUT = _error_ROOT / "docs/_static/quasilinear_error_anatomy.png"
_error_DEFAULT_CORE_EXCLUDED_CASES = (
    "solovev_reference_repair_dt002_amp1em5_n48_t250",
    "shaped_tokamak_pressure_external_vmec_t650_high_grid_window",
)
_error_SHORT_LABELS = {
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


def _error__json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _error__json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_error__json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _error__json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _error__read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _error__finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _error__geometry_group(geometry: str) -> str:
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


def _error__case_metadata(
    screening: dict[str, Any], saturation: dict[str, Any]
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
                "observed_heat_flux": _error__finite_float(
                    item.get("observed_heat_flux")
                ),
                "observed_heat_flux_std": _error__finite_float(
                    item.get("observed_heat_flux_std")
                ),
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
                    "geometry": str(
                        item.get("geometry", metadata[case].get("geometry", ""))
                    ),
                    "split": str(item.get("split", metadata[case].get("split", ""))),
                    "label": str(
                        item.get("label", _error_SHORT_LABELS.get(case, case))
                    ),
                }
            )
    return metadata


def _error__model_metrics(screening: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in screening.get("models", []):
        if not isinstance(model, dict):
            continue
        rows.append(
            {
                "model": str(model.get("model", "")),
                "label": str(model.get("label", model.get("model", ""))),
                "mean_abs_relative_error": _error__finite_float(
                    model.get("mean_abs_relative_error")
                ),
                "holdout_mean_abs_relative_error": _error__finite_float(
                    model.get("holdout_mean_abs_relative_error")
                ),
                "spearman": _error__finite_float(model.get("spearman")),
                "holdout_spearman": _error__finite_float(model.get("holdout_spearman")),
                "pairwise_order_accuracy": _error__finite_float(
                    model.get("pairwise_order_accuracy")
                ),
                "holdout_pairwise_order_accuracy": _error__finite_float(
                    model.get("holdout_pairwise_order_accuracy")
                ),
                "absolute_flux_gate_passed": bool(
                    model.get("absolute_flux_gate_passed", False)
                ),
                "screening_gate_passed": bool(
                    model.get("screening_gate_passed", False)
                ),
                "holdout_screening_gate_passed": bool(
                    model.get("holdout_screening_gate_passed", False)
                ),
            }
        )
    return rows


def _error__average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        for pos in range(start, stop):
            ranks[order[pos]] = rank
        start = stop
    return ranks


def _error__pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    da = [value - mean_a for value in a]
    db = [value - mean_b for value in b]
    var_a = sum((value * value for value in da))
    var_b = sum((value * value for value in db))
    if var_a <= 0.0 or var_b <= 0.0:
        return float("nan")
    return sum((x * y for x, y in zip(da, db, strict=True))) / math.sqrt(var_a * var_b)


def _error__spearman(rows: list[dict[str, Any]]) -> float:
    observed = [float(row["observed_heat_flux"]) for row in rows]
    predicted = [float(row["predicted_heat_flux"]) for row in rows]
    return _error__pearson(
        _error__average_ranks(observed), _error__average_ranks(predicted)
    )


def _error__pairwise_order_accuracy(rows: list[dict[str, Any]]) -> float:
    total = 0
    correct = 0
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            observed_delta = float(left["observed_heat_flux"]) - float(
                right["observed_heat_flux"]
            )
            predicted_delta = float(left["predicted_heat_flux"]) - float(
                right["predicted_heat_flux"]
            )
            if observed_delta == 0.0 or predicted_delta == 0.0:
                continue
            total += 1
            correct += int(observed_delta * predicted_delta > 0.0)
    return correct / total if total else float("nan")


def _error__core_portfolio_gate(
    rows: list[dict[str, Any]],
    *,
    excluded_cases: tuple[str, ...] = _error_DEFAULT_CORE_EXCLUDED_CASES,
    transport_gate: float,
    interval_coverage_gate: float = 0.75,
    screening_gate: float = 0.75,
) -> dict[str, Any]:
    excluded = [row for row in rows if row["case"] in set(excluded_cases)]
    core = [row for row in rows if row["case"] not in set(excluded_cases)]
    holdout = [row for row in core if row["split"] == "holdout"]
    errors = [float(row["absolute_relative_error"]) for row in core]
    holdout_errors = [float(row["absolute_relative_error"]) for row in holdout]
    coverage = (
        sum((bool(row["prediction_interval_contains_observed"]) for row in core))
        / len(core)
        if core
        else float("nan")
    )
    mean_error = sum(errors) / len(errors) if errors else float("nan")
    holdout_mean_error = (
        sum(holdout_errors) / len(holdout_errors) if holdout_errors else float("nan")
    )
    spearman = _error__spearman(core)
    holdout_spearman = _error__spearman(holdout)
    pairwise = _error__pairwise_order_accuracy(core)
    holdout_pairwise = _error__pairwise_order_accuracy(holdout)
    blockers: list[str] = []
    if not (math.isfinite(mean_error) and mean_error <= transport_gate):
        blockers.append("core_mean_abs_relative_error_exceeds_gate")
    if not (math.isfinite(holdout_mean_error) and holdout_mean_error <= transport_gate):
        blockers.append("core_holdout_mean_abs_relative_error_exceeds_gate")
    if not (math.isfinite(coverage) and coverage >= interval_coverage_gate):
        blockers.append("core_interval_coverage_below_gate")
    return {
        "passed": not blockers,
        "claim_level": "scoped_core_portfolio_absolute_flux_diagnostic",
        "blockers": blockers,
        "transport_gate": transport_gate,
        "interval_coverage_gate": interval_coverage_gate,
        "screening_gate": screening_gate,
        "core_case_count": len(core),
        "core_holdout_count": len(holdout),
        "excluded_case_count": len(excluded),
        "excluded_cases": [
            {
                "case": str(row["case"]),
                "label": str(row["label"]),
                "absolute_relative_error": float(row["absolute_relative_error"]),
                "reason": "declared stress outlier retained outside the scoped core portfolio",
            }
            for row in excluded
        ],
        "core_mean_abs_relative_error": mean_error,
        "core_holdout_mean_abs_relative_error": holdout_mean_error,
        "core_max_abs_relative_error": max(errors) if errors else float("nan"),
        "core_prediction_interval_coverage": coverage,
        "core_spearman": spearman,
        "core_holdout_spearman": holdout_spearman,
        "core_pairwise_order_accuracy": pairwise,
        "core_holdout_pairwise_order_accuracy": holdout_pairwise,
        "screening_gate_passed": bool(
            math.isfinite(spearman)
            and math.isfinite(holdout_spearman)
            and math.isfinite(pairwise)
            and math.isfinite(holdout_pairwise)
            and (spearman >= screening_gate)
            and (holdout_spearman >= screening_gate)
            and (pairwise >= screening_gate)
            and (holdout_pairwise >= screening_gate)
        ),
        "claim_boundary": "This gate closes only the scoped core-portfolio absolute-flux diagnostic. The excluded stress cases remain visible negative evidence and prevent a universal absolute-flux claim.",
    }


def _error_build_error_anatomy_report(
    *,
    candidate_uncertainty: str | Path = _error_DEFAULT_UNCERTAINTY,
    screening_skill: str | Path = _error_DEFAULT_SCREENING,
    saturation_sweep: str | Path = _error_DEFAULT_SATURATION,
    candidate: str = "spectral_envelope_ridge",
    transport_gate: float | None = None,
) -> dict[str, Any]:
    """Return per-case residual anatomy for the current QL candidate."""
    uncertainty = _error__read_json(candidate_uncertainty)
    screening = _error__read_json(screening_skill)
    saturation = _error__read_json(saturation_sweep)
    candidates = uncertainty.get("candidates", {})
    if not isinstance(candidates, dict) or candidate not in candidates:
        raise ValueError(
            f"candidate {candidate!r} not found in {candidate_uncertainty}"
        )
    candidate_payload = candidates[candidate]
    if not isinstance(candidate_payload, dict):
        raise ValueError(f"candidate {candidate!r} payload is not an object")
    rows_in = candidate_payload.get("rows", [])
    if not isinstance(rows_in, list) or not rows_in:
        raise ValueError(f"candidate {candidate!r} has no rows")
    gate = (
        float(transport_gate)
        if transport_gate is not None
        else _error__finite_float(uncertainty.get("transport_gate"), default=0.35)
    )
    metadata = _error__case_metadata(screening, saturation)
    rows: list[dict[str, Any]] = []
    for item in rows_in:
        if not isinstance(item, dict):
            continue
        case = str(item.get("holdout_case", ""))
        meta = metadata.get(case, {"case": case})
        observed = _error__finite_float(item.get("observed_heat_flux"))
        predicted = _error__finite_float(item.get("predicted_heat_flux"))
        rel_error = _error__finite_float(item.get("absolute_relative_error"))
        ratio = (
            predicted / observed
            if observed and math.isfinite(observed)
            else float("nan")
        )
        geometry = str(meta.get("geometry", ""))
        row = {
            "case": case,
            "label": str(meta.get("label", _error_SHORT_LABELS.get(case, case))),
            "split": str(meta.get("split", "")),
            "geometry": geometry,
            "geometry_group": _error__geometry_group(geometry),
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
            "prediction_interval_low": _error__finite_float(
                item.get("prediction_interval_low")
            ),
            "prediction_interval_high": _error__finite_float(
                item.get("prediction_interval_high")
            ),
        }
        rows.append(row)
    rows.sort(key=lambda row: float(row["absolute_relative_error"]), reverse=True)
    finite_errors = np.asarray(
        [
            row["absolute_relative_error"]
            for row in rows
            if math.isfinite(float(row["absolute_relative_error"]))
        ],
        dtype=float,
    )
    total_error = float(np.sum(finite_errors)) if finite_errors.size else float("nan")
    for row in rows:
        err = float(row["absolute_relative_error"])
        row["error_budget_fraction"] = (
            err / total_error if total_error > 0.0 else float("nan")
        )
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
    model_rows = _error__model_metrics(screening)
    model_rows.sort(key=lambda row: float(row["mean_abs_relative_error"]))
    screening_gates = (
        screening.get("gates", {})
        if isinstance(screening.get("gates", {}), dict)
        else {}
    )
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
    if any((bool(row["above_transport_gate"]) for row in rows)):
        blockers.append("case_residuals_exceed_transport_gate")
    dominant_residuals = [
        {
            "case": str(row["case"]),
            "label": str(row["label"]),
            "geometry_group": str(row["geometry_group"]),
            "absolute_relative_error": float(row["absolute_relative_error"]),
            "error_budget_fraction": float(row["error_budget_fraction"]),
            "prediction_to_observed_ratio": float(row["prediction_to_observed_ratio"]),
        }
        for row in rows[:3]
    ]
    core_gate = _error__core_portfolio_gate(rows, transport_gate=gate)
    return {
        "kind": "quasilinear_error_anatomy",
        "claim_level": "model_development_residual_anatomy_not_absolute_flux_promotion",
        "candidate": candidate,
        "transport_gate": gate,
        "case_count": len(rows),
        "holdout_count": sum((1 for row in rows if row["split"] == "holdout")),
        "candidate_mean_abs_relative_error": _error__finite_float(
            candidate_payload.get("mean_abs_relative_error")
        ),
        "candidate_prediction_interval_coverage": _error__finite_float(
            candidate_payload.get("prediction_interval_coverage")
        ),
        "rows": rows,
        "geometry_group_summary": group_rows,
        "model_summary": model_rows,
        "promotion_gate": {
            "passed": False,
            "blockers": blockers,
            "claim_boundary": "This artifact diagnoses why quasilinear model candidates fail or pass. It does not promote a runtime/TOML absolute-flux predictor.",
        },
        "core_portfolio_gate": core_gate,
        "frozen_ledger_policy": {
            "additional_holdout_collection_active": False,
            "ledger_case_count": len(rows),
            "ledger_holdout_count": sum(
                (1 for row in rows if row["split"] == "holdout")
            ),
            "active_next_step": "use the passing scoped core portfolio for current QL examples; keep stress outliers as deferred model-physics evidence",
            "do_not_promote_until": [
                "transport mean relative error gate passes",
                "prediction interval coverage gate passes",
                "screening/rank gates pass for the intended claim scope",
                "promotion guardrails still classify the candidate as non-diagnostic",
            ],
        },
        "dominant_residuals": dominant_residuals,
        "model_development_requirements": [
            "reduce the external-axisymmetric residual budget, especially the Solovev low-flux stress case, without loosening the 0.35 transport gate",
            "add saturation-amplitude physics that separates pressure shaping, axisymmetric VMEC stress cases, and stellarator benchmark windows",
            "preserve the comparatively good HSX/W7-X errors while improving the frozen-ledger mean and held-out rank/correlation metrics",
        ],
        "source_artifacts": {
            "candidate_uncertainty": str(Path(candidate_uncertainty)),
            "screening_skill": str(Path(screening_skill)),
            "saturation_sweep": str(Path(saturation_sweep)),
        },
    }


def _error_write_error_anatomy_figure(
    report: dict[str, Any],
    *,
    out: str | Path = _error_DEFAULT_OUT,
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
    ax0.plot(
        [lim_lo, lim_hi], [lim_lo, lim_hi], color="0.25", linestyle="--", linewidth=1.3
    )
    ax0.scatter(
        observed, predicted, c=colors, s=52, edgecolor="white", linewidth=0.7, zorder=3
    )
    for row in rows[:5]:
        ax0.annotate(
            str(row["label"]),
            (float(row["observed_heat_flux"]), float(row["predicted_heat_flux"])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.16",
                "fc": "white",
                "ec": "0.82",
                "alpha": 0.82,
            },
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
    ax1.axvline(
        gate, color="#111827", linestyle="--", linewidth=1.2, label=f"{gate:.2f} gate"
    )
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
    subtitle = f"{report['candidate']} | mean error = {float(report['candidate_mean_abs_relative_error']):.3f} | core mean = {float(report['core_portfolio_gate']['core_mean_abs_relative_error']):.3f} {('PASS' if report['core_portfolio_gate']['passed'] else 'FAIL')}"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13.5, fontweight="bold")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    outputs = {"png": str(out_path)}
    if write_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs["pdf"] = str(pdf_path)
    plt.close(fig)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(_error__json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def _error_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-uncertainty", type=Path, default=_error_DEFAULT_UNCERTAINTY
    )
    parser.add_argument(
        "--screening-skill", type=Path, default=_error_DEFAULT_SCREENING
    )
    parser.add_argument(
        "--saturation-sweep", type=Path, default=_error_DEFAULT_SATURATION
    )
    parser.add_argument("--candidate", default="spectral_envelope_ridge")
    parser.add_argument("--out", type=Path, default=_error_DEFAULT_OUT)
    parser.add_argument("--title", default="Quasilinear residual anatomy")
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument("--write-pdf", action="store_true")
    return parser


def _error_main(argv: list[str] | None = None) -> int:
    args = _error_build_parser().parse_args(argv)
    report = _error_build_error_anatomy_report(
        candidate_uncertainty=args.candidate_uncertainty,
        screening_skill=args.screening_skill,
        saturation_sweep=args.saturation_sweep,
        candidate=str(args.candidate),
    )
    outputs = _error_write_error_anatomy_figure(
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


build_screening_skill_report = _screening_build_report
write_screening_skill_figure = _screening_write_figure
write_screening_skill_csv = _screening_write_csv
build_stellarator_usefulness_report = _stellarator_build_report
write_stellarator_usefulness_figure = _stellarator_write_figure
write_stellarator_usefulness_csv = _stellarator_write_csv
build_error_anatomy_report = _error_build_error_anatomy_report
write_error_anatomy_figure = _error_write_error_anatomy_figure


def build_screening_skill_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render quasilinear screening-skill artifacts."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_screening_STATIC / "quasilinear_screening_skill.png",
    )
    parser.add_argument(
        "--title", default="Quasilinear screening skill needs nonlinear calibration"
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def screening_skill_main(argv: list[str] | None = None) -> int:
    args = build_screening_skill_parser().parse_args(argv)
    report = build_screening_skill_report()
    paths = write_screening_skill_figure(
        report, out=args.out, title=args.title, dpi=args.dpi
    )
    print(json.dumps(paths, indent=2))
    return 0


def build_stellarator_usefulness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render stellarator quasilinear-usefulness artifacts."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_stellarator_STATIC / "quasilinear_stellarator_usefulness.png",
    )
    parser.add_argument(
        "--title", default="Quasilinear models need nonlinear stellarator calibration"
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def stellarator_usefulness_main(argv: list[str] | None = None) -> int:
    args = build_stellarator_usefulness_parser().parse_args(argv)
    report = build_stellarator_usefulness_report()
    paths = write_stellarator_usefulness_figure(
        report, out=args.out, title=args.title, dpi=args.dpi
    )
    print(json.dumps(paths, indent=2))
    return 0


def build_error_anatomy_parser() -> argparse.ArgumentParser:
    return _error_build_parser()


def error_anatomy_main(argv: list[str] | None = None) -> int:
    args = build_error_anatomy_parser().parse_args(argv)
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

    screening = build_screening_skill_parser()
    screening.description = "Render quasilinear screening-skill artifacts."
    sub_screening = sub.add_parser(
        "screening-skill",
        help="render quasilinear screening/ranking skill artifacts",
        parents=[screening],
        add_help=False,
    )
    sub_screening.set_defaults(func=screening_skill_main)

    stellarator = build_stellarator_usefulness_parser()
    stellarator.description = "Render stellarator quasilinear-usefulness artifacts."
    sub_stellarator = sub.add_parser(
        "stellarator-usefulness",
        help="render stellarator quasilinear usefulness artifacts",
        parents=[stellarator],
        add_help=False,
    )
    sub_stellarator.set_defaults(func=stellarator_usefulness_main)

    error = build_error_anatomy_parser()
    error.description = "Render quasilinear residual-error anatomy artifacts."
    sub_error = sub.add_parser(
        "error-anatomy",
        help="render quasilinear residual-error anatomy artifacts",
        parents=[error],
        add_help=False,
    )
    sub_error.set_defaults(func=error_anatomy_main)
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
    if command == "screening-skill":
        return screening_skill_main(remaining)
    if command == "stellarator-usefulness":
        return stellarator_usefulness_main(remaining)
    if command == "error-anatomy":
        return error_anatomy_main(remaining)
    build_parser().error(f"unknown command {command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
