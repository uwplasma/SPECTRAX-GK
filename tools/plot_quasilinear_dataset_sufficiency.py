#!/usr/bin/env python3
"""Audit whether the quasilinear calibration set is large enough for promotion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # type: ignore[import-untyped]  # noqa: E402

from plot_quasilinear_saturation_rule_sweep import (  # noqa: E402
    DEFAULT_CASES,
    SaturationCase,
    nonlinear_input_validation_report,
)  # type: ignore[import-not-found]


ROOT = Path(__file__).resolve().parents[1]


CANDIDATE_PARAMETER_COUNTS = {
    "linear_weight": 1,
    "shape_power_law": 2,
    "linear_state_ridge": 5,
}

CANDIDATE_LABELS = {
    "linear_weight": "linear weight",
    "shape_power_law": "shape power law",
    "linear_state_ridge": "linear-state ridge",
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


def _nonlinear_index_rows(path: Path | None, used_gate_cases: set[str]) -> list[dict[str, Any]]:
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
    nonlinear_index: str | Path | None = ROOT / "docs/_static/nonlinear_window_statistics.json",
    candidate_gate: str | Path | None = ROOT / "docs/_static/quasilinear_candidate_uncertainty.json",
    saturation_gate: str | Path | None = ROOT / "docs/_static/quasilinear_saturation_rule_sweep.json",
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

    train_geometries = sorted({row["geometry"] for row in case_rows if row["split"] == "train"})
    holdout_geometries = sorted({row["geometry"] for row in case_rows if row["split"] == "holdout"})
    all_geometries = sorted({row["geometry"] for row in case_rows})
    used_gate_cases = {str(row["gate_case"]) for row in case_rows}
    excluded_cases = _nonlinear_index_rows(Path(nonlinear_index) if nonlinear_index is not None else None, used_gate_cases)

    n_cases = len(case_rows)
    loo_train_cases_per_fold = max(n_cases - 1, 0)
    candidate_rows = []
    for name, n_parameters in CANDIDATE_PARAMETER_COUNTS.items():
        ratio = float("inf") if n_parameters == 0 else float(loo_train_cases_per_fold / n_parameters)
        candidate_rows.append(
            {
                "candidate": name,
                "label": CANDIDATE_LABELS[name],
                "n_parameters": int(n_parameters),
                "leave_one_out_train_cases_per_fold": int(loo_train_cases_per_fold),
                "train_to_parameter_ratio": ratio,
                "min_train_to_parameter_ratio": float(min_leave_one_out_train_to_parameter_ratio),
                "data_volume_passed": bool(ratio >= min_leave_one_out_train_to_parameter_ratio),
            }
        )

    requirements = {
        "validated_input_gates": bool(input_validation["passed"]),
        "minimum_total_electrostatic_cases": n_cases >= min_total_electrostatic_cases,
        "minimum_explicit_train_geometries": len(train_geometries) >= min_explicit_train_geometries,
        "minimum_holdout_geometries": len(holdout_geometries) >= min_holdout_geometries,
        "candidate_data_volume": any(bool(row["data_volume_passed"]) for row in candidate_rows),
    }
    blockers = [name for name, passed in requirements.items() if not passed]

    candidate_gate_payload = _load_promotion_gate(Path(candidate_gate) if candidate_gate is not None else None)
    saturation_gate_payload = _load_promotion_gate(Path(saturation_gate) if saturation_gate is not None else None)
    downstream_gates = {
        "candidate_uncertainty": candidate_gate_payload,
        "saturation_rule_sweep": saturation_gate_payload,
    }
    downstream_passed = all(gate is not None and bool(gate["passed"]) for gate in downstream_gates.values())
    if not downstream_passed:
        blockers.append("downstream_candidate_skill_gates_not_passed")

    return {
        "kind": "quasilinear_dataset_sufficiency",
        "claim_level": "promotion_blocked_until_more_converged_electrostatic_holdouts",
        "input_validation": input_validation,
        "requirements": {
            "min_total_electrostatic_cases": int(min_total_electrostatic_cases),
            "min_explicit_train_geometries": int(min_explicit_train_geometries),
            "min_holdout_geometries": int(min_holdout_geometries),
            "min_leave_one_out_train_to_parameter_ratio": float(min_leave_one_out_train_to_parameter_ratio),
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
            "electrostatic-compatible, and downstream held-out skill gates pass."
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
        ax0.scatter(0.0, idx, s=130, color=color, marker=marker, edgecolor="white", linewidth=0.8)
        ax0.text(0.08, idx, f"{row['geometry']} / {row['split']}", va="center", fontsize=9)
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
    ax1.bar(x + 0.18, required, width=0.36, color="#e5e7eb", edgecolor="#374151", label="required")
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
    ratios = np.asarray([row["train_to_parameter_ratio"] for row in candidate_rows], dtype=float)
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
    ax2.axhline(ratio_gate, color="#111827", linestyle="--", linewidth=1.2, label=f"{ratio_gate:g}x gate")
    ax2.set_xticks(np.arange(len(labels)), labels)
    ax2.set_ylabel("LOO train cases / parameters")
    ax2.set_title("Candidate data-volume guard")
    ax2.set_ylim(0.0, max(float(np.max(ratios)), ratio_gate) + 0.75)
    ax2.grid(True, axis="y", alpha=0.24)
    ax2.legend(loc="upper right", fontsize=8)

    blockers = list(report["promotion_gate"]["blockers"])
    text_lines = [
        "Promotion gate: " + ("PASS" if report["promotion_gate"]["passed"] else "BLOCKED"),
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
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
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
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["json"] = str(json_path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "docs/_static/quasilinear_dataset_sufficiency.png"))
    parser.add_argument("--title", default="Quasilinear dataset-sufficiency gate")
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG and JSON artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
