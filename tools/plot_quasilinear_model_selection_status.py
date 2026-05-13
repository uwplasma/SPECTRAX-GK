#!/usr/bin/env python3
"""Render the quasilinear model-selection claim-boundary status."""

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
from spectraxgk.quasilinear_model_selection import (  # noqa: E402
    DEFAULT_REQUIRED_CANDIDATE,
    build_quasilinear_model_selection_status_from_paths,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs/_static/quasilinear_model_selection_status.png"
DEFAULT_DATASET = ROOT / "docs/_static/quasilinear_dataset_sufficiency.json"
DEFAULT_CANDIDATE = ROOT / "docs/_static/quasilinear_candidate_uncertainty.json"
DEFAULT_CALIBRATION_REPORTS = tuple(
    sorted((ROOT / "docs/_static").glob("quasilinear_*train_holdout_report.json"))
)

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
    finite_values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
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
    bars = ax0.bar(x, [float(v) if v is not None else np.nan for v in values], color=colors)
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


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--required-candidate", default=DEFAULT_REQUIRED_CANDIDATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default="Quasilinear model-selection status")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-pdf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = (
        tuple(args.calibration_report)
        if args.calibration_report is not None
        else DEFAULT_CALIBRATION_REPORTS
    )
    status = build_quasilinear_model_selection_status_from_paths(
        dataset_sufficiency=args.dataset,
        candidate_uncertainty=args.candidate,
        calibration_reports=reports,
        required_candidate=args.required_candidate,
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


if __name__ == "__main__":
    raise SystemExit(main())
