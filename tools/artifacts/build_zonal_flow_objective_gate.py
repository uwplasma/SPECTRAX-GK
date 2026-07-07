#!/usr/bin/env python3
"""Build a zonal-flow objective-row artifact from validated response summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.objectives.zonal import (  # noqa: E402
    ZonalFlowObjectiveConfig,
    zonal_flow_objective_artifact_from_records,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv"
DEFAULT_COMPARISON = ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv"
DEFAULT_OUT_JSON = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.json"
DEFAULT_OUT_CSV = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.csv"
DEFAULT_OUT_PNG = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.png"


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none", "null"}:
            return None
        value = stripped
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def _kx_key(value: Any) -> float:
    scalar = _optional_float(value)
    if scalar is None:
        raise ValueError(f"missing finite kx value: {value!r}")
    return round(float(scalar), 10)


def _comparison_by_kx(path: Path | None) -> dict[float, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    table = _read_csv(path)
    rows: dict[float, dict[str, str]] = {}
    for row in table:
        if "kx" in row:
            rows[_kx_key(row["kx"])] = row
        elif "kx_target" in row:
            rows[_kx_key(row["kx_target"])] = row
    return rows


def _tail_std_ratio(row: dict[str, str] | None) -> float | None:
    if row is None:
        return None
    direct = _optional_float(row.get("tail_std_ratio"))
    if direct is not None:
        return direct
    tail_std = _optional_float(row.get("tail_std"))
    reference_tail_std = _optional_float(row.get("reference_tail_std"))
    if tail_std is None or reference_tail_std is None or reference_tail_std <= 0.0:
        return None
    return tail_std / reference_tail_std


def _recurrence_value(
    *,
    summary_row: dict[str, str],
    comparison_row: dict[str, str] | None,
    source: str,
) -> float | None:
    if source == "residual_std":
        return _optional_float(summary_row.get("residual_std"))
    if source == "tail_std":
        return (
            None
            if comparison_row is None
            else _optional_float(comparison_row.get("tail_std"))
        )
    if source == "tail_std_ratio":
        return _tail_std_ratio(comparison_row)
    if source != "auto":
        raise ValueError(f"unknown recurrence source: {source}")
    ratio = _tail_std_ratio(comparison_row)
    if ratio is not None:
        return ratio
    return _optional_float(summary_row.get("residual_std"))


def records_from_w7x_summary(
    summary_csv: Path,
    *,
    comparison_csv: Path | None = None,
    recurrence_source: str = "auto",
) -> list[dict[str, object]]:
    """Return normalized zonal-objective records from the W7-X summary CSV."""

    summary = _read_csv(summary_csv)
    comparison = _comparison_by_kx(comparison_csv)
    records: list[dict[str, object]] = []
    for row in summary:
        kx = _kx_key(row.get("kx_target", row.get("kx")))
        comparison_row = comparison.get(kx)
        recurrence = _recurrence_value(
            summary_row=row,
            comparison_row=comparison_row,
            source=recurrence_source,
        )
        records.append(
            {
                "surface": _optional_float(row.get("surface")) or 0.0,
                "alpha": _optional_float(row.get("alpha")) or 0.0,
                "kx": float(kx),
                "residual_level": row.get("residual_level"),
                "damping_rate": row.get("gam_damping_rate", row.get("damping_rate")),
                "linear_growth_rate": row.get("linear_growth_rate", 0.0),
                "recurrence_amplitude": recurrence,
            }
        )
    return records


def _write_row_csv(path: Path, payload: dict[str, object]) -> None:
    rows = list(payload["row_table"])
    if not rows:
        raise ValueError("cannot write an empty zonal-flow objective table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "surface",
        "alpha",
        "kx",
        "residual_level",
        "damping_rate",
        "linear_growth_rate",
        "recurrence_amplitude",
        "inverse_residual",
        "growth_over_residual",
        "sample_objective",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _plot_payload(path: Path, payload: dict[str, object]) -> None:
    set_plot_style()
    table = list(payload["row_table"])
    kx = np.asarray([float(row["kx"]) for row in table], dtype=float)
    order = np.argsort(kx)
    labels = [f"{kx[index]:.2f}" for index in order]
    x = np.arange(order.size)
    metrics = {
        "Residual response\n(higher is better)": [
            float(table[index]["residual_level"]) for index in order
        ],
        "Damping penalty\n(lower is better)": [
            float(table[index]["damping_rate"]) for index in order
        ],
        "Recurrence/tail penalty\n(lower is better)": [
            float(table[index]["recurrence_amplitude"]) for index in order
        ],
        "Weighted sample objective\n(lower is better)": [
            float(table[index]["sample_objective"]) for index in order
        ],
    }
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, (title, values), color in zip(
        axes.ravel(), metrics.items(), colors, strict=True
    ):
        ax.bar(x, values, color=color, alpha=0.86, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_xlabel(r"$k_x \rho_i$")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Zonal-flow objective row gate", fontsize=15)
    status = "promotion-ready" if payload["promotion_ready"] else "diagnostic only"
    fig.text(
        0.5,
        0.01,
        (
            f"Status: {status}; missing damping rows: {payload['missing_damping_count']}; "
            f"claim: {payload['claim_level']}"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--recurrence-source",
        choices=("auto", "residual_std", "tail_std", "tail_std_ratio"),
        default="auto",
        help="Late-envelope recurrence metric used for the fourth objective column.",
    )
    parser.add_argument(
        "--missing-damping-policy",
        choices=("zero", "fail"),
        default="zero",
        help=(
            "Use 'fail' for promoted physics gates. The default 'zero' writes a "
            "diagnostic W7-X row artifact while preserving promotion_ready=false."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--damping-weight", type=float, default=1.0)
    parser.add_argument("--growth-over-residual-weight", type=float, default=0.0)
    parser.add_argument("--recurrence-weight", type=float, default=0.25)
    parser.add_argument("--residual-floor", type=float, default=1.0e-6)
    parser.add_argument(
        "--claim-level",
        default="diagnostic_zonal_objective_row_producer_not_promoted_w7x_optimization_claim",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    records = records_from_w7x_summary(
        args.summary_csv,
        comparison_csv=args.comparison_csv,
        recurrence_source=args.recurrence_source,
    )
    config = ZonalFlowObjectiveConfig(
        residual_weight=args.residual_weight,
        damping_weight=args.damping_weight,
        growth_over_residual_weight=args.growth_over_residual_weight,
        recurrence_weight=args.recurrence_weight,
        residual_floor=args.residual_floor,
    )
    payload = zonal_flow_objective_artifact_from_records(
        records,
        config=config,
        missing_damping_policy=args.missing_damping_policy,
        claim_level=args.claim_level,
        source_paths=[
            _repo_relative(args.summary_csv),
            _repo_relative(args.comparison_csv),
        ],
    )
    payload["input_summary_csv"] = _repo_relative(args.summary_csv)
    payload["input_comparison_csv"] = _repo_relative(args.comparison_csv)
    payload["recurrence_source"] = args.recurrence_source
    payload["validation_status"] = (
        "closed" if payload["promotion_ready"] else "diagnostic"
    )
    payload["gate_index_include"] = False
    payload["notes"] = [
        "This artifact verifies the row-production contract for zonal-flow optimization objectives.",
        "W7-X rows with missing GAM damping remain diagnostic and are not promoted to an optimization claim.",
        "Use --missing-damping-policy=fail for closed QA/QH/Miller-style promotion gates.",
    ]
    json.dumps(payload, allow_nan=False)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_row_csv(args.out_csv, payload)
    _plot_payload(args.out_png, payload)
    print(
        "wrote zonal-flow objective gate "
        f"samples={payload['sample_count']} promotion_ready={payload['promotion_ready']} "
        f"json={_repo_relative(args.out_json)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
