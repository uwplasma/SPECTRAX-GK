#!/usr/bin/env python3
"""Build a time-horizon stability gate from external-VMEC grid gates."""

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

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.diagnostics.validation_gates import (
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
)  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "_static" / "external_vmec_time_horizon_gate.png"
DEFAULT_MAX_RELATIVE_CHANGE = 0.15


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


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _max_symmetric_relative_difference(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    max_diff = 0.0
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            denom = max(abs(left) + abs(right), 1.0e-300)
            max_diff = max(max_diff, float(2.0 * abs(left - right) / denom))
    return max_diff


def _parse_entry(raw: str) -> tuple[float, Path]:
    if ":" not in raw:
        raise ValueError("entries must have format HORIZON:PATH")
    horizon_raw, path_raw = raw.split(":", maxsplit=1)
    horizon = float(horizon_raw)
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")
    path = Path(path_raw)
    if not path.is_absolute():
        path = ROOT / path
    return horizon, path


def _load_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "external_vmec_nonlinear_grid_convergence_gate":
        raise ValueError(f"{path} is not an external-VMEC nonlinear grid gate")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError(f"{path} must contain at least two run rows")
    return payload


def build_time_horizon_payload(
    entries: list[tuple[float, Path]],
    *,
    case: str = "External-VMEC high-grid time-horizon stability",
    max_relative_change: float = DEFAULT_MAX_RELATIVE_CHANGE,
) -> dict[str, Any]:
    """Return a JSON-ready time-horizon stability report."""

    if len(entries) < 2:
        raise ValueError("at least two horizon entries are required")
    if max_relative_change <= 0.0:
        raise ValueError("max_relative_change must be positive")

    rows = []
    for horizon, path in sorted(entries, key=lambda item: item[0]):
        gate = _load_gate(path)
        common_means = [
            float(run["common_window"]["heat_flux_mean"]) for run in gate["runs"]
        ]
        least_means = [
            float(run["least_trending_window"]["heat_flux_mean"])
            for run in gate["runs"]
        ]
        rows.append(
            {
                "horizon": float(horizon),
                "gate_json": _repo_relative_path(path),
                "grid_gate_passed": bool(gate.get("passed")),
                "grid_common_relative_difference": float(
                    gate["common_window"][
                        "max_pairwise_heat_flux_symmetric_relative_difference"
                    ]
                ),
                "grid_least_relative_difference": float(
                    gate["least_windows"][
                        "max_pairwise_heat_flux_symmetric_relative_difference"
                    ]
                ),
                "common_heat_flux_mean": float(np.mean(common_means)),
                "common_heat_flux_half_spread": float(
                    0.5 * (max(common_means) - min(common_means))
                ),
                "least_heat_flux_mean": float(np.mean(least_means)),
                "least_heat_flux_half_spread": float(
                    0.5 * (max(least_means) - min(least_means))
                ),
                "grid_labels": [str(run["label"]) for run in gate["runs"]],
                "grid_common_means": common_means,
                "grid_least_means": least_means,
            }
        )

    common_change = _max_symmetric_relative_difference(
        [float(row["common_heat_flux_mean"]) for row in rows]
    )
    least_change = _max_symmetric_relative_difference(
        [float(row["least_heat_flux_mean"]) for row in rows]
    )
    failed_grid_gates = sum(0 if row["grid_gate_passed"] else 1 for row in rows)
    gates = [
        evaluate_scalar_gate(
            "failed_grid_gate_count",
            float(failed_grid_gates),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="Every input high-grid gate must pass before checking horizon stability.",
        ),
        evaluate_scalar_gate(
            "common_window_time_horizon_relative_change",
            common_change,
            0.0,
            atol=max_relative_change,
            rtol=0.0,
            notes="Pairwise symmetric relative change of high-grid averaged common-window means.",
        ),
        evaluate_scalar_gate(
            "least_window_time_horizon_relative_change",
            least_change,
            0.0,
            atol=max_relative_change,
            rtol=0.0,
            notes="Pairwise symmetric relative change of high-grid averaged least-trending means.",
        ),
    ]
    report = gate_report(case, "external_vmec_high_grid_horizon_gates", gates)
    passed = bool(report.passed)
    return _json_clean(
        {
            "kind": "external_vmec_time_horizon_gate",
            "case": case,
            "passed": passed,
            "gate_index_include": False,
            "claim_level": (
                "passed_high_grid_time_horizon_candidate_not_replicated_holdout"
                if passed
                else "negative_time_horizon_result_not_transport_validation"
            ),
            "thresholds": {"max_relative_change": float(max_relative_change)},
            "common_window_time_horizon_relative_change": common_change,
            "least_window_time_horizon_relative_change": least_change,
            "rows": rows,
            "gate_report": gate_report_to_dict(report),
            "promotion_gate": {
                "passed": False,
                "reason": (
                    "time-horizon stability is necessary but not sufficient; "
                    "replicate/seed/timestep evidence is still required before holdout admission"
                    if passed
                    else "time-horizon stability gate failed"
                ),
            },
        }
    )


def write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    """Write horizon rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "horizon",
        "grid_gate_passed",
        "grid_common_relative_difference",
        "grid_least_relative_difference",
        "common_heat_flux_mean",
        "common_heat_flux_half_spread",
        "least_heat_flux_mean",
        "least_heat_flux_half_spread",
        "gate_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row[key] for key in fields})


def write_time_horizon_panel(
    entries: list[tuple[float, Path]],
    *,
    out: str | Path = DEFAULT_OUT,
    case: str = "External-VMEC high-grid time-horizon stability",
    max_relative_change: float = DEFAULT_MAX_RELATIVE_CHANGE,
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for the time-horizon gate."""

    payload = build_time_horizon_payload(
        entries, case=case, max_relative_change=max_relative_change
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    horizons = np.asarray(
        [float(row["horizon"]) for row in payload["rows"]], dtype=float
    )
    common = np.asarray(
        [float(row["common_heat_flux_mean"]) for row in payload["rows"]], dtype=float
    )
    common_err = np.asarray(
        [float(row["common_heat_flux_half_spread"]) for row in payload["rows"]],
        dtype=float,
    )
    least = np.asarray(
        [float(row["least_heat_flux_mean"]) for row in payload["rows"]], dtype=float
    )
    least_err = np.asarray(
        [float(row["least_heat_flux_half_spread"]) for row in payload["rows"]],
        dtype=float,
    )

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    ax, ax_text = axes
    ax.errorbar(
        horizons,
        common,
        yerr=common_err,
        marker="o",
        linewidth=2.2,
        capsize=4,
        label="common window",
    )
    ax.errorbar(
        horizons,
        least,
        yerr=least_err,
        marker="s",
        linewidth=2.2,
        capsize=4,
        label="least-trending window",
    )
    ax.set_xlabel("final time")
    ax.set_ylabel("high-grid averaged heat flux")
    ax.set_title("High-grid horizon stability")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    failed = [
        item for item in payload["gate_report"]["gates"] if not bool(item["passed"])
    ]
    lines = [
        "Gate status: " + ("PASS" if payload["passed"] else "FAIL"),
        f"common horizon diff: {float(payload['common_window_time_horizon_relative_change']):.3f}",
        f"least horizon diff: {float(payload['least_window_time_horizon_relative_change']):.3f}",
        f"allowed diff: {float(payload['thresholds']['max_relative_change']):.3f}",
        "",
        "Failed metrics:",
    ]
    lines.extend(
        f"- {item['metric']}: {float(item['observed']):.3g} > {float(item['atol']):.3g}"
        for item in failed
    )
    if not failed:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Interpretation:",
            "necessary horizon check only;",
            "replicate evidence is still",
            "required before calibration use.",
        ]
    )
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )
    fig.suptitle(case, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    payload = dict(payload)
    payload.update(
        {
            "png": _repo_relative_path(out_path),
            "pdf": _repo_relative_path(pdf_path),
            "csv": _repo_relative_path(csv_path),
        }
    )
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_summary_csv(csv_path, payload)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Horizon/gate entry encoded as HORIZON:PATH",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path.")
    parser.add_argument(
        "--case", default="External-VMEC high-grid time-horizon stability"
    )
    parser.add_argument(
        "--max-relative-change", type=float, default=DEFAULT_MAX_RELATIVE_CHANGE
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_time_horizon_panel(
        [_parse_entry(raw) for raw in args.entry],
        out=args.out,
        case=args.case,
        max_relative_change=args.max_relative_change,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
