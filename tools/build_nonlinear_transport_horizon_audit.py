#!/usr/bin/env python3
"""Audit nonlinear artifacts for transport-window time coverage."""

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
from matplotlib.patches import Patch  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "nonlinear_transport_time_horizon_audit.png"
MIN_TRANSPORT_TIME = 50.0

STATUS_COLORS = {
    "release_transport_gate_passed": "#2a9d8f",
    "long_feasibility_pending_convergence": "#e9c46a",
    "long_but_failed_convergence": "#d1495b",
    "short_or_startup_not_transport_average": "#f4a261",
    "reduced_estimator_not_transport_average": "#7c3aed",
    "missing_or_unclassified": "#6b7280",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


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


def _netcdf_tmax(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        from netCDF4 import Dataset
    except ImportError:
        return None
    try:
        with Dataset(path, "r") as root:
            if "Grids" in root.groups and "time" in root.groups["Grids"].variables:
                time = np.asarray(root.groups["Grids"].variables["time"][:], dtype=float)
            elif "time" in root.variables:
                time = np.asarray(root.variables["time"][:], dtype=float)
            else:
                return None
    except OSError:
        return None
    time = time[np.isfinite(time)]
    return float(np.max(time)) if time.size else None


def _csv_tmax(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            names = set(reader.fieldnames or [])
            time_name = "t" if "t" in names else "time" if "time" in names else None
            if time_name is None:
                return None
            values = [_finite_float(row.get(time_name)) for row in reader]
    except OSError:
        return None
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _source_tmax(root: Path, *raw_paths: object) -> float | None:
    values: list[float] = []
    for raw in raw_paths:
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        suffixes = "".join(path.suffixes).lower()
        value = _netcdf_tmax(path) if suffixes.endswith(".nc") else _csv_tmax(path)
        if value is not None:
            values.append(value)
    return max(values) if values else None


def _max_float(*values: object) -> float | None:
    finite = [_finite_float(value) for value in values]
    finite_values = [value for value in finite if value is not None]
    return max(finite_values) if finite_values else None


def classify_record(record: dict[str, Any], *, min_transport_time: float = MIN_TRANSPORT_TIME) -> str:
    """Classify a nonlinear artifact without promoting short or reduced outputs."""

    claim = str(record.get("claim_level", "")).lower()
    kind = str(record.get("kind", "")).lower()
    artifact = str(record.get("artifact", "")).lower()
    notes = str(record.get("notes", "")).lower()
    tmax = _finite_float(record.get("effective_tmax"), _finite_float(record.get("tmax"), 0.0)) or 0.0
    threshold_tmax = _max_float(record.get("effective_tmax"), record.get("source_tmax")) or tmax
    gate_passed = bool(record.get("gate_passed", False))
    transport_gate = record.get("transport_average_gate", None)
    promotion_gate = record.get("promotion_gate_passed", None)
    convergence_passed = record.get("convergence_gate_passed", None)

    if "startup" in claim or transport_gate is False:
        return "short_or_startup_not_transport_average"
    reduced_text = " ".join([claim, kind, artifact, notes])
    if (
        "estimator" in reduced_text
        or "optimization" in reduced_text
        or "smooth_logistic_heat_flux_envelope" in reduced_text
        or "reduced nonlinear-window" in reduced_text
    ):
        return "reduced_estimator_not_transport_average"
    if gate_passed and (tmax >= min_transport_time or threshold_tmax >= min_transport_time):
        return "release_transport_gate_passed"
    if tmax >= min_transport_time and convergence_passed is False:
        return "long_but_failed_convergence"
    if tmax >= min_transport_time and promotion_gate is False:
        return "long_feasibility_pending_convergence"
    if tmax < min_transport_time:
        return "short_or_startup_not_transport_average"
    return "missing_or_unclassified"


def _release_gate_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in [
        "docs/_static/nonlinear_cyclone_gate_summary.json",
        "docs/_static/nonlinear_cyclone_miller_gate_summary.json",
        "docs/_static/nonlinear_kbm_gate_summary.json",
        "docs/_static/nonlinear_w7x_gate_summary.json",
        "docs/_static/nonlinear_hsx_gate_summary.json",
    ]:
        payload = _read_json(root / relative)
        if payload is None:
            continue
        source_tmax = _source_tmax(root, payload.get("spectrax"), payload.get("gx"))
        summary_tmax = _finite_float(payload.get("tmax"))
        effective_tmax = summary_tmax if summary_tmax is not None else source_tmax
        case = str(payload.get("case", Path(relative).stem))
        records.append(
            {
                "case": case,
                "artifact": relative,
                "claim_level": "matched_nonlinear_transport_comparison_gate",
                "kind": "release_nonlinear_gate",
                "gate_passed": bool(payload.get("gate_passed", False)),
                "summary_tmax": summary_tmax,
                "source_tmax": source_tmax,
                "effective_tmax": effective_tmax,
                "tmax": effective_tmax,
                "notes": "matched-code nonlinear transport comparison gate",
            }
        )
    return records


def _pilot_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in [
        "docs/_static/external_vmec_qh_nonlinear_t150_pilot.json",
        "docs/_static/external_vmec_qh_nonlinear_t150_n48_pilot.json",
        "docs/_static/external_vmec_qh_nonlinear_t150_n64_pilot.json",
        "docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.json",
        "docs/_static/external_vmec_cth_like_nonlinear_t150_n48_pilot.json",
    ]:
        payload = _read_json(root / relative)
        if payload is None:
            continue
        promotion = payload.get("promotion_gate", {})
        records.append(
            {
                "case": str(payload.get("label", Path(relative).stem)),
                "artifact": relative,
                "claim_level": str(payload.get("claim_level", "")),
                "kind": str(payload.get("kind", "nonlinear_feasibility_pilot")),
                "gate_passed": bool((promotion or {}).get("passed", False)),
                "promotion_gate_passed": bool((promotion or {}).get("passed", False)),
                "summary_tmax": _finite_float(payload.get("tmax")),
                "source_tmax": _source_tmax(root, payload.get("source")),
                "effective_tmax": _max_float(payload.get("tmax"), _source_tmax(root, payload.get("source"))),
                "tmax": _max_float(payload.get("tmax"), _source_tmax(root, payload.get("source"))),
                "notes": str((promotion or {}).get("reason", "feasibility pilot")),
            }
        )
    for relative in [
        "docs/_static/external_vmec_qh_grid_convergence_gate.json",
        "docs/_static/external_vmec_qh_high_grid_convergence_gate.json",
        "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
    ]:
        gate_payload = _read_json(root / relative)
        if gate_payload is None:
            continue
        gate_report = gate_payload.get("gate_report", {})
        runs = gate_payload.get("runs", [])
        run_tmax = [
            _finite_float(run.get("tmax"))
            for run in runs
            if isinstance(run, dict) and _finite_float(run.get("tmax")) is not None
        ]
        passed = bool(gate_report.get("passed", False))
        records.append(
            {
                "case": str(gate_payload.get("case", Path(relative).stem)),
                "artifact": relative,
                "claim_level": str(gate_payload.get("claim_level", "")),
                "kind": str(gate_payload.get("kind", "external_vmec_nonlinear_grid_convergence_gate")),
                "gate_passed": passed,
                "convergence_gate_passed": passed,
                "summary_tmax": max(run_tmax) if run_tmax else None,
                "source_tmax": None,
                "effective_tmax": max(run_tmax) if run_tmax else None,
                "tmax": max(run_tmax) if run_tmax else None,
                "notes": (
                    "long nonlinear pilots pass the grid/window convergence gate"
                    if passed
                    else "long nonlinear pilots exist, but grid/window convergence gate fails"
                ),
            }
        )
    return records


def _startup_and_reduced_records(root: Path) -> list[dict[str, Any]]:
    specs = [
        ("docs/_static/nonlinear_window_fd_audit.json", "Compact nonlinear FD startup audit"),
        ("docs/_static/vmec_boozer_nonlinear_window_fd_audit.json", "VMEC/Boozer nonlinear FD startup audit"),
        ("docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json", "QH reduced nonlinear-window estimator gate"),
        ("docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.json", "Li383 reduced nonlinear-window estimator gate"),
        ("docs/_static/stellarator_itg_nonlinear_optimization.json", "Reduced stellarator nonlinear objective example"),
    ]
    records: list[dict[str, Any]] = []
    for relative, label in specs:
        payload = _read_json(root / relative)
        if payload is None:
            continue
        tmax = _finite_float(payload.get("tmax"))
        if tmax is None:
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                tmax = _finite_float(metrics.get("max_tmax"))
            runs = payload.get("runs", [])
            if tmax is None and isinstance(runs, list):
                run_tmax = []
                for run in runs:
                    if not isinstance(run, dict):
                        continue
                    window = run.get("window", {})
                    if isinstance(window, dict):
                        run_tmax.append(_finite_float(window.get("t_max")))
                    time = run.get("time")
                    if isinstance(time, list):
                        run_tmax.extend(_finite_float(item) for item in time)
                finite_run_tmax = [value for value in run_tmax if value is not None]
                tmax = max(finite_run_tmax) if finite_run_tmax else None
            cfg = payload.get("nonlinear_window_config", {})
            if tmax is None and isinstance(cfg, dict):
                dt = _finite_float(cfg.get("dt"))
                steps = _finite_float(cfg.get("steps"))
                if dt is not None and steps is not None:
                    tmax = dt * steps
            trace = payload.get("nonlinear_trace", {})
            if tmax is None and isinstance(trace, dict):
                times = trace.get("times")
                if isinstance(times, list):
                    finite_times = [_finite_float(item) for item in times]
                    tmax_values = [value for value in finite_times if value is not None]
                    tmax = max(tmax_values) if tmax_values else None
        records.append(
            {
                "case": label,
                "artifact": relative,
                "claim_level": str(payload.get("claim_level", "")),
                "kind": str(payload.get("kind", "")),
                "gate_passed": bool(payload.get("passed", False)),
                "transport_average_gate": payload.get("transport_average_gate"),
                "summary_tmax": tmax,
                "source_tmax": None,
                "effective_tmax": tmax,
                "tmax": tmax,
                "notes": str(payload.get("next_action", "not a promoted nonlinear transport average")),
            }
        )
    return records


def build_payload(root: Path = ROOT, *, min_transport_time: float = MIN_TRANSPORT_TIME) -> dict[str, Any]:
    """Build the nonlinear transport time-horizon audit payload."""

    root = Path(root)
    records = _release_gate_records(root) + _pilot_records(root) + _startup_and_reduced_records(root)
    for record in records:
        record["status"] = classify_record(record, min_transport_time=min_transport_time)
        record["requires_longer_or_convergence_action"] = record["status"] in {
            "short_or_startup_not_transport_average",
            "long_feasibility_pending_convergence",
            "long_but_failed_convergence",
        }
    records.sort(key=lambda item: (str(item["status"]), str(item["case"])))
    counts: dict[str, int] = {}
    for record in records:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
    return {
        "kind": "nonlinear_transport_time_horizon_audit",
        "min_transport_time": float(min_transport_time),
        "summary": {
            "n_records": len(records),
            "status_counts": counts,
            "release_transport_gate_passed": counts.get("release_transport_gate_passed", 0),
            "short_or_reduced_not_transport": counts.get("short_or_startup_not_transport_average", 0)
            + counts.get("reduced_estimator_not_transport_average", 0),
        },
        "interpretation": (
            "Heat-flux values from startup or reduced-estimator artifacts must not be used as nonlinear "
            "transport averages. Long windows require post-transient averaging plus either matched-code "
            "comparison or grid/window convergence gates."
        ),
        "records": records,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "status",
        "effective_tmax",
        "summary_tmax",
        "source_tmax",
        "gate_passed",
        "artifact",
        "claim_level",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def write_artifacts(payload: dict[str, Any], *, out: Path = DEFAULT_OUT) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV nonlinear time-horizon audit artifacts."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = list(payload["records"])
    csv_path = out.with_suffix(".csv")
    json_path = out.with_suffix(".json")
    pdf_path = out.with_suffix(".pdf")
    _write_csv(csv_path, records)
    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    set_plot_style()
    fig_h = max(5.5, 0.42 * len(records) + 1.6)
    fig, ax = plt.subplots(figsize=(13.2, fig_h), constrained_layout=True)
    labels = [str(record["case"]) for record in records]
    tmax = np.asarray([
        max(float(record.get("effective_tmax") or 1.0e-3), 1.0e-3)
        for record in records
    ])
    colors = [STATUS_COLORS.get(str(record["status"]), STATUS_COLORS["missing_or_unclassified"]) for record in records]
    y = np.arange(len(records))
    ax.barh(y, tmax, color=colors, alpha=0.88)
    ax.axvline(float(payload["min_transport_time"]), color="#111827", linestyle="--", linewidth=1.7)
    ax.set_xscale("log")
    ax.set_xlabel("maximum simulated time in artifact")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Nonlinear heat-flux artifact time-horizon audit")
    ax.grid(True, axis="x", which="both", alpha=0.22)
    statuses = list(dict.fromkeys(str(record["status"]) for record in records))
    handles = [
        Patch(facecolor=STATUS_COLORS.get(status, STATUS_COLORS["missing_or_unclassified"]), label=status.replace("_", " "))
        for status in statuses
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=9)
    ax.text(
        float(payload["min_transport_time"]) * 1.05,
        -0.75,
        "minimum long-window threshold",
        fontsize=9,
        ha="left",
        va="center",
    )
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out), "pdf": str(pdf_path), "json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path.")
    parser.add_argument("--min-transport-time", type=float, default=MIN_TRANSPORT_TIME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(ROOT, min_transport_time=float(args.min_transport_time))
    paths = write_artifacts(payload, out=Path(args.out))
    for key in ("png", "pdf", "json", "csv"):
        print(f"saved {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
