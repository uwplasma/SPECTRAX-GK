#!/usr/bin/env python3
"""Build the multi-equilibrium VMEC/Boozer equal-arc parity matrix.

The matrix is a lightweight publication artifact around the optional
``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` bridge.  It intentionally uses a
minimum Boozer spectral resolution of ``mboz=nboz=21`` because lower mode
counts under-resolve the QI drift gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from spectraxgk.geometry.differentiable import vmec_jax_flux_tube_array_parity_report  # type: ignore[import-untyped]  # noqa: E402
from spectraxgk.plotting import set_plot_style  # type: ignore[import-untyped]  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_parity_matrix.png"
MIN_BOOZER_MODE_COUNT = 21


class ParityCase(NamedTuple):
    """One VMEC example equilibrium used by the parity matrix."""

    case_name: str
    label: str
    family: str
    ntheta: int
    mboz: int = MIN_BOOZER_MODE_COUNT
    nboz: int = MIN_BOOZER_MODE_COUNT


DEFAULT_CASES: tuple[ParityCase, ...] = (
    ParityCase("nfp4_QH_warm_start", "QH warm start", "quasi-helical", 16),
    ParityCase("nfp3_QI_fixed_resolution_final", "QI fixed resolution", "quasi-isodynamic", 8),
    ParityCase("shaped_tokamak_pressure", "shaped tokamak", "axisymmetric finite-beta", 8),
)


Reporter = Callable[..., dict[str, object]]


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


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _validate_case(case: ParityCase) -> None:
    if int(case.mboz) < MIN_BOOZER_MODE_COUNT or int(case.nboz) < MIN_BOOZER_MODE_COUNT:
        raise ValueError(f"mboz and nboz must both be >= {MIN_BOOZER_MODE_COUNT}")
    if int(case.ntheta) < 4:
        raise ValueError("ntheta must be >= 4")


def _row_from_report(case: ParityCase, report: dict[str, object]) -> dict[str, object]:
    available = bool(report.get("available", False))
    mboz = int(cast(Any, report.get("mboz", case.mboz)))
    nboz = int(cast(Any, report.get("nboz", case.nboz)))
    row = {
        "case_name": case.case_name,
        "label": case.label,
        "family": case.family,
        "ntheta": int(case.ntheta),
        "mboz": mboz,
        "nboz": nboz,
        "mode_floor_passed": mboz >= MIN_BOOZER_MODE_COUNT and nboz >= MIN_BOOZER_MODE_COUNT,
        "available": available,
        "status": str(report.get("status", "unavailable")),
        "reason": report.get("reason"),
        "error": report.get("error"),
        "equal_arc_core_worst_normalized_max_abs": _finite_float(
            report.get("equal_arc_core_worst_normalized_max_abs")
        ),
        "equal_arc_core_worst_scalar_rel": _finite_float(report.get("equal_arc_core_worst_scalar_rel")),
        "equal_arc_derivative_worst_normalized_max_abs": _finite_float(
            report.get("equal_arc_derivative_worst_normalized_max_abs")
        ),
        "equal_arc_metric_worst_normalized_max_abs": _finite_float(
            report.get("equal_arc_metric_worst_normalized_max_abs")
        ),
        "equal_arc_drift_worst_normalized_max_abs": _finite_float(
            report.get("equal_arc_drift_worst_normalized_max_abs")
        ),
        "equal_arc_core_tolerance": _finite_float(report.get("equal_arc_core_tolerance")),
        "equal_arc_derivative_tolerance": _finite_float(report.get("equal_arc_derivative_tolerance")),
        "equal_arc_metric_tolerance": _finite_float(report.get("equal_arc_metric_tolerance")),
        "equal_arc_drift_tolerance": _finite_float(report.get("equal_arc_drift_tolerance")),
        "equal_arc_core_passed": bool(report.get("equal_arc_core_passed", False)),
        "equal_arc_bgrad_passed": bool(report.get("equal_arc_derivative_passed", False)),
        "equal_arc_metric_passed": bool(report.get("equal_arc_metric_passed", False)),
        "equal_arc_drift_passed": bool(report.get("equal_arc_drift_passed", False)),
        "production_parity_passed": bool(report.get("production_parity_passed", False)),
        "worst_core_normalized_max_abs": _finite_float(report.get("worst_core_normalized_max_abs")),
        "worst_scalar_rel": _finite_float(report.get("worst_scalar_rel")),
        "input_path": report.get("input_path"),
        "wout_path": report.get("wout_path"),
    }
    row["equal_arc_all_passed"] = bool(
        available
        and row["mode_floor_passed"]
        and row["equal_arc_core_passed"]
        and row["equal_arc_bgrad_passed"]
        and row["equal_arc_metric_passed"]
        and row["equal_arc_drift_passed"]
    )
    return row


def build_parity_matrix(
    cases: tuple[ParityCase, ...] = DEFAULT_CASES,
    *,
    reporter: Reporter = vmec_jax_flux_tube_array_parity_report,
) -> dict[str, object]:
    """Return a JSON-ready multi-equilibrium Boozer parity report."""

    rows: list[dict[str, object]] = []
    for case in cases:
        _validate_case(case)
        try:
            report = reporter(
                case_name=case.case_name,
                ntheta=int(case.ntheta),
                mboz=int(case.mboz),
                nboz=int(case.nboz),
            )
        except Exception as exc:  # pragma: no cover - optional-backend diagnostic detail
            report = {
                "available": False,
                "case_name": case.case_name,
                "mboz": int(case.mboz),
                "nboz": int(case.nboz),
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(_row_from_report(case, report))

    available_rows = [row for row in rows if bool(row["available"])]
    all_available = len(available_rows) == len(rows) and bool(rows)
    all_equal_arc_passed = all_available and all(bool(row["equal_arc_all_passed"]) for row in rows)
    return {
        "kind": "vmec_boozer_parity_matrix",
        "claim_level": "multi_equilibrium_zero_beta_equal_arc_parity_gate_not_full_transport_gradient_claim",
        "minimum_boozer_mode_count": MIN_BOOZER_MODE_COUNT,
        "cases": [case._asdict() for case in cases],
        "rows": rows,
        "summary": {
            "n_cases": len(rows),
            "n_available": len(available_rows),
            "n_equal_arc_passed": sum(1 for row in rows if bool(row["equal_arc_all_passed"])),
            "all_available": all_available,
            "all_equal_arc_passed": all_equal_arc_passed,
        },
        "notes": (
            "The parity matrix checks the JAX-native Boozer equal-arc core, bgrad, "
            "zero-beta metric, and loaded-convention drift profiles against the "
            "imported VMEC/EIK runtime path. It enforces mboz,nboz >= 21 so QI "
            "drift parity is not evaluated on an under-resolved Boozer spectrum."
        ),
    }


def _metric_arrays(rows: list[dict[str, object]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    metric_keys = [
        "equal_arc_core_worst_normalized_max_abs",
        "equal_arc_core_worst_scalar_rel",
        "equal_arc_derivative_worst_normalized_max_abs",
        "equal_arc_metric_worst_normalized_max_abs",
        "equal_arc_drift_worst_normalized_max_abs",
    ]
    tolerance_keys = [
        "equal_arc_core_tolerance",
        "equal_arc_core_tolerance",
        "equal_arc_derivative_tolerance",
        "equal_arc_metric_tolerance",
        "equal_arc_drift_tolerance",
    ]
    pass_keys = [
        "equal_arc_core_passed",
        "equal_arc_core_passed",
        "equal_arc_bgrad_passed",
        "equal_arc_metric_passed",
        "equal_arc_drift_passed",
    ]
    values = np.full((len(rows), len(metric_keys)), np.nan)
    tolerances = np.full_like(values, np.nan)
    passed = np.zeros_like(values, dtype=bool)
    for i, row in enumerate(rows):
        for j, (metric_key, tolerance_key, pass_key) in enumerate(zip(metric_keys, tolerance_keys, pass_keys, strict=True)):
            value = _finite_float(row.get(metric_key))
            tolerance = _finite_float(row.get(tolerance_key))
            if value is not None:
                values[i, j] = value
            if tolerance is not None:
                tolerances[i, j] = tolerance
            passed[i, j] = bool(row.get(pass_key, False))
    return ["core", "scalar", "bgrad", "metric", "drift"], values, tolerances, passed


def write_parity_matrix_artifacts(payload: dict[str, object], *, out: str | Path = DEFAULT_OUT) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for a parity-matrix payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    raw_rows = payload.get("rows", [])
    if not isinstance(raw_rows, list):
        raise ValueError("payload rows must be a list")
    rows = [cast(dict[str, object], row) for row in raw_rows]

    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [
        "case_name",
        "label",
        "family",
        "ntheta",
        "mboz",
        "nboz",
        "available",
        "equal_arc_all_passed",
        "equal_arc_core_worst_normalized_max_abs",
        "equal_arc_core_worst_scalar_rel",
        "equal_arc_derivative_worst_normalized_max_abs",
        "equal_arc_metric_worst_normalized_max_abs",
        "equal_arc_drift_worst_normalized_max_abs",
        "status",
        "reason",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    set_plot_style()
    labels = [str(row["label"]) for row in rows]
    cols, values, tolerances, passed = _metric_arrays(rows)
    ratios = values / tolerances
    ratios[~np.isfinite(ratios)] = np.nan
    masked = np.ma.masked_invalid(ratios)
    vmax = 2.0 if np.all(masked.mask) else max(2.0, float(np.nanmax(ratios)) * 1.15)

    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    im = ax.imshow(masked, cmap=cmap, norm=LogNorm(vmin=1.0e-3, vmax=vmax), aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("equal-arc parity subgate")
    ax.set_title("VMEC/Boozer equal-arc parity matrix at mboz=nboz=21")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isfinite(values[i, j]):
                text = "n/a"
                color = "#333333"
            else:
                mark = "pass" if passed[i, j] else "open"
                text = f"{values[i, j]:.1e}\n{mark}"
                color = "white" if ratios[i, j] > 0.08 else "#111111"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.0, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("mismatch / tolerance")
    raw_summary = payload.get("summary", {})
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    ax.text(
        1.01,
        1.02,
        (
            f"available: {summary.get('n_available')}/{summary.get('n_cases')}\n"
            f"passed: {summary.get('n_equal_arc_passed')}/{summary.get('n_cases')}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_parity_matrix()
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_parity_matrix_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
