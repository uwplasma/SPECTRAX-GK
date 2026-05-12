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
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402


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
    ParityCase("nfp3_QI_fixed_resolution_final", "QI fixed resolution", "quasi-isodynamic", 16),
    ParityCase("shaped_tokamak_pressure", "shaped tokamak", "axisymmetric finite-beta", 8),
)

DEFAULT_QI_VARIANTS: tuple[ParityCase, ...] = (
    ParityCase("nfp1_QI", "QI input nfp1", "quasi-isodynamic input variant", 8),
    ParityCase("nfp2_QI", "QI input nfp2", "quasi-isodynamic input variant", 8),
    ParityCase(
        "nfp3_QI_fixed_resolution_final",
        "QI fixed resolution, ntheta=8",
        "quasi-isodynamic evaluated reference",
        8,
    ),
    ParityCase(
        "nfp3_QI_fixed_resolution_final",
        "QI fixed resolution, ntheta=16",
        "quasi-isodynamic evaluated reference",
        16,
    ),
    ParityCase("nfp4_QI_finite_beta", "QI finite-beta input", "quasi-isodynamic input variant", 8),
)


Reporter = Callable[..., dict[str, object]]
ArtifactResolver = Callable[[str], tuple[str | None, str | None, str | None]]


def _discover_differentiable_geometry_backends() -> dict[str, object]:
    """Import optional backend discovery lazily so fake-reporter tests stay local."""

    try:
        mod = importlib.import_module("spectraxgk.geometry.differentiable")
        discover = getattr(mod, "discover_differentiable_geometry_backends")
        return cast(dict[str, object], discover())
    except Exception as exc:  # pragma: no cover - optional-backend diagnostic detail
        return {
            "vmec_jax_available": False,
            "booz_xform_jax_api_available": False,
            "import_error": f"{type(exc).__name__}: {exc}",
        }


def _default_reporter(**kwargs: object) -> dict[str, object]:
    """Call the real VMEC/Boozer reporter only when a live report is requested."""

    try:
        mod = importlib.import_module("spectraxgk.geometry.differentiable")
        reporter = getattr(mod, "vmec_jax_flux_tube_array_parity_report")
        return cast(dict[str, object], reporter(**kwargs))
    except Exception as exc:  # pragma: no cover - optional-backend diagnostic detail
        return {
            "available": False,
            "case_name": str(kwargs.get("case_name", "unknown")),
            "mboz": int(cast(Any, kwargs.get("mboz", MIN_BOOZER_MODE_COUNT))),
            "nboz": int(cast(Any, kwargs.get("nboz", MIN_BOOZER_MODE_COUNT))),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _set_plot_style() -> None:
    try:
        mod = importlib.import_module("spectraxgk.plotting")
        cast(Callable[[], None], getattr(mod, "set_plot_style"))()
    except Exception:  # pragma: no cover - plotting style is cosmetic
        plt.rcParams.update({"figure.dpi": 140})


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


def _vmec_example_artifacts(case_name: str) -> tuple[str | None, str | None, str | None]:
    """Return VMEC example input/wout artifacts without launching a solve."""

    info = _discover_differentiable_geometry_backends()
    if not bool(info.get("vmec_jax_available", False)):
        return None, None, "vmec_jax is not available for example artifact lookup"

    try:
        driver = importlib.import_module("vmec_jax.driver")
    except Exception as exc:  # pragma: no cover - optional-backend diagnostic detail
        return None, None, f"vmec_jax_example_paths_unavailable: {type(exc).__name__}: {exc}"

    try:
        input_path, wout_path = driver.example_paths(str(case_name))
    except Exception as exc:  # pragma: no cover - optional-backend diagnostic detail
        return None, None, f"vmec_jax_example_paths_failed: {type(exc).__name__}: {exc}"

    return (
        None if input_path is None else str(input_path),
        None if wout_path is None else str(wout_path),
        None,
    )


def _rejection_row(
    case: ParityCase,
    *,
    artifact_reason: str,
    rejection_reason: str,
    input_path: str | None = None,
    wout_path: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    mode_floor_passed = int(case.mboz) >= MIN_BOOZER_MODE_COUNT and int(case.nboz) >= MIN_BOOZER_MODE_COUNT
    return {
        "validation_scope": "qi_seed_robustness",
        "case_name": case.case_name,
        "label": case.label,
        "family": case.family,
        "ntheta": int(case.ntheta),
        "mboz": int(case.mboz),
        "nboz": int(case.nboz),
        "mode_floor_passed": mode_floor_passed,
        "available": False,
        "status": "artifact_rejected",
        "reason": rejection_reason,
        "error": error,
        "artifact_reason": artifact_reason,
        "rejection_reason": rejection_reason,
        "failure_reason": None,
        "qi_gate_status": "artifact_rejected",
        "qi_validation_evaluated": False,
        "qi_validation_passed": False,
        "qi_validation_rejected": True,
        "equal_arc_core_worst_normalized_max_abs": None,
        "equal_arc_core_worst_scalar_rel": None,
        "equal_arc_derivative_worst_normalized_max_abs": None,
        "equal_arc_metric_worst_normalized_max_abs": None,
        "equal_arc_drift_worst_normalized_max_abs": None,
        "equal_arc_core_tolerance": None,
        "equal_arc_derivative_tolerance": None,
        "equal_arc_metric_tolerance": None,
        "equal_arc_drift_tolerance": None,
        "equal_arc_core_passed": False,
        "equal_arc_bgrad_passed": False,
        "equal_arc_metric_passed": False,
        "equal_arc_drift_passed": False,
        "equal_arc_all_passed": False,
        "production_parity_passed": False,
        "worst_core_normalized_max_abs": None,
        "worst_scalar_rel": None,
        "input_path": input_path,
        "wout_path": wout_path,
    }


def _row_from_report(case: ParityCase, report: dict[str, object]) -> dict[str, object]:
    available = bool(report.get("available", False))
    mboz = int(cast(Any, report.get("mboz", case.mboz)))
    nboz = int(cast(Any, report.get("nboz", case.nboz)))
    row = {
        "validation_scope": "parity_matrix",
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


def _qi_row_from_evaluated_row(case: ParityCase, row: dict[str, object]) -> dict[str, object]:
    qi_row = dict(row)
    qi_row.update(
        {
            "validation_scope": "qi_seed_robustness",
            "family": case.family,
            "label": case.label,
            "qi_validation_evaluated": bool(row.get("available", False)),
            "qi_validation_passed": bool(row.get("equal_arc_all_passed", False)),
            "qi_validation_rejected": False,
            "rejection_reason": None,
        }
    )
    if bool(row.get("equal_arc_all_passed", False)):
        qi_row.update(
            {
                "qi_gate_status": "passed",
                "artifact_reason": "mode21_qi_tolerance_passed",
                "failure_reason": None,
            }
        )
    else:
        qi_row.update(
            {
                "qi_gate_status": "fragile_open",
                "artifact_reason": "mode21_qi_tolerance_exceeded",
                "failure_reason": (
                    "evaluated QI variant did not pass all mode-21 equal-arc "
                    "core/scalar/bgrad/metric/drift tolerances"
                ),
            }
        )
    return qi_row


def _qi_rejection_from_unavailable_row(case: ParityCase, row: dict[str, object]) -> dict[str, object]:
    detail = row.get("reason") or row.get("error") or "mode-21 QI report was unavailable"
    return _rejection_row(
        case,
        artifact_reason="mode21_qi_report_unavailable",
        rejection_reason=str(detail),
        input_path=cast(str | None, row.get("input_path")),
        wout_path=cast(str | None, row.get("wout_path")),
        error=cast(str | None, row.get("error")),
    )


def _can_reuse_row(case: ParityCase, row: dict[str, object] | None) -> bool:
    if row is None or not bool(row.get("available", False)):
        return False
    return (
        int(cast(Any, row.get("ntheta", -1))) == int(case.ntheta)
        and int(cast(Any, row.get("mboz", -1))) == int(case.mboz)
        and int(cast(Any, row.get("nboz", -1))) == int(case.nboz)
    )


def _count_by_artifact_reason(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("artifact_reason")
        if reason is None:
            continue
        key = str(reason)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_qi_seed_robustness(
    variants: tuple[ParityCase, ...] = DEFAULT_QI_VARIANTS,
    *,
    reporter: Reporter = _default_reporter,
    artifact_resolver: ArtifactResolver = _vmec_example_artifacts,
    evaluated_rows: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return QI input-variant accounting for the mode-21 parity gate."""

    rows: list[dict[str, object]] = []
    evaluated_rows = evaluated_rows or {}
    for case in variants:
        if int(case.ntheta) < 4:
            rows.append(
                _rejection_row(
                    case,
                    artifact_reason="invalid_theta_grid",
                    rejection_reason="ntheta must be >= 4 for the QI parity gate",
                )
            )
            continue
        if int(case.mboz) < MIN_BOOZER_MODE_COUNT or int(case.nboz) < MIN_BOOZER_MODE_COUNT:
            rows.append(
                _rejection_row(
                    case,
                    artifact_reason="underresolved_boozer_modes",
                    rejection_reason=(
                        f"mboz and nboz must both be >= {MIN_BOOZER_MODE_COUNT} "
                        "for the mode-21 QI parity gate"
                    ),
                )
            )
            continue

        reusable = evaluated_rows.get(case.case_name)
        if _can_reuse_row(case, reusable):
            rows.append(_qi_row_from_evaluated_row(case, cast(dict[str, object], reusable)))
            continue

        input_path, wout_path, lookup_error = artifact_resolver(case.case_name)
        if lookup_error is not None:
            rows.append(
                _rejection_row(
                    case,
                    artifact_reason="vmec_example_artifact_lookup_failed",
                    rejection_reason=str(lookup_error),
                    error=str(lookup_error),
                )
            )
            continue
        if input_path is None:
            rows.append(
                _rejection_row(
                    case,
                    artifact_reason="missing_vmec_input_artifact",
                    rejection_reason="QI variant has no vmec_jax example input artifact",
                    input_path=input_path,
                    wout_path=wout_path,
                )
            )
            continue
        if wout_path is None:
            rows.append(
                _rejection_row(
                    case,
                    artifact_reason="missing_bundled_wout_reference",
                    rejection_reason=(
                        "QI variant has an input artifact but no bundled wout reference; "
                        "the bounded parity matrix does not launch VMEC solves"
                    ),
                    input_path=input_path,
                    wout_path=wout_path,
                )
            )
            continue
        if not Path(wout_path).exists():
            rows.append(
                _rejection_row(
                    case,
                    artifact_reason="missing_wout_file",
                    rejection_reason="QI variant advertises a wout reference that is not present on disk",
                    input_path=input_path,
                    wout_path=wout_path,
                )
            )
            continue

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
                "input_path": input_path,
                "wout_path": wout_path,
                "error": f"{type(exc).__name__}: {exc}",
            }
        row = _row_from_report(case, report)
        row["input_path"] = row.get("input_path") or input_path
        row["wout_path"] = row.get("wout_path") or wout_path
        if bool(row.get("available", False)):
            rows.append(_qi_row_from_evaluated_row(case, row))
        else:
            rows.append(_qi_rejection_from_unavailable_row(case, row))

    evaluated = [row for row in rows if bool(row.get("qi_validation_evaluated", False))]
    passed = [row for row in evaluated if bool(row.get("qi_validation_passed", False))]
    failed = [row for row in evaluated if not bool(row.get("qi_validation_passed", False))]
    rejected = [row for row in rows if bool(row.get("qi_validation_rejected", False))]
    unaccounted = [
        row
        for row in rows
        if not (
            bool(row.get("qi_validation_evaluated", False))
            or bool(row.get("qi_validation_rejected", False))
        )
    ]
    full_declared_seed_campaign_passed = bool(rows) and len(passed) == len(rows)
    evaluated_gate_passed = bool(evaluated) and not failed and not unaccounted
    if full_declared_seed_campaign_passed:
        robustness_status = "passed"
    elif evaluated_gate_passed and rejected:
        robustness_status = "artifact_limited_passed"
    elif evaluated_gate_passed:
        robustness_status = "evaluated_passed"
    elif failed:
        robustness_status = "fragile_open"
    elif rejected or unaccounted:
        robustness_status = "artifact_limited"
    else:
        robustness_status = "open"
    return {
        "kind": "qi_seed_robustness",
        "minimum_boozer_mode_count": MIN_BOOZER_MODE_COUNT,
        "rows": rows,
        "summary": {
            "n_variants": len(rows),
            "n_evaluated": len(evaluated),
            "n_passed": len(passed),
            "n_failed": len(failed),
            "n_rejected": len(rejected),
            "n_unaccounted": len(unaccounted),
            "all_evaluated_variants_passed": evaluated_gate_passed,
            "all_variants_accounted_for": not unaccounted,
            "all_variants_evaluated": len(evaluated) == len(rows) and bool(rows),
            "seed_robust_gate_passed": evaluated_gate_passed,
            "full_declared_seed_campaign_passed": full_declared_seed_campaign_passed,
            "evaluated_reference_gate_passed": evaluated_gate_passed,
            "robustness_status": robustness_status,
            "artifact_reason_counts": _count_by_artifact_reason(rows),
        },
        "notes": (
            "QI seed/input variants with a bundled wout reference are evaluated at "
            "mboz=nboz=21. Input-only variants are rejected with artifact_reason "
            "rather than triggering a VMEC solve inside this bounded gate. The "
            "available-seed robustness gate passes when all evaluated variants pass "
            "the existing tolerances; the full declared seed campaign remains "
            "artifact-limited until every declared input seed has a bundled wout "
            "reference. Tolerances are not relaxed here."
        ),
    }


def build_parity_matrix(
    cases: tuple[ParityCase, ...] = DEFAULT_CASES,
    *,
    reporter: Reporter = _default_reporter,
    qi_variants: tuple[ParityCase, ...] = DEFAULT_QI_VARIANTS,
    artifact_resolver: ArtifactResolver = _vmec_example_artifacts,
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
    qi_seed_robustness = build_qi_seed_robustness(
        variants=qi_variants,
        reporter=reporter,
        artifact_resolver=artifact_resolver,
        evaluated_rows={str(row["case_name"]): row for row in rows},
    )
    return {
        "kind": "vmec_boozer_parity_matrix",
        "claim_level": "multi_equilibrium_zero_beta_equal_arc_parity_gate_not_full_transport_gradient_claim",
        "minimum_boozer_mode_count": MIN_BOOZER_MODE_COUNT,
        "cases": [case._asdict() for case in cases],
        "qi_seed_robustness_cases": [case._asdict() for case in qi_variants],
        "rows": rows,
        "qi_seed_robustness": qi_seed_robustness,
        "summary": {
            "n_cases": len(rows),
            "n_available": len(available_rows),
            "n_equal_arc_passed": sum(1 for row in rows if bool(row["equal_arc_all_passed"])),
            "all_available": all_available,
            "all_equal_arc_passed": all_equal_arc_passed,
            "qi_seed_robust_gate_passed": bool(
                cast(dict[str, object], qi_seed_robustness["summary"]).get("seed_robust_gate_passed", False)
            ),
        },
        "notes": (
            "The parity matrix checks the JAX-native Boozer equal-arc core, bgrad, "
            "zero-beta metric, and loaded-convention drift profiles against the "
            "imported VMEC/EIK runtime path. It enforces mboz,nboz >= 21 so QI "
            "drift parity is not evaluated on an under-resolved Boozer spectrum. "
            "The JSON and CSV also include a QI seed/input robustness lane where "
            "mode-21 variants either pass the same tolerance checks or carry an "
            "explicit artifact_reason explaining why they were rejected."
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
    raw_qi = payload.get("qi_seed_robustness", {})
    qi_payload = raw_qi if isinstance(raw_qi, dict) else {}
    raw_qi_rows = qi_payload.get("rows", [])
    qi_rows = [cast(dict[str, object], row) for row in raw_qi_rows] if isinstance(raw_qi_rows, list) else []
    csv_rows = rows + qi_rows

    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [
        "validation_scope",
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
        "qi_gate_status",
        "artifact_reason",
        "rejection_reason",
        "failure_reason",
        "reason",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    _set_plot_style()
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
    raw_qi_summary = qi_payload.get("summary", {})
    qi_summary = raw_qi_summary if isinstance(raw_qi_summary, dict) else {}
    qi_status = str(qi_summary.get("robustness_status", "open"))
    ax.text(
        1.01,
        1.02,
        (
            f"available: {summary.get('n_available')}/{summary.get('n_cases')}\n"
            f"passed: {summary.get('n_equal_arc_passed')}/{summary.get('n_cases')}\n"
            f"QI robust: {qi_status}"
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
