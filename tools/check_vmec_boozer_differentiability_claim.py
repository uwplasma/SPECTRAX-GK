#!/usr/bin/env python3
"""Check the scoped VMEC/Boozer differentiability release claim.

This checker validates already-tracked evidence artifacts. It does not rerun
VMEC, Boozer transforms, or transport solves. Its job is claim hygiene: the
release may claim differentiable reduced solver objectives through the
``vmec_jax -> booz_xform_jax -> FluxTubeGeometryData`` path only when the
equal-arc parity and AD/finite-difference gradient gates pass, while direct
VMEC tensor-vs-imported-EIK parity and production nonlinear transport gradients
remain explicitly scoped as open.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "_static" / "vmec_boozer_differentiability_claim_guard.json"

DEFAULT_PARITY_MATRIX = "docs/_static/vmec_boozer_parity_matrix.json"
DEFAULT_GRADIENT_MATRIX = "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
DEFAULT_BRIDGE = "docs/_static/differentiable_geometry_bridge.json"
DEFAULT_NONLINEAR_FD_AUDIT = "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json"

REQUIRED_GRADIENT_GATE_TYPES = {
    "frequency",
    "quasilinear",
    "nonlinear-window estimator",
}
REQUIRED_SOURCE_SCOPE = "mode21_vmec_boozer_state"
MINIMUM_BOOZER_MODE_COUNT = 21
MINIMUM_GRADIENT_CASES = 2


def _read_json(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON artifact: {rel_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {rel_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{rel_path}: expected a JSON object")
    return payload


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _path_rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _parity_matrix_checks(parity: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    summary = _as_dict(parity.get("summary"))
    rows = _as_rows(parity.get("rows"))
    minimum_mode_count = parity.get("minimum_boozer_mode_count")
    mode_floor_passed = isinstance(minimum_mode_count, (int, float)) and minimum_mode_count >= MINIMUM_BOOZER_MODE_COUNT
    row_mode_failures = [
        row.get("case_name", "<unknown>")
        for row in rows
        if min(float(row.get("mboz", 0.0)), float(row.get("nboz", 0.0))) < MINIMUM_BOOZER_MODE_COUNT
        or row.get("mode_floor_passed") is False
    ]
    direct_open_rows = [
        row.get("case_name", "<unknown>")
        for row in rows
        if row.get("production_parity_passed") is False and row.get("status") == "diagnostic_open"
    ]
    hidden_direct_failures = [
        row.get("case_name", "<unknown>")
        for row in rows
        if row.get("production_parity_passed") is False and row.get("status") != "diagnostic_open"
    ]
    claim_level = str(parity.get("claim_level", ""))
    claim_scoped = "not_full_transport_gradient_claim" in claim_level
    checks = {
        "all_available": _bool(summary.get("all_available")),
        "all_equal_arc_passed": _bool(summary.get("all_equal_arc_passed")),
        "minimum_boozer_mode_count": minimum_mode_count,
        "mode_floor_passed": mode_floor_passed and not row_mode_failures,
        "n_cases": summary.get("n_cases", len(rows)),
        "n_equal_arc_passed": summary.get("n_equal_arc_passed"),
        "direct_tensor_parity_open_rows": sorted(str(item) for item in direct_open_rows),
        "direct_tensor_parity_hidden_failures": sorted(str(item) for item in hidden_direct_failures),
        "claim_level": claim_level,
        "claim_scoped_not_full_transport_gradient": claim_scoped,
    }
    blockers: list[str] = []
    if not checks["all_available"]:
        blockers.append("parity_matrix_not_all_available")
    if not checks["all_equal_arc_passed"]:
        blockers.append("equal_arc_parity_matrix_failed")
    if not checks["mode_floor_passed"]:
        blockers.append("boozer_mode_floor_failed")
    if hidden_direct_failures:
        blockers.append("direct_tensor_parity_failure_not_marked_diagnostic_open")
    if not claim_scoped:
        blockers.append("parity_matrix_claim_not_scoped")
    return checks, blockers


def _gradient_matrix_checks(gradient: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    summary = _as_dict(gradient.get("summary"))
    rows = _as_rows(gradient.get("rows"))
    cases = {str(row.get("case_name")) for row in rows if row.get("case_name")}
    gate_types = {str(row.get("gate_type")) for row in rows if row.get("gate_type")}
    source_scope_failures = [
        row.get("path") or row.get("case_name", "<unknown>")
        for row in rows
        if row.get("source_scope") != REQUIRED_SOURCE_SCOPE
    ]
    mode_failures = [
        row.get("path") or row.get("case_name", "<unknown>")
        for row in rows
        if min(float(row.get("mboz", 0.0)), float(row.get("nboz", 0.0))) < MINIMUM_BOOZER_MODE_COUNT
    ]
    failed_rows = [
        row.get("path") or row.get("case_name", "<unknown>")
        for row in rows
        if row.get("passed") is not True
    ]
    claim_level = str(gradient.get("claim_level", ""))
    claim_scoped = "not_production_nonlinear_optimization" in claim_level
    checks = {
        "passed": _bool(gradient.get("passed")),
        "all_gates_passed": _bool(summary.get("all_gates_passed")),
        "all_mode21_source_scope": _bool(summary.get("all_mode21_source_scope")),
        "all_mboz_nboz_at_least_21": _bool(summary.get("all_mboz_nboz_at_least_21")),
        "n_cases": len(cases),
        "cases": sorted(cases),
        "gate_types": sorted(gate_types),
        "required_gate_types": sorted(REQUIRED_GRADIENT_GATE_TYPES),
        "missing_gate_types": sorted(REQUIRED_GRADIENT_GATE_TYPES - gate_types),
        "source_scope_failures": sorted(str(item) for item in source_scope_failures),
        "mode_failures": sorted(str(item) for item in mode_failures),
        "failed_rows": sorted(str(item) for item in failed_rows),
        "claim_level": claim_level,
        "claim_scoped_not_production_nonlinear_optimization": claim_scoped,
    }
    blockers: list[str] = []
    if not checks["passed"] or not checks["all_gates_passed"] or failed_rows:
        blockers.append("gradient_holdout_matrix_failed")
    if checks["n_cases"] < MINIMUM_GRADIENT_CASES:
        blockers.append("gradient_holdout_case_count_below_two")
    if checks["missing_gate_types"]:
        blockers.append("gradient_holdout_missing_required_gate_types")
    if source_scope_failures or not checks["all_mode21_source_scope"]:
        blockers.append("gradient_holdout_wrong_source_scope")
    if mode_failures or not checks["all_mboz_nboz_at_least_21"]:
        blockers.append("gradient_holdout_mode_floor_failed")
    if not claim_scoped:
        blockers.append("gradient_holdout_claim_not_scoped")
    return checks, blockers


def _bridge_checks(bridge: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    direct = _as_dict(bridge.get("vmec_jax_flux_tube_array_parity"))
    status = str(direct.get("status", ""))
    production_parity_passed = direct.get("production_parity_passed")
    interpretation = str(direct.get("interpretation", ""))
    direct_gap_scoped = (
        production_parity_passed is True
        or (production_parity_passed is False and status == "diagnostic_open" and bool(interpretation))
    )
    equal_arc_keys = (
        "equal_arc_core_passed",
        "equal_arc_derivative_passed",
        "equal_arc_metric_passed",
        "equal_arc_drift_passed",
    )
    equal_arc_checks = {key: direct.get(key) for key in equal_arc_keys if key in direct}
    checks = {
        "direct_status": status,
        "direct_production_parity_passed": production_parity_passed,
        "direct_interpretation_present": bool(interpretation),
        "direct_tensor_gap_explicitly_scoped": direct_gap_scoped,
        "equal_arc_checks": equal_arc_checks,
        "vmec_jax_boozer_available": _bool(_as_dict(bridge.get("vmec_jax_boozer_flux_tube")).get("available")),
        "booz_xform_flux_tube_available": _bool(_as_dict(bridge.get("booz_xform_flux_tube")).get("available")),
    }
    blockers: list[str] = []
    if not direct_gap_scoped:
        blockers.append("direct_tensor_parity_gap_not_explicitly_scoped")
    if equal_arc_checks and not all(value is True for value in equal_arc_checks.values()):
        blockers.append("bridge_equal_arc_subcheck_failed")
    if not checks["vmec_jax_boozer_available"]:
        blockers.append("vmec_jax_boozer_bridge_unavailable")
    if not checks["booz_xform_flux_tube_available"]:
        blockers.append("booz_xform_flux_tube_bridge_unavailable")
    return checks, blockers


def _nonlinear_fd_checks(audit: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    claim_level = str(audit.get("claim_level", ""))
    production_gate = _bool(audit.get("production_nonlinear_window_gradient_gate"))
    transport_average_gate = _bool(audit.get("transport_average_gate"))
    production_path_gate = _bool(audit.get("vmec_boozer_production_nonlinear_observable_fd_path_gate"))
    startup_gate = _bool(audit.get("vmec_boozer_startup_nonlinear_plumbing_fd_path_gate"))
    scoped_startup = "not_transport_average" in claim_level
    checks = {
        "passed": _bool(audit.get("passed")),
        "claim_level": claim_level,
        "startup_plumbing_gate": startup_gate,
        "transport_average_gate": transport_average_gate,
        "production_nonlinear_window_gradient_gate": production_gate,
        "production_nonlinear_observable_fd_path_gate": production_path_gate,
        "claim_scoped_not_transport_average": scoped_startup,
    }
    blockers: list[str] = []
    if not checks["passed"]:
        blockers.append("nonlinear_fd_startup_audit_failed")
    if not startup_gate:
        blockers.append("nonlinear_fd_startup_plumbing_gate_failed")
    if production_gate or transport_average_gate or production_path_gate:
        blockers.append("startup_fd_audit_attempts_production_nonlinear_claim")
    if not scoped_startup:
        blockers.append("nonlinear_fd_claim_not_scoped_to_startup")
    return checks, blockers


def build_vmec_boozer_differentiability_claim_guard(
    root: Path = REPO_ROOT,
    *,
    parity_matrix_path: str = DEFAULT_PARITY_MATRIX,
    gradient_matrix_path: str = DEFAULT_GRADIENT_MATRIX,
    bridge_path: str = DEFAULT_BRIDGE,
    nonlinear_fd_audit_path: str = DEFAULT_NONLINEAR_FD_AUDIT,
) -> dict[str, Any]:
    """Return a machine-checkable VMEC/Boozer differentiability claim guard."""

    root = root.resolve()
    artifacts = {
        "parity_matrix": parity_matrix_path,
        "gradient_holdout_matrix": gradient_matrix_path,
        "differentiable_geometry_bridge": bridge_path,
        "nonlinear_fd_audit": nonlinear_fd_audit_path,
    }
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    try:
        parity = _read_json(root, parity_matrix_path)
        checks["parity_matrix"], parity_blockers = _parity_matrix_checks(parity)
        blockers.extend(parity_blockers)
    except ValueError as exc:
        checks["parity_matrix"] = {"error": str(exc)}
        blockers.append("parity_matrix_unreadable")

    try:
        gradient = _read_json(root, gradient_matrix_path)
        checks["gradient_holdout_matrix"], gradient_blockers = _gradient_matrix_checks(gradient)
        blockers.extend(gradient_blockers)
    except ValueError as exc:
        checks["gradient_holdout_matrix"] = {"error": str(exc)}
        blockers.append("gradient_holdout_matrix_unreadable")

    try:
        bridge = _read_json(root, bridge_path)
        checks["differentiable_geometry_bridge"], bridge_blockers = _bridge_checks(bridge)
        blockers.extend(bridge_blockers)
    except ValueError as exc:
        checks["differentiable_geometry_bridge"] = {"error": str(exc)}
        blockers.append("differentiable_geometry_bridge_unreadable")

    try:
        nonlinear_fd = _read_json(root, nonlinear_fd_audit_path)
        checks["nonlinear_fd_audit"], nonlinear_fd_blockers = _nonlinear_fd_checks(nonlinear_fd)
        blockers.extend(nonlinear_fd_blockers)
    except ValueError as exc:
        checks["nonlinear_fd_audit"] = {"error": str(exc)}
        blockers.append("nonlinear_fd_audit_unreadable")

    unique_blockers = sorted(set(blockers))
    return {
        "kind": "vmec_boozer_differentiability_claim_guard",
        "claim_level": (
            "reduced_full_chain_differentiability_guard_not_full_nonlinear_transport_optimization"
        ),
        "passed": not unique_blockers,
        "blockers": unique_blockers,
        "minimum_boozer_mode_count": MINIMUM_BOOZER_MODE_COUNT,
        "minimum_gradient_cases": MINIMUM_GRADIENT_CASES,
        "artifacts": artifacts,
        "checks": checks,
        "scope": (
            "Passes only the release-level reduced differentiability claim: "
            "VMEC/Boozer equal-arc parity and AD/finite-difference gradients for "
            "linear, quasilinear, and nonlinear-window estimator objectives. "
            "Direct VMEC tensor-vs-imported-EIK parity and converged nonlinear "
            "transport-gradient optimization remain explicitly out of scope."
        ),
        "root": str(root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parity-matrix", default=DEFAULT_PARITY_MATRIX)
    parser.add_argument("--gradient-matrix", default=DEFAULT_GRADIENT_MATRIX)
    parser.add_argument("--bridge", default=DEFAULT_BRIDGE)
    parser.add_argument("--nonlinear-fd-audit", default=DEFAULT_NONLINEAR_FD_AUDIT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    out_json = args.out_json
    if out_json is not None and not out_json.is_absolute():
        out_json = root / out_json
    report = build_vmec_boozer_differentiability_claim_guard(
        root=root,
        parity_matrix_path=args.parity_matrix,
        gradient_matrix_path=args.gradient_matrix,
        bridge_path=args.bridge,
        nonlinear_fd_audit_path=args.nonlinear_fd_audit,
    )
    if out_json is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {_path_rel(root, out_json)}")
    if not report["passed"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
