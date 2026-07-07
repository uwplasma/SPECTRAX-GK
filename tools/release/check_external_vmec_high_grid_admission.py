#!/usr/bin/env python3
"""Fail-closed high-grid admission gate for external-VMEC nonlinear holdouts.

This checker is for the narrow case where a full grid ladder fails only because
the lowest grid is demonstrably unconverged, while the highest grids, time
horizons, and seed/timestep replicated transport windows are stable.  It does
not replace the normal grid-convergence gate; it records a scoped exception
with enough metadata to prevent accidental promotion to a broader absolute-flux
claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.gates import (
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
)  # noqa: E402


DEFAULT_OUT = ROOT / "docs" / "_static" / "external_vmec_high_grid_admission_gate.json"
DEFAULT_ALLOWED_FULL_GRID_FAILURES = {
    "common_window_pairwise_heat_flux_symmetric_relative_difference",
    "least_window_pairwise_heat_flux_symmetric_relative_difference",
}
DEFAULT_MAX_PAIRWISE_RELATIVE_DIFFERENCE = 0.15
DEFAULT_MAX_TIME_HORIZON_RELATIVE_CHANGE = 0.15
DEFAULT_MAX_MEAN_REL_SPREAD = 0.15
DEFAULT_MAX_COMBINED_SEM_REL = 0.25
DEFAULT_MIN_REPLICATES = 3
DEFAULT_VALUE_FLOOR = 1.0e-12


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_json(path: Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise ValueError(
            f"{path} has kind {payload.get('kind')!r}; expected {expected_kind!r}"
        )
    return payload


def _failed_metrics(payload: dict[str, Any]) -> set[str]:
    report = payload.get("gate_report", {})
    gates = report.get("gates", []) if isinstance(report, dict) else []
    return {
        str(gate.get("metric", "unknown"))
        for gate in gates
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    }


def _grid_labels(payload: dict[str, Any]) -> list[str]:
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return []
    return [str(run.get("label", "")) for run in runs if isinstance(run, dict)]


def _threshold(payload: dict[str, Any], key: str, default: float) -> float:
    thresholds = payload.get("thresholds", {})
    if isinstance(thresholds, dict):
        try:
            return float(thresholds.get(key, default))
        except (TypeError, ValueError):
            return float(default)
    return float(default)


def _metric(
    payload: dict[str, Any], path: tuple[str, ...], default: float = float("inf")
) -> float:
    item: Any = payload
    for key in path:
        if not isinstance(item, dict) or key not in item:
            return float(default)
        item = item[key]
    try:
        out = float(item)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def build_high_grid_admission_payload(
    *,
    full_grid_gate_path: Path,
    high_grid_gate_paths: list[Path],
    time_horizon_gate_path: Path,
    replicate_ensemble_path: Path,
    excluded_grid_labels: list[str],
    retained_grid_labels: list[str],
    case: str = "External-VMEC high-grid admission",
    allowed_full_grid_failures: set[str] | None = None,
    min_replicates: int = DEFAULT_MIN_REPLICATES,
    value_floor: float = DEFAULT_VALUE_FLOOR,
) -> dict[str, Any]:
    """Return a JSON-ready scoped high-grid admission report."""

    if not high_grid_gate_paths:
        raise ValueError("at least one high-grid gate is required")
    if not excluded_grid_labels:
        raise ValueError("at least one excluded coarse-grid label is required")
    if len(retained_grid_labels) < 2:
        raise ValueError("at least two retained high-grid labels are required")
    if min_replicates < 2:
        raise ValueError("min_replicates must be at least 2")
    if value_floor < 0.0:
        raise ValueError("value_floor must be non-negative")

    allowed_failures = set(
        allowed_full_grid_failures or DEFAULT_ALLOWED_FULL_GRID_FAILURES
    )
    full_grid = _load_json(
        full_grid_gate_path,
        expected_kind="external_vmec_nonlinear_grid_convergence_gate",
    )
    high_grid_gates = [
        _load_json(path, expected_kind="external_vmec_nonlinear_grid_convergence_gate")
        for path in high_grid_gate_paths
    ]
    time_horizon = _load_json(
        time_horizon_gate_path, expected_kind="external_vmec_time_horizon_gate"
    )
    replicate = _load_json(replicate_ensemble_path)
    if replicate.get("kind") not in {
        "nonlinear_window_ensemble_report",
        "nonlinear_window_ensemble_gate",
    }:
        raise ValueError(
            f"{replicate_ensemble_path} is not a nonlinear-window ensemble gate"
        )

    full_labels = set(_grid_labels(full_grid))
    missing_excluded = sorted(set(excluded_grid_labels) - full_labels)
    missing_retained = sorted(set(retained_grid_labels) - full_labels)
    unexpected_full_failures = sorted(_failed_metrics(full_grid) - allowed_failures)
    high_grid_failed_count = sum(
        0 if bool(gate.get("passed", False)) else 1 for gate in high_grid_gates
    )
    high_grid_label_mismatch_count = sum(
        0 if set(_grid_labels(gate)) == set(retained_grid_labels) else 1
        for gate in high_grid_gates
    )

    max_high_grid_common_diff = max(
        _metric(
            gate,
            ("common_window", "max_pairwise_heat_flux_symmetric_relative_difference"),
        )
        for gate in high_grid_gates
    )
    max_high_grid_least_diff = max(
        _metric(
            gate,
            ("least_windows", "max_pairwise_heat_flux_symmetric_relative_difference"),
        )
        for gate in high_grid_gates
    )
    max_pairwise_threshold = min(
        _threshold(
            gate,
            "max_pairwise_relative_difference",
            DEFAULT_MAX_PAIRWISE_RELATIVE_DIFFERENCE,
        )
        for gate in high_grid_gates
    )

    horizon_common_change = _metric(
        time_horizon, ("common_window_time_horizon_relative_change",)
    )
    horizon_least_change = _metric(
        time_horizon, ("least_window_time_horizon_relative_change",)
    )
    horizon_threshold = _threshold(
        time_horizon,
        "max_relative_change",
        DEFAULT_MAX_TIME_HORIZON_RELATIVE_CHANGE,
    )

    statistics = replicate.get("statistics", {})
    config = replicate.get("config", {})
    if not isinstance(statistics, dict):
        statistics = {}
    if not isinstance(config, dict):
        config = {}
    n_reports = int(statistics.get("n_reports", 0) or 0)
    n_finite = int(statistics.get("n_finite_means", 0) or 0)
    ensemble_mean = float(statistics.get("ensemble_mean", float("nan")))
    mean_rel_spread = float(statistics.get("mean_rel_spread", float("inf")))
    combined_sem_rel = float(statistics.get("combined_sem_rel", float("inf")))
    spread_threshold = float(
        config.get("max_mean_rel_spread", DEFAULT_MAX_MEAN_REL_SPREAD)
    )
    sem_threshold = float(
        config.get("max_combined_sem_rel", DEFAULT_MAX_COMBINED_SEM_REL)
    )

    gates = [
        evaluate_scalar_gate(
            "full_grid_gate_failed_before_coarse_exclusion",
            0.0 if not bool(full_grid.get("passed", False)) else 1.0,
            0.0,
            atol=0.0,
            rtol=0.0,
            notes="High-grid admission is only meaningful as an exception to a failed full-grid ladder.",
        ),
        evaluate_scalar_gate(
            "full_grid_failure_limited_to_grid_difference",
            float(len(unexpected_full_failures)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="metrics",
            notes="The full-grid gate may fail only grid-difference metrics, not stationarity/sample gates.",
        ),
        evaluate_scalar_gate(
            "excluded_coarse_grid_labels_present",
            float(len(missing_excluded)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="labels",
            notes="Every excluded grid must be present in the failed full-grid ladder.",
        ),
        evaluate_scalar_gate(
            "retained_high_grid_labels_present",
            float(len(missing_retained)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="labels",
            notes="The retained high grids must also be present in the failed full-grid ladder.",
        ),
        evaluate_scalar_gate(
            "high_grid_gate_failure_count",
            float(high_grid_failed_count),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="Every retained high-grid convergence gate used for admission must pass.",
        ),
        evaluate_scalar_gate(
            "high_grid_label_mismatch_count",
            float(high_grid_label_mismatch_count),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="Every high-grid gate must use exactly the retained-grid labels.",
        ),
        evaluate_scalar_gate(
            "high_grid_common_window_pairwise_difference",
            max_high_grid_common_diff,
            0.0,
            atol=max_pairwise_threshold,
            rtol=0.0,
            notes="Highest-grid common-window heat-flux means must agree under the production threshold.",
        ),
        evaluate_scalar_gate(
            "high_grid_least_window_pairwise_difference",
            max_high_grid_least_diff,
            0.0,
            atol=max_pairwise_threshold,
            rtol=0.0,
            notes="Highest-grid least-trending-window heat-flux means must agree under the production threshold.",
        ),
        evaluate_scalar_gate(
            "time_horizon_gate_failure_count",
            0.0 if bool(time_horizon.get("passed", False)) else 1.0,
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="Late high-grid horizons must be stable before coarse-grid exclusion can be considered.",
        ),
        evaluate_scalar_gate(
            "time_horizon_common_window_change",
            horizon_common_change,
            0.0,
            atol=horizon_threshold,
            rtol=0.0,
            notes="High-grid averaged common-window means must be stable across late final times.",
        ),
        evaluate_scalar_gate(
            "time_horizon_least_window_change",
            horizon_least_change,
            0.0,
            atol=horizon_threshold,
            rtol=0.0,
            notes="High-grid averaged least-window means must be stable across late final times.",
        ),
        evaluate_scalar_gate(
            "replicate_ensemble_gate_failure_count",
            0.0 if bool(replicate.get("passed", False)) else 1.0,
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="A seed/timestep replicated late transport window must pass.",
        ),
        evaluate_scalar_gate(
            "replicate_count_shortfall",
            float(max(0, min_replicates - n_reports)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="replicates",
            notes="At least three replicated reports are required for this scoped exception.",
        ),
        evaluate_scalar_gate(
            "finite_replicate_count_shortfall",
            float(max(0, min_replicates - n_finite)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="replicates",
            notes="Replicate means must be finite.",
        ),
        evaluate_scalar_gate(
            "replicate_mean_relative_spread",
            mean_rel_spread,
            0.0,
            atol=spread_threshold,
            rtol=0.0,
            notes="Seed/timestep spread of late-window means.",
        ),
        evaluate_scalar_gate(
            "replicate_combined_sem_relative",
            combined_sem_rel,
            0.0,
            atol=sem_threshold,
            rtol=0.0,
            notes="Combined uncertainty estimate normalized by the ensemble mean.",
        ),
        evaluate_scalar_gate(
            "nonzero_transport_mean_floor",
            0.0
            if math.isfinite(ensemble_mean) and abs(ensemble_mean) >= value_floor
            else 1.0,
            0.0,
            atol=0.0,
            rtol=0.0,
            notes="Admission must not be based on startup/noise-floor heat flux.",
        ),
    ]
    report = gate_report(case, "external_vmec_high_grid_admission_policy", gates)
    blockers = [gate.metric for gate in report.gates if not gate.passed]
    passed = bool(report.passed)
    return _json_clean(
        {
            "kind": "external_vmec_high_grid_admission_gate",
            "case": case,
            "passed": passed,
            "gate_index_include": True,
            "claim_level": (
                "passed_high_grid_transport_holdout_admission_under_coarse_grid_exclusion"
                if passed
                else "blocked_high_grid_transport_holdout_admission"
            ),
            "claim_scope": (
                "The case is eligible as a high-grid nonlinear transport holdout with "
                "the listed coarse grids excluded. This is not a universal absolute-flux "
                "prediction or normal full-ladder convergence claim."
            ),
            "inputs": {
                "full_grid_gate": _repo_relative_path(full_grid_gate_path),
                "high_grid_gates": [
                    _repo_relative_path(path) for path in high_grid_gate_paths
                ],
                "time_horizon_gate": _repo_relative_path(time_horizon_gate_path),
                "replicate_ensemble_gate": _repo_relative_path(replicate_ensemble_path),
            },
            "policy": {
                "excluded_grid_labels": list(excluded_grid_labels),
                "retained_grid_labels": list(retained_grid_labels),
                "allowed_full_grid_failure_metrics": sorted(allowed_failures),
                "min_replicates": int(min_replicates),
                "value_floor": float(value_floor),
                "calibration_use": "eligible_as_scoped_high_grid_holdout"
                if passed
                else "blocked",
                "restrictions": [
                    "do not describe as full n48/n64/n80 convergence",
                    "do not use for universal absolute-flux promotion without separate calibration gates",
                    "retain the coarse-grid failure sidecar with the admitted artifact",
                    "rerun if physics, dissipation, flux-tube, or resolution settings change",
                ],
            },
            "summary": {
                "full_grid_failed_metrics": sorted(_failed_metrics(full_grid)),
                "unexpected_full_grid_failed_metrics": unexpected_full_failures,
                "missing_excluded_grid_labels": missing_excluded,
                "missing_retained_grid_labels": missing_retained,
                "max_high_grid_common_difference": max_high_grid_common_diff,
                "max_high_grid_least_difference": max_high_grid_least_diff,
                "high_grid_pairwise_threshold": max_pairwise_threshold,
                "time_horizon_common_change": horizon_common_change,
                "time_horizon_least_change": horizon_least_change,
                "time_horizon_threshold": horizon_threshold,
                "replicate_n_reports": n_reports,
                "replicate_n_finite_means": n_finite,
                "replicate_ensemble_mean": ensemble_mean,
                "replicate_mean_rel_spread": mean_rel_spread,
                "replicate_combined_sem_rel": combined_sem_rel,
                "replicate_mean_rel_spread_threshold": spread_threshold,
                "replicate_combined_sem_rel_threshold": sem_threshold,
            },
            "literature_policy": {
                "summary": (
                    "Nonlinear gyrokinetic heat-flux admission follows the benchmark "
                    "practice of comparing saturated late-time traces, resolution ladders, "
                    "and uncertainty of time averages. Coarse-grid exclusion is allowed "
                    "only when the retained higher grids, horizon stability, and replicated "
                    "transport windows pass fail-closed gates."
                ),
                "anchors": [
                    {
                        "name": "Dimits et al. 2000 Cyclone nonlinear heat-flux benchmark",
                        "url": "https://doi.org/10.1063/1.873896",
                    },
                    {
                        "name": "Gonzalez-Jerez et al. 2022 W7-X stella/GENE benchmark",
                        "url": "https://doi.org/10.1017/S0022377822000393",
                    },
                    {
                        "name": "Mandell et al. 2024 GX nonlinear convergence and benchmark practice",
                        "url": "https://doi.org/10.1017/S0022377822000617",
                    },
                    {
                        "name": "Hoffmann, Frei & Ricci 2023 nonlinear moment/GK convergence study",
                        "url": "https://arxiv.org/abs/2308.01016",
                    },
                    {
                        "name": "Oberparleiter et al. 2016 uncertainty and stopping rule for nonlinear gyrokinetics",
                        "url": "https://doi.org/10.1063/1.4960039",
                    },
                ],
            },
            "promotion_gate": {
                "passed": passed,
                "blockers": blockers,
                "reason": (
                    "high-grid admission policy passed; eligible for scoped holdout metadata"
                    if passed
                    else "high-grid admission policy failed"
                ),
            },
            "gate_report": gate_report_to_dict(report),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-grid-gate", type=Path, required=True)
    parser.add_argument("--high-grid-gate", type=Path, action="append", required=True)
    parser.add_argument("--time-horizon-gate", type=Path, required=True)
    parser.add_argument("--replicate-ensemble", type=Path, required=True)
    parser.add_argument("--excluded-grid-label", action="append", required=True)
    parser.add_argument("--retained-grid-label", action="append", required=True)
    parser.add_argument("--case", default="External-VMEC high-grid admission")
    parser.add_argument("--min-replicates", type=int, default=DEFAULT_MIN_REPLICATES)
    parser.add_argument("--value-floor", type=float, default=DEFAULT_VALUE_FLOOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_high_grid_admission_payload(
        full_grid_gate_path=args.full_grid_gate,
        high_grid_gate_paths=args.high_grid_gate,
        time_horizon_gate_path=args.time_horizon_gate,
        replicate_ensemble_path=args.replicate_ensemble,
        excluded_grid_labels=args.excluded_grid_label,
        retained_grid_labels=args.retained_grid_label,
        case=args.case,
        min_replicates=args.min_replicates,
        value_floor=args.value_floor,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"out": _repo_relative_path(args.out), "passed": payload["passed"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
