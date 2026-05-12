#!/usr/bin/env python3
"""Build a bounded QI VMEC/Boozer robustness scan artifact.

The default path is intentionally artifact-only: it summarizes the tracked
parity JSON and any explicitly supplied rerun observation.  Use ``--live`` to
run optional backend scan points; live points are bounded by ``--timeout-sec``
and ``--max-evaluations``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import sys
from typing import Any, Callable, NamedTuple, cast

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARITY_SCRIPT = ROOT / "tools" / "build_vmec_boozer_parity_matrix.py"
DEFAULT_PARITY_JSON = ROOT / "docs" / "_static" / "vmec_boozer_parity_matrix.json"
DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_qi_robustness.json"
MIN_BOOZER_MODE_COUNT = 21
DRIFT_TOLERANCE = 8.0e-2


class ScanPoint(NamedTuple):
    case_name: str
    label: str
    ntheta: int
    mboz: int = MIN_BOOZER_MODE_COUNT
    nboz: int = MIN_BOOZER_MODE_COUNT


DEFAULT_SCAN_POINTS: tuple[ScanPoint, ...] = (
    ScanPoint("nfp3_QI_fixed_resolution_final", "QI fixed resolution", 8),
    ScanPoint("nfp3_QI_fixed_resolution_final", "QI fixed resolution", 16),
    ScanPoint("nfp3_QI_fixed_resolution_final", "QI fixed resolution", 16, 25, 25),
)


Reporter = Callable[..., dict[str, object]]


def _load_parity_module() -> Any:
    spec = importlib.util.spec_from_file_location("build_vmec_boozer_parity_matrix", PARITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PARITY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _bool(value: object) -> bool:
    return bool(value) if value is not None else False


def _max_ratio(row: dict[str, object]) -> float | None:
    pairs = (
        ("equal_arc_core_worst_normalized_max_abs", "equal_arc_core_tolerance"),
        ("equal_arc_core_worst_scalar_rel", "equal_arc_core_tolerance"),
        ("equal_arc_derivative_worst_normalized_max_abs", "equal_arc_derivative_tolerance"),
        ("equal_arc_metric_worst_normalized_max_abs", "equal_arc_metric_tolerance"),
        ("equal_arc_drift_worst_normalized_max_abs", "equal_arc_drift_tolerance"),
    )
    ratios: list[float] = []
    for value_key, tolerance_key in pairs:
        value = _finite_float(row.get(value_key))
        tolerance = _finite_float(row.get(tolerance_key))
        if value is None or tolerance is None or tolerance <= 0.0:
            continue
        ratios.append(value / tolerance)
    return max(ratios) if ratios else None


def _row_from_report(point: ScanPoint, report: dict[str, object], *, source: str) -> dict[str, object]:
    available = _bool(report.get("available"))
    mboz = int(cast(Any, report.get("mboz", point.mboz)))
    nboz = int(cast(Any, report.get("nboz", point.nboz)))
    row: dict[str, object] = {
        "source": source,
        "case_name": point.case_name,
        "label": point.label,
        "ntheta": int(point.ntheta),
        "mboz": mboz,
        "nboz": nboz,
        "mode_floor_passed": mboz >= MIN_BOOZER_MODE_COUNT and nboz >= MIN_BOOZER_MODE_COUNT,
        "available": available,
        "status": str(report.get("status", "unavailable")),
        "reason": report.get("reason"),
        "error": report.get("error"),
        "input_path": report.get("input_path"),
        "wout_path": report.get("wout_path"),
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
        "equal_arc_core_passed": _bool(report.get("equal_arc_core_passed")),
        "equal_arc_bgrad_passed": _bool(report.get("equal_arc_bgrad_passed", report.get("equal_arc_derivative_passed"))),
        "equal_arc_metric_passed": _bool(report.get("equal_arc_metric_passed")),
        "equal_arc_drift_passed": _bool(report.get("equal_arc_drift_passed")),
    }
    row["equal_arc_all_passed"] = bool(
        row["available"]
        and row["mode_floor_passed"]
        and row["equal_arc_core_passed"]
        and row["equal_arc_bgrad_passed"]
        and row["equal_arc_metric_passed"]
        and row["equal_arc_drift_passed"]
    )
    row["cost_rank"] = int(point.ntheta) * int(mboz) * int(nboz)
    row["max_tolerance_ratio"] = _max_ratio(row)
    return row


def _drift_only_observation(*, drift: float, source: str) -> dict[str, object]:
    passed = float(drift) <= DRIFT_TOLERANCE
    row: dict[str, object] = {
        "source": source,
        "case_name": "nfp3_QI_fixed_resolution_final",
        "label": "QI fixed resolution rerun observation",
        "ntheta": 8,
        "mboz": MIN_BOOZER_MODE_COUNT,
        "nboz": MIN_BOOZER_MODE_COUNT,
        "mode_floor_passed": True,
        "available": True,
        "status": "drift_observation_only",
        "reason": "rerun supplied only the limiting drift mismatch",
        "error": None,
        "equal_arc_core_worst_normalized_max_abs": None,
        "equal_arc_core_worst_scalar_rel": None,
        "equal_arc_derivative_worst_normalized_max_abs": None,
        "equal_arc_metric_worst_normalized_max_abs": None,
        "equal_arc_drift_worst_normalized_max_abs": float(drift),
        "equal_arc_core_tolerance": None,
        "equal_arc_derivative_tolerance": None,
        "equal_arc_metric_tolerance": None,
        "equal_arc_drift_tolerance": DRIFT_TOLERANCE,
        "equal_arc_core_passed": False,
        "equal_arc_bgrad_passed": False,
        "equal_arc_metric_passed": False,
        "equal_arc_drift_passed": passed,
        "equal_arc_all_passed": False,
        "cost_rank": 8 * MIN_BOOZER_MODE_COUNT * MIN_BOOZER_MODE_COUNT,
        "max_tolerance_ratio": float(drift) / DRIFT_TOLERANCE,
    }
    return row


def _rows_from_parity_json(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_qi = payload.get("qi_seed_robustness", {})
    qi_rows = raw_qi.get("rows", []) if isinstance(raw_qi, dict) else []
    rows: list[dict[str, object]] = []
    if not isinstance(qi_rows, list):
        return rows
    for raw in qi_rows:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("case_name")) != "nfp3_QI_fixed_resolution_final":
            continue
        if not _bool(raw.get("qi_validation_evaluated", raw.get("available"))):
            continue
        point = ScanPoint(
            str(raw.get("case_name")),
            str(raw.get("label", "QI fixed resolution")),
            int(cast(Any, raw.get("ntheta", 8))),
            int(cast(Any, raw.get("mboz", MIN_BOOZER_MODE_COUNT))),
            int(cast(Any, raw.get("nboz", MIN_BOOZER_MODE_COUNT))),
        )
        rows.append(_row_from_report(point, raw, source=f"parity_json:{path.name}"))
    return rows


def _worker(payload: tuple[str, int, int, int], out: mp.Queue[dict[str, object]]) -> None:
    case_name, ntheta, mboz, nboz = payload
    try:
        mod = _load_parity_module()
        reporter = cast(Reporter, getattr(mod, "_default_reporter"))
        out.put(reporter(case_name=case_name, ntheta=ntheta, mboz=mboz, nboz=nboz))
    except Exception as exc:  # pragma: no cover - subprocess diagnostic detail
        out.put(
            {
                "available": False,
                "case_name": case_name,
                "ntheta": ntheta,
                "mboz": mboz,
                "nboz": nboz,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_point_with_timeout(point: ScanPoint, timeout_sec: float) -> dict[str, object]:
    ctx = mp.get_context("spawn")
    out: mp.Queue[dict[str, object]] = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_worker,
        args=((point.case_name, int(point.ntheta), int(point.mboz), int(point.nboz)), out),
    )
    proc.start()
    proc.join(float(timeout_sec))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {
            "available": False,
            "case_name": point.case_name,
            "ntheta": int(point.ntheta),
            "mboz": int(point.mboz),
            "nboz": int(point.nboz),
            "error": f"timeout after {float(timeout_sec):.1f} s",
        }
    try:
        return out.get_nowait()
    except queue.Empty:
        return {
            "available": False,
            "case_name": point.case_name,
            "ntheta": int(point.ntheta),
            "mboz": int(point.mboz),
            "nboz": int(point.nboz),
            "error": f"backend process exited with code {proc.exitcode}",
        }


def _config_key(row: dict[str, object]) -> tuple[str, int, int, int]:
    return (
        str(row.get("case_name")),
        int(cast(Any, row.get("ntheta", -1))),
        int(cast(Any, row.get("mboz", -1))),
        int(cast(Any, row.get("nboz", -1))),
    )


def _drift_only_failed_configs(rows: list[dict[str, object]]) -> set[tuple[str, int, int, int]]:
    return {
        _config_key(row)
        for row in rows
        if str(row.get("status")) == "drift_observation_only"
        and not bool(row.get("equal_arc_drift_passed", False))
    }


def _select_floor(rows: list[dict[str, object]]) -> dict[str, object] | None:
    blocked_configs = _drift_only_failed_configs(rows)
    complete_passes = [
        row
        for row in rows
        if bool(row.get("equal_arc_all_passed", False)) and _config_key(row) not in blocked_configs
    ]
    if not complete_passes:
        return None
    return min(
        complete_passes,
        key=lambda row: (
            int(cast(Any, row.get("cost_rank", 10**18))),
            float(cast(Any, row.get("max_tolerance_ratio", 10**18))),
        ),
    )


def build_qi_robustness_scan(
    *,
    scan_points: tuple[ScanPoint, ...] = DEFAULT_SCAN_POINTS,
    reporter: Reporter | None = None,
    parity_json: Path | None = DEFAULT_PARITY_JSON,
    known_rerun_drift: float | None = None,
    live: bool = False,
    max_evaluations: int | None = None,
    timeout_sec: float = 120.0,
) -> dict[str, object]:
    """Return a JSON-ready robustness scan over QI field-line/Boozer settings."""

    rows: list[dict[str, object]] = []
    if parity_json is not None:
        rows.extend(_rows_from_parity_json(Path(parity_json)))
    if known_rerun_drift is not None:
        rows.append(_drift_only_observation(drift=float(known_rerun_drift), source="reported_json_only_rerun"))

    live_points = list(scan_points[: max_evaluations if max_evaluations is not None else len(scan_points)])
    if live:
        for point in live_points:
            if int(point.mboz) < MIN_BOOZER_MODE_COUNT or int(point.nboz) < MIN_BOOZER_MODE_COUNT:
                rows.append(
                    {
                        "source": "live_scan",
                        "case_name": point.case_name,
                        "label": point.label,
                        "ntheta": int(point.ntheta),
                        "mboz": int(point.mboz),
                        "nboz": int(point.nboz),
                        "mode_floor_passed": False,
                        "available": False,
                        "status": "underresolved_boozer_modes",
                        "error": None,
                        "equal_arc_all_passed": False,
                    }
                )
                continue
            if reporter is not None:
                report = reporter(
                    case_name=point.case_name,
                    ntheta=int(point.ntheta),
                    mboz=int(point.mboz),
                    nboz=int(point.nboz),
                )
            else:
                report = _run_point_with_timeout(point, timeout_sec)
            rows.append(_row_from_report(point, report, source="live_scan"))

    selected = _select_floor(rows)
    blocked_configs = _drift_only_failed_configs(rows)
    available_rows = [row for row in rows if bool(row.get("available", False))]
    complete_rows = [row for row in rows if _max_ratio(row) is not None and str(row.get("status")) != "drift_observation_only"]
    failed_complete_rows = [row for row in complete_rows if not bool(row.get("equal_arc_all_passed", False))]
    drift_only_failures = [
        row
        for row in rows
        if str(row.get("status")) == "drift_observation_only"
        and not bool(row.get("equal_arc_drift_passed", False))
    ]
    return {
        "kind": "vmec_boozer_qi_robustness_scan",
        "minimum_boozer_mode_count": MIN_BOOZER_MODE_COUNT,
        "drift_tolerance": DRIFT_TOLERANCE,
        "live_scan_requested": bool(live),
        "timeout_sec": float(timeout_sec),
        "scan_points": [point._asdict() for point in scan_points],
        "rows": rows,
        "selected_floor": None
        if selected is None
        else {
            "case_name": selected["case_name"],
            "ntheta": selected["ntheta"],
            "mboz": selected["mboz"],
            "nboz": selected["nboz"],
            "equal_arc_drift_worst_normalized_max_abs": selected.get(
                "equal_arc_drift_worst_normalized_max_abs"
            ),
            "max_tolerance_ratio": selected.get("max_tolerance_ratio"),
            "source": selected.get("source"),
        },
        "summary": {
            "n_rows": len(rows),
            "n_available": len(available_rows),
            "n_complete_rows": len(complete_rows),
            "n_complete_failed": len(failed_complete_rows),
            "n_drift_only_failures": len(drift_only_failures),
            "n_configs_blocked_by_newer_drift_observation": len(blocked_configs),
            "selected_floor_passed": selected is not None,
            "robustness_status": "floor_selected" if selected is not None else "open",
        },
        "notes": (
            "This scan does not relax the 0.08 QI drift tolerance. Complete rows "
            "must pass core/scalar/bgrad/metric/drift subgates before they can be "
            "selected as the per-case floor. Drift-only rerun observations are "
            "tracked as evidence but are not sufficient to select a passing floor."
        ),
    }


def write_qi_robustness_artifact(payload: dict[str, object], *, out: Path = DEFAULT_OUT) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(out)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--from-parity-json", type=Path, default=DEFAULT_PARITY_JSON)
    parser.add_argument("--known-rerun-drift", type=float, default=None)
    parser.add_argument("--live", action="store_true", help="Run bounded optional-backend scan points.")
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_qi_robustness_scan(
        parity_json=args.from_parity_json,
        known_rerun_drift=args.known_rerun_drift,
        live=bool(args.live),
        max_evaluations=args.max_evaluations,
        timeout_sec=float(args.timeout_sec),
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    print(json.dumps(write_qi_robustness_artifact(payload, out=args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
