#!/usr/bin/env python3
"""Build a machine-readable status summary for open research validation lanes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "_static" / "open_research_lane_status.png"

STATUS_ORDER = {"closed": 0, "partial": 1, "open": 2, "blocked": 3}
STATUS_COLORS = {
    "closed": "#2a9d8f",
    "partial": "#e9c46a",
    "open": "#f4a261",
    "blocked": "#d1495b",
}


def _read_json(root: Path, relative: str) -> dict[str, Any] | None:
    path = root / relative
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative} must contain a JSON object")
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


def _gate_failures(gate_report: dict[str, Any] | None) -> list[str]:
    if not gate_report:
        return []
    failures: list[str] = []
    for gate in gate_report.get("gates", []):
        if isinstance(gate, dict) and not bool(gate.get("passed", False)):
            failures.append(str(gate.get("metric", "unknown")))
    return failures


def _best_recurrence_candidate(payload: dict[str, Any] | None) -> dict[str, Any]:
    rows = [] if payload is None else payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {"label": None, "mean_abs_error": None, "tail_std_ratio": None, "hermite_tail": None}
    finite_rows = [row for row in rows if isinstance(row, dict) and _finite_float(row.get("mean_abs_error")) is not None]
    if not finite_rows:
        return {"label": None, "mean_abs_error": None, "tail_std_ratio": None, "hermite_tail": None}
    best = min(finite_rows, key=lambda row: float(row["mean_abs_error"]))
    tail_std = _finite_float(best.get("tail_std"), 0.0) or 0.0
    ref_tail_std = _finite_float(best.get("reference_tail_std"), 0.0) or 0.0
    ratio = None if ref_tail_std <= 0.0 else tail_std / ref_tail_std
    return {
        "label": str(best.get("label", "unknown")),
        "mean_abs_error": _finite_float(best.get("mean_abs_error")),
        "tail_std_ratio": ratio,
        "hermite_tail": _finite_float(best.get("hermite_tail_at_tmax")),
        "source_path": best.get("source_path"),
    }


def _best_hypercollision_probe(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    rows = [] if payload is None else payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return None
    finite_rows = [row for row in rows if isinstance(row, dict) and _finite_float(row.get("mean_abs_error")) is not None]
    if not finite_rows:
        return None
    best = min(finite_rows, key=lambda row: float(row["mean_abs_error"]))
    tail_std = _finite_float(best.get("tail_std"), 0.0) or 0.0
    ref_tail_std = _finite_float(best.get("reference_tail_std"), 0.0) or 0.0
    ratio = None if ref_tail_std <= 0.0 else tail_std / ref_tail_std
    return {
        "label": str(best.get("label", "unknown")),
        "mean_abs_error": _finite_float(best.get("mean_abs_error")),
        "tail_std_ratio": ratio,
        "hermite_tail": _finite_float(best.get("hermite_tail_at_tmax")),
        "free_energy_ratio": _finite_float(best.get("free_energy_at_tmax_over_initial")),
        "source_path": best.get("source_path"),
        "validation_status": payload.get("validation_status"),
    }


def _holdout_counts(report: dict[str, Any] | None) -> tuple[int, int, list[str]]:
    if report is None:
        return 0, 0, []
    points = report.get("points", [])
    if not isinstance(points, list):
        return 0, 0, []
    train = 0
    holdout = 0
    names: list[str] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        split = str(point.get("split", ""))
        if split == "train":
            train += 1
        if split == "holdout":
            holdout += 1
            names.append(str(point.get("case", "unknown")))
    return train, holdout, names


def build_status_payload(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return a JSON-ready status payload for active research lanes."""

    root = Path(root)
    zonal_ref = _read_json(root, "docs/_static/w7x_zonal_reference_compare.json")
    zonal_recurrence = _read_json(root, "docs/_static/w7x_zonal_recurrence_sweep_kx070.json")
    zonal_hypercollision = _read_json(root, "docs/_static/w7x_zonal_hypercollision_probe_kx070.json")
    fluct = _read_json(root, "docs/_static/w7x_fluctuation_spectrum_panel.json")
    ql_inputs = _read_json(root, "docs/_static/quasilinear_validated_calibration_inputs.json")
    ql_report = _read_json(root, "docs/_static/quasilinear_stellarator_train_holdout_report.json")
    cth_gate = _read_json(root, "docs/_static/external_vmec_cth_like_grid_convergence_gate.json")
    geom = _read_json(root, "docs/_static/differentiable_geometry_bridge.json")
    profile = _read_json(root, "docs/_static/nonlinear_sharding_profile_office_gpu.json")

    zonal_failures = _gate_failures(zonal_ref.get("gate_report") if zonal_ref else None)
    best_recurrence = _best_recurrence_candidate(zonal_recurrence)
    best_hypercollision = _best_hypercollision_probe(zonal_hypercollision)
    zonal_status = "closed" if zonal_ref and not zonal_failures and zonal_ref.get("validation_status") == "closed" else "open"

    train_count, holdout_count, holdout_names = _holdout_counts(ql_report)
    ql_passed = bool(ql_report.get("passed", False)) if ql_report else False
    cth_passed = bool((cth_gate or {}).get("gate_report", {}).get("passed", False))

    geom_sensitivity = (geom or {}).get("sensitivity", {}) if isinstance((geom or {}).get("sensitivity", {}), dict) else {}
    geom_inverse = (geom or {}).get("geometry_inverse_design_report", {})
    geom_uq = (geom or {}).get("uq", {})
    geom_max_abs = _finite_float(geom_sensitivity.get("max_abs_ad_fd_error"))
    geom_inverse_res = _finite_float(geom_inverse.get("final_residual_norm")) if isinstance(geom_inverse, dict) else None
    geom_rank = int(geom_uq.get("sensitivity_map_rank", 0)) if isinstance(geom_uq, dict) else 0

    profile_identity = bool((profile or {}).get("identity_gate_pass", False))
    profile_speedup = _finite_float((profile or {}).get("engineering_speedup"))

    lanes = [
        {
            "lane": "W7-X zonal long-window recurrence/damping",
            "status": zonal_status,
            "claim_level": "open_physical_closure_not_normalization",
            "primary_artifacts": [
                "docs/_static/w7x_zonal_reference_compare.json",
                "docs/_static/w7x_zonal_recurrence_sweep_kx070.json",
                "docs/_static/w7x_zonal_hypercollision_probe_kx070.json",
            ],
            "key_metrics": {
                "failed_reference_gates": zonal_failures,
                "best_bounded_candidate": best_recurrence,
                "best_constant_hypercollision_probe": best_hypercollision,
            },
            "next_action": (
                "Move beyond constant Hermite damping: test a physically motivated closure/operator and promote only if "
                "residual, tail-envelope, and moment-tail gates pass together."
            ),
        },
        {
            "lane": "W7-X fluctuation spectrum and TEM/multi-flux extension",
            "status": "partial" if bool(fluct and fluct.get("source_gate_passed")) else "open",
            "claim_level": "validated_simulation_spectrum_tem_extension_open",
            "primary_artifacts": ["docs/_static/w7x_fluctuation_spectrum_panel.json", "docs/_static/tem_mismatch_table.csv"],
            "key_metrics": {
                "time_samples": (fluct or {}).get("time_samples"),
                "time_window": [(fluct or {}).get("time_min"), (fluct or {}).get("time_max")],
                "dominant_phi_ky": (fluct or {}).get("dominant_phi_ky"),
                "dominant_heat_flux_ky": (fluct or {}).get("dominant_heat_flux_ky"),
            },
            "next_action": (
                "Add W7-X multi-alpha/multi-surface ITG and kinetic-electron density-gradient/TEM scans before "
                "broad stellarator-validation claims."
            ),
        },
        {
            "lane": "Nonlinear holdouts for quasilinear absolute-flux promotion",
            "status": "closed" if ql_passed else "open",
            "claim_level": "diagnostic_calibration_dataset_not_absolute_flux" if not ql_passed else "calibrated_absolute_flux",
            "primary_artifacts": [
                "docs/_static/quasilinear_validated_calibration_inputs.json",
                "docs/_static/quasilinear_stellarator_train_holdout_report.json",
                "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
            ],
            "key_metrics": {
                "validated_inputs_passed": bool((ql_inputs or {}).get("passed", False)),
                "train_points": train_count,
                "holdout_points": holdout_count,
                "holdout_cases": holdout_names,
                "calibration_report_passed": ql_passed,
                "cth_like_external_vmec_converged": cth_passed,
            },
            "next_action": (
                "Add at least one more grid/window-converged nonlinear holdout; keep CTH-like external VMEC excluded "
                "until its common-window and grid-refinement gates pass."
            ),
        },
        {
            "lane": "vmec_jax / booz_xform_jax differentiable geometry bridge",
            "status": "partial" if geom_max_abs is not None and geom_rank >= 2 else "open",
            "claim_level": "contract_gradient_gate_not_full_stellarator_optimization",
            "primary_artifacts": ["docs/_static/differentiable_geometry_bridge.json"],
            "key_metrics": {
                "max_abs_ad_fd_error": geom_max_abs,
                "inverse_residual_norm": geom_inverse_res,
                "sensitivity_rank": geom_rank,
                "vmec_jax_available": (geom or {}).get("backend_info", {}).get("vmec_jax_available"),
                "booz_xform_jax_api_available": (geom or {}).get("booz_xform_jax_api_available"),
            },
            "next_action": (
                "Connect a real in-memory vmec_jax/booz_xform_jax output to FluxTubeGeometryData and add parity plus "
                "geometry-gradient gates before optimization claims."
            ),
        },
        {
            "lane": "Profiler-backed nonlinear hot-path optimization",
            "status": "partial" if profile_identity else "open",
            "claim_level": "profile_identity_artifact_no_speedup_claim",
            "primary_artifacts": ["docs/_static/nonlinear_sharding_profile_office_gpu.json"],
            "key_metrics": {
                "identity_gate_pass": profile_identity,
                "engineering_speedup": profile_speedup,
                "device_count": (profile or {}).get("device_count"),
                "backend": (profile or {}).get("default_backend"),
            },
            "next_action": (
                "Collect matched CPU/GPU profiler traces and optimize only persistent nonlinear bracket/field-solve hot paths; "
                "do not publish speedup claims until fresh profiler artifacts pass identity gates."
            ),
        },
    ]

    return {
        "kind": "open_research_lane_status",
        "claim_scope": "post_v1_5_development_tracking_no_unvalidated_promotion",
        "status_order": STATUS_ORDER,
        "lanes": lanes,
        "summary": {
            "n_lanes": len(lanes),
            "n_closed": sum(1 for lane in lanes if lane["status"] == "closed"),
            "n_partial": sum(1 for lane in lanes if lane["status"] == "partial"),
            "n_open": sum(1 for lane in lanes if lane["status"] == "open"),
            "n_blocked": sum(1 for lane in lanes if lane["status"] == "blocked"),
        },
    }


def write_status_artifacts(payload: dict[str, Any], *, out_png: Path = DEFAULT_OUT) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for the lane-status payload."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_png.with_suffix(".json")
    out_csv = out_png.with_suffix(".csv")
    out_pdf = out_png.with_suffix(".pdf")

    out_json.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = ["lane", "status", "claim_level", "primary_artifacts", "next_action"]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for lane in payload["lanes"]:
            writer.writerow(
                {
                    "lane": lane["lane"],
                    "status": lane["status"],
                    "claim_level": lane["claim_level"],
                    "primary_artifacts": ";".join(lane["primary_artifacts"]),
                    "next_action": lane["next_action"],
                }
            )

    set_plot_style()
    lanes = payload["lanes"]
    y = np.arange(len(lanes))
    colors = [STATUS_COLORS.get(str(lane["status"]), "#777777") for lane in lanes]
    values = [STATUS_ORDER.get(str(lane["status"]), 3) for lane in lanes]
    labels = [textwrap.fill(str(lane["lane"]), width=38) for lane in lanes]

    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    ax.barh(y, values, color=colors, edgecolor="#333333", alpha=0.95)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 3.2)
    ax.set_xticks([0, 1, 2, 3], ["closed", "partial", "open", "blocked"])
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("Open research lanes: executable status and claim scope")
    ax.grid(axis="x", alpha=0.25)
    for yi, lane, value in zip(y, lanes, values, strict=True):
        metric = ""
        key_metrics = lane.get("key_metrics", {})
        if lane["lane"].startswith("W7-X zonal"):
            failed = key_metrics.get("failed_reference_gates", [])
            metric = f"failed gates: {len(failed)}"
        elif lane["lane"].startswith("W7-X fluctuation"):
            metric = f"samples: {key_metrics.get('time_samples')}"
        elif lane["lane"].startswith("Nonlinear holdouts"):
            metric = f"holdouts: {key_metrics.get('holdout_points')}, promoted: {key_metrics.get('calibration_report_passed')}"
        elif lane["lane"].startswith("vmec_jax"):
            metric = f"AD-FD max: {key_metrics.get('max_abs_ad_fd_error'):.1e}"
        elif lane["lane"].startswith("Profiler"):
            speed = key_metrics.get("engineering_speedup")
            metric = "speedup: n/a" if speed is None else f"speedup: {speed:.2f}x"
        ax.text(min(value + 0.06, 3.05), yi, metric, va="center", ha="left", fontsize=8.2)

    caption = (
        "Partial means a bounded diagnostic/gate exists, but the broader manuscript claim remains scoped. "
        "Open means no promotion until the listed physics or profiler gate passes."
    )
    fig.text(0.5, 0.025, caption, fontsize=8.2, color="#333333", ha="center")
    fig.subplots_adjust(left=0.35, right=0.97, top=0.90, bottom=0.14)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {"png": str(out_png), "pdf": str(out_pdf), "json": str(out_json), "csv": str(out_csv)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_status_payload(Path(args.root))
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_status_artifacts(payload, out_png=Path(args.out))
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
