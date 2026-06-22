#!/usr/bin/env python3
"""Build a matched baseline-to-optimized nonlinear transport audit.

This tool consumes already-replicated nonlinear-window ensemble artifacts. It
never launches simulations and fails closed when either side of the matched pair
is absent or not promotion-ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


DEFAULT_BASELINE_ENSEMBLE = (
    ROOT
    / "docs"
    / "_static"
    / "external_vmec_circular_replicates"
    / "circular_replicate_t700_ensemble_gate.json"
)
DEFAULT_OPTIMIZED_ENSEMBLE = (
    ROOT
    / "docs"
    / "_static"
    / "optimized_equilibrium_replicates"
    / "optimized_equilibrium_replicate_t700_ensemble_gate.json"
)
DEFAULT_SELECTED_OPTIMIZED_AUDIT = ROOT / "docs" / "_static" / "production_nonlinear_optimization_guard.json"


def _repo_relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing JSON artifact: {_repo_relative(path)}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON artifact {_repo_relative(path)}: {exc}"
    if not isinstance(payload, dict):
        return None, f"artifact is not a JSON object: {_repo_relative(path)}"
    return payload, None


def _artifact_passed(payload: Mapping[str, Any]) -> bool:
    if bool(payload.get("passed", False)):
        return True
    for key in ("gate_report", "promotion_gate"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and bool(nested.get("passed", False)):
            return True
    return False


def _ensemble_row(path: Path, payload: Mapping[str, Any] | None, *, role: str) -> dict[str, Any]:
    if payload is None:
        return {
            "role": role,
            "path": _repo_relative(path),
            "present": False,
            "passed": False,
            "qualifies": False,
            "blockers": [f"{role}_ensemble_missing"],
        }

    stats = payload.get("statistics")
    stats_map: Mapping[str, Any] = stats if isinstance(stats, Mapping) else {}
    kind = str(payload.get("kind", ""))
    mean = _finite_float(stats_map.get("ensemble_mean"))
    combined_sem = _finite_float(stats_map.get("combined_sem"))
    combined_sem_rel = _finite_float(stats_map.get("combined_sem_rel"))
    mean_spread = _finite_float(stats_map.get("mean_spread"))
    mean_rel_spread = _finite_float(stats_map.get("mean_rel_spread"))
    n_reports = int(_finite_float(stats_map.get("n_reports")) or 0)
    passed = _artifact_passed(payload)

    blockers: list[str] = []
    if kind != "nonlinear_window_ensemble_report":
        blockers.append(f"{role}_not_nonlinear_window_ensemble_report")
    if not passed:
        blockers.append(f"{role}_ensemble_gate_not_passed")
    if mean is None:
        blockers.append(f"{role}_missing_finite_ensemble_mean")
    if combined_sem is None or combined_sem_rel is None:
        blockers.append(f"{role}_missing_uncertainty_metadata")
    if mean_spread is None or mean_rel_spread is None:
        blockers.append(f"{role}_missing_spread_metadata")
    if n_reports < 2:
        blockers.append(f"{role}_insufficient_replicates")

    return {
        "role": role,
        "path": _repo_relative(path),
        "present": True,
        "kind": kind,
        "case": str(payload.get("case", "")),
        "comparison": str(payload.get("comparison", "")),
        "claim_level": str(payload.get("claim_level", "")),
        "passed": passed,
        "qualifies": not blockers,
        "blockers": blockers,
        "ensemble_mean": mean,
        "combined_sem": combined_sem,
        "combined_sem_rel": combined_sem_rel,
        "mean_spread": mean_spread,
        "mean_rel_spread": mean_rel_spread,
        "n_reports": n_reports,
        "rows": payload.get("rows", []),
    }


def _selected_audit_row(
    path: Path | None,
    payload: Mapping[str, Any] | None,
    *,
    optimized_ensemble_path: Path,
) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "present": False,
            "required": False,
            "passed": True,
            "optimized_ensemble_selected": True,
            "blockers": [],
        }
    if payload is None:
        return {
            "path": _repo_relative(path),
            "present": False,
            "required": True,
            "passed": False,
            "optimized_ensemble_selected": False,
            "blockers": ["selected_optimized_audit_missing"],
        }

    promotion_gate = payload.get("promotion_gate")
    promotion_passed = isinstance(promotion_gate, Mapping) and bool(promotion_gate.get("passed", False))
    optimized_path = _repo_relative(optimized_ensemble_path)
    optimized_rows = payload.get("optimized_equilibrium_artifacts")
    selected_paths = []
    if isinstance(optimized_rows, list):
        selected_paths = [
            str(row.get("path", ""))
            for row in optimized_rows
            if isinstance(row, Mapping) and bool(row.get("qualifies_for_production_optimization", False))
        ]
    selected = bool(optimized_path and optimized_path in selected_paths)
    blockers: list[str] = []
    if not promotion_passed:
        blockers.append("selected_optimized_audit_promotion_gate_not_passed")
    if not selected:
        blockers.append("optimized_ensemble_not_selected_by_audit")
    return {
        "path": _repo_relative(path),
        "present": True,
        "required": True,
        "passed": promotion_passed,
        "optimized_ensemble_selected": selected,
        "selected_optimized_ensemble_paths": selected_paths,
        "blockers": blockers,
    }


def build_audit(
    *,
    baseline_path: Path,
    optimized_path: Path,
    selected_optimized_audit_path: Path | None = DEFAULT_SELECTED_OPTIMIZED_AUDIT,
    case: str = "baseline_to_optimized_nonlinear_transport_audit",
    min_relative_reduction: float = 0.0,
    require_uncertainty_separation: bool = True,
) -> dict[str, Any]:
    """Return a fail-closed matched baseline-to-optimized transport audit."""

    baseline_payload, baseline_error = _load_json(baseline_path)
    optimized_payload, optimized_error = _load_json(optimized_path)
    selected_payload: dict[str, Any] | None = None
    selected_error: str | None = None
    if selected_optimized_audit_path is not None:
        selected_payload, selected_error = _load_json(selected_optimized_audit_path)

    baseline = _ensemble_row(baseline_path, baseline_payload, role="baseline")
    optimized = _ensemble_row(optimized_path, optimized_payload, role="optimized")
    selected_audit = _selected_audit_row(
        selected_optimized_audit_path,
        selected_payload,
        optimized_ensemble_path=optimized_path,
    )

    load_blockers = [
        item
        for item in (baseline_error, optimized_error, selected_error)
        if item is not None
    ]
    comparison: dict[str, Any] = {
        "baseline_mean": baseline.get("ensemble_mean"),
        "optimized_mean": optimized.get("ensemble_mean"),
        "absolute_delta": None,
        "relative_delta": None,
        "relative_reduction": None,
        "combined_uncertainty": None,
        "uncertainty_separation_sigma": None,
        "baseline_spread": baseline.get("mean_spread"),
        "optimized_spread": optimized.get("mean_spread"),
        "baseline_mean_rel_spread": baseline.get("mean_rel_spread"),
        "optimized_mean_rel_spread": optimized.get("mean_rel_spread"),
    }
    baseline_mean = _finite_float(baseline.get("ensemble_mean"))
    optimized_mean = _finite_float(optimized.get("ensemble_mean"))
    baseline_sem = _finite_float(baseline.get("combined_sem"))
    optimized_sem = _finite_float(optimized.get("combined_sem"))
    if baseline_mean is not None and optimized_mean is not None:
        delta = optimized_mean - baseline_mean
        scale = max(abs(baseline_mean), 1.0e-12)
        comparison["absolute_delta"] = delta
        comparison["relative_delta"] = delta / scale
        comparison["relative_reduction"] = (baseline_mean - optimized_mean) / scale
    if baseline_sem is not None and optimized_sem is not None:
        combined_uncertainty = math.sqrt(baseline_sem**2 + optimized_sem**2)
        comparison["combined_uncertainty"] = combined_uncertainty
        if comparison["absolute_delta"] is not None and combined_uncertainty > 0.0:
            comparison["uncertainty_separation_sigma"] = abs(float(comparison["absolute_delta"])) / combined_uncertainty

    blockers: list[str] = []
    blockers.extend(load_blockers)
    blockers.extend(str(item) for item in baseline.get("blockers", []))
    blockers.extend(str(item) for item in optimized.get("blockers", []))
    blockers.extend(str(item) for item in selected_audit.get("blockers", []))
    relative_reduction = _finite_float(comparison.get("relative_reduction"))
    reduction_passed = relative_reduction is not None and relative_reduction >= float(min_relative_reduction)
    if not reduction_passed:
        blockers.append("optimized_heat_flux_not_reduced_vs_baseline")
    separation = _finite_float(comparison.get("uncertainty_separation_sigma"))
    uncertainty_passed = separation is not None and separation >= 1.0
    if require_uncertainty_separation and not uncertainty_passed:
        blockers.append("baseline_optimized_difference_not_uncertainty_separated")

    gates = [
        {
            "metric": "baseline_replicated_ensemble_present",
            "passed": baseline.get("present") is True,
            "detail": str(baseline.get("path")),
        },
        {
            "metric": "optimized_replicated_ensemble_present",
            "passed": optimized.get("present") is True,
            "detail": str(optimized.get("path")),
        },
        {
            "metric": "baseline_replicated_ensemble_qualified",
            "passed": bool(baseline.get("qualifies", False)),
            "detail": "; ".join(str(item) for item in baseline.get("blockers", [])) or "qualified",
        },
        {
            "metric": "optimized_replicated_ensemble_qualified",
            "passed": bool(optimized.get("qualifies", False)),
            "detail": "; ".join(str(item) for item in optimized.get("blockers", [])) or "qualified",
        },
        {
            "metric": "selected_optimized_equilibrium_audit",
            "passed": bool(selected_audit.get("passed", False)) and bool(selected_audit.get("optimized_ensemble_selected", False)),
            "detail": "; ".join(str(item) for item in selected_audit.get("blockers", [])) or "selected optimized audit closed",
        },
        {
            "metric": "optimized_heat_flux_reduction",
            "passed": reduction_passed,
            "detail": f"relative_reduction={relative_reduction} min={min_relative_reduction}",
        },
        {
            "metric": "uncertainty_separated_difference",
            "passed": (not require_uncertainty_separation) or uncertainty_passed,
            "detail": f"sigma={separation} min=1.0 required={require_uncertainty_separation}",
        },
    ]
    passed = not blockers and all(bool(gate["passed"]) for gate in gates)
    return {
        "kind": "baseline_optimized_nonlinear_transport_audit",
        "claim_level": "matched_baseline_to_optimized_replicated_nonlinear_transport_audit",
        "case": str(case),
        "passed": passed,
        "blockers": blockers,
        "gates": gates,
        "baseline_ensemble": baseline,
        "optimized_ensemble": optimized,
        "selected_optimized_audit": selected_audit,
        "comparison": comparison,
        "config": {
            "min_relative_reduction": float(min_relative_reduction),
            "require_uncertainty_separation": bool(require_uncertainty_separation),
        },
    }


def _write_json(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "role",
        "path",
        "case",
        "passed",
        "qualifies",
        "ensemble_mean",
        "combined_sem",
        "combined_sem_rel",
        "mean_spread",
        "mean_rel_spread",
        "n_reports",
        "blockers",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in ("baseline_ensemble", "optimized_ensemble"):
            row = report[key]
            writer.writerow(
                {
                    field: (
                        ";".join(str(item) for item in row.get(field, []))
                        if field == "blockers"
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def _write_plot(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    rows = [report["baseline_ensemble"], report["optimized_ensemble"]]
    labels = ["no-ESS\nreference", "optimized\nQA/ESS"]
    means = [_finite_float(row.get("ensemble_mean")) or 0.0 for row in rows]
    sems = [_finite_float(row.get("combined_sem")) or 0.0 for row in rows]
    colors = ["#2563eb", "#0f766e"] if bool(report["passed"]) else ["#6b7280", "#b91c1c"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.bar(labels, means, yerr=sems, capsize=5, color=colors, alpha=0.88, edgecolor="0.2")
    ax.set_ylabel(r"post-transient $\langle Q_i\rangle/Q_{gB}$")
    reduction = report["comparison"].get("relative_reduction")
    sigma = report["comparison"].get("uncertainty_separation_sigma")
    status = "passed" if report["passed"] else "failed closed"
    ax.set_title(f"Baseline-to-optimized nonlinear audit: {status}")
    ax.grid(axis="y", alpha=0.25)
    caption = (
        f"relative reduction = {reduction:.3f}; separation = {sigma:.2f} sigma"
        if reduction is not None and sigma is not None
        else "comparison unavailable"
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.23)
    fig.text(0.5, 0.055, caption, ha="center", fontsize=9.5, color="0.25")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_audit_artifacts(
    report: Mapping[str, Any],
    *,
    out_json: Path,
    out_csv: Path | None = None,
    out_png: Path | None = None,
) -> None:
    """Write the JSON/CSV/PNG sidecars for a matched nonlinear audit."""

    _write_json(report, out_json)
    _write_csv(report, out_csv or out_json.with_suffix(".csv"))
    if out_png is not None:
        _write_plot(report, out_png)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ensemble", type=Path, default=DEFAULT_BASELINE_ENSEMBLE)
    parser.add_argument("--optimized-ensemble", type=Path, default=DEFAULT_OPTIMIZED_ENSEMBLE)
    parser.add_argument("--selected-optimized-audit", type=Path, default=DEFAULT_SELECTED_OPTIMIZED_AUDIT)
    parser.add_argument(
        "--no-selected-optimized-audit",
        action="store_true",
        help="Do not require a separate selected optimized-equilibrium audit artifact.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-png", type=Path)
    parser.add_argument("--case", default="baseline_to_optimized_nonlinear_transport_audit")
    parser.add_argument("--min-relative-reduction", type=float, default=0.0)
    parser.add_argument(
        "--allow-uncertainty-overlap",
        action="store_true",
        help="Do not require the baseline/optimized mean difference to exceed one combined SEM.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_path = None if args.no_selected_optimized_audit else args.selected_optimized_audit
    report = build_audit(
        baseline_path=args.baseline_ensemble,
        optimized_path=args.optimized_ensemble,
        selected_optimized_audit_path=selected_path,
        case=str(args.case),
        min_relative_reduction=float(args.min_relative_reduction),
        require_uncertainty_separation=not bool(args.allow_uncertainty_overlap),
    )
    write_audit_artifacts(
        report,
        out_json=args.out_json,
        out_csv=args.out_csv,
        out_png=args.out_png,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "blockers": report["blockers"],
                "comparison": report["comparison"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
