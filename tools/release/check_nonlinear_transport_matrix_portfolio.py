#!/usr/bin/env python3
"""Select a promotable broad nonlinear transport matrix from candidate families.

Each input matrix report must come from
``tools/build_matched_nonlinear_transport_matrix.py report``. This portfolio
gate is intentionally separate from single-point matched audits: a nonlinear
turbulent-flux optimization family is broadly promotable only if at least one
candidate report passes the multi-surface, multi-field-line, multi-ky matrix
policy. Strict growth/QL/nonlinear-window transfer rows can be supplied as
excluded comparisons; they are recorded as negative evidence but never counted
toward promotion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return raw.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"expected LABEL=PATH, got {raw!r}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty label in {raw!r}")
    return label, Path(path)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _matrix_row(
    *,
    label: str,
    path: Path,
    min_total_samples: int,
    min_surfaces: int,
    min_alphas: int,
    min_ky_values: int,
    min_pass_fraction: float,
    min_mean_relative_reduction: float,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "label": label,
            "path": _repo_relative(path),
            "exists": False,
            "passed": False,
            "qualifies_for_broad_promotion": False,
            "blockers": ["missing matrix report"],
        }
    payload = _load_json(path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    total_samples = int(_finite_float(summary.get("total_samples")) or 0)
    completed_samples = int(_finite_float(summary.get("completed_samples")) or 0)
    passed_samples = int(_finite_float(summary.get("passed_samples")) or 0)
    pass_fraction = _finite_float(summary.get("pass_fraction"))
    mean_reduction = _finite_float(summary.get("mean_relative_reduction"))
    surfaces = summary.get("surfaces") if isinstance(summary.get("surfaces"), list) else []
    alphas = summary.get("alphas") if isinstance(summary.get("alphas"), list) else []
    ky_values = summary.get("ky_values") if isinstance(summary.get("ky_values"), list) else []
    blockers: list[str] = []
    if payload.get("kind") != "matched_nonlinear_transport_matrix_report":
        blockers.append("not a matched nonlinear transport matrix report")
    if not bool(payload.get("passed", False)):
        blockers.append("matrix report failed its internal gate")
    if total_samples < int(min_total_samples):
        blockers.append(f"total_samples {total_samples} < {int(min_total_samples)}")
    if completed_samples < total_samples:
        blockers.append(f"completed_samples {completed_samples} < total_samples {total_samples}")
    if len(set(surfaces)) < int(min_surfaces):
        blockers.append(f"surfaces {len(set(surfaces))} < {int(min_surfaces)}")
    if len(set(alphas)) < int(min_alphas):
        blockers.append(f"field_line_labels {len(set(alphas))} < {int(min_alphas)}")
    if len(set(ky_values)) < int(min_ky_values):
        blockers.append(f"ky_values {len(set(ky_values))} < {int(min_ky_values)}")
    if pass_fraction is None or pass_fraction < float(min_pass_fraction):
        blockers.append(
            "pass_fraction "
            f"{'n/a' if pass_fraction is None else f'{pass_fraction:.6g}'} "
            f"< {float(min_pass_fraction):.6g}"
        )
    if mean_reduction is None or mean_reduction < float(min_mean_relative_reduction):
        blockers.append(
            "mean_relative_reduction "
            f"{'n/a' if mean_reduction is None else f'{mean_reduction:.6g}'} "
            f"< {float(min_mean_relative_reduction):.6g}"
        )
    return {
        "label": label,
        "path": _repo_relative(path),
        "exists": True,
        "passed": bool(payload.get("passed", False)),
        "qualifies_for_broad_promotion": not blockers,
        "blockers": blockers,
        "summary": {
            "total_samples": total_samples,
            "completed_samples": completed_samples,
            "passed_samples": passed_samples,
            "pass_fraction": pass_fraction,
            "mean_relative_reduction": mean_reduction,
            "surfaces": surfaces,
            "alphas": alphas,
            "ky_values": ky_values,
        },
    }


def _excluded_row(label: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"label": label, "path": _repo_relative(path), "exists": False}
    payload = _load_json(path)
    stats = payload.get("statistics") if isinstance(payload.get("statistics"), Mapping) else {}
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), Mapping) else {}
    return {
        "label": label,
        "path": _repo_relative(path),
        "exists": True,
        "passed": bool(payload.get("passed", False)),
        "relative_reduction": _finite_float(
            comparison.get("relative_reduction", stats.get("relative_reduction"))
        ),
        "uncertainty_z_score": _finite_float(
            comparison.get("uncertainty_z_score", stats.get("uncertainty_z_score"))
        ),
        "note": "recorded as negative/diagnostic evidence; excluded from broad matrix promotion",
    }


def build_report(
    *,
    matrix_reports: Mapping[str, Path],
    excluded_comparisons: Mapping[str, Path],
    min_total_samples: int = 18,
    min_surfaces: int = 3,
    min_alphas: int = 2,
    min_ky_values: int = 3,
    min_pass_fraction: float = 1.0,
    min_mean_relative_reduction: float = 0.02,
) -> dict[str, Any]:
    rows = [
        _matrix_row(
            label=label,
            path=path,
            min_total_samples=min_total_samples,
            min_surfaces=min_surfaces,
            min_alphas=min_alphas,
            min_ky_values=min_ky_values,
            min_pass_fraction=min_pass_fraction,
            min_mean_relative_reduction=min_mean_relative_reduction,
        )
        for label, path in sorted(matrix_reports.items())
    ]
    qualified = [row for row in rows if bool(row["qualifies_for_broad_promotion"])]
    selected = max(
        qualified,
        key=lambda row: float(row["summary"].get("mean_relative_reduction") or 0.0),
        default=None,
    )
    blockers: list[str] = []
    if not rows:
        blockers.append("no matrix reports supplied")
    if selected is None:
        blockers.append("no candidate family passed the broad matrix gate")
    return {
        "kind": "nonlinear_transport_matrix_portfolio_gate",
        "claim_level": "broad_nonlinear_turbulent_flux_optimization_family_selection",
        "passed": selected is not None,
        "selected_family": None if selected is None else selected["label"],
        "selected_report": selected,
        "config": {
            "min_total_samples": int(min_total_samples),
            "min_surfaces": int(min_surfaces),
            "min_field_line_labels": int(min_alphas),
            "min_ky_values": int(min_ky_values),
            "min_pass_fraction": float(min_pass_fraction),
            "min_mean_relative_reduction": float(min_mean_relative_reduction),
        },
        "matrix_reports": rows,
        "excluded_comparisons": [
            _excluded_row(label, path) for label, path in sorted(excluded_comparisons.items())
        ],
        "blockers": blockers,
        "notes": [
            "A single-point matched audit is not enough for this broad promotion gate.",
            "Strict growth/QL/nonlinear-window transfer rows are listed only as excluded evidence.",
        ],
    }


def _write_figure(report: Mapping[str, Any], path: Path) -> None:
    rows = [row for row in report.get("matrix_reports", []) if isinstance(row, Mapping)]
    path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    labels = [str(row.get("label")) for row in rows]
    reductions = [
        100.0 * float((row.get("summary") or {}).get("mean_relative_reduction") or 0.0)
        for row in rows
    ]
    colors = [
        "#0f766e" if bool(row.get("qualifies_for_broad_promotion")) else "#b45309"
        for row in rows
    ]
    if rows:
        ax.bar(range(len(rows)), reductions, color=colors, edgecolor="0.2", linewidth=0.6)
        ax.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
        ax.set_ylabel("mean heat-flux reduction across matrix (%)")
        ax.axhline(0.0, color="0.2", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No matrix reports supplied", ha="center", va="center")
        ax.set_xticks([])
    status = "passes" if bool(report.get("passed", False)) else "blocked"
    selected = report.get("selected_family") or "none"
    ax.set_title(f"Nonlinear transport matrix portfolio: {status}; selected={selected}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Candidate-family matrix report JSON. May be supplied multiple times.",
    )
    parser.add_argument(
        "--excluded-comparison",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Strict negative-transfer comparison JSON to record but not promote.",
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-figure", type=Path)
    parser.add_argument("--min-total-samples", type=int, default=18)
    parser.add_argument("--min-surfaces", type=int, default=3)
    parser.add_argument("--min-field-line-labels", type=int, default=2)
    parser.add_argument("--min-ky-values", type=int, default=3)
    parser.add_argument("--min-pass-fraction", type=float, default=1.0)
    parser.add_argument("--min-mean-relative-reduction", type=float, default=0.02)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix_reports = dict(_parse_labeled_path(raw) for raw in args.matrix_report)
    excluded = dict(_parse_labeled_path(raw) for raw in args.excluded_comparison)
    report = build_report(
        matrix_reports=matrix_reports,
        excluded_comparisons=excluded,
        min_total_samples=int(args.min_total_samples),
        min_surfaces=int(args.min_surfaces),
        min_alphas=int(args.min_field_line_labels),
        min_ky_values=int(args.min_ky_values),
        min_pass_fraction=float(args.min_pass_fraction),
        min_mean_relative_reduction=float(args.min_mean_relative_reduction),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_figure is not None:
        _write_figure(report, args.out_figure)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "selected_family": report["selected_family"],
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_blocked and not bool(report["passed"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
