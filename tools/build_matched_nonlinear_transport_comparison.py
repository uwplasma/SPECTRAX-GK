#!/usr/bin/env python3
"""Compare matched replicated nonlinear transport ensembles.

The production nonlinear-optimization claim needs a paired baseline-vs-candidate
comparison after each state has independently passed a replicated late-window
transport gate. This tool consumes those two ensemble JSON files and writes a
small machine-readable comparison plus an optional publication-style figure.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite, sqrt
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.plotting import set_plot_style  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _ensemble_stats(payload: dict[str, Any], *, label: str) -> dict[str, float | bool]:
    stats = payload.get("statistics")
    if not isinstance(stats, dict):
        raise ValueError(f"{label} ensemble is missing statistics")
    mean = float(stats.get("ensemble_mean", float("nan")))
    sem = float(stats.get("combined_sem", stats.get("sample_sem", float("nan"))))
    spread = float(stats.get("mean_rel_spread", float("nan")))
    sem_rel = float(stats.get("combined_sem_rel", float("nan")))
    if not isfinite(mean):
        raise ValueError(f"{label} ensemble_mean is not finite")
    if not isfinite(sem):
        sem = 0.0
    return {
        "passed": bool(payload.get("passed", False)),
        "ensemble_mean": mean,
        "combined_sem": max(0.0, sem),
        "mean_relative_spread": spread,
        "combined_sem_relative": sem_rel,
    }


def build_comparison(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_artifact: Path,
    candidate_artifact: Path,
    case: str,
    min_relative_reduction: float,
) -> dict[str, Any]:
    """Return a matched nonlinear transport comparison report."""

    base = _ensemble_stats(baseline, label="baseline")
    cand = _ensemble_stats(candidate, label="candidate")
    baseline_mean = float(base["ensemble_mean"])
    candidate_mean = float(cand["ensemble_mean"])
    delta = baseline_mean - candidate_mean
    scale = max(abs(baseline_mean), 1.0e-30)
    relative_reduction = delta / scale
    combined_uncertainty = sqrt(float(base["combined_sem"]) ** 2 + float(cand["combined_sem"]) ** 2)
    z_score = delta / combined_uncertainty if combined_uncertainty > 0.0 else float("inf")
    passed = (
        bool(base["passed"])
        and bool(cand["passed"])
        and isfinite(relative_reduction)
        and relative_reduction >= float(min_relative_reduction)
    )
    return {
        "kind": "matched_nonlinear_transport_comparison",
        "case": str(case),
        "passed": bool(passed),
        "claim_level": "matched_replicated_late_window_transport_comparison",
        "artifacts": {
            "baseline": _repo_relative(baseline_artifact),
            "candidate": _repo_relative(candidate_artifact),
        },
        "config": {
            "min_relative_reduction": float(min_relative_reduction),
        },
        "baseline": base,
        "candidate": cand,
        "statistics": {
            "absolute_reduction": delta,
            "relative_reduction": relative_reduction,
            "combined_uncertainty": combined_uncertainty,
            "uncertainty_z_score": z_score,
        },
        "gates": [
            {
                "metric": "baseline_ensemble_passed",
                "passed": bool(base["passed"]),
                "detail": f"baseline passed={bool(base['passed'])}",
            },
            {
                "metric": "candidate_ensemble_passed",
                "passed": bool(cand["passed"]),
                "detail": f"candidate passed={bool(cand['passed'])}",
            },
            {
                "metric": "relative_transport_reduction",
                "passed": bool(relative_reduction >= float(min_relative_reduction)),
                "detail": (
                    f"relative reduction {relative_reduction:.6g} >= "
                    f"{float(min_relative_reduction):.6g}"
                ),
            },
        ],
    }


def _write_plot(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    base = report["baseline"]
    cand = report["candidate"]
    stats = report["statistics"]
    means = [float(base["ensemble_mean"]), float(cand["ensemble_mean"])]
    sems = [float(base["combined_sem"]), float(cand["combined_sem"])]
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    colors = ["#334155", "#0f766e"]
    ax.bar([0, 1], means, yerr=sems, capsize=5, color=colors, alpha=0.9, edgecolor="0.15", linewidth=0.6)
    ax.set_xticks([0, 1], ["baseline", "transport\ncandidate"])
    ax.set_ylabel(r"late-window $\langle Q_i\rangle/Q_{gB}$")
    rel = float(stats["relative_reduction"])
    z_score = float(stats["uncertainty_z_score"])
    status = "passes" if bool(report["passed"]) else "not promoted"
    ax.set_title(f"Matched nonlinear transport comparison: {status}")
    ax.text(
        0.5,
        0.96,
        f"relative reduction = {100.0 * rel:.2f}%\nuncertainty z-score = {z_score:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ensemble", type=Path, required=True)
    parser.add_argument("--candidate-ensemble", type=Path, required=True)
    parser.add_argument("--case", default="matched_nonlinear_transport")
    parser.add_argument("--min-relative-reduction", type=float, default=0.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-figure", type=Path)
    parser.add_argument("--fail-on-unpromoted", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_comparison(
        baseline=_load_json(args.baseline_ensemble),
        candidate=_load_json(args.candidate_ensemble),
        baseline_artifact=args.baseline_ensemble,
        candidate_artifact=args.candidate_ensemble,
        case=str(args.case),
        min_relative_reduction=float(args.min_relative_reduction),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_figure is not None:
        _write_plot(report, args.out_figure)
    print(json.dumps({"passed": report["passed"], "statistics": report["statistics"]}, indent=2, sort_keys=True))
    if args.fail_on_unpromoted and not bool(report["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
