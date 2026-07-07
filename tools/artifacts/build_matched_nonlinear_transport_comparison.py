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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


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


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def _ensemble_stats(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    stats = payload.get("statistics")
    if not isinstance(stats, dict):
        raise ValueError(f"{label} ensemble is missing statistics")
    mean = _optional_float(stats.get("ensemble_mean"))
    sem = _optional_float(stats.get("combined_sem", stats.get("sample_sem")))
    spread = _optional_float(stats.get("mean_rel_spread"))
    sem_rel = _optional_float(stats.get("combined_sem_rel"))
    finite_mean = mean is not None
    return {
        "passed": bool(payload.get("passed", False)) and finite_mean,
        "raw_passed": bool(payload.get("passed", False)),
        "finite_mean": bool(finite_mean),
        "ensemble_mean": mean,
        "combined_sem": max(0.0, sem) if sem is not None else None,
        "mean_relative_spread": spread,
        "combined_sem_relative": sem_rel,
        "failure": None if finite_mean else f"{label} ensemble_mean is not finite",
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
    if bool(base["finite_mean"]) and bool(cand["finite_mean"]):
        baseline_mean = float(base["ensemble_mean"])
        candidate_mean = float(cand["ensemble_mean"])
        delta = baseline_mean - candidate_mean
        scale = max(abs(baseline_mean), 1.0e-30)
        relative_reduction = delta / scale
        base_sem = float(base["combined_sem"] or 0.0)
        cand_sem = float(cand["combined_sem"] or 0.0)
        combined_uncertainty = sqrt(base_sem**2 + cand_sem**2)
        z_score = (
            delta / combined_uncertainty if combined_uncertainty > 0.0 else float("inf")
        )
    else:
        delta = None
        relative_reduction = None
        combined_uncertainty = None
        z_score = None
    passed = (
        bool(base["passed"])
        and bool(cand["passed"])
        and relative_reduction is not None
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
                "detail": f"baseline passed={bool(base['passed'])}; {base.get('failure') or 'finite mean available'}",
            },
            {
                "metric": "candidate_ensemble_passed",
                "passed": bool(cand["passed"]),
                "detail": f"candidate passed={bool(cand['passed'])}; {cand.get('failure') or 'finite mean available'}",
            },
            {
                "metric": "relative_transport_reduction",
                "passed": bool(
                    relative_reduction is not None
                    and relative_reduction >= float(min_relative_reduction)
                ),
                "detail": (
                    f"relative reduction {_format_optional(relative_reduction)} >= "
                    f"{float(min_relative_reduction):.6g}"
                ),
            },
        ],
    }


def _format_optional(value: Any) -> str:
    out = _optional_float(value)
    return "n/a" if out is None else f"{out:.6g}"


def _write_plot(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    base = report["baseline"]
    cand = report["candidate"]
    stats = report["statistics"]
    means = [
        float(base["ensemble_mean"])
        if base["ensemble_mean"] is not None
        else float("nan"),
        float(cand["ensemble_mean"])
        if cand["ensemble_mean"] is not None
        else float("nan"),
    ]
    sems = [
        float(base["combined_sem"]) if base["combined_sem"] is not None else 0.0,
        float(cand["combined_sem"]) if cand["combined_sem"] is not None else 0.0,
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = ["#334155", "#0f766e"]
    finite_means = [isfinite(mean) for mean in means]
    if any(finite_means):
        ax.bar(
            [0, 1],
            means,
            yerr=sems,
            capsize=5,
            color=colors,
            alpha=0.9,
            edgecolor="0.15",
            linewidth=0.6,
        )
    else:
        ax.text(
            0.5,
            0.48,
            "No matched transport reduction can be computed:\n"
            "baseline/candidate ensembles have no finite\n"
            "accepted-window means.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#7f1d1d",
            bbox={"boxstyle": "round,pad=0.45", "fc": "#fff7ed", "ec": "#fed7aa"},
        )
    ax.set_xticks([0, 1], ["baseline", "transport\ncandidate"])
    ax.set_ylabel(r"late-window $\langle Q_i\rangle/Q_{gB}$")
    rel = _optional_float(stats.get("relative_reduction"))
    z_score = _optional_float(stats.get("uncertainty_z_score"))
    status = "passes" if bool(report["passed"]) else "not promoted"
    ax.set_title(f"Matched nonlinear transport comparison: {status}", pad=12)
    finite_y = [mean + sem for mean, sem in zip(means, sems) if isfinite(mean)]
    ymax = max(finite_y) if finite_y else 1.0
    ax.set_ylim(0.0, ymax * 1.30)
    ax.text(
        0.50,
        0.88,
        "relative reduction = {rel}\nuncertainty z-score = {z}".format(
            rel="n/a" if rel is None else f"{100.0 * rel:.2f}%",
            z="n/a" if z_score is None else f"{z_score:.2f}",
        ),
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
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.out_figure is not None:
        _write_plot(report, args.out_figure)
    print(
        json.dumps(
            {"passed": report["passed"], "statistics": report["statistics"]},
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_unpromoted and not bool(report["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
