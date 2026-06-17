#!/usr/bin/env python3
"""Check replicated nonlinear-window summaries for robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.quasilinear.window import (  # noqa: E402
    NonlinearWindowEnsembleConfig,
    nonlinear_window_ensemble_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Nonlinear-window convergence JSON reports.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path)
    parser.add_argument("--case", default="nonlinear_window_ensemble")
    parser.add_argument("--comparison", default="replicate_uncertainty")
    parser.add_argument("--min-reports", type=int, default=2)
    parser.add_argument("--max-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--value-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--allow-failed-individual",
        action="store_true",
        help="Do not require each input window to pass the promotion-ready gate.",
    )
    return parser


def _load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} did not contain a JSON object")
        reports.append(payload)
    return reports


def _write_png(report: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    rows = list(report["rows"])
    labels = [str(row["case"]) for row in rows]
    means = np.asarray([np.nan if row["late_mean"] is None else float(row["late_mean"]) for row in rows])
    sem = np.asarray([0.0 if row["sem"] is None else float(row["sem"]) for row in rows])
    ensemble_mean = report["statistics"]["ensemble_mean"]
    passed = bool(report["passed"])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    x = np.arange(len(rows))
    colors = ["#0f766e" if bool(row["promotion_ready"]) else "#b91c1c" for row in rows]
    ax.bar(x, means, yerr=sem, capsize=4, color=colors, alpha=0.88)
    if ensemble_mean is not None:
        ax.axhline(float(ensemble_mean), color="0.2", lw=1.3, ls="--", label="ensemble mean")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("late-window mean")
    status = "passed" if passed else "failed closed"
    ax.set_title(f"{report['comparison']} ensemble gate {status}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = _load_reports(list(args.reports))
    summary = nonlinear_window_ensemble_report(
        reports,
        case=args.case,
        comparison=args.comparison,
        config=NonlinearWindowEnsembleConfig(
            min_reports=args.min_reports,
            max_mean_rel_spread=args.max_mean_rel_spread,
            max_combined_sem_rel=args.max_combined_sem_rel,
            value_floor=args.value_floor,
            require_individual_passed=not args.allow_failed_individual,
        ),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_png is not None:
        _write_png(summary, args.out_png)
    print(json.dumps(summary["gate_report"], indent=2, sort_keys=True))
    return 0 if bool(summary["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
