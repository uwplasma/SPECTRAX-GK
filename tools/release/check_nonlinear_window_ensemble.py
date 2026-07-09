#!/usr/bin/env python3
"""Check nonlinear-window convergence, readiness, and replicated ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.transport_windows import (  # noqa: E402
    NonlinearWindowConvergenceConfig,
    NonlinearWindowEnsembleConfig,
    NonlinearWindowEnsembleManifestConfig,
    nonlinear_window_convergence_from_csv,
    nonlinear_window_convergence_from_summary,
    nonlinear_window_ensemble_artifact_manifest,
    nonlinear_window_ensemble_report,
)


VARIANT_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "seed": ("seed", "random_seed", "rng_seed", "simulation_seed"),
    "timestep": ("timestep", "time_step", "dt"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check replicated nonlinear-window summaries for robustness.")
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Nonlinear-window convergence JSON reports.",
    )
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


def build_convergence_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check nonlinear late-window convergence metadata without rerunning a solve."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Diagnostics CSV with time trace.")
    source.add_argument(
        "--summary",
        type=Path,
        help="Window summary JSON containing a diagnostics source path.",
    )
    parser.add_argument("--diagnostics-source", default="spectrax")
    parser.add_argument("--time-column", default="t")
    parser.add_argument("--value-column", default="heat_flux")
    parser.add_argument("--case", default=None)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--max-running-mean-rel-drift", type=float, default=0.15)
    parser.add_argument("--terminal-fraction", type=float, default=0.25)
    parser.add_argument("--min-terminal-samples", type=int, default=8)
    parser.add_argument("--max-terminal-mean-rel-delta", type=float, default=0.10)
    parser.add_argument("--max-sem-rel", type=float, default=0.25)
    parser.add_argument("--value-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--allow-nonfinite",
        action="store_true",
        help="Ignore non-finite samples inside the late window instead of failing.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def build_readiness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert transport-window summaries into ensemble-readiness metadata."
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="Transport-window summary JSON files with trace CSV provenance.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Optional directory for per-summary convergence report JSON files.",
    )
    parser.add_argument("--case", default="nonlinear_window_ensemble_readiness")
    parser.add_argument("--time-column", default="t")
    parser.add_argument("--value-column", default="heat_flux")
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--max-running-mean-rel-drift", type=float, default=0.15)
    parser.add_argument("--max-terminal-mean-rel-delta", type=float, default=0.10)
    parser.add_argument("--max-sem-rel", type=float, default=0.25)
    parser.add_argument("--min-replicates-per-case", type=int, default=2)
    parser.add_argument(
        "--variant-axis",
        action="append",
        choices=tuple(VARIANT_KEY_ALIASES),
        help="Required replicated variant axis. Defaults to seed and timestep.",
    )
    parser.add_argument(
        "--allow-failed-observed-window",
        action="store_true",
        help="Record failed convergence reports but do not block on them.",
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


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _repo_relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    raw_path = Path(path)
    try:
        return raw_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _safe_report_name(summary_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", summary_path.stem)
    return f"{stem}.convergence.json"


def _nested_search(payload: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    queue: list[dict[str, Any]] = [payload]
    seen = 0
    while queue and seen < 64:
        current = queue.pop(0)
        seen += 1
        for key in aliases:
            if key in current and current[key] not in (None, ""):
                return current[key]
        for key in (
            "variant",
            "metadata",
            "run",
            "simulation",
            "config",
            "nonlinear_config",
        ):
            nested = current.get(key)
            if isinstance(nested, dict):
                queue.append(nested)
    return None


def _variant_from_summary(
    summary: dict[str, Any], summary_path: Path
) -> dict[str, Any]:
    text = summary_path.stem.lower()
    variant: dict[str, Any] = {}
    for axis, aliases in VARIANT_KEY_ALIASES.items():
        value = _nested_search(summary, aliases)
        if value is None:
            regexes = (
                [r"(?:seed|rng)[_-]?([0-9]+)"]
                if axis == "seed"
                else [r"(?:dt|timestep)[_-]?([0-9]+(?:p[0-9]+)?)"]
            )
            for pattern in regexes:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).replace("p", ".")
                    break
        variant[axis] = value
    return variant


def _summary_case(summary: dict[str, Any], path: Path) -> str:
    return str(summary.get("case") or summary.get("case_name") or path.stem)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_png(report: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    rows = list(report["rows"])
    labels = [str(row["case"]) for row in rows]
    means = np.asarray(
        [
            np.nan if row["late_mean"] is None else float(row["late_mean"])
            for row in rows
        ]
    )
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
        ax.axhline(
            float(ensemble_mean), color="0.2", lw=1.3, ls="--", label="ensemble mean"
        )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("late-window mean")
    status = "passed" if passed else "failed closed"
    ax.set_title(f"{report['comparison']} ensemble gate {status}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _convergence_config(args: argparse.Namespace) -> NonlinearWindowConvergenceConfig:
    return NonlinearWindowConvergenceConfig(
        tmin=args.tmin,
        tmax=args.tmax,
        transient_fraction=args.transient_fraction,
        min_samples=args.min_samples,
        min_blocks=args.min_blocks,
        block_size=args.block_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        max_running_mean_rel_drift=args.max_running_mean_rel_drift,
        terminal_fraction=args.terminal_fraction,
        min_terminal_samples=args.min_terminal_samples,
        max_terminal_mean_rel_delta=args.max_terminal_mean_rel_delta,
        max_sem_rel=args.max_sem_rel,
        value_floor=args.value_floor,
        require_all_finite=not args.allow_nonfinite,
    )


def main_convergence(argv: list[str] | None = None) -> int:
    args = build_convergence_parser().parse_args(argv)
    cfg = _convergence_config(args)
    if args.csv is not None:
        report = nonlinear_window_convergence_from_csv(
            args.csv,
            time_column=args.time_column,
            value_column=args.value_column,
            case=args.case,
            config=cfg,
        )
    else:
        report = nonlinear_window_convergence_from_summary(
            args.summary,
            diagnostics_source=args.diagnostics_source,
            time_column=args.time_column,
            value_column=args.value_column,
            case=args.case,
            config=cfg,
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stats = report["statistics"]
    print(f"wrote {args.out_json}")
    print(
        "nonlinear_window_converged={passed} late_mean={mean} sem={sem}".format(
            passed=report["passed"],
            mean=stats["late_mean"],
            sem=stats["sem"],
        )
    )
    return 0 if report["passed"] else 2


def main_readiness(argv: list[str] | None = None) -> int:
    args = build_readiness_parser().parse_args(argv)
    axes = tuple(args.variant_axis or ("seed", "timestep"))
    records: list[dict[str, Any]] = []
    for summary_path in args.summaries:
        summary = _load_json_object(summary_path)
        convergence_config = NonlinearWindowConvergenceConfig(
            tmin=summary.get("tmin"),
            tmax=summary.get("tmax"),
            min_samples=args.min_samples,
            min_blocks=args.min_blocks,
            bootstrap_samples=args.bootstrap_samples,
            max_running_mean_rel_drift=args.max_running_mean_rel_drift,
            max_terminal_mean_rel_delta=args.max_terminal_mean_rel_delta,
            max_sem_rel=args.max_sem_rel,
        )
        report = nonlinear_window_convergence_from_summary(
            summary_path,
            time_column=args.time_column,
            value_column=args.value_column,
            case=_summary_case(summary, summary_path),
            config=convergence_config,
        )
        report_path: Path | None = None
        if args.reports_dir is not None:
            report_path = args.reports_dir / _safe_report_name(summary_path)
            _write_json(report_path, report)
        records.append(
            {
                "case": _summary_case(summary, summary_path),
                "summary_artifact": _repo_relative(summary_path),
                "source_artifact": _repo_relative(
                    report["provenance"]["source_artifact"]
                ),
                "convergence_report_artifact": _repo_relative(report_path),
                "variant": _variant_from_summary(summary, summary_path),
                "report": report,
            }
        )

    manifest = nonlinear_window_ensemble_artifact_manifest(
        records,
        case=args.case,
        config=NonlinearWindowEnsembleManifestConfig(
            min_replicates_per_case=args.min_replicates_per_case,
            required_variant_axes=axes,
            require_observed_windows_ready=not args.allow_failed_observed_window,
        ),
    )
    _write_json(args.out_json, manifest)
    print(json.dumps(manifest["promotion_gate"], indent=2, sort_keys=True))
    return 0 if bool(manifest["passed"]) else 1


def main_ensemble(argv: list[str] | None = None) -> int:
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
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.out_png is not None:
        _write_png(summary, args.out_png)
    print(json.dumps(summary["gate_report"], indent=2, sort_keys=True))
    return 0 if bool(summary["passed"]) else 1


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "convergence":
        return main_convergence(tokens[1:])
    if tokens and tokens[0] == "readiness":
        return main_readiness(tokens[1:])
    return main_ensemble(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
