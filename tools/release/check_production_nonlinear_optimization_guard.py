#!/usr/bin/env python3
"""Check production nonlinear turbulent-flux optimization promotion guardrails."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.nonlinear_transport.optimization_guard import (  # noqa: E402
    ProductionNonlinearOptimizationGuardConfig,
    production_nonlinear_optimization_guard_report,
)
from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


DEFAULT_OPTIMIZATION_ARTIFACT = (
    ROOT / "docs/_static/stellarator_itg_optimization_comparison.json"
)
DEFAULT_REDUCED_ARTIFACTS = (
    ROOT / "docs/_static/nonlinear_window_fd_audit.json",
    ROOT / "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json",
)
DEFAULT_REPLICATED_ENSEMBLES = (
    ROOT
    / "docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json",
    ROOT
    / "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json",
    ROOT
    / "docs/_static/vmec_boozer_holdout_transport/vmec_boozer_qh_torflux078_alpha120_holdout_ensemble_gate.json",
)
DEFAULT_OPTIMIZED_EQUILIBRIUM_ENSEMBLES = (
    ROOT
    / "docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.json",
    ROOT
    / "docs/_static/vmec_qa_t1500_replicates/growth_from_strict_baseline_t1500_ensemble_gate.json",
    ROOT
    / "docs/_static/vmec_qa_t1500_replicates/quasilinear_from_strict_baseline_t1500_ensemble_gate.json",
    ROOT
    / "docs/_static/vmec_qa_t1500_replicates/nonlinear_window_from_strict_baseline_t1500_ensemble_gate.json",
)
DEFAULT_MATCHED_OPTIMIZED_AUDITS = (
    ROOT / "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json",
    ROOT / "docs/_static/vmec_jax_qa_projected_weight_0p0005_matched_comparison.json",
    ROOT / "docs/_static/vmec_jax_qa_projected_weight_0p001_matched_comparison.json",
    ROOT / "docs/_static/vmec_qa_t1500_baseline_to_growth_comparison.json",
    ROOT / "docs/_static/vmec_qa_t1500_baseline_to_quasilinear_comparison.json",
    ROOT / "docs/_static/vmec_qa_t1500_baseline_to_nonlinear_window_comparison.json",
)
DEFAULT_OUT_JSON = ROOT / "docs/_static/production_nonlinear_optimization_guard.json"
DEFAULT_OUT_PNG = ROOT / "docs/_static/production_nonlinear_optimization_guard.png"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_mapping(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {_repo_relative(path): _load_json(path) for path in paths}


def _write_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for gate in report["gates"]:
        rows.append(
            {
                "group": "safety"
                if gate["metric"] in report["safety_gate"]["requirements"]
                else "",
                "metric": gate["metric"],
                "passed": gate["passed"],
                "detail": gate["detail"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["metric", "passed", "detail"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ("metric", "passed", "detail")})


def _write_plot(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    gates = list(report["gates"])
    labels = [str(gate["metric"]).replace("_", "\n") for gate in gates]
    values = [1 if bool(gate["passed"]) else 0 for gate in gates]
    colors = ["#0f766e" if value else "#b91c1c" for value in values]
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    x = range(len(gates))
    axs[0].bar(x, values, color=colors, alpha=0.9)
    axs[0].set_ylim(0.0, 1.15)
    axs[0].set_yticks([0, 1], ["blocked", "passed"])
    axs[0].set_xticks(list(x), labels, rotation=0, ha="center", fontsize=8)
    axs[0].set_title("Release safety and production-promotion gates")
    axs[0].grid(axis="y", alpha=0.25)

    summaries = report["summary"]
    counts = [
        summaries["qualifying_replicated_holdout_ensembles"],
        summaries["qualifying_optimized_equilibrium_ensembles"],
        summaries["qualifying_matched_optimized_transport_audits"],
    ]
    axs[1].bar(
        [0, 1, 2],
        counts,
        color=["#2563eb", "#f97316", "#7c3aed"],
        alpha=0.88,
    )
    axs[1].axhline(
        report["config"]["min_replicated_ensembles"], color="#2563eb", ls=":", lw=1.5
    )
    axs[1].axhline(
        report["config"]["min_optimized_equilibrium_ensembles"],
        color="#f97316",
        ls="--",
        lw=1.2,
    )
    axs[1].axhline(
        report["config"]["min_matched_optimized_audits"],
        color="#7c3aed",
        ls="-.",
        lw=1.2,
    )
    axs[1].set_xticks(
        [0, 1, 2],
        [
            "long-window\nholdouts",
            "optimized-equilibrium\ntransport",
            "matched\noptimization audits",
        ],
        fontsize=9,
    )
    axs[1].set_ylabel("qualifying artifacts")
    axs[1].set_title("Evidence available for nonlinear optimization")
    axs[1].grid(axis="y", alpha=0.25)
    axs[1].text(
        0.03,
        0.95,
        (
            f"matched audits total: {summaries.get('total_matched_optimized_transport_audits', 0)}\n"
            f"strict negatives: {summaries.get('failed_matched_optimized_transport_audits', 0)}"
        ),
        transform=axs[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.9},
    )

    status = (
        "production promoted"
        if report["production_nonlinear_optimization_promoted"]
        else "production blocked; release-safe scoped evidence"
    )
    fig.suptitle(
        f"Production nonlinear turbulent-flux optimization guard: {status}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimization-artifact", type=Path, default=DEFAULT_OPTIMIZATION_ARTIFACT
    )
    parser.add_argument("--reduced-artifact", action="append", type=Path, default=[])
    parser.add_argument("--replicated-ensemble", action="append", type=Path, default=[])
    parser.add_argument(
        "--optimized-equilibrium-ensemble", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--matched-optimized-audit", action="append", type=Path, default=[]
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out-pdf", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--min-replicated-ensembles", type=int, default=2)
    parser.add_argument("--min-reports-per-ensemble", type=int, default=2)
    parser.add_argument("--max-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--min-optimized-equilibrium-ensembles", type=int, default=3)
    parser.add_argument("--min-matched-optimized-audits", type=int, default=3)
    parser.add_argument(
        "--min-matched-optimized-relative-reduction", type=float, default=0.02
    )
    parser.add_argument(
        "--min-matched-optimized-uncertainty-sigma", type=float, default=1.0
    )
    parser.add_argument(
        "--allow-missing-optimized-equilibrium-transport",
        action="store_true",
        help="Do not require optimized-equilibrium replicated transport for production promotion.",
    )
    parser.add_argument(
        "--allow-missing-matched-optimized-audit",
        action="store_true",
        help="Do not require a matched baseline-to-optimized nonlinear transport audit for production promotion.",
    )
    parser.add_argument(
        "--fail-on-unsafe",
        action="store_true",
        help="Return non-zero if reduced/startup evidence is unsafe for release.",
    )
    parser.add_argument(
        "--fail-on-unpromoted",
        action="store_true",
        help="Return non-zero if production nonlinear optimization is not promoted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reduced_paths = list(args.reduced_artifact) or list(DEFAULT_REDUCED_ARTIFACTS)
    replicated_paths = list(args.replicated_ensemble) or list(
        DEFAULT_REPLICATED_ENSEMBLES
    )
    optimized_paths = list(args.optimized_equilibrium_ensemble) or list(
        DEFAULT_OPTIMIZED_EQUILIBRIUM_ENSEMBLES
    )
    matched_paths = list(args.matched_optimized_audit) or list(
        DEFAULT_MATCHED_OPTIMIZED_AUDITS
    )
    cfg = ProductionNonlinearOptimizationGuardConfig(
        min_replicated_ensembles=args.min_replicated_ensembles,
        min_reports_per_ensemble=args.min_reports_per_ensemble,
        max_mean_rel_spread=args.max_mean_rel_spread,
        max_combined_sem_rel=args.max_combined_sem_rel,
        require_optimized_equilibrium_transport=not args.allow_missing_optimized_equilibrium_transport,
        require_matched_optimized_transport_audit=not args.allow_missing_matched_optimized_audit,
        min_optimized_equilibrium_ensembles=args.min_optimized_equilibrium_ensembles,
        min_matched_optimized_audits=args.min_matched_optimized_audits,
        min_matched_optimized_relative_reduction=args.min_matched_optimized_relative_reduction,
        min_matched_optimized_uncertainty_sigma=args.min_matched_optimized_uncertainty_sigma,
    )
    report = production_nonlinear_optimization_guard_report(
        optimization_artifact=_load_json(args.optimization_artifact),
        optimization_artifact_path=_repo_relative(args.optimization_artifact),
        reduced_artifacts=_load_mapping(reduced_paths),
        replicated_ensemble_artifacts=_load_mapping(replicated_paths),
        optimized_equilibrium_artifacts=_load_mapping(optimized_paths),
        matched_optimized_transport_artifacts=_load_mapping(matched_paths),
        config=cfg,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.out_png is not None:
        _write_plot(report, args.out_png)
    if args.out_pdf is not None:
        _write_plot(report, args.out_pdf)
    _write_csv(report, args.out_csv or args.out_json.with_suffix(".csv"))
    print(
        json.dumps(
            {
                "safe_to_release": report["safe_to_release"],
                "promoted": report["production_nonlinear_optimization_promoted"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_unsafe and not bool(report["safe_to_release"]):
        return 1
    if args.fail_on_unpromoted and not bool(
        report["production_nonlinear_optimization_promoted"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
