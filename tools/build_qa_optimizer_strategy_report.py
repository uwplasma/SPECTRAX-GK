#!/usr/bin/env python3
"""Build the QA transport-optimizer strategy report from tracked evidence.

The report is intentionally evidence-driven: it combines the solved QA
optimizer rows with the admitted long-window RBC(1,1) nonlinear landscape.
It should be regenerated whenever either input artifact changes before making
claims about which optimizer family is appropriate for the next campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.plotting import set_plot_style  # noqa: E402


DEFAULT_PANEL_JSON = ROOT / "docs" / "_static" / "vmec_jax_qa_full_sweep_panel.json"
DEFAULT_LANDSCAPE_JSON = ROOT / "docs" / "_static" / "vmec_boundary_transport_landscape_rbc11_full.json"
DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "vmec_jax_qa_optimizer_strategy_report"
MIN_IOTA = 0.41


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve(strict=False).relative_to(ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _correlation(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    x = np.asarray([item[0] for item in pairs], dtype=float)
    y = np.asarray([item[1] for item in pairs], dtype=float)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 0.0 else math.nan


def _case_summary(panel: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in panel.get("cases", []):
        if not isinstance(case, dict):
            continue
        history = case.get("history") if isinstance(case.get("history"), dict) else {}
        setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
        objective_initial = _finite(history.get("objective_initial"))
        objective_final = _finite(history.get("objective_final"))
        reduction = (
            1.0 - objective_final / objective_initial
            if math.isfinite(objective_initial) and objective_initial != 0.0 and math.isfinite(objective_final)
            else math.nan
        )
        iota_final = _finite(history.get("iota_final"))
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "label": str(case.get("label", case.get("case_id", ""))).replace("\n", " "),
                "transport_kind": setup.get("transport_kind"),
                "optimizer_method": (setup.get("optimizer") or {}).get("method")
                if isinstance(setup.get("optimizer"), dict)
                else None,
                "spectrax_weight": _finite(setup.get("spectrax_weight")),
                "objective_initial": objective_initial,
                "objective_final": objective_final,
                "objective_reduction_fraction": reduction,
                "aspect_final": _finite(history.get("aspect_final")),
                "iota_final": iota_final,
                "iota_shortfall": max(0.0, MIN_IOTA - iota_final) if math.isfinite(iota_final) else math.nan,
                "qs_final": _finite(history.get("qs_final")),
                "transport_metric_final": _finite(history.get("transport_metric_final")),
                "gate_passed": bool(case.get("gate_passed")),
                "diagnostic_gate_passed": bool(case.get("diagnostic_gate_passed")),
                "gate_blockers": tuple(str(item) for item in case.get("gate_blockers", [])),
                "message": history.get("message"),
                "nfev": int(_finite(history.get("nfev"), 0.0)),
            }
        )
    return rows


def _landscape_summary(landscape: dict[str, Any]) -> dict[str, Any]:
    rows = landscape.get("rows", [])
    row_by_coef: dict[float, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            row_by_coef[round(_finite(row.get("coefficient_value")), 12)] = row

    nonlinear_points: list[dict[str, Any]] = []
    for point in landscape.get("nonlinear_ensemble_points", []):
        if not isinstance(point, dict) or not point.get("passed"):
            continue
        coef = round(_finite(point.get("coefficient_value")), 12)
        row = row_by_coef.get(coef, {})
        metrics = row.get("reduced_metrics") if isinstance(row.get("reduced_metrics"), dict) else {}
        nonlinear_points.append(
            {
                "label": row.get("label", point.get("case", "")),
                "relative_fraction": _finite(row.get("relative_fraction")),
                "coefficient_value": _finite(point.get("coefficient_value")),
                "mean": _finite(point.get("mean")),
                "sem": _finite(point.get("sem")),
                "metrics": {str(key): _finite(value) for key, value in metrics.items()},
            }
        )
    nonlinear_points.sort(key=lambda item: item["relative_fraction"])

    baseline = next((item for item in nonlinear_points if abs(item["relative_fraction"]) < 1e-12), None)
    best = min(nonlinear_points, key=lambda item: item["mean"]) if nonlinear_points else None
    baseline_mean = _finite(baseline.get("mean") if baseline else None)
    best_mean = _finite(best.get("mean") if best else None)
    best_reduction = (
        1.0 - best_mean / baseline_mean
        if math.isfinite(baseline_mean) and baseline_mean != 0.0 and math.isfinite(best_mean)
        else math.nan
    )

    metric_names = sorted({key for item in nonlinear_points for key in item["metrics"]})
    correlations: list[dict[str, Any]] = []
    y = [item["mean"] for item in nonlinear_points]
    for metric in metric_names:
        x = [item["metrics"].get(metric, math.nan) for item in nonlinear_points]
        correlations.append(
            {
                "metric": metric,
                "pearson_with_nonlinear_q": _correlation(x, y),
                "spearman_with_nonlinear_q": _correlation(_rank(x), _rank(y)),
            }
        )

    return {
        "n_admitted_nonlinear_points": len(nonlinear_points),
        "baseline_point": baseline,
        "best_point": best,
        "best_reduction_fraction_vs_baseline": best_reduction,
        "nonlinear_points": nonlinear_points,
        "metric_correlations": correlations,
    }


def build_report(panel_json: Path, landscape_json: Path) -> dict[str, Any]:
    panel = _read_json(panel_json)
    landscape = _read_json(landscape_json)
    cases = _case_summary(panel)
    landscape_summary = _landscape_summary(landscape)

    deterministic_cases = [row for row in cases if row["case_id"] != "qa_baseline_scipy"]
    all_transport_gates_pass = all(row["gate_passed"] for row in deterministic_cases) if deterministic_cases else False
    nonlinear_points = int(landscape_summary["n_admitted_nonlinear_points"])
    best_reduction = _finite(landscape_summary["best_reduction_fraction_vs_baseline"])

    recommendations = [
        {
            "stage": "strict_qa_baseline",
            "method": "vmec_jax_exact_discrete_adjoint_least_squares",
            "status": "preferred_for_smooth_aspect_iota_qs_constraints",
            "reason": "The admitted max-mode-5 baseline reaches the target aspect, iota, and QS gate; keep this as the reference seed.",
        },
        {
            "stage": "linear_and_quasilinear_transport",
            "method": "constraint_aware_adjoint_trust_or_lbfgs_with_weight_continuation",
            "status": "next_deterministic_campaign",
            "reason": (
                "The current scalar-trust rows move the internal objective but remain just below the strict iota gate. "
                "Use transport-weight continuation, active iota/aspect admission filters, multistarts, and RBC(1,1)-informed warm starts."
            ),
        },
        {
            "stage": "long_window_nonlinear_heat_flux",
            "method": "spsa_common_random_numbers_then_cma_es_or_bo_for_low_dimensional_projected_controls",
            "status": "preferred_noisy_outer_loop",
            "reason": (
                "Long-window Q has seed/timestep variability and cannot be promoted from reduced-window gradients. "
                "Use two-sided noisy objectives with common random numbers, then admit only matched t=1500 replicated audits."
            ),
        },
        {
            "stage": "quasilinear_claims",
            "method": "screening_and_model_development_only",
            "status": "absolute_flux_promotion_blocked",
            "reason": (
                "On the admitted RBC(1,1) landscape, simple linear/QL metrics are not reliable absolute nonlinear-Q predictors; "
                "promote correlation/screening only when held-out gates pass."
            ),
        },
    ]

    return {
        "kind": "vmec_jax_qa_optimizer_strategy_report",
        "schema_version": 1,
        "panel_json": _repo_relative(panel_json),
        "landscape_json": _repo_relative(landscape_json),
        "claim_scope": (
            "Optimizer-strategy evidence and campaign design; not a promoted nonlinear turbulent-flux optimization claim."
        ),
        "cases": cases,
        "landscape": landscape_summary,
        "gates": {
            "deterministic_transport_rows_all_strict_gates_pass": all_transport_gates_pass,
            "has_admitted_long_window_landscape": nonlinear_points >= 3,
            "has_material_landscape_reduction_direction": bool(math.isfinite(best_reduction) and best_reduction > 0.10),
            "nonlinear_absolute_optimization_promoted": False,
        },
        "optimizer_recommendations": recommendations,
        "literature_context": [
            {
                "topic": "smooth_QA_constraints",
                "source": "VMEC-JAX discrete-adjoint documentation",
                "url": "https://vmec-jax.readthedocs.io/en/latest/discrete_adjoint.html",
            },
            {
                "topic": "linear_quasilinear_microstability_optimization",
                "source": "Direct microstability optimization of stellarator devices",
                "url": "https://doi.org/10.1103/PhysRevE.110.035201",
            },
            {
                "topic": "noisy_nonlinear_heat_flux_optimization",
                "source": "Optimization of nonlinear turbulence in stellarators",
                "url": "https://doi.org/10.1017/S0022377824000369",
            },
        ],
    }


def _write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "label",
        "transport_kind",
        "optimizer_method",
        "spectrax_weight",
        "objective_initial",
        "objective_final",
        "objective_reduction_fraction",
        "iota_final",
        "iota_shortfall",
        "qs_final",
        "transport_metric_final",
        "gate_passed",
        "diagnostic_gate_passed",
        "gate_blockers",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["cases"]:
            out = {field: row.get(field) for field in fields}
            out["gate_blockers"] = ",".join(row.get("gate_blockers", ()))
            writer.writerow(out)


def _plot(report: dict[str, Any], path: Path) -> None:
    set_plot_style()
    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    cases = report["cases"]
    labels = [row["label"].replace(" from ", "\nfrom ") for row in cases]
    initial = np.asarray([row["objective_initial"] for row in cases], dtype=float)
    final = np.asarray([row["objective_final"] for row in cases], dtype=float)
    x = np.arange(len(cases))
    ax0.semilogy(x - 0.18, initial, "o", label="initial objective", color="#5b6770")
    ax0.semilogy(x + 0.18, final, "o", label="final objective", color="#c84630")
    for i, row in enumerate(cases):
        ax0.plot([i - 0.18, i + 0.18], [initial[i], final[i]], color="#b8b0a2", lw=1.2)
        if row["iota_shortfall"] > 0:
            ax0.text(i, final[i] * 1.25, "iota gate", ha="center", va="bottom", fontsize=7, rotation=90)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=18, ha="right")
    ax0.set_ylabel("optimizer objective")
    ax0.set_title("Current optimizer rows reduce the internal objective")
    ax0.legend(frameon=False, fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    landscape = report["landscape"]
    points = landscape["nonlinear_points"]
    frac = np.asarray([point["relative_fraction"] for point in points], dtype=float)
    q = np.asarray([point["mean"] for point in points], dtype=float)
    sem = np.asarray([point["sem"] for point in points], dtype=float)
    ax1.errorbar(frac, q, yerr=sem, marker="o", lw=1.5, capsize=2.5, color="#155f83")
    baseline = landscape.get("baseline_point") or {}
    best = landscape.get("best_point") or {}
    if baseline:
        ax1.axhline(baseline["mean"], color="#5b6770", ls="--", lw=1.0, label="baseline Q")
    if best:
        ax1.scatter([best["relative_fraction"]], [best["mean"]], s=70, color="#c84630", zorder=4, label="best admitted")
    ax1.set_xlabel("RBC(1,1) relative perturbation")
    ax1.set_ylabel("late-window nonlinear <Q_i>")
    ax1.set_title("A real lower-Q direction exists in the long-window landscape")
    ax1.legend(frameon=False, fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    corr = report["landscape"]["metric_correlations"]
    corr = [item for item in corr if math.isfinite(item["spearman_with_nonlinear_q"])]
    names = [item["metric"].replace("quasilinear_flux_", "QL ") for item in corr]
    values = [item["spearman_with_nonlinear_q"] for item in corr]
    colors = ["#c84630" if value < 0 else "#155f83" for value in values]
    ax2.barh(np.arange(len(corr)), values, color=colors)
    ax2.axvline(0.0, color="#3d3d3d", lw=0.8)
    ax2.set_yticks(np.arange(len(corr)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_xlabel("Spearman correlation with long-window Q")
    ax2.set_title("QL rules are screening evidence here, not absolute-Q predictors")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    lines = [
        "Recommended optimizer ladder",
        "",
        "1. Smooth QA constraints: VMEC-JAX exact-adjoint least squares.",
        "2. Linear/QL: adjoint trust/L-BFGS with transport-weight continuation,",
        "   multistarts, and active iota/aspect admission filters.",
        "3. Nonlinear Q: SPSA with common random numbers; compare CMA-ES/BO",
        "   only in a low-dimensional projected subspace.",
        "4. Promotion: matched t=1500 replicated nonlinear audits only.",
    ]
    reduction = _finite(landscape.get("best_reduction_fraction_vs_baseline"))
    if math.isfinite(reduction):
        lines += ["", f"Best admitted RBC(1,1) Q reduction: {100.0 * reduction:.1f}%"]
    ax3.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=10)

    fig.suptitle("QA transport optimization strategy from current evidence", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-json", type=Path, default=DEFAULT_PANEL_JSON)
    parser.add_argument("--landscape-json", type=Path, default=DEFAULT_LANDSCAPE_JSON)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.panel_json, args.landscape_json)
    out_json = args.out_prefix.with_suffix(".json")
    out_csv = args.out_prefix.with_suffix(".csv")
    out_png = args.out_prefix.with_suffix(".png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(report, out_csv)
    _plot(report, out_png)
    print(
        json.dumps(
            {
                "out_json": _repo_relative(out_json),
                "out_csv": _repo_relative(out_csv),
                "out_png": _repo_relative(out_png),
                "best_reduction_fraction_vs_baseline": report["landscape"][
                    "best_reduction_fraction_vs_baseline"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
