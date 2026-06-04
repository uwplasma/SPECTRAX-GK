#!/usr/bin/env python3
"""Build a fail-closed VMEC-JAX QA transport-optimization status panel.

The panel summarizes the current paper-facing state of the VMEC-JAX/SPECTRAX-GK
QA optimization lane from tracked artifacts.  It intentionally separates
solved-equilibrium admission gates from reduced transport objectives and from
long-window nonlinear audit evidence, so a reduced or failed candidate cannot
be mistaken for a promoted turbulent-flux optimization.
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


DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_jax_qa_transport_optimization_status.png"
DEFAULT_CONSTRAINTS_DIR = (
    ROOT / "tools_out" / "vmec_jax_qa_transport_authoritative_sidecar" / "constraints"
)
DEFAULT_DIRECT_TRANSPORT_DIR = (
    ROOT / "tools_out" / "vmec_jax_qa_transport_authoritative_sidecar" / "transport"
)
DEFAULT_PROJECTED_BASELINE_DIR = (
    ROOT
    / "tools_out"
    / "multisample_projected_line_search_smoke_grid_compatible"
    / "baseline_transport_metric_eval"
)
DEFAULT_PROJECTED_STEP_DIR = (
    ROOT
    / "tools_out"
    / "multisample_projected_line_search_smoke_grid_compatible"
    / "step_1p0em04"
    / "transport_metric_eval"
)
DEFAULT_LINE_SEARCH_JSON = ROOT / "docs" / "_static" / "vmec_boozer_aggregate_line_search_comparison.json"
DEFAULT_QL_RULE_JSON = ROOT / "docs" / "_static" / "quasilinear_saturation_rule_sweep.json"
DEFAULT_QL_MODEL_JSON = ROOT / "docs" / "_static" / "quasilinear_model_selection_status.json"
DEFAULT_NONLINEAR_AUDIT_JSON = ROOT / "docs" / "_static" / "qa_no_ess_to_optimized_nonlinear_audit.json"


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(ROOT.resolve(strict=False)))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_ready(value: Any) -> Any:
    """Return ``value`` with non-finite floats converted to JSON ``null``."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _candidate_row(label: str, root: Path, *, objective_kind: str) -> dict[str, Any]:
    history = _read_json(root / "history.json")
    gate = _read_json(root / "solved_wout_gate.json")
    wout_repro_gate_path = root / "wout_reproducibility_gate.json"
    wout_repro_gate = _read_json(wout_repro_gate_path) if wout_repro_gate_path.exists() else None
    checks = gate.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}

    def check_passed(name: str) -> bool:
        item = checks.get(name, {})
        return bool(item.get("passed", False)) if isinstance(item, dict) else False

    def check_margin(name: str) -> float:
        item = checks.get(name, {})
        if not isinstance(item, dict):
            return math.nan
        if name == "aspect":
            return _finite_float(item.get("absolute_tolerance")) - _finite_float(item.get("absolute_error"))
        if name == "iota_profile":
            return _finite_float(item.get("minimum_iotas_excluding_axis")) - _finite_float(item.get("floor"))
        return _finite_float(item.get("margin"))

    transport_metric = _finite_float(
        history.get("transport_metric_final", history.get("transport_objective_final")),
        default=math.nan,
    )
    solved_gate_passed = bool(gate.get("passed", False))
    wout_repro_gate_passed = (
        None if wout_repro_gate is None else bool(wout_repro_gate.get("passed", False))
    )
    admission_gate_passed = solved_gate_passed and (
        wout_repro_gate_passed is None or bool(wout_repro_gate_passed)
    )
    return {
        "label": label,
        "root": _repo_relative(root),
        "objective_kind": objective_kind,
        "passed_solved_wout_gate": bool(admission_gate_passed),
        "solved_wout_gate_passed": bool(solved_gate_passed),
        "wout_reproducibility_gate_passed": wout_repro_gate_passed,
        "wout_reproducibility_gate": wout_repro_gate,
        "aspect_final": _finite_float(history.get("aspect_final")),
        "iota_final": _finite_float(history.get("iota_final")),
        "qs_final": _finite_float(history.get("qs_final")),
        "objective_initial": _finite_float(history.get("objective_initial")),
        "objective_final": _finite_float(history.get("objective_final")),
        "transport_metric_final": transport_metric,
        "transport_metric_kind": history.get("transport_metric_kind"),
        "total_wall_time_s": _finite_float(history.get("total_wall_time_s")),
        "gate_checks": {
            "aspect": check_passed("aspect"),
            "mean_iota": check_passed("mean_iota"),
            "iota_profile": check_passed("iota_profile"),
            "quasisymmetry": check_passed("quasisymmetry"),
        },
        "gate_margins": {
            "aspect": check_margin("aspect"),
            "mean_iota": check_margin("mean_iota"),
            "iota_profile": check_margin("iota_profile"),
            "quasisymmetry": check_margin("quasisymmetry"),
        },
        "next_action": gate.get("next_action"),
    }


def _line_search_rows(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "objective": str(row.get("objective", "")),
                "passed": bool(row.get("passed", False)),
                "initial_objective": _finite_float(row.get("initial_objective")),
                "final_objective": _finite_float(row.get("final_objective")),
                "relative_reduction": _finite_float(row.get("relative_reduction")),
                "initial_update_direction": str(row.get("initial_update_direction", "")),
            }
        )
    return out


def _quasilinear_model_rows(rule_path: Path, model_path: Path) -> list[dict[str, Any]]:
    rule_data = _read_json(rule_path)
    model_data = _read_json(model_path)
    rows: list[dict[str, Any]] = []
    rules = rule_data.get("rules", {})
    if isinstance(rules, dict):
        for name, item in rules.items():
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "model": str(name),
                    "label": str(item.get("label", name)),
                    "source": "simple_saturation_rule",
                    "mean_abs_relative_error": _finite_float(item.get("holdout_mean_abs_relative_error")),
                    "passed": bool(item.get("holdout_gate_passed", False)),
                }
            )
    metrics = model_data.get("metrics", {})
    if isinstance(metrics, dict):
        rows.append(
            {
                "model": str(model_data.get("required_candidate", "spectral_envelope_ridge")),
                "label": "spectral envelope ridge",
                "source": "model_selection_status",
                "mean_abs_relative_error": _finite_float(metrics.get("candidate_mean_abs_relative_error")),
                "passed": bool(model_data.get("passed", False)),
            }
        )
    return rows


def build_payload(
    *,
    constraints_dir: Path = DEFAULT_CONSTRAINTS_DIR,
    direct_transport_dir: Path = DEFAULT_DIRECT_TRANSPORT_DIR,
    projected_baseline_dir: Path = DEFAULT_PROJECTED_BASELINE_DIR,
    projected_step_dir: Path = DEFAULT_PROJECTED_STEP_DIR,
    line_search_json: Path = DEFAULT_LINE_SEARCH_JSON,
    ql_rule_json: Path = DEFAULT_QL_RULE_JSON,
    ql_model_json: Path = DEFAULT_QL_MODEL_JSON,
    nonlinear_audit_json: Path = DEFAULT_NONLINEAR_AUDIT_JSON,
) -> dict[str, Any]:
    """Return a JSON-ready optimization status report from tracked artifacts."""

    candidates = [
        _candidate_row(
            "QA max_mode=5 baseline",
            constraints_dir,
            objective_kind="constraints_only",
        ),
        _candidate_row(
            "projected transport step",
            projected_step_dir,
            objective_kind="constraint_preserving_projected_nonlinear_window_metric",
        ),
        _candidate_row(
            "direct scalar transport",
            direct_transport_dir,
            objective_kind="scalar_transport_residual",
        ),
    ]
    projected_baseline = _candidate_row(
        "projected baseline metric",
        projected_baseline_dir,
        objective_kind="baseline_nonlinear_window_metric",
    )
    line_search = _line_search_rows(line_search_json)
    ql_rows = _quasilinear_model_rows(ql_rule_json, ql_model_json)
    nonlinear = _read_json(nonlinear_audit_json)
    comparison = nonlinear.get("comparison", {})
    if not isinstance(comparison, dict):
        comparison = {}
    baseline_metric = projected_baseline["transport_metric_final"]
    projected_metric = candidates[1]["transport_metric_final"]
    projected_relative_change = (
        (projected_metric - baseline_metric) / baseline_metric
        if math.isfinite(baseline_metric) and baseline_metric != 0.0 and math.isfinite(projected_metric)
        else math.nan
    )
    direct_transport_candidate = candidates[2]
    blocked = [
        row["label"]
        for row in candidates
        if not bool(row["passed_solved_wout_gate"])
        or any(not bool(v) for v in row["gate_checks"].values())
    ]
    return {
        "kind": "vmec_jax_qa_transport_optimization_status",
        "claim_scope": (
            "max_mode=5 VMEC-JAX QA/SPECTRAX-GK optimization status; solved-equilibrium "
            "gates, reduced line searches, quasilinear model-development diagnostics, and "
            "long-window nonlinear audit anchors are reported separately"
        ),
        "inputs": {
            "constraints_dir": _repo_relative(constraints_dir),
            "direct_transport_dir": _repo_relative(direct_transport_dir),
            "projected_baseline_dir": _repo_relative(projected_baseline_dir),
            "projected_step_dir": _repo_relative(projected_step_dir),
            "line_search_json": _repo_relative(line_search_json),
            "ql_rule_json": _repo_relative(ql_rule_json),
            "ql_model_json": _repo_relative(ql_model_json),
            "nonlinear_audit_json": _repo_relative(nonlinear_audit_json),
        },
        "projected_baseline": projected_baseline,
        "candidates": candidates,
        "line_search_rows": line_search,
        "quasilinear_model_rows": ql_rows,
        "long_window_nonlinear_audit": {
            "passed": bool(nonlinear.get("passed", False)),
            "baseline_mean": _finite_float(comparison.get("baseline_mean")),
            "optimized_mean": _finite_float(comparison.get("optimized_mean")),
            "relative_reduction": _finite_float(comparison.get("relative_reduction")),
            "uncertainty_separation_sigma": _finite_float(comparison.get("uncertainty_separation_sigma")),
            "claim_level": nonlinear.get("claim_level"),
        },
        "summary": {
            "qa_baseline_gate_passed": bool(candidates[0]["passed_solved_wout_gate"]),
            "direct_scalar_transport_gate_passed": bool(direct_transport_candidate["passed_solved_wout_gate"]),
            "direct_scalar_transport_blocked": not bool(direct_transport_candidate["passed_solved_wout_gate"]),
            "projected_transport_gate_passed": bool(candidates[1]["passed_solved_wout_gate"]),
            "projected_transport_relative_metric_change": projected_relative_change,
            "projected_transport_improved": bool(math.isfinite(projected_relative_change) and projected_relative_change < 0.0),
            "quasilinear_model_selection_passed": any(
                row["model"] == "spectral_envelope_ridge" and bool(row["passed"]) for row in ql_rows
            ),
            "simple_quasilinear_absolute_flux_promoted": any(
                row["source"] == "simple_saturation_rule" and bool(row["passed"]) for row in ql_rows
            ),
            "long_window_nonlinear_audit_passed": bool(nonlinear.get("passed", False)),
            "blocked_candidates": blocked,
            "next_action": (
                "Use the passing QA baseline and any constraint-preserving candidates for long-window nonlinear "
                "audits; do not promote direct scalar transport branches that fail aspect/iota/QS gates."
            ),
        },
    }


def _write_csv(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for row in payload["candidates"]:
        rows.append(
            {
                "section": "candidate",
                "label": row["label"],
                "passed": row["passed_solved_wout_gate"],
                "metric": row["transport_metric_final"],
                "relative_change": "",
                "objective_kind": row["objective_kind"],
            }
        )
    for row in payload["line_search_rows"]:
        rows.append(
            {
                "section": "line_search",
                "label": row["objective"],
                "passed": row["passed"],
                "metric": row["final_objective"],
                "relative_change": row["relative_reduction"],
                "objective_kind": row["objective"],
            }
        )
    for row in payload["quasilinear_model_rows"]:
        rows.append(
            {
                "section": "quasilinear_model",
                "label": row["model"],
                "passed": row["passed"],
                "metric": row["mean_abs_relative_error"],
                "relative_change": "",
                "objective_kind": row["source"],
            }
        )
    audit = payload["long_window_nonlinear_audit"]
    rows.append(
        {
            "section": "long_window_nonlinear_audit",
            "label": "optimized_vs_baseline",
            "passed": audit["passed"],
            "metric": audit["optimized_mean"],
            "relative_change": audit["relative_reduction"],
            "objective_kind": audit["claim_level"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _annotate_status(ax: plt.Axes, x: float, y: float, text: str, *, passed: bool) -> None:
    color = "#2a9d8f" if passed else "#c43c39"
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.0,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "none", "alpha": 0.95},
    )


def plot_payload(payload: dict[str, Any], out: Path) -> None:
    """Render the optimization-status panel."""

    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8), constrained_layout=True)
    colors = {
        "QA max_mode=5 baseline": "#244c66",
        "projected transport step": "#2a9d8f",
        "direct scalar transport": "#b45f2a",
    }

    ax = axes[0, 0]
    gate_names = ("aspect", "mean_iota", "iota_profile", "quasisymmetry")
    gate_labels = ("aspect", r"$|\bar\iota|$", r"$\min\iota(s)$", "QS")
    x = np.arange(len(gate_names), dtype=float)
    width = 0.23
    offsets = (-width, 0.0, width)
    for offset, row in zip(offsets, payload["candidates"], strict=True):
        margins = [row["gate_margins"][name] for name in gate_names]
        ax.bar(
            x + offset,
            margins,
            width=width,
            label=row["label"].replace(" max_mode=5", ""),
            color=colors.get(row["label"], "#6b7280"),
            edgecolor="#1f2937",
            linewidth=0.5,
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(gate_labels)
    ax.set_ylabel("positive solved-gate margin")
    ax.set_title("Full max_mode=5 solved-equilibrium gates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[0, 1]
    baseline = payload["projected_baseline"]
    projected = payload["candidates"][1]
    metric_labels = ["QA metric eval", "projected step"]
    metric_values = [baseline["transport_metric_final"], projected["transport_metric_final"]]
    bars = ax.bar(
        np.arange(2),
        metric_values,
        color=["#94d2bd", "#2a9d8f"],
        edgecolor="#1f2937",
    )
    for bar, value in zip(bars, metric_values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    rel_change = payload["summary"]["projected_transport_relative_metric_change"]
    title = "nonlinear-window metric"
    if math.isfinite(float(rel_change)):
        title += f" ({rel_change:+.2%})"
    ax.set_xticks(np.arange(2), metric_labels, rotation=12, ha="right")
    ax.set_ylabel("lower is better")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 2]
    rows = payload["line_search_rows"]
    labels = [row["objective"].replace("_", " ") for row in rows]
    initial = np.asarray([row["initial_objective"] for row in rows], dtype=float)
    final = np.asarray([row["final_objective"] for row in rows], dtype=float)
    denom = np.where(np.isfinite(initial) & (np.abs(initial) > 0.0), initial, np.nan)
    final_norm = final / denom
    xr = np.arange(max(len(rows), 1), dtype=float)
    width = 0.34
    if rows:
        ax.bar(xr - width / 2.0, np.ones_like(xr), width, label="initial", color="#e9c46a", edgecolor="#1f2937")
        ax.bar(xr + width / 2.0, final_norm, width, label="final/initial", color="#264653", edgecolor="#1f2937")
        for xi, row in zip(xr, rows, strict=True):
            reduction = row["relative_reduction"]
            text = f"{reduction:.3%}" if math.isfinite(float(reduction)) else "n/a"
            ax.text(xi, 1.03, text, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xr, labels or ["objective"], rotation=12, ha="right")
    ax.set_ylim(0.97, 1.04)
    ax.set_ylabel("normalized objective")
    ax.set_title("Reduced growth / QL line searches")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ql_rows = payload["quasilinear_model_rows"]
    ql_labels = [row["model"].replace("_", "\n") for row in ql_rows]
    ql_values = np.asarray([row["mean_abs_relative_error"] for row in ql_rows], dtype=float)
    ql_colors = ["#2a9d8f" if row["passed"] else "#c43c39" for row in ql_rows]
    ax.bar(np.arange(len(ql_rows)), ql_values, color=ql_colors, edgecolor="#1f2937")
    ax.axhline(0.35, color="#111827", linestyle=":", linewidth=1.1, label="promotion gate")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(ql_rows)), ql_labels, rotation=0)
    ax.set_ylabel("holdout mean abs. rel. error")
    ax.set_title("Quasilinear model candidates")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    audit = payload["long_window_nonlinear_audit"]
    flux_labels = ["matched baseline", "optimized QA/ESS"]
    flux_values = [audit["baseline_mean"], audit["optimized_mean"]]
    ax.bar(np.arange(2), flux_values, color=["#8ecae6", "#fb8500"], edgecolor="#1f2937")
    for xi, value in enumerate(flux_values):
        ax.text(xi, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(2), flux_labels, rotation=12, ha="right")
    ax.set_ylabel(r"late-window $\langle Q_i\rangle$")
    reduction = audit["relative_reduction"]
    sigma = audit["uncertainty_separation_sigma"]
    title = "Long-window nonlinear audit"
    if math.isfinite(float(reduction)) and math.isfinite(float(sigma)):
        title += f" ({reduction:.1%}, {sigma:.1f} SEM)"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 2]
    ax.axis("off")
    statuses = [
        ("QA gate", payload["summary"]["qa_baseline_gate_passed"]),
        ("direct scalar transport", not payload["summary"]["direct_scalar_transport_blocked"]),
        ("projected metric improves", payload["summary"]["projected_transport_improved"]),
        ("QL model selection", payload["summary"]["quasilinear_model_selection_passed"]),
        ("simple abs-flux QL", payload["summary"]["simple_quasilinear_absolute_flux_promoted"]),
        ("nonlinear audit", payload["summary"]["long_window_nonlinear_audit_passed"]),
    ]
    ax.set_title("Claim boundary")
    for idx, (label, passed) in enumerate(statuses):
        y = 0.88 - idx * 0.14
        _annotate_status(ax, 0.32, y, "PASS" if passed else "BLOCKED", passed=passed)
        ax.text(0.48, y, label, ha="left", va="center", fontsize=9.0, color="#111827")
    ax.text(
        0.02,
        0.03,
        "Reduced objectives guide candidates; only solved-gate-passing WOUTs enter long-window nonlinear audits.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        wrap=True,
        color="#374151",
    )

    fig.suptitle(
        "VMEC-JAX QA + SPECTRAX-GK transport optimization status",
        fontsize=15,
        fontweight="bold",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints-dir", type=Path, default=DEFAULT_CONSTRAINTS_DIR)
    parser.add_argument("--direct-transport-dir", type=Path, default=DEFAULT_DIRECT_TRANSPORT_DIR)
    parser.add_argument("--projected-baseline-dir", type=Path, default=DEFAULT_PROJECTED_BASELINE_DIR)
    parser.add_argument("--projected-step-dir", type=Path, default=DEFAULT_PROJECTED_STEP_DIR)
    parser.add_argument("--line-search-json", type=Path, default=DEFAULT_LINE_SEARCH_JSON)
    parser.add_argument("--ql-rule-json", type=Path, default=DEFAULT_QL_RULE_JSON)
    parser.add_argument("--ql-model-json", type=Path, default=DEFAULT_QL_MODEL_JSON)
    parser.add_argument("--nonlinear-audit-json", type=Path, default=DEFAULT_NONLINEAR_AUDIT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pdf", action="store_true", help="also write a PDF companion")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_payload(
        constraints_dir=args.constraints_dir,
        direct_transport_dir=args.direct_transport_dir,
        projected_baseline_dir=args.projected_baseline_dir,
        projected_step_dir=args.projected_step_dir,
        line_search_json=args.line_search_json,
        ql_rule_json=args.ql_rule_json,
        ql_model_json=args.ql_model_json,
        nonlinear_audit_json=args.nonlinear_audit_json,
    )
    base = args.out.with_suffix("")
    base.with_suffix(".json").write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(payload, base.with_suffix(".csv"))
    plot_payload(payload, args.out)
    if args.pdf:
        plot_payload(payload, base.with_suffix(".pdf"))
    print(
        json.dumps(
            {
                "qa_baseline_gate_passed": payload["summary"]["qa_baseline_gate_passed"],
                "direct_scalar_transport_blocked": payload["summary"]["direct_scalar_transport_blocked"],
                "projected_transport_relative_metric_change": payload["summary"][
                    "projected_transport_relative_metric_change"
                ],
                "quasilinear_model_selection_passed": payload["summary"]["quasilinear_model_selection_passed"],
                "long_window_nonlinear_audit_passed": payload["summary"]["long_window_nonlinear_audit_passed"],
                "paths": {
                    "png": str(args.out),
                    "json": str(base.with_suffix(".json")),
                    "csv": str(base.with_suffix(".csv")),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
