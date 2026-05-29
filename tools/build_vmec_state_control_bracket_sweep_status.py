#!/usr/bin/env python3
"""Summarize VMEC-state nonlinear bracket-amplitude sweep gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_vmec_boundary_perturbation_inputs import _json_clean  # noqa: E402


DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_state_control_bracket_sweep_status"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _short_parameter(raw: str) -> str:
    return raw.removeprefix("state_control_")


def _row_from_gate(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{path} missing metrics object")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        config = {}
    parameter = str(payload.get("parameter_name", path.stem))
    return {
        "artifact": str(path),
        "alpha_delta": float(payload.get("delta_parameter", float("nan"))),
        "parameter_name": parameter,
        "state_parameter": _short_parameter(parameter),
        "passed": bool(payload.get("passed")),
        "blockers": list(payload.get("blockers", [])),
        "response_fraction": float(metrics.get("response_fraction", float("nan"))),
        "min_response_fraction": float(config.get("min_fd_response_fraction", 0.03)),
        "fd_asymmetry_rel": float(metrics.get("fd_asymmetry_rel", float("nan"))),
        "max_fd_asymmetry_rel": float(config.get("max_fd_asymmetry_rel", 0.5)),
        "gradient_uncertainty_rel": float(metrics.get("gradient_uncertainty_rel", float("nan"))),
        "max_gradient_uncertainty_rel": float(config.get("max_gradient_uncertainty_rel", 0.5)),
        "baseline_window_mean": float(metrics.get("baseline_window_mean", float("nan"))),
        "plus_window_mean": float(metrics.get("plus_window_mean", float("nan"))),
        "minus_window_mean": float(metrics.get("minus_window_mean", float("nan"))),
    }


def build_bracket_sweep_status(
    gate_paths: Sequence[Path],
    *,
    run_summary: Path | None = None,
    out_prefix: Path = DEFAULT_OUT_PREFIX,
) -> dict[str, Any]:
    rows = [_row_from_gate(path) for path in gate_paths]
    if not rows:
        raise ValueError("at least one central-FD gate is required")
    rows = sorted(rows, key=lambda row: (row["alpha_delta"], row["state_parameter"]))
    run_payload: dict[str, Any] | None = None
    if run_summary is not None:
        run_payload = _load_json(run_summary)
    summary = {
        "central_fd_gates_passed": sum(1 for row in rows if row["passed"]),
        "central_fd_gates_total": len(rows),
        "max_response_fraction": max(row["response_fraction"] for row in rows),
        "min_gradient_uncertainty_rel": min(row["gradient_uncertainty_rel"] for row in rows),
        "nonlinear_runs_completed": None if run_payload is None else int(run_payload.get("successes", 0)),
        "nonlinear_run_failures": None if run_payload is None else len(run_payload.get("failures", [])),
        "nonlinear_run_seconds": None if run_payload is None else float(run_payload.get("finished_at", 0.0) - run_payload.get("started_at", 0.0)),
    }
    passed = bool(summary["central_fd_gates_passed"] == summary["central_fd_gates_total"])
    report = {
        "kind": "vmec_state_control_bracket_sweep_status",
        "claim_level": "bracket_amplitude_sweep_negative_nonlinear_gradient_evidence",
        "passed": passed,
        "summary": summary,
        "rows": rows,
        "next_action": (
            "do not increase single-control bracket amplitude further; the 3e-3 and 1e-2 sweeps are runtime-stable "
            "but response fractions remain well below the resolved-gradient gate, so the next scientific step is "
            "variance reduction through longer replicated windows or a better-conditioned multi-control observable"
        ),
    }
    _write_outputs(out_prefix, report)
    return report


def _write_outputs(out_prefix: Path, report: Mapping[str, Any]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "alpha_delta",
                "state_parameter",
                "passed",
                "response_fraction",
                "min_response_fraction",
                "fd_asymmetry_rel",
                "max_fd_asymmetry_rel",
                "gradient_uncertainty_rel",
                "max_gradient_uncertainty_rel",
                "blockers",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in report["rows"]:
            out = dict(row)
            out["blockers"] = ";".join(row["blockers"])
            writer.writerow({field: out[field] for field in writer.fieldnames})
    _plot(png_path, report)


def _plot(path: Path, report: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from spectraxgk.plotting import set_plot_style

    rows = list(report["rows"])
    labels = [f"{row['state_parameter']}\n$\\alpha$={row['alpha_delta']:.3g}" for row in rows]
    x = np.arange(len(rows))
    response = np.asarray([row["response_fraction"] for row in rows], dtype=float)
    asymmetry = np.asarray([row["fd_asymmetry_rel"] for row in rows], dtype=float)
    uncertainty = np.asarray([row["gradient_uncertainty_rel"] for row in rows], dtype=float)
    min_response = float(rows[0]["min_response_fraction"])
    max_asymmetry = float(rows[0]["max_fd_asymmetry_rel"])
    max_uncertainty = float(rows[0]["max_gradient_uncertainty_rel"])

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    axes[0].bar(x, response, color="#377eb8", edgecolor="0.2")
    axes[0].axhline(min_response, color="#e41a1c", lw=1.6, ls="--", label=f"gate {min_response:.2f}")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("FD response fraction")
    axes[0].set_title("Resolved-response gate")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.25)

    width = 0.38
    axes[1].bar(x - width / 2, asymmetry, width=width, color="#ff7f00", edgecolor="0.2", label="FD asymmetry")
    axes[1].bar(x + width / 2, uncertainty, width=width, color="#4daf4a", edgecolor="0.2", label="gradient uncertainty")
    axes[1].axhline(max_asymmetry, color="0.25", lw=1.2, ls="--", label="gate 0.5")
    axes[1].axhline(max_uncertainty, color="0.25", lw=1.2, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("relative diagnostic")
    axes[1].set_title("Asymmetry and uncertainty gates")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    summary = report["summary"]
    fig.suptitle(
        f"VMEC-state nonlinear bracket sweep: {summary['central_fd_gates_passed']}/"
        f"{summary['central_fd_gates_total']} central-FD gates passed",
        fontsize=14,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gates", nargs="+", type=Path)
    parser.add_argument("--run-summary", type=Path)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    args = parser.parse_args(argv)
    report = build_bracket_sweep_status(args.gates, run_summary=args.run_summary, out_prefix=args.out_prefix)
    print(
        json.dumps(
            {
                "json": str(args.out_prefix.with_suffix(".json")),
                "passed": report["passed"],
                "central_fd_gates_passed": report["summary"]["central_fd_gates_passed"],
                "central_fd_gates_total": report["summary"]["central_fd_gates_total"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
