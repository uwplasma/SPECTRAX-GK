#!/usr/bin/env python3
"""Build a machine-readable parallelization closure status artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC = REPO_ROOT / "docs" / "_static"
DEFAULT_OUT_PREFIX = STATIC / "parallelization_completion_status"

PRODUCTION_THRESHOLDS = {
    "independent_ky_scan": {"cpu": 5.0, "gpu": 1.5},
    "quasilinear_uq_ensemble": {"cpu": 5.0, "gpu": 1.5},
}

ARTIFACTS = {
    "independent_ky_scan": "independent_ky_scan_scaling_large.json",
    "quasilinear_uq_ensemble": "quasilinear_uq_ensemble_scaling_large.json",
    "whole_state_nonlinear_sharding": "nonlinear_sharding_strong_scaling_large.json",
    "fft_axis_domain": "nonlinear_spectral_communication_identity_gate.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _best_speedups_by_backend(payload: dict[str, Any]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in _rows(payload):
        backend = str(row.get("backend", payload.get("backend", ""))).strip().lower()
        if not backend:
            continue
        speedup = float(row.get("strong_speedup_vs_1_device", 0.0))
        best[backend] = max(best.get(backend, 0.0), speedup)
    return best


def _production_lane(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    thresholds = PRODUCTION_THRESHOLDS[name]
    speedups = _best_speedups_by_backend(payload)
    identity_passed = bool(payload.get("identity_passed")) and all(
        row.get("identity_gate_pass") is True for row in _rows(payload)
    )
    threshold_passed = all(
        speedups.get(backend, 0.0) >= threshold
        for backend, threshold in thresholds.items()
    )
    passed = bool(identity_passed and threshold_passed)
    return {
        "lane": name,
        "status": "production_closed" if passed else "open",
        "claim_level": "production_parallelization",
        "identity_passed": identity_passed,
        "threshold_passed": threshold_passed,
        "thresholds": thresholds,
        "best_speedups": speedups,
        "source": f"docs/_static/{ARTIFACTS[name]}",
        "summary": (
            "Production independent-work strong scaling with serial numerical identity."
            if passed
            else "Production independent-work scaling is not yet closed."
        ),
    }


def _diagnostic_nonlinear_lane(payload: dict[str, Any]) -> dict[str, Any]:
    speedups = _best_speedups_by_backend(payload)
    identity_passed = bool(payload.get("identity_passed")) and all(
        row.get("identity_gate_pass") is True for row in _rows(payload)
    )
    claim_scope = str(payload.get("claim_scope", ""))
    scoped = "not a production speedup claim" in claim_scope
    return {
        "lane": "whole_state_nonlinear_sharding",
        "status": "diagnostic_closed_not_production" if identity_passed and scoped else "open",
        "claim_level": "diagnostic_identity_and_profiler_evidence",
        "identity_passed": identity_passed,
        "threshold_passed": None,
        "best_speedups": speedups,
        "source": f"docs/_static/{ARTIFACTS['whole_state_nonlinear_sharding']}",
        "summary": (
            "Whole-state nonlinear sharding preserves state identity but is not promoted as "
            "production nonlinear domain-decomposition speedup."
        ),
    }


def _spectral_lane(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload.get("gate", {}) if isinstance(payload.get("gate"), dict) else {}
    identity_passed = bool(payload.get("identity_passed", gate.get("identity_passed"))) or bool(
        gate.get("identity_passed")
    )
    rows_passed = all(row.get("identity_passed") is True for row in _rows(payload))
    claim_scope = str(payload.get("claim_scope", ""))
    scoped = "no production routing or speedup claim" in claim_scope
    passed = bool(identity_passed and rows_passed and scoped)
    return {
        "lane": "fft_axis_domain",
        "status": "diagnostic_identity_closed" if passed else "open",
        "claim_level": "diagnostic_communication_identity",
        "identity_passed": passed,
        "threshold_passed": None,
        "best_speedups": {},
        "source": f"docs/_static/{ARTIFACTS['fft_axis_domain']}",
        "summary": (
            "Spectral split/reassemble, FFT round-trip, bracket, and field-layout identity are closed; "
            "runtime distributed FFT routing remains out of production scope."
        ),
    }


def build_status(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return release-scoped parallelization completion status."""

    static = root / "docs" / "_static"
    payloads = {name: _read_json(static / artifact) for name, artifact in ARTIFACTS.items()}
    lanes = [
        _production_lane("independent_ky_scan", payloads["independent_ky_scan"]),
        _production_lane("quasilinear_uq_ensemble", payloads["quasilinear_uq_ensemble"]),
        _diagnostic_nonlinear_lane(payloads["whole_state_nonlinear_sharding"]),
        _spectral_lane(payloads["fft_axis_domain"]),
    ]
    production = [lane for lane in lanes if lane["claim_level"] == "production_parallelization"]
    production_closed = [lane for lane in production if lane["status"] == "production_closed"]
    diagnostic_closed = [lane for lane in lanes if str(lane["status"]).startswith("diagnostic")]
    production_completion = 100.0 * len(production_closed) / max(len(production), 1)
    overall_completion = 100.0 * (
        len(production_closed) + 0.75 * len(diagnostic_closed)
    ) / max(len(lanes), 1)
    passed = production_completion == 100.0 and all(lane["identity_passed"] for lane in lanes)
    return {
        "kind": "parallelization_completion_status",
        "claim_scope": (
            "Release production parallelization is closed for independent ky scans and "
            "quasilinear/UQ ensembles with CPU/GPU strong-scaling and serial numerical "
            "identity. Whole-state nonlinear sharding and FFT-axis decomposition remain "
            "diagnostic until runtime distributed communication, conservation, transport-window, "
            "and profiler-backed speedup gates pass."
        ),
        "passed": passed,
        "production_completion_percent": production_completion,
        "overall_parallelization_percent": overall_completion,
        "lanes": lanes,
    }


def _json_path_for_prefix(out_prefix: Path) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    return out_prefix.with_suffix(".json")


def write_json_artifact(status: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    json_path = _json_path_for_prefix(out_prefix)
    json_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(json_path)}


def write_artifacts(status: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    paths = write_json_artifact(status, out_prefix)

    lanes = list(status["lanes"])
    labels = [str(lane["lane"]).replace("_", "\n") for lane in lanes]
    cpu = [float(lane.get("best_speedups", {}).get("cpu", math.nan)) for lane in lanes]
    gpu = [float(lane.get("best_speedups", {}).get("gpu", math.nan)) for lane in lanes]
    colors = [
        "#2f7f5f" if str(lane["status"]).startswith(("production", "diagnostic")) else "#b44a3c"
        for lane in lanes
    ]

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), constrained_layout=True)
    y = list(range(len(lanes)))
    score = [
        100.0
        if lane["status"] == "production_closed"
        else 75.0
        if str(lane["status"]).startswith("diagnostic")
        else 0.0
        for lane in lanes
    ]
    axes[0].barh(y, score, color=colors, alpha=0.88)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 105.0)
    axes[0].set_xlabel("closure score [%]")
    axes[0].set_title("Parallelization claim status")
    axes[0].axvline(100.0, color="0.25", ls=":", lw=1.1)
    for idx, lane in enumerate(lanes):
        status_label = {
            "production_closed": "production closed",
            "diagnostic_closed_not_production": "diagnostic only",
            "diagnostic_identity_closed": "identity gate",
        }.get(str(lane["status"]), str(lane["status"]).replace("_", " "))
        axes[0].text(
            max(score[idx] - 2.0, 2.0),
            idx,
            status_label,
            va="center",
            ha="right",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    width = 0.36
    x = list(range(len(lanes)))
    cpu_plot = [0.0 if math.isnan(value) else value for value in cpu]
    gpu_plot = [0.0 if math.isnan(value) else value for value in gpu]
    axes[1].bar([value - width / 2 for value in x], cpu_plot, width, label="CPU", color="#276b8e")
    axes[1].bar([value + width / 2 for value in x], gpu_plot, width, label="GPU", color="#c45a14")
    axes[1].axhline(5.0, color="#276b8e", ls=":", lw=1.1, label="CPU prod. gate")
    axes[1].axhline(1.5, color="#c45a14", ls="--", lw=1.1, label="GPU prod. gate")
    axes[1].set_xticks(x, labels, rotation=0)
    axes[1].set_ylabel("best strong speedup")
    axes[1].set_title("Production speedup evidence")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("SPECTRAX-GK parallelization closure", fontsize=13, fontweight="bold")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    paths.update({"png": str(png_path), "pdf": str(pdf_path)})
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--fail-under-production", type=float, default=100.0)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write only the JSON status artifact; useful in dependency-free CI hygiene jobs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = build_status(args.root)
    paths = (
        write_json_artifact(status, Path(args.out_prefix))
        if args.skip_figures
        else write_artifacts(status, Path(args.out_prefix))
    )
    print(json.dumps({"passed": status["passed"], "paths": paths}, indent=2))
    return (
        0
        if float(status["production_completion_percent"]) >= float(args.fail_under_production)
        and status["passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
