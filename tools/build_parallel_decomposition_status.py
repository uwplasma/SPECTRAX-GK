#!/usr/bin/env python3
"""Build a decomposition-contract status artifact from existing parallel gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from spectraxgk.parallel.decomposition import (
    DecompositionContract,
    build_diagnostic_nonlinear_domain_decomposition,
    build_independent_portfolio_decomposition,
    serial_reconstruction_identity_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC = REPO_ROOT / "docs" / "_static"
DEFAULT_OUT_PREFIX = STATIC / "parallel_decomposition_status"

ARTIFACTS = {
    "independent_ky_scan": "parallel_ky_scan_gate.json",
    "uq_ensemble": "quasilinear_runtime_parallel_gate.json",
    "diagnostic_nonlinear_domain": "nonlinear_domain_parallel_identity_gate.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _identity_passed(payload: dict[str, Any]) -> bool:
    gate = payload.get("gate", {}) if isinstance(payload.get("gate"), dict) else {}
    if "identity_passed" in gate:
        return bool(gate["identity_passed"])
    return bool(payload.get("identity_passed"))


def _parallel_count(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            count = int(value)
            if count >= 1:
                return count
    return 1


def _ky_contract(payload: dict[str, Any]) -> DecompositionContract:
    n_items = len(payload.get("ky_values", [])) or len(_rows(payload))
    return build_independent_portfolio_decomposition(
        n_items,
        requested_shards=_parallel_count(payload, "parallel_batch", "actual_devices"),
        workload="independent_ky_scan",
    )


def _uq_contract(payload: dict[str, Any]) -> DecompositionContract:
    n_items = len(payload.get("ky_values", [])) or len(_rows(payload))
    return build_independent_portfolio_decomposition(
        n_items,
        requested_shards=_parallel_count(payload, "workers", "actual_workers"),
        workload="uq_ensemble",
    )


def _diagnostic_domain_contract(payload: dict[str, Any]) -> DecompositionContract:
    gate = payload.get("gate", {})
    if not isinstance(gate, dict):
        raise ValueError("nonlinear domain artifact must contain a gate object")
    plan = gate.get("plan", {})
    if not isinstance(plan, dict):
        raise ValueError("nonlinear domain artifact gate must contain a plan object")
    state_shape = plan.get("state_shape", payload.get("state_shape"))
    chunk_sizes = plan.get("chunk_sizes", ())
    if not state_shape:
        raise ValueError("nonlinear domain artifact must provide state_shape")
    if not chunk_sizes:
        raise ValueError("nonlinear domain artifact must provide chunk_sizes")
    return build_diagnostic_nonlinear_domain_decomposition(
        state_shape,
        axis=int(plan.get("axis", 0)),
        requested_shards=len(chunk_sizes),
    )


def _claim_separation_passed(contract: DecompositionContract) -> bool:
    claim = contract.claim_label.lower()
    if contract.production_independent_batching:
        return (
            "production independent batching" in claim
            and "not a nonlinear state-domain decomposition" in claim
            and contract.independent_work
            and not contract.changes_solver_layout
        )
    return (
        "diagnostic nonlinear state-domain partition" in claim
        and "no production routing or speedup claim" in claim
        and not contract.independent_work
        and contract.changes_solver_layout
    )


def _lane(
    name: str,
    source: str,
    payload: dict[str, Any],
    contract: DecompositionContract,
) -> dict[str, Any]:
    values = tuple(range(contract.n_items))
    reconstruction = serial_reconstruction_identity_report(values, contract)
    artifact_identity_passed = _identity_passed(payload)
    claim_separation_passed = _claim_separation_passed(contract)
    passed = bool(
        artifact_identity_passed
        and reconstruction.identity_passed
        and claim_separation_passed
    )
    return {
        "lane": name,
        "source": f"docs/_static/{source}",
        "claim_level": contract.claim_level,
        "claim_label": contract.claim_label,
        "n_items": contract.n_items,
        "requested_shards": contract.requested_shards,
        "actual_shards": contract.actual_shards,
        "artifact_identity_passed": artifact_identity_passed,
        "reconstruction_identity_passed": reconstruction.identity_passed,
        "claim_separation_passed": claim_separation_passed,
        "passed": passed,
        "contract": contract.to_dict(),
        "reconstruction": reconstruction.to_dict(),
    }


def build_status(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return decomposition-contract status from existing static artifacts."""

    static = root / "docs" / "_static"
    payloads = {
        name: _read_json(static / artifact)
        for name, artifact in ARTIFACTS.items()
    }
    lanes = [
        _lane(
            "independent_ky_scan",
            ARTIFACTS["independent_ky_scan"],
            payloads["independent_ky_scan"],
            _ky_contract(payloads["independent_ky_scan"]),
        ),
        _lane(
            "uq_ensemble",
            ARTIFACTS["uq_ensemble"],
            payloads["uq_ensemble"],
            _uq_contract(payloads["uq_ensemble"]),
        ),
        _lane(
            "diagnostic_nonlinear_domain",
            ARTIFACTS["diagnostic_nonlinear_domain"],
            payloads["diagnostic_nonlinear_domain"],
            _diagnostic_domain_contract(payloads["diagnostic_nonlinear_domain"]),
        ),
    ]
    passed = all(lane["passed"] for lane in lanes)
    production_lanes = [
        lane
        for lane in lanes
        if lane["claim_level"] == "production_independent_batching"
    ]
    diagnostic_lanes = [
        lane
        for lane in lanes
        if lane["claim_level"] == "diagnostic_nonlinear_domain_partition"
    ]
    return {
        "kind": "parallel_decomposition_status",
        "claim_scope": (
            "Deterministic decomposition-contract status only. Independent ky "
            "and UQ portfolios are production independent-batching contracts "
            "with serial reconstruction identity. Nonlinear state-domain "
            "partitioning remains diagnostic metadata with no production "
            "routing or speedup claim."
        ),
        "passed": passed,
        "production_independent_lanes": len(production_lanes),
        "diagnostic_nonlinear_lanes": len(diagnostic_lanes),
        "lanes": lanes,
    }


def write_json_artifact(status: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    path = out_prefix.with_suffix(".json")
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(path)}


def write_csv_artifact(status: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    path = out_prefix.with_suffix(".csv")
    fields = [
        "lane",
        "claim_level",
        "n_items",
        "requested_shards",
        "actual_shards",
        "artifact_identity_passed",
        "reconstruction_identity_passed",
        "claim_separation_passed",
        "passed",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for lane in status["lanes"]:
            writer.writerow({field: lane[field] for field in fields})
    return {"csv": str(path)}


def write_artifacts(status: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = write_json_artifact(status, out_prefix)
    paths.update(write_csv_artifact(status, out_prefix))

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    lanes = list(status["lanes"])
    labels = [str(lane["lane"]).replace("_", "\n") for lane in lanes]
    scores = [100.0 if lane["passed"] else 0.0 for lane in lanes]
    colors = ["#2f7f5f" if lane["passed"] else "#b44a3c" for lane in lanes]

    fig, ax = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
    y = list(range(len(lanes)))
    ax.barh(y, scores, color=colors, alpha=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 105.0)
    ax.set_xlabel("contract gates passed [%]")
    ax.set_title("Parallel decomposition contract status")
    ax.axvline(100.0, color="0.25", ls=":", lw=1.0)
    for idx, lane in enumerate(lanes):
        label = "pass" if lane["passed"] else "open"
        ax.text(
            max(scores[idx] - 3.0, 2.0),
            idx,
            label,
            va="center",
            ha="right" if lane["passed"] else "left",
            color="white" if lane["passed"] else "black",
            fontsize=9,
            fontweight="bold",
        )
    ax.grid(True, axis="x", alpha=0.2)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    paths.update({"png": str(png_path), "pdf": str(pdf_path)})
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write only JSON and CSV artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = build_status(args.root)
    paths = (
        {**write_json_artifact(status, args.out_prefix), **write_csv_artifact(status, args.out_prefix)}
        if args.skip_figures
        else write_artifacts(status, args.out_prefix)
    )
    print(json.dumps({"passed": status["passed"], "paths": paths}, indent=2))
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
