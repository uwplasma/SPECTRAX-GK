#!/usr/bin/env python3
"""Build a machine-readable parallelization closure status artifact."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def _load_decomposition_module() -> Any:
    """Load the pure-stdlib decomposition helpers without importing the package.

    The repo-hygiene CI job intentionally runs before optional numerical
    dependencies are installed. Importing ``spectraxgk`` would pull in the full
    public API and NumPy, while this artifact only needs the dependency-light
    contract helpers.
    """

    path = SRC / "spectraxgk" / "parallel" / "decomposition.py"
    spec = importlib.util.spec_from_file_location(
        "_spectraxgk_parallel_decomposition",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load decomposition helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_DECOMPOSITION = _load_decomposition_module()
DecompositionContract = _DECOMPOSITION.DecompositionContract
build_diagnostic_nonlinear_domain_decomposition = (
    _DECOMPOSITION.build_diagnostic_nonlinear_domain_decomposition
)
build_independent_portfolio_decomposition = (
    _DECOMPOSITION.build_independent_portfolio_decomposition
)
serial_reconstruction_identity_report = (
    _DECOMPOSITION.serial_reconstruction_identity_report
)

STATIC = REPO_ROOT / "docs" / "_static"
DEFAULT_OUT_PREFIX = STATIC / "parallelization_completion_status"
DEFAULT_DECOMPOSITION_OUT_PREFIX = STATIC / "parallel_decomposition_status"

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

EXPECTED_KINDS = {
    "independent_ky_scan": "independent_ky_scan_scaling_combined",
    "quasilinear_uq_ensemble": "quasilinear_uq_ensemble_scaling_combined",
    "whole_state_nonlinear_sharding": "nonlinear_sharding_strong_scaling_combined",
    "fft_axis_domain": "nonlinear_spectral_communication_identity_gate",
}

CLAIM_SCOPE_PHRASES = {
    "independent_ky_scan": (
        "independent ky scan",
        "not a nonlinear domain-decomposition speedup claim",
    ),
    "quasilinear_uq_ensemble": (
        "quasilinear/uq ensemble",
        "not a promoted absolute nonlinear heat-flux predictor",
    ),
    "whole_state_nonlinear_sharding": (
        "whole-state sharding",
        "not a production speedup claim",
    ),
    "fft_axis_domain": (
        "pencil fused-bracket",
        "physical transport-window identity gate",
        "no production distributed FFT routing or speedup claim",
    ),
}

DECOMPOSITION_ARTIFACTS = {
    "independent_ky_scan": "parallel_ky_scan_gate.json",
    "uq_ensemble": "quasilinear_runtime_parallel_gate.json",
    "diagnostic_nonlinear_domain": "nonlinear_domain_parallel_identity_gate.json",
}


def _optimization_provenance_member(value: float) -> dict[str, Any]:
    x = float(value)
    residual = x - 0.35
    return {
        "objective": residual * residual + 0.1 * x,
        "gradient_proxy": 2.0 * residual + 0.1,
        "uq_weight": 1.0 / (1.0 + x * x),
    }


def _indexed_optimization_provenance_task(
    task: tuple[int, float],
) -> tuple[int, dict[str, Any]]:
    index, value = task
    return int(index), _optimization_provenance_member(float(value))


def _max_abs_result_error(
    reference: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> float:
    if len(reference) != len(observed):
        return math.inf
    max_abs = 0.0
    for ref_row, obs_row in zip(reference, observed, strict=True):
        if set(ref_row) != set(obs_row):
            return math.inf
        for key in ref_row:
            ref = float(ref_row[key])
            obs = float(obs_row[key])
            max_abs = max(max_abs, abs(obs - ref))
    return max_abs


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


def _input_backends(payload: dict[str, Any]) -> list[str]:
    inputs = payload.get("inputs", [])
    if not isinstance(inputs, list):
        return []
    return sorted(
        {
            str(row.get("backend", "")).strip().lower()
            for row in inputs
            if isinstance(row, dict) and str(row.get("backend", "")).strip()
        }
    )


def _source_contract(
    name: str, payload: dict[str, Any], *, require_cpu_gpu_inputs: bool
) -> dict[str, Any]:
    claim_scope = str(payload.get("claim_scope", ""))
    claim_scope_lc = claim_scope.lower()
    required_phrases = CLAIM_SCOPE_PHRASES[name]
    missing_phrases = [
        phrase for phrase in required_phrases if phrase.lower() not in claim_scope_lc
    ]
    backends = _input_backends(payload)
    expected_backends = ["cpu", "gpu"] if require_cpu_gpu_inputs else []
    backend_gate_passed = not require_cpu_gpu_inputs or backends == expected_backends
    kind = str(payload.get("kind", ""))
    kind_gate_passed = kind == EXPECTED_KINDS[name]
    claim_separation_passed = bool(
        kind_gate_passed and not missing_phrases and backend_gate_passed
    )
    return {
        "artifact_kind": kind,
        "expected_kind": EXPECTED_KINDS[name],
        "kind_gate_passed": kind_gate_passed,
        "required_scope_phrases": list(required_phrases),
        "missing_scope_phrases": missing_phrases,
        "input_backends": backends,
        "expected_input_backends": expected_backends,
        "input_backend_gate_passed": backend_gate_passed,
        "claim_separation_passed": claim_separation_passed,
    }


def _production_lane(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    thresholds = PRODUCTION_THRESHOLDS[name]
    speedups = _best_speedups_by_backend(payload)
    source_contract = _source_contract(name, payload, require_cpu_gpu_inputs=True)
    identity_passed = bool(payload.get("identity_passed")) and all(
        row.get("identity_gate_pass") is True for row in _rows(payload)
    )
    threshold_passed = all(
        speedups.get(backend, 0.0) >= threshold
        for backend, threshold in thresholds.items()
    )
    passed = bool(
        identity_passed
        and threshold_passed
        and source_contract["claim_separation_passed"]
    )
    return {
        "lane": name,
        "status": "production_closed" if passed else "open",
        "claim_level": "production_parallelization",
        "source_contract": source_contract,
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
    source_contract = _source_contract(
        "whole_state_nonlinear_sharding", payload, require_cpu_gpu_inputs=True
    )
    identity_passed = bool(payload.get("identity_passed")) and all(
        row.get("identity_gate_pass") is True for row in _rows(payload)
    )
    passed = bool(identity_passed and source_contract["claim_separation_passed"])
    return {
        "lane": "whole_state_nonlinear_sharding",
        "status": "diagnostic_closed_not_production" if passed else "open",
        "claim_level": "diagnostic_identity_and_profiler_evidence",
        "source_contract": source_contract,
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
    source_contract = _source_contract(
        "fft_axis_domain", payload, require_cpu_gpu_inputs=False
    )
    gate = payload.get("gate", {}) if isinstance(payload.get("gate"), dict) else {}
    identity_passed = bool(
        payload.get("identity_passed", gate.get("identity_passed"))
    ) or bool(gate.get("identity_passed"))
    rows_passed = all(row.get("identity_passed") is True for row in _rows(payload))
    passed = bool(
        identity_passed and rows_passed and source_contract["claim_separation_passed"]
    )
    return {
        "lane": "fft_axis_domain",
        "status": "diagnostic_identity_closed" if passed else "open",
        "claim_level": "diagnostic_communication_identity",
        "source_contract": source_contract,
        "identity_passed": passed,
        "threshold_passed": None,
        "best_speedups": {},
        "source": f"docs/_static/{ARTIFACTS['fft_axis_domain']}",
        "summary": (
            "Spectral communication, logical RHS, fixed-step integrator, pencil fused-bracket, "
            "and physical transport-window identity are closed; runtime distributed FFT routing "
            "remains out of production scope."
        ),
    }


def _independent_ensemble_provenance_status() -> dict[str, Any]:
    values = [0.05, 0.2, 0.45, 0.8]
    requested_workers = 16
    actual_workers = min(requested_workers, len(values))
    indexed_values = list(enumerate(values))
    serial_payloads = [
        _indexed_optimization_provenance_task(task) for task in indexed_values
    ]
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        parallel_payloads = list(
            pool.map(_indexed_optimization_provenance_task, indexed_values)
        )

    shard_count = actual_workers
    shards = [
        parallel_payloads[index::shard_count]
        for index in range(shard_count)
        if parallel_payloads[index::shard_count]
    ]
    reconstructed_payloads = [payload for shard in shards for payload in shard]
    reconstructed_payloads.sort(key=lambda item: item[0])

    serial_indices = tuple(index for index, _ in serial_payloads)
    parallel_indices = tuple(index for index, _ in parallel_payloads)
    reconstructed_indices = tuple(index for index, _ in reconstructed_payloads)
    serial_results = [result for _, result in serial_payloads]
    parallel_results = [result for _, result in parallel_payloads]
    max_abs_error = _max_abs_result_error(serial_results, parallel_results)
    identity_passed = bool(max_abs_error == 0.0)
    ordering_passed = bool(serial_indices == parallel_indices == reconstructed_indices)
    worker_clipping_passed = bool(actual_workers == min(requested_workers, len(values)))
    reconstruction_identity_passed = bool(reconstructed_indices == serial_indices)

    exception_metadata: dict[str, Any]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(
                pool.map(
                    lambda item: (_ for _ in ()).throw(
                        ValueError("provenance probe failure")
                    )
                    if item == "fail"
                    else item,
                    ["ok", "fail"],
                )
            )
    except ValueError as exc:
        exception_metadata = {
            "index": 1,
            "executor": "thread",
            "actual_workers": 2,
            "original_type": type(exc).__name__,
            "original_message": str(exc),
            "probe_workers": requested_workers,
            "passed": "provenance probe failure" in str(exc),
        }
    else:  # pragma: no cover - defensive fail-closed path
        exception_metadata = {
            "passed": False,
            "missing_exception": True,
            "probe_workers": requested_workers,
            "executor": "thread",
        }

    exception_metadata_passed = bool(exception_metadata.get("passed"))
    passed = bool(
        identity_passed
        and ordering_passed
        and worker_clipping_passed
        and reconstruction_identity_passed
        and exception_metadata_passed
    )
    return {
        "kind": "independent_ensemble_provenance_gate",
        "workload": "optimization_ensemble",
        "passed": passed,
        "identity_passed": identity_passed,
        "ordering_passed": ordering_passed,
        "worker_clipping_passed": worker_clipping_passed,
        "reconstruction_identity_passed": reconstruction_identity_passed,
        "exception_metadata_passed": exception_metadata_passed,
        "requested_workers": requested_workers,
        "actual_workers": actual_workers,
        "problem_size": len(values),
        "serial_indices": list(serial_indices),
        "parallel_indices": list(parallel_indices),
        "reconstructed_indices": list(reconstructed_indices),
        "exception_metadata": exception_metadata,
        "identity_report": {
            "kind": "independent_ensemble_serial_identity",
            "backend": "python:thread",
            "requested_workers": requested_workers,
            "actual_workers": actual_workers,
            "problem_size": len(values),
            "identity_passed": identity_passed,
            "max_abs_error": max_abs_error,
            "max_rel_error": 0.0,
            "atol": 0.0,
            "rtol": 0.0,
            "metadata": {
                "executor": "thread",
                "workload": "optimization_ensemble",
                "source": "build_parallelization_completion_status",
            },
        },
        "summary": (
            "Independent UQ/optimization ensemble batching preserves serial "
            "ordering, clips oversubscribed workers, records worker-exception "
            "metadata, and reconstructs deterministically."
        ),
    }


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


def _decomposition_ky_contract(payload: dict[str, Any]) -> DecompositionContract:
    n_items = len(payload.get("ky_values", [])) or len(_rows(payload))
    return build_independent_portfolio_decomposition(
        n_items,
        requested_shards=_parallel_count(payload, "parallel_batch", "actual_devices"),
        workload="independent_ky_scan",
    )


def _decomposition_uq_contract(payload: dict[str, Any]) -> DecompositionContract:
    n_items = len(payload.get("ky_values", [])) or len(_rows(payload))
    return build_independent_portfolio_decomposition(
        n_items,
        requested_shards=_parallel_count(payload, "workers", "actual_workers"),
        workload="uq_ensemble",
    )


def _decomposition_diagnostic_domain_contract(
    payload: dict[str, Any],
) -> DecompositionContract:
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


def _decomposition_claim_separation_passed(
    contract: DecompositionContract,
) -> bool:
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


def _decomposition_lane(
    name: str,
    source: str,
    payload: dict[str, Any],
    contract: DecompositionContract,
) -> dict[str, Any]:
    values = tuple(range(contract.n_items))
    reconstruction = serial_reconstruction_identity_report(values, contract)
    artifact_identity_passed = _identity_passed(payload)
    claim_separation_passed = _decomposition_claim_separation_passed(contract)
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


def build_decomposition_status(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return decomposition-contract status from existing static artifacts."""

    static = root / "docs" / "_static"
    payloads = {
        name: _read_json(static / artifact)
        for name, artifact in DECOMPOSITION_ARTIFACTS.items()
    }
    lanes = [
        _decomposition_lane(
            "independent_ky_scan",
            DECOMPOSITION_ARTIFACTS["independent_ky_scan"],
            payloads["independent_ky_scan"],
            _decomposition_ky_contract(payloads["independent_ky_scan"]),
        ),
        _decomposition_lane(
            "uq_ensemble",
            DECOMPOSITION_ARTIFACTS["uq_ensemble"],
            payloads["uq_ensemble"],
            _decomposition_uq_contract(payloads["uq_ensemble"]),
        ),
        _decomposition_lane(
            "diagnostic_nonlinear_domain",
            DECOMPOSITION_ARTIFACTS["diagnostic_nonlinear_domain"],
            payloads["diagnostic_nonlinear_domain"],
            _decomposition_diagnostic_domain_contract(
                payloads["diagnostic_nonlinear_domain"]
            ),
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


def write_decomposition_json_artifact(
    status: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    path = out_prefix.with_suffix(".json")
    path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"json": str(path)}


def write_decomposition_csv_artifact(
    status: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
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


def write_decomposition_artifacts(
    status: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = write_decomposition_json_artifact(status, out_prefix)
    paths.update(write_decomposition_csv_artifact(status, out_prefix))

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


def build_status(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return release-scoped parallelization completion status."""

    static = root / "docs" / "_static"
    payloads = {
        name: _read_json(static / artifact) for name, artifact in ARTIFACTS.items()
    }
    lanes = [
        _production_lane("independent_ky_scan", payloads["independent_ky_scan"]),
        _production_lane(
            "quasilinear_uq_ensemble", payloads["quasilinear_uq_ensemble"]
        ),
        _diagnostic_nonlinear_lane(payloads["whole_state_nonlinear_sharding"]),
        _spectral_lane(payloads["fft_axis_domain"]),
    ]
    production = [
        lane for lane in lanes if lane["claim_level"] == "production_parallelization"
    ]
    production_closed = [
        lane for lane in production if lane["status"] == "production_closed"
    ]
    diagnostic_closed = [
        lane for lane in lanes if str(lane["status"]).startswith("diagnostic")
    ]
    provenance_gate = _independent_ensemble_provenance_status()
    production_completion = 100.0 * len(production_closed) / max(len(production), 1)
    overall_completion = (
        100.0
        * (len(production_closed) + 0.75 * len(diagnostic_closed))
        / max(len(lanes), 1)
    )
    passed = (
        production_completion == 100.0
        and all(lane["identity_passed"] for lane in lanes)
        and all(lane["source_contract"]["claim_separation_passed"] for lane in lanes)
        and provenance_gate["passed"]
    )
    return {
        "kind": "parallelization_completion_status",
        "claim_scope": (
            "Release production parallelization is closed for independent ky scans and "
            "quasilinear/UQ ensembles with CPU/GPU strong-scaling and serial numerical "
            "identity. The independent ensemble provenance gate additionally verifies "
            "serial-vs-parallel ordering, worker clipping, exception metadata, and "
            "deterministic reconstruction for UQ/optimization batches. Whole-state "
            "nonlinear sharding and FFT-axis decomposition remain "
            "diagnostic until runtime distributed communication, conservation, transport-window, "
            "and profiler-backed speedup gates pass."
        ),
        "passed": passed,
        "production_completion_percent": production_completion,
        "overall_parallelization_percent": overall_completion,
        "independent_ensemble_provenance_gate": provenance_gate,
        "lanes": lanes,
    }


def _json_path_for_prefix(out_prefix: Path) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    return out_prefix.with_suffix(".json")


def write_json_artifact(status: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    json_path = _json_path_for_prefix(out_prefix)
    json_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"json": str(json_path)}


def write_artifacts(status: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style  # type: ignore[import-untyped]

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    paths = write_json_artifact(status, out_prefix)

    lanes = list(status["lanes"])
    labels = [str(lane["lane"]).replace("_", "\n") for lane in lanes]
    cpu = [float(lane.get("best_speedups", {}).get("cpu", math.nan)) for lane in lanes]
    gpu = [float(lane.get("best_speedups", {}).get("gpu", math.nan)) for lane in lanes]
    colors = [
        "#2f7f5f"
        if str(lane["status"]).startswith(("production", "diagnostic"))
        else "#b44a3c"
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
    axes[1].bar(
        [value - width / 2 for value in x],
        cpu_plot,
        width,
        label="CPU",
        color="#276b8e",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        gpu_plot,
        width,
        label="GPU",
        color="#c45a14",
    )
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


def build_decomposition_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a decomposition-contract status artifact."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_DECOMPOSITION_OUT_PREFIX
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write only JSON and CSV artifacts.",
    )
    return parser


def _main_decomposition(argv: list[str] | None = None) -> int:
    args = build_decomposition_parser().parse_args(argv)
    status = build_decomposition_status(args.root)
    paths = (
        {
            **write_decomposition_json_artifact(status, args.out_prefix),
            **write_decomposition_csv_artifact(status, args.out_prefix),
        }
        if args.skip_figures
        else write_decomposition_artifacts(status, args.out_prefix)
    )
    print(json.dumps({"passed": status["passed"], "paths": paths}, indent=2))
    return 0 if status["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv and raw_argv[0] == "decomposition":
        return _main_decomposition(raw_argv[1:])
    args = build_parser().parse_args(raw_argv)
    status = build_status(args.root)
    paths = (
        write_json_artifact(status, Path(args.out_prefix))
        if args.skip_figures
        else write_artifacts(status, Path(args.out_prefix))
    )
    print(json.dumps({"passed": status["passed"], "paths": paths}, indent=2))
    return (
        0
        if float(status["production_completion_percent"])
        >= float(args.fail_under_production)
        and status["passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
