#!/usr/bin/env python3
"""Fast contract checks for tracked parallel scaling artifacts.

This checker validates the large-run evidence that is already checked into
``docs/_static``. It intentionally does not rerun profilers or assert any
minimum speedup; it verifies artifact completeness, numerical identity gates,
positive timing payloads, and manifest registration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC = REPO_ROOT / "docs" / "_static"
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "performance_optimization_manifest.toml"
SIDE_EXTENSIONS = (".json", ".csv", ".png")
PRODUCTION_GATE_JSON = "nonlinear_sharding_production_speedup_gate.json"
PRODUCTION_GATE_CSV = "nonlinear_sharding_production_speedup_gate.csv"
PRODUCTION_GATE_ARTIFACT_PATHS = (
    f"docs/_static/{PRODUCTION_GATE_JSON}",
    f"docs/_static/{PRODUCTION_GATE_CSV}",
)
PRODUCTION_GATE_SOURCE_FIELDS = (
    "backend",
    "requested_devices",
    "actual_devices",
    "best_spec",
    "state_sharding_active",
    "identity_gate_pass",
    "strong_speedup_vs_1_device",
    "max_abs_state_error",
    "max_rel_state_error",
)
PRODUCTION_GATE_CLASSIFICATIONS = {
    "reference_only",
    "identity_failed",
    "inactive_or_fallback",
    "timing_incomplete",
    "identity_preserving_regression",
    "identity_only_insufficient_speedup",
    "production_candidate",
    "profile_error",
    "diagnostic_only",
}


@dataclass(frozen=True)
class ArtifactFamily:
    name: str
    combined: str
    split: tuple[str, ...]
    expected_combined_kind: str
    expected_split_kind: str | None
    identity_claim_phrase: str
    split_identity_claim_phrase: str | None
    timing_fields: tuple[str, ...]
    error_fields: tuple[str, ...]
    row_identity_key: str = "identity_gate_pass"
    combined_has_inputs: bool = True
    profile_payloads_required: bool = False
    min_rows: int = 1
    min_grid: tuple[tuple[str, int], ...] = ()
    min_steps: int | None = None

    @property
    def json_files(self) -> tuple[str, ...]:
        return (self.combined, *self.split)

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        stems = [Path(name).with_suffix("").name for name in self.json_files]
        return tuple(
            f"docs/_static/{stem}{extension}"
            for stem in stems
            for extension in SIDE_EXTENSIONS
        )


FAMILIES = (
    ArtifactFamily(
        name="independent_ky_scan",
        combined="independent_ky_scan_scaling_large.json",
        split=(
            "independent_ky_scan_scaling_cpu_large.json",
            "independent_ky_scan_scaling_gpu_large.json",
        ),
        expected_combined_kind="independent_ky_scan_scaling_combined",
        expected_split_kind="independent_ky_scan_strong_scaling",
        identity_claim_phrase="not a nonlinear domain-decomposition",
        split_identity_claim_phrase=None,
        timing_fields=("timed_wall_s", "wall_s", "strong_speedup_vs_1_device", "parallel_efficiency"),
        error_fields=("max_gamma_abs_error", "max_gamma_rel_error", "max_omega_abs_error"),
        min_rows=2,
        min_grid=(("Nl", 2), ("Nm", 4), ("Ny", 64), ("Nz", 32)),
        min_steps=100,
    ),
    ArtifactFamily(
        name="quasilinear_uq_ensemble",
        combined="quasilinear_uq_ensemble_scaling_large.json",
        split=(
            "quasilinear_uq_ensemble_scaling_cpu_large.json",
            "quasilinear_uq_ensemble_scaling_gpu_large.json",
        ),
        expected_combined_kind="quasilinear_uq_ensemble_scaling_combined",
        expected_split_kind="quasilinear_uq_ensemble_scaling",
        identity_claim_phrase="not a promoted absolute nonlinear heat-flux predictor",
        split_identity_claim_phrase="not an absolute nonlinear heat-flux validation claim",
        timing_fields=("timed_wall_s", "wall_s", "strong_speedup_vs_1_device", "parallel_efficiency"),
        error_fields=("max_gamma_abs_error", "max_heat_flux_proxy_abs_error", "max_heat_flux_proxy_rel_error"),
        min_rows=2,
        min_grid=(("Nl", 2), ("Nm", 4), ("Ny", 64), ("Nz", 32)),
        min_steps=500,
    ),
    ArtifactFamily(
        name="nonlinear_sharding",
        combined="nonlinear_sharding_strong_scaling_large.json",
        split=(
            "nonlinear_sharding_strong_scaling_cpu_large.json",
            "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
        ),
        expected_combined_kind="nonlinear_sharding_strong_scaling_combined",
        expected_split_kind="nonlinear_sharding_strong_scaling_sweep",
        identity_claim_phrase="not a production speedup claim",
        split_identity_claim_phrase="not as a broad production speedup claim",
        timing_fields=("parallel_median_s", "serial_median_s", "same_process_speedup", "strong_speedup_vs_1_device"),
        error_fields=("max_abs_state_error", "max_rel_state_error"),
        profile_payloads_required=True,
        min_rows=2,
        min_grid=(("Nl", 2), ("Nm", 4), ("Nx", 8), ("Ny", 24), ("Nz", 32)),
        min_steps=1,
    ),
    ArtifactFamily(
        name="linear_rhs_parallel_slices",
        combined="linear_rhs_parallel_slices_sweep.json",
        split=(),
        expected_combined_kind="linear_rhs_parallel_slices_sweep",
        expected_split_kind=None,
        identity_claim_phrase="not a publication speedup claim",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s", "sharded_median_s", "speedup"),
        error_fields=("max_abs_error", "max_rel_error", "max_phi_abs_error"),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
        min_rows=4,
        min_grid=(("Nl", 2), ("Ny", 16), ("Nz", 64)),
    ),
)


def _load_json(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name}: JSON artifact must be an object")
    return payload


def _as_rows(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name}: rows must be a non-empty list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{name}: each row must be an object")
    return rows


def _finite_positive(value: object, field: str, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{context}: {field} must be finite and positive")
    return number


def _finite_nonnegative(value: object, field: str, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{context}: {field} must be finite and non-negative")
    return number


def _path_basename(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return Path(value).name


def _assert_sidecars_exist(root: Path, family: ArtifactFamily) -> int:
    checked = 0
    for artifact in family.artifact_paths:
        path = REPO_ROOT / artifact if root == STATIC else root / Path(artifact).name
        if not path.exists():
            raise ValueError(f"{family.name}: missing sidecar artifact {artifact}")
        checked += 1
    return checked


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return sum(1 for _ in reader)


def _assert_csv_matches_json(root: Path, family: ArtifactFamily, name: str, row_count: int) -> None:
    csv_path = root / f"{Path(name).stem}.csv"
    count = _csv_row_count(csv_path)
    if count != row_count:
        raise ValueError(f"{family.name}: {csv_path.name} has {count} rows, expected {row_count}")


def _assert_claim_scope(payload: dict[str, Any], phrase: str, context: str) -> None:
    claim_scope = payload.get("claim_scope")
    if not isinstance(claim_scope, str) or phrase not in claim_scope:
        raise ValueError(f"{context}: claim_scope must include {phrase!r}")


def _grid_value(grid: dict[str, Any], name: str) -> int | None:
    aliases = {
        "Ny": ("Ny", "Ny_requested"),
        "Nx": ("Nx", "Nx_requested"),
    }
    for key in aliases.get(name, (name,)):
        if key in grid:
            return int(grid[key])
    return None


def _assert_problem_metadata(
    payload: dict[str, Any],
    family: ArtifactFamily,
    context: str,
) -> None:
    """Reject artifacts that lack representative problem-size metadata."""

    rows = _as_rows(payload, context)
    if len(rows) < family.min_rows:
        raise ValueError(f"{context}: expected at least {family.min_rows} rows")
    if family.combined_has_inputs and "inputs" in payload and "grid" not in payload:
        # Combined CPU/GPU summary artifacts carry provenance rows and delegate
        # representative grid/time checks to the split source artifacts.
        return
    if family.min_grid:
        grid = payload.get("grid")
        if not isinstance(grid, dict):
            raise ValueError(f"{context}: grid metadata must be present")
        for field, minimum in family.min_grid:
            value = _grid_value(grid, field)
            if value is None:
                raise ValueError(f"{context}: grid metadata missing {field!r}")
            if value < minimum:
                raise ValueError(
                    f"{context}: grid {field}={value} is below required {minimum}"
                )
    if family.min_steps is not None:
        steps = payload.get("steps")
        time = payload.get("time")
        if steps is None and isinstance(time, dict):
            steps = time.get("steps")
        if steps is None:
            raise ValueError(f"{context}: time-step metadata must be present")
        if int(steps) < family.min_steps:
            raise ValueError(
                f"{context}: steps={int(steps)} is below required {family.min_steps}"
            )


def _assert_identity_payload(
    payload: dict[str, Any],
    family: ArtifactFamily,
    context: str,
    *,
    expected_kind: str,
    claim_phrase: str | None = None,
) -> list[dict[str, Any]]:
    if payload.get("kind") != expected_kind:
        raise ValueError(f"{context}: kind must be {expected_kind!r}")
    if payload.get("identity_passed") is not True:
        raise ValueError(f"{context}: identity_passed must be true")
    _assert_claim_scope(payload, claim_phrase or family.identity_claim_phrase, context)
    _assert_problem_metadata(payload, family, context)

    rows = _as_rows(payload, context)
    seen_requested: set[int] = set()
    for index, row in enumerate(rows):
        row_context = f"{context}: row {index}"
        if row.get("error") is not None:
            raise ValueError(f"{row_context}: error must be null")
        if row.get(family.row_identity_key) is not True:
            raise ValueError(f"{row_context}: {family.row_identity_key} must be true")
        requested = int(row.get("requested_devices", 0))
        if requested < 1:
            raise ValueError(f"{row_context}: requested_devices must be >= 1")
        seen_requested.add(requested)
        actual = row.get("actual_workers", row.get("actual_devices", requested))
        if int(actual) < 1 or int(actual) > requested:
            raise ValueError(f"{row_context}: actual worker/device count is invalid")
        for field in family.timing_fields:
            _finite_positive(row.get(field), field, row_context)
        for field in family.error_fields:
            _finite_nonnegative(row.get(field), field, row_context)
    if 1 not in seen_requested:
        raise ValueError(f"{context}: rows must include a one-device/worker baseline")
    return rows


def _assert_worker_stats(rows: list[dict[str, Any]], context: str) -> None:
    for row in rows:
        workers = row.get("worker_stats")
        if not isinstance(workers, list) or len(workers) != int(row["actual_workers"]):
            raise ValueError(f"{context}: worker_stats must match actual_workers")
        for index, worker in enumerate(workers):
            worker_context = f"{context}: worker {index}"
            if not isinstance(worker, dict):
                raise ValueError(f"{worker_context}: worker_stats entries must be objects")
            samples = worker.get("samples_s")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"{worker_context}: samples_s must be non-empty")
            for sample in samples:
                _finite_positive(sample, "sample", worker_context)
            stats = worker.get("stats_s")
            if not isinstance(stats, dict):
                raise ValueError(f"{worker_context}: stats_s must be an object")
            for field in ("min", "median", "mean", "max"):
                _finite_positive(stats.get(field), field, worker_context)
            _finite_nonnegative(stats.get("std"), "std", worker_context)


def _assert_profile_payloads(payload: dict[str, Any], rows: list[dict[str, Any]], context: str) -> None:
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"{context}: profiles must be an object")
    requested_keys = {str(row["requested_devices"]) for row in rows}
    if set(profiles) != requested_keys:
        raise ValueError(f"{context}: profiles must match requested_devices rows")
    for row in rows:
        profile = profiles[str(row["requested_devices"])]
        if not isinstance(profile, dict):
            raise ValueError(f"{context}: profile entries must be objects")
        if profile.get("_profile_json") != row.get("profile_json"):
            raise ValueError(f"{context}: profile_json pointer mismatch")
        if profile.get("identity_gate_pass") is not True:
            raise ValueError(f"{context}: embedded profile identity gate must pass")
        _assert_claim_scope(profile, "Do not use as a published runtime claim", context)
        trace = profile.get("profiler_trace")
        if not isinstance(trace, dict) or trace.get("error") is not None:
            raise ValueError(f"{context}: profiler_trace must be present and error-free")


def _nonlinear_sharding_split_artifacts() -> tuple[str, ...]:
    for family in FAMILIES:
        if family.name == "nonlinear_sharding":
            return family.split
    raise AssertionError("nonlinear_sharding artifact family is not configured")


def _production_gate_candidate_key(row: dict[str, Any]) -> tuple[str, int, int, str, float]:
    return (
        str(row.get("backend", "")).lower(),
        int(row.get("requested_devices") or 0),
        int(row.get("actual_devices") or 0),
        _path_basename(row.get("source")),
        float(row.get("strong_speedup_vs_1_device")),
    )


def _production_gate_source_row_key(row: dict[str, Any], source_name: str) -> tuple[str, str, int, int]:
    return (
        source_name,
        str(row.get("backend", "")).lower(),
        int(row.get("requested_devices") or 0),
        int(row.get("actual_devices") or 0),
    )


def _assert_production_gate_row_matches_source(
    row: dict[str, Any],
    *,
    index: int,
    source_rows: dict[tuple[str, str, int, int], dict[str, Any]],
) -> None:
    source_name = _path_basename(row.get("source"))
    key = _production_gate_source_row_key(row, source_name)
    source = source_rows.get(key)
    if source is None:
        raise ValueError(
            f"{PRODUCTION_GATE_JSON}: row {index}: no matching source row in {source_name}"
        )
    for field in PRODUCTION_GATE_SOURCE_FIELDS:
        if row.get(field) != source.get(field):
            raise ValueError(
                f"{PRODUCTION_GATE_JSON}: row {index}: {field} is stale relative "
                f"to source artifact {source_name}"
            )


def _load_production_gate_source_rows(root: Path) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    source_rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for source_name in _nonlinear_sharding_split_artifacts():
        payload = _load_json(root, source_name)
        for index, row in enumerate(_as_rows(payload, source_name)):
            key = _production_gate_source_row_key(row, source_name)
            if key in source_rows:
                raise ValueError(
                    f"{source_name}: duplicate source row for backend/device key at row {index}"
                )
            source_rows[key] = row
    return source_rows


def _production_gate_source_artifacts_exist(root: Path) -> bool:
    return all((root / source_name).exists() for source_name in _nonlinear_sharding_split_artifacts())


def validate_nonlinear_sharding_production_gate(
    root: Path,
    *,
    check_sidecars: bool = True,
) -> dict[str, Any]:
    """Validate the fail-closed nonlinear sharding promotion gate artifact."""

    root = root.resolve()
    sidecars = 0
    if check_sidecars:
        for artifact in PRODUCTION_GATE_ARTIFACT_PATHS:
            path = REPO_ROOT / artifact if root == STATIC else root / Path(artifact).name
            if not path.exists():
                raise ValueError(
                    f"nonlinear_sharding_production_gate: missing sidecar artifact {artifact}"
                )
            sidecars += 1

    payload = _load_json(root, PRODUCTION_GATE_JSON)
    context = PRODUCTION_GATE_JSON
    if payload.get("kind") != "nonlinear_sharding_production_speedup_gate":
        raise ValueError(f"{context}: kind must be 'nonlinear_sharding_production_speedup_gate'")
    _assert_claim_scope(payload, "Whole-state nonlinear sharding", context)
    _assert_claim_scope(payload, "diagnostic identity/profiler artifact", context)

    gate_passed = payload.get("gate_passed")
    if not isinstance(gate_passed, bool):
        raise ValueError(f"{context}: gate_passed must be boolean")
    if payload.get("production_speedup_claim_allowed") is not gate_passed:
        raise ValueError(
            f"{context}: production_speedup_claim_allowed must match gate_passed"
        )
    expected_status = "production_speedup_candidate" if gate_passed else "diagnostic_only"
    if payload.get("status") != expected_status:
        raise ValueError(f"{context}: status must be {expected_status!r}")

    required_backends = payload.get("required_backends")
    if not isinstance(required_backends, list) or not required_backends:
        raise ValueError(f"{context}: required_backends must be a non-empty list")
    required = tuple(str(backend).lower() for backend in required_backends)
    if len(set(required)) != len(required):
        raise ValueError(f"{context}: required_backends must not contain duplicates")

    min_devices = int(payload.get("min_devices", 0))
    min_speedup = _finite_positive(
        payload.get("min_speedup_vs_1_device"),
        "min_speedup_vs_1_device",
        context,
    )
    min_efficiency = _finite_positive(
        payload.get("min_parallel_efficiency"),
        "min_parallel_efficiency",
        context,
    )
    identity_atol = _finite_nonnegative(payload.get("identity_atol"), "identity_atol", context)
    identity_rtol = _finite_nonnegative(payload.get("identity_rtol"), "identity_rtol", context)
    rows = _as_rows(payload, context)

    if check_sidecars:
        count = _csv_row_count(root / PRODUCTION_GATE_CSV)
        if count != len(rows):
            raise ValueError(
                f"nonlinear_sharding_production_gate: {PRODUCTION_GATE_CSV} has "
                f"{count} rows, expected {len(rows)}"
            )

    source_artifacts = set(_nonlinear_sharding_split_artifacts())
    source_rows = (
        _load_production_gate_source_rows(root)
        if check_sidecars or _production_gate_source_artifacts_exist(root)
        else {}
    )
    candidates_by_backend: dict[str, list[dict[str, Any]]] = {backend: [] for backend in required}
    observed_backends: set[str] = set()
    for index, row in enumerate(rows):
        row_context = f"{context}: row {index}"
        backend = str(row.get("backend", "")).lower()
        if backend not in required:
            raise ValueError(f"{row_context}: backend {backend!r} is not required")
        observed_backends.add(backend)
        source_name = _path_basename(row.get("source"))
        if source_name not in source_artifacts:
            raise ValueError(f"{row_context}: source must be a nonlinear sharding split artifact")
        if source_rows:
            _assert_production_gate_row_matches_source(
                row,
                index=index,
                source_rows=source_rows,
            )
        requested_devices = int(row.get("requested_devices") or 0)
        actual_devices = int(row.get("actual_devices") or 0)
        if requested_devices < 1 or actual_devices < 1 or actual_devices > requested_devices:
            raise ValueError(f"{row_context}: device counts are invalid")

        blockers = row.get("blockers")
        if not isinstance(blockers, list):
            raise ValueError(f"{row_context}: blockers must be a list")
        candidate_passed = row.get("candidate_passed")
        if not isinstance(candidate_passed, bool):
            raise ValueError(f"{row_context}: candidate_passed must be boolean")
        if candidate_passed != (not blockers):
            raise ValueError(
                f"{row_context}: candidate_passed must be true exactly when blockers are empty"
            )

        classification = str(row.get("classification", ""))
        if classification not in PRODUCTION_GATE_CLASSIFICATIONS:
            raise ValueError(f"{row_context}: unknown classification {classification!r}")
        if candidate_passed and classification != "production_candidate":
            raise ValueError(
                f"{row_context}: passing candidates must be classified as production_candidate"
            )
        if classification == "production_candidate" and not candidate_passed:
            raise ValueError(f"{row_context}: production_candidate rows must pass")

        max_abs = _finite_nonnegative(row.get("max_abs_state_error"), "max_abs_state_error", row_context)
        max_rel = _finite_nonnegative(row.get("max_rel_state_error"), "max_rel_state_error", row_context)
        speedup = _finite_positive(
            row.get("strong_speedup_vs_1_device"),
            "strong_speedup_vs_1_device",
            row_context,
        )
        efficiency = _finite_positive(row.get("parallel_efficiency"), "parallel_efficiency", row_context)
        identity_passed = bool(row.get("identity_gate_pass", False))
        if identity_passed and (max_abs > identity_atol or max_rel > identity_rtol):
            raise ValueError(f"{row_context}: identity errors exceed gate tolerances")
        if candidate_passed:
            if not identity_passed:
                raise ValueError(f"{row_context}: passing candidate must pass identity")
            if not bool(row.get("state_sharding_active", False)):
                raise ValueError(f"{row_context}: passing candidate must use active sharding")
            if actual_devices < min_devices:
                raise ValueError(f"{row_context}: passing candidate is below min_devices")
            if speedup < min_speedup:
                raise ValueError(f"{row_context}: passing candidate is below min_speedup")
            if efficiency < min_efficiency:
                raise ValueError(f"{row_context}: passing candidate is below min_parallel_efficiency")
            candidates_by_backend[backend].append(row)

    missing_backend_rows = sorted(set(required) - observed_backends)
    if missing_backend_rows:
        raise ValueError(f"{context}: missing rows for backend(s): {', '.join(missing_backend_rows)}")

    best_candidates = payload.get("best_candidates")
    if not isinstance(best_candidates, dict):
        raise ValueError(f"{context}: best_candidates must be an object")
    gate_blockers = list(payload.get("blockers", []))
    if not all(isinstance(blocker, str) for blocker in gate_blockers):
        raise ValueError(f"{context}: blockers must be strings")

    expected_gate_blockers: list[str] = []
    for backend in required:
        candidates = candidates_by_backend.get(backend, [])
        best = best_candidates.get(backend)
        if not candidates:
            expected_gate_blockers.append(f"{backend}_production_speedup_candidate_missing")
            if best is not None:
                raise ValueError(f"{context}: best_candidates[{backend!r}] must be null")
            continue
        if not isinstance(best, dict):
            raise ValueError(f"{context}: best_candidates[{backend!r}] must be an object")
        expected_best = max(
            candidates,
            key=lambda item: (
                float(item["strong_speedup_vs_1_device"]),
                int(item["requested_devices"]),
            ),
        )
        if _production_gate_candidate_key(best) != _production_gate_candidate_key(expected_best):
            raise ValueError(f"{context}: best candidate for {backend} is stale")

    if gate_blockers != expected_gate_blockers:
        raise ValueError(f"{context}: blockers do not match missing backend candidates")
    expected_gate_passed = not expected_gate_blockers
    if gate_passed != expected_gate_passed:
        raise ValueError(f"{context}: gate_passed is inconsistent with candidate rows")

    return {
        "name": "nonlinear_sharding_production_speedup_gate",
        "json": PRODUCTION_GATE_JSON,
        "n_rows": len(rows),
        "n_sidecars": sidecars,
        "required_backends": list(required),
        "gate_passed": gate_passed,
        "production_speedup_claim_allowed": bool(payload["production_speedup_claim_allowed"]),
        "status": payload["status"],
        "blockers": gate_blockers,
        "production_candidate_backends": [
            backend for backend in required if candidates_by_backend.get(backend)
        ],
    }


def validate_family(root: Path, family: ArtifactFamily, *, check_sidecars: bool = True) -> dict[str, Any]:
    """Validate one artifact family under ``root`` and return a compact summary."""

    root = root.resolve()
    sidecars = _assert_sidecars_exist(root, family) if check_sidecars else 0
    combined = _load_json(root, family.combined)
    rows = _assert_identity_payload(
        combined,
        family,
        family.combined,
        expected_kind=family.expected_combined_kind,
    )
    _assert_csv_matches_json(root, family, family.combined, len(rows))

    if family.combined_has_inputs:
        inputs = combined.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{family.combined}: inputs must list split source artifacts")
        input_names = {_path_basename(entry.get("path")) for entry in inputs if isinstance(entry, dict)}
        source_names = {_path_basename(row.get("source")) for row in rows}
        expected_names = set(family.split)
        if input_names != expected_names:
            raise ValueError(f"{family.combined}: inputs do not match expected split artifacts")
        if source_names != expected_names:
            raise ValueError(f"{family.combined}: row sources do not match expected split artifacts")
        for entry in inputs:
            if not isinstance(entry, dict) or entry.get("identity_passed") is not True:
                raise ValueError(f"{family.combined}: each split input must pass identity")

    backends: set[str] = set()
    split_rows = 0
    for split_name in family.split:
        split = _load_json(root, split_name)
        assert family.expected_split_kind is not None
        rows = _assert_identity_payload(
            split,
            family,
            split_name,
            expected_kind=family.expected_split_kind,
            claim_phrase=family.split_identity_claim_phrase,
        )
        _assert_csv_matches_json(root, family, split_name, len(rows))
        if family.name in {"independent_ky_scan", "quasilinear_uq_ensemble"}:
            _assert_worker_stats(rows, split_name)
        if family.profile_payloads_required:
            _assert_profile_payloads(split, rows, split_name)
        backend = split.get("backend")
        if not isinstance(backend, str) or not backend:
            raise ValueError(f"{split_name}: backend must be a non-empty string")
        backends.add(backend)
        split_rows += len(rows)

    return {
        "name": family.name,
        "combined": family.combined,
        "n_combined_rows": len(_as_rows(combined, family.combined)),
        "n_split_rows": split_rows,
        "n_sidecars": sidecars,
        "backends": sorted(backends),
    }


def _manifest_parallel_artifacts(manifest: Path) -> set[str]:
    with manifest.open("rb") as stream:
        data = tomllib.load(stream)
    lanes = data.get("lanes")
    if not isinstance(lanes, list):
        raise ValueError(f"{manifest}: lanes must be a list")
    for lane in lanes:
        if isinstance(lane, dict) and lane.get("name") == "parallel_scaling":
            paths = lane.get("artifact_paths")
            if not isinstance(paths, list):
                raise ValueError("parallel_scaling artifact_paths must be a list")
            return {path for path in paths if isinstance(path, str)}
    raise ValueError("performance manifest must contain a parallel_scaling lane")


def validate_all(
    *,
    root: Path = STATIC,
    manifest: Path = DEFAULT_MANIFEST,
    check_manifest: bool = True,
    check_sidecars: bool = True,
) -> dict[str, Any]:
    """Validate all tracked parallel scaling families."""

    summaries = [validate_family(root, family, check_sidecars=check_sidecars) for family in FAMILIES]
    production_gate = validate_nonlinear_sharding_production_gate(
        root,
        check_sidecars=check_sidecars,
    )

    missing_from_manifest: list[str] = []
    if check_manifest:
        manifest_paths = _manifest_parallel_artifacts(manifest)
        required = {path for family in FAMILIES for path in family.artifact_paths}
        required.update(PRODUCTION_GATE_ARTIFACT_PATHS)
        missing_from_manifest = sorted(required - manifest_paths)
        if missing_from_manifest:
            raise ValueError(
                "parallel_scaling manifest is missing artifacts: "
                + ", ".join(missing_from_manifest)
            )

    return {
        "n_families": len(summaries),
        "n_json_artifacts": sum(1 + len(family.split) for family in FAMILIES) + 1,
        "n_sidecars": sum(summary["n_sidecars"] for summary in summaries)
        + int(production_gate["n_sidecars"]),
        "manifest_checked": check_manifest,
        "missing_from_manifest": missing_from_manifest,
        "families": summaries,
        "production_gate": production_gate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=STATIC, help="Directory containing scaling artifacts.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--skip-manifest-check", action="store_true")
    parser.add_argument("--skip-sidecar-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_all(
        root=args.root,
        manifest=args.manifest,
        check_manifest=not args.skip_manifest_check,
        check_sidecars=not args.skip_sidecar_check,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out_json is None:
        print(payload, end="")
    else:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
