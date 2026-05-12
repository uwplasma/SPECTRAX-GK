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
SIDE_EXTENSIONS = (".json", ".csv", ".png", ".pdf")


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

    missing_from_manifest: list[str] = []
    if check_manifest:
        manifest_paths = _manifest_parallel_artifacts(manifest)
        required = {path for family in FAMILIES for path in family.artifact_paths}
        missing_from_manifest = sorted(required - manifest_paths)
        if missing_from_manifest:
            raise ValueError(
                "parallel_scaling manifest is missing artifacts: "
                + ", ".join(missing_from_manifest)
            )

    return {
        "n_families": len(summaries),
        "n_json_artifacts": sum(1 + len(family.split) for family in FAMILIES),
        "n_sidecars": sum(summary["n_sidecars"] for summary in summaries),
        "manifest_checked": check_manifest,
        "missing_from_manifest": missing_from_manifest,
        "families": summaries,
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
