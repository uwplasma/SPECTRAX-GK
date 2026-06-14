#!/usr/bin/env python3
"""Validate the differentiable architecture refactor manifest.

This checker keeps the large refactor plan executable. It does not prove that
future refactors are done; it proves that every planned high-risk split declares
source ownership, compatibility facade, tests, parity gates, literature anchors,
autodiff gates, and extension points before implementation starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "differentiable_refactor_manifest.toml"
ALLOWED_STATUSES = {"planned", "active", "closed", "deferred"}
REQUIRED_HOTSPOT_FIELDS = (
    "module",
    "source_path",
    "current_reason",
    "target_packages",
    "public_facade",
    "line_target",
    "status",
    "existing_fast_tests",
    "planned_tests",
    "parity_gates",
    "literature_anchors",
    "autodiff_gates",
    "extension_points",
)
REQUIRED_LAYER_FIELDS = (
    "name",
    "responsibility",
    "planned_packages",
    "extension_points",
)
REQUIRED_CONTRACT_MODULE_FIELDS = (
    "module",
    "source_path",
    "status",
    "responsibility",
    "public_exports",
    "fast_tests",
    "doc_pages",
)


def _repo_path(raw: str) -> Path:
    return (REPO_ROOT / raw).resolve()


def _as_nonempty_string(value: object, field: str, row: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{row}: {field} must be a non-empty string")
    return value.strip()


def _as_nonempty_list(value: object, field: str, row: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{row}: {field} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{row}: {field} entries must be non-empty strings")
        result.append(item.strip())
    if len(set(result)) != len(result):
        raise ValueError(f"{row}: {field} contains duplicate entries")
    return result


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load a TOML manifest as a dictionary."""

    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a TOML table")
    return data


def _validate_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manifest must contain [metadata]")
    if metadata.get("schema_version") != 1:
        raise ValueError("metadata.schema_version must be 1")
    for field in ("title", "owner_lane", "status", "compatibility_policy"):
        _as_nonempty_string(metadata.get(field), field, "metadata")
    status = metadata["status"]
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"metadata.status must be one of {sorted(ALLOWED_STATUSES)}")
    for field in ("max_public_module_lines_target", "max_internal_module_lines_target"):
        value = metadata.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"metadata.{field} must be a positive integer")
    return metadata


def _validate_global_acceptance(data: dict[str, Any]) -> dict[str, Any]:
    acceptance = data.get("global_acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("manifest must contain [global_acceptance]")
    coverage = acceptance.get("required_package_coverage_percent")
    if not isinstance(coverage, (int, float)) or not 0 < float(coverage) <= 100:
        raise ValueError("global_acceptance.required_package_coverage_percent must be in (0, 100]")
    budget = acceptance.get("max_fast_test_budget_seconds")
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError("global_acceptance.max_fast_test_budget_seconds must be positive")
    for field in (
        "require_public_api_facades",
        "require_reference_adapter_isolation",
        "require_jax_transform_contracts",
        "require_docstring_policy",
    ):
        if acceptance.get(field) is not True:
            raise ValueError(f"global_acceptance.{field} must be true")
    return acceptance


def _validate_validation_policy(data: dict[str, Any]) -> None:
    policy = data.get("validation_policy")
    if not isinstance(policy, dict):
        raise ValueError("manifest must contain [validation_policy]")
    for field in (
        "parity_reference_scope",
        "literature_gate_scope",
        "autodiff_gate_scope",
        "performance_gate_scope",
    ):
        _as_nonempty_string(policy.get(field), field, "validation_policy")


def _validate_layers(data: dict[str, Any]) -> list[dict[str, Any]]:
    layers = data.get("architecture_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("manifest must contain [[architecture_layers]] entries")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in layers:
        if not isinstance(raw, dict):
            raise ValueError("architecture layer entries must be TOML tables")
        name = _as_nonempty_string(raw.get("name"), "name", "architecture_layers")
        if name in seen:
            raise ValueError(f"duplicate architecture layer: {name}")
        seen.add(name)
        for field in REQUIRED_LAYER_FIELDS:
            if field not in raw:
                raise ValueError(f"architecture layer {name}: missing {field}")
        _as_nonempty_string(raw.get("responsibility"), "responsibility", name)
        _as_nonempty_list(raw.get("planned_packages"), "planned_packages", name)
        _as_nonempty_list(raw.get("extension_points"), "extension_points", name)
        validated.append(raw)
    return validated


def _validate_phase1_contract_modules(data: dict[str, Any]) -> list[dict[str, Any]]:
    contract_modules = data.get("phase1_contract_modules")
    if not isinstance(contract_modules, list) or not contract_modules:
        raise ValueError("manifest must contain [[phase1_contract_modules]] entries")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in contract_modules:
        if not isinstance(raw, dict):
            raise ValueError("phase1 contract module entries must be TOML tables")
        for field in REQUIRED_CONTRACT_MODULE_FIELDS:
            if field not in raw:
                module = raw.get("module", "<unknown>")
                raise ValueError(f"{module}: missing required phase1 field {field}")
        module = _as_nonempty_string(raw.get("module"), "module", "phase1_contract_modules")
        if module in seen:
            raise ValueError(f"duplicate phase1 contract module: {module}")
        seen.add(module)
        if not module.startswith("spectraxgk."):
            raise ValueError(f"{module}: module must start with spectraxgk.")
        source_path = _as_nonempty_string(raw.get("source_path"), "source_path", module)
        source = _repo_path(source_path)
        if not source.exists() or not source.is_file():
            raise ValueError(f"{module}: source path does not exist: {source_path}")
        status = _as_nonempty_string(raw.get("status"), "status", module)
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{module}: status must be one of {sorted(ALLOWED_STATUSES)}")
        _as_nonempty_string(raw.get("responsibility"), "responsibility", module)
        for field in ("public_exports", "fast_tests", "doc_pages"):
            _as_nonempty_list(raw.get(field), field, module)
        for test_path in raw["fast_tests"]:
            test = _repo_path(test_path)
            if not test.exists() or not test.is_file():
                raise ValueError(f"{module}: fast test does not exist: {test_path}")
        for doc_page in raw["doc_pages"]:
            doc = _repo_path(doc_page)
            if not doc.exists() or not doc.is_file():
                raise ValueError(f"{module}: doc page does not exist: {doc_page}")
        validated.append(raw)
    return validated


def _validate_hotspots(data: dict[str, Any]) -> list[dict[str, Any]]:
    hotspots = data.get("hotspots")
    if not isinstance(hotspots, list) or not hotspots:
        raise ValueError("manifest must contain [[hotspots]] entries")
    seen_modules: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in hotspots:
        if not isinstance(raw, dict):
            raise ValueError("hotspot entries must be TOML tables")
        for field in REQUIRED_HOTSPOT_FIELDS:
            if field not in raw:
                module = raw.get("module", "<unknown>")
                raise ValueError(f"{module}: missing required field {field}")
        module = _as_nonempty_string(raw.get("module"), "module", "hotspots")
        if module in seen_modules:
            raise ValueError(f"duplicate hotspot module: {module}")
        seen_modules.add(module)
        if not module.startswith("spectraxgk."):
            raise ValueError(f"{module}: module must start with spectraxgk.")
        source_path = _as_nonempty_string(raw.get("source_path"), "source_path", module)
        resolved_source = _repo_path(source_path)
        if not resolved_source.exists():
            raise ValueError(f"{module}: source path does not exist: {source_path}")
        if not resolved_source.is_file():
            raise ValueError(f"{module}: source path must be a file: {source_path}")
        public_facade = _as_nonempty_string(raw.get("public_facade"), "public_facade", module)
        if public_facade != module:
            raise ValueError(f"{module}: public_facade must match module for compatibility planning")
        line_target = raw.get("line_target")
        if not isinstance(line_target, int) or line_target <= 0:
            raise ValueError(f"{module}: line_target must be a positive integer")
        status = _as_nonempty_string(raw.get("status"), "status", module)
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{module}: status must be one of {sorted(ALLOWED_STATUSES)}")
        for field in (
            "target_packages",
            "existing_fast_tests",
            "planned_tests",
            "parity_gates",
            "literature_anchors",
            "autodiff_gates",
            "extension_points",
        ):
            _as_nonempty_list(raw.get(field), field, module)
        for test_path in raw["existing_fast_tests"]:
            resolved_test = _repo_path(test_path)
            if not resolved_test.exists() or not resolved_test.is_file():
                raise ValueError(f"{module}: existing fast test does not exist: {test_path}")
            try:
                resolved_test.relative_to((REPO_ROOT / "tests").resolve())
            except ValueError as exc:
                raise ValueError(f"{module}: existing fast test must live under tests/: {test_path}") from exc
        validated.append(raw)
    return validated


def validate_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the manifest and return a summary dictionary."""

    metadata = _validate_metadata(data)
    acceptance = _validate_global_acceptance(data)
    _validate_validation_policy(data)
    layers = _validate_layers(data)
    contract_modules = _validate_phase1_contract_modules(data)
    hotspots = _validate_hotspots(data)
    high_risk_modules = {row["module"] for row in hotspots}
    required_hotspots = {
        "spectraxgk.benchmarks",
        "spectraxgk.geometry.differentiable",
        "spectraxgk.nonlinear_parallel",
        "spectraxgk.solver_objective_gradients",
        "spectraxgk.nonlinear",
        "spectraxgk.runtime_artifacts",
        "spectraxgk.runtime",
        "spectraxgk.linear",
        "spectraxgk.cli",
    }
    missing = sorted(required_hotspots - high_risk_modules)
    if missing:
        raise ValueError(f"manifest missing required high-risk hotspots: {missing}")
    return {
        "title": metadata["title"],
        "status": metadata["status"],
        "required_package_coverage_percent": float(acceptance["required_package_coverage_percent"]),
        "n_architecture_layers": len(layers),
        "n_phase1_contract_modules": len(contract_modules),
        "n_hotspots": len(hotspots),
        "phase1_contract_modules": sorted(row["module"] for row in contract_modules),
        "hotspot_modules": sorted(high_risk_modules),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(argv)
    summary = validate_manifest(load_manifest(args.manifest))
    if args.out_json:
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
