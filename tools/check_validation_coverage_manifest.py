#!/usr/bin/env python3
"""Validate the refactor/validation coverage traceability manifest.

The manifest is intentionally stricter than a planning document.  Each critical
module must map to reference anchors, physics/numerics contracts, fast tests,
artifacts, and next tests so the 95% coverage target remains tied to useful
scientific and numerical checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "validation_coverage_manifest.toml"

ALLOWED_STATUSES = {"closed", "active", "open", "planned"}
ALLOWED_PRIORITIES = {"high", "medium", "low"}
REQUIRED_STRING_FIELDS = (
    "module",
    "path",
    "owner_lane",
    "status",
    "coverage_priority",
)
REQUIRED_LIST_FIELDS = (
    "reference_anchors",
    "physics_contracts",
    "numerics_contracts",
    "fast_tests",
    "next_tests",
)
OPTIONAL_LIST_FIELDS = ("owned_modules",)


def _repo_path(raw: str) -> Path:
    return (REPO_ROOT / raw).resolve()


def _module_to_source_path(module: str) -> Path:
    return _repo_path("src/" + module.replace(".", "/") + ".py")


def _source_path_to_module(path: Path) -> str:
    rel = path.relative_to((REPO_ROOT / "src").resolve()).with_suffix("")
    return ".".join(rel.parts)


def _coverage_filename_to_module(filename: str) -> str | None:
    normalized = filename.replace("\\", "/").lstrip("./")
    marker = "src/spectraxgk/"
    if marker in normalized:
        normalized = "spectraxgk/" + normalized.split(marker, 1)[1]
    if not normalized.startswith("spectraxgk/") or not normalized.endswith(".py"):
        return None
    return normalized[:-3].replace("/", ".")


def _validate_fast_test_path(resolved: Path, raw: str, module: str) -> None:
    """Require manifest fast tests to be discoverable by pytest."""

    tests_root = (REPO_ROOT / "tests").resolve()
    if not resolved.exists():
        raise ValueError(f"{module}: fast test does not exist: {raw}")
    if not resolved.is_file():
        raise ValueError(f"{module}: fast test must be a file: {raw}")
    try:
        rel = resolved.relative_to(tests_root)
    except ValueError as exc:
        raise ValueError(f"{module}: fast test must live under tests/: {raw}") from exc
    if not rel.name.startswith("test_") or rel.suffix != ".py":
        raise ValueError(
            f"{module}: fast test must be a tests/**/test_*.py file "
            f"discoverable by pytest and run_wide_coverage_gate.py: {raw}"
        )


def _as_nonempty_string(value: object, field: str, module: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{module}: {field} must be a non-empty string")
    return value.strip()


def _as_nonempty_list(value: object, field: str, module: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{module}: {field} must be a non-empty list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{module}: {field} entries must be non-empty strings")
        items.append(item.strip())
    return items


def _reject_duplicate_values(values: list[str], field: str, module: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{module}: {field} contains duplicate entries: {duplicates}")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load a TOML manifest as a dictionary."""

    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a TOML table")
    return data


def _coverage_xml_summary(
    coverage_xml: Path,
    *,
    package_modules: set[str],
    target_by_module: dict[str, float],
    owner_by_module: dict[str, str],
    package_target: float,
) -> dict[str, Any]:
    """Return package and module coverage from a Cobertura XML report."""

    tree = ET.parse(coverage_xml)
    root = tree.getroot()
    try:
        package_coverage_percent = float(root.attrib["line-rate"]) * 100.0
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{coverage_xml}: missing numeric coverage line-rate") from exc

    measured: dict[str, float] = {}
    for cls in root.findall(".//class"):
        module = _coverage_filename_to_module(cls.attrib.get("filename", ""))
        if module is None or module not in package_modules:
            continue
        if module in measured:
            raise ValueError(f"{coverage_xml}: duplicate coverage entry for {module}")
        try:
            measured[module] = float(cls.attrib["line-rate"]) * 100.0
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{coverage_xml}: invalid line-rate for {module}") from exc

    tracked_modules = set(target_by_module)
    module_rows: list[dict[str, Any]] = []
    for module in sorted(tracked_modules):
        coverage = measured.get(module)
        target = float(target_by_module[module])
        passed = coverage is not None and coverage >= target
        module_rows.append(
            {
                "module": module,
                "owner": owner_by_module.get(module, module),
                "coverage_percent": coverage,
                "target_percent": target,
                "passed": bool(passed),
            }
        )

    missing_measured_modules = sorted(
        module for module in tracked_modules if module not in measured
    )
    modules_below_target = [
        row["module"]
        for row in module_rows
        if row["coverage_percent"] is not None
        and row["coverage_percent"] < row["target_percent"]
    ]

    return {
        "coverage_xml": str(coverage_xml),
        "package_coverage_percent": package_coverage_percent,
        "package_coverage_target_percent": float(package_target),
        "package_coverage_passed": bool(
            package_coverage_percent >= float(package_target)
        ),
        "n_measured_modules": len(measured),
        "n_tracked_modules": len(tracked_modules),
        "n_missing_measured_modules": len(missing_measured_modules),
        "missing_measured_modules": missing_measured_modules,
        "n_modules_below_target": len(modules_below_target),
        "modules_below_target": modules_below_target,
        "module_rows": module_rows,
    }


def validate_manifest(
    data: dict[str, Any],
    *,
    check_artifacts: bool = True,
    coverage_xml: Path | None = None,
    enforce_package_coverage: bool = False,
    enforce_module_coverage: bool = False,
) -> dict[str, Any]:
    """Validate manifest content and return a compact summary."""

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manifest must contain a [metadata] table")
    target = metadata.get("package_coverage_target_percent")
    if not isinstance(target, (float, int)) or not (0.0 < float(target) <= 100.0):
        raise ValueError("metadata.package_coverage_target_percent must be in (0, 100]")

    modules = data.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("manifest must contain at least one [[modules]] entry")

    inventory = data.get("coverage_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("manifest must contain a [coverage_inventory] table")
    require_all_owned = inventory.get("require_all_package_modules_owned")
    if require_all_owned is not True:
        raise ValueError(
            "coverage_inventory.require_all_package_modules_owned must be true"
        )
    excluded_module_list = _as_nonempty_list(
        inventory.get("excluded_modules"),
        "excluded_modules",
        "coverage_inventory",
    )
    _reject_duplicate_values(
        excluded_module_list,
        "excluded_modules",
        "coverage_inventory",
    )
    excluded_modules = set(excluded_module_list)
    for module in excluded_modules:
        if not module.startswith("spectraxgk."):
            raise ValueError(
                f"coverage_inventory: excluded module must start with 'spectraxgk.': {module}"
            )
        if not _module_to_source_path(module).exists():
            raise ValueError(
                f"coverage_inventory: excluded module source does not exist: {module}"
            )

    seen_modules: set[str] = set()
    rows: list[dict[str, Any]] = []
    owned_modules_by_owner: dict[str, list[str]] = {}
    module_owners: dict[str, str] = {}
    target_by_module: dict[str, float] = {}
    for entry in modules:
        if not isinstance(entry, dict):
            raise ValueError("each [[modules]] entry must be a TOML table")
        module = _as_nonempty_string(entry.get("module"), "module", "<unknown>")
        if module in seen_modules:
            raise ValueError(f"{module}: duplicate module entry")
        seen_modules.add(module)
        if not module.startswith("spectraxgk."):
            raise ValueError(f"{module}: module must start with 'spectraxgk.'")

        strings = {
            field: _as_nonempty_string(entry.get(field), field, module)
            for field in REQUIRED_STRING_FIELDS
        }
        lists = {
            field: _as_nonempty_list(entry.get(field), field, module)
            for field in REQUIRED_LIST_FIELDS
        }
        for field, values in lists.items():
            _reject_duplicate_values(values, field, module)
        optional_lists = {
            field: _as_nonempty_list(entry[field], field, module)
            for field in OPTIONAL_LIST_FIELDS
            if field in entry
        }
        for field, values in optional_lists.items():
            _reject_duplicate_values(values, field, module)
        status = strings["status"]
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{module}: invalid status {status!r}")
        priority = strings["coverage_priority"]
        if priority not in ALLOWED_PRIORITIES:
            raise ValueError(f"{module}: invalid coverage_priority {priority!r}")
        coverage = entry.get("coverage_target_percent")
        if not isinstance(coverage, (float, int)) or not (
            0.0 < float(coverage) <= 100.0
        ):
            raise ValueError(f"{module}: coverage_target_percent must be in (0, 100]")
        target_by_module[module] = float(coverage)

        source_path = _repo_path(strings["path"])
        if not source_path.exists():
            raise ValueError(f"{module}: source path does not exist: {strings['path']}")
        expected_source_path = _module_to_source_path(module)
        if source_path != expected_source_path:
            raise ValueError(
                f"{module}: path must match module import path "
                f"{expected_source_path.relative_to(REPO_ROOT)}"
            )

        owned_modules = optional_lists.get("owned_modules", [])
        owned_modules_by_owner[module] = owned_modules
        for owned_module in owned_modules:
            if not owned_module.startswith("spectraxgk."):
                raise ValueError(
                    f"{module}: owned module must start with 'spectraxgk.': {owned_module}"
                )
            if owned_module in seen_modules:
                raise ValueError(
                    f"{module}: owned_modules must not include direct manifest row: {owned_module}"
                )
            if owned_module in excluded_modules:
                raise ValueError(
                    f"{module}: owned_modules must not include excluded module: {owned_module}"
                )
            if not _module_to_source_path(owned_module).exists():
                raise ValueError(
                    f"{module}: owned module source does not exist: {owned_module}"
                )
            previous_owner = module_owners.setdefault(owned_module, module)
            if previous_owner != module:
                raise ValueError(
                    f"{owned_module}: duplicate coverage ownership by {previous_owner} and {module}"
                )
            target_by_module[owned_module] = float(coverage)

        for test_path in lists["fast_tests"]:
            resolved = _repo_path(test_path)
            _validate_fast_test_path(resolved, test_path, module)

        artifacts = _as_nonempty_list(
            entry.get("artifact_paths"), "artifact_paths", module
        )
        _reject_duplicate_values(artifacts, "artifact_paths", module)
        if check_artifacts:
            for artifact in artifacts:
                resolved_artifact = _repo_path(artifact)
                if not resolved_artifact.exists():
                    raise ValueError(
                        f"{module}: artifact path does not exist: {artifact}"
                    )
                if not resolved_artifact.is_file():
                    raise ValueError(
                        f"{module}: artifact path must be a file: {artifact}"
                    )

        rows.append(
            {
                "module": module,
                "path": strings["path"],
                "status": status,
                "coverage_priority": priority,
                "coverage_target_percent": float(coverage),
                "n_reference_anchors": len(lists["reference_anchors"]),
                "n_physics_contracts": len(lists["physics_contracts"]),
                "n_numerics_contracts": len(lists["numerics_contracts"]),
                "n_fast_tests": len(lists["fast_tests"]),
                "n_artifacts": len(artifacts),
                "n_next_tests": len(lists["next_tests"]),
                "n_owned_modules": len(owned_modules),
            }
        )

    package_modules = {
        _source_path_to_module(path)
        for path in (REPO_ROOT / "src" / "spectraxgk").rglob("*.py")
        if path.is_file()
    }
    direct_modules = seen_modules
    owned_modules = set(module_owners)
    directly_owned_modules = sorted(direct_modules & owned_modules)
    if directly_owned_modules:
        raise ValueError(
            f"direct manifest rows must not be listed as owned modules: {directly_owned_modules}"
        )
    unowned_modules = sorted(
        package_modules - direct_modules - owned_modules - excluded_modules
    )
    stale_owned_modules = sorted(
        (direct_modules | owned_modules | excluded_modules) - package_modules
    )
    if stale_owned_modules:
        raise ValueError(
            f"manifest references missing package modules: {stale_owned_modules}"
        )
    if unowned_modules:
        raise ValueError(f"package modules lack coverage ownership: {unowned_modules}")

    high_priority_open = [
        row["module"]
        for row in rows
        if row["coverage_priority"] == "high"
        and row["status"] in {"active", "open", "planned"}
    ]
    summary = {
        "package_coverage_target_percent": float(target),
        "n_modules": len(rows),
        "n_high_priority_open": len(high_priority_open),
        "high_priority_open": high_priority_open,
        "n_direct_modules": len(direct_modules),
        "n_owned_modules": len(owned_modules),
        "n_excluded_modules": len(excluded_modules),
        "n_package_modules": len(package_modules),
        "owned_modules_by_owner": owned_modules_by_owner,
        "rows": rows,
    }
    if coverage_xml is not None:
        coverage_summary = _coverage_xml_summary(
            coverage_xml,
            package_modules=package_modules,
            target_by_module=target_by_module,
            owner_by_module=module_owners,
            package_target=float(target),
        )
        summary["coverage_xml_summary"] = coverage_summary
        if enforce_package_coverage and not coverage_summary["package_coverage_passed"]:
            raise ValueError(
                "package coverage below manifest target: "
                f"{coverage_summary['package_coverage_percent']:.2f}% < {float(target):.2f}%"
            )
        if enforce_module_coverage:
            failures = list(coverage_summary["missing_measured_modules"]) + list(
                coverage_summary["modules_below_target"]
            )
            if failures:
                raise ValueError(
                    f"module coverage below manifest target or missing: {failures}"
                )
    elif enforce_package_coverage or enforce_module_coverage:
        raise ValueError("coverage enforcement requires --coverage-xml")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--skip-artifact-check",
        action="store_true",
        help="Validate schema and tests without requiring artifact files to exist.",
    )
    parser.add_argument(
        "--coverage-xml",
        type=Path,
        default=None,
        help="Optional Cobertura XML report used to attach measured package/module coverage.",
    )
    parser.add_argument(
        "--enforce-package-coverage",
        action="store_true",
        help="Fail if the XML package line-rate is below metadata.package_coverage_target_percent.",
    )
    parser.add_argument(
        "--enforce-module-coverage",
        action="store_true",
        help="Fail if tracked direct/owned modules are missing or below their manifest target.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_manifest(
        load_manifest(args.manifest),
        check_artifacts=not args.skip_artifact_check,
        coverage_xml=args.coverage_xml,
        enforce_package_coverage=args.enforce_package_coverage,
        enforce_module_coverage=args.enforce_module_coverage,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out_json}")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
