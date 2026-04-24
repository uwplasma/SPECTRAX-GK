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


def _repo_path(raw: str) -> Path:
    return (REPO_ROOT / raw).resolve()


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


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load a TOML manifest as a dictionary."""

    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a TOML table")
    return data


def validate_manifest(data: dict[str, Any], *, check_artifacts: bool = True) -> dict[str, Any]:
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

    seen_modules: set[str] = set()
    rows: list[dict[str, Any]] = []
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
        status = strings["status"]
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{module}: invalid status {status!r}")
        priority = strings["coverage_priority"]
        if priority not in ALLOWED_PRIORITIES:
            raise ValueError(f"{module}: invalid coverage_priority {priority!r}")
        coverage = entry.get("coverage_target_percent")
        if not isinstance(coverage, (float, int)) or not (0.0 < float(coverage) <= 100.0):
            raise ValueError(f"{module}: coverage_target_percent must be in (0, 100]")

        source_path = _repo_path(strings["path"])
        if not source_path.exists():
            raise ValueError(f"{module}: source path does not exist: {strings['path']}")

        for test_path in lists["fast_tests"]:
            resolved = _repo_path(test_path)
            if not resolved.exists():
                raise ValueError(f"{module}: fast test does not exist: {test_path}")
            try:
                resolved.relative_to((REPO_ROOT / "tests").resolve())
            except ValueError as exc:
                raise ValueError(f"{module}: fast test must live under tests/: {test_path}") from exc

        artifacts = _as_nonempty_list(entry.get("artifact_paths"), "artifact_paths", module)
        if check_artifacts:
            for artifact in artifacts:
                if not _repo_path(artifact).exists():
                    raise ValueError(f"{module}: artifact path does not exist: {artifact}")

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
            }
        )

    high_priority_open = [
        row["module"]
        for row in rows
        if row["coverage_priority"] == "high" and row["status"] in {"active", "open", "planned"}
    ]
    return {
        "package_coverage_target_percent": float(target),
        "n_modules": len(rows),
        "n_high_priority_open": len(high_priority_open),
        "high_priority_open": high_priority_open,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--skip-artifact-check",
        action="store_true",
        help="Validate schema and tests without requiring artifact files to exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_manifest(load_manifest(args.manifest), check_artifacts=not args.skip_artifact_check)
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
