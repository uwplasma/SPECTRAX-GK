#!/usr/bin/env python3
"""Validate the package architecture policy manifest.

The checker prevents new root-level prefix modules from appearing while the
codebase migrates toward domain packages. Existing prefix modules are allowed
only when they are explicitly listed as temporary migration scaffolding.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "package_architecture_manifest.toml"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "src" / "spectraxgk"


def _as_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _as_nonempty_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return _as_string_list(value, field)


def _as_string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings")
        result.append(item.strip())
    duplicates = sorted({item for item in result if result.count(item) > 1})
    if duplicates:
        raise ValueError(f"{field} contains duplicate entries: {duplicates}")
    return result


def _as_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _as_nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _repo_path(raw: str) -> Path:
    return (REPO_ROOT / raw).resolve()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load an architecture TOML manifest."""

    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a TOML table")
    return data


def _root_module_path(module: str, source_root: Path) -> Path:
    if not module.startswith("spectraxgk."):
        raise ValueError(f"root-prefix module must start with spectraxgk.: {module}")
    remainder = module.removeprefix("spectraxgk.")
    if "." in remainder:
        raise ValueError(f"root-prefix allowlist entries must be root modules: {module}")
    return source_root / f"{remainder}.py"


def _package_path(package: str, source_root: Path) -> Path:
    if not package.startswith("spectraxgk."):
        raise ValueError(f"required package must start with spectraxgk.: {package}")
    remainder = package.removeprefix("spectraxgk.")
    return source_root.joinpath(*remainder.split(".")) / "__init__.py"


def _count_matching_files(path: Path, *, pattern: str, recursive: bool) -> int:
    """Count files for topology gates, treating deleted targets as zero."""

    if not path.exists():
        return 0
    iterator = path.rglob(pattern) if recursive else path.glob(pattern)
    return sum(1 for item in iterator if item.is_file())


def _validate_topology_policy(
    data: dict[str, Any],
    *,
    require_targets: bool = False,
) -> list[dict[str, Any]]:
    topology = data.get("topology_policy")
    if topology is None:
        return []
    if not isinstance(topology, dict):
        raise ValueError("topology_policy must be a TOML table")
    mode = _as_nonempty_string(topology.get("mode"), "topology_policy.mode")
    if mode != "no_regression_until_target":
        raise ValueError(
            "topology_policy.mode must be 'no_regression_until_target'"
        )
    _as_nonempty_string(
        topology.get("description"),
        "topology_policy.description",
    )
    counts = topology.get("counts")
    if not isinstance(counts, list) or not counts:
        raise ValueError("topology_policy.counts must be a non-empty list")

    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(counts):
        if not isinstance(entry, dict):
            raise ValueError(f"topology_policy.counts[{index}] must be a table")
        prefix = f"topology_policy.counts[{index}]"
        name = _as_nonempty_string(entry.get("name"), f"{prefix}.name")
        raw_path = _as_nonempty_string(entry.get("path"), f"{prefix}.path")
        pattern = _as_nonempty_string(entry.get("pattern"), f"{prefix}.pattern")
        recursive = _as_bool(entry.get("recursive"), f"{prefix}.recursive")
        baseline = _as_nonnegative_int(entry.get("baseline"), f"{prefix}.baseline")
        target = _as_nonnegative_int(entry.get("target"), f"{prefix}.target")
        if target > baseline:
            raise ValueError(
                f"{name}: topology target {target} cannot exceed baseline {baseline}"
            )
        count = _count_matching_files(
            _repo_path(raw_path),
            pattern=pattern,
            recursive=recursive,
        )
        if count > baseline:
            raise ValueError(
                f"{name}: topology count regressed to {count}, above baseline {baseline}"
            )
        if require_targets and count > target:
            raise ValueError(
                f"{name}: topology target not met; count {count} exceeds target {target}"
            )
        rows.append(
            {
                "name": name,
                "path": raw_path,
                "pattern": pattern,
                "recursive": recursive,
                "count": count,
                "baseline": baseline,
                "target": target,
                "remaining_to_target": max(0, count - target),
                "target_met": count <= target,
            }
        )
    return rows


def validate_architecture_policy(
    data: dict[str, Any],
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    check_paths: bool = True,
    require_topology_targets: bool = False,
) -> dict[str, Any]:
    """Validate architecture-policy content and return a compact summary."""

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manifest must contain [metadata]")
    if metadata.get("schema_version") != 1:
        raise ValueError("metadata.schema_version must be 1")
    for field in ("title", "layout_authority", "status"):
        _as_nonempty_string(metadata.get(field), f"metadata.{field}")
    if check_paths:
        authority = _repo_path(str(metadata["layout_authority"]))
        if not authority.is_file():
            raise ValueError(f"layout authority does not exist: {authority}")

    root_policy = data.get("root_prefix_policy")
    if not isinstance(root_policy, dict):
        raise ValueError("manifest must contain [root_prefix_policy]")
    blocked_prefixes = _as_nonempty_list(
        root_policy.get("blocked_prefixes"),
        "root_prefix_policy.blocked_prefixes",
    )
    allowed_modules = _as_string_list(
        root_policy.get("allowed_root_prefix_modules"),
        "root_prefix_policy.allowed_root_prefix_modules",
    )
    allowed_set = set(allowed_modules)

    root_modules: list[str] = []
    for path in sorted(source_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = f"spectraxgk.{path.stem}"
        if path.stem.startswith(tuple(blocked_prefixes)):
            root_modules.append(module)

    new_modules = sorted(set(root_modules) - allowed_set)
    stale_allowlist = sorted(allowed_set - set(root_modules))
    if new_modules:
        raise ValueError(
            "root-level prefix modules must be moved into domain packages or "
            f"listed as temporary facades: {new_modules}"
        )
    if stale_allowlist:
        raise ValueError(
            "root-prefix allowlist contains modules that no longer exist; "
            f"remove them from the manifest: {stale_allowlist}"
        )
    if check_paths:
        for module in allowed_modules:
            path = _root_module_path(module, source_root)
            if not path.is_file():
                raise ValueError(f"allowed root-prefix module source does not exist: {module}")

    package_policy = data.get("package_policy")
    if not isinstance(package_policy, dict):
        raise ValueError("manifest must contain [package_policy]")
    required_packages = _as_nonempty_list(
        package_policy.get("required_domain_packages"),
        "package_policy.required_domain_packages",
    )
    required_docs = _as_nonempty_list(
        package_policy.get("required_docs"),
        "package_policy.required_docs",
    )
    if check_paths:
        for package in required_packages:
            path = _package_path(package, source_root)
            if not path.is_file():
                raise ValueError(f"required domain package is missing __init__.py: {package}")
        for doc in required_docs:
            path = _repo_path(doc)
            if not path.is_file():
                raise ValueError(f"required architecture doc does not exist: {doc}")

    topology_counts = _validate_topology_policy(
        data,
        require_targets=require_topology_targets,
    )

    return {
        "layout_authority": str(metadata["layout_authority"]),
        "n_blocked_prefixes": len(blocked_prefixes),
        "n_allowed_root_prefix_modules": len(allowed_modules),
        "n_current_root_prefix_modules": len(root_modules),
        "n_required_domain_packages": len(required_packages),
        "required_domain_packages": required_packages,
        "n_topology_counts": len(topology_counts),
        "topology_counts": topology_counts,
        "topology_targets_met": all(row["target_met"] for row in topology_counts),
        "status": str(metadata["status"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--require-topology-targets",
        action="store_true",
        help="Fail unless all topology counts are already at their final targets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_architecture_policy(
        load_manifest(args.manifest),
        source_root=args.source_root,
        check_paths=True,
        require_topology_targets=args.require_topology_targets,
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
