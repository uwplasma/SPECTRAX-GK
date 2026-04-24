#!/usr/bin/env python3
"""Validate the CPU/GPU profiling and optimization manifest."""

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
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "performance_optimization_manifest.toml"
ALLOWED_STATUSES = {"closed", "active", "open", "planned"}
ALLOWED_PRIORITIES = {"high", "medium", "low"}
REQUIRED_STRING_FIELDS = ("name", "owner", "status", "priority")
REQUIRED_LIST_FIELDS = (
    "platforms",
    "cases",
    "profiling_tools",
    "metrics",
    "artifact_paths",
    "bottleneck_hypotheses",
    "optimization_actions",
    "gates",
)


def _repo_path(raw: str) -> Path:
    return (REPO_ROOT / raw).resolve()


def _as_nonempty_string(value: object, field: str, lane: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{lane}: {field} must be a non-empty string")
    return value.strip()


def _as_nonempty_list(value: object, field: str, lane: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{lane}: {field} must be a non-empty list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{lane}: {field} entries must be non-empty strings")
        items.append(item.strip())
    return items


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load the profiling/optimization TOML manifest."""

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
    schema = metadata.get("schema_version")
    if schema != 1:
        raise ValueError("metadata.schema_version must be 1")

    lanes = data.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("manifest must contain at least one [[lanes]] entry")

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for entry in lanes:
        if not isinstance(entry, dict):
            raise ValueError("each [[lanes]] entry must be a TOML table")
        name = _as_nonempty_string(entry.get("name"), "name", "<unknown>")
        if name in seen:
            raise ValueError(f"{name}: duplicate lane entry")
        seen.add(name)

        strings = {field: _as_nonempty_string(entry.get(field), field, name) for field in REQUIRED_STRING_FIELDS}
        lists = {field: _as_nonempty_list(entry.get(field), field, name) for field in REQUIRED_LIST_FIELDS}
        status = strings["status"]
        priority = strings["priority"]
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{name}: invalid status {status!r}")
        if priority not in ALLOWED_PRIORITIES:
            raise ValueError(f"{name}: invalid priority {priority!r}")

        for tool in lists["profiling_tools"]:
            resolved = _repo_path(tool)
            if not resolved.exists():
                raise ValueError(f"{name}: profiling tool does not exist: {tool}")
            try:
                resolved.relative_to((REPO_ROOT / "tools").resolve())
            except ValueError as exc:
                raise ValueError(f"{name}: profiling tool must live under tools/: {tool}") from exc

        if check_artifacts:
            for artifact in lists["artifact_paths"]:
                if not _repo_path(artifact).exists():
                    raise ValueError(f"{name}: artifact path does not exist: {artifact}")

        rows.append(
            {
                "name": name,
                "owner": strings["owner"],
                "status": status,
                "priority": priority,
                "n_platforms": len(lists["platforms"]),
                "n_cases": len(lists["cases"]),
                "n_tools": len(lists["profiling_tools"]),
                "n_metrics": len(lists["metrics"]),
                "n_artifacts": len(lists["artifact_paths"]),
                "n_hypotheses": len(lists["bottleneck_hypotheses"]),
                "n_actions": len(lists["optimization_actions"]),
                "n_gates": len(lists["gates"]),
            }
        )

    high_priority_active = [
        row["name"]
        for row in rows
        if row["priority"] == "high" and row["status"] in {"active", "open", "planned"}
    ]
    return {
        "n_lanes": len(rows),
        "n_high_priority_active": len(high_priority_active),
        "high_priority_active": high_priority_active,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--skip-artifact-check",
        action="store_true",
        help="Validate schema and tools without requiring artifact files to exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_manifest(load_manifest(args.manifest), check_artifacts=not args.skip_artifact_check)
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
