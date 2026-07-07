#!/usr/bin/env python3
"""Check tracked repository size against the artifact-hygiene manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

try:
    from .audit_repository_size import ROOT, build_repository_size_report
except ImportError:  # pragma: no cover - direct script execution
    from audit_repository_size import ROOT, build_repository_size_report


DEFAULT_MANIFEST = ROOT / "tools/repository_size_manifest.toml"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    policy = data.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{path} must contain a [policy] table")
    for key in ("max_tracked_total_bytes", "max_unlisted_tracked_file_bytes"):
        value = policy.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path} policy.{key} must be a positive integer")
    allowed = data.get("allowed_large_files", [])
    if not isinstance(allowed, list):
        raise ValueError(f"{path} allowed_large_files must be an array of tables")
    paths: set[str] = set()
    for idx, item in enumerate(allowed):
        if not isinstance(item, dict):
            raise ValueError(f"{path} allowed_large_files[{idx}] must be a table")
        item_path = item.get("path")
        max_bytes = item.get("max_bytes")
        reason = item.get("reason")
        if not isinstance(item_path, str) or not item_path:
            raise ValueError(f"{path} allowed_large_files[{idx}].path must be a non-empty string")
        if item_path in paths:
            raise ValueError(f"{path} lists duplicate allowed_large_files path {item_path!r}")
        paths.add(item_path)
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError(f"{path} allowed_large_files[{idx}].max_bytes must be a positive integer")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path} allowed_large_files[{idx}].reason must be a non-empty string")
    return data


def check_repository_size_manifest(
    *,
    root: str | Path = ROOT,
    manifest: str | Path = DEFAULT_MANIFEST,
    top_n: int = 50,
) -> dict[str, Any]:
    """Return a JSON-ready pass/fail report for tracked repository size policy."""

    root_path = Path(root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root_path / manifest_path
    data = _load_manifest(manifest_path)
    policy = data["policy"]
    allowed = {item["path"]: item for item in data.get("allowed_large_files", [])}
    audit = build_repository_size_report(root_path, top_n=max(top_n, len(allowed) + 10))

    failures: list[str] = []
    if int(audit["tracked_total_bytes"]) > int(policy["max_tracked_total_bytes"]):
        failures.append(
            "tracked_total_bytes={actual} exceeds max_tracked_total_bytes={limit}".format(
                actual=audit["tracked_total_bytes"],
                limit=policy["max_tracked_total_bytes"],
            )
        )

    tracked = {row["path"]: int(row["bytes"]) for row in audit["largest_tracked_files"]}
    if len(tracked) < int(audit["tracked_file_count"]):
        for row in build_repository_size_report(root_path, top_n=int(audit["tracked_file_count"]))["largest_tracked_files"]:
            tracked[row["path"]] = int(row["bytes"])

    for path, item in allowed.items():
        if path not in tracked:
            failures.append(f"allowed large file {path!r} is not tracked")
            continue
        if tracked[path] > int(item["max_bytes"]):
            failures.append(f"{path} has {tracked[path]} bytes, exceeding allowed max_bytes={item['max_bytes']}")

    threshold = int(policy["max_unlisted_tracked_file_bytes"])
    unlisted_large = [
        {"path": path, "bytes": size}
        for path, size in sorted(tracked.items(), key=lambda item: item[1], reverse=True)
        if size > threshold and path not in allowed
    ]
    for item in unlisted_large:
        failures.append(
            "{path} has {bytes} bytes, exceeding max_unlisted_tracked_file_bytes={threshold} and is not whitelisted".format(
                path=item["path"],
                bytes=item["bytes"],
                threshold=threshold,
            )
        )

    return {
        "kind": "repository_size_manifest_check",
        "manifest": str(manifest_path),
        "passed": not failures,
        "failures": failures,
        "policy": policy,
        "allowed_large_files": list(allowed.values()),
        "unlisted_large_files": unlisted_large,
        "audit": audit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repository root to check.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Repository size manifest TOML.")
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_repository_size_manifest(root=args.root, manifest=args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {out}")
    else:
        print(text)
    if not report["passed"]:
        for failure in report["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "repository size check passed: tracked_total_bytes={bytes}".format(
            bytes=report["audit"]["tracked_total_bytes"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
