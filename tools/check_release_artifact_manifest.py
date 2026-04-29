#!/usr/bin/env python3
"""Validate the release-artifact manifest for large tracked assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tools/release_artifact_manifest.toml"
VALID_ACTIONS = {"keep_in_repo", "move_to_release"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    policy = data.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{path} must contain a [policy] table")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{path} must contain at least one [[artifacts]] table")
    seen: set[str] = set()
    for idx, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ValueError(f"{path} artifacts[{idx}] must be a table")
        for key in ("path", "size_bytes", "sha256", "action", "reason", "replay_command"):
            if key not in item:
                raise ValueError(f"{path} artifacts[{idx}] is missing {key!r}")
        artifact_path = item["path"]
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError(f"{path} artifacts[{idx}].path must be a non-empty string")
        if artifact_path in seen:
            raise ValueError(f"{path} lists duplicate artifact path {artifact_path!r}")
        seen.add(artifact_path)
        if not isinstance(item["size_bytes"], int) or int(item["size_bytes"]) <= 0:
            raise ValueError(f"{path} artifacts[{idx}].size_bytes must be a positive integer")
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise ValueError(f"{path} artifacts[{idx}].sha256 must be a hex sha256 string")
        if item["action"] not in VALID_ACTIONS:
            raise ValueError(f"{path} artifacts[{idx}].action must be one of {sorted(VALID_ACTIONS)}")
        if item["action"] == "move_to_release":
            for key in ("release_asset_name", "preview_strategy"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"{path} artifacts[{idx}] move_to_release entries need {key!r}")
    return data


def check_release_artifact_manifest(
    *,
    root: str | Path = ROOT,
    manifest: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Return a JSON-ready validation report for the release-artifact manifest."""

    root_path = Path(root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root_path / manifest_path
    data = _load_manifest(manifest_path)
    rows = []
    failures = []
    total_move_bytes = 0
    total_keep_bytes = 0
    for item in data["artifacts"]:
        rel = Path(item["path"])
        path = root_path / rel
        exists = path.exists()
        actual_size = path.stat().st_size if exists else None
        actual_sha = _sha256(path) if exists and path.is_file() else None
        action = str(item["action"])
        if action == "move_to_release" and actual_size is not None:
            total_move_bytes += int(actual_size)
        if action == "keep_in_repo" and actual_size is not None:
            total_keep_bytes += int(actual_size)
        row = {
            "path": item["path"],
            "action": action,
            "exists": exists,
            "expected_size_bytes": item["size_bytes"],
            "actual_size_bytes": actual_size,
            "expected_sha256": item["sha256"],
            "actual_sha256": actual_sha,
            "release_asset_name": item.get("release_asset_name"),
            "preview_strategy": item.get("preview_strategy"),
        }
        rows.append(row)
        if not exists:
            failures.append(f"{item['path']} does not exist")
            continue
        if actual_size != int(item["size_bytes"]):
            failures.append(f"{item['path']} size {actual_size} != manifest size {item['size_bytes']}")
        if actual_sha != item["sha256"]:
            failures.append(f"{item['path']} sha256 {actual_sha} != manifest sha256 {item['sha256']}")

    return {
        "kind": "release_artifact_manifest_check",
        "manifest": str(manifest_path),
        "passed": not failures,
        "failures": failures,
        "policy": data["policy"],
        "artifact_count": len(rows),
        "move_to_release_bytes": total_move_bytes,
        "keep_in_repo_bytes": total_keep_bytes,
        "artifacts": rows,
        "notes": (
            "This validates provenance for large tracked assets. It does not upload or "
            "delete files; release migration should happen in a separate explicit step."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repository root to check.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Release artifact manifest TOML.")
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_release_artifact_manifest(root=args.root, manifest=args.manifest)
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
        "release artifact manifest check passed: artifacts={count} move_to_release_bytes={bytes}".format(
            count=report["artifact_count"],
            bytes=report["move_to_release_bytes"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
