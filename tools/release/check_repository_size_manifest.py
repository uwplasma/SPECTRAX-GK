#!/usr/bin/env python3
"""Check tracked repository size against the artifact-hygiene manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tools/repository_size_manifest.toml"
DEFAULT_RELEASE_ARTIFACT_MANIFEST = ROOT / "tools/release_artifact_manifest.toml"
VALID_RELEASE_ARTIFACT_ACTIONS = {
    "keep_in_repo",
    "keep_preview_in_repo",
    "move_to_release",
}
LOCAL_ARTIFACT_ROOTS = (
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "docs/_build",
    "tools_out",
)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _tracked_paths(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    raw = proc.stdout.split(b"\0")
    return [Path(item.decode()) for item in raw if item]


def _category(path: Path) -> str:
    parts = path.parts
    if not parts:
        return "."
    if len(parts) >= 2 and parts[0] == "docs" and parts[1] == "_static":
        return "docs/_static"
    if len(parts) >= 2 and parts[0] == "docs" and parts[1] == "_build":
        return "docs/_build"
    return parts[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_repository_size_report(
    root: str | Path = ROOT, *, top_n: int = 30
) -> dict[str, Any]:
    """Return a JSON-ready size report for tracked files and local artifact roots."""

    root_path = Path(root).resolve()
    rows = []
    by_category: dict[str, int] = defaultdict(int)
    for rel in _tracked_paths(root_path):
        path = root_path / rel
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rows.append({"path": str(rel), "bytes": int(size), "category": _category(rel)})
        by_category[_category(rel)] += int(size)

    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    local_roots = []
    for rel in LOCAL_ARTIFACT_ROOTS:
        path = root_path / rel
        local_roots.append(
            {"path": rel, "bytes": int(_directory_size(path)), "exists": path.exists()}
        )

    return {
        "kind": "repository_size_audit",
        "root": str(root_path),
        "tracked_total_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "tracked_file_count": len(rows),
        "tracked_by_category": dict(
            sorted(by_category.items(), key=lambda item: item[1], reverse=True)
        ),
        "largest_tracked_files": rows[:top_n],
        "local_artifact_roots": local_roots,
        "notes": (
            "This report is non-destructive. Move large tracked validation artifacts to "
            "GitHub Releases or another artifact store before considering a coordinated "
            "history rewrite."
        ),
    }


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
            raise ValueError(
                f"{path} allowed_large_files[{idx}].path must be a non-empty string"
            )
        if item_path in paths:
            raise ValueError(
                f"{path} lists duplicate allowed_large_files path {item_path!r}"
            )
        paths.add(item_path)
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError(
                f"{path} allowed_large_files[{idx}].max_bytes must be a positive integer"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"{path} allowed_large_files[{idx}].reason must be a non-empty string"
            )
    return data


def _load_release_artifact_manifest(path: Path) -> dict[str, Any]:
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
        for key in (
            "path",
            "size_bytes",
            "sha256",
            "action",
            "reason",
            "replay_command",
        ):
            if key not in item:
                raise ValueError(f"{path} artifacts[{idx}] is missing {key!r}")
        artifact_path = item["path"]
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError(f"{path} artifacts[{idx}].path must be a non-empty string")
        if artifact_path in seen:
            raise ValueError(f"{path} lists duplicate artifact path {artifact_path!r}")
        seen.add(artifact_path)
        if not isinstance(item["size_bytes"], int) or int(item["size_bytes"]) <= 0:
            raise ValueError(
                f"{path} artifacts[{idx}].size_bytes must be a positive integer"
            )
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise ValueError(
                f"{path} artifacts[{idx}].sha256 must be a hex sha256 string"
            )
        if item["action"] not in VALID_RELEASE_ARTIFACT_ACTIONS:
            raise ValueError(
                f"{path} artifacts[{idx}].action must be one of {sorted(VALID_RELEASE_ARTIFACT_ACTIONS)}"
            )
        if item["action"] in {"move_to_release", "keep_preview_in_repo"}:
            for key in ("release_asset_name", "preview_strategy"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(
                        f"{path} artifacts[{idx}] {item['action']} entries need {key!r}"
                    )
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
        for row in build_repository_size_report(
            root_path, top_n=int(audit["tracked_file_count"])
        )["largest_tracked_files"]:
            tracked[row["path"]] = int(row["bytes"])

    for path, item in allowed.items():
        if path not in tracked:
            failures.append(f"allowed large file {path!r} is not tracked")
            continue
        if tracked[path] > int(item["max_bytes"]):
            failures.append(
                f"{path} has {tracked[path]} bytes, exceeding allowed max_bytes={item['max_bytes']}"
            )

    threshold = int(policy["max_unlisted_tracked_file_bytes"])
    unlisted_large = [
        {"path": path, "bytes": size}
        for path, size in sorted(
            tracked.items(), key=lambda item: item[1], reverse=True
        )
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


def check_release_artifact_manifest(
    *,
    root: str | Path = ROOT,
    manifest: str | Path = DEFAULT_RELEASE_ARTIFACT_MANIFEST,
) -> dict[str, Any]:
    """Return a JSON-ready validation report for the release-artifact manifest."""

    root_path = Path(root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root_path / manifest_path
    data = _load_release_artifact_manifest(manifest_path)
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
        if action == "move_to_release":
            total_move_bytes += int(item["size_bytes"])
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
            "release_tag": item.get("release_tag"),
            "release_url": item.get("release_url"),
        }
        rows.append(row)
        if not exists:
            release_url = item.get("release_url")
            release_tag = item.get("release_tag")
            if (
                action == "move_to_release"
                and isinstance(release_url, str)
                and release_url.strip()
            ):
                if not isinstance(release_tag, str) or not release_tag.strip():
                    failures.append(
                        f"{item['path']} moved to release but is missing release_tag"
                    )
                continue
            failures.append(f"{item['path']} does not exist")
            continue
        if actual_size != int(item["size_bytes"]):
            failures.append(
                f"{item['path']} size {actual_size} != manifest size {item['size_bytes']}"
            )
        if actual_sha != item["sha256"]:
            failures.append(
                f"{item['path']} sha256 {actual_sha} != manifest sha256 {item['sha256']}"
            )

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
            "This validates provenance for large tracked assets. Existing files are "
            "checked against size and sha256. move_to_release entries may be absent "
            "from Git only when the manifest records release_tag and release_url."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repository root to check.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Repository size manifest TOML.",
    )
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def build_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit tracked file sizes and local artifact roots without modifying the repo."
    )
    parser.add_argument("--root", default=str(ROOT), help="Repository root to audit.")
    parser.add_argument(
        "--top", type=int, default=30, help="Number of tracked files to list."
    )
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def build_release_artifact_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the release-artifact manifest for large tracked assets."
    )
    parser.add_argument("--root", default=str(ROOT), help="Repository root to check.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_RELEASE_ARTIFACT_MANIFEST),
        help="Release artifact manifest TOML.",
    )
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def main_audit(argv: list[str] | None = None) -> int:
    args = build_audit_parser().parse_args(argv)
    report = build_repository_size_report(args.root, top_n=args.top)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {out}")
    else:
        print(text)
    return 0


def main_release_artifacts(argv: list[str] | None = None) -> int:
    args = build_release_artifact_parser().parse_args(argv)
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


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "audit":
        return main_audit(tokens[1:])
    if tokens and tokens[0] == "release-artifacts":
        return main_release_artifacts(tokens[1:])

    args = build_parser().parse_args(tokens)
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
