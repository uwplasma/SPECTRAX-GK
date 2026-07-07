#!/usr/bin/env python3
"""Audit tracked file sizes and local artifact roots without modifying the repo."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
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
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
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


def build_repository_size_report(root: str | Path = ROOT, *, top_n: int = 30) -> dict[str, Any]:
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
        local_roots.append({"path": rel, "bytes": int(_directory_size(path)), "exists": path.exists()})

    return {
        "kind": "repository_size_audit",
        "root": str(root_path),
        "tracked_total_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "tracked_file_count": len(rows),
        "tracked_by_category": dict(sorted(by_category.items(), key=lambda item: item[1], reverse=True)),
        "largest_tracked_files": rows[:top_n],
        "local_artifact_roots": local_roots,
        "notes": (
            "This report is non-destructive. Move large tracked validation artifacts to "
            "GitHub Releases or another artifact store before considering a coordinated "
            "history rewrite."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repository root to audit.")
    parser.add_argument("--top", type=int, default=30, help="Number of tracked files to list.")
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
