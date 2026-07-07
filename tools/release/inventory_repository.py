#!/usr/bin/env python3
"""Build a repository inventory for consolidation planning.

The refactor plan uses this tool to classify tracked files by ownership and
whether they should stay in the installable package, move to maintenance areas,
or be reviewed for deletion. The output is intentionally machine-readable so the
classification can be diffed during large topology changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT_MARKERS = ("pyproject.toml", ".git")
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".rst",
    ".md",
    ".yml",
    ".yaml",
    ".json",
    ".csv",
    ".txt",
}


@dataclass(frozen=True)
class InventoryRow:
    path: str
    area: str
    role: str
    action: str
    suffix: str
    bytes: int
    lines: int | None
    notes: str


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    for parent in (current, *current.parents):
        if all((parent / marker).exists() for marker in ROOT_MARKERS):
            return parent
    raise RuntimeError(f"could not find repository root from {start}")


def _git_ls_files(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return [root / line for line in proc.stdout.splitlines() if line]


def _line_count(path: Path) -> int | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        return sum(1 for _ in path.open("rb"))
    except OSError:
        return None


def _area(rel: Path) -> str:
    if len(rel.parts) == 1:
        return "root"
    if rel.parts[0] == "src" and len(rel.parts) > 2 and rel.parts[1] == "spectraxgk":
        return (
            "src/spectraxgk"
            if len(rel.parts) == 3
            else f"src/spectraxgk/{rel.parts[2]}"
        )
    if rel.parts[0] in {"tests", "tools", "examples", "benchmarks", "docs"}:
        return rel.parts[0] if len(rel.parts) == 1 else f"{rel.parts[0]}/{rel.parts[1]}"
    return rel.parts[0]


def _role_and_action(rel: Path) -> tuple[str, str, str]:
    parts = rel.parts
    name = rel.name
    stem = rel.stem
    path = rel.as_posix()

    if parts[0] == "src" and len(parts) > 2 and parts[1] == "spectraxgk":
        if len(parts) > 3 and parts[2] == "validation":
            return (
                "installable validation/campaign code",
                "move-or-shrink",
                "candidate for benchmarks/, tools/campaigns/, tests/validation, or small metric facade",
            )
        return (
            "promoted library code",
            "keep-and-consolidate",
            "must preserve public behavior and JAX contracts",
        )

    if parts[0] == "tests":
        if len(parts) == 2:
            return (
                "flat test",
                "move-or-merge",
                "root-level tests should move into domain folders",
            )
        return (
            "organized test",
            "keep-or-merge",
            "keep coverage while reducing one-file-per-wrapper tests",
        )

    if parts[0] == "tools":
        if name == "README.md":
            return "tool documentation", "keep", "documents maintenance-tool ownership"
        if len(parts) == 2 and rel.suffix == ".py":
            if stem.startswith(("probe_", "debug_")):
                return (
                    "probe/debug tool",
                    "delete-or-move-out-of-main",
                    "not a maintained release entry point",
                )
            if stem.startswith(("compare_", "generate_gx_")):
                return (
                    "comparison utility",
                    "move",
                    "belongs under tools/comparison if still current",
                )
            if stem.startswith(("profile_", "benchmark_")):
                return (
                    "profiling/performance tool",
                    "move",
                    "belongs under tools/profiling or benchmarks",
                )
            if stem.startswith(
                ("build_", "plot_", "make_", "digitize_", "derive_", "compress_")
            ):
                return (
                    "artifact builder",
                    "move-or-merge",
                    "belongs under tools/artifacts if output is referenced",
                )
            if stem.startswith(("check_", "audit_", "run_tests", "run_wide_coverage")):
                return "release gate", "move", "belongs under tools/release"
            if stem.startswith(
                (
                    "write_",
                    "run_",
                    "postprocess_",
                    "prepare_",
                    "finalize_",
                    "import_",
                    "rank_",
                    "design_",
                )
            ):
                return (
                    "campaign helper",
                    "move-or-delete",
                    "keep only if documented active campaign",
                )
            return "flat maintenance tool", "classify", "needs owner review"
        if len(parts) > 2:
            folder = parts[1]
            role = {
                "release": "release gate",
                "artifacts": "artifact builder",
                "profiling": "profiling/performance tool",
                "comparison": "comparison utility",
                "campaigns": "campaign helper",
            }.get(folder, "organized maintenance tool")
            return role, "keep-or-merge", f"owned by tools/{folder}"
        return "tool support file", "keep-or-review", "non-python tool asset"

    if parts[0] == "examples":
        return (
            "public example",
            "keep-or-scope",
            "must be runnable or explicitly marked long/manual",
        )

    if parts[0] == "benchmarks":
        if parts[1:2] == ("results",):
            return "benchmark result index", "keep-small", "no raw solver outputs"
        return "benchmark driver", "keep-small", "root benchmark layer is canonical"

    if parts[0] == "docs":
        if len(parts) > 1 and parts[1] == "_static":
            return (
                "docs artifact",
                "keep-if-referenced",
                "delete stale generated companions",
            )
        return "documentation", "keep-current", "must match promoted claims and layout"

    if parts[0] in {".github", "pyproject.toml", "uv.lock"} or path in {
        "pyproject.toml",
        "uv.lock",
    }:
        return "project infrastructure", "keep", "release/build/CI infrastructure"

    return "repository support", "keep-or-review", "classify during consolidation"


def build_inventory(root: Path) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for path in _git_ls_files(root):
        if not path.exists():
            # Let the inventory run during refactors before deletions are staged.
            # Once the deletion is committed, git ls-files will no longer report it.
            continue
        rel = path.relative_to(root)
        role, action, notes = _role_and_action(rel)
        rows.append(
            InventoryRow(
                path=rel.as_posix(),
                area=_area(rel),
                role=role,
                action=action,
                suffix=path.suffix,
                bytes=path.stat().st_size,
                lines=_line_count(path),
                notes=notes,
            )
        )
    return rows


def _write_json(rows: Iterable[InventoryRow], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(rows: Iterable[InventoryRow], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(asdict(rows[0]).keys())
            if rows
            else list(InventoryRow.__annotations__),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _summary(rows: Iterable[InventoryRow]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = summary.setdefault(row.action, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += row.bytes
    return dict(sorted(summary.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root or any path inside it.",
    )
    parser.add_argument("--json-out", type=Path, help="Write full inventory JSON.")
    parser.add_argument("--csv-out", type=Path, help="Write full inventory CSV.")
    parser.add_argument(
        "--summary-json-out", type=Path, help="Write action summary JSON."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = _repo_root(args.root)
    rows = build_inventory(root)
    if args.json_out:
        _write_json(rows, args.json_out)
    if args.csv_out:
        _write_csv(rows, args.csv_out)
    summary = _summary(rows)
    if args.summary_json_out:
        args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {"tracked_files": len(rows), "actions": summary}, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
