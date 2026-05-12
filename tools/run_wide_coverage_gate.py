#!/usr/bin/env python3
"""Run package-wide coverage in bounded shards and enforce a combined gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DIR = REPO_ROOT / "tests"
COVERAGE_DATA_RE = re.compile(r"^\.coverage\.shard-(?P<shard>[0-9]+)\.")


def _resolve_test_dir(test_dir: Path) -> Path:
    """Resolve relative test directories against the repository root."""

    return (
        (REPO_ROOT / test_dir).resolve()
        if not test_dir.is_absolute()
        else test_dir.resolve()
    )


def discover_test_files(test_dir: Path = DEFAULT_TEST_DIR) -> list[Path]:
    """Return top-level pytest files in deterministic order."""

    return sorted(_resolve_test_dir(test_dir).glob("test_*.py"))


def split_shards(items: list[Path], nshards: int) -> list[list[Path]]:
    """Split paths into round-robin shards with near-equal file counts.

    Alphabetical test discovery groups related plotting tests together. A
    round-robin split keeps deterministic membership while spreading expensive
    modules across CI workers, which makes the package-wide coverage gate less
    sensitive to runner memory and shutdown pressure.
    """

    if nshards < 1:
        raise ValueError("nshards must be >= 1")
    shards: list[list[Path]] = [[] for _ in range(nshards)]
    for idx, item in enumerate(items):
        shards[idx % nshards].append(item)
    return shards


def discover_coverage_data(root: Path = REPO_ROOT) -> list[Path]:
    """Return coverage.py data files under ``root`` without descending into docs."""

    return sorted(path for path in root.glob(".coverage.*") if path.is_file())


def discover_empty_shard_markers(root: Path = REPO_ROOT) -> list[Path]:
    """Return sentinel files written when a CI shard produced no coverage data."""

    return sorted(path for path in root.glob("EMPTY_SHARD_*") if path.is_file())


def build_coverage_shard_report(root: Path, nshards: int) -> dict[str, object]:
    """Return a JSON-ready report for combine-time shard artifact validation."""

    coverage_files = discover_coverage_data(root)
    markers = discover_empty_shard_markers(root)
    labeled: dict[int, list[Path]] = {idx: [] for idx in range(1, nshards + 1)}
    unlabeled: list[Path] = []
    out_of_range: list[Path] = []
    for path in coverage_files:
        match = COVERAGE_DATA_RE.match(path.name)
        if not match:
            unlabeled.append(path)
            continue
        shard = int(match.group("shard"))
        if 1 <= shard <= nshards:
            labeled[shard].append(path)
        else:
            out_of_range.append(path)

    def _rel(paths: list[Path]) -> list[str]:
        return [str(path.relative_to(root)) for path in paths]

    return {
        "kind": "wide_coverage_shard_report",
        "root": str(root),
        "expected_shards": int(nshards),
        "coverage_data_files": _rel(coverage_files),
        "coverage_data_file_count": len(coverage_files),
        "empty_shard_markers": _rel(markers),
        "unlabeled_coverage_data_files": _rel(unlabeled),
        "out_of_range_labeled_coverage_data_files": _rel(out_of_range),
        "labeled_shards": {
            str(idx): _rel(paths) for idx, paths in labeled.items() if paths
        },
        "missing_labeled_shards": [
            idx for idx, paths in labeled.items() if not paths
        ],
    }


def validate_coverage_shard_report(
    report: dict[str, object], *, require_labeled_shards: bool
) -> list[str]:
    """Return validation failures for a combine-time coverage shard report."""

    failures: list[str] = []
    if int(report["coverage_data_file_count"]) == 0:
        failures.append("no coverage.py data files were found")
    markers = list(report["empty_shard_markers"])
    if markers:
        failures.append(f"empty shard markers found: {markers}")
    out_of_range = list(report["out_of_range_labeled_coverage_data_files"])
    if out_of_range:
        failures.append(f"out-of-range labeled coverage data files found: {out_of_range}")
    missing = list(report["missing_labeled_shards"])
    if require_labeled_shards and missing:
        failures.append(f"missing labeled coverage data for shards: {missing}")
    return failures


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic JSON for CI artifacts and local diagnostics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(cmd: list[str], *, timeout: int | None, cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=cwd, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"command timed out after {timeout}s: {' '.join(cmd)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", type=int, default=6, help="Number of bounded test shards."
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Per-shard timeout in seconds."
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=95.0,
        help="Combined package coverage threshold.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("coverage-wide.xml"),
        help="Combined XML report path.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Directory containing test_*.py files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print shard membership without running tests.",
    )
    parser.add_argument(
        "--only-shard",
        type=int,
        default=None,
        help="Run only one 1-based shard. Useful for local bounded shard execution.",
    )
    parser.add_argument(
        "--keep-existing-coverage",
        action="store_true",
        help="Do not erase existing .coverage data before running selected shard(s).",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Run selected shard(s) without combining/reporting coverage data.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip pytest execution and only combine/report existing shard coverage data.",
    )
    parser.add_argument(
        "--require-shard-data",
        action="store_true",
        help=(
            "Before combining, require one or more labeled .coverage.shard-N.* "
            "files for every configured shard and reject EMPTY_SHARD_N markers."
        ),
    )
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        default=None,
        help="Optional JSON report path for combine-time coverage shard data.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed to each pytest shard; repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_dir = _resolve_test_dir(args.test_dir)
    tests = discover_test_files(test_dir)
    if not tests:
        raise SystemExit(f"no test_*.py files found under {test_dir}")
    shards = split_shards(tests, int(args.shards))

    for idx, shard in enumerate(shards):
        rel = [str(path.relative_to(REPO_ROOT)) for path in shard]
        print(f"shard {idx + 1}/{len(shards)}: {len(rel)} files")
        for path in rel:
            print(f"  {path}")

    if args.dry_run:
        return

    if args.combine_only:
        report = build_coverage_shard_report(REPO_ROOT, int(args.shards))
        if args.shard_manifest is not None:
            write_json(args.shard_manifest, report)
        failures = validate_coverage_shard_report(
            report, require_labeled_shards=bool(args.require_shard_data)
        )
        if failures:
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            raise SystemExit("wide coverage shard validation failed: " + "; ".join(failures))
        _run([sys.executable, "-m", "coverage", "combine"], timeout=120, cwd=REPO_ROOT)
        _run(
            [sys.executable, "-m", "coverage", "xml", "-o", str(args.xml)],
            timeout=120,
            cwd=REPO_ROOT,
        )
        _run(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                f"--fail-under={float(args.fail_under):.6g}",
            ],
            timeout=120,
            cwd=REPO_ROOT,
        )
        return

    if args.only_shard is not None and not (1 <= int(args.only_shard) <= len(shards)):
        raise SystemExit(f"--only-shard must be in [1, {len(shards)}]")
    selected = (
        [(int(args.only_shard) - 1, shards[int(args.only_shard) - 1])]
        if args.only_shard is not None
        else list(enumerate(shards))
    )

    if not args.keep_existing_coverage:
        _run([sys.executable, "-m", "coverage", "erase"], timeout=None, cwd=REPO_ROOT)
    for idx, shard in selected:
        if not shard:
            continue
        rel = [str(path.relative_to(REPO_ROOT)) for path in shard]
        cmd = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--source=spectraxgk",
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "--disable-warnings",
            *args.pytest_arg,
            *rel,
        ]
        print(f"running coverage shard {idx + 1}/{len(shards)}", flush=True)
        _run(cmd, timeout=int(args.timeout), cwd=REPO_ROOT)

    if args.skip_combine:
        return

    _run([sys.executable, "-m", "coverage", "combine"], timeout=120, cwd=REPO_ROOT)
    _run(
        [sys.executable, "-m", "coverage", "xml", "-o", str(args.xml)],
        timeout=120,
        cwd=REPO_ROOT,
    )
    _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            f"--fail-under={float(args.fail_under):.6g}",
        ],
        timeout=120,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
