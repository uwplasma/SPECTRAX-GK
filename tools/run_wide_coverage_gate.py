#!/usr/bin/env python3
"""Run package-wide coverage in bounded shards and enforce a combined gate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DIR = REPO_ROOT / "tests"


def discover_test_files(test_dir: Path = DEFAULT_TEST_DIR) -> list[Path]:
    """Return top-level pytest files in deterministic order."""

    return sorted(test_dir.glob("test_*.py"))


def split_shards(items: list[Path], nshards: int) -> list[list[Path]]:
    """Split paths into contiguous shards with near-equal file counts."""

    if nshards < 1:
        raise ValueError("nshards must be >= 1")
    if not items:
        return [[] for _ in range(nshards)]
    shard_size = int(math.ceil(len(items) / float(nshards)))
    return [items[i * shard_size : (i + 1) * shard_size] for i in range(nshards)]


def _run(cmd: list[str], *, timeout: int | None, cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=cwd, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"command timed out after {timeout}s: {' '.join(cmd)}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=6, help="Number of bounded test shards.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-shard timeout in seconds.")
    parser.add_argument("--fail-under", type=float, default=95.0, help="Combined package coverage threshold.")
    parser.add_argument("--xml", type=Path, default=Path("coverage-wide.xml"), help="Combined XML report path.")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Directory containing test_*.py files.")
    parser.add_argument("--dry-run", action="store_true", help="Print shard membership without running tests.")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed to each pytest shard; repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tests = discover_test_files(args.test_dir)
    if not tests:
        raise SystemExit(f"no test_*.py files found under {args.test_dir}")
    shards = split_shards(tests, int(args.shards))

    for idx, shard in enumerate(shards):
        rel = [str(path.relative_to(REPO_ROOT)) for path in shard]
        print(f"shard {idx + 1}/{len(shards)}: {len(rel)} files")
        for path in rel:
            print(f"  {path}")

    if args.dry_run:
        return

    _run([sys.executable, "-m", "coverage", "erase"], timeout=None, cwd=REPO_ROOT)
    for idx, shard in enumerate(shards):
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

    _run([sys.executable, "-m", "coverage", "combine"], timeout=120, cwd=REPO_ROOT)
    _run([sys.executable, "-m", "coverage", "xml", "-o", str(args.xml)], timeout=120, cwd=REPO_ROOT)
    _run(
        [sys.executable, "-m", "coverage", "report", f"--fail-under={float(args.fail_under):.6g}"],
        timeout=120,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
