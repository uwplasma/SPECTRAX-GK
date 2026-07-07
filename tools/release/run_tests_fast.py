#!/usr/bin/env python3
"""Run pytest files with bounded per-file and total local-test budgets."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DIR = REPO_ROOT / "tests"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", type=Path, help="Optional explicit test files."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Directory containing test_*.py files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-file pytest timeout in seconds.",
    )
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=300.0,
        help="Total runner budget in seconds; use 0 to disable the whole-run cap.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed to each pytest invocation; repeat as needed.",
    )
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_tests(
    files: list[Path],
    *,
    per_file_timeout_s: float,
    total_timeout_s: float,
    pytest_args: list[str] | None = None,
) -> tuple[int, list[tuple[str, str, float]]]:
    """Run each pytest file and return ``(exit_code, results)``.

    Exit code ``124`` means at least one subprocess hit a timeout or the
    configured total budget expired before all files were attempted.
    """

    pytest_args = list(pytest_args or [])
    deadline = time.monotonic() + total_timeout_s if total_timeout_s > 0.0 else None
    results: list[tuple[str, str, float]] = []
    timed_out = False
    failed = False

    for path in files:
        label = _relative(path)
        if deadline is not None and time.monotonic() >= deadline:
            results.append((label, "not_run(total_timeout)", 0.0))
            timed_out = True
            continue

        timeout_s = float(per_file_timeout_s)
        if deadline is not None:
            timeout_s = max(1.0, min(timeout_s, deadline - time.monotonic()))

        t0 = time.monotonic()
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "--disable-warnings",
            *pytest_args,
            str(path),
        ]
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, timeout=timeout_s)
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
            timed_out = True
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 5:
                status = "skipped(no_tests_collected)"
            else:
                status = f"fail({exc.returncode})"
                failed = True
        dt = time.monotonic() - t0
        results.append((label, status, dt))
        print(f"{label}: {status} ({dt:.1f}s)", flush=True)

    print("SUMMARY", flush=True)
    for path, status, dt in results:
        print(f"{path}\t{status}\t{dt:.1f}s", flush=True)

    if timed_out:
        return 124, results
    if failed:
        return 1, results
    return 0, results


def main() -> int:
    args = parse_args()
    files = [path if path.is_absolute() else (REPO_ROOT / path) for path in args.files]
    if not files:
        files = discover_test_files(args.test_dir)
    if not files:
        raise SystemExit(f"no test_*.py files found under {args.test_dir}")
    code, _results = run_tests(
        files,
        per_file_timeout_s=float(args.timeout),
        total_timeout_s=float(args.total_timeout),
        pytest_args=list(args.pytest_arg),
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
