#!/usr/bin/env python3
"""Run pytest per file with a per-file timeout for fast local feedback."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    root = Path("tests")
    files = sorted(root.rglob("test_*.py"))
    timeout_s = 300
    results: list[tuple[str, str, float]] = []
    for path in files:
        t0 = time.time()
        try:
            subprocess.run(["pytest", str(path)], check=True, timeout=timeout_s)
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except subprocess.CalledProcessError as exc:
            status = f"fail({exc.returncode})"
        dt = time.time() - t0
        results.append((str(path), status, dt))
        print(f"{path}: {status} ({dt:.1f}s)")

    print("SUMMARY")
    for path, status, dt in results:
        print(f"{path}\t{status}\t{dt:.1f}s")
    return 0 if all(status == "ok" for _, status, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
