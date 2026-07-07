"""Shared path helpers for tests.

Tests move across topology folders during refactors; keep repository paths here
instead of encoding parent-depth assumptions in individual test files.
"""

from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
