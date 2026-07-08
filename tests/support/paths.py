"""Shared path helpers for tests.

Tests move across topology folders during refactors; keep repository paths here
instead of encoding parent-depth assumptions in individual test files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"


def load_artifact_tool(script_name: str) -> ModuleType:
    """Load a ``tools/artifacts`` script directly from the checkout."""

    tools_dir = TOOLS_ROOT / "artifacts"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"test_loaded_{script_name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
