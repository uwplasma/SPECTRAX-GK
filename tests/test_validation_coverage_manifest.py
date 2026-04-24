from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_validation_coverage_manifest.py"
    spec = importlib.util.spec_from_file_location("check_validation_coverage_manifest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_text(
    *,
    source: str,
    test: str,
    artifact: str,
    module: str = "spectraxgk.runtime",
    status: str = "active",
) -> str:
    return f"""
[metadata]
package_coverage_target_percent = 95.0

[[modules]]
module = "{module}"
path = "{source}"
owner_lane = "runtime lane"
status = "{status}"
coverage_priority = "high"
coverage_target_percent = 95.0
reference_anchors = ["reference"]
physics_contracts = ["physics"]
numerics_contracts = ["numerics"]
fast_tests = ["{test}"]
artifact_paths = ["{artifact}"]
next_tests = ["next"]
"""


def test_repository_validation_manifest_is_well_formed() -> None:
    mod = _load_tool_module()
    summary = mod.validate_manifest(mod.load_manifest())

    assert summary["package_coverage_target_percent"] == 95.0
    assert summary["n_modules"] >= 10
    rows = {row["module"]: row for row in summary["rows"]}
    assert rows["spectraxgk.linear"]["coverage_target_percent"] == 95.0
    assert rows["spectraxgk.validation_gates"]["n_physics_contracts"] >= 2
    assert "spectraxgk.nonlinear" in summary["high_priority_open"]


def test_validation_manifest_main_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out_json = tmp_path / "summary.json"

    assert mod.main(["--out-json", str(out_json)]) == 0

    payload = json.loads(out_json.read_text())
    assert payload["n_modules"] >= 10
    assert payload["package_coverage_target_percent"] == 95.0


def test_validation_manifest_rejects_missing_fast_test(tmp_path: Path) -> None:
    mod = _load_tool_module()
    source = tmp_path / "src" / "spectraxgk" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/spectraxgk/runtime.py",
            test="tests/missing.py",
            artifact="docs/_static/gate.json",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="fast test does not exist"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_invalid_status(tmp_path: Path) -> None:
    mod = _load_tool_module()
    source = tmp_path / "src" / "spectraxgk" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source\n")
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/spectraxgk/runtime.py",
            test="tests/test_runtime.py",
            artifact="docs/_static/gate.json",
            status="halfway",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="invalid status"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root
