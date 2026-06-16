from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "check_validation_coverage_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_validation_coverage_manifest", path
    )
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

[coverage_inventory]
require_all_package_modules_owned = true
excluded_modules = ["spectraxgk.__init__"]

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


def _write_minimal_package(tmp_path: Path, *modules: str) -> None:
    package = tmp_path / "src" / "spectraxgk"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("# package\n")
    for module in modules:
        assert module.startswith("spectraxgk.")
        module_path = tmp_path / "src" / Path(*module.split(".")).with_suffix(".py")
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("# source\n")


def test_repository_validation_manifest_is_well_formed() -> None:
    mod = _load_tool_module()
    summary = mod.validate_manifest(mod.load_manifest())

    assert summary["package_coverage_target_percent"] == 95.0
    assert summary["n_modules"] >= 10
    assert summary["n_package_modules"] == (
        summary["n_direct_modules"]
        + summary["n_owned_modules"]
        + summary["n_excluded_modules"]
    )
    rows = {row["module"]: row for row in summary["rows"]}
    assert rows["spectraxgk.linear"]["coverage_target_percent"] == 95.0
    assert rows["spectraxgk.runtime"]["n_owned_modules"] >= 5
    assert rows["spectraxgk.validation_gates"]["n_physics_contracts"] >= 2
    assert (
        rows["spectraxgk.solver_ready_gradient_gates"]["coverage_target_percent"]
        == 95.0
    )
    assert (
        rows["spectraxgk.solver_vmec_boozer_gradient_gates"]["n_numerics_contracts"]
        >= 2
    )
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
    _write_minimal_package(tmp_path, "spectraxgk.runtime")
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
    _write_minimal_package(tmp_path, "spectraxgk.runtime")
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


def test_validation_manifest_rejects_duplicate_manifest_list_entries(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    _write_minimal_package(tmp_path, "spectraxgk.runtime", "spectraxgk.config")
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[metadata]
package_coverage_target_percent = 95.0

[coverage_inventory]
require_all_package_modules_owned = true
excluded_modules = ["spectraxgk.__init__"]

[[modules]]
module = "spectraxgk.runtime"
path = "src/spectraxgk/runtime.py"
owned_modules = ["spectraxgk.config", "spectraxgk.config"]
owner_lane = "runtime lane"
status = "active"
coverage_priority = "high"
coverage_target_percent = 95.0
reference_anchors = ["reference"]
physics_contracts = ["physics"]
numerics_contracts = ["numerics"]
fast_tests = ["tests/test_runtime.py"]
artifact_paths = ["docs/_static/gate.json"]
next_tests = ["next"]
""".strip()
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(
            ValueError, match="owned_modules contains duplicate entries"
        ):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_directory_fast_test(tmp_path: Path) -> None:
    mod = _load_tool_module()
    _write_minimal_package(tmp_path, "spectraxgk.runtime")
    test_dir = tmp_path / "tests" / "runtime_cases"
    test_dir.mkdir(parents=True)
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/spectraxgk/runtime.py",
            test="tests/runtime_cases",
            artifact="docs/_static/gate.json",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="fast test must be a file"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_nested_fast_test_not_seen_by_wide_gate(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    _write_minimal_package(tmp_path, "spectraxgk.runtime")
    test = tmp_path / "tests" / "runtime" / "test_runtime.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/spectraxgk/runtime.py",
            test="tests/runtime/test_runtime.py",
            artifact="docs/_static/gate.json",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="discoverable by run_wide_coverage_gate"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_non_pytest_fast_test_name(tmp_path: Path) -> None:
    mod = _load_tool_module()
    _write_minimal_package(tmp_path, "spectraxgk.runtime")
    test = tmp_path / "tests" / "runtime_cases.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/spectraxgk/runtime.py",
            test="tests/runtime_cases.py",
            artifact="docs/_static/gate.json",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="top-level tests/test_\\*.py"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_attaches_measured_package_coverage(tmp_path: Path) -> None:
    mod = _load_tool_module()
    _write_minimal_package(tmp_path, "spectraxgk.runtime")
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
        )
    )
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """
<coverage line-rate="0.96">
  <packages>
    <package name="spectraxgk">
      <classes>
        <class filename="src/spectraxgk/runtime.py" line-rate="0.97" />
      </classes>
    </package>
  </packages>
</coverage>
""".strip()
    )

    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        summary = mod.validate_manifest(
            mod.load_manifest(manifest),
            coverage_xml=coverage_xml,
            enforce_package_coverage=True,
        )
    finally:
        mod.REPO_ROOT = old_root

    measured = summary["coverage_xml_summary"]
    assert measured["package_coverage_passed"] is True
    assert measured["package_coverage_percent"] == pytest.approx(96.0)
    assert measured["n_modules_below_target"] == 0
    assert measured["module_rows"][0]["coverage_percent"] == pytest.approx(97.0)


def test_validation_manifest_rejects_duplicate_coverage_xml_module_entries(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    _write_minimal_package(tmp_path, "spectraxgk.runtime")
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
        )
    )
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """
<coverage line-rate="0.96">
  <packages>
    <package name="spectraxgk">
      <classes>
        <class filename="src/spectraxgk/runtime.py" line-rate="0.97" />
        <class filename="spectraxgk/runtime.py" line-rate="0.50" />
      </classes>
    </package>
  </packages>
</coverage>
""".strip()
    )

    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(
            ValueError, match="duplicate coverage entry for spectraxgk.runtime"
        ):
            mod.validate_manifest(
                mod.load_manifest(manifest), coverage_xml=coverage_xml
            )
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_package_coverage_below_target(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    _write_minimal_package(tmp_path, "spectraxgk.runtime")
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
        )
    )
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """
<coverage line-rate="0.949">
  <packages>
    <package name="spectraxgk">
      <classes>
        <class filename="spectraxgk/runtime.py" line-rate="1.0" />
      </classes>
    </package>
  </packages>
</coverage>
""".strip()
    )

    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="package coverage below manifest target"):
            mod.validate_manifest(
                mod.load_manifest(manifest),
                coverage_xml=coverage_xml,
                enforce_package_coverage=True,
            )
    finally:
        mod.REPO_ROOT = old_root
