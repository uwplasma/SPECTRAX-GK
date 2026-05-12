from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_validation_coverage_manifest.py"
    spec = importlib.util.spec_from_file_location("check_validation_coverage_manifest_refactor", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_package(tmp_path: Path, *modules: str) -> None:
    package = tmp_path / "src" / "spectraxgk"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("# package\n")
    for module in modules:
        module_path = tmp_path / "src" / Path(*module.split(".")).with_suffix(".py")
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("# source\n")


def _write_fast_inputs(tmp_path: Path) -> None:
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}\n")


def _row(module: str, owned_modules: list[str] | None = None) -> str:
    owned = ""
    if owned_modules is not None:
        body = "\n".join(f'  "{owned_module}",' for owned_module in owned_modules)
        owned = f"owned_modules = [\n{body}\n]\n"
    return f"""
[[modules]]
module = "{module}"
path = "src/{module.replace('.', '/')}.py"
{owned}owner_lane = "runtime lane"
status = "active"
coverage_priority = "high"
coverage_target_percent = 95.0
reference_anchors = ["reference"]
physics_contracts = ["physics"]
numerics_contracts = ["numerics"]
fast_tests = ["tests/test_runtime.py"]
artifact_paths = ["docs/_static/gate.json"]
next_tests = ["next"]
"""


def _manifest(*rows: str) -> str:
    return """
[metadata]
package_coverage_target_percent = 95.0

[coverage_inventory]
require_all_package_modules_owned = true
excluded_modules = ["spectraxgk.__init__"]
""" + "".join(rows)


def _validate_tmp_manifest(tmp_path: Path, manifest_text: str):
    mod = _load_tool_module()
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(manifest_text)
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        return mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_manifest_accepts_owned_refactor_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "spectraxgk.runtime", "spectraxgk.runtime_config")
    _write_fast_inputs(tmp_path)

    summary = _validate_tmp_manifest(
        tmp_path,
        _manifest(_row("spectraxgk.runtime", owned_modules=["spectraxgk.runtime_config"])),
    )

    assert summary["n_direct_modules"] == 1
    assert summary["n_owned_modules"] == 1
    assert summary["n_excluded_modules"] == 1
    assert summary["owned_modules_by_owner"]["spectraxgk.runtime"] == ["spectraxgk.runtime_config"]


def test_manifest_rejects_unowned_package_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "spectraxgk.runtime", "spectraxgk.runtime_config")
    _write_fast_inputs(tmp_path)

    with pytest.raises(ValueError, match="package modules lack coverage ownership"):
        _validate_tmp_manifest(tmp_path, _manifest(_row("spectraxgk.runtime")))


def test_manifest_rejects_duplicate_owned_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "spectraxgk.runtime", "spectraxgk.linear", "spectraxgk.runtime_config")
    _write_fast_inputs(tmp_path)

    manifest = _manifest(
        _row("spectraxgk.runtime", owned_modules=["spectraxgk.runtime_config"]),
        _row("spectraxgk.linear", owned_modules=["spectraxgk.runtime_config"]),
    )
    with pytest.raises(ValueError, match="duplicate coverage ownership"):
        _validate_tmp_manifest(tmp_path, manifest)


def test_manifest_rejects_direct_rows_listed_as_owned_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "spectraxgk.runtime", "spectraxgk.linear")
    _write_fast_inputs(tmp_path)

    manifest = _manifest(
        _row("spectraxgk.runtime", owned_modules=["spectraxgk.linear"]),
        _row("spectraxgk.linear"),
    )
    with pytest.raises(ValueError, match="direct manifest rows must not be listed as owned modules"):
        _validate_tmp_manifest(tmp_path, manifest)
