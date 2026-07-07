from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LARGE_MODULE_DIRECT_ROW_MIN_SOURCE_LINES = 2_000
PUBLIC_PACKAGE_API_INIT_EXCEPTIONS = {
    "spectraxgk.api",
    "spectraxgk.geometry",
    "spectraxgk.operators",
    "spectraxgk.operators.linear",
}


def _load_tool_module():
    path = ROOT / "tools" / "check_validation_coverage_manifest.py"
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


def _repository_manifest_sets() -> tuple[set[str], set[str], set[str]]:
    mod = _load_tool_module()
    data = mod.load_manifest()
    summary = mod.validate_manifest(data)
    direct_modules = {row["module"] for row in summary["rows"]}
    owned_modules = {
        owned_module
        for modules in summary["owned_modules_by_owner"].values()
        for owned_module in modules
    }
    excluded_modules = set(data["coverage_inventory"]["excluded_modules"])
    return direct_modules, owned_modules, excluded_modules


def _documented_public_api_modules() -> set[str]:
    api_reference = (ROOT / "docs" / "api.rst").read_text(encoding="utf-8")
    return set(
        re.findall(
            r"^\.\. automodule:: (spectraxgk(?:\.[A-Za-z_]\w*)*)\s*$",
            api_reference,
            flags=re.MULTILINE,
        )
    )


def _manifest_candidates_for_api_module(module: str) -> set[str]:
    source_base = ROOT / "src" / Path(*module.split("."))
    candidates: set[str] = set()
    if source_base.with_suffix(".py").exists():
        candidates.add(module)
    if (source_base / "__init__.py").exists():
        candidates.add(f"{module}.__init__")
    return candidates


def _source_module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)


def _source_line_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_documented_public_api_modules_have_manifest_tracking() -> None:
    direct_modules, owned_modules, excluded_modules = _repository_manifest_sets()
    tracked_modules = direct_modules | owned_modules | excluded_modules
    public_modules = _documented_public_api_modules()

    missing_source = sorted(
        module for module in public_modules if not _manifest_candidates_for_api_module(module)
    )
    missing_manifest = {
        module: sorted(candidates)
        for module in sorted(public_modules)
        if (candidates := _manifest_candidates_for_api_module(module))
        and candidates.isdisjoint(tracked_modules)
    }
    excluded_package_api = {
        module
        for module in public_modules
        if f"{module}.__init__" in _manifest_candidates_for_api_module(module)
        and f"{module}.__init__" in excluded_modules
    }

    assert not missing_source
    assert not missing_manifest
    assert excluded_package_api <= PUBLIC_PACKAGE_API_INIT_EXCEPTIONS


def test_large_modules_have_direct_manifest_rows() -> None:
    direct_modules, _, _ = _repository_manifest_sets()
    large_modules_without_direct_rows: dict[str, int] = {}
    for path in (ROOT / "src" / "spectraxgk").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        source_lines = _source_line_count(path)
        module = _source_module_name(path)
        if (
            source_lines >= LARGE_MODULE_DIRECT_ROW_MIN_SOURCE_LINES
            and module not in direct_modules
        ):
            large_modules_without_direct_rows[module] = source_lines

    assert not large_modules_without_direct_rows


def test_manifest_accepts_owned_refactor_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "spectraxgk.runtime", "spectraxgk.workflows.runtime.config")
    _write_fast_inputs(tmp_path)

    summary = _validate_tmp_manifest(
        tmp_path,
        _manifest(_row("spectraxgk.runtime", owned_modules=["spectraxgk.workflows.runtime.config"])),
    )

    assert summary["n_direct_modules"] == 1
    assert summary["n_owned_modules"] == 1
    assert summary["n_excluded_modules"] == 1
    assert summary["owned_modules_by_owner"]["spectraxgk.runtime"] == ["spectraxgk.workflows.runtime.config"]


def test_manifest_rejects_unowned_package_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "spectraxgk.runtime", "spectraxgk.workflows.runtime.config")
    _write_fast_inputs(tmp_path)

    with pytest.raises(ValueError, match="package modules lack coverage ownership"):
        _validate_tmp_manifest(tmp_path, _manifest(_row("spectraxgk.runtime")))


def test_manifest_rejects_duplicate_owned_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "spectraxgk.runtime", "spectraxgk.linear", "spectraxgk.workflows.runtime.config")
    _write_fast_inputs(tmp_path)

    manifest = _manifest(
        _row("spectraxgk.runtime", owned_modules=["spectraxgk.workflows.runtime.config"]),
        _row("spectraxgk.linear", owned_modules=["spectraxgk.workflows.runtime.config"]),
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
