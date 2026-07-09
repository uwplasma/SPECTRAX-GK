from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from support.paths import REPO_ROOT, load_release_tool
from tools.release.check_package_architecture_manifest import (
    validate_architecture_policy,
)


ROOT = REPO_ROOT
LARGE_MODULE_DIRECT_ROW_MIN_SOURCE_LINES = 2_000
PUBLIC_PACKAGE_API_INIT_EXCEPTIONS = {
    "spectraxgk.api",
    "spectraxgk.geometry",
    "spectraxgk.operators",
    "spectraxgk.operators.linear",
}


def _load_differentiable_refactor_tool():
    return load_release_tool("check_differentiable_refactor_manifest")


def _load_performance_manifest_tool():
    return load_release_tool("check_performance_optimization_manifest")


def _load_validation_coverage_tool():
    return load_release_tool("check_validation_coverage_manifest")


def _architecture_manifest(*, allowed: list[str]) -> dict[str, object]:
    return {
        "metadata": {
            "schema_version": 1,
            "title": "test architecture policy",
            "layout_authority": "docs/architecture_refactor_plan.rst",
            "status": "active",
        },
        "root_prefix_policy": {
            "blocked_prefixes": ["runtime_", "nonlinear_"],
            "allowed_root_prefix_modules": allowed,
        },
        "package_policy": {
            "required_domain_packages": ["spectraxgk.operators"],
            "required_docs": ["docs/architecture_refactor_plan.rst"],
        },
    }


def _architecture_manifest_with_topology(
    *, count_path: str, baseline: int, target: int
) -> dict[str, object]:
    data = _architecture_manifest(allowed=[])
    data["topology_policy"] = {
        "mode": "no_regression_until_target",
        "description": "test topology policy",
        "counts": [
            {
                "name": "test_python_files",
                "path": count_path,
                "pattern": "*.py",
                "recursive": True,
                "baseline": baseline,
                "target": target,
            }
        ],
    }
    return data


def _performance_manifest_text(
    *, tool: str, artifact: str, status: str = "active"
) -> str:
    return f"""
[metadata]
schema_version = 1

[[lanes]]
name = "lane"
owner = "owner"
status = "{status}"
priority = "high"
platforms = ["cpu"]
cases = ["case"]
profiling_tools = ["{tool}"]
metrics = ["runtime_s"]
artifact_paths = ["{artifact}"]
bottleneck_hypotheses = ["hypothesis"]
optimization_actions = ["action"]
gates = ["gate"]
"""


def _write_package(tmp_path: Path, *modules: str) -> None:
    package = tmp_path / "src" / "spectraxgk"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    for module in modules:
        module_path = tmp_path / "src" / Path(*module.split(".")).with_suffix(".py")
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("# source\n", encoding="utf-8")


def _write_fast_inputs(tmp_path: Path) -> None:
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}\n", encoding="utf-8")


def _coverage_row(module: str, owned_modules: list[str] | None = None) -> str:
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


def _coverage_manifest(*rows: str) -> str:
    return """
[metadata]
package_coverage_target_percent = 95.0

[coverage_inventory]
require_all_package_modules_owned = true
excluded_modules = ["spectraxgk.__init__"]
""" + "".join(rows)


def _validate_tmp_coverage_manifest(tmp_path: Path, manifest_text: str):
    mod = _load_validation_coverage_tool()
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(manifest_text, encoding="utf-8")
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        return mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def _repository_manifest_sets() -> tuple[set[str], set[str], set[str]]:
    mod = _load_validation_coverage_tool()
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


def test_differentiable_refactor_manifest_is_well_formed() -> None:
    mod = _load_differentiable_refactor_tool()
    summary = mod.validate_manifest(mod.load_manifest())
    manifest = mod.load_manifest()
    assert summary["required_package_coverage_percent"] >= 95.0
    assert manifest["global_acceptance"]["require_adaptive_derivative_policy"] is True
    assert (
        "adaptive-branch derivative policy"
        in manifest["validation_policy"]["autodiff_gate_scope"]
    )
    assert summary["n_architecture_layers"] >= 8
    assert summary["n_phase1_contract_modules"] >= 2
    assert summary["n_phase1_split_modules"] >= 16
    assert summary["n_hotspots"] >= 9
    assert "spectraxgk.core.contracts" in summary["phase1_contract_modules"]
    assert "spectraxgk.core.extension_points" in summary["phase1_contract_modules"]
    assert "spectraxgk.diagnostics.growth_rates" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.backend_discovery" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.autodiff_checks" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.numerics" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.flux_tube_contract" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.sensitivity" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.booz_xform_bridge" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.vmec_state_sensitivity" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.geometry.vmec_boozer_core" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.geometry.vmec_flux_tube_reports" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.geometry.vmec_tensor_mapping" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.objectives.gradient_gates" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.objectives.vmec_boozer_gradients" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.objectives.vmec_boozer_fd" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.objectives.vmec_boozer_line_search" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.objectives.vmec_boozer" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.nonlinear.rhs" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.nonlinear.diagnostic_state" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.solvers.nonlinear.explicit" in summary[
        "phase1_split_modules"
    ]
    assert "spectraxgk.solvers.nonlinear.imex" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.linear.cache_builder" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.linear.moments" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.linear.params" in summary["phase1_split_modules"]
    assert "spectraxgk.solvers.linear.krylov" in summary["phase1_split_modules"]
    assert "spectraxgk.solvers.linear.parallel" in summary["phase1_split_modules"]
    assert "spectraxgk.workflows.cases" in summary["phase1_split_modules"]
    assert "spectraxgk.artifacts.io" in summary["phase1_split_modules"]
    assert "spectraxgk.artifacts.linear" in summary["phase1_split_modules"]
    assert "spectraxgk.artifacts.nonlinear" in summary["phase1_split_modules"]
    assert "spectraxgk.artifacts.nonlinear_diagnostics" in summary[
        "phase1_split_modules"
    ]
    for module in (
        "spectraxgk.benchmarks",
        "spectraxgk.geometry.differentiable",
        "spectraxgk.operators.nonlinear.parallel",
        "spectraxgk.objectives.solver_gradients",
        "spectraxgk.nonlinear",
        "spectraxgk.workflows.runtime.artifacts",
        "spectraxgk.runtime",
        "spectraxgk.linear",
        "spectraxgk.cli",
    ):
        assert module in summary["hotspot_modules"]


def test_differentiable_refactor_manifest_main_writes_summary_json(
    tmp_path: Path,
) -> None:
    mod = _load_differentiable_refactor_tool()
    out_json = tmp_path / "summary.json"
    assert mod.main(["--out-json", str(out_json)]) == 0
    payload = out_json.read_text(encoding="utf-8")
    assert "Differentiable architecture refactor plan" in payload
    assert "spectraxgk.geometry.differentiable" in payload


def test_validate_architecture_policy_accepts_manifested_root_facade(tmp_path):
    source_root = tmp_path / "spectraxgk"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "nonlinear_removed_helper.py").write_text("", encoding="utf-8")

    summary = validate_architecture_policy(
        _architecture_manifest(allowed=["spectraxgk.nonlinear_removed_helper"]),
        source_root=source_root,
        check_paths=False,
    )

    assert summary["n_current_root_prefix_modules"] == 1
    assert summary["n_allowed_root_prefix_modules"] == 1


def test_validate_architecture_policy_rejects_new_root_prefix_module(tmp_path):
    source_root = tmp_path / "spectraxgk"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "runtime_extra.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="root-level prefix modules"):
        validate_architecture_policy(
            _architecture_manifest(allowed=[]),
            source_root=source_root,
            check_paths=False,
        )


def test_validate_architecture_policy_reports_topology_gap(tmp_path):
    source_root = tmp_path / "spectraxgk"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(3):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    summary = validate_architecture_policy(
        _architecture_manifest_with_topology(
            count_path=str(count_root), baseline=5, target=2
        ),
        source_root=source_root,
        check_paths=False,
    )

    row = summary["topology_counts"][0]
    assert row["count"] == 3
    assert row["baseline"] == 5
    assert row["target"] == 2
    assert row["remaining_to_target"] == 1
    assert row["target_met"] is False
    assert summary["topology_targets_met"] is False


def test_validate_architecture_policy_rejects_topology_regression(tmp_path):
    source_root = tmp_path / "spectraxgk"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(3):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="above baseline"):
        validate_architecture_policy(
            _architecture_manifest_with_topology(
                count_path=str(count_root), baseline=2, target=1
            ),
            source_root=source_root,
            check_paths=False,
        )


def test_validate_architecture_policy_can_require_topology_targets(tmp_path):
    source_root = tmp_path / "spectraxgk"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(2):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="target not met"):
        validate_architecture_policy(
            _architecture_manifest_with_topology(
                count_path=str(count_root), baseline=3, target=1
            ),
            source_root=source_root,
            check_paths=False,
            require_topology_targets=True,
        )

    summary = validate_architecture_policy(
        _architecture_manifest_with_topology(
            count_path=str(count_root), baseline=3, target=2
        ),
        source_root=source_root,
        check_paths=False,
        require_topology_targets=True,
    )
    assert summary["topology_targets_met"] is True


def test_package_architecture_inventory_classifies_repository_areas() -> None:
    mod = load_release_tool("check_package_architecture_manifest")

    role, action, notes = mod._role_and_action(
        Path("src/spectraxgk/operators/nonlinear/rhs.py")
    )
    tool_role, tool_action, tool_notes = mod._role_and_action(
        Path("tools/artifacts/make_figures.py")
    )
    summary = mod._summary(
        [
            mod.InventoryRow(
                path="src/spectraxgk/operators/nonlinear/rhs.py",
                area="src/spectraxgk/operators",
                role=role,
                action=action,
                suffix=".py",
                bytes=12,
                lines=1,
                notes=notes,
            ),
            mod.InventoryRow(
                path="tools/artifacts/make_figures.py",
                area="tools/artifacts",
                role=tool_role,
                action=tool_action,
                suffix=".py",
                bytes=8,
                lines=1,
                notes=tool_notes,
            ),
        ]
    )

    assert role == "promoted library code"
    assert action == "keep-and-consolidate"
    assert tool_role == "artifact builder"
    assert tool_action == "keep-or-merge"
    assert summary["keep-and-consolidate"] == {"files": 1, "bytes": 12}
    assert summary["keep-or-merge"] == {"files": 1, "bytes": 8}


def test_validate_architecture_policy_rejects_stale_allowlist(tmp_path):
    source_root = tmp_path / "spectraxgk"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="allowlist contains modules"):
        validate_architecture_policy(
            _architecture_manifest(allowed=["spectraxgk.nonlinear_removed_helper"]),
            source_root=source_root,
            check_paths=False,
        )


def test_repository_performance_manifest_is_well_formed() -> None:
    mod = _load_performance_manifest_tool()
    summary = mod.validate_manifest(mod.load_manifest())

    assert summary["n_lanes"] >= 5
    active = set(summary["high_priority_active"])
    assert "cold_start_compile" in active
    assert "nonlinear_warm_throughput" in active
    rows = {row["name"]: row for row in summary["rows"]}
    assert rows["end_to_end_runtime_memory"]["n_tools"] >= 2
    assert rows["parallel_scaling"]["priority"] == "medium"


def test_performance_manifest_main_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_performance_manifest_tool()
    out_json = tmp_path / "summary.json"

    assert mod.main(["--out-json", str(out_json)]) == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_lanes"] >= 5
    assert "memory_efficiency" in {row["name"] for row in payload["rows"]}


def test_performance_manifest_rejects_missing_tool(tmp_path: Path) -> None:
    mod = _load_performance_manifest_tool()
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _performance_manifest_text(
            tool="tools/missing.py", artifact="docs/_static/runtime.png"
        ),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="profiling tool does not exist"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_performance_manifest_accepts_benchmark_performance_driver(
    tmp_path: Path,
) -> None:
    mod = _load_performance_manifest_tool()
    tool = tmp_path / "benchmarks" / "performance" / "benchmark_runtime_memory.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# benchmark\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _performance_manifest_text(
            tool="benchmarks/performance/benchmark_runtime_memory.py",
            artifact="docs/_static/runtime.png",
        ),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        summary = mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root

    assert summary["rows"][0]["n_tools"] == 1


def test_performance_manifest_rejects_unowned_driver_path(tmp_path: Path) -> None:
    mod = _load_performance_manifest_tool()
    tool = tmp_path / "scripts" / "benchmark.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# benchmark\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _performance_manifest_text(
            tool="scripts/benchmark.py", artifact="docs/_static/runtime.png"
        ),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(
            ValueError,
            match=r"tools/ or benchmarks/performance/",
        ):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_performance_manifest_rejects_invalid_status(tmp_path: Path) -> None:
    mod = _load_performance_manifest_tool()
    tool = tmp_path / "tools" / "profile.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# tool\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _performance_manifest_text(
            tool="tools/profile.py",
            artifact="docs/_static/runtime.png",
            status="halfway",
        ),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="invalid status"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_documented_public_api_modules_have_manifest_tracking() -> None:
    direct_modules, owned_modules, excluded_modules = _repository_manifest_sets()
    tracked_modules = direct_modules | owned_modules | excluded_modules
    public_modules = _documented_public_api_modules()

    missing_source = sorted(
        module
        for module in public_modules
        if not _manifest_candidates_for_api_module(module)
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
    _write_package(
        tmp_path, "spectraxgk.runtime", "spectraxgk.workflows.runtime.config"
    )
    _write_fast_inputs(tmp_path)

    summary = _validate_tmp_coverage_manifest(
        tmp_path,
        _coverage_manifest(
            _coverage_row(
                "spectraxgk.runtime",
                owned_modules=["spectraxgk.workflows.runtime.config"],
            )
        ),
    )

    assert summary["n_direct_modules"] == 1
    assert summary["n_owned_modules"] == 1
    assert summary["n_excluded_modules"] == 1
    assert summary["owned_modules_by_owner"]["spectraxgk.runtime"] == [
        "spectraxgk.workflows.runtime.config"
    ]


def test_manifest_rejects_unowned_package_modules(tmp_path: Path) -> None:
    _write_package(
        tmp_path, "spectraxgk.runtime", "spectraxgk.workflows.runtime.config"
    )
    _write_fast_inputs(tmp_path)

    with pytest.raises(ValueError, match="package modules lack coverage ownership"):
        _validate_tmp_coverage_manifest(
            tmp_path, _coverage_manifest(_coverage_row("spectraxgk.runtime"))
        )


def test_manifest_rejects_duplicate_owned_modules(tmp_path: Path) -> None:
    _write_package(
        tmp_path,
        "spectraxgk.runtime",
        "spectraxgk.linear",
        "spectraxgk.workflows.runtime.config",
    )
    _write_fast_inputs(tmp_path)

    manifest = _coverage_manifest(
        _coverage_row(
            "spectraxgk.runtime", owned_modules=["spectraxgk.workflows.runtime.config"]
        ),
        _coverage_row(
            "spectraxgk.linear", owned_modules=["spectraxgk.workflows.runtime.config"]
        ),
    )
    with pytest.raises(ValueError, match="duplicate coverage ownership"):
        _validate_tmp_coverage_manifest(tmp_path, manifest)


def test_manifest_rejects_direct_rows_listed_as_owned_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "spectraxgk.runtime", "spectraxgk.linear")
    _write_fast_inputs(tmp_path)

    manifest = _coverage_manifest(
        _coverage_row("spectraxgk.runtime", owned_modules=["spectraxgk.linear"]),
        _coverage_row("spectraxgk.linear"),
    )
    with pytest.raises(
        ValueError, match="direct manifest rows must not be listed as owned modules"
    ):
        _validate_tmp_coverage_manifest(tmp_path, manifest)
