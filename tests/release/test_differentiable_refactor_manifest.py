from __future__ import annotations

import importlib.util
from pathlib import Path

from support.paths import REPO_ROOT


def _load_tool_module():
    path = REPO_ROOT / "tools" / "release" / "check_differentiable_refactor_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "check_differentiable_refactor_manifest", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_differentiable_refactor_manifest_is_well_formed() -> None:
    mod = _load_tool_module()
    summary = mod.validate_manifest(mod.load_manifest())
    manifest = mod.load_manifest()
    assert summary["required_package_coverage_percent"] >= 95.0
    assert manifest["global_acceptance"]["require_adaptive_derivative_policy"] is True
    assert (
        "adaptive-branch derivative policy"
        in (manifest["validation_policy"]["autodiff_gate_scope"])
    )
    assert summary["n_architecture_layers"] >= 8
    assert summary["n_phase1_contract_modules"] >= 2
    assert summary["n_phase1_split_modules"] >= 16
    assert summary["n_hotspots"] >= 9
    assert "spectraxgk.core.contracts" in summary["phase1_contract_modules"]
    assert "spectraxgk.core.extension_points" in summary["phase1_contract_modules"]
    assert (
        "spectraxgk.validation.benchmarks.initialization"
        in summary["phase1_split_modules"]
    )
    assert "spectraxgk.validation.benchmarks.species" in summary["phase1_split_modules"]
    assert (
        "spectraxgk.diagnostics.growth_rates"
        in summary["phase1_split_modules"]
    )
    assert "spectraxgk.geometry.backend_discovery" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.autodiff_checks" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.numerics" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.flux_tube_contract" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.sensitivity" in summary["phase1_split_modules"]
    assert "spectraxgk.geometry.booz_xform_bridge" in summary["phase1_split_modules"]
    assert (
        "spectraxgk.geometry.vmec_state_sensitivity" in summary["phase1_split_modules"]
    )
    assert "spectraxgk.geometry.vmec_boozer_core" in summary["phase1_split_modules"]
    assert (
        "spectraxgk.geometry.vmec_flux_tube_reports" in summary["phase1_split_modules"]
    )
    assert "spectraxgk.geometry.vmec_tensor_mapping" in summary["phase1_split_modules"]
    assert "spectraxgk.objectives.gradient_gates" in summary["phase1_split_modules"]
    assert (
        "spectraxgk.objectives.vmec_boozer_gradients" in summary["phase1_split_modules"]
    )
    assert "spectraxgk.objectives.vmec_boozer_fd" in summary["phase1_split_modules"]
    assert (
        "spectraxgk.objectives.vmec_boozer_line_search"
        in summary["phase1_split_modules"]
    )
    assert "spectraxgk.objectives.vmec_boozer" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.nonlinear.rhs" in summary["phase1_split_modules"]
    assert (
        "spectraxgk.operators.nonlinear.diagnostic_state"
        in summary["phase1_split_modules"]
    )
    assert "spectraxgk.solvers.nonlinear.explicit" in summary["phase1_split_modules"]
    assert "spectraxgk.solvers.nonlinear.imex" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.linear.cache" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.linear.moments" in summary["phase1_split_modules"]
    assert "spectraxgk.operators.linear.params" in summary["phase1_split_modules"]
    assert "spectraxgk.solvers.linear.krylov" in summary["phase1_split_modules"]
    assert "spectraxgk.solvers.linear.parallel" in summary["phase1_split_modules"]
    assert "spectraxgk.workflows.cases" in summary["phase1_split_modules"]
    assert "spectraxgk.artifacts.validation" in summary["phase1_split_modules"]
    assert "spectraxgk.artifacts.io" in summary["phase1_split_modules"]
    assert "spectraxgk.artifacts.linear" in summary["phase1_split_modules"]
    assert "spectraxgk.artifacts.nonlinear" in summary["phase1_split_modules"]
    assert (
        "spectraxgk.artifacts.nonlinear_diagnostics" in summary["phase1_split_modules"]
    )
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
    mod = _load_tool_module()
    out_json = tmp_path / "summary.json"
    assert mod.main(["--out-json", str(out_json)]) == 0
    payload = out_json.read_text(encoding="utf-8")
    assert "Differentiable architecture refactor plan" in payload
    assert "spectraxgk.geometry.differentiable" in payload
