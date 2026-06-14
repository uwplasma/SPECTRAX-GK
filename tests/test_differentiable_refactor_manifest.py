from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool_module():
    path = REPO_ROOT / "tools" / "check_differentiable_refactor_manifest.py"
    spec = importlib.util.spec_from_file_location("check_differentiable_refactor_manifest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_differentiable_refactor_manifest_is_well_formed() -> None:
    mod = _load_tool_module()
    summary = mod.validate_manifest(mod.load_manifest())
    assert summary["required_package_coverage_percent"] >= 95.0
    assert summary["n_architecture_layers"] >= 8
    assert summary["n_hotspots"] >= 9
    for module in (
        "spectraxgk.benchmarks",
        "spectraxgk.geometry.differentiable",
        "spectraxgk.nonlinear_parallel",
        "spectraxgk.solver_objective_gradients",
        "spectraxgk.nonlinear",
        "spectraxgk.runtime_artifacts",
        "spectraxgk.runtime",
        "spectraxgk.linear",
        "spectraxgk.cli",
    ):
        assert module in summary["hotspot_modules"]


def test_differentiable_refactor_manifest_main_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out_json = tmp_path / "summary.json"
    assert mod.main(["--out-json", str(out_json)]) == 0
    payload = out_json.read_text(encoding="utf-8")
    assert "Differentiable architecture refactor plan" in payload
    assert "spectraxgk.geometry.differentiable" in payload
