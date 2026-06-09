from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "write_vmec_qa_t1500_postprocess_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("write_vmec_qa_t1500_postprocess_manifest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_contains_fail_closed_case_and_comparison_commands() -> None:
    mod = _load_tool_module()

    manifest = mod.build_manifest(
        run_root="tools_out/audits",
        netcdf_root="office:/work/audits",
        cases=("baseline", "quasilinear"),
        min_relative_reduction=0.04,
    )

    assert manifest["kind"] == "vmec_qa_t1500_postprocess_manifest"
    assert manifest["window"] == {"tmin": 1100.0, "tmax": 1500.0}
    assert len(manifest["case_commands"]) == 2
    ql = manifest["case_commands"][1]
    assert ql["case"] == "quasilinear"
    assert "vmec_qa_full_sweep_quasilinear_from_strict_baseline" in ql["outputs"][0]
    assert ql["output_gate_json"] == "docs/_static/vmec_qa_t1500_quasilinear_output_gate.json"
    assert "--min-window-samples 80" in ql["check_outputs_command"]
    assert "--json-out docs/_static/vmec_qa_t1500_quasilinear_output_gate.json" in ql[
        "check_outputs_command"
    ]
    assert "compact_replicate_ensemble_bundle.py" in ql["compact_bundle_command"]
    assert "--output-gate-json docs/_static/vmec_qa_t1500_quasilinear_output_gate.json" in ql[
        "compact_bundle_command"
    ]
    assert "office:/work/audits" in ql["compact_bundle_command"]
    comparisons = manifest["comparison_commands"]
    assert len(comparisons) == 1
    assert comparisons[0]["candidate"] == "quasilinear"
    assert "--min-relative-reduction 0.04" in comparisons[0]["command"]
    assert "qa_baseline_scipy_t1500_ensemble_gate.json" in comparisons[0]["command"]


def test_manifest_cli_writes_selected_cases(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out = tmp_path / "manifest.json"

    rc = mod.main(
        [
            "--case",
            "baseline",
            "--case",
            "growth",
            "--run-root",
            "tools_out/demo",
            "--netcdf-root",
            "office:/demo",
            "--out-json",
            str(out),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [row["case"] for row in payload["case_commands"]] == ["baseline", "growth"]
    assert payload["comparison_commands"][0]["candidate"] == "growth"
    assert payload["case_commands"][0]["outputs"][2].endswith("_dt0p04.out.nc")
