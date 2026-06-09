from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "compact_replicate_ensemble_bundle.py"
    )
    spec = importlib.util.spec_from_file_location("compact_replicate_ensemble_bundle", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload() -> dict:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "passed": True,
        "rows": [
            {
                "index": 0,
                "source_artifact": (
                    "docs/_static/demo/demo_case_nonlinear_t1500_n64_seed32"
                    "_heat_flux_trace.csv"
                ),
                "summary_artifact": (
                    "docs/_static/demo/demo_case_nonlinear_t1500_n64_seed32"
                    "_transport_window.json"
                ),
            },
            {
                "index": 1,
                "source_artifact": (
                    "docs/_static/demo/demo_case_nonlinear_t1500_n64_dt0p04"
                    "_heat_flux_trace.csv"
                ),
                "summary_artifact": (
                    "docs/_static/demo/demo_case_nonlinear_t1500_n64_dt0p04"
                    "_transport_window.json"
                ),
            },
        ],
    }


def test_compact_ensemble_payload_rewrites_rows_to_authoritative_netcdf() -> None:
    mod = _load_tool_module()

    compact = mod.compact_ensemble_payload(
        _payload(),
        output_gate_json="docs/_static/demo_output_gate.json",
        netcdf_root="office:/work/run/tools_out/audits",
    )

    assert compact["compact_bundle_policy"]["output_gate_json"] == (
        "docs/_static/demo_output_gate.json"
    )
    assert compact["rows"][0]["generated_trace_artifact"].endswith("_heat_flux_trace.csv")
    assert compact["rows"][0]["source_artifact"] == (
        "office:/work/run/tools_out/audits/demo_case/"
        "demo_case_nonlinear_t1500_n64_seed32.out.nc"
    )
    assert compact["rows"][1]["source_artifact"] == (
        "office:/work/run/tools_out/audits/demo_case/"
        "demo_case_nonlinear_t1500_n64_dt0p04.out.nc"
    )
    assert compact["rows"][1]["summary_artifact"] == (
        "docs/_static/demo_output_gate.json#rows[1]"
    )


def test_compact_ensemble_cli_writes_requested_output(tmp_path: Path) -> None:
    mod = _load_tool_module()
    ensemble = tmp_path / "ensemble.json"
    out = tmp_path / "compact.json"
    ensemble.write_text(json.dumps(_payload()), encoding="utf-8")

    rc = mod.main(
        [
            "--ensemble-json",
            str(ensemble),
            "--output-gate-json",
            "docs/_static/demo_output_gate.json",
            "--netcdf-root",
            "office:/work/run/tools_out/audits/",
            "--out-json",
            str(out),
        ]
    )

    assert rc == 0
    compact = json.loads(out.read_text(encoding="utf-8"))
    assert compact["rows"][0]["source_artifact"].startswith(
        "office:/work/run/tools_out/audits/demo_case/"
    )
    assert ensemble.read_text(encoding="utf-8") == json.dumps(_payload())
