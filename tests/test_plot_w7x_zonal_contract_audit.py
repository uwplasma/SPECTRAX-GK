from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_w7x_zonal_contract_audit.py"
    spec = importlib.util.spec_from_file_location("plot_w7x_zonal_contract_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    reference_trace_rows = []
    spectrax_trace_rows = []
    compare_rows = []
    residual_rows = []
    for kx in (0.05, 0.07, 0.10, 0.30):
        t = np.linspace(0.0, 20.0, 31)
        response = 0.1 + kx + 0.04 * np.exp(-0.12 * t) * np.cos(0.7 * t)
        for code, offset in (("stella", -0.002), ("GENE", 0.002)):
            for time_value, value in zip(t, response + offset, strict=True):
                reference_trace_rows.append(
                    {
                        "kx_rhoi": kx,
                        "code": code,
                        "t_vti_over_a": time_value,
                        "response": value,
                    }
                )
            residual_rows.append(
                {
                    "kx_rhoi": kx,
                    "code": code,
                    "residual_median": 0.1 + kx + offset,
                }
            )
        spectrax_response = response + (0.004 if kx < 0.1 else 0.02)
        for time_value, value in zip(t, spectrax_response, strict=True):
            spectrax_trace_rows.append(
                {
                    "kx_target": kx,
                    "kx_selected": kx,
                    "t_reference": time_value,
                    "phi_zonal_real": value,
                    "response_normalized": value,
                    "initial_level": 1.0,
                    "initial_normalization": "line_first",
                    "source_path": "synthetic.nc",
                }
            )
        compare_rows.append(
            {
                "kx": kx,
                "spectrax_residual": 0.1 + kx + (0.004 if kx < 0.1 else 0.02),
                "spectrax_residual_std": 0.01,
                "spectrax_tmax": 20.0,
                "reference_residual": 0.1 + kx,
                "reference_min": 0.1 + kx - 0.002,
                "reference_max": 0.1 + kx + 0.002,
                "reference_tmax": 20.0,
                "coverage_ratio": 1.0,
                "residual_abs_error": 0.004 if kx < 0.1 else 0.02,
                "residual_atol_effective": 0.02,
                "trace_available": 1,
                "tail_std": 0.01 + 0.1 * kx,
                "reference_tail_std": 0.01,
                "tail_mean_abs_error": 0.003,
                "tail_max_abs_error": 0.005,
            }
        )
    ref_traces = tmp_path / "reference_traces.csv"
    ref_residuals = tmp_path / "reference_residuals.csv"
    summary = tmp_path / "spectrax_summary.csv"
    traces = tmp_path / "spectrax_traces.csv"
    compare = tmp_path / "compare.csv"
    pd.DataFrame(reference_trace_rows).to_csv(ref_traces, index=False)
    pd.DataFrame(residual_rows).to_csv(ref_residuals, index=False)
    pd.DataFrame({"kx_target": [0.05], "residual_level": [0.1], "residual_std": [0.01], "tmax": [20.0]}).to_csv(
        summary,
        index=False,
    )
    pd.DataFrame(spectrax_trace_rows).to_csv(traces, index=False)
    pd.DataFrame(compare_rows).to_csv(compare, index=False)
    return ref_traces, ref_residuals, summary, traces, compare


def test_w7x_zonal_contract_audit_rows_and_main(tmp_path: Path) -> None:
    mod = _load_tool_module()
    ref_traces, ref_residuals, summary, traces, compare = _write_inputs(tmp_path)
    rows = mod.load_audit_rows(compare)

    assert len(rows) == 4
    assert rows[0]["residual_gate_passed"] is True
    assert rows[-1]["tail_std_ratio"] > 1.0

    out_png = tmp_path / "audit.png"
    out_csv = tmp_path / "audit.csv"
    out_json = tmp_path / "audit.json"
    rc = mod.main(
        [
            "--reference-traces",
            str(ref_traces),
            "--reference-residuals",
            str(ref_residuals),
            "--spectrax-summary",
            str(summary),
            "--spectrax-traces",
            str(traces),
            "--compare-csv",
            str(compare),
            "--out-png",
            str(out_png),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_csv.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
    assert payload["reference_contract"]["normalization"].startswith("line-averaged potential")
