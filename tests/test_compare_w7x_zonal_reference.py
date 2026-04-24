from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "compare_w7x_zonal_reference.py"
    spec = importlib.util.spec_from_file_location("compare_w7x_zonal_reference", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_reference(tmp_path: Path) -> tuple[Path, Path]:
    traces = []
    residuals = []
    for kx in (0.05, 0.07, 0.10, 0.30):
        tmax = 3500.0 if kx == 0.05 else 2000.0
        t = np.linspace(0.0, tmax, 21)
        for code, offset in (("stella", -0.002), ("GENE", 0.002)):
            residual = 0.1 + kx + offset
            for tv in t:
                traces.append({"kx_rhoi": kx, "code": code, "t_vti_over_a": tv, "response": residual + np.exp(-tv / 200.0)})
            residuals.append(
                {
                    "panel": "x",
                    "kx_rhoi": kx,
                    "code": code,
                    "residual_mean": residual,
                    "residual_median": residual,
                    "residual_min": residual,
                    "residual_max": residual,
                    "n_pixels": 5,
                }
            )
    trace_csv = tmp_path / "ref_traces.csv"
    residual_csv = tmp_path / "ref_residuals.csv"
    pd.DataFrame(traces).to_csv(trace_csv, index=False)
    pd.DataFrame(residuals).to_csv(residual_csv, index=False)
    return trace_csv, residual_csv


def _write_summary(tmp_path: Path, *, tmax_scale: float = 1.0, residual_shift: float = 0.0) -> Path:
    rows = []
    for kx in (0.05, 0.07, 0.10, 0.30):
        ref_tmax = 3500.0 if kx == 0.05 else 2000.0
        rows.append(
            {
                "kx_target": kx,
                "residual_level": 0.1 + kx + residual_shift,
                "residual_std": 0.01,
                "tmax": ref_tmax * tmax_scale,
            }
        )
    path = tmp_path / "spectrax_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_w7x_zonal_reference_comparison_passes_closed_synthetic_case(tmp_path: Path) -> None:
    mod = _load_tool_module()
    ref_traces, ref_residuals = _write_reference(tmp_path)
    summary = _write_summary(tmp_path)

    rows, report = mod.build_comparison(
        spectrax_summary=summary,
        reference_traces=ref_traces,
        reference_residuals=ref_residuals,
    )

    assert report.passed is True
    assert rows["coverage_ratio"].min() == 1.0
    assert rows["residual_abs_error"].max() <= 1.0e-12


def test_w7x_zonal_reference_comparison_fails_short_window(tmp_path: Path) -> None:
    mod = _load_tool_module()
    ref_traces, ref_residuals = _write_reference(tmp_path)
    summary = _write_summary(tmp_path, tmax_scale=0.03)

    rows, report = mod.build_comparison(
        spectrax_summary=summary,
        reference_traces=ref_traces,
        reference_residuals=ref_residuals,
    )

    assert report.passed is False
    assert rows["coverage_ratio"].max() < 0.98
    failed = [gate.metric for gate in report.gates if not gate.passed]
    assert "time_coverage_kx050" in failed


def test_w7x_zonal_reference_main_writes_open_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    ref_traces, ref_residuals = _write_reference(tmp_path)
    summary = _write_summary(tmp_path, residual_shift=1.0)
    out_csv = tmp_path / "compare.csv"
    out_json = tmp_path / "compare.json"
    out_png = tmp_path / "compare.png"

    rc = mod.main(
        [
            "--spectrax-summary",
            str(summary),
            "--reference-traces",
            str(ref_traces),
            "--reference-residuals",
            str(ref_residuals),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 2
    assert out_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
