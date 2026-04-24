from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from spectraxgk.benchmarking import EigenfunctionComparisonMetrics, save_eigenfunction_reference_bundle


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "generate_w7x_reference_overlay.py"
    spec = importlib.util.spec_from_file_location("generate_w7x_reference_overlay", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_w7x_reference_loader_rejects_nonfinite_bundle(tmp_path: Path) -> None:
    mod = _load_tool_module()
    bundle = tmp_path / "bad_w7x_ref.npz"
    save_eigenfunction_reference_bundle(
        bundle,
        theta=np.array([-1.0, 0.0, 1.0]),
        mode=np.array([1.0 + 0.0j, np.nan + 0.0j, 0.5 + 0.0j]),
        source="GX",
        case="w7x_linear",
    )

    with pytest.raises(ValueError, match="non-finite reference mode"):
        mod._load_finite_reference(bundle)


def test_w7x_eigenfunction_gate_report_uses_strict_publication_thresholds() -> None:
    mod = _load_tool_module()

    report = mod._w7x_eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.50, relative_l2=0.80, phase_shift=0.0)
    )

    assert report.case == "w7x_linear_eigenfunction_ky0p3000"
    assert report.source == "GX raw eigenfunction bundle"
    assert report.passed is False
    assert mod.W7X_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"] == 0.95
    assert mod.W7X_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"] == 0.25


def test_w7x_overlay_main_writes_gate_artifacts(tmp_path: Path, monkeypatch) -> None:
    mod = _load_tool_module()
    theta = np.linspace(-np.pi, np.pi, 32)
    reference = np.cos(theta) + 0.25j * np.sin(theta)
    bundle = tmp_path / "w7x_ref.npz"
    spectrax_csv = tmp_path / "w7x_spectrax.csv"
    out_png = tmp_path / "w7x_overlay.png"
    out_json = tmp_path / "w7x_overlay.json"
    save_eigenfunction_reference_bundle(
        bundle,
        theta=theta,
        mode=reference,
        source="GX",
        case="w7x_linear",
        metadata={"ky": 0.3},
    )

    def fake_gx_reference(_path):
        time = np.array([0.0, 1.0])
        ky = np.array([0.0, 0.3])
        kx = np.array([0.0])
        zero = np.zeros((time.size, ky.size, kx.size), dtype=float)
        return time, ky, kx, zero, zero, zero, zero, zero

    def fake_run(_args, *, reference_times, output_steps):
        assert np.array_equal(reference_times, np.array([0.0, 1.0]))
        assert np.array_equal(output_steps, np.array([0, 1]))
        return {
            "theta": theta,
            "mode": reference * np.exp(0.37j),
            "gamma_last": 0.0093,
            "omega_last": -0.2319,
            "Wg_last": 1.0,
            "Wphi_last": 2.0,
            "Wapar_last": 0.0,
            "Phi2_last": 3.0,
            "t_final": 1.0,
            "nl": 8,
            "nm": 16,
            "ny": 82,
            "kx_local": 0.0,
            "kx_ref": 0.0,
        }

    monkeypatch.setattr(mod, "_load_gx_reference", fake_gx_reference)
    monkeypatch.setattr(mod, "_run_w7x_spectrax_mode", fake_run)

    mod.main(
        [
            "--gx",
            str(tmp_path / "dummy.out.nc"),
            "--gx-input",
            str(tmp_path / "dummy.in"),
            "--geometry-file",
            str(tmp_path / "dummy.eik.nc"),
            "--bundle-out",
            str(bundle),
            "--out-csv",
            str(spectrax_csv),
            "--out-png",
            str(out_png),
            "--out-json",
            str(out_json),
        ]
    )

    assert spectrax_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["eigenfunction_gate_passed"] is True
    assert data["validation_status"] == "closed"
    assert data["gate_report"]["case"] == "w7x_linear_eigenfunction_ky0p3000"
