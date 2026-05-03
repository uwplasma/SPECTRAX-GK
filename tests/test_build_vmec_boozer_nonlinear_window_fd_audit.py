from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_vmec_boozer_nonlinear_window_fd_audit.py"
    spec = importlib.util.spec_from_file_location("build_vmec_boozer_nonlinear_window_fd_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_geom(scale: float = 1.0):
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    ones = np.ones_like(theta)
    return SimpleNamespace(
        theta=theta,
        gradpar=lambda: 0.7 * scale,
        bmag_profile=scale * (1.0 + 0.05 * np.cos(theta)),
        bgrad_profile=scale * 0.05 * np.sin(theta),
        gds2_profile=scale * ones,
        gds21_profile=0.02 * scale * np.sin(theta),
        gds22_profile=scale * (1.0 + 0.03 * np.cos(theta)),
        cv_profile=0.1 * scale * np.cos(theta),
        gb_profile=0.1 * scale * np.cos(theta),
        cv0_profile=np.zeros_like(theta),
        gb0_profile=np.zeros_like(theta),
        jacobian_profile=ones / scale,
        grho_profile=scale * ones,
        q=1.4 * scale,
        s_hat=0.8 * scale,
        epsilon=0.18,
        R0=2.7,
        alpha=0.0,
        kxfac=1.0,
        theta_scale=1.0,
        nfp=4,
    )


def _synthetic_run(label: str, perturbation: float, scale: float = 1.0) -> dict:
    mod = _load_tool_module()
    time = np.linspace(0.0, 1.0, 8)
    heat = scale * (1.0 + 0.08 * time)
    return {
        "label": label,
        "perturbation": perturbation,
        "geometry_file_name": f"{label}.nc",
        "geometry_response": mod.geometry_response_metrics(_fake_geom(), _fake_geom(1.0 + abs(perturbation))),
        "time": time.tolist(),
        "heat_flux": heat.tolist(),
        "window": mod.late_window_metrics(time, heat, tail_fraction=0.5),
    }


def test_geometry_response_metrics_reports_profile_and_scalar_changes() -> None:
    mod = _load_tool_module()

    metrics = mod.geometry_response_metrics(_fake_geom(), _fake_geom(1.01))

    assert metrics["max_relative_change"] > 0.0
    assert metrics["per_profile"]["bmag"] > 0.0
    assert metrics["per_scalar"]["q"] > 0.0


def test_build_vmec_boozer_audit_payload_passes_conditioned_response() -> None:
    mod = _load_tool_module()
    runs = [
        _synthetic_run("minus", -1.0e-5, 0.80),
        _synthetic_run("base", 0.0, 1.00),
        _synthetic_run("plus", 1.0e-5, 1.25),
        _synthetic_run("base_repeat", 0.0, 1.00),
    ]

    payload = mod.build_vmec_boozer_audit_payload(
        runs,
        case_name="nfp4_QH_warm_start",
        parameter_name="Rcos_r1_m1",
        perturbation_step=1.0e-5,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
        min_geometry_relative_change=1.0e-7,
    )

    assert payload["passed"] is True
    assert payload["vmec_boozer_production_nonlinear_observable_fd_path_gate"] is True
    assert payload["production_nonlinear_window_gradient_gate"] is False
    assert payload["gates"]["geometry_perturbation_resolved"] is True
    assert payload["metrics"]["central_fd_dq_dparameter"] > 0.0


def test_build_vmec_boozer_audit_payload_blocks_unresolved_geometry() -> None:
    mod = _load_tool_module()
    runs = [
        _synthetic_run("minus", -1.0e-10, 0.80),
        _synthetic_run("base", 0.0, 1.00),
        _synthetic_run("plus", 1.0e-10, 1.25),
        _synthetic_run("base_repeat", 0.0, 1.00),
    ]

    payload = mod.build_vmec_boozer_audit_payload(
        runs,
        case_name="nfp4_QH_warm_start",
        parameter_name="Rcos_r1_m1",
        perturbation_step=1.0e-10,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
        min_geometry_relative_change=1.0e-7,
    )

    assert payload["passed"] is False
    assert payload["gates"]["geometry_perturbation_resolved"] is False


def test_main_writes_artifacts_without_running_solver(monkeypatch, tmp_path: Path) -> None:
    mod = _load_tool_module()

    monkeypatch.setattr(
        mod,
        "_mode21_vmec_boozer_linear_context",
        lambda **_kwargs: {"parameter_names": ("Rcos_r1_m1",), "geometry_for": lambda _x: _fake_geom()},
    )

    def fake_run_vmec_boozer_window(*, label: str, perturbation: float, **_kwargs):
        scale = {"minus": 0.80, "base": 1.00, "plus": 1.25, "base_repeat": 1.00}[label]
        return _synthetic_run(label, perturbation, scale)

    monkeypatch.setattr(mod, "run_vmec_boozer_window", fake_run_vmec_boozer_window)
    out = tmp_path / "audit.png"

    assert mod.main(["--out", str(out), "--tail-fraction", "0.5"]) == 0

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["passed"] is True
    assert (
        meta["claim_level"]
        == "vmec_boozer_geometry_perturbed_production_nonlinear_observable_fd_path_not_optimization_claim"
    )
