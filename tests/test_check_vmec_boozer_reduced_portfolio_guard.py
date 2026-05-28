from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from spectraxgk.stellarator_objective_portfolio import (
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_vmec_boozer_reduced_portfolio_guard.py"
spec = importlib.util.spec_from_file_location("check_vmec_boozer_reduced_portfolio_guard", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _row_artifact() -> dict[str, object]:
    samples = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "artifact_kind": "vmec_boozer_multi_point_objective_gate",
        "builder": "tools/build_vmec_boozer_multi_point_objective_gate.py",
        "passed": True,
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": "real VMEC/Boozer reduced QL rows; not a nonlinear turbulent transport claim",
        "next_action": "Nonlinear transport optimization still requires separate long-window gates.",
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "input_path": "/tmp/input.nfp4_QH_warm_start",
        "wout_path": "/tmp/wout_nfp4_QH_warm_start.nc",
        "options": {"mboz": 21, "nboz": 21},
        "n_samples": 4,
        "samples": samples,
        "objective_names": ["gamma", "omega", "kperp_eff2", "mixing_length_heat_flux_proxy"],
        "minus_sample_values": [0.7, 0.9, 0.8, 1.0],
        "base_sample_values": [0.8, 1.0, 0.9, 1.1],
        "plus_sample_values": [0.9, 1.1, 1.0, 1.2],
        "minus_objective_table": [
            [0.10, -0.2, 0.5, 0.7],
            [0.12, -0.3, 0.6, 0.9],
            [0.11, -0.1, 0.4, 0.8],
            [0.13, -0.4, 0.7, 1.0],
        ],
        "base_objective_table": [
            [0.11, -0.2, 0.5, 0.8],
            [0.13, -0.3, 0.6, 1.0],
            [0.12, -0.1, 0.4, 0.9],
            [0.14, -0.4, 0.7, 1.1],
        ],
        "plus_objective_table": [
            [0.12, -0.2, 0.5, 0.9],
            [0.14, -0.3, 0.6, 1.1],
            [0.13, -0.1, 0.4, 1.0],
            [0.15, -0.4, 0.7, 1.2],
        ],
        "base_value": 0.95,
        "minus_value": 0.85,
        "plus_value": 1.05,
        "central_derivative": 1.0,
        "response_abs": 0.2,
        "curvature_ratio": 0.01,
        "finite_values": True,
        "finite_difference_consistent": True,
        "response_resolved": True,
    }


def _gradient_artifact() -> dict[str, object]:
    return {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": True,
        "objective_gates": [
            {
                "objective": "gamma",
                "parameter": "Rcos_mid_surface_m1",
                "passed": True,
                "implicit": 10.0,
                "finite_difference": 10.01,
                "abs_error": 0.01,
                "rel_error": 0.001,
            },
            {
                "objective": "mixing_length_heat_flux_proxy",
                "parameter": "Rcos_mid_surface_m1",
                "passed": True,
                "implicit": 20.0,
                "finite_difference": 20.02,
                "abs_error": 0.02,
                "rel_error": 0.001,
            },
        ],
    }


def test_reduced_portfolio_guard_passes_real_metadata_contract() -> None:
    report = reduced_portfolio_artifact_guard_report(
        _row_artifact(),
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is True
    assert report["provenance_gate"]["passed"] is True
    assert report["coverage_gate"]["n_alphas"] == 2
    assert report["coverage_gate"]["n_ky"] == 2
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == [1, 2, 2, 1]
    assert report["ad_fd_gradient_gate"]["has_growth_ad_fd_gate"] is True
    assert report["ad_fd_gradient_gate"]["has_quasilinear_ad_fd_gate"] is True
    assert report["claim_scope_gate"]["passed"] is True


def test_reduced_portfolio_guard_distinguishes_physical_torflux_surfaces() -> None:
    artifact = _row_artifact()
    artifact["samples"] = [
        {
            "surface_index": None,
            "torflux": 0.5,
            "surface": 0.5,
            "alpha": 0.0,
            "ky": 0.1,
            "selected_ky_index": 1,
            "weight": 0.25,
        },
        {
            "surface_index": None,
            "torflux": 0.5,
            "surface": 0.5,
            "alpha": 0.0,
            "ky": 0.2,
            "selected_ky_index": 2,
            "weight": 0.25,
        },
        {
            "surface_index": None,
            "torflux": 0.7,
            "surface": 0.7,
            "alpha": 0.0,
            "ky": 0.1,
            "selected_ky_index": 1,
            "weight": 0.25,
        },
        {
            "surface_index": None,
            "torflux": 0.7,
            "surface": 0.7,
            "alpha": 0.0,
            "ky": 0.2,
            "selected_ky_index": 2,
            "weight": 0.25,
        },
    ]
    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
        config=ReducedPortfolioArtifactGuardConfig(min_alphas=1),
    )

    assert report["passed"] is True
    assert report["coverage_gate"]["n_surfaces"] == 2
    assert report["coverage_gate"]["n_alphas"] == 1
    assert report["coverage_gate"]["n_ky"] == 2
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == [2, 1, 2, 1]


def test_reduced_portfolio_guard_fails_single_alpha_or_missing_gradient_gate() -> None:
    artifact = _row_artifact()
    artifact["samples"] = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.5},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.5},
    ]
    artifact["base_sample_values"] = [0.8, 1.0]
    artifact["minus_sample_values"] = [0.7, 0.9]
    artifact["plus_sample_values"] = [0.9, 1.1]
    artifact["base_objective_table"] = [[0.11, -0.2, 0.5, 0.8], [0.13, -0.3, 0.6, 1.0]]
    artifact["minus_objective_table"] = [[0.10, -0.2, 0.5, 0.7], [0.12, -0.3, 0.6, 0.9]]
    artifact["plus_objective_table"] = [[0.12, -0.2, 0.5, 0.9], [0.14, -0.3, 0.6, 1.1]]
    artifact["base_value"] = 0.9
    artifact["minus_value"] = 0.8
    artifact["plus_value"] = 1.0

    report = reduced_portfolio_artifact_guard_report(artifact, gradient_artifacts=[])

    assert report["passed"] is False
    assert report["coverage_gate"]["passed"] is False
    assert report["ad_fd_gradient_gate"]["passed"] is False


def test_reduced_portfolio_guard_fails_production_nonlinear_claim() -> None:
    artifact = _row_artifact()
    artifact["objective"] = "nonlinear_heat_flux"
    artifact["claim_scope"] = "production nonlinear turbulent transport optimization claim"

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["claim_scope_gate"]["passed"] is False


def test_reduced_portfolio_guard_validates_config() -> None:
    with pytest.raises(ValueError, match="min_alphas"):
        ReducedPortfolioArtifactGuardConfig(min_alphas=0)


def test_tool_writes_guard_artifact(tmp_path: Path) -> None:
    row_path = tmp_path / "row.json"
    gradient_path = tmp_path / "gradient.json"
    out_path = tmp_path / "guard.json"
    row_path.write_text(json.dumps(_row_artifact()), encoding="utf-8")
    gradient_path.write_text(json.dumps(_gradient_artifact()), encoding="utf-8")

    payload = mod.build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=row_path,
        gradient_artifacts=[gradient_path],
    )
    written = mod.write_vmec_boozer_reduced_portfolio_guard_artifact(payload, out=out_path)

    assert Path(written) == out_path
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["row_artifact"] == str(row_path)


def test_tool_exposes_reducer_value_tolerances(tmp_path: Path) -> None:
    row = _row_artifact()
    row["base_value"] = 0.95000004
    row_path = tmp_path / "row.json"
    gradient_path = tmp_path / "gradient.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")
    gradient_path.write_text(json.dumps(_gradient_artifact()), encoding="utf-8")

    strict_payload = mod.build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=row_path,
        gradient_artifacts=[gradient_path],
    )
    loose_payload = mod.build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=row_path,
        gradient_artifacts=[gradient_path],
        value_rtol=1.0e-6,
        value_atol=1.0e-6,
    )

    assert strict_payload["portfolio_reducer_gate"]["passed"] is False
    assert strict_payload["passed"] is False
    assert loose_payload["portfolio_reducer_gate"]["passed"] is True
    assert loose_payload["passed"] is True


def test_tool_main_returns_nonzero_for_failed_guard(tmp_path: Path) -> None:
    row = _row_artifact()
    row["options"] = {"mboz": 8, "nboz": 8}
    row_path = tmp_path / "row.json"
    gradient_path = tmp_path / "gradient.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")
    gradient_path.write_text(json.dumps(_gradient_artifact()), encoding="utf-8")

    result = mod.main(
        [
            "--row-artifact",
            str(row_path),
            "--gradient-artifact",
            str(gradient_path),
            "--out",
            str(tmp_path / "guard.json"),
        ]
    )

    assert result == 1
