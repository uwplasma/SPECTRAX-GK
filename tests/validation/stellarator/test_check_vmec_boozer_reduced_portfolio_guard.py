from __future__ import annotations

import json
from pathlib import Path

from support.paths import load_release_tool

import pytest

from spectraxgk.objectives.portfolio_artifacts import (
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)


mod = load_release_tool("check_vmec_boozer_reduced_portfolio_guard")


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
        "builder": "tools/artifacts/build_vmec_boozer_multi_point_objective_gate.py",
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
        "objective_names": [
            "gamma",
            "omega",
            "kperp_eff2",
            "mixing_length_heat_flux_proxy",
        ],
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


@pytest.mark.parametrize(
    ("reduction", "base_value", "expected_shape"),
    [
        ("weighted_mean", 0.95, [1, 2, 2, 1]),
        ("max", 1.0, [1, 2, 2, 1]),
    ],
)
def test_reduced_portfolio_guard_accepts_declared_reducer_semantics(
    reduction: str,
    base_value: float,
    expected_shape: list[int],
) -> None:
    artifact = _row_artifact()
    artifact["reduction"] = reduction
    if reduction == "max":
        artifact["base_sample_values"] = [0.5, 0.75, 0.875, 1.0]
    artifact["base_value"] = base_value

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is True
    assert report["portfolio_reducer_gate"]["reduction"] == reduction
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == expected_shape


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


def test_reduced_portfolio_guard_rejects_duplicate_or_incomplete_sample_grids() -> None:
    duplicate = _row_artifact()
    duplicate["samples"] = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        reduced_portfolio_artifact_guard_report(
            duplicate,
            gradient_artifacts=[_gradient_artifact()],
        )

    incomplete = _row_artifact()
    incomplete["samples"] = incomplete["samples"][:-1]  # type: ignore[index]
    incomplete["base_sample_values"] = incomplete["base_sample_values"][:-1]  # type: ignore[index]
    incomplete["minus_sample_values"] = incomplete["minus_sample_values"][:-1]  # type: ignore[index]
    incomplete["plus_sample_values"] = incomplete["plus_sample_values"][:-1]  # type: ignore[index]
    incomplete["base_objective_table"] = incomplete["base_objective_table"][:-1]  # type: ignore[index]
    incomplete["minus_objective_table"] = incomplete["minus_objective_table"][:-1]  # type: ignore[index]
    incomplete["plus_objective_table"] = incomplete["plus_objective_table"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="complete rectangular"):
        reduced_portfolio_artifact_guard_report(
            incomplete,
            gradient_artifacts=[_gradient_artifact()],
        )


def test_reduced_portfolio_guard_marks_bad_gradient_gate_without_crashing() -> None:
    bad_gradient = {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": False,
        "objective_gates": [
            {
                "objective": "gamma",
                "passed": True,
                "implicit": 1.0,
                "finite_difference": float("nan"),
                "abs_error": 0.0,
                "rel_error": 0.0,
            },
            "not-a-dict",
        ],
    }

    report = reduced_portfolio_artifact_guard_report(
        _row_artifact(),
        gradient_artifacts=[bad_gradient],  # type: ignore[list-item]
    )

    assert report["passed"] is False
    assert report["ad_fd_gradient_gate"]["passed"] is False
    assert report["ad_fd_gradient_gate"]["finite_ad_fd_values"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda artifact: artifact.update({"reduction": "median"}),
            "artifact reduction",
        ),
        (
            lambda artifact: artifact.update({"base_sample_values": [0.8, 1.0]}),
            "base_sample_values",
        ),
        (
            lambda artifact: artifact.update(
                {"base_objective_table": [0.8, 1.0, 0.9, 1.1]}
            ),
            "two-dimensional",
        ),
        (
            lambda artifact: artifact.update(
                {"samples": [*artifact["samples"][:-1], "bad-sample"]}
            ),
            "all samples",
        ),
    ],
)
def test_reduced_portfolio_guard_rejects_malformed_artifact_shapes(
    mutator,
    message: str,
) -> None:
    artifact = _row_artifact()
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        reduced_portfolio_artifact_guard_report(
            artifact,
            gradient_artifacts=[_gradient_artifact()],
        )


def test_reduced_portfolio_guard_reports_objective_and_provenance_blockers() -> None:
    artifact = _row_artifact()
    artifact["objective_names"] = ["omega", "kperp_eff2"]
    artifact["options"] = {"mboz": 8, "nboz": 8}
    artifact["input_path"] = ""
    artifact["wout_path"] = ""

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["objective_name_gate"]["passed"] is False
    assert report["objective_name_gate"]["has_growth_objective"] is False
    assert report["objective_name_gate"]["has_quasilinear_objective"] is False
    assert report["provenance_gate"]["passed"] is False
    assert report["provenance_gate"]["mboz"] == 8
    assert report["provenance_gate"]["has_input_and_wout_paths"] is False


def test_reduced_portfolio_guard_reports_unresolved_finite_difference_diagnostics() -> (
    None
):
    artifact = _row_artifact()
    artifact["response_resolved"] = False
    artifact["finite_difference_consistent"] = False
    artifact["plus_value"] = float("inf")

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["finite_difference_gate"]["passed"] is False
    assert report["finite_difference_gate"]["response_resolved"] is False
    assert report["finite_difference_gate"]["finite_difference_consistent"] is False
    assert (
        report["finite_difference_gate"]["finite_scalar_fields"]["plus_value"] is False
    )


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
    artifact["claim_scope"] = (
        "production nonlinear turbulent transport optimization claim"
    )

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["claim_scope_gate"]["passed"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_alphas": 0}, "min_alphas"),
        ({"min_ky": 0}, "min_ky"),
        ({"min_objectives": 0}, "min_objectives"),
        ({"min_boozer_mode": 0}, "min_boozer_mode"),
        ({"value_rtol": -1.0e-8}, "tolerances"),
        ({"value_atol": -1.0e-8}, "tolerances"),
    ],
)
def test_reduced_portfolio_guard_validates_config(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReducedPortfolioArtifactGuardConfig(**kwargs)


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
    written = mod.write_vmec_boozer_reduced_portfolio_guard_artifact(
        payload, out=out_path
    )

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
