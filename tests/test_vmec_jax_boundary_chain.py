from __future__ import annotations

import json
import math

import pytest

import spectraxgk
from spectraxgk.geometry.vmec_boundary_chain import (
    boundary_chain_summary_from_probe,
    build_boundary_chain_collection_summary,
    build_boundary_chain_summary,
)


def test_boundary_chain_summary_classifies_frozen_axis_branch_sensitivity() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=0.5936446584,
        final_cot_dot_exact_final_fd=0.5238431385,
        frozen_axis_replay_cost_gradient=0.10354209293,
        frozen_axis_vjp_cost_gradient=0.10354209294,
        raw_initial_replay_cost_gradient=0.46125484804,
        raw_initial_fd_norm=209.00303407,
        frozen_axis_initial_fd_norm=1.81142209,
    )

    assert summary["finite"] is True
    assert (
        summary["classification"]
        == "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive"
    )
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["metrics"]["raw_to_frozen_initial_norm_ratio"] == pytest.approx(
        115.379, rel=1.0e-3
    )
    assert "magnetic-axis initialization branch" in summary["next_action"]
    json.dumps(summary, allow_nan=False)


def test_boundary_chain_summary_verifies_frozen_axis_convention_despite_raw_exact_branch_sensitivity() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=1.0,
        final_cot_dot_exact_final_fd=0.98,
        frozen_axis_replay_cost_gradient=0.5,
        frozen_axis_vjp_cost_gradient=0.5 + 1.0e-13,
        frozen_axis_linear_replay_cost_gradient=0.5 + 2.0e-13,
        frozen_axis_linear_vjp_cost_gradient=0.5,
        frozen_axis_initial_fd_vs_linear_abs_norm=1.0e-13,
        frozen_axis_initial_fd_vs_linear_rel=1.0e-13,
        raw_initial_replay_cost_gradient=3.0,
        raw_initial_fd_norm=120.0,
        frozen_axis_initial_fd_norm=1.0,
        exact_relative_tolerance=0.1,
    )

    assert summary["classification"] == (
        "frozen_axis_convention_verified_but_exact_fd_branch_sensitive"
    )
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_fd_matches_linear_tangent"] is True
    assert summary["passes"]["frozen_axis_fd_jvp_matches_linear_jvp"] is True
    assert summary["passes"]["frozen_axis_linear_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_fd_vjp_matches_linear_vjp"] is True
    assert summary["passes"]["frozen_axis_convention_verified"] is True
    assert summary["metrics"]["raw_to_frozen_initial_norm_ratio"] == pytest.approx(120.0)
    assert "explicit tangent column" in summary["next_action"]


def test_boundary_chain_summary_anchors_latest_rc14_probe() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=0.07727649731974727,
        final_cot_dot_exact_final_fd=0.07362260082651849,
        frozen_axis_replay_cost_gradient=0.11282399953461382,
        frozen_axis_vjp_cost_gradient=0.11282399953439505,
        raw_initial_replay_cost_gradient=0.10166856394763844,
        raw_initial_fd_norm=70.29927985025522,
        frozen_axis_initial_fd_norm=1.8114220932737295,
        exact_relative_tolerance=0.1,
    )

    assert (
        summary["classification"]
        == "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive"
    )
    assert summary["passes"]["final_state_matches_exact_fd"] is True
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["errors"]["final_state_vs_exact_fd_rel"] == pytest.approx(
        0.04728341242111478
    )
    assert summary["errors"]["frozen_axis_vs_exact_fd_rel"] == pytest.approx(
        0.31507039602829146
    )


def test_boundary_chain_summary_anchors_latest_zs13_probe() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=-0.11873119671939811,
        final_cot_dot_exact_final_fd=-0.11900005540528957,
        frozen_axis_replay_cost_gradient=-0.14192363954706916,
        frozen_axis_vjp_cost_gradient=-0.14192363954695417,
        raw_initial_replay_cost_gradient=-0.1419716858450226,
        raw_initial_fd_norm=1.8888557263064212,
        frozen_axis_initial_fd_norm=1.8114220932737053,
        exact_relative_tolerance=0.1,
    )

    assert (
        summary["classification"]
        == "frozen_axis_replay_consistent_but_exact_fd_inconsistent"
    )
    assert summary["passes"]["final_state_matches_exact_fd"] is True
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["errors"]["final_state_vs_exact_fd_rel"] == pytest.approx(
        0.0022593156362472975
    )
    assert summary["errors"]["frozen_axis_vs_exact_fd_rel"] == pytest.approx(
        0.16341493849570599
    )
    assert summary["metrics"]["raw_to_frozen_initial_norm_ratio"] == pytest.approx(
        1.0427474266325047
    )


def test_boundary_chain_summary_anchors_coefficient_local_rc14_probe() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=-0.008248825352711719,
        final_cot_dot_exact_final_fd=-0.0073983155214745865,
        frozen_axis_replay_cost_gradient=0.11282399953752609,
        frozen_axis_vjp_cost_gradient=0.11282399953729642,
        raw_initial_replay_cost_gradient=0.11313073901185483,
        raw_initial_fd_norm=1.939974024843866,
        frozen_axis_initial_fd_norm=1.8114220932738312,
        exact_relative_tolerance=0.1,
    )

    assert summary["classification"] == "final_state_cotangent_mismatch"
    assert summary["passes"]["final_state_matches_exact_fd"] is False
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["errors"]["final_state_vs_exact_fd_rel"] == pytest.approx(
        0.10310678125310722
    )
    assert summary["errors"]["frozen_axis_vs_exact_fd_rel"] == pytest.approx(
        1.0731123288176654
    )


def test_boundary_chain_summary_accepts_converged_sparse_fd_window() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=0.02245397716,
        final_cot_dot_exact_final_fd=0.0221,
        frozen_axis_replay_cost_gradient=0.02079152741,
        frozen_axis_vjp_cost_gradient=0.02079152740,
        raw_initial_replay_cost_gradient=0.0208,
        raw_initial_fd_norm=1.9,
        frozen_axis_initial_fd_norm=1.8,
        exact_relative_tolerance=0.1,
    )

    assert summary["classification"] == "exact_fd_and_frozen_axis_replay_consistent"
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is True
    assert summary["errors"]["frozen_axis_vs_exact_fd_rel"] == pytest.approx(
        0.0740380977
    )


def test_boundary_chain_summary_reports_nonfinite_payload() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=math.nan,
        final_cot_dot_exact_final_fd=1.0,
        frozen_axis_replay_cost_gradient=1.0,
        frozen_axis_vjp_cost_gradient=1.0,
    )

    assert summary["finite"] is False
    assert summary["classification"] == "nonfinite_boundary_chain_probe"
    assert summary["errors"] == {}


def test_boundary_chain_summary_from_probe_and_public_api() -> None:
    payload = {
        "exact_fd_cost_gradient": 0.02245397716,
        "final_cot_dot_exact_final_fd": 0.0221,
        "final_cot_dot_tape_jvp_frozen_axis_fd": 0.02079152741,
        "initial_cot_dot_frozen_axis_fd": 0.02079152740,
        "final_cot_dot_tape_jvp_frozen_axis_linear": 0.02079152741001,
        "initial_cot_dot_frozen_axis_linear": 0.02079152740001,
        "frozen_axis_initial_fd_vs_linear_abs_norm": 1.0e-13,
        "frozen_axis_initial_fd_vs_linear_rel": 1.0e-13,
        "final_cot_dot_tape_jvp_raw_initial_fd": 0.0208,
        "raw_initial_fd_norm": 1.9,
        "frozen_axis_initial_fd_norm": 1.8,
    }

    assert spectraxgk.build_boundary_chain_summary is build_boundary_chain_summary
    assert (
        spectraxgk.boundary_chain_summary_from_probe
        is boundary_chain_summary_from_probe
    )
    summary = boundary_chain_summary_from_probe(payload, exact_relative_tolerance=0.1)
    assert summary["classification"] == "exact_fd_and_frozen_axis_replay_consistent"
    assert summary["passes"]["frozen_axis_convention_verified"] is True


def test_boundary_chain_collection_summary_counts_mixed_modes() -> None:
    branch_sensitive = {
        "index": 22,
        "name": "rc11",
        "summary": build_boundary_chain_summary(
            exact_fd_cost_gradient=-0.0135,
            final_cot_dot_exact_final_fd=-0.014,
            frozen_axis_replay_cost_gradient=-0.0172,
            frozen_axis_vjp_cost_gradient=-0.0172,
            raw_initial_replay_cost_gradient=-0.092,
            raw_initial_fd_norm=120.0,
            frozen_axis_initial_fd_norm=1.0,
            exact_relative_tolerance=0.1,
        ),
    }
    consistent = {
        "index": 28,
        "name": "rc14",
        "summary": build_boundary_chain_summary(
            exact_fd_cost_gradient=0.02245397716,
            final_cot_dot_exact_final_fd=0.0221,
            frozen_axis_replay_cost_gradient=0.02079152741,
            frozen_axis_vjp_cost_gradient=0.02079152740,
            raw_initial_replay_cost_gradient=0.0208,
            raw_initial_fd_norm=1.9,
            frozen_axis_initial_fd_norm=1.8,
            exact_relative_tolerance=0.1,
        ),
        "growth_branch_locality": {
            "enabled": True,
            "passed": True,
            "classification": "all_samples_dominant_growth_branch_locally_consistent",
        },
    }

    summary = build_boundary_chain_collection_summary(
        [branch_sensitive, consistent],
        exact_relative_tolerance=0.1,
    )

    assert (
        summary["classification"]
        == "mixed_exact_fd_consistency_with_branch_sensitive_modes"
    )
    assert summary["counts"] == {
        "n_total": 2,
        "n_finite": 2,
        "n_frozen_axis_internal_pass": 2,
        "n_frozen_axis_convention_verified": 0,
        "n_exact_fd_consistent": 1,
        "n_branch_sensitive": 1,
        "n_growth_branch_locality_checked": 1,
        "n_growth_branch_locality_passed": 1,
    }
    assert summary["rows"][0]["name"] == "rc11"
    assert summary["rows"][1]["frozen_axis_matches_exact_fd"] is True
    assert summary["rows"][1]["exact_fd_consistent"] is True
    assert summary["rows"][1]["growth_branch_locality_passed"] is True
    assert (
        summary["rows"][1]["growth_branch_locality_classification"]
        == "all_samples_dominant_growth_branch_locally_consistent"
    )
    assert "exclude or regularize branch-sensitive modes" in summary["next_action"]


def test_boundary_chain_collection_counts_verified_frozen_axis_convention_separately() -> None:
    verified = {
        "index": 3,
        "name": "rc11",
        "summary": build_boundary_chain_summary(
            exact_fd_cost_gradient=1.0,
            final_cot_dot_exact_final_fd=0.99,
            frozen_axis_replay_cost_gradient=0.5,
            frozen_axis_vjp_cost_gradient=0.5,
            frozen_axis_linear_replay_cost_gradient=0.5,
            frozen_axis_linear_vjp_cost_gradient=0.5,
            frozen_axis_initial_fd_vs_linear_abs_norm=0.0,
            frozen_axis_initial_fd_vs_linear_rel=0.0,
            raw_initial_replay_cost_gradient=4.0,
            raw_initial_fd_norm=100.0,
            frozen_axis_initial_fd_norm=1.0,
        ),
    }

    summary = build_boundary_chain_collection_summary([verified])

    assert summary["classification"] == "all_components_frozen_axis_convention_verified"
    assert summary["counts"]["n_exact_fd_consistent"] == 0
    assert summary["counts"]["n_frozen_axis_convention_verified"] == 1
    assert summary["rows"][0]["frozen_axis_matches_exact_fd"] is False
    assert summary["rows"][0]["exact_fd_consistent"] is False
    assert summary["rows"][0]["frozen_axis_convention_verified"] is True
    assert "growth-branch" in summary["next_action"]


def test_boundary_chain_collection_summary_fails_closed_when_empty() -> None:
    summary = build_boundary_chain_collection_summary([])

    assert summary["finite"] is False
    assert summary["classification"] == "empty_boundary_chain_collection"
    assert summary["counts"]["n_total"] == 0
    assert summary["counts"]["n_frozen_axis_convention_verified"] == 0
    assert summary["counts"]["n_growth_branch_locality_checked"] == 0
