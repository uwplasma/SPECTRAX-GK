from __future__ import annotations

import numpy as np
import pytest

import spectraxgk
from spectraxgk import benchmarking
from spectraxgk.validation_gates import (
    BranchContinuationMetrics,
    EigenfunctionComparisonMetrics,
    GateReport,
    LateTimeLinearMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
    ScalarGateResult,
    ZonalFlowResponseMetrics,
    branch_continuity_gate_report,
    eigenfunction_gate_report,
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
    linear_metrics_gate_report,
    nonlinear_window_gate_report,
    observed_order_gate_report,
    zonal_response_gate_report,
)


def test_validation_gate_primitives_are_public_and_backward_compatible() -> None:
    assert spectraxgk.evaluate_scalar_gate is evaluate_scalar_gate
    assert benchmarking.evaluate_scalar_gate is evaluate_scalar_gate
    assert benchmarking.observed_order_gate_report is observed_order_gate_report
    assert benchmarking.branch_continuity_gate_report is branch_continuity_gate_report


def test_scalar_gate_and_json_report_are_strict_and_serializable() -> None:
    passed = evaluate_scalar_gate("gamma", 1.01, 1.0, atol=0.0, rtol=0.02)
    failed = evaluate_scalar_gate("omega", 0.7, 1.0, atol=0.0, rtol=0.02)
    report = gate_report("case", "reference", [passed, failed])
    payload = gate_report_to_dict(report)

    assert isinstance(passed, ScalarGateResult)
    assert isinstance(report, GateReport)
    assert report.passed is False
    assert payload["gates"][0]["metric"] == "gamma"
    assert payload["gates"][1]["passed"] is False

    with pytest.raises(ValueError):
        gate_report("empty", "reference", [])
    with pytest.raises(ValueError):
        evaluate_scalar_gate("bad", 1.0, 1.0, atol=-1.0, rtol=0.0)


def test_validation_gate_family_helpers_cover_physics_observables() -> None:
    linear = LateTimeLinearMetrics(
        gamma_fit=1.0,
        omega_fit=2.0,
        gamma_tail_mean=1.0,
        omega_tail_mean=2.0,
        gamma_tail_std=0.01,
        omega_tail_std=0.02,
        tmin=1.0,
        tmax=2.0,
        nsamples=10,
        signal_source="mode",
    )
    nonlinear = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=10,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.05,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=4.0,
        phi_mode_envelope_std=0.4,
        phi_mode_envelope_max=4.5,
    )
    zonal = ZonalFlowResponseMetrics(
        initial_level=1.0,
        initial_policy="first_abs",
        residual_level=0.2,
        residual_std=0.01,
        response_rms=0.3,
        gam_frequency=2.0,
        gam_damping_rate=0.1,
        damping_method="branchwise_extrema",
        frequency_method="hilbert_phase",
        peak_count=4,
        peak_fit_count=4,
        tmin=0.0,
        tmax=10.0,
        fit_tmin=0.0,
        fit_tmax=5.0,
        peak_times=np.array([1.0, 2.0]),
        peak_envelope=np.array([0.5, 0.4]),
        max_peak_times=np.array([1.0]),
        max_peak_values=np.array([0.5]),
        min_peak_times=np.array([2.0]),
        min_peak_values=np.array([-0.4]),
    )

    assert linear_metrics_gate_report(linear, linear, case="linear", source="self").passed is True
    assert nonlinear_window_gate_report(nonlinear, nonlinear, case="nonlinear", source="self").passed is True
    assert (
        zonal_response_gate_report(
            zonal,
            zonal,
            case="zonal",
            source="self",
            residual_atol=0.0,
            frequency_atol=0.0,
            damping_atol=0.0,
        ).passed
        is True
    )
    assert eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.99, relative_l2=0.01, phase_shift=0.0),
        case="mode",
        source="self",
    ).passed is True


def test_order_and_branch_gates_preserve_open_lane_failures() -> None:
    observed = ObservedOrderMetrics(
        step_sizes=np.array([0.4, 0.2, 0.1]),
        errors=np.array([0.01, 0.02, 0.002]),
        orders=np.array([-1.0, 3.32192809]),
        asymptotic_order=3.32192809,
    )
    order_report = observed_order_gate_report(
        observed,
        case="nonmonotone",
        source="synthetic",
        min_asymptotic_order=1.0,
        min_pairwise_order=0.0,
    )
    assert order_report.passed is False
    assert order_report.gates[1].metric == "min_pairwise_order_deficit"

    branch = BranchContinuationMetrics(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.1, 0.3]),
        omega=np.array([1.0, 1.1]),
        rel_gamma_jumps=np.array([0.666]),
        rel_omega_jumps=np.array([0.091]),
        max_rel_gamma_jump=0.666,
        max_rel_omega_jump=0.091,
        min_successive_overlap=None,
    )
    branch_report = branch_continuity_gate_report(
        branch,
        case="branch",
        source="synthetic",
        max_rel_gamma_jump=0.5,
        max_rel_omega_jump=0.5,
        min_successive_overlap=0.95,
    )
    assert branch_report.passed is False
    assert branch_report.gates[-1].metric == "successive_overlap_deficit"
