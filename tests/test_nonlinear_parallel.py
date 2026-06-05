from __future__ import annotations

import json

import pytest

import spectraxgk
import spectraxgk.nonlinear_parallel as nonlinear_parallel
from spectraxgk.nonlinear_parallel import (
    NonlinearDomainDecompositionPlan,
    NonlinearDomainIdentityReport,
    NonlinearDomainTransportWindowReport,
    NonlinearParallelStrategy,
    NonlinearSpectralCommunicationReport,
    NonlinearSpectralRHSIdentityReport,
    classify_nonlinear_parallel_strategy,
    nonlinear_parallel_strategies,
    nonlinear_parallel_strategy,
    release_ready_nonlinear_parallel_strategies,
)


def test_nonlinear_parallel_public_api_exports_are_stable() -> None:
    public_names = (
        "NonlinearDomainDecompositionPlan",
        "NonlinearDomainIdentityReport",
        "NonlinearDomainTransportWindowReport",
        "NonlinearParallelStrategy",
        "NonlinearSpectralCommunicationReport",
        "NonlinearSpectralRHSIdentityReport",
        "build_nonlinear_domain_decomposition_plan",
        "classify_nonlinear_parallel_strategy",
        "deterministic_nonlinear_domain_state",
        "deterministic_nonlinear_spectral_state",
        "nonlinear_domain_identity_report",
        "nonlinear_domain_parallel_identity_gate",
        "nonlinear_domain_transport_window_identity_gate",
        "nonlinear_parallel_strategies",
        "nonlinear_parallel_strategy",
        "nonlinear_spectral_communication_identity_gate",
        "nonlinear_spectral_communication_identity_report",
        "nonlinear_spectral_rhs_identity_gate",
        "nonlinear_spectral_rhs_identity_report",
        "prototype_nonlinear_domain_decomposed_step",
        "prototype_nonlinear_domain_serial_step",
        "release_ready_nonlinear_parallel_strategies",
    )

    assert set(public_names) <= set(spectraxgk.__all__)
    assert set(public_names) <= set(nonlinear_parallel.__all__)
    assert NonlinearParallelStrategy is nonlinear_parallel.NonlinearParallelStrategy
    assert NonlinearDomainDecompositionPlan is nonlinear_parallel.NonlinearDomainDecompositionPlan
    assert NonlinearDomainIdentityReport is nonlinear_parallel.NonlinearDomainIdentityReport
    assert (
        NonlinearDomainTransportWindowReport
        is nonlinear_parallel.NonlinearDomainTransportWindowReport
    )
    assert (
        NonlinearSpectralCommunicationReport
        is nonlinear_parallel.NonlinearSpectralCommunicationReport
    )
    assert (
        NonlinearSpectralRHSIdentityReport
        is nonlinear_parallel.NonlinearSpectralRHSIdentityReport
    )
    for name in public_names:
        assert getattr(spectraxgk, name) is getattr(nonlinear_parallel, name)


def test_fft_axis_domain_sharding_is_diagnostic_until_runtime_fft_gates_exist() -> (
    None
):
    strategy = nonlinear_parallel_strategy("fft_axis_domain")

    assert strategy.diagnostic_only is True
    assert strategy.release_ready is False
    assert strategy.readiness == "diagnostic"
    assert "distributed_fft_forward_inverse_identity" in strategy.identity_gates
    assert "distributed_fft_nonlinear_bracket_identity" in strategy.identity_gates
    assert "distributed_fft_field_solve_identity" in strategy.identity_gates
    assert "split/reassemble spectral communication identity" in strategy.notes
    assert "runtime distributed FFT routing" in strategy.notes


def test_whole_state_kx_ky_is_diagnostic_only() -> None:
    strategy = nonlinear_parallel_strategy("whole_state_kx_ky")

    assert classify_nonlinear_parallel_strategy("whole_state_kx_ky") == "diagnostic"
    assert strategy.diagnostic_only is True
    assert strategy.release_ready is False
    assert strategy.independent_work is False
    assert strategy.changes_solver_layout is True
    assert "whole_state_kx_ky_final_state_identity" in strategy.identity_gates
    assert "nonlinear_domain_transport_window_identity" in strategy.physics_gates
    assert "matched_cpu_gpu_whole_state_scaling_profile" in strategy.profiler_gates
    assert "not a speedup claim" in strategy.notes


def test_independent_ky_and_uq_are_release_ready_independent_work_paths() -> None:
    release_ready_strategies = release_ready_nonlinear_parallel_strategies()
    release_ready = {strategy.name: strategy for strategy in release_ready_strategies}

    assert [strategy.name for strategy in release_ready_strategies] == [
        "independent_ky_scan",
        "uq_ensemble",
    ]
    assert set(release_ready) == {"independent_ky_scan", "uq_ensemble"}
    for name in ("independent_ky_scan", "uq_ensemble"):
        strategy = release_ready[name]
        assert strategy.release_ready is True
        assert strategy.independent_work is True
        assert strategy.changes_solver_layout is False
        assert strategy.identity_gates
        assert strategy.physics_gates
        assert strategy.profiler_gates


def test_velocity_species_hermite_exposes_identity_physics_and_profiler_gates() -> None:
    strategy = nonlinear_parallel_strategy("velocity_species_hermite")

    assert strategy.diagnostic_only is True
    assert "hermite_ghost_exchange_identity" in strategy.identity_gates
    assert "velocity_field_reduce_broadcast_identity" in strategy.identity_gates
    assert "nonlinear_fixed_step_identity_gate" in strategy.physics_gates
    assert "matched_velocity_species_hermite_scaling_profile" in strategy.profiler_gates
    assert set(strategy.required_gates) == set(
        strategy.identity_gates + strategy.physics_gates + strategy.profiler_gates
    )


def test_strategy_registry_is_complete_json_friendly_and_rejects_unknown_names() -> (
    None
):
    strategies = nonlinear_parallel_strategies()
    by_name = {strategy.name: strategy for strategy in strategies}

    assert set(by_name) == {
        "independent_ky_scan",
        "uq_ensemble",
        "whole_state_kx_ky",
        "velocity_species_hermite",
        "fft_axis_domain",
    }
    strategy_table = [strategy.to_dict() for strategy in strategies]
    decoded_table = json.loads(json.dumps(strategy_table, sort_keys=True))

    assert [row["name"] for row in decoded_table] == [
        strategy.name for strategy in strategies
    ]
    assert decoded_table[0]["name"] == "independent_ky_scan"
    assert decoded_table[0]["readiness"] == "release_ready"
    for row in decoded_table:
        assert set(row) == {
            "name",
            "readiness",
            "independent_work",
            "changes_solver_layout",
            "identity_gates",
            "physics_gates",
            "profiler_gates",
            "notes",
        }
        assert isinstance(row["identity_gates"], list)
        assert isinstance(row["physics_gates"], list)
        assert isinstance(row["profiler_gates"], list)
    with pytest.raises(ValueError, match="unknown nonlinear parallelization strategy"):
        nonlinear_parallel_strategy("banana")  # type: ignore[arg-type]


def test_strategy_table_keeps_release_ready_paths_independent_and_layout_preserving() -> (
    None
):
    strategies = nonlinear_parallel_strategies()

    assert len({strategy.name for strategy in strategies}) == len(strategies)
    for strategy in strategies:
        assert strategy.required_gates
        assert len(strategy.required_gates) == len(set(strategy.required_gates))
        if strategy.release_ready:
            assert strategy.independent_work is True
            assert strategy.changes_solver_layout is False
            assert "speedup claim" not in strategy.notes.lower()
        else:
            assert strategy.changes_solver_layout is True
            assert strategy.name not in {
                item.name for item in release_ready_nonlinear_parallel_strategies()
            }


def test_strategy_policy_separates_identity_physics_and_profiler_gates() -> None:
    for strategy in nonlinear_parallel_strategies():
        assert all("identity" in gate for gate in strategy.identity_gates)
        assert all(
            gate not in strategy.identity_gates for gate in strategy.physics_gates
        )
        assert all(
            "profile" in gate or "scaling" in gate for gate in strategy.profiler_gates
        )

    whole_state = nonlinear_parallel_strategy("whole_state_kx_ky")
    assert whole_state.diagnostic_only
    assert "final_state" in " ".join(whole_state.identity_gates)
    assert "nonlinear_window" in " ".join(whole_state.physics_gates)
    assert "whole_state_scaling_profile" in " ".join(whole_state.profiler_gates)
