from __future__ import annotations

import pytest

import spectraxgk
from spectraxgk.nonlinear_parallel import (
    NonlinearParallelStrategy,
    classify_nonlinear_parallel_strategy,
    nonlinear_parallel_strategies,
    nonlinear_parallel_strategy,
    release_ready_nonlinear_parallel_strategies,
)


def test_fft_axis_domain_sharding_is_blocked_until_distributed_fft_identity_gates_exist() -> (
    None
):
    assert spectraxgk.NonlinearParallelStrategy is NonlinearParallelStrategy
    assert spectraxgk.nonlinear_parallel_strategy is nonlinear_parallel_strategy

    strategy = nonlinear_parallel_strategy("fft_axis_domain")

    assert strategy.blocked is True
    assert strategy.release_ready is False
    assert strategy.readiness == "blocked"
    assert "distributed_fft_forward_inverse_identity" in strategy.identity_gates
    assert "distributed_fft_nonlinear_bracket_identity" in strategy.identity_gates
    assert "distributed_fft_field_solve_identity" in strategy.identity_gates
    assert "distributed FFT identity gates" in strategy.notes


def test_whole_state_kx_ky_is_diagnostic_only() -> None:
    strategy = nonlinear_parallel_strategy("whole_state_kx_ky")

    assert classify_nonlinear_parallel_strategy("whole_state_kx_ky") == "diagnostic"
    assert strategy.diagnostic_only is True
    assert strategy.release_ready is False
    assert strategy.independent_work is False
    assert strategy.changes_solver_layout is True
    assert "whole_state_kx_ky_final_state_identity" in strategy.identity_gates
    assert "matched_cpu_gpu_whole_state_scaling_profile" in strategy.profiler_gates


def test_independent_ky_and_uq_are_release_ready_independent_work_paths() -> None:
    release_ready = {
        strategy.name: strategy
        for strategy in release_ready_nonlinear_parallel_strategies()
    }

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
    payload = by_name["independent_ky_scan"].to_dict()
    assert payload["name"] == "independent_ky_scan"
    assert payload["readiness"] == "release_ready"
    with pytest.raises(ValueError, match="unknown nonlinear parallelization strategy"):
        nonlinear_parallel_strategy("banana")  # type: ignore[arg-type]
