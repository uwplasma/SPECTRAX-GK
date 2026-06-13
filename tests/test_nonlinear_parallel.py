from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
import spectraxgk.nonlinear_parallel as nonlinear_parallel
from spectraxgk.nonlinear_parallel import (
    NonlinearDomainDecompositionPlan,
    NonlinearDomainIdentityReport,
    NonlinearDomainTransportWindowReport,
    NonlinearParallelStrategy,
    NonlinearSpectralCommunicationReport,
    NonlinearSpectralDevicePencilRHSIdentityReport,
    NonlinearSpectralDevicePencilTransportWindowReport,
    NonlinearSpectralDomainWorkModel,
    NonlinearSpectralPencilRHSIdentityReport,
    NonlinearSpectralPencilTransportWindowReport,
    NonlinearSpectralPencilWorkModel,
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
        "NonlinearSpectralDevicePencilRHSIdentityReport",
        "NonlinearSpectralDevicePencilTransportWindowReport",
        "NonlinearSpectralDomainWorkModel",
        "NonlinearSpectralPencilRHSIdentityReport",
        "NonlinearSpectralPencilTransportWindowReport",
        "NonlinearSpectralPencilWorkModel",
        "NonlinearSpectralRHSIdentityReport",
        "build_nonlinear_domain_decomposition_plan",
        "classify_nonlinear_parallel_strategy",
        "deterministic_nonlinear_domain_state",
        "deterministic_nonlinear_spectral_state",
        "device_z_pencil_nonlinear_spectral_rhs",
        "device_z_pencil_nonlinear_spectral_transport_window_identity_gate",
        "integrate_logical_decomposed_nonlinear_spectral",
        "nonlinear_domain_identity_report",
        "nonlinear_domain_parallel_identity_gate",
        "nonlinear_domain_transport_window_identity_gate",
        "nonlinear_parallel_strategies",
        "nonlinear_parallel_strategy",
        "nonlinear_spectral_communication_identity_gate",
        "nonlinear_spectral_communication_identity_report",
        "nonlinear_spectral_domain_work_model",
        "nonlinear_spectral_pencil_rhs_identity_gate",
        "nonlinear_spectral_pencil_transport_window_identity_gate",
        "nonlinear_spectral_pencil_work_model",
        "nonlinear_spectral_rhs_identity_gate",
        "nonlinear_spectral_rhs_identity_report",
        "pencil_decomposed_nonlinear_spectral_rhs",
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
        NonlinearSpectralDevicePencilRHSIdentityReport
        is nonlinear_parallel.NonlinearSpectralDevicePencilRHSIdentityReport
    )
    assert (
        NonlinearSpectralDevicePencilTransportWindowReport
        is nonlinear_parallel.NonlinearSpectralDevicePencilTransportWindowReport
    )
    assert (
        NonlinearSpectralDomainWorkModel
        is nonlinear_parallel.NonlinearSpectralDomainWorkModel
    )
    assert (
        NonlinearSpectralPencilRHSIdentityReport
        is nonlinear_parallel.NonlinearSpectralPencilRHSIdentityReport
    )
    assert (
        NonlinearSpectralPencilTransportWindowReport
        is nonlinear_parallel.NonlinearSpectralPencilTransportWindowReport
    )
    assert (
        NonlinearSpectralPencilWorkModel
        is nonlinear_parallel.NonlinearSpectralPencilWorkModel
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
    assert "pencil_fft_fused_nonlinear_rhs_identity" in strategy.identity_gates
    assert "pencil_fft_physical_transport_window_identity" in strategy.identity_gates
    assert "device_z_pencil_fused_nonlinear_rhs_identity" in strategy.identity_gates
    assert "device_z_pencil_physical_transport_window_identity" in strategy.identity_gates
    assert "pencil-FFT fused-bracket identity" in strategy.notes
    assert "device-level transport-window routing" in strategy.notes


def test_logical_decomposed_spectral_integrator_routes_after_identity_gate() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 5))

    routed, report = nonlinear_parallel.integrate_logical_decomposed_nonlinear_spectral(
        state,
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        dt=0.0025,
        steps=3,
        atol=5.0e-6,
        rtol=5.0e-6,
    )
    serial = state
    dt = jnp.asarray(0.0025, dtype=jnp.real(state).dtype)
    for _ in range(3):
        _field, _bracket, rhs = nonlinear_parallel._serial_nonlinear_spectral_rhs(serial)
        serial = serial + dt * rhs

    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert routed.shape == state.shape
    assert jnp.allclose(routed, serial, atol=5.0e-6, rtol=5.0e-6)
    assert "no production distributed FFT" in report.claim_scope


def test_nonlinear_spectral_domain_work_model_blocks_global_reconstruction_speedup() -> None:
    model = nonlinear_parallel.nonlinear_spectral_domain_work_model(
        (2, 4, 32, 32, 4),
        y_chunks=(16, 16),
        x_chunks=(16, 16),
    )

    assert isinstance(model, NonlinearSpectralDomainWorkModel)
    assert model.num_tiles == 4
    assert model.state_elements == 32768
    assert model.field_elements == 4096
    assert model.owned_state_elements_per_step == model.state_elements
    assert model.state_allgather_elements_per_step == 3 * model.state_elements
    assert model.bracket_allgather_elements_per_step == 3 * model.state_elements
    assert model.field_broadcast_elements_per_step == 3 * model.field_elements
    assert model.total_communication_elements_per_step == 208896
    assert model.communication_to_owned_work_ratio == pytest.approx(6.375)
    assert model.parallel_efficiency_ceiling == pytest.approx(1.0 / 7.375)
    assert model.production_speedup_feasible is False
    assert model.feasibility_blockers == (
        "global_reconstruction_communication_dominates_owned_work",
    )
    assert "not a distributed FFT performance claim" in model.claim_scope
    assert model.to_dict()["production_speedup_feasible"] is False


def test_nonlinear_spectral_pencil_work_model_gates_plausible_scaling() -> None:
    model = nonlinear_parallel.nonlinear_spectral_pencil_work_model(
        (2, 4, 32, 32, 4),
        y_chunks=(16, 16),
        x_chunks=(16, 16),
    )

    assert isinstance(model, NonlinearSpectralPencilWorkModel)
    assert model.num_tiles == 4
    assert model.global_reconstruction_elements_per_step == 0
    assert (
        model.transform_payload_elements_per_step
        == 3 * model.state_elements + 2 * model.field_elements
    )
    assert model.pencil_transpose_elements_per_step > 0
    assert model.communication_to_fft_work_ratio < model.max_communication_to_fft_work_ratio
    assert model.predicted_speedup_ceiling >= model.min_predicted_speedup
    assert model.production_speedup_feasible is True
    assert model.feasibility_blockers == ()
    assert "no global reconstruction" in model.claim_scope

    blocked = nonlinear_parallel.nonlinear_spectral_pencil_work_model(
        (2, 4, 8, 8, 2),
        y_chunks=(8,),
        x_chunks=(8,),
    )
    assert blocked.production_speedup_feasible is False
    assert "single_tile_no_domain_decomposition" in blocked.feasibility_blockers


def test_pencil_fft_route_matches_serial_fft_and_rhs_without_reconstruction() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    roundtrip_serial = jnp.fft.fft2(jnp.fft.ifft2(state, axes=(-3, -2)), axes=(-3, -2))
    roundtrip_pencil = nonlinear_parallel._pencil_fft2(
        nonlinear_parallel._pencil_ifft2(state, y_axis=-3, x_axis=-2),
        y_axis=-3,
        x_axis=-2,
    )
    assert jnp.allclose(roundtrip_pencil, roundtrip_serial, atol=5.0e-6, rtol=5.0e-6)

    report = nonlinear_parallel.nonlinear_spectral_pencil_rhs_identity_gate(
        state,
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=1.0e-5,
    )
    assert isinstance(report, NonlinearSpectralPencilRHSIdentityReport)
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.work_model.global_reconstruction_elements_per_step == 0
    assert report.rhs_max_abs_error <= report.atol
    assert report.rhs_max_rel_error <= report.rtol

    routed_rhs, routed_report = nonlinear_parallel.pencil_decomposed_nonlinear_spectral_rhs(
        state,
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=1.0e-5,
    )
    _field, _bracket, serial_rhs = nonlinear_parallel._serial_nonlinear_spectral_rhs(state)
    assert routed_report.identity_passed is True
    assert routed_report.decomposed_path_enabled is True
    assert jnp.allclose(routed_rhs, serial_rhs, atol=5.0e-6, rtol=5.0e-6)


def test_z_chunked_pencil_bracket_matches_unchunked_route() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 5))
    field = nonlinear_parallel._field_from_state(state)

    unchunked = nonlinear_parallel._pencil_spectral_bracket(state, field)
    chunked = nonlinear_parallel._pencil_spectral_bracket_z_chunked(
        state,
        field,
        z_chunk_size=2,
    )
    chunked_rhs = nonlinear_parallel._pencil_nonlinear_spectral_rhs_z_chunked(
        state,
        z_chunk_size=2,
    )[2]
    unchunked_rhs = nonlinear_parallel._pencil_nonlinear_spectral_rhs(state)[2]

    assert jnp.allclose(chunked, unchunked, atol=5.0e-6, rtol=1.0e-5)
    assert jnp.allclose(chunked_rhs, unchunked_rhs, atol=5.0e-6, rtol=1.0e-5)
    assert jnp.allclose(
        nonlinear_parallel._pencil_spectral_bracket_z_chunked(
            state,
            field,
            z_chunk_size=99,
        ),
        unchunked,
        atol=5.0e-6,
        rtol=1.0e-5,
    )
    with pytest.raises(ValueError, match="z_chunk_size must be at least one"):
        nonlinear_parallel._pencil_spectral_bracket_z_chunked(
            state,
            field,
            z_chunk_size=0,
        )


def test_device_z_pencil_route_fails_closed_without_two_devices() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    routed_rhs, report = nonlinear_parallel.device_z_pencil_nonlinear_spectral_rhs(
        state,
        devices=(),
        atol=5.0e-6,
        rtol=1.0e-5,
    )
    _field, _bracket, serial_rhs = nonlinear_parallel._serial_nonlinear_spectral_rhs(state)

    assert isinstance(report, NonlinearSpectralDevicePencilRHSIdentityReport)
    assert report.identity_passed is False
    assert report.device_sharding_active is False
    assert report.decomposed_path_enabled is False
    assert report.blocked_reasons == ("requires_at_least_two_devices",)
    assert jnp.allclose(routed_rhs, serial_rhs, atol=0.0, rtol=0.0)

    transport_report = (
        nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
            state,
            devices=(),
            steps=2,
            atol=5.0e-6,
            rtol=1.0e-5,
        )
    )
    assert isinstance(transport_report, NonlinearSpectralDevicePencilTransportWindowReport)
    assert transport_report.identity_passed is False
    assert transport_report.device_sharding_active is False
    assert transport_report.decomposed_path_enabled is False
    assert transport_report.blocked_reasons == ("requires_at_least_two_devices",)
    assert "cannot support the z shard" in transport_report.claim_scope


def test_host_staged_array_for_sharding_materializes_numpy_copy() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    staged = nonlinear_parallel._host_staged_array_for_sharding(state)

    assert isinstance(staged, np.ndarray)
    assert staged.shape == state.shape
    assert np.allclose(staged, np.asarray(jax.device_get(state)))


def test_abs_or_rel_tolerance_policy_matches_allclose_style_gate() -> None:
    assert nonlinear_parallel._within_abs_or_rel_tolerance(
        4.0e-6,
        2.0e-2,
        atol=5.0e-6,
        rtol=1.0e-4,
    )
    assert nonlinear_parallel._within_abs_or_rel_tolerance(
        1.0e-3,
        4.0e-7,
        atol=5.0e-6,
        rtol=1.0e-4,
    )
    assert not nonlinear_parallel._within_abs_or_rel_tolerance(
        1.0e-3,
        1.0e-2,
        atol=5.0e-6,
        rtol=1.0e-4,
    )


def test_device_z_transport_window_rejects_invalid_step_count() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    with pytest.raises(ValueError, match="steps must be at least one"):
        nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
            state,
            steps=0,
        )


def test_device_z_transport_window_blocks_nondivisible_z_extent() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 3))

    report = nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
        state,
        devices=(jax.devices()[0], jax.devices()[0]),
        steps=2,
    )

    assert report.identity_passed is False
    assert report.device_sharding_active is False
    assert report.active_device_count == 0
    assert report.requested_device_count == 2
    assert report.blocked_reasons == ("z_extent_not_divisible_by_device_count",)
    assert report.final_state_max_abs_error == float("inf")


def test_device_z_pencil_transport_window_identity_on_available_devices() -> None:
    if jax.local_device_count() < 2:
        pytest.skip("requires at least two local JAX devices")
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
        state,
        devices=jax.devices()[:2],
        steps=2,
        dt=0.0025,
        atol=5.0e-6,
        rtol=1.0e-4,
    )

    assert report.identity_passed is True
    assert report.device_sharding_active is True
    assert report.decomposed_path_enabled is True
    assert report.final_state_max_abs_error <= report.atol
    assert report.final_state_max_rel_error <= report.rtol
    assert report.physical_flux_trace_max_abs_error <= report.atol
    assert report.bracket_rms_trace_max_rel_error <= report.rtol
    assert len(report.serial_free_energy_trace) == report.steps + 1
    assert len(report.device_free_energy_trace) == report.steps + 1
    assert "transport-window identity gate" in report.claim_scope


def test_pencil_fft_physical_transport_window_identity_gate() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_parallel.nonlinear_spectral_pencil_transport_window_identity_gate(
        state,
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        dt=0.001,
        steps=3,
        atol=5.0e-6,
        rtol=1.0e-5,
    )

    assert isinstance(report, NonlinearSpectralPencilTransportWindowReport)
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.final_state_max_abs_error <= report.atol
    assert report.physical_flux_trace_max_abs_error <= report.atol
    assert len(report.serial_physical_flux_trace) == 4
    assert len(report.pencil_physical_flux_trace) == 4
    assert max(report.serial_physical_flux_trace) > 0.0
    assert "not an absolute nonlinear turbulent heat-flux claim" in report.claim_scope


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
