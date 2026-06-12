from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectraxgk.nonlinear_parallel import (
    NonlinearSpectralCommunicationReport,
    NonlinearSpectralIntegratorIdentityReport,
    NonlinearSpectralRHSIdentityReport,
    deterministic_nonlinear_spectral_state,
    logical_decomposed_nonlinear_spectral_rhs,
    nonlinear_spectral_communication_identity_gate,
    nonlinear_spectral_communication_identity_report,
    nonlinear_spectral_integrator_identity_gate,
    nonlinear_spectral_rhs_identity_gate,
    nonlinear_spectral_rhs_identity_report,
)


def test_nonlinear_spectral_communication_gate_closes_fft_bracket_and_field_layouts() -> None:
    state = deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_spectral_communication_identity_gate(
        state,
        y_chunks=(2, 2, 2),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert isinstance(report, NonlinearSpectralCommunicationReport)
    assert report.state_shape == (2, 3, 6, 4, 2)
    assert report.y_chunks == (2, 2, 2)
    assert report.x_chunks == (2, 2)
    assert report.y_offsets == (0, 2, 4)
    assert report.x_offsets == (0, 2)
    assert report.blocked_reasons == ()
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.fft_max_abs_error <= report.atol
    assert report.fft_max_rel_error <= report.rtol
    assert report.bracket_max_abs_error <= report.atol
    assert report.bracket_max_rel_error <= report.rtol
    assert report.field_max_abs_error <= report.atol
    assert report.field_max_rel_error <= report.rtol
    assert "no production routing or speedup claim" in report.claim_scope
    assert report.to_dict()["identity_passed"] is True


def test_nonlinear_spectral_communication_report_fails_closed_on_mismatch() -> None:
    reference = jnp.ones((2, 3, 6, 4, 2), dtype=jnp.complex64)
    perturbed = reference.at[0, 0, 0, 0, 0].add(1.0e-3)
    field = jnp.ones((6, 4, 2), dtype=jnp.complex64)

    report = nonlinear_spectral_communication_identity_report(
        reference,
        perturbed,
        reference,
        reference,
        field,
        field,
        state_shape=(2, 3, 6, 4, 2),
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.fft_max_abs_error > report.atol
    assert report.blocked_reasons == ()


def test_nonlinear_spectral_communication_report_fails_closed_on_shape_blocker() -> None:
    reference = jnp.ones((2, 3, 6, 4, 2), dtype=jnp.complex64)
    communicated = jnp.ones((2, 3, 5, 4, 2), dtype=jnp.complex64)
    field = jnp.ones((6, 4, 2), dtype=jnp.complex64)

    report = nonlinear_spectral_communication_identity_report(
        reference,
        communicated,
        reference,
        reference,
        field,
        field,
        state_shape=(2, 3, 6, 4, 2),
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.fft_max_abs_error == float("inf")
    assert report.fft_max_rel_error == float("inf")
    assert report.blocked_reasons == (
        "communicated_fft_roundtrip_shape_mismatch",
    )


def test_nonlinear_spectral_rhs_identity_gate_reconstructs_ordered_logical_tiles() -> None:
    state = deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_spectral_rhs_identity_gate(
        state,
        y_chunks=(2, 1, 3),
        x_chunks=(1, 3),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert isinstance(report, NonlinearSpectralRHSIdentityReport)
    assert report.state_shape == (2, 3, 6, 4, 2)
    assert report.y_chunks == (2, 1, 3)
    assert report.x_chunks == (1, 3)
    assert report.y_offsets == (0, 2, 3)
    assert report.x_offsets == (0, 1)
    assert report.tile_bounds == (
        (0, 2, 0, 1),
        (0, 2, 1, 4),
        (2, 3, 0, 1),
        (2, 3, 1, 4),
        (3, 6, 0, 1),
        (3, 6, 1, 4),
    )
    assert report.blocked_reasons == ()
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.reconstruction_max_abs_error <= report.atol
    assert report.reconstruction_max_rel_error <= report.rtol
    assert report.field_max_abs_error <= report.atol
    assert report.field_max_rel_error <= report.rtol
    assert report.bracket_max_abs_error <= report.atol
    assert report.bracket_max_rel_error <= report.rtol
    assert report.rhs_max_abs_error <= report.atol
    assert report.rhs_max_rel_error <= report.rtol
    assert "existing bracket contribution" in report.claim_scope
    assert "no production routing or speedup claim" in report.claim_scope
    assert report.to_dict()["tile_bounds"] == report.tile_bounds


def test_logical_decomposed_nonlinear_spectral_rhs_returns_gated_rhs() -> None:
    state = deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    rhs, report = logical_decomposed_nonlinear_spectral_rhs(
        state,
        y_chunks=(2, 1, 3),
        x_chunks=(1, 3),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert isinstance(report, NonlinearSpectralRHSIdentityReport)
    assert rhs.shape == state.shape
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.rhs_max_abs_error <= report.atol
    assert report.rhs_max_rel_error <= report.rtol


def test_nonlinear_spectral_integrator_identity_gate_closes_multistep_route() -> None:
    state = deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_spectral_integrator_identity_gate(
        state,
        y_chunks=(2, 1, 3),
        x_chunks=(1, 3),
        steps=3,
        dt=0.0025,
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert isinstance(report, NonlinearSpectralIntegratorIdentityReport)
    assert report.state_shape == (2, 3, 6, 4, 2)
    assert report.y_offsets == (0, 2, 3)
    assert report.x_offsets == (0, 1)
    assert report.steps == 3
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.final_state_max_abs_error <= report.atol
    assert report.final_state_max_rel_error <= report.rtol
    assert report.free_energy_trace_max_abs_error <= report.atol
    assert report.field_energy_trace_max_abs_error <= report.atol
    assert report.flux_proxy_trace_max_abs_error <= report.atol
    assert len(report.serial_free_energy_trace) == report.steps + 1
    assert "no production distributed FFT routing or speedup claim" in report.claim_scope
    assert report.to_dict()["identity_passed"] is True


def test_nonlinear_spectral_rhs_report_fails_closed_on_rhs_mismatch() -> None:
    reference = jnp.ones((2, 3, 6, 4, 2), dtype=jnp.complex64)
    field = jnp.ones((6, 4, 2), dtype=jnp.complex64)
    bracket = 0.25j * reference
    rhs = -bracket
    perturbed_rhs = rhs.at[0, 0, 0, 0, 0].add(1.0e-3)

    report = nonlinear_spectral_rhs_identity_report(
        reference,
        reference,
        field,
        field,
        bracket,
        bracket,
        rhs,
        perturbed_rhs,
        state_shape=(2, 3, 6, 4, 2),
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        tile_bounds=(
            (0, 3, 0, 2),
            (0, 3, 2, 4),
            (3, 6, 0, 2),
            (3, 6, 2, 4),
        ),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.blocked_reasons == ()
    assert report.rhs_max_abs_error > report.atol


def test_nonlinear_spectral_state_and_chunk_validation_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="spectral state shape"):
        deterministic_nonlinear_spectral_state((2, 3, 4, 5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Ny and Nx >= 2"):
        deterministic_nonlinear_spectral_state((1, 1, 1, 4, 1))

    state = deterministic_nonlinear_spectral_state((2, 2, 6, 4, 1))
    with pytest.raises(ValueError, match="y_chunks must sum"):
        nonlinear_spectral_communication_identity_gate(state, y_chunks=(2, 2), x_chunks=(2, 2))
    with pytest.raises(ValueError, match="x_chunks entries must be positive"):
        nonlinear_spectral_communication_identity_gate(state, y_chunks=(3, 3), x_chunks=(4, 0))
    with pytest.raises(ValueError, match="y_chunks must sum"):
        nonlinear_spectral_rhs_identity_gate(state, y_chunks=(2, 2), x_chunks=(2, 2))
    with pytest.raises(ValueError, match="steps must be at least one"):
        nonlinear_spectral_integrator_identity_gate(state, y_chunks=(3, 3), x_chunks=(2, 2), steps=0)
