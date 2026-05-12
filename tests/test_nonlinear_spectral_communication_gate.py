from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectraxgk.nonlinear_parallel import (
    NonlinearSpectralCommunicationReport,
    deterministic_nonlinear_spectral_state,
    nonlinear_spectral_communication_identity_gate,
    nonlinear_spectral_communication_identity_report,
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
