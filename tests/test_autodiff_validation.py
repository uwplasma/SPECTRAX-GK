from __future__ import annotations

import numpy as np
import pytest

import spectraxgk
from spectraxgk.autodiff_validation import covariance_diagnostics


def test_covariance_diagnostics_reports_uq_and_sensitivity_metadata() -> None:
    assert spectraxgk.covariance_diagnostics is covariance_diagnostics
    jac = np.array([[1.0, 0.2], [0.1, 0.8], [0.4, -0.3]])
    residual = np.array([1.0e-3, -2.0e-3, 1.5e-3])

    out = covariance_diagnostics(jac, residual, regularization=1.0e-8)
    cov = np.asarray(out["covariance"])
    corr = np.asarray(out["covariance_correlation"])
    eig = np.asarray(out["covariance_eigenvalues"])

    assert out["sensitivity_map_rank"] == 2
    assert np.isfinite(float(out["jacobian_condition_number"]))
    assert np.allclose(cov, cov.T)
    assert np.all(eig > 0.0)
    assert np.all(np.asarray(out["covariance_std"]) > 0.0)
    assert np.allclose(np.diag(corr), 1.0)
    assert float(out["uq_ellipse_area_1sigma"]) > 0.0


def test_covariance_diagnostics_matches_closed_form_gauss_newton() -> None:
    jac = np.eye(2)
    residual = np.array([2.0e-3, -4.0e-3])

    out = covariance_diagnostics(jac, residual, regularization=0.0)
    sigma2 = float(np.mean(residual**2) + 1.0e-12)

    np.testing.assert_allclose(np.asarray(out["covariance"]), sigma2 * np.eye(2))
    np.testing.assert_allclose(np.asarray(out["covariance_std"]), np.sqrt(sigma2) * np.ones(2))
    np.testing.assert_allclose(np.asarray(out["covariance_correlation"]), np.eye(2))
    np.testing.assert_allclose(float(out["uq_ellipse_area_1sigma"]), np.pi * sigma2)
    assert out["sensitivity_map_rank"] == 2
    assert float(out["jacobian_condition_number"]) == 1.0


def test_covariance_diagnostics_flags_rank_deficient_sensitivity_map() -> None:
    jac = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    residual = np.array([1.0e-3, 2.0e-3, -1.0e-3])

    out = covariance_diagnostics(jac, residual, regularization=1.0e-6)

    assert out["sensitivity_map_rank"] == 1
    assert np.isinf(float(out["jacobian_condition_number"]))
    cov = np.asarray(out["covariance"])
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) >= 0.0)
    assert np.asarray(out["covariance_std"])[1] > np.asarray(out["covariance_std"])[0]


def test_covariance_diagnostics_handles_scalar_and_empty_parameter_maps() -> None:
    scalar = covariance_diagnostics(np.ones((3, 1)), np.array([1.0e-3, -1.0e-3, 0.0]))
    assert scalar["sensitivity_map_rank"] == 1
    assert np.asarray(scalar["covariance"]).shape == (1, 1)
    assert float(scalar["uq_ellipse_area_1sigma"]) == 0.0

    with pytest.raises(ValueError, match="at least one parameter column"):
        covariance_diagnostics(np.empty((2, 0)), np.array([1.0e-3, -1.0e-3]))


def test_covariance_diagnostics_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2, 1)), np.ones(2))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2)), np.ones(3))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2)), np.ones(2), regularization=-1.0)
