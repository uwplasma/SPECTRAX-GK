from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import spectraxgk
from spectraxgk.autodiff_validation import (
    autodiff_finite_difference_report,
    central_finite_difference_jacobian,
    covariance_diagnostics,
)
from spectraxgk.quasilinear import quasilinear_feature_objective


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


def test_autodiff_finite_difference_report_matches_closed_form_jacobian() -> None:
    assert spectraxgk.autodiff_finite_difference_report is autodiff_finite_difference_report
    assert spectraxgk.central_finite_difference_jacobian is central_finite_difference_jacobian

    def fn(x):
        return jnp.asarray([x[0] ** 2 + 3.0 * x[1], x[0] * x[1]])

    p = jnp.asarray([0.4, -0.2])
    report = autodiff_finite_difference_report(fn, p, step=1.0e-3, rtol=5.0e-4, atol=5.0e-6)

    assert report["passed"] is True
    jac_ad = np.asarray(report["jacobian_ad"])
    np.testing.assert_allclose(jac_ad, np.asarray([[0.8, 3.0], [-0.2, 0.4]]), rtol=1.0e-6)
    assert float(report["tangent_max_abs_error"]) < 1.0e-4


def test_quasilinear_feature_objective_derivative_gate() -> None:
    features = jnp.asarray([0.2, 0.8, 1.5])
    report = autodiff_finite_difference_report(
        lambda x: quasilinear_feature_objective(x, csat=0.7),
        features,
        step=1.0e-3,
        rtol=1.0e-4,
        atol=1.0e-5,
    )

    assert report["passed"] is True
    jac = np.asarray(report["jacobian_ad"]).reshape(3)
    expected = np.asarray([0.7 * 1.5 / 0.8, -0.7 * 1.5 * 0.2 / 0.8**2, 0.7 * 0.2 / 0.8])
    np.testing.assert_allclose(jac, expected, rtol=1.0e-6)


def test_autodiff_finite_difference_report_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones((2, 1)))
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones(2), step=0.0)
    with pytest.raises(ValueError):
        autodiff_finite_difference_report(lambda x: x, jnp.ones((2, 1)))
    with pytest.raises(ValueError):
        autodiff_finite_difference_report(lambda x: x, jnp.ones(2), direction=jnp.ones(3))
