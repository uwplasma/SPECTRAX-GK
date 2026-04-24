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


def test_covariance_diagnostics_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2, 1)), np.ones(2))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2)), np.ones(3))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2)), np.ones(2), regularization=-1.0)
