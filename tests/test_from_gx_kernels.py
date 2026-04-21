from __future__ import annotations

import numpy as np
import pytest

from spectraxgk.from_gx.kernels import (
    finite_diff_nonuniform,
    gx_derm,
    gx_dermv,
    gx_nperiod_data_extend,
    gx_reflect_n_append,
    nperiod_contract,
    nperiod_mask,
)


def test_nperiod_helpers_contract_arrays() -> None:
    theta = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    values = np.arange(theta.size)
    mask = np.asarray(nperiod_mask(theta, 1.0))
    assert mask.tolist() == [False, True, True, True, False]
    contracted_values, contracted_theta = nperiod_contract.__wrapped__(values, theta, 1.0)
    np.testing.assert_allclose(np.asarray(contracted_values), [1, 2, 3])
    np.testing.assert_allclose(np.asarray(contracted_theta), [-1.0, 0.0, 1.0])


def test_finite_diff_nonuniform_matches_quadratic_derivative() -> None:
    grid = np.array([0.0, 0.5, 1.5, 3.0])
    values = grid**2
    diff = np.asarray(finite_diff_nonuniform(values, grid))
    np.testing.assert_allclose(diff[1:-1], 2.0 * grid[1:-1], atol=1.0e-6)
    assert np.isfinite(diff[[0, -1]]).all()
    np.testing.assert_allclose(np.asarray(finite_diff_nonuniform(np.array([1.0, 2.0]), np.array([0.0, 1.0]))), [0.0, 0.0])


def test_gx_derm_handles_1d_and_2d_axes() -> None:
    arr = np.array([1.0, 2.0, 4.0, 7.0])
    np.testing.assert_allclose(np.asarray(gx_derm(arr, axis="l", parity="e")), [0.0, 3.0, 5.0, 0.0])
    np.testing.assert_allclose(np.asarray(gx_derm(arr, axis="r")), [2.0, 3.0, 5.0, 6.0])
    arr2 = np.arange(12.0).reshape(3, 4)
    out_r = np.asarray(gx_derm(arr2, axis="r"))
    out_l = np.asarray(gx_derm(arr2, axis="l", parity="o"))
    assert out_r.shape == arr2.shape
    assert out_l.shape == arr2.shape
    with pytest.raises(ValueError):
        gx_derm(np.ones((2, 2, 2)), axis="l")
    with pytest.raises(ValueError):
        gx_derm(arr, axis="bad")


def test_gx_dermv_handles_weighted_derivatives() -> None:
    grid = np.array([0.0, 1.0, 2.0, 4.0])
    values = grid**2
    out_l = np.asarray(gx_dermv(values, grid, axis="l", parity="o"))
    assert out_l.shape == values.shape
    assert np.isfinite(out_l).all()

    arr2 = np.vstack([values, values + 1.0])
    grid2 = np.vstack([grid, grid])
    out2 = np.asarray(gx_dermv(arr2, grid2, axis="r"))
    assert out2.shape == arr2.shape

    with pytest.raises(ValueError):
        gx_dermv(np.ones(3), np.ones(4), axis="l")
    with pytest.raises(ValueError):
        gx_dermv(np.ones((2, 2, 2)), np.ones((2, 2, 2)), axis="l")
    with pytest.raises(ValueError):
        gx_dermv(values, grid, axis="bad")


def test_nperiod_extension_and_reflection_helpers() -> None:
    base = np.array([0.0, 1.0, 2.0])
    even = np.asarray(gx_nperiod_data_extend(base, 2, istheta=False, parity="e"))
    odd = np.asarray(gx_nperiod_data_extend(base, 2, istheta=False, parity="o"))
    theta = np.asarray(gx_nperiod_data_extend(base, 2, istheta=True))
    assert even.size == 7
    assert odd.size == 7
    assert theta.size == 7
    np.testing.assert_allclose(np.asarray(gx_reflect_n_append(base, "e")), [2.0, 1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(gx_reflect_n_append(base, "o")), [-2.0, -1.0, 0.0, 1.0, 2.0])
