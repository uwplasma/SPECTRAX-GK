from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.runtime_policies import (
    _active_kx_indices,
    _active_ky_indices,
    _nearest_index_from_candidates,
    _select_nonlinear_mode_indices,
    _validate_dealias_mask_shape,
)


def test_nearest_index_from_candidates_locks_retained_mode_tie_order() -> None:
    values = np.asarray([0.0, 0.5, 1.0, 1.5])

    assert _nearest_index_from_candidates(values, 0.75, np.asarray([1, 2])) == 1
    assert _nearest_index_from_candidates(values, 1.4, np.asarray([0, 2])) == 2

    with pytest.raises(ValueError, match="candidate indices"):
        _nearest_index_from_candidates(values, 1.0, np.asarray([], dtype=int))


def test_active_dealias_indices_fall_back_to_full_axis_for_empty_masks() -> None:
    empty_mask = np.zeros((3, 4), dtype=bool)

    np.testing.assert_array_equal(_active_ky_indices(empty_mask, 3), [0, 1, 2])
    np.testing.assert_array_equal(_active_kx_indices(empty_mask, 1, 4), [0, 1, 2, 3])

    mixed_mask = np.asarray(
        [
            [False, False, False, True],
            [False, False, False, False],
            [True, False, False, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(_active_ky_indices(mixed_mask, 3), [0, 2])
    np.testing.assert_array_equal(_active_kx_indices(mixed_mask, 2, 4), [0])


def test_select_nonlinear_mode_indices_uses_nearest_retained_dealiased_mode() -> None:
    grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.4, 0.8]),
        kx=np.asarray([-1.0, 0.0, 1.0]),
        dealias_mask=np.asarray(
            [
                [False, False, True],
                [False, False, False],
                [True, False, False],
            ],
            dtype=bool,
        ),
    )

    ky_idx, kx_idx = _select_nonlinear_mode_indices(
        grid,
        ky_target=0.39,
        kx_target=0.2,
        use_dealias_mask=True,
    )

    assert (ky_idx, kx_idx) == (0, 2)


def test_select_nonlinear_mode_indices_validates_dealias_mask_shape() -> None:
    bad_grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.4]),
        kx=np.asarray([-1.0, 0.0, 1.0]),
        dealias_mask=np.ones((2, 2), dtype=bool),
    )

    with pytest.raises(ValueError, match="dealias_mask shape"):
        _select_nonlinear_mode_indices(
            bad_grid,
            ky_target=0.4,
            kx_target=0.0,
            use_dealias_mask=True,
        )


def test_validate_dealias_mask_shape_returns_boolean_view() -> None:
    mask = _validate_dealias_mask_shape(
        np.asarray([[1, 0], [0, 1]], dtype=int),
        ky_size=2,
        kx_size=2,
    )

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])
