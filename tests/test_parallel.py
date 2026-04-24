from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
import spectraxgk.parallel as parallel


def test_ky_scan_batches_are_balanced_and_order_preserving() -> None:
    assert spectraxgk.ky_scan_batches is parallel.ky_scan_batches
    assert spectraxgk.batch_map is parallel.batch_map
    ky = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    chunks = parallel.ky_scan_batches(ky, n_batches=2)

    assert [chunk.tolist() for chunk in chunks] == [[0.1, 0.2, 0.3], [0.4, 0.5]]
    assert np.allclose(np.concatenate(chunks), ky)


def test_batch_map_matches_vmap_on_single_device() -> None:
    values = jnp.linspace(0.0, 1.0, 7)

    def fn(x):
        return jnp.asarray([x, x**2 + 1.0])

    observed = parallel.batch_map(fn, values, batch_size=3, devices=[jax.devices()[0]])
    expected = jax.vmap(fn)(values)

    assert np.allclose(np.asarray(observed), np.asarray(expected))


def test_pad_to_multiple_preserves_prefix_and_reports_original_size() -> None:
    padded, original_n = parallel.pad_to_multiple(jnp.asarray([1.0, 2.0, 3.0]), 4)

    assert original_n == 3
    assert np.allclose(np.asarray(padded), np.asarray([1.0, 2.0, 3.0, 3.0]))


def test_parallel_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        parallel.split_evenly(np.arange(3), 0)
    with pytest.raises(ValueError):
        parallel.ky_scan_batches(np.ones((2, 2)), n_batches=2)
    with pytest.raises(ValueError):
        parallel.batch_map(lambda x: x, jnp.asarray([]))
    with pytest.raises(ValueError):
        parallel.pad_to_multiple(jnp.asarray([1.0]), 0)
