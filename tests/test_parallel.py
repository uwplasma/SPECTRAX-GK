from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
import spectraxgk.parallel as parallel


def test_parallel_public_api_exports_are_stable() -> None:
    public_names = (
        "ParallelIdentityReport",
        "batch_map",
        "batch_map_identity_report",
        "independent_map",
        "ky_scan_batches",
        "parallel_identity_report",
    )

    assert set(public_names) <= set(spectraxgk.__all__)
    assert set(public_names) <= set(parallel.__all__)
    for name in public_names:
        assert getattr(spectraxgk, name) is getattr(parallel, name)


def test_ky_scan_batches_are_balanced_and_order_preserving() -> None:
    ky = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    chunks = parallel.ky_scan_batches(ky, n_batches=2)

    assert [chunk.tolist() for chunk in chunks] == [[0.1, 0.2, 0.3], [0.4, 0.5]]
    assert np.allclose(np.concatenate(chunks), ky)


def test_split_evenly_handles_multidimensional_batches_without_empty_chunks() -> None:
    values = np.arange(15).reshape(5, 3)
    chunks = parallel.split_evenly(values, n_parts=8)

    assert [chunk.shape for chunk in chunks] == [(1, 3)] * 5
    np.testing.assert_array_equal(np.concatenate(chunks, axis=0), values)


def test_batch_map_matches_vmap_on_single_device() -> None:
    values = jnp.linspace(0.0, 1.0, 7)

    def fn(x):
        return jnp.asarray([x, x**2 + 1.0])

    observed = parallel.batch_map(fn, values, batch_size=3, devices=[jax.devices()[0]])
    expected = jax.vmap(fn)(values)

    assert np.allclose(np.asarray(observed), np.asarray(expected))


def test_batch_map_matches_vmap_for_pytree_outputs_single_device() -> None:
    values = jnp.linspace(0.0, 1.0, 6)

    def fn(x):
        return {
            "features": jnp.asarray([x, x**2 + 1.0]),
            "moments": (x + 2.0, jnp.asarray([x - 1.0])),
        }

    observed = parallel.batch_map(fn, values, batch_size=2, devices=[jax.devices()[0]])
    expected = jax.vmap(fn)(values)

    jax.tree_util.tree_map(
        lambda obs, exp: np.testing.assert_allclose(np.asarray(obs), np.asarray(exp)),
        observed,
        expected,
    )


def test_parallel_identity_report_records_tree_errors_and_metadata() -> None:
    reference = {
        "gamma": jnp.asarray([1.0, 2.0, 4.0]),
        "omega": (jnp.asarray([0.25]),),
    }
    observed = {
        "gamma": jnp.asarray([1.0, 2.0 + 1e-6, 4.0]),
        "omega": (jnp.asarray([0.25]),),
    }

    report = parallel.parallel_identity_report(
        reference,
        observed,
        kind="unit_identity",
        problem_size=3,
        requested_workers=2,
        actual_workers=2,
        backend="cpu",
        atol=1e-5,
        rtol=1e-5,
        metadata={"observable": "linear_scan"},
    )

    payload = report.to_dict()
    assert report.identity_passed is True
    assert payload["kind"] == "unit_identity"
    assert payload["backend"] == "cpu"
    assert payload["problem_size"] == 3
    assert payload["metadata"] == {"observable": "linear_scan"}
    assert payload["max_abs_error"] > 0.0


def test_parallel_identity_report_rejects_mismatched_pytrees_and_bad_counts() -> None:
    with pytest.raises(ValueError, match="different structures"):
        parallel.parallel_identity_report(
            {"a": jnp.asarray([1.0])},
            {"b": jnp.asarray([1.0])},
            kind="bad",
            problem_size=1,
            requested_workers=1,
        )
    with pytest.raises(ValueError, match="problem_size"):
        parallel.parallel_identity_report(
            jnp.asarray([1.0]),
            jnp.asarray([1.0]),
            kind="bad",
            problem_size=0,
            requested_workers=1,
        )
    with pytest.raises(ValueError, match="actual_workers"):
        parallel.parallel_identity_report(
            jnp.asarray([1.0]),
            jnp.asarray([1.0]),
            kind="bad",
            problem_size=1,
            requested_workers=1,
            actual_workers=2,
        )


def test_batch_map_identity_report_is_serial_vmap_identity_gate() -> None:
    values = jnp.linspace(0.1, 0.9, 9)

    def fn(x):
        return {
            "mode": jnp.asarray([x, x**2, jnp.sin(x)]),
            "flux_proxy": x**3 + 0.5,
        }

    report = parallel.batch_map_identity_report(
        fn,
        values,
        batch_size=4,
        devices=[jax.devices()[0]],
        atol=0.0,
        rtol=0.0,
    )

    assert report.kind == "batch_map_serial_identity"
    assert report.identity_passed is True
    assert report.problem_size == 9
    assert report.actual_workers == 1
    assert report.metadata["batch_size"] == 4


def test_pad_to_multiple_preserves_prefix_and_reports_original_size() -> None:
    padded, original_n = parallel.pad_to_multiple(jnp.asarray([1.0, 2.0, 3.0]), 4)

    assert original_n == 3
    assert np.allclose(np.asarray(padded), np.asarray([1.0, 2.0, 3.0, 3.0]))


def test_pad_to_multiple_preserves_batch_tail_for_multidimensional_values() -> None:
    values = jnp.arange(10, dtype=jnp.float32).reshape(5, 2)
    padded, original_n = parallel.pad_to_multiple(values, 4)

    assert original_n == 5
    assert padded.shape == (8, 2)
    np.testing.assert_array_equal(np.asarray(padded[:5]), np.asarray(values))
    np.testing.assert_array_equal(
        np.asarray(padded[5:]),
        np.repeat(np.asarray(values[-1:]), 3, axis=0),
    )


def test_pad_to_multiple_noops_when_already_aligned_and_split_empty() -> None:
    values = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    padded, original_n = parallel.pad_to_multiple(values, 2)

    assert original_n == 4
    assert np.allclose(np.asarray(padded), np.asarray(values))
    assert parallel.split_evenly(np.asarray([]), 3) == []


def test_batch_map_multi_device_branch_preserves_vmap_identity(monkeypatch) -> None:
    def fake_pmap(fn, devices):
        assert len(devices) == 2

        def mapped(sharded):
            return jnp.stack([fn(shard) for shard in sharded], axis=0)

        return mapped

    monkeypatch.setattr(parallel.jax, "pmap", fake_pmap)
    values = jnp.linspace(0.0, 1.0, 5)

    def fn(x):
        return jnp.asarray([x, x + 1.0])

    observed = parallel.batch_map(
        fn, values, batch_size=3, devices=[object(), object()]
    )
    expected = jax.vmap(fn)(values)

    assert np.allclose(np.asarray(observed), np.asarray(expected))


def test_batch_map_multi_device_branch_drops_padding_and_preserves_chunk_order(
    monkeypatch,
) -> None:
    seen_shards: list[np.ndarray] = []

    def fake_pmap(fn, devices):
        assert len(devices) == 3

        def mapped(sharded):
            seen_shards.append(np.asarray(sharded))
            return jnp.stack([fn(shard) for shard in sharded], axis=0)

        return mapped

    monkeypatch.setattr(parallel.jax, "pmap", fake_pmap)
    values = jnp.arange(7, dtype=jnp.float32)

    def fn(x):
        return jnp.asarray([x, 10.0 * x + jnp.mod(x, 2.0)])

    observed = parallel.batch_map(
        fn, values, batch_size=5, devices=[object(), object(), object()]
    )
    expected = jax.vmap(fn)(values)

    assert np.allclose(np.asarray(observed), np.asarray(expected))
    assert [tuple(shard.shape) for shard in seen_shards] == [(3, 2), (3, 2)]
    np.testing.assert_allclose(
        seen_shards[0].reshape(-1),
        np.asarray([0, 1, 2, 3, 3, 3], dtype=float),
    )
    np.testing.assert_allclose(
        seen_shards[1].reshape(-1),
        np.asarray([4, 5, 6, 6, 6, 6], dtype=float),
    )


def test_batch_map_single_device_fallback_never_calls_pmap(monkeypatch) -> None:
    monkeypatch.setattr(
        parallel.jax,
        "pmap",
        lambda *args, **kwargs: pytest.fail(
            "single-device fallback must use vmap chunks, not pmap"
        ),
    )
    values = jnp.arange(5, dtype=jnp.float32)

    def fn(x):
        return {"linear": x + 1.0, "quadratic": jnp.asarray([x**2])}

    observed = parallel.batch_map(fn, values, batch_size=2, devices=[jax.devices()[0]])
    expected = jax.vmap(fn)(values)

    jax.tree_util.tree_map(
        lambda obs, exp: np.testing.assert_allclose(np.asarray(obs), np.asarray(exp)),
        observed,
        expected,
    )


def test_batch_map_multi_device_branch_preserves_pytree_identity(monkeypatch) -> None:
    def fake_pmap(fn, devices):
        assert len(devices) == 2

        def mapped(sharded):
            return jax.tree_util.tree_map(
                lambda *parts: jnp.stack(parts, axis=0),
                *[fn(shard) for shard in sharded],
            )

        return mapped

    monkeypatch.setattr(parallel.jax, "pmap", fake_pmap)
    values = jnp.linspace(0.0, 1.0, 5)

    def fn(x):
        return {"field": jnp.asarray([x, x + 1.0]), "flux": x**2}

    observed = parallel.batch_map(
        fn, values, batch_size=3, devices=[object(), object()]
    )
    expected = jax.vmap(fn)(values)

    jax.tree_util.tree_map(
        lambda obs, exp: np.testing.assert_allclose(np.asarray(obs), np.asarray(exp)),
        observed,
        expected,
    )


def test_parallel_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        parallel.split_evenly(np.arange(3), 0)
    with pytest.raises(ValueError):
        parallel.ky_scan_batches(np.ones((2, 2)), n_batches=2)
    with pytest.raises(ValueError):
        parallel.batch_map(lambda x: x, jnp.asarray([]))
    with pytest.raises(ValueError):
        parallel.batch_map(lambda x: x, jnp.asarray([1.0]), batch_size=0)
    with pytest.raises(ValueError):
        parallel.pad_to_multiple(jnp.asarray([1.0]), 0)
    with pytest.raises(ValueError):
        parallel.pad_to_multiple(jnp.asarray([]), 2)
    with pytest.raises(ValueError):
        parallel.independent_map(lambda x: x, [1], workers=0)
    with pytest.raises(ValueError):
        parallel.independent_map(lambda x: x, [1], workers=2, executor="mpi")


def test_independent_map_preserves_serial_order_and_nested_outputs() -> None:
    values = [3, 1, 2]

    def fn(value: int) -> dict[str, int]:
        return {"x": value, "x2": value * value}

    serial = parallel.independent_map(fn, values, workers=1)
    threaded = parallel.independent_map(fn, values, workers=2)

    assert serial == [{"x": 3, "x2": 9}, {"x": 1, "x2": 1}, {"x": 2, "x2": 4}]
    assert threaded == serial
    assert parallel.independent_map(fn, [], workers=3) == []


def test_independent_map_clips_thread_workers_and_accepts_executor_aliases(
    monkeypatch,
) -> None:
    records: list[tuple[int, tuple[int, ...]]] = []

    class FakeThreadPool:
        def __init__(self, *, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            materialized = tuple(items)
            records.append((self.max_workers, materialized))
            return [fn(item) for item in materialized]

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", FakeThreadPool)

    observed = parallel.independent_map(
        lambda value: value * 11, [3, 1, 4], workers=99, executor="threads"
    )

    assert observed == [33, 11, 44]
    assert records == [(3, (3, 1, 4))]
