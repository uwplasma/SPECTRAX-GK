from __future__ import annotations

import pytest

import jax
import jax.numpy as jnp
import numpy as np

import spectraxgk
from spectraxgk.velocity_sharding import (
    build_velocity_sharding_plan,
    hermite_neighbor_reference,
    hermite_neighbor_shard_map,
    hermite_streaming_ladder_reference,
    hermite_streaming_ladder_shard_map,
    periodic_streaming_reference,
    periodic_streaming_shard_map,
    velocity_field_reduce_reference,
    velocity_field_reduce_shard_map,
)


def test_velocity_sharding_plan_prefers_species_then_hermite_for_6d_state() -> None:
    plan = build_velocity_sharding_plan((2, 4, 8, 16, 4, 32), num_devices=4)

    assert spectraxgk.build_velocity_sharding_plan is build_velocity_sharding_plan
    assert plan.dims == ("s", "l", "m", "ky", "kx", "z")
    assert plan.chunks["s"] == 2
    assert plan.chunks["m"] == 2
    assert plan.chunks["l"] == 1
    assert plan.active_axes == ("s", "m")
    assert plan.shard_shape == (1, 4, 4, 16, 4, 32)
    assert plan.needs_hermite_exchange is True
    assert plan.needs_field_reduction is True
    assert plan.field_reduction_axes == ("s", "m")
    assert plan.communication_pattern == "hermite_ghost_exchange+field_reduce_broadcast"
    assert plan.load_balance == pytest.approx(1.0)
    assert plan.to_dict()["num_devices"] == 4


def test_velocity_sharding_plan_uses_hermite_for_single_species_5d_state() -> None:
    plan = build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=2)

    assert plan.dims == ("l", "m", "ky", "kx", "z")
    assert plan.chunks["m"] == 2
    assert plan.active_axes == ("m",)
    assert plan.shard_shape == (4, 4, 16, 4, 32)
    assert plan.field_reduction_axes == ("m",)


def test_velocity_sharding_plan_supports_explicit_laguerre_fallback() -> None:
    plan = build_velocity_sharding_plan((1, 4, 3, 16, 4, 32), num_devices=4, axes=("hermite", "laguerre"))

    assert plan.chunks["m"] == 2
    assert plan.chunks["l"] == 2
    assert plan.active_axes == ("l", "m")


def test_velocity_sharding_plan_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="5 or 6"):
        build_velocity_sharding_plan((2, 3, 4), num_devices=2)
    with pytest.raises(ValueError, match="num_devices"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=0)
    with pytest.raises(ValueError, match="species sharding"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=2, axes=("species",))
    with pytest.raises(ValueError, match="could not be factored"):
        build_velocity_sharding_plan((1, 2, 3, 4, 1, 8), num_devices=5)
    with pytest.raises(ValueError, match="Unknown"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=2, axes=("banana",))


def test_hermite_neighbor_reference_matches_manual_shift() -> None:
    state = jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4))

    lower, upper = hermite_neighbor_reference(state)

    expected_lower = np.zeros_like(np.asarray(state))
    expected_upper = np.zeros_like(np.asarray(state))
    expected_lower[:, 1:, ...] = np.asarray(state)[:, :-1, ...]
    expected_upper[:, :-1, ...] = np.asarray(state)[:, 1:, ...]
    np.testing.assert_allclose(np.asarray(lower), expected_lower)
    np.testing.assert_allclose(np.asarray(upper), expected_upper)
    assert spectraxgk.hermite_neighbor_reference is hermite_neighbor_reference
    assert spectraxgk.hermite_neighbor_shard_map is hermite_neighbor_shard_map


def test_hermite_neighbor_shard_map_noops_to_reference_for_single_chunk() -> None:
    state = jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4))
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    lower, upper = hermite_neighbor_shard_map(state, plan, devices=[jax.devices()[0]])
    expected_lower, expected_upper = hermite_neighbor_reference(state)

    np.testing.assert_allclose(np.asarray(lower), np.asarray(expected_lower))
    np.testing.assert_allclose(np.asarray(upper), np.asarray(expected_upper))


def test_hermite_neighbor_shard_map_matches_reference_when_logical_devices_available() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices; artifact generator sets logical CPU devices")
    state = (jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4)) + 1j).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    lower, upper = hermite_neighbor_shard_map(state, plan, devices=devices[:2])
    expected_lower, expected_upper = hermite_neighbor_reference(state)

    np.testing.assert_allclose(np.asarray(lower), np.asarray(expected_lower))
    np.testing.assert_allclose(np.asarray(upper), np.asarray(expected_upper))


def test_hermite_neighbor_shard_map_rejects_unsupported_multi_axis_plan() -> None:
    state = jnp.zeros((2, 4, 8, 3, 1, 4), dtype=jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=4)

    with pytest.raises(NotImplementedError, match="only an active 'm' axis"):
        hermite_neighbor_shard_map(state, plan, devices=[jax.devices()[0]] * 4)


def test_velocity_field_reduce_reference_matches_manual_sum() -> None:
    state = jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4))

    reduced = velocity_field_reduce_reference(state, axis="hermite")

    np.testing.assert_allclose(np.asarray(reduced), np.asarray(state).sum(axis=1))
    assert spectraxgk.velocity_field_reduce_reference is velocity_field_reduce_reference
    assert spectraxgk.velocity_field_reduce_shard_map is velocity_field_reduce_shard_map


def test_velocity_field_reduce_shard_map_noops_to_reference_for_single_chunk() -> None:
    state = jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4))
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    reduced = velocity_field_reduce_shard_map(state, plan, devices=[jax.devices()[0]])
    expected = velocity_field_reduce_reference(state)

    np.testing.assert_allclose(np.asarray(reduced), np.asarray(expected))


def test_velocity_field_reduce_shard_map_matches_reference_when_logical_devices_available() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices; artifact generator sets logical CPU devices")
    state = (jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4)) + 1j).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    reduced = velocity_field_reduce_shard_map(state, plan, devices=devices[:2])
    expected = velocity_field_reduce_reference(state)

    np.testing.assert_allclose(np.asarray(reduced), np.asarray(expected))


def test_hermite_streaming_ladder_reference_matches_manual_coefficients() -> None:
    state = jnp.zeros((1, 4, 1, 1, 1), dtype=jnp.complex64)
    state = state.at[0, 2, 0, 0, 0].set(3.0 + 2.0j)

    ladder = hermite_streaming_ladder_reference(state, vth=2.0)

    expected = np.zeros_like(np.asarray(state))
    expected[0, 1, 0, 0, 0] = 2.0 * np.sqrt(2.0) * (3.0 + 2.0j)
    expected[0, 3, 0, 0, 0] = 2.0 * np.sqrt(3.0) * (3.0 + 2.0j)
    np.testing.assert_allclose(np.asarray(ladder), expected, rtol=1.0e-6, atol=1.0e-6)
    assert spectraxgk.hermite_streaming_ladder_reference is hermite_streaming_ladder_reference
    assert spectraxgk.hermite_streaming_ladder_shard_map is hermite_streaming_ladder_shard_map


def test_hermite_streaming_ladder_shard_map_noops_to_reference_for_single_chunk() -> None:
    state = (jnp.arange(2 * 5 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 5, 3, 1, 4)) + 1j).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    ladder = hermite_streaming_ladder_shard_map(state, plan, vth=1.5, devices=[jax.devices()[0]])
    expected = hermite_streaming_ladder_reference(state, vth=1.5)

    np.testing.assert_allclose(np.asarray(ladder), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)


def test_hermite_streaming_ladder_shard_map_matches_reference_when_logical_devices_available() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices; artifact generator sets logical CPU devices")
    state = (jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4)) + 1j).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    ladder = hermite_streaming_ladder_shard_map(state, plan, vth=1.5, devices=devices[:2])
    expected = hermite_streaming_ladder_reference(state, vth=1.5)

    np.testing.assert_allclose(np.asarray(ladder), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)


def test_periodic_streaming_reference_matches_production_streaming_term() -> None:
    from spectraxgk.basis import hermite_ladder_coeffs
    from spectraxgk.terms.operators import streaming_term

    ns, nl, nm, ny, nx, nz = 1, 2, 4, 2, 1, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    kz = jnp.fft.fftfreq(nz, d=float(z[1] - z[0])) * 2.0 * jnp.pi
    state = jnp.zeros((ns, nl, nm, ny, nx, nz), dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, 0, :].set(jnp.exp(1j * z))
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_p = sqrt_p[:nm].reshape((1, 1, nm, 1, 1, 1))
    sqrt_m = sqrt_m[:nm].reshape((1, 1, nm, 1, 1, 1))
    vth = jnp.asarray([1.5], dtype=jnp.float32)

    observed = periodic_streaming_reference(state, kz=kz, vth=vth)
    expected = streaming_term(
        state,
        kz=kz,
        vth=vth.reshape((1, 1, 1, 1, 1, 1)),
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)
    assert spectraxgk.periodic_streaming_reference is periodic_streaming_reference
    assert spectraxgk.periodic_streaming_shard_map is periodic_streaming_shard_map


def test_periodic_streaming_shard_map_noops_to_reference_for_single_chunk() -> None:
    nz = 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    kz = jnp.fft.fftfreq(nz, d=float(z[1] - z[0])) * 2.0 * jnp.pi
    state = jnp.zeros((1, 2, 5, 2, 1, nz), dtype=jnp.complex64)
    state = state.at[0, 0, 2, 0, 0, :].set(jnp.exp(2j * z))
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    observed = periodic_streaming_shard_map(state, plan, kz=kz, vth=jnp.asarray([1.2]), devices=[jax.devices()[0]])
    expected = periodic_streaming_reference(state, kz=kz, vth=jnp.asarray([1.2]))

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)


def test_periodic_streaming_shard_map_matches_reference_when_logical_devices_available() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices; artifact generator sets logical CPU devices")
    nz = 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    kz = jnp.fft.fftfreq(nz, d=float(z[1] - z[0])) * 2.0 * jnp.pi
    state = jnp.zeros((1, 2, 6, 2, 1, nz), dtype=jnp.complex64)
    state = state.at[0, 0, 2, 0, 0, :].set(jnp.exp(2j * z))
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    observed = periodic_streaming_shard_map(state, plan, kz=kz, vth=jnp.asarray([1.2]), devices=devices[:2])
    expected = periodic_streaming_reference(state, kz=kz, vth=jnp.asarray([1.2]))

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)
