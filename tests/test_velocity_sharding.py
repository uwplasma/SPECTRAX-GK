from __future__ import annotations

from dataclasses import replace

import pytest

import jax
import jax.numpy as jnp
import numpy as np

import spectraxgk
from spectraxgk.velocity_sharding import (
    build_velocity_sharding_plan,
    curvature_gradb_drift_reference,
    curvature_gradb_drift_shard_map,
    diamagnetic_drive_reference,
    diamagnetic_drive_shard_map,
    electrostatic_phi_reference,
    electrostatic_phi_shard_map,
    hermite_neighbor_reference,
    hermite_neighbor_shard_map,
    hermite_shift_reference,
    hermite_shift_shard_map,
    hermite_streaming_ladder_reference,
    hermite_streaming_ladder_shard_map,
    mirror_drift_reference,
    mirror_drift_shard_map,
    periodic_streaming_reference,
    periodic_streaming_shard_map,
    velocity_field_reduce_reference,
    velocity_field_reduce_shard_map,
)


def _install_fake_two_way_shard_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise shard-map branches without requiring CI-visible logical devices."""

    class FakeMesh:
        def __init__(self, devices, names):
            self.devices = devices
            self.names = names

    class FakeNamedSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    def fake_partition_spec(*axes):
        return axes

    def local_hermite_slice(arr):
        if getattr(arr, "ndim", 0) not in (5, 6):
            return arr
        m_axis = 1 if arr.ndim == 5 else 2
        local_m = int(arr.shape[m_axis]) // 2
        index = [slice(None)] * arr.ndim
        index[m_axis] = slice(0, local_m)
        return arr[tuple(index)]

    def fake_shard_map(fn, **_kwargs):
        def mapped(arr):
            return fn(local_hermite_slice(arr))

        return mapped

    monkeypatch.setattr("jax.sharding.Mesh", FakeMesh)
    monkeypatch.setattr("jax.sharding.NamedSharding", FakeNamedSharding)
    monkeypatch.setattr("jax.sharding.PartitionSpec", fake_partition_spec)
    monkeypatch.setattr(jax, "shard_map", fake_shard_map)
    monkeypatch.setattr(jax, "device_put", lambda arr, _sharding=None: arr)
    monkeypatch.setattr(jax.lax, "ppermute", lambda value, *_args, **_kwargs: jnp.zeros_like(value))
    monkeypatch.setattr(jax.lax, "psum", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(jax.lax, "axis_index", lambda *_args, **_kwargs: 0)


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
    with pytest.raises(ValueError, match="entries"):
        build_velocity_sharding_plan((4, 0, 16, 4, 32), num_devices=1)
    with pytest.raises(ValueError, match="num_devices"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=0)
    with pytest.raises(ValueError, match="ghost"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=1, hermite_ghost_depth=-1)
    with pytest.raises(ValueError, match="species sharding"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=2, axes=("species",))
    with pytest.raises(ValueError, match="could not be factored"):
        build_velocity_sharding_plan((1, 2, 3, 4, 1, 8), num_devices=5)
    with pytest.raises(ValueError, match="Unknown"):
        build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=2, axes=("banana",))
    field_only = build_velocity_sharding_plan((4, 8, 16, 4, 32), num_devices=2, axes=("laguerre",))
    assert field_only.communication_pattern == "field_reduce_broadcast"


def test_velocity_sharding_rejects_invalid_hermite_exchange_requests() -> None:
    state = jnp.zeros((2, 6, 3, 1, 4), dtype=jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    bad_shape = state[:, :5, ...]
    bad_chunks = replace(plan, chunks={**plan.chunks, "m": 0})
    uneven = replace(plan, state_shape=bad_shape.shape)
    laguerre_plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("laguerre",))

    with pytest.raises(ValueError, match="state shape"):
        hermite_neighbor_shard_map(bad_shape, plan)
    with pytest.raises(ValueError, match="chunk count"):
        hermite_neighbor_shard_map(state, bad_chunks)
    with pytest.raises(ValueError, match="divide evenly"):
        hermite_neighbor_shard_map(bad_shape, uneven, devices=[object(), object()])
    with pytest.raises(ValueError, match="not enough devices"):
        hermite_neighbor_shard_map(state, plan, devices=[object()])

    np.testing.assert_allclose(np.asarray(hermite_shift_shard_map(state, plan, offset=0)), np.asarray(state))
    np.testing.assert_allclose(np.asarray(hermite_shift_shard_map(state, plan, offset=99)), np.zeros_like(np.asarray(state)))
    with pytest.raises(ValueError, match="state shape"):
        hermite_shift_shard_map(bad_shape, plan, offset=1)
    with pytest.raises(NotImplementedError, match="active 'm'"):
        hermite_shift_shard_map(state, laguerre_plan, offset=1)
    with pytest.raises(ValueError, match="divide evenly"):
        hermite_shift_shard_map(bad_shape, uneven, offset=1, devices=[object(), object()])
    with pytest.raises(NotImplementedError, match="local shard size"):
        hermite_shift_shard_map(state, plan, offset=4, devices=[object(), object()])
    with pytest.raises(ValueError, match="not enough devices"):
        hermite_shift_shard_map(state, plan, offset=1, devices=[object()])


def test_velocity_field_reduce_rejects_invalid_axes_and_plans() -> None:
    state = jnp.zeros((2, 6, 3, 1, 4), dtype=jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    bad_shape = state[:, :5, ...]
    uneven = replace(plan, state_shape=bad_shape.shape)
    laguerre_plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("laguerre",))

    with pytest.raises(ValueError, match="Unknown"):
        velocity_field_reduce_reference(state, axis="banana")
    with pytest.raises(ValueError, match="not present"):
        velocity_field_reduce_reference(state, axis="species")
    with pytest.raises(ValueError, match="state shape"):
        velocity_field_reduce_shard_map(bad_shape, plan)
    with pytest.raises(ValueError, match="Unknown"):
        velocity_field_reduce_shard_map(state, plan, axis="banana")
    with pytest.raises(ValueError, match="not present"):
        velocity_field_reduce_shard_map(state, plan, axis="species")
    with pytest.raises(NotImplementedError, match="Hermite axis"):
        velocity_field_reduce_shard_map(state, plan, axis="laguerre")
    with pytest.raises(NotImplementedError, match="active 'm'"):
        velocity_field_reduce_shard_map(state, laguerre_plan)
    with pytest.raises(ValueError, match="divide evenly"):
        velocity_field_reduce_shard_map(bad_shape, uneven, devices=[object(), object()])
    with pytest.raises(ValueError, match="not enough devices"):
        velocity_field_reduce_shard_map(state, plan, devices=[object()])


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


def test_mocked_shard_map_exercises_multi_device_velocity_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_two_way_shard_map(monkeypatch)
    state = (jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4)) + 1j).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    devices = [object(), object()]

    lower, upper = hermite_neighbor_shard_map(state, plan, devices=devices)
    assert lower.shape == (2, 3, 3, 1, 4)
    assert upper.shape == (2, 3, 3, 1, 4)
    assert hermite_shift_shard_map(state, plan, offset=1, devices=devices).shape == (2, 3, 3, 1, 4)
    assert hermite_shift_shard_map(state, plan, offset=-1, devices=devices).shape == (2, 3, 3, 1, 4)
    assert velocity_field_reduce_shard_map(state, plan, devices=devices).shape == (2, 3, 1, 4)

    kz = jnp.asarray([0.0, 1.0, -1.0, -2.0], dtype=jnp.float32)
    local_coeffs = jnp.ones((1, 3, 1, 1, 1), dtype=state.dtype)
    monkeypatch.setattr(
        "spectraxgk.velocity_sharding._hermite_ladder_coefficients",
        lambda _arr: (local_coeffs, local_coeffs, 1),
    )
    assert hermite_streaming_ladder_shard_map(state, plan, vth=1.2, devices=devices).shape == (2, 3, 3, 1, 4)
    assert periodic_streaming_shard_map(state, plan, kz=kz, vth=1.2, devices=devices).shape == (2, 3, 3, 1, 4)

    local_m = 3
    local_m_shape = (1, local_m, 1, 1, 1)
    ell = jnp.arange(state.shape[0], dtype=jnp.float32).reshape((state.shape[0], 1, 1, 1, 1))
    local_m_values = jnp.arange(local_m, dtype=jnp.float32).reshape(local_m_shape)
    mirror = mirror_drift_shard_map(
        state,
        plan,
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        bgrad=jnp.ones((state.shape[-1],), dtype=jnp.float32),
        ell=ell,
        sqrt_m=jnp.sqrt(local_m_values),
        sqrt_m_p1=jnp.sqrt(local_m_values + 1.0),
        devices=devices,
    )
    assert mirror.shape == (2, 3, 3, 1, 4)
    Jl = jnp.ones((state.shape[0], state.shape[2], state.shape[3], state.shape[4]), dtype=jnp.float32)
    phi = electrostatic_phi_shard_map(
        state,
        plan,
        Jl=Jl,
        tau_e=1.0,
        charge=1.0,
        density=1.0,
        tz=1.0,
        mask0=jnp.zeros((state.shape[2], state.shape[3], state.shape[4]), dtype=bool),
        devices=devices,
    )
    assert phi.shape == (3, 1, 4)
    assert electrostatic_phi_shard_map(
        state,
        plan,
        Jl=Jl[None, ...],
        tau_e=1.0,
        devices=devices,
    ).shape == (3, 1, 4)
    diamagnetic = diamagnetic_drive_shard_map(
        state,
        plan,
        phi=phi,
        Jl=Jl,
        l4=jnp.arange(state.shape[0], dtype=jnp.float32).reshape((state.shape[0], 1, 1, 1)),
        tprim=6.9,
        fprim=2.2,
        omega_star_scale=1.0,
        ky=jnp.asarray([0.0, 0.3, 0.6], dtype=jnp.float32),
        devices=devices,
    )
    assert diamagnetic.shape == (2, 3, 3, 1, 4)


def test_hermite_shift_reference_matches_manual_offsets() -> None:
    state = jnp.arange(2 * 6 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 6, 3, 1, 4))

    upper2 = hermite_shift_reference(state, offset=2)
    lower2 = hermite_shift_reference(state, offset=-2)

    expected_upper2 = np.zeros_like(np.asarray(state))
    expected_lower2 = np.zeros_like(np.asarray(state))
    expected_upper2[:, :-2, ...] = np.asarray(state)[:, 2:, ...]
    expected_lower2[:, 2:, ...] = np.asarray(state)[:, :-2, ...]
    np.testing.assert_allclose(np.asarray(upper2), expected_upper2)
    np.testing.assert_allclose(np.asarray(lower2), expected_lower2)
    assert spectraxgk.hermite_shift_reference is hermite_shift_reference
    assert spectraxgk.hermite_shift_shard_map is hermite_shift_shard_map


def test_hermite_shift_shard_map_matches_reference_when_logical_devices_available() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices; artifact generator sets logical CPU devices")
    state = (jnp.arange(2 * 8 * 3 * 1 * 4, dtype=jnp.float32).reshape((2, 8, 3, 1, 4)) + 1j).astype(jnp.complex64)
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    for offset in (-2, -1, 1, 2):
        observed = hermite_shift_shard_map(state, plan, offset=offset, devices=devices[:2])
        expected = hermite_shift_reference(state, offset=offset)
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected))


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


def _small_periodic_field_problem():
    from spectraxgk.config import CycloneBaseCase, GridConfig
    from spectraxgk.geometry import SAlphaGeometry
    from spectraxgk.grids import build_spectral_grid
    from spectraxgk.linear import LinearParams, build_linear_cache

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic"))
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(beta=0.0, fapar=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=6)
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros((2, 6, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    state = state.at[0, 0, min(1, grid.ky.size - 1), 0, :].set(0.2 * jnp.exp(1j * z))
    state = state.at[1, 0, min(2, grid.ky.size - 1), 0, :].set(0.1 * jnp.exp(2j * z))
    return state, cache, params


def test_electrostatic_phi_reference_matches_production_field_solve() -> None:
    from spectraxgk.linear import LinearTerms, linear_rhs_cached

    state, cache, params = _small_periodic_field_problem()
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    _rhs, phi = linear_rhs_cached(state, cache, params, terms=terms, use_jit=False, use_custom_vjp=False)
    observed = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )

    assert float(jnp.linalg.norm(phi)) > 0.0
    np.testing.assert_allclose(np.asarray(observed), np.asarray(phi), rtol=2.0e-6, atol=2.0e-6)
    assert spectraxgk.electrostatic_phi_reference is electrostatic_phi_reference
    assert spectraxgk.electrostatic_phi_shard_map is electrostatic_phi_shard_map


def test_electrostatic_phi_shard_map_noops_to_reference_for_single_chunk() -> None:
    state, cache, params = _small_periodic_field_problem()
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    observed = electrostatic_phi_shard_map(
        state,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=[jax.devices()[0]],
    )
    expected = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6)


def test_electrostatic_phi_shard_map_matches_reference_when_logical_devices_available() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices; artifact generator sets logical CPU devices")
    state, cache, params = _small_periodic_field_problem()
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    observed = electrostatic_phi_shard_map(
        state,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=devices[:2],
    )
    expected = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6)


def test_electrostatic_phi_rejects_invalid_shapes_and_plans() -> None:
    state, cache, params = _small_periodic_field_problem()
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    bad_shape = state[:, :5, ...]
    uneven = replace(plan, state_shape=bad_shape.shape)
    laguerre_plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("laguerre",))

    with pytest.raises(NotImplementedError, match="single-species"):
        electrostatic_phi_reference(state[None, ...], Jl=cache.Jl, tau_e=params.tau_e)
    with pytest.raises(ValueError, match="Jl"):
        electrostatic_phi_reference(state, Jl=jnp.ones((state.shape[0], state.shape[2], state.shape[4])), tau_e=params.tau_e)
    with pytest.raises(NotImplementedError, match="single-species"):
        electrostatic_phi_shard_map(state[None, ...], plan, Jl=cache.Jl, tau_e=params.tau_e)
    with pytest.raises(ValueError, match="state shape"):
        electrostatic_phi_shard_map(bad_shape, plan, Jl=cache.Jl, tau_e=params.tau_e)
    with pytest.raises(NotImplementedError, match="active 'm'"):
        electrostatic_phi_shard_map(state, laguerre_plan, Jl=cache.Jl, tau_e=params.tau_e)
    with pytest.raises(ValueError, match="divide evenly"):
        electrostatic_phi_shard_map(bad_shape, uneven, Jl=cache.Jl, tau_e=params.tau_e, devices=[object(), object()])
    with pytest.raises(ValueError, match="Jl"):
        electrostatic_phi_shard_map(state, plan, Jl=jnp.ones((state.shape[0], state.shape[2], state.shape[4])), tau_e=params.tau_e)
    with pytest.raises(ValueError, match="not enough devices"):
        electrostatic_phi_shard_map(state, plan, Jl=cache.Jl, tau_e=params.tau_e, devices=[object()])


def test_mirror_and_curvature_gradb_drift_shard_maps_match_production_terms() -> None:
    from spectraxgk.linear import build_H
    from spectraxgk.terms.linear_terms import curvature_gradb_contribution, mirror_contribution

    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    H = build_H(state, cache.Jl, phi, jnp.asarray([params.tz]))
    plan = build_velocity_sharding_plan(H.shape, num_devices=1, axes=("hermite",))
    vth = jnp.asarray([params.vth], dtype=jnp.float32)
    tz = jnp.asarray([params.tz], dtype=jnp.float32)

    mirror_expected = mirror_contribution(
        H[None, ...],
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )[0]
    mirror_observed = mirror_drift_shard_map(
        H,
        plan,
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        devices=[jax.devices()[0]],
    )
    np.testing.assert_allclose(np.asarray(mirror_observed), np.asarray(mirror_expected), rtol=2.0e-6, atol=2.0e-6)

    drift_expected = curvature_gradb_contribution(
        H[None, ...],
        tz=tz,
        omega_d_scale=jnp.asarray(params.omega_d_scale, dtype=jnp.float32),
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        imag=jnp.asarray(1j, dtype=H.dtype),
        weight_curv=jnp.asarray(1.0, dtype=jnp.float32),
        weight_gradb=jnp.asarray(1.0, dtype=jnp.float32),
    )[0]
    drift_observed = curvature_gradb_drift_shard_map(
        H,
        plan,
        tz=tz,
        omega_d_scale=params.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        devices=[jax.devices()[0]],
    )
    np.testing.assert_allclose(np.asarray(drift_observed), np.asarray(drift_expected), rtol=2.0e-6, atol=2.0e-6)
    assert spectraxgk.mirror_drift_reference is mirror_drift_reference
    assert spectraxgk.mirror_drift_shard_map is mirror_drift_shard_map
    assert spectraxgk.curvature_gradb_drift_reference is curvature_gradb_drift_reference
    assert spectraxgk.curvature_gradb_drift_shard_map is curvature_gradb_drift_shard_map


def test_mirror_and_curvature_gradb_drift_shard_maps_match_reference_when_logical_devices_available() -> None:
    from spectraxgk.linear import build_H

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices; artifact generator sets logical CPU devices")
    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    H = build_H(state, cache.Jl, phi, jnp.asarray([params.tz]))
    plan = build_velocity_sharding_plan(H.shape, num_devices=2, axes=("hermite",))
    vth = jnp.asarray([params.vth], dtype=jnp.float32)
    tz = jnp.asarray([params.tz], dtype=jnp.float32)

    mirror_observed = mirror_drift_shard_map(
        H,
        plan,
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        devices=devices[:2],
    )
    mirror_expected = mirror_drift_reference(
        H,
        vth=vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
    )
    np.testing.assert_allclose(np.asarray(mirror_observed), np.asarray(mirror_expected), rtol=2.0e-6, atol=2.0e-6)

    drift_observed = curvature_gradb_drift_shard_map(
        H,
        plan,
        tz=tz,
        omega_d_scale=params.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        devices=devices[:2],
    )
    drift_expected = curvature_gradb_drift_reference(
        H,
        tz=tz,
        omega_d_scale=params.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
    )
    np.testing.assert_allclose(np.asarray(drift_observed), np.asarray(drift_expected), rtol=2.0e-6, atol=2.0e-6)


def test_six_dimensional_mirror_and_drift_paths_preserve_species_broadcasts() -> None:
    state = (jnp.arange(2 * 2 * 4 * 2 * 1 * 3, dtype=jnp.float32).reshape((2, 2, 4, 2, 1, 3)) + 1j).astype(
        jnp.complex64
    )
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))
    ell = jnp.arange(state.shape[1], dtype=jnp.float32).reshape((1, state.shape[1], 1, 1, 1, 1))
    m = jnp.arange(state.shape[2], dtype=jnp.float32).reshape((1, 1, state.shape[2], 1, 1, 1))
    sqrt_m = jnp.sqrt(m)
    sqrt_m_p1 = jnp.sqrt(m + 1.0)
    vth = jnp.asarray([1.0, 1.5], dtype=jnp.float32)
    tz = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    cv_d = jnp.ones((state.shape[3], state.shape[4], state.shape[5]), dtype=jnp.float32)
    gb_d = 0.25 * cv_d

    mirror_expected = mirror_drift_reference(
        state,
        vth=vth,
        bgrad=jnp.ones((state.shape[-1],), dtype=jnp.float32),
        ell=ell,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
    )
    mirror_observed = mirror_drift_shard_map(
        state,
        plan,
        vth=vth,
        bgrad=jnp.ones((state.shape[-1],), dtype=jnp.float32),
        ell=ell,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
        devices=[jax.devices()[0]],
    )
    np.testing.assert_allclose(np.asarray(mirror_observed), np.asarray(mirror_expected), rtol=2.0e-6, atol=2.0e-6)

    drift_expected = curvature_gradb_drift_reference(
        state,
        tz=tz,
        omega_d_scale=0.7,
        cv_d=cv_d,
        gb_d=gb_d,
        ell=ell,
        m=m,
    )
    drift_observed = curvature_gradb_drift_shard_map(
        state,
        plan,
        tz=tz,
        omega_d_scale=0.7,
        cv_d=cv_d,
        gb_d=gb_d,
        ell=ell,
        m=m,
        devices=[jax.devices()[0]],
    )
    assert drift_observed.shape == state.shape
    np.testing.assert_allclose(np.asarray(drift_observed), np.asarray(drift_expected), rtol=2.0e-6, atol=2.0e-6)


def test_diamagnetic_drive_shard_map_matches_production_term() -> None:
    from spectraxgk.linear import LinearTerms, linear_rhs_cached

    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    plan = build_velocity_sharding_plan(state.shape, num_devices=1, axes=("hermite",))

    observed = diamagnetic_drive_shard_map(
        state,
        plan,
        phi=phi,
        Jl=cache.Jl,
        l4=cache.l4,
        tprim=params.R_over_LTi,
        fprim=params.R_over_Ln,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
        devices=[jax.devices()[0]],
    )
    expected, _phi = linear_rhs_cached(
        state,
        cache,
        params,
        terms=LinearTerms(
            streaming=0.0,
            mirror=0.0,
            curvature=0.0,
            gradb=0.0,
            diamagnetic=1.0,
            collisions=0.0,
            hypercollisions=0.0,
            end_damping=0.0,
            apar=0.0,
            bpar=0.0,
        ),
        use_jit=False,
        use_custom_vjp=False,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6)
    assert spectraxgk.diamagnetic_drive_reference is diamagnetic_drive_reference
    assert spectraxgk.diamagnetic_drive_shard_map is diamagnetic_drive_shard_map


def test_diamagnetic_drive_shard_map_matches_reference_when_logical_devices_available() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices; artifact generator sets logical CPU devices")
    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))

    observed = diamagnetic_drive_shard_map(
        state,
        plan,
        phi=phi,
        Jl=cache.Jl,
        l4=cache.l4,
        tprim=params.R_over_LTi,
        fprim=params.R_over_Ln,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
        devices=devices[:2],
    )
    expected = diamagnetic_drive_reference(
        state,
        phi=phi,
        Jl=cache.Jl,
        l4=cache.l4,
        tprim=params.R_over_LTi,
        fprim=params.R_over_Ln,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6)


def test_diamagnetic_drive_rejects_invalid_shapes_and_plans() -> None:
    state, cache, params = _small_periodic_field_problem()
    phi = electrostatic_phi_reference(
        state,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
    )
    plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("hermite",))
    bad_shape = state[:, :5, ...]
    uneven = replace(plan, state_shape=bad_shape.shape)
    laguerre_plan = build_velocity_sharding_plan(state.shape, num_devices=2, axes=("laguerre",))
    kwargs = dict(
        phi=phi,
        Jl=cache.Jl,
        l4=cache.l4,
        tprim=params.R_over_LTi,
        fprim=params.R_over_Ln,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
    )

    with pytest.raises(NotImplementedError, match="single-species"):
        diamagnetic_drive_reference(state[None, ...], **kwargs)
    with pytest.raises(ValueError, match="Jl"):
        diamagnetic_drive_reference(state, **{**kwargs, "Jl": jnp.ones((state.shape[0], state.shape[2], state.shape[4]))})
    jl4 = cache.Jl[0] if cache.Jl.ndim == 5 else cache.Jl
    assert diamagnetic_drive_reference(state, **{**kwargs, "Jl": jl4[None, ...]}).shape == state.shape
    with pytest.raises(NotImplementedError, match="single-species"):
        diamagnetic_drive_shard_map(state[None, ...], plan, **kwargs)
    with pytest.raises(ValueError, match="state shape"):
        diamagnetic_drive_shard_map(bad_shape, plan, **kwargs)
    with pytest.raises(NotImplementedError, match="active 'm'"):
        diamagnetic_drive_shard_map(state, laguerre_plan, **kwargs)
    with pytest.raises(ValueError, match="divide evenly"):
        diamagnetic_drive_shard_map(bad_shape, uneven, devices=[object(), object()], **kwargs)
    with pytest.raises(ValueError, match="not enough devices"):
        diamagnetic_drive_shard_map(state, plan, devices=[object()], **kwargs)


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


def test_linear_rhs_parallel_cached_streaming_only_matches_serial_call_graph() -> None:
    from spectraxgk.config import CycloneBaseCase, GridConfig
    from spectraxgk.geometry import SAlphaGeometry
    from spectraxgk.grids import build_spectral_grid
    from spectraxgk.linear import (
        LinearParams,
        LinearTerms,
        build_linear_cache,
        linear_rhs_cached,
        linear_rhs_parallel_cached,
    )
    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic"))
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros((2, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    state = state.at[0, 2, min(1, grid.ky.size - 1), 0, :].set(jnp.exp(1j * z))
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    parallel = RuntimeParallelConfig(strategy="velocity", backend="streaming_only", axis="hermite", num_devices=1)

    serial, phi_serial = linear_rhs_cached(state, cache, params, terms=terms, use_jit=False)
    sharded, phi_sharded = linear_rhs_parallel_cached(state, cache, params, terms=terms, parallel=parallel)

    np.testing.assert_allclose(np.asarray(sharded), np.asarray(serial), rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(phi_sharded), np.asarray(phi_serial), rtol=0.0, atol=0.0)
    assert spectraxgk.linear_rhs_parallel_cached is linear_rhs_parallel_cached
    assert spectraxgk.linear_rhs_streaming_velocity_sharded(state, cache, params, num_devices=1)[0].shape == state.shape


def test_linear_rhs_parallel_cached_rejects_non_streaming_velocity_route() -> None:
    from spectraxgk.linear import LinearParams, LinearTerms, linear_rhs_parallel_cached
    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    class Cache:
        kz = jnp.asarray([0.0, 1.0, -1.0])

    state = jnp.zeros((1, 4, 1, 1, 3), dtype=jnp.complex64)
    parallel = RuntimeParallelConfig(strategy="velocity", backend="streaming_only", axis="hermite", num_devices=1)

    with pytest.raises(NotImplementedError, match="streaming-only LinearTerms"):
        linear_rhs_parallel_cached(state, Cache(), LinearParams(), terms=LinearTerms(), parallel=parallel)


def test_linear_rhs_parallel_cached_electrostatic_streaming_matches_serial_call_graph() -> None:
    from spectraxgk.config import CycloneBaseCase, GridConfig
    from spectraxgk.geometry import SAlphaGeometry
    from spectraxgk.grids import build_spectral_grid
    from spectraxgk.linear import (
        LinearParams,
        LinearTerms,
        build_linear_cache,
        linear_rhs_cached,
        linear_rhs_parallel_cached,
    )
    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic"))
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(beta=0.0, fapar=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros((2, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    state = state.at[0, 0, min(1, grid.ky.size - 1), 0, :].set(0.2 * jnp.exp(1j * z))
    state = state.at[1, 2, min(1, grid.ky.size - 1), 0, :].set(0.1 * jnp.exp(2j * z))
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    parallel = RuntimeParallelConfig(
        strategy="velocity",
        backend="streaming_electrostatic",
        axis="hermite",
        num_devices=1,
    )

    serial, phi_serial = linear_rhs_cached(state, cache, params, terms=terms, use_jit=False, use_custom_vjp=False)
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel,
        use_custom_vjp=False,
    )

    assert float(jnp.linalg.norm(phi_serial)) > 0.0
    np.testing.assert_allclose(np.asarray(phi_sharded), np.asarray(phi_serial), rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(sharded), np.asarray(serial), rtol=2.0e-6, atol=2.0e-6)
    assert spectraxgk.linear_rhs_streaming_electrostatic_velocity_sharded(state, cache, params, num_devices=1)[0].shape == state.shape


def test_linear_rhs_parallel_cached_electrostatic_linear_slices_match_serial_call_graph() -> None:
    from spectraxgk.linear import LinearTerms, linear_rhs_cached, linear_rhs_parallel_cached
    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    state, cache, params = _small_periodic_field_problem()
    z = jnp.linspace(0.0, 2.0 * jnp.pi, state.shape[-1], endpoint=False)
    state = state.at[0, 2, min(1, state.shape[2] - 1), 0, :].set(0.07 * jnp.exp(2j * z))
    state = state.at[1, 3, min(1, state.shape[2] - 1), 0, :].set(0.03 * jnp.exp(3j * z))
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    parallel = RuntimeParallelConfig(
        strategy="velocity",
        backend="electrostatic_linear_slices",
        axis="hermite",
        num_devices=1,
    )

    serial, phi_serial = linear_rhs_cached(state, cache, params, terms=terms, use_jit=False, use_custom_vjp=False)
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel,
        use_custom_vjp=False,
    )

    np.testing.assert_allclose(np.asarray(phi_sharded), np.asarray(phi_serial), rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(sharded), np.asarray(serial), rtol=2.0e-6, atol=2.0e-6)
    assert spectraxgk.linear_rhs_electrostatic_slices_velocity_sharded(state, cache, params, terms=terms, num_devices=1)[0].shape == state.shape


def test_linear_rhs_parallel_cached_auto_backend_selects_gated_electrostatic_slices() -> None:
    from spectraxgk.linear import LinearTerms, linear_rhs_cached, linear_rhs_parallel_cached
    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    state, cache, params = _small_periodic_field_problem()
    z = jnp.linspace(0.0, 2.0 * jnp.pi, state.shape[-1], endpoint=False)
    state = state.at[0, 2, min(1, state.shape[2] - 1), 0, :].set(0.07 * jnp.exp(2j * z))
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    parallel = RuntimeParallelConfig(strategy="velocity", backend="auto", axis="hermite", num_devices=1)

    serial, phi_serial = linear_rhs_cached(state, cache, params, terms=terms, use_jit=False, use_custom_vjp=False)
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel,
        use_custom_vjp=False,
    )

    np.testing.assert_allclose(np.asarray(phi_sharded), np.asarray(phi_serial), rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(sharded), np.asarray(serial), rtol=2.0e-6, atol=2.0e-6)


def test_linear_rhs_parallel_cached_electrostatic_linear_slices_rejects_ungated_terms() -> None:
    from spectraxgk.linear import LinearParams, LinearTerms, linear_rhs_parallel_cached
    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    class Cache:
        use_twist_shift = False

    state = jnp.zeros((1, 4, 1, 1, 3), dtype=jnp.complex64)
    parallel = RuntimeParallelConfig(strategy="velocity", backend="electrostatic_linear_slices", axis="hermite", num_devices=1)

    with pytest.raises(NotImplementedError, match="collision/EM"):
        linear_rhs_parallel_cached(
            state,
            Cache(),
            LinearParams(),
            terms=LinearTerms(collisions=1.0),
            parallel=parallel,
        )

    auto_parallel = RuntimeParallelConfig(strategy="velocity", backend="auto", axis="hermite", num_devices=1)
    with pytest.raises(NotImplementedError, match="backend='auto'"):
        linear_rhs_parallel_cached(
            state,
            Cache(),
            LinearParams(),
            terms=LinearTerms(collisions=1.0),
            parallel=auto_parallel,
        )
