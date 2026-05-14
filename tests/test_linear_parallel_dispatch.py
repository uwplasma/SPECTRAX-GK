from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from spectraxgk import linear_parallel
from spectraxgk.linear_params import LinearParams, LinearTerms


def _state() -> jnp.ndarray:
    return jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)


def _richer_state() -> jnp.ndarray:
    values = jnp.arange(2 * 3 * 2 * 1 * 4, dtype=jnp.float32).reshape((2, 3, 2, 1, 4))
    return values.astype(jnp.complex64)


def _cache_for_richer_state(**overrides):
    cache = SimpleNamespace(
        use_twist_shift=False,
        kz=jnp.asarray([0.0, 1.0, -1.0, 2.0], dtype=jnp.float32),
        Jl=jnp.ones((2, 2, 1, 4), dtype=jnp.float32),
        mask0=None,
        bgrad=jnp.ones((4,), dtype=jnp.float32),
        l=jnp.arange(2, dtype=jnp.float32),
        l4=jnp.arange(2, dtype=jnp.float32).reshape((2, 1, 1, 1)),
        m=jnp.arange(3, dtype=jnp.float32),
        sqrt_m=jnp.sqrt(jnp.arange(3, dtype=jnp.float32)),
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 4, dtype=jnp.float32)),
        cv_d=jnp.ones((2, 1, 4), dtype=jnp.float32),
        gb_d=jnp.ones((2, 1, 4), dtype=jnp.float32),
        ky=jnp.asarray([0.2, 0.5], dtype=jnp.float32),
    )
    for key, value in overrides.items():
        setattr(cache, key, value)
    return cache


def _sentinel(name: str):
    return jnp.asarray([len(name)], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)


def _streaming_only_terms() -> LinearTerms:
    return LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


def _electrostatic_slice_terms() -> LinearTerms:
    return LinearTerms(
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


def test_linear_parallel_term_classifiers_cover_release_routes() -> None:
    assert linear_parallel._is_streaming_only_terms(_streaming_only_terms()) is True
    assert linear_parallel._is_streaming_only_terms(_electrostatic_slice_terms()) is False
    assert linear_parallel._is_electrostatic_slice_terms(_electrostatic_slice_terms()) is True
    assert linear_parallel._is_electrostatic_slice_terms(LinearTerms(collisions=1.0)) is False
    assert linear_parallel._is_electrostatic_field_terms(LinearTerms(apar=0.0, bpar=0.0)) is True
    assert linear_parallel._is_electrostatic_field_terms(LinearTerms(apar=1.0, bpar=0.0)) is False


def test_resolve_parallel_devices_validates_explicit_and_requested_counts(monkeypatch) -> None:
    devices = [object(), object()]

    assert linear_parallel._resolve_parallel_devices(devices=devices) == devices
    assert linear_parallel._resolve_parallel_devices(devices=devices, num_devices=2) == devices
    with pytest.raises(ValueError, match="num_devices"):
        linear_parallel._resolve_parallel_devices(devices=devices, num_devices=1)
    with pytest.raises(ValueError, match="num_devices"):
        linear_parallel._resolve_parallel_devices(num_devices=0)
    monkeypatch.setattr(linear_parallel.jax, "devices", lambda: [object()])
    with pytest.raises(ValueError, match="only"):
        linear_parallel._resolve_parallel_devices(num_devices=2)
    with pytest.raises(ValueError, match="at least one"):
        linear_parallel._resolve_parallel_devices(devices=[])


def test_streaming_velocity_sharded_route_validates_shape_and_returns_zero_phi(monkeypatch) -> None:
    import spectraxgk.velocity_sharding as velocity_sharding

    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_streaming(arr, plan, **kwargs):
        calls.append((tuple(arr.shape), int(plan.num_devices)))
        assert kwargs["kz"].shape == (4,)
        assert kwargs["vth"] == 1.0
        assert len(kwargs["devices"]) == 1
        return jnp.ones_like(arr) * (2.0 + 0.0j)

    monkeypatch.setattr(velocity_sharding, "periodic_streaming_shard_map", fake_streaming)

    with pytest.raises(ValueError, match="G must have shape"):
        linear_parallel.linear_rhs_streaming_velocity_sharded(
            jnp.zeros((2, 3, 4), dtype=jnp.complex64),
            _cache_for_richer_state(),
            LinearParams(),
            devices=[object()],
        )

    dG, phi = linear_parallel.linear_rhs_streaming_velocity_sharded(
        _richer_state(),
        _cache_for_richer_state(),
        LinearParams(),
        devices=[object()],
    )

    assert calls == [((2, 3, 2, 1, 4), 1)]
    assert dG.shape == _richer_state().shape
    assert jnp.all(dG == -(2.0 + 0.0j))
    assert phi.shape == (2, 1, 4)
    assert jnp.all(phi == 0.0)


def test_electrostatic_streaming_field_rhs_and_sharded_phi_path(monkeypatch) -> None:
    import spectraxgk.terms.operators as operators
    import spectraxgk.velocity_sharding as velocity_sharding

    arr = _richer_state()
    cache = _cache_for_richer_state()
    phi = jnp.ones(arr.shape[-3:], dtype=arr.dtype) * (1.5 + 0.0j)

    field_rhs = linear_parallel._electrostatic_streaming_field_rhs(
        arr[None, ...],
        phi=phi,
        Jl=cache.Jl,
        tz=jnp.asarray([2.0], dtype=jnp.float32),
        vth=jnp.asarray([3.0], dtype=jnp.float32),
    )
    assert field_rhs.shape == (1,) + arr.shape
    assert jnp.all(field_rhs[:, :, 0] == 0.0)
    assert jnp.all(field_rhs[:, :, 2] == 0.0)
    assert jnp.allclose(field_rhs[:, :, 1], -2.25 + 0.0j)

    def fake_streaming(local_arr, plan, **kwargs):
        assert tuple(local_arr.shape) == tuple(arr.shape)
        assert int(plan.num_devices) == 1
        assert len(kwargs["devices"]) == 1
        return jnp.ones_like(local_arr) * (4.0 + 0.0j)

    def fake_grad_z(value, *, kz):
        assert kz.shape == (4,)
        return jnp.ones_like(value) * (0.25 + 0.0j)

    monkeypatch.setattr(velocity_sharding, "periodic_streaming_shard_map", fake_streaming)
    monkeypatch.setattr(operators, "grad_z_periodic", fake_grad_z)

    from spectraxgk.velocity_sharding import build_velocity_sharding_plan

    out = linear_parallel._streaming_electrostatic_from_phi_velocity_sharded(
        arr,
        cache,
        LinearParams(kpar_scale=2.0, tz=2.0, vth=3.0),
        phi=phi,
        plan=build_velocity_sharding_plan(arr.shape, num_devices=1, axes=("hermite",)),
        devices=[object()],
    )

    assert out.shape == arr.shape
    assert jnp.allclose(out, -(4.0 + 0.0j) + 0.5)


def test_streaming_electrostatic_velocity_sharded_fail_closed_and_uses_phi(monkeypatch) -> None:
    import spectraxgk.velocity_sharding as velocity_sharding

    arr = _richer_state()
    cache = _cache_for_richer_state()
    phi_expected = jnp.ones(arr.shape[-3:], dtype=arr.dtype) * (0.75 + 0.0j)

    with pytest.raises(NotImplementedError, match="single-species 5D"):
        linear_parallel.linear_rhs_streaming_electrostatic_velocity_sharded(
            arr[None, ...],
            cache,
            LinearParams(),
            devices=[object()],
        )
    with pytest.raises(NotImplementedError, match="periodic z grid"):
        linear_parallel.linear_rhs_streaming_electrostatic_velocity_sharded(
            arr,
            _cache_for_richer_state(use_twist_shift=True),
            LinearParams(),
            devices=[object()],
        )

    def fake_phi(local_arr, plan, **kwargs):
        assert tuple(local_arr.shape) == tuple(arr.shape)
        assert kwargs["tau_e"] == 1.0
        assert len(kwargs["devices"]) == 1
        return phi_expected

    def fake_rhs(local_arr, local_cache, local_params, *, phi, plan, devices):
        assert local_arr is not arr or tuple(local_arr.shape) == tuple(arr.shape)
        assert jnp.all(phi == phi_expected)
        assert int(plan.num_devices) == 1
        assert len(devices) == 1
        return jnp.ones_like(local_arr) * (6.0 + 0.0j)

    monkeypatch.setattr(velocity_sharding, "electrostatic_phi_shard_map", fake_phi)
    monkeypatch.setattr(linear_parallel, "_streaming_electrostatic_from_phi_velocity_sharded", fake_rhs)

    dG, phi = linear_parallel.linear_rhs_streaming_electrostatic_velocity_sharded(
        arr,
        cache,
        LinearParams(),
        devices=[object()],
    )

    assert jnp.all(dG == 6.0 + 0.0j)
    assert jnp.all(phi == phi_expected)


def test_electrostatic_slices_velocity_sharded_fail_closed_and_weighted_routes(monkeypatch) -> None:
    import spectraxgk.velocity_sharding as velocity_sharding

    arr = _richer_state()
    cache = _cache_for_richer_state()
    phi_expected = jnp.ones(arr.shape[-3:], dtype=arr.dtype) * (0.25 + 0.0j)
    calls: list[str] = []

    with pytest.raises(NotImplementedError, match="allows only electrostatic"):
        linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded(
            arr,
            cache,
            LinearParams(),
            LinearTerms(collisions=1.0),
            devices=[object()],
        )
    with pytest.raises(NotImplementedError, match="single-species 5D"):
        linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded(
            arr[None, ...],
            cache,
            LinearParams(),
            _electrostatic_slice_terms(),
            devices=[object()],
        )
    with pytest.raises(NotImplementedError, match="periodic z grid"):
        linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded(
            arr,
            _cache_for_richer_state(use_twist_shift=True),
            LinearParams(),
            _electrostatic_slice_terms(),
            devices=[object()],
        )

    def fake_phi(*args, **kwargs):
        calls.append("phi")
        return phi_expected

    def fake_streaming(local_arr, *_args, **_kwargs):
        calls.append("streaming")
        return jnp.ones_like(local_arr) * (1.0 + 0.0j)

    def fake_build_h(local_arr, *_args, **_kwargs):
        calls.append("H")
        return local_arr + (0.5 + 0.0j)

    def fake_mirror(local_h, *_args, **kwargs):
        calls.append("mirror")
        assert float(kwargs["weight"]) == pytest.approx(0.25)
        return jnp.ones_like(local_h) * (2.0 * kwargs["weight"] + 0.0j)

    def fake_curv(local_h, *_args, **kwargs):
        calls.append("curv")
        assert float(kwargs["weight_curv"]) == pytest.approx(0.5)
        assert float(kwargs["weight_gradb"]) == pytest.approx(0.0)
        return jnp.ones_like(local_h) * (3.0 * kwargs["weight_curv"] + 0.0j)

    def fake_diamag(local_arr, *_args, **kwargs):
        calls.append("diamag")
        assert float(kwargs["weight"]) == pytest.approx(0.125)
        return jnp.ones_like(local_arr) * (4.0 * kwargs["weight"] + 0.0j)

    monkeypatch.setattr(velocity_sharding, "electrostatic_phi_shard_map", fake_phi)
    monkeypatch.setattr(
        linear_parallel,
        "_streaming_electrostatic_from_phi_velocity_sharded",
        fake_streaming,
    )
    monkeypatch.setattr(linear_parallel, "build_H", fake_build_h)
    monkeypatch.setattr(velocity_sharding, "mirror_drift_shard_map", fake_mirror)
    monkeypatch.setattr(velocity_sharding, "curvature_gradb_drift_shard_map", fake_curv)
    monkeypatch.setattr(velocity_sharding, "diamagnetic_drive_shard_map", fake_diamag)

    terms = LinearTerms(
        streaming=0.5,
        mirror=0.25,
        curvature=0.5,
        gradb=0.0,
        diamagnetic=0.125,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    dG, phi = linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded(
        arr,
        cache,
        LinearParams(),
        terms,
        devices=[object()],
    )

    assert calls == ["phi", "streaming", "H", "mirror", "curv", "diamag"]
    assert jnp.all(phi == phi_expected)
    assert jnp.allclose(dG, 0.5 * 1.0 + 0.25 * 2.0 + 0.5 * 3.0 + 0.125 * 4.0)


def test_linear_rhs_parallel_cached_serial_dispatch(monkeypatch) -> None:
    import spectraxgk.linear as linear_compat

    calls: list[str] = []

    def fake_serial(*args, **kwargs):
        calls.append("serial")
        assert kwargs["use_jit"] is False
        assert kwargs["use_custom_vjp"] is False
        assert kwargs["dt"] == 0.125
        return _sentinel("serial")

    monkeypatch.setattr(linear_compat, "linear_rhs_cached", fake_serial)

    out = linear_parallel.linear_rhs_parallel_cached(
        _state(),
        SimpleNamespace(),
        LinearParams(),
        parallel=None,
        use_jit=False,
        use_custom_vjp=False,
        dt=0.125,
    )

    assert calls == ["serial"]
    assert int(out[0][0]) == len("serial")


def test_linear_rhs_parallel_cached_velocity_auto_selects_electrostatic_slice(monkeypatch) -> None:
    import spectraxgk.linear as linear_compat

    calls: list[tuple[str, int | None]] = []

    def fake_slice(*args, **kwargs):
        calls.append(("slice", kwargs["num_devices"]))
        assert kwargs["terms"] == _electrostatic_slice_terms()
        return _sentinel("slice")

    monkeypatch.setattr(linear_compat, "linear_rhs_electrostatic_slices_velocity_sharded", fake_slice)
    parallel = SimpleNamespace(strategy="velocity", backend="auto", axis="hermite", num_devices=3)

    out = linear_parallel.linear_rhs_parallel_cached(
        _state(),
        SimpleNamespace(),
        LinearParams(),
        terms=_electrostatic_slice_terms(),
        parallel=parallel,
    )

    assert calls == [("slice", 3)]
    assert int(out[0][0]) == len("slice")


def test_linear_rhs_parallel_cached_explicit_streaming_routes(monkeypatch) -> None:
    import spectraxgk.linear as linear_compat

    calls: list[str] = []

    def fake_streaming(*args, **kwargs):
        calls.append("streaming")
        assert kwargs["num_devices"] == 2
        return _sentinel("streaming")

    def fake_electrostatic(*args, **kwargs):
        calls.append("electrostatic")
        assert kwargs["num_devices"] == 2
        assert kwargs["use_custom_vjp"] is False
        return _sentinel("electrostatic")

    monkeypatch.setattr(linear_compat, "linear_rhs_streaming_velocity_sharded", fake_streaming)
    monkeypatch.setattr(
        linear_compat,
        "linear_rhs_streaming_electrostatic_velocity_sharded",
        fake_electrostatic,
    )
    terms = _streaming_only_terms()

    linear_parallel.linear_rhs_parallel_cached(
        _state(),
        SimpleNamespace(),
        LinearParams(),
        terms=terms,
        parallel=SimpleNamespace(strategy="velocity", backend="streaming_only", axis="m", num_devices=2),
    )
    linear_parallel.linear_rhs_parallel_cached(
        _state(),
        SimpleNamespace(),
        LinearParams(),
        terms=terms,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="streaming_electrostatic",
            axis="hermite",
            num_devices=2,
        ),
        use_custom_vjp=False,
    )

    assert calls == ["streaming", "electrostatic"]


def test_linear_rhs_parallel_cached_rejects_unsupported_velocity_requests() -> None:
    g = _state()
    cache = SimpleNamespace()
    params = LinearParams()

    with pytest.raises(NotImplementedError, match="Hermite axis"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=_electrostatic_slice_terms(),
            parallel=SimpleNamespace(strategy="velocity", backend="auto", axis="kx"),
        )
    with pytest.raises(NotImplementedError, match="backend='auto'"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=LinearTerms(apar=1.0),
            parallel=SimpleNamespace(strategy="velocity", backend="auto", axis="hermite"),
        )
    with pytest.raises(NotImplementedError, match="streaming-only"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=_streaming_only_terms(),
            parallel=SimpleNamespace(strategy="velocity", backend="streaming_only", axis="ky"),
        )
    with pytest.raises(NotImplementedError, match="requires streaming-only"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=_electrostatic_slice_terms(),
            parallel=SimpleNamespace(strategy="velocity", backend="streaming_only", axis="hermite"),
        )
    with pytest.raises(NotImplementedError, match="collision/EM"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=LinearTerms(apar=1.0),
            parallel=SimpleNamespace(
                strategy="velocity",
                backend="electrostatic_linear_slices",
                axis="hermite",
            ),
        )
    with pytest.raises(NotImplementedError, match="strategy='velocity'"):
        linear_parallel.linear_rhs_parallel_cached(
            g,
            cache,
            params,
            terms=_electrostatic_slice_terms(),
            parallel=SimpleNamespace(strategy="domain", backend="unknown", axis="hermite"),
        )
