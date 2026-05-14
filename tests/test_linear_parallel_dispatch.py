from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from spectraxgk import linear_parallel
from spectraxgk.linear_params import LinearParams, LinearTerms


def _state() -> jnp.ndarray:
    return jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)


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
