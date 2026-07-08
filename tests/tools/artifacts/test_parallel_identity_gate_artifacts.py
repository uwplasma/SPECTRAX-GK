from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spectraxgk.benchmarks import CycloneScanResult
from spectraxgk.workflows.runtime.results import RuntimeLinearScanResult
from tools.artifacts import generate_electrostatic_diamagnetic_gate as diamagnetic_gate
from tools.artifacts import generate_electrostatic_drift_gate as drift_gate
from tools.artifacts import (
    generate_electrostatic_field_reduce_gate as field_reduce_gate,
)
from tools.artifacts import (
    generate_hermite_streaming_ladder_gate as hermite_ladder_gate,
)
from tools.artifacts import generate_linear_rhs_electrostatic_slices_gate as slices_gate
from tools.artifacts import (
    generate_linear_rhs_streaming_electrostatic_gate as electrostatic_gate,
)
from tools.artifacts import generate_linear_rhs_streaming_gate as streaming_gate
from tools.artifacts import generate_parallel_identity_gate as parallel_identity_gate
from tools.artifacts import (
    generate_periodic_streaming_microkernel_gate as periodic_gate,
)
from tools.artifacts import generate_velocity_parallel_gates as velocity_parallel_gates

hermite_exchange_gate = velocity_parallel_gates
velocity_reduce_gate = velocity_parallel_gates


class _VelocityPlan:
    def __init__(self, shape: tuple[int, ...], pattern: str) -> None:
        self.shape = shape
        self.pattern = pattern

    def to_dict(self) -> dict[str, object]:
        return {
            "state_shape": self.shape,
            "chunks": {"m": 2},
            "active_axes": ("m",),
            "communication_pattern": self.pattern,
        }


class _Grid:
    ky = np.asarray([0.0, 0.3])
    z = np.asarray([0.0, 1.0, 2.0, 3.0])


def _fake_devices(_kind: str | None = None) -> list[object]:
    return [object(), object()]


def _assert_standard_artifacts(
    paths: dict[str, str] | None, out_prefix: Path, csv_token: str
) -> None:
    json_path = Path(paths["json"]) if paths else out_prefix.with_suffix(".json")
    csv_path = Path(paths["csv"]) if paths else out_prefix.with_suffix(".csv")
    png_path = Path(paths["png"]) if paths else out_prefix.with_suffix(".png")
    pdf_path = Path(paths["pdf"]) if paths else out_prefix.with_suffix(".pdf")

    assert json.loads(json_path.read_text(encoding="utf-8"))["identity_passed"] is True
    assert csv_token in csv_path.read_text(encoding="utf-8")
    assert png_path.exists()
    assert pdf_path.exists()


def _runtime_scan(ky_values: np.ndarray, *, workers: int) -> RuntimeLinearScanResult:
    gamma = np.asarray(ky_values, dtype=float) + 1.0
    omega = -(np.asarray(ky_values, dtype=float) + 2.0)
    quasilinear = tuple(
        {
            "ky": float(ky),
            "gamma": float(gamma_i),
            "omega": float(omega_i),
            "kperp_eff2": 0.5 + float(ky),
            "heat_flux_weight_total": 2.0 * float(ky),
            "particle_flux_weight_total": 0.1 * float(ky),
            "amplitude2": 0.3 * float(ky),
            "saturated_heat_flux_total": 0.6 * float(ky),
            "saturated_particle_flux_total": 0.03 * float(ky),
        }
        for ky, gamma_i, omega_i in zip(ky_values, gamma, omega, strict=True)
    )
    return RuntimeLinearScanResult(
        ky=np.asarray(ky_values, dtype=float),
        gamma=gamma,
        omega=omega,
        quasilinear=quasilinear,
        parallel={
            "requested_workers": int(workers),
            "effective_workers": min(int(workers), len(ky_values)),
            "executor": "thread",
            "identity_contract": "test",
            "quasilinear_state_extraction": True,
        },
    )


def test_velocity_field_reduce_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 2, 1, 1)
        return _VelocityPlan(shape, "field_reduce_broadcast")

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.arange(8, dtype=jnp.float32).reshape((1, 4, 2, 1, 1))

    def fake_reduce(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sum(state, axis=1)

    monkeypatch.setattr(velocity_reduce_gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_reference", fake_reduce
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_shard_map", fake_reduce
    )

    summary = velocity_reduce_gate.build_velocity_field_reduce_gate(
        shape=(1, 4, 2, 1, 1), requested_devices=2, atol=1.0e-12, rtol=1.0e-10
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["rtol"] == 1.0e-10
    assert summary["max_allowed_error"] > summary["atol"]
    assert summary["max_rel_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_hermite_exchange_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 1, 1, 1)
        return _VelocityPlan(shape, "hermite_ghost_exchange")

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]])

    def fake_reference(state):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[0.0]]], [[[1.0]]], [[[2.0]]], [[[3.0]]]]]), jnp.asarray(
            [[[[[2.0]]], [[[3.0]]], [[[4.0]]], [[[0.0]]]]]
        )

    monkeypatch.setattr(hermite_exchange_gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_neighbor_reference", fake_reference
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_neighbor_shard_map",
        lambda state, plan, devices: fake_reference(state),
    )

    summary = hermite_exchange_gate.build_hermite_exchange_gate(
        shape=(1, 4, 1, 1, 1), requested_devices=2, atol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["plan"]["communication_pattern"] == "hermite_ghost_exchange"
    assert summary["max_lower_abs_error"] == 0.0
    assert summary["max_upper_abs_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_hermite_streaming_ladder_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 1, 1, 1)
        return _VelocityPlan(shape, "hermite_ghost_exchange+field_reduce_broadcast")

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]])

    def fake_ladder(state, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sqrt(state + 1.0)

    def fake_reduce(state, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sum(state, axis=1)

    monkeypatch.setattr(hermite_ladder_gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_streaming_ladder_reference", fake_ladder
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_streaming_ladder_shard_map",
        lambda state, plan, **kwargs: fake_ladder(state),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_reference", fake_reduce
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_shard_map",
        lambda state, plan, **kwargs: fake_reduce(state),
    )

    summary = hermite_ladder_gate.build_hermite_streaming_ladder_gate(
        shape=(1, 4, 1, 1, 1),
        requested_devices=2,
        vth=1.7,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_ladder_abs_error"] == 0.0
    assert summary["max_ladder_rel_error"] == 0.0
    assert summary["max_reduction_abs_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_periodic_streaming_microkernel_gate_builds_identity_summary(
    monkeypatch,
) -> None:
    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 1, 4, 1, 1, 4)
        return _VelocityPlan(shape, "hermite_ghost_exchange+field_reduce_broadcast")

    def fake_state(shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones(shape, dtype=jnp.complex64), jnp.linspace(0.0, 1.0, shape[-1])

    def fake_streaming(state, **_kwargs):  # type: ignore[no-untyped-def]
        return 2.0 * state

    monkeypatch.setattr(periodic_gate, "_state", fake_state)
    monkeypatch.setattr(periodic_gate, "_production_streaming_term", fake_streaming)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.periodic_streaming_reference", fake_streaming
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.periodic_streaming_shard_map",
        lambda state, plan, **kwargs: fake_streaming(state),
    )

    summary = periodic_gate.build_periodic_streaming_microkernel_gate(
        shape=(1, 1, 4, 1, 1, 4),
        requested_devices=2,
        vth=1.7,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_sharded_abs_error"] == 0.0
    assert summary["max_sharded_rel_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_electrostatic_field_reduce_gate_builds_identity_summary(monkeypatch) -> None:
    class FakeCache:
        Jl = 1.0
        mask0 = False

    class FakeParams:
        tau_e = 1.0
        charge_sign = 1.0
        density = 1.0
        tz = 1.0

    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            FakeCache(),
            FakeParams(),
            _Grid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return state * 0.0, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    def fake_phi(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(field_reduce_gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.electrostatic_phi_shard_map", fake_phi
    )

    summary = field_reduce_gate.build_electrostatic_field_reduce_gate(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert len(summary["rows"]) == 4


def test_electrostatic_diamagnetic_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        class Cache:
            Jl = jnp.ones((1, 1, 2, 1, 4), dtype=jnp.float32)
            mask0 = jnp.zeros((2, 1, 4), dtype=bool)
            l4 = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
            ky = jnp.asarray([0.0, 0.3], dtype=jnp.float32)

        class Params:
            tau_e = 1.0
            charge_sign = 1.0
            density = 1.0
            tz = 1.0
            R_over_LTi = 6.9
            R_over_Ln = 2.2
            omega_star_scale = 1.0

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            Cache(),
            Params(),
            _Grid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return 2.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    def fake_phi(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(diamagnetic_gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.electrostatic_phi_shard_map", fake_phi
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.diamagnetic_drive_shard_map",
        lambda state, *_args, **_kwargs: 2.0 * state,
    )

    summary = diamagnetic_gate.build_electrostatic_diamagnetic_gate(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["max_phi_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert len(summary["rows"]) == 4


def test_electrostatic_drift_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        class Cache:
            Jl = jnp.ones((1, 1, 2, 1, 4), dtype=jnp.float32)
            mask0 = jnp.zeros((2, 1, 4), dtype=bool)
            bgrad = jnp.ones((4,), dtype=jnp.float32)
            m = jnp.ones((1, 4, 1, 1, 1), dtype=jnp.float32)
            sqrt_m = jnp.ones((1, 4, 1, 1, 1), dtype=jnp.float32)
            sqrt_m_p1 = jnp.ones((1, 4, 1, 1, 1), dtype=jnp.float32)
            cv_d = jnp.ones((2, 1, 4), dtype=jnp.float32)
            gb_d = jnp.ones((2, 1, 4), dtype=jnp.float32)

        setattr(Cache, "l", jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32))

        class Params:
            tau_e = 1.0
            charge_sign = 1.0
            density = 1.0
            tz = 1.0
            vth = 1.0
            omega_d_scale = 1.0

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            Cache(),
            Params(),
            _Grid(),
        )

    def fake_rhs(state, *_args, **kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        terms = kwargs.get("terms")
        scale = (
            float(getattr(terms, "mirror", 0.0))
            + float(getattr(terms, "curvature", 0.0))
            + float(getattr(terms, "gradb", 0.0))
        )
        return scale * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    def fake_phi(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(drift_gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.linear.build_H", lambda state, *_args, **_kwargs: state
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.electrostatic_phi_shard_map", fake_phi
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.mirror_drift_shard_map",
        lambda state, *_args, **_kwargs: state,
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.curvature_gradb_drift_shard_map",
        lambda state, *_args, **_kwargs: 2.0 * state,
    )

    summary = drift_gate.build_electrostatic_drift_gate(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert {row["component"] for row in summary["rows"]} == {
        "mirror",
        "curvature_gradb",
        "total",
    }


def test_linear_rhs_streaming_gate_builds_identity_summary(monkeypatch) -> None:
    class FakeCache:
        kz = [0.0, 1.0, -2.0, -1.0]

    class FakeParams:
        vth = 1.0

    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            FakeCache(),
            FakeParams(),
            _Grid(),
            object(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return -2.0 * state, jnp.zeros((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(streaming_gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.periodic_streaming_shard_map",
        lambda state, *_args, **_kwargs: 2.0 * state,
    )

    summary = streaming_gate.build_linear_rhs_streaming_gate(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["phi_norm"] == 0.0
    assert len(summary["rows"]) == 4


@pytest.mark.parametrize(
    ("gate", "builder"),
    [
        (electrostatic_gate, "build_linear_rhs_streaming_electrostatic_gate"),
        (slices_gate, "build_linear_rhs_electrostatic_slices_gate"),
    ],
    ids=["streaming_electrostatic", "electrostatic_slices"],
)
def test_linear_rhs_electrostatic_routes_build_identity_summary(
    monkeypatch, gate: Any, builder: str
) -> None:
    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            object(),
            object(),
            _Grid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return 3.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_parallel_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )

    summary = getattr(gate, builder)(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["max_phi_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert len(summary["rows"]) == 4


def test_parallel_ky_scan_gate_builds_identity_summary(monkeypatch) -> None:
    calls: list[int] = []

    def fake_scan(ky_values, *, ky_batch, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(int(ky_batch))
        result = CycloneScanResult(
            ky=np.asarray(ky_values, dtype=float),
            gamma=np.asarray([0.1, 0.2], dtype=float),
            omega=np.asarray([0.3, 0.4], dtype=float),
        )
        return result, 4.0 if ky_batch == 1 else 2.0

    monkeypatch.setattr(parallel_identity_gate, "_timed_cyclone_scan", fake_scan)
    summary = parallel_identity_gate.build_parallel_ky_scan_gate(
        ky_values=np.asarray([0.1, 0.2]),
        serial_batch=1,
        parallel_batch=2,
        gamma_rtol=1.0e-12,
        omega_atol=1.0e-12,
        steps=4,
        dt=0.1,
        nx=1,
        ny=4,
        nz=8,
        nlaguerre=2,
        nhermite=3,
    )

    assert calls == [1, 2]
    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["max_gamma_rel_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_logical_cpu_parallel_scan_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_devices(requested_devices: int):  # type: ignore[no-untyped-def]
        return [object()] * requested_devices

    def fake_scan(ky_values, *, batch_size, devices):  # type: ignore[no-untyped-def]
        assert devices
        ky = np.asarray(ky_values, dtype=float)
        return {
            "gamma": ky + 0.1,
            "omega": -ky,
            "kperp2": ky**2 + 0.08,
            "ql_proxy": (ky + 0.1) / (ky**2 + 0.08),
        }, 4.0 if batch_size == 1 else 2.0

    monkeypatch.setattr(parallel_identity_gate, "_select_devices", fake_devices)
    monkeypatch.setattr(parallel_identity_gate, "_timed_scan_model", fake_scan)

    summary = parallel_identity_gate.build_logical_cpu_parallel_scan_gate(
        ky_values=np.asarray([0.1, 0.2]),
        serial_batch=1,
        parallel_batch=2,
        requested_devices=2,
        gamma_rtol=1.0e-12,
        omega_atol=1.0e-12,
        ql_rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["device_parallel_config"]["strategy"] == "device_batch"
    assert summary["device_parallel_config"]["num_devices"] == 2
    assert summary["max_ql_rel_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_quasilinear_runtime_parallel_gate_builds_identity_summary(monkeypatch) -> None:
    calls: list[int] = []

    def fake_timed_scan(_cfg, ky_values, *, workers, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(int(workers))
        return _runtime_scan(np.asarray(ky_values, dtype=float), workers=workers), (
            4.0 if workers == 1 else 2.0
        )

    monkeypatch.setattr(parallel_identity_gate, "_timed_runtime_scan", fake_timed_scan)
    summary = parallel_identity_gate.build_quasilinear_runtime_parallel_gate(
        ky_values=np.asarray([0.1, 0.2]),
        workers=2,
        rtol=1.0e-12,
        atol=1.0e-12,
        solver="krylov",
        nx=1,
        ny=8,
        nz=12,
        nlaguerre=2,
        nhermite=2,
    )

    assert calls == [1, 2]
    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["serial_parallel_metadata"]["requested_workers"] == 2
    assert len(summary["rows"]) == 2
    assert summary["rows"][0]["heat_flux_weight_total_abs_error"] == 0.0


def _writer_name(prefix: str) -> str:
    if prefix == "parallel_gate":
        return "write_parallel_ky_scan_artifacts"
    if prefix == "logical_cpu_parallel_gate":
        return "write_logical_cpu_parallel_scan_artifacts"
    if prefix == "ql_parallel_gate":
        return "write_quasilinear_runtime_parallel_artifacts"
    return "write_artifacts"


@pytest.mark.parametrize(
    ("gate", "prefix", "summary", "csv_token"),
    [
        (
            velocity_reduce_gate,
            "velocity_field_reduce_gate",
            {
                "rows": [
                    {
                        "ky_index": 0,
                        "reduced_real": 1.0,
                        "reference_real": 1.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-7,
                "max_allowed_error": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            hermite_exchange_gate,
            "hermite_exchange_gate",
            {
                "rows": [
                    {
                        "m": 0,
                        "center_real": 1.0,
                        "lower_real": 0.0,
                        "upper_real": 2.0,
                        "lower_reference_real": 0.0,
                        "upper_reference_real": 2.0,
                        "lower_abs_error": 0.0,
                        "upper_abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "identity_passed": True,
            },
            "lower_abs_error",
        ),
        (
            hermite_ladder_gate,
            "hermite_streaming_ladder_gate",
            {
                "rows": [
                    {
                        "m": 0,
                        "state_real": 1.0,
                        "ladder_real": 2.0,
                        "reference_real": 2.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            periodic_gate,
            "periodic_streaming_microkernel_gate",
            {
                "rows": [
                    {
                        "m": 0,
                        "state_abs": 1.0,
                        "production_abs": 2.0,
                        "sharded_abs": 2.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            field_reduce_gate,
            "electrostatic_field_reduce_gate",
            {
                "rows": [
                    {
                        "z_index": 0,
                        "serial_abs": 2.0,
                        "sharded_abs": 2.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            diamagnetic_gate,
            "electrostatic_diamagnetic_gate",
            {
                "rows": [
                    {"m": 0, "serial_norm": 2.0, "sharded_norm": 2.0, "abs_error": 0.0}
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            drift_gate,
            "electrostatic_drift_gate",
            {
                "rows": [
                    {
                        "component": "mirror",
                        "serial_norm": 2.0,
                        "sharded_norm": 2.0,
                        "abs_error": 0.0,
                        "rel_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            streaming_gate,
            "linear_rhs_streaming_gate",
            {
                "rows": [
                    {
                        "m": 0,
                        "production_abs": 2.0,
                        "sharded_abs": 2.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            electrostatic_gate,
            "linear_rhs_streaming_electrostatic_gate",
            {
                "rows": [
                    {"m": 0, "serial_abs": 2.0, "sharded_abs": 2.0, "abs_error": 0.0}
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            slices_gate,
            "linear_rhs_electrostatic_slices_gate",
            {
                "rows": [
                    {"m": 0, "serial_norm": 2.0, "sharded_norm": 2.0, "abs_error": 0.0}
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            parallel_identity_gate,
            "parallel_gate",
            {
                "rows": [
                    {
                        "ky": 0.1,
                        "serial_gamma": 0.1,
                        "batched_gamma": 0.1,
                        "gamma_rel_error": 0.0,
                        "serial_omega": 0.2,
                        "batched_omega": 0.2,
                        "omega_abs_error": 0.0,
                    }
                ],
                "gamma_rtol": 1.0e-8,
                "omega_atol": 1.0e-8,
                "serial_elapsed_s": 2.0,
                "batched_elapsed_s": 1.0,
                "observed_speedup": 2.0,
                "identity_passed": True,
            },
            "gamma_rel_error",
        ),
        (
            parallel_identity_gate,
            "logical_cpu_parallel_gate",
            {
                "rows": [
                    {
                        "ky": 0.1,
                        "serial_gamma": 0.2,
                        "batched_gamma": 0.2,
                        "gamma_rel_error": 0.0,
                        "serial_omega": -0.1,
                        "batched_omega": -0.1,
                        "omega_abs_error": 0.0,
                        "serial_ql_proxy": 1.5,
                        "batched_ql_proxy": 1.5,
                        "ql_rel_error": 0.0,
                    }
                ],
                "gamma_rtol": 1.0e-8,
                "omega_atol": 1.0e-8,
                "ql_rtol": 1.0e-8,
                "serial_elapsed_s": 2.0,
                "batched_elapsed_s": 1.0,
                "observed_speedup": 2.0,
                "identity_passed": True,
            },
            "ql_rel_error",
        ),
        (
            parallel_identity_gate,
            "ql_parallel_gate",
            {
                "identity_passed": True,
                "observed_speedup": 2.0,
                "atol": 1.0e-12,
                "rows": [
                    {
                        "ky": 0.1,
                        "serial_heat_flux_weight_total": 0.2,
                        "parallel_heat_flux_weight_total": 0.2,
                        "heat_flux_weight_total_abs_error": 0.0,
                        "serial_saturated_heat_flux_total": 0.06,
                        "parallel_saturated_heat_flux_total": 0.06,
                        "saturated_heat_flux_total_abs_error": 0.0,
                    }
                ],
            },
            "heat_flux_weight_total_abs_error",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_parallel_identity_gate_writes_artifacts(
    tmp_path: Path, gate: Any, prefix: str, summary: dict[str, object], csv_token: str
) -> None:
    out = tmp_path / prefix
    paths = getattr(gate, _writer_name(prefix))(summary, out)
    _assert_standard_artifacts(paths, out, csv_token)
