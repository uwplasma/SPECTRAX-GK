from __future__ import annotations

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.basis import hermite_normed, laguerre  # noqa: E402
from spectraxgk.velocity_maps import (  # noqa: E402
    ModalGate,
    VelocityMapConfig,
    gate_regularization,
    hermite_derivative_matrix,
    hermite_multiply_matrix,
    laguerre_multiply_matrix,
    mapped_parallel_operators,
    mapped_perpendicular_energy_matrix,
)


def _finite_difference(f, x: float, *, eps: float = 1.0e-6) -> float:
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def test_hermite_multiply_matrix_matches_basis_recurrence():
    n = 7
    x = jnp.linspace(-1.2, 1.3, 23)
    psi = hermite_normed(x, n - 1)
    mat = hermite_multiply_matrix(n)

    # Multiplication by the highest retained mode has an out-of-basis component.
    for mode in range(n - 1):
        represented = mat[:, mode] @ psi
        direct = x * psi[mode]
        np.testing.assert_allclose(np.asarray(represented), np.asarray(direct), rtol=1.0e-10, atol=1.0e-10)


def test_hermite_derivative_matrix_matches_basis_derivative():
    n = 7
    x = jnp.linspace(-1.1, 1.4, 19)
    psi = hermite_normed(x, n - 1)
    deriv = hermite_derivative_matrix(n)

    for mode in range(1, n):
        represented = deriv[:, mode] @ psi
        expected = jnp.sqrt(2.0 * mode) * psi[mode - 1]
        np.testing.assert_allclose(np.asarray(represented), np.asarray(expected), rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(deriv[:, 0] @ psi), 0.0, rtol=0.0, atol=1.0e-14)


def test_laguerre_multiply_matrix_matches_recurrence():
    n = 6
    x = jnp.linspace(0.0, 4.0, 31)
    basis = laguerre(x, n - 1)
    mat = laguerre_multiply_matrix(n)

    # Multiplication by the highest retained Laguerre mode has an out-of-basis component.
    for mode in range(n - 1):
        represented = mat[:, mode] @ basis
        direct = x * basis[mode]
        np.testing.assert_allclose(np.asarray(represented), np.asarray(direct), rtol=1.0e-10, atol=1.0e-10)


def test_mapped_parallel_operators_identity_and_shift_scale():
    n = 5
    identity_ops = mapped_parallel_operators(n)
    base_v = hermite_multiply_matrix(n)
    base_d = hermite_derivative_matrix(n)

    np.testing.assert_allclose(np.asarray(identity_ops.multiply), np.asarray(base_v), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(identity_ops.derivative), np.asarray(base_d), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.asarray(identity_ops.energy), np.asarray(base_v @ base_v), rtol=0.0, atol=1.0e-14
    )

    cfg = VelocityMapConfig(parallel_shift=0.3, parallel_log_scale=jnp.log(1.7))
    ops = mapped_parallel_operators(n, cfg)
    expected_v = 0.3 * jnp.eye(n, dtype=jnp.float64) + 1.7 * base_v
    expected_d = base_d / 1.7
    np.testing.assert_allclose(np.asarray(ops.multiply), np.asarray(expected_v), rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(ops.derivative), np.asarray(expected_d), rtol=1.0e-14, atol=1.0e-14)


def test_mapped_parallel_operator_gradients_match_finite_difference():
    n = 6
    coeff = jnp.linspace(0.2, 1.1, n)

    def objective(log_scale):
        cfg = VelocityMapConfig(parallel_shift=0.1, parallel_log_scale=log_scale)
        ops = mapped_parallel_operators(n, cfg)
        return jnp.vdot(coeff, ops.energy @ coeff).real + 0.05 * ops.regularization["parallel_log_scale_sq"]

    x0 = 0.2
    ad = float(jax.grad(objective)(jnp.asarray(x0, dtype=jnp.float64)))
    fd = _finite_difference(lambda z: objective(jnp.asarray(z, dtype=jnp.float64)), x0)
    np.testing.assert_allclose(ad, fd, rtol=1.0e-6, atol=1.0e-6)


def test_perpendicular_energy_scale_and_gradient():
    n = 5
    base = laguerre_multiply_matrix(n)

    def objective(log_scale):
        cfg = VelocityMapConfig(perpendicular_log_scale=log_scale)
        mat = mapped_perpendicular_energy_matrix(n, cfg)
        return jnp.sum(mat * mat)

    cfg = VelocityMapConfig(perpendicular_log_scale=jnp.log(2.0))
    mat = mapped_perpendicular_energy_matrix(n, cfg)
    np.testing.assert_allclose(np.asarray(mat), np.asarray(2.0 * base), rtol=1.0e-14, atol=1.0e-14)

    x0 = 0.1
    ad = float(jax.grad(objective)(jnp.asarray(x0, dtype=jnp.float64)))
    fd = _finite_difference(lambda z: objective(jnp.asarray(z, dtype=jnp.float64)), x0)
    np.testing.assert_allclose(ad, fd, rtol=1.0e-6, atol=1.0e-6)


def test_modal_gate_bounds_application_and_gradients():
    gate = ModalGate(cutoff=3.0, width=0.75)
    vals = gate.values(8)
    assert bool(jnp.all(vals >= 0.0))
    assert bool(jnp.all(vals <= 1.0))
    assert float(vals[0]) > float(vals[-1])

    arr = jnp.ones((2, 8, 3), dtype=jnp.float64)
    gated = gate.apply(arr, axis=1)
    np.testing.assert_allclose(np.asarray(gated[0, :, 0]), np.asarray(vals), rtol=0.0, atol=1.0e-14)

    reg = gate_regularization(gate, 8)
    assert float(reg["gate_total_removed"]) > 0.0
    assert float(reg["gate_roughness"]) > 0.0

    def objective(cutoff):
        g = ModalGate(cutoff=cutoff, width=0.75)
        return jnp.sum(g.values(8) ** 2)

    x0 = 3.0
    ad = float(jax.grad(objective)(jnp.asarray(x0, dtype=jnp.float64)))
    fd = _finite_difference(lambda z: objective(jnp.asarray(z, dtype=jnp.float64)), x0)
    np.testing.assert_allclose(ad, fd, rtol=1.0e-6, atol=1.0e-6)
