from __future__ import annotations

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.linear import (  # noqa: E402
    apply_hermite_v,
    apply_hermite_v2,
    apply_laguerre_x,
    apply_mapped_hermite_v,
    apply_mapped_hermite_v2,
    apply_mapped_laguerre_x,
    diamagnetic_drive_coeffs,
    energy_operator,
)
from spectraxgk.velocity_maps import VelocityMapConfig  # noqa: E402


def _finite_difference_component(f, x: jnp.ndarray, i: int, *, eps: float = 1.0e-6) -> float:
    dx = jnp.zeros_like(x).at[i].set(eps)
    return float((f(x + dx) - f(x - dx)) / (2.0 * eps))


def test_mapped_linear_helpers_identity_regression():
    G = jnp.arange(2 * 4 * 1 * 1 * 1, dtype=jnp.float64).reshape((2, 4, 1, 1, 1))

    np.testing.assert_allclose(
        np.asarray(apply_mapped_hermite_v(G)),
        np.asarray(apply_hermite_v(G)),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(apply_mapped_hermite_v2(G)),
        np.asarray(apply_hermite_v2(G)),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(apply_mapped_laguerre_x(G)),
        np.asarray(apply_laguerre_x(G)),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_mapped_hermite_shift_scale_algebra():
    G = jnp.zeros((2, 5, 1, 1, 1), dtype=jnp.float64)
    G = G.at[0, 0, 0, 0, 0].set(0.7)
    G = G.at[1, 2, 0, 0, 0].set(-0.4)
    cfg = VelocityMapConfig(parallel_shift=0.25, parallel_log_scale=jnp.log(1.6))

    vG = apply_hermite_v(G)
    v2G = apply_hermite_v2(G)
    mapped_v = apply_mapped_hermite_v(G, cfg)
    mapped_v2 = apply_mapped_hermite_v2(G, cfg)

    expected_v = 0.25 * G + 1.6 * vG
    expected_v2 = (0.25**2) * G + 2.0 * 0.25 * 1.6 * vG + (1.6**2) * v2G
    np.testing.assert_allclose(np.asarray(mapped_v), np.asarray(expected_v), rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(mapped_v2), np.asarray(expected_v2), rtol=1.0e-14, atol=1.0e-14)


def test_mapped_laguerre_perpendicular_scale():
    G = jnp.zeros((4, 3, 1, 1, 1), dtype=jnp.float64)
    G = G.at[2, 1, 0, 0, 0].set(1.0)
    cfg = VelocityMapConfig(perpendicular_log_scale=jnp.log(0.6))

    np.testing.assert_allclose(
        np.asarray(apply_mapped_laguerre_x(G, cfg)),
        np.asarray(0.6 * apply_laguerre_x(G)),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_mapped_energy_operator_gradients_match_finite_difference():
    G = jnp.linspace(-0.4, 0.8, 4 * 5, dtype=jnp.float64).reshape((4, 5, 1, 1, 1))

    def objective(theta):
        cfg = VelocityMapConfig(
            parallel_shift=theta[0],
            parallel_log_scale=theta[1],
            perpendicular_log_scale=theta[2],
        )
        out = energy_operator(G, coeff_const=0.3, coeff_par=0.7, coeff_perp=0.2, velocity_map=cfg)
        return jnp.vdot(out, out).real

    theta0 = jnp.asarray([0.15, -0.2, 0.1], dtype=jnp.float64)
    ad = jax.grad(objective)(theta0)
    fd = jnp.asarray([_finite_difference_component(objective, theta0, i) for i in range(theta0.size)])
    np.testing.assert_allclose(np.asarray(ad), np.asarray(fd), rtol=1.0e-6, atol=1.0e-6)


def test_mapped_diamagnetic_drive_coefficients_expose_shifted_mode():
    eta_i = jnp.asarray(1.0, dtype=jnp.float64)
    default_coeffs = diamagnetic_drive_coeffs(
        3, 4, eta_i=eta_i, coeff_const=1.0, coeff_par=0.5, coeff_perp=1.0
    )
    identity_coeffs = diamagnetic_drive_coeffs(
        3,
        4,
        eta_i=eta_i,
        coeff_const=1.0,
        coeff_par=0.5,
        coeff_perp=1.0,
        velocity_map=VelocityMapConfig(),
    )
    np.testing.assert_allclose(np.asarray(identity_coeffs), np.asarray(default_coeffs), rtol=0.0, atol=1.0e-14)

    shifted_coeffs = diamagnetic_drive_coeffs(
        3,
        4,
        eta_i=eta_i,
        coeff_const=1.0,
        coeff_par=0.5,
        coeff_perp=1.0,
        velocity_map=VelocityMapConfig(parallel_shift=0.2),
    )
    assert abs(float(shifted_coeffs[0, 1])) > 0.0
