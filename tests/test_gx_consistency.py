import math
import numpy as np
import jax.numpy as jnp

from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import J_l_all
from spectraxgk.terms.linear_terms import hypercollisions_contribution


def _gx_jflr(l: int, b: jnp.ndarray) -> jnp.ndarray:
    """GX Jflr: exp(-b/2) * (-b/2)^l / l!."""

    return jnp.exp(-0.5 * b) * ((-0.5 * b) ** l) / jnp.asarray(math.factorial(l), dtype=b.dtype)


def test_gyroaverage_matches_gx_jflr():
    b = jnp.asarray([0.0, 0.3, 1.0], dtype=jnp.float32)
    Jl = J_l_all(b, l_max=4)
    for l in range(5):
        expected = _gx_jflr(l, b)
        assert jnp.allclose(Jl[l], expected, rtol=1.0e-6, atol=1.0e-7)


def test_salpha_geometry_matches_gx_formulas():
    geom = SAlphaGeometry(
        q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0, alpha=0.0, drift_scale=1.0
    )
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)

    bmag = 1.0 / (1.0 + geom.epsilon * jnp.cos(theta))
    bgrad = geom.gradpar() * geom.epsilon * jnp.sin(theta) * bmag
    gds2 = 1.0 + shear * shear
    gds21 = -geom.s_hat * shear
    gds22 = jnp.asarray(geom.s_hat * geom.s_hat, dtype=jnp.float32)
    cv = (jnp.cos(theta) + shear * jnp.sin(theta)) / geom.R0
    gb = cv
    cv0 = (-geom.s_hat * jnp.sin(theta)) / geom.R0
    gb0 = cv0

    bmag_gx = geom.bmag(theta)
    bgrad_gx = geom.bgrad(theta)
    gds2_gx, gds21_gx, gds22_gx = geom.metric_coeffs(theta)
    cv_d, gb_d = geom.drift_components(jnp.asarray([0.1]), jnp.asarray([0.2]), theta)
    cv_d = cv_d[0, 0]
    gb_d = gb_d[0, 0]

    assert jnp.allclose(bmag_gx, bmag, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(bgrad_gx, bgrad, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(gds2_gx, gds2, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(gds21_gx, gds21, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(gds22_gx, gds22, rtol=1.0e-6, atol=1.0e-8)

    kx0 = jnp.asarray([0.1])
    ky0 = jnp.asarray([0.2])
    kx_hat = kx0 / geom.s_hat
    cv_d_expected = ky0[:, None] * cv + kx_hat[:, None] * cv0
    gb_d_expected = ky0[:, None] * gb + kx_hat[:, None] * gb0
    assert jnp.allclose(cv_d, cv_d_expected[0], rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(gb_d, gb_d_expected[0], rtol=1.0e-10, atol=1.0e-12)

    kperp2 = geom.k_perp2(kx0, ky0, theta)
    bmag_inv = 1.0 / bmag
    shat_inv = 1.0 / geom.s_hat
    gds22_match = jnp.asarray(gds22_gx, dtype=gds2.dtype)
    kperp2_expected = (
        ky0[:, None] * (ky0[:, None] * gds2 + 2.0 * kx0[:, None] * shat_inv * gds21)
        + (kx0[:, None] * shat_inv) ** 2 * gds22_match
    ) * (bmag_inv * bmag_inv)
    assert jnp.allclose(kperp2, kperp2_expected[0], rtol=1.0e-9, atol=1.0e-11)


def test_salpha_geometry_kperp2_matches_gs2_formula():
    """GS2-style kperp2 omits the bmag^{-2} factor."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0, alpha=0.0, kperp2_bmag=False)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)
    gds2 = 1.0 + shear * shear
    gds21 = -geom.s_hat * shear
    gds22 = jnp.asarray(geom.s_hat * geom.s_hat, dtype=jnp.float32)

    kx0 = jnp.asarray([0.1])
    ky0 = jnp.asarray([0.2])
    kx_hat = kx0 / geom.s_hat
    kperp2 = geom.k_perp2(kx0, ky0, theta)
    kperp2_expected = ky0[:, None] * (ky0[:, None] * gds2 + 2.0 * kx_hat[:, None] * gds21) + (
        kx_hat[:, None] ** 2
    ) * gds22
    assert jnp.allclose(kperp2, kperp2_expected[0], rtol=1.0e-9, atol=1.0e-11)


def test_hypercollisions_matches_gx_formula():
    Nl, Nm = 6, 12
    G = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.complex64)
    l = jnp.arange(Nl, dtype=jnp.float32)[:, None, None, None, None]
    m = jnp.arange(Nm, dtype=jnp.float32)[None, :, None, None, None]
    vth = jnp.asarray([1.0], dtype=jnp.float32)
    nu_hyper_l = jnp.asarray(0.5, dtype=jnp.float32)
    nu_hyper_m = jnp.asarray(0.5, dtype=jnp.float32)
    nu_hyper_lm = jnp.asarray(0.0, dtype=jnp.float32)
    p_hyper_l = jnp.asarray(6.0, dtype=jnp.float32)
    p_hyper_m = jnp.asarray(6.0, dtype=jnp.float32)
    p_hyper_lm = jnp.asarray(6.0, dtype=jnp.float32)
    nu_hyper = jnp.asarray(0.0, dtype=jnp.float32)
    hyper_ratio = jnp.zeros((Nl, Nm, 1, 1, 1), dtype=jnp.float32)
    ratio_l = (l / float(Nl)) ** p_hyper_l
    ratio_m = (m / float(Nm)) ** p_hyper_m
    ratio_lm = ((2.0 * l + m) / (2.0 * float(Nl) + float(Nm))) ** p_hyper_lm
    mask_const = (m > 2.0) | (l > 1.0)
    mask_kz = jnp.zeros_like(mask_const)
    m_pow = m ** p_hyper_m
    m_norm_kz = float(max(Nm - 1, 1))
    m_norm_kz_factor = (p_hyper_m + 0.5) / (m_norm_kz ** (p_hyper_m + 0.5))
    kz = jnp.asarray([0.0], dtype=jnp.float32)
    kpar_scale = jnp.asarray(1.0, dtype=jnp.float32)

    out = hypercollisions_contribution(
        G,
        vth=vth,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hyper_ratio=hyper_ratio,
        ratio_l=ratio_l,
        ratio_m=ratio_m,
        ratio_lm=ratio_lm,
        mask_const=mask_const,
        mask_kz=mask_kz,
        m_pow=m_pow,
        m_norm_kz_factor=m_norm_kz_factor,
        kz=kz,
        kpar_scale=kpar_scale,
        hypercollisions_const=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(0.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    l_norm = float(Nl)
    m_norm = float(Nm)
    scaled_nu_l = l_norm * nu_hyper_l
    scaled_nu_m = m_norm * nu_hyper_m
    hyper_term = -vth[:, None, None, None, None, None] * (
        scaled_nu_l * ratio_l + scaled_nu_m * ratio_m
    ) - nu_hyper_lm * ratio_lm
    expected = jnp.where(mask_const, hyper_term, 0.0) * G

    assert jnp.allclose(out, expected, rtol=1.0e-6, atol=1.0e-7)
