import math

import jax.numpy as jnp

from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.velocity import J_l_all, single_precision_factorial
from spectraxgk.terms import linear_terms as linear_terms_module
from spectraxgk.terms.linear_terms import (
    diamagnetic_contribution,
    end_damping_contribution,
    hypercollisions_contribution,
    hyperdiffusion_contribution,
    streaming_contribution,
    linked_streaming_contribution,
)


def _gx_jflr(ell: int, b: jnp.ndarray) -> jnp.ndarray:
    """GX Jflr: exp(-b/2) * (-b/2)^ell / ell!."""

    return (
        jnp.exp(-0.5 * b)
        * ((-0.5 * b) ** ell)
        / jnp.asarray(math.factorial(ell), dtype=b.dtype)
    )


def test_gyroaverage_matches_gx_jflr():
    b = jnp.asarray([0.0, 0.3, 1.0], dtype=jnp.float32)
    Jl = J_l_all(b, l_max=4)
    for ell in range(5):
        expected = _gx_jflr(ell, b)
        assert jnp.allclose(Jl[ell], expected, rtol=1.0e-6, atol=1.0e-7)


def test_single_precision_factorial_matches_stirling_branch():
    m = jnp.asarray([7.0, 8.0, 12.0], dtype=jnp.float32)
    expected = jnp.asarray(
        [
            math.sqrt(2.0 * math.pi * x)
            * (x**x)
            * math.exp(-x)
            * (1.0 + 1.0 / (12.0 * x) + 1.0 / (288.0 * x * x))
            for x in (7.0, 8.0, 12.0)
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(
        single_precision_factorial(m), expected, rtol=1.0e-7, atol=1.0e-7
    )


def test_gyroaverage_matches_gx_jflr_stirling_branch():
    b = jnp.asarray([0.3, 1.0, 2.5], dtype=jnp.float32)
    ell = 7
    expected = (
        jnp.exp(-0.5 * b)
        * ((-0.5 * b) ** ell)
        / single_precision_factorial(jnp.asarray(float(ell), dtype=b.dtype))
    )
    Jl = J_l_all(b, l_max=ell)
    assert jnp.allclose(Jl[ell], expected, rtol=1.0e-7, atol=1.0e-8)


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
    # Allow one-ulp level differences from mixed float32/float64 intermediates.
    assert jnp.allclose(kperp2, kperp2_expected[0], rtol=1.0e-8, atol=5.0e-10)


def test_salpha_geometry_kperp2_matches_alternate_formula():
    """Alternate kperp2 convention omits the bmag^{-2} factor."""
    geom = SAlphaGeometry(
        q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0, alpha=0.0, kperp2_bmag=False
    )
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)
    gds2 = 1.0 + shear * shear
    gds21 = -geom.s_hat * shear
    gds22 = jnp.asarray(geom.s_hat * geom.s_hat, dtype=jnp.float32)

    kx0 = jnp.asarray([0.1])
    ky0 = jnp.asarray([0.2])
    kx_hat = kx0 / geom.s_hat
    kperp2 = geom.k_perp2(kx0, ky0, theta)
    kperp2_expected = (
        ky0[:, None] * (ky0[:, None] * gds2 + 2.0 * kx_hat[:, None] * gds21)
        + (kx_hat[:, None] ** 2) * gds22
    )
    # Allow one-ulp level differences from mixed float32/float64 intermediates.
    assert jnp.allclose(kperp2, kperp2_expected[0], rtol=1.0e-8, atol=5.0e-10)


def test_hypercollisions_matches_gx_formula():
    Nl, Nm = 6, 12
    G = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.complex64)
    ell = jnp.arange(Nl, dtype=jnp.float32)[:, None, None, None, None]
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
    ratio_l = (ell / float(Nl)) ** p_hyper_l
    ratio_m = (m / float(Nm)) ** p_hyper_m
    ratio_lm = ((2.0 * ell + m) / (2.0 * float(Nl) + float(Nm))) ** p_hyper_lm
    mask_const = (m > 2.0) | (ell > 1.0)
    mask_kz = jnp.zeros_like(mask_const)
    m_pow = m**p_hyper_m
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
    hyper_term = (
        -vth[:, None, None, None, None, None]
        * (scaled_nu_l * ratio_l + scaled_nu_m * ratio_m)
        - nu_hyper_lm * ratio_lm
    )
    expected = jnp.where(mask_const, hyper_term, 0.0) * G

    assert jnp.allclose(out, expected, rtol=1.0e-6, atol=1.0e-7)


def test_hypercollisions_skips_linked_abs_kz_when_kz_weight_is_zero(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError(
            "abs_z_linked_fft should not run when hypercollisions_kz is zero"
        )

    monkeypatch.setattr(linear_terms_module, "abs_z_linked_fft", _fail)

    Nl, Nm = 2, 4
    G = jnp.ones((1, Nl, Nm, 1, 1, 2), dtype=jnp.complex64)
    zeros_lm = jnp.zeros((Nl, Nm, 1, 1, 1), dtype=jnp.float32)
    mask_const = jnp.zeros((1, Nl, Nm, 1, 1, 1), dtype=bool)
    mask_kz = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=bool)

    out = hypercollisions_contribution(
        G,
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        nu_hyper=jnp.asarray([0.0], dtype=jnp.float32),
        nu_hyper_l=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(1.0, dtype=jnp.float32),
        nu_hyper_lm=jnp.asarray(0.0, dtype=jnp.float32),
        hyper_ratio=zeros_lm,
        ratio_l=zeros_lm,
        ratio_m=zeros_lm,
        ratio_lm=zeros_lm,
        mask_const=mask_const,
        mask_kz=mask_kz,
        m_pow=jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(1.0, dtype=jnp.float32),
        kz=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_const=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(0.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        linked_indices=(jnp.asarray([[0]], dtype=jnp.int32),),
        linked_kz=(jnp.asarray([0.0, 1.0], dtype=jnp.float32),),
        linked_inverse_permutation=jnp.asarray([0], dtype=jnp.int32),
        linked_full_cover=True,
        linked_gather_map=jnp.asarray([0], dtype=jnp.int32),
        linked_gather_mask=jnp.asarray([True], dtype=bool),
        linked_use_gather=True,
    )

    assert jnp.allclose(out, jnp.zeros_like(G))


def test_hypercollisions_static_zero_operator_skips_linked_abs_kz(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError(
            "abs_z_linked_fft should not run for an exactly zero hypercollision operator"
        )

    monkeypatch.setattr(linear_terms_module, "abs_z_linked_fft", _fail)

    Nl, Nm = 2, 4
    G = jnp.ones((1, Nl, Nm, 1, 1, 2), dtype=jnp.complex64)
    zeros_lm = jnp.zeros((Nl, Nm, 1, 1, 1), dtype=jnp.float32)
    mask = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=bool)

    out = hypercollisions_contribution(
        G,
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        nu_hyper=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_l=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_lm=jnp.asarray(0.0, dtype=jnp.float32),
        hyper_ratio=zeros_lm,
        ratio_l=zeros_lm,
        ratio_m=zeros_lm,
        ratio_lm=zeros_lm,
        mask_const=mask,
        mask_kz=mask,
        m_pow=jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(1.0, dtype=jnp.float32),
        kz=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_const=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        linked_indices=(jnp.asarray([[0]], dtype=jnp.int32),),
        linked_kz=(jnp.asarray([0.0, 1.0], dtype=jnp.float32),),
        linked_inverse_permutation=jnp.asarray([0], dtype=jnp.int32),
        linked_full_cover=True,
        linked_gather_map=jnp.asarray([0], dtype=jnp.int32),
        linked_gather_mask=jnp.asarray([True], dtype=bool),
        linked_use_gather=True,
    )

    assert jnp.allclose(out, jnp.zeros_like(G))


def test_linked_kz_hypercollisions_activate_for_z_varying_state():
    Nl, Nm, Nz = 2, 4, 4
    zeros_lm = jnp.zeros((Nl, Nm, 1, 1, 1), dtype=jnp.float32)
    mask_const = jnp.zeros((1, Nl, Nm, 1, 1, 1), dtype=bool)
    mask_kz = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=bool)
    kwargs = dict(
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        nu_hyper=jnp.asarray([0.0], dtype=jnp.float32),
        nu_hyper_l=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(1.0, dtype=jnp.float32),
        nu_hyper_lm=jnp.asarray(0.0, dtype=jnp.float32),
        hyper_ratio=zeros_lm,
        ratio_l=zeros_lm,
        ratio_m=zeros_lm,
        ratio_lm=zeros_lm,
        mask_const=mask_const,
        mask_kz=mask_kz,
        m_pow=jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(1.0, dtype=jnp.float32),
        kz=jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_const=jnp.asarray(0.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        linked_indices=(jnp.asarray([[0]], dtype=jnp.int32),),
        linked_kz=(jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32),),
        linked_inverse_permutation=jnp.asarray([0], dtype=jnp.int32),
        linked_full_cover=True,
    )

    constant = jnp.ones((1, Nl, Nm, 1, 1, Nz), dtype=jnp.complex64)
    z_varying = constant * jnp.asarray([0.0, 1.0, 0.0, -1.0], dtype=jnp.complex64)

    constant_out = hypercollisions_contribution(constant, **kwargs)
    varying_out = hypercollisions_contribution(z_varying, **kwargs)

    assert jnp.linalg.norm(constant_out) < 1.0e-6
    assert jnp.linalg.norm(varying_out) > 1.0e-3


def test_static_zero_linear_term_guards_skip_expensive_operators(monkeypatch):
    def _fail_streaming(*args, **kwargs):
        raise AssertionError(
            "streaming_term should not run when streaming weight is zero"
        )

    def _fail_grad(*args, **kwargs):
        raise AssertionError(
            "grad_z_periodic should not run when GX streaming weight is zero"
        )

    monkeypatch.setattr(linear_terms_module, "streaming_term", _fail_streaming)
    monkeypatch.setattr(linear_terms_module, "grad_z_periodic", _fail_grad)

    G = jnp.ones((1, 2, 3, 1, 1, 4), dtype=jnp.complex64)
    out = streaming_contribution(
        G,
        kz=jnp.ones((4,), dtype=jnp.float32),
        dz=jnp.asarray(1.0, dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        sqrt_p=jnp.ones((2, 3, 1, 1, 1), dtype=jnp.float32),
        sqrt_m=jnp.ones((2, 3, 1, 1, 1), dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert jnp.allclose(out, jnp.zeros_like(G))

    field = jnp.zeros((1, 1, 4), dtype=jnp.complex64)
    out_gx = linked_streaming_contribution(
        G,
        phi=field,
        apar=field,
        bpar=field,
        Jl=jnp.ones((1, 2, 1, 1, 4), dtype=jnp.float32),
        JlB=jnp.ones((1, 2, 1, 1, 4), dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        sqrt_p=jnp.ones((2, 3, 1, 1, 1), dtype=jnp.float32),
        sqrt_m=jnp.ones((2, 3, 1, 1, 1), dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(0.0, dtype=jnp.float32),
        kz=jnp.ones((4,), dtype=jnp.float32),
        dz=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert jnp.allclose(out_gx, jnp.zeros_like(G))


def test_disabled_em_fields_match_explicit_zero_arrays_in_streaming_and_diamagnetic():
    G = jnp.arange(1 * 2 * 4 * 2 * 1 * 3, dtype=jnp.float32).reshape(1, 2, 4, 2, 1, 3)
    G = (G + 1j * (G + 1.0)).astype(jnp.complex64) * 1.0e-3
    phi = jnp.ones((2, 1, 3), dtype=jnp.complex64) * (0.2 + 0.1j)
    zero_field = jnp.zeros_like(phi)
    Jl = jnp.ones((1, 2, 2, 1, 3), dtype=jnp.float32)
    JlB = 0.5 * Jl
    tz = jnp.asarray([1.0], dtype=jnp.float32)
    vth = jnp.asarray([1.2], dtype=jnp.float32)
    sqrt_p = jnp.ones((2, 4, 1, 1, 1), dtype=jnp.float32)
    sqrt_m = 0.75 * sqrt_p
    common_streaming = dict(
        G=G,
        phi=phi,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        vth=vth,
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        kz=jnp.asarray([0.0, 1.0, -1.0], dtype=jnp.float32),
        dz=jnp.asarray(1.0, dtype=jnp.float32),
    )
    explicit_streaming = linked_streaming_contribution(
        apar=zero_field, bpar=zero_field, **common_streaming
    )
    pruned_streaming = linked_streaming_contribution(
        apar=None, bpar=None, **common_streaming
    )
    assert jnp.allclose(pruned_streaming, explicit_streaming, rtol=1.0e-6, atol=1.0e-7)

    l4 = jnp.arange(2, dtype=jnp.float32)[:, None, None, None]
    common_diamagnetic = dict(
        dG=jnp.zeros_like(G),
        phi=phi,
        Jl=Jl,
        JlB=JlB,
        l4=l4,
        tprim=jnp.asarray([2.0], dtype=jnp.float32),
        fprim=jnp.asarray([0.8], dtype=jnp.float32),
        tz=tz,
        vth=vth,
        omega_star_scale=jnp.asarray(1.0, dtype=jnp.float32),
        ky=jnp.asarray([0.0, 0.3], dtype=jnp.float32),
        imag=jnp.asarray(1j, dtype=jnp.complex64),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )
    explicit_diamagnetic = diamagnetic_contribution(
        apar=zero_field, bpar=zero_field, **common_diamagnetic
    )
    pruned_diamagnetic = diamagnetic_contribution(
        apar=None, bpar=None, **common_diamagnetic
    )
    assert jnp.allclose(
        pruned_diamagnetic, explicit_diamagnetic, rtol=1.0e-6, atol=1.0e-7
    )


def test_static_zero_damping_guards_return_zero_without_profiles():
    G = jnp.ones((1, 2, 3, 2, 2, 4), dtype=jnp.complex64)

    hyperdiff = hyperdiffusion_contribution(
        G,
        kx=jnp.asarray([], dtype=jnp.float32),
        ky=jnp.asarray([], dtype=jnp.float32),
        dealias_mask=jnp.zeros((0, 0), dtype=bool),
        D_hyper=jnp.asarray(0.0, dtype=jnp.float32),
        p_hyper_kperp=jnp.asarray(2.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert jnp.allclose(hyperdiff, jnp.zeros_like(G))

    damp = end_damping_contribution(
        G,
        ky=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        damp_profile=jnp.ones((4,), dtype=jnp.float32),
        linked_damp_profile=jnp.ones((3, 5), dtype=jnp.float32),
        damp_amp=jnp.asarray(0.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert jnp.allclose(damp, jnp.zeros_like(G))
