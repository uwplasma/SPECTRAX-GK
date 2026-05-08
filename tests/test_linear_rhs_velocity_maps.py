from __future__ import annotations

from dataclasses import replace

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.config import GridConfig  # noqa: E402
from spectraxgk.geometry import SAlphaGeometry  # noqa: E402
from spectraxgk.grids import build_spectral_grid, select_ky_grid  # noqa: E402
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, linear_rhs_cached  # noqa: E402
from spectraxgk.terms.assembly import assemble_rhs_cached, assemble_rhs_terms_cached  # noqa: E402
from spectraxgk.terms.config import TermConfig  # noqa: E402
from spectraxgk.velocity_maps import VelocityMapConfig  # noqa: E402


def _finite_difference_component(f, x: jnp.ndarray, i: int, *, eps: float = 1.0e-6) -> float:
    dx = jnp.zeros_like(x).at[i].set(eps)
    return float((f(x + dx) - f(x - dx)) / (2.0 * eps))


def _small_linear_case(seed: int = 0):
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        R_over_Ln=0.8,
        R_over_LTi=2.49,
        R_over_LTe=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    Nl, Nm = 4, 5
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    rng = np.random.default_rng(seed)
    G0 = rng.normal(size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    )
    return jnp.asarray(G0), cache, params


def _mapped_terms() -> TermConfig:
    return TermConfig(
        streaming=1.0,
        mirror=0.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


def test_linear_rhs_velocity_map_identity_matches_default_rhs():
    G0, cache, params = _small_linear_case()
    terms = _mapped_terms()

    rhs_default, fields_default = assemble_rhs_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_custom_vjp=False,
    )
    rhs_identity, fields_identity = assemble_rhs_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_custom_vjp=False,
        velocity_map=VelocityMapConfig(),
    )

    np.testing.assert_allclose(np.asarray(fields_identity.phi), np.asarray(fields_default.phi), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(rhs_identity), np.asarray(rhs_default), rtol=1.0e-12, atol=1.0e-12)


def test_linear_rhs_velocity_map_params_and_keyword_paths_match():
    G0, cache, params = _small_linear_case(seed=1)
    cfg = VelocityMapConfig(
        parallel_shift=0.12,
        parallel_log_scale=-0.08,
        perpendicular_log_scale=0.05,
    )
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    rhs_from_keyword, phi_from_keyword = linear_rhs_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        velocity_map=cfg,
    )
    rhs_from_params, phi_from_params = linear_rhs_cached(
        G0,
        cache,
        replace(params, velocity_map=cfg),
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
    )
    rhs_from_jit, phi_from_jit = linear_rhs_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_jit=True,
        velocity_map=cfg,
    )

    np.testing.assert_allclose(np.asarray(phi_from_params), np.asarray(phi_from_keyword), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(rhs_from_params), np.asarray(rhs_from_keyword), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(phi_from_jit), np.asarray(phi_from_keyword), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(rhs_from_jit), np.asarray(rhs_from_keyword), rtol=1.0e-12, atol=1.0e-12)


def test_linear_rhs_velocity_map_gradients_match_finite_difference():
    G0, cache, params = _small_linear_case(seed=2)
    terms = _mapped_terms()

    def objective(theta):
        cfg = VelocityMapConfig(
            parallel_shift=theta[0],
            parallel_log_scale=theta[1],
            perpendicular_log_scale=theta[2],
        )
        rhs, _fields = assemble_rhs_cached(
            G0,
            cache,
            params,
            terms=terms,
            use_custom_vjp=False,
            velocity_map=cfg,
        )
        return jnp.vdot(rhs, rhs).real

    theta0 = jnp.asarray([0.08, -0.05, 0.07], dtype=jnp.float64)
    ad = jax.grad(objective)(theta0)
    fd = jnp.asarray([_finite_difference_component(objective, theta0, i) for i in range(theta0.size)])
    np.testing.assert_allclose(np.asarray(ad), np.asarray(fd), rtol=1.0e-5, atol=1.0e-5)


def test_linear_rhs_velocity_map_term_sum_matches_total():
    G0, cache, params = _small_linear_case(seed=3)
    cfg = VelocityMapConfig(
        parallel_shift=0.15,
        parallel_log_scale=0.1,
        perpendicular_log_scale=-0.12,
    )
    terms = _mapped_terms()

    rhs_total, _fields = assemble_rhs_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_custom_vjp=False,
        velocity_map=cfg,
    )
    rhs_terms, _fields_terms, contrib = assemble_rhs_terms_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_custom_vjp=False,
        velocity_map=cfg,
    )
    rhs_sum = (
        contrib["streaming"]
        + contrib["mirror"]
        + contrib["curvature"]
        + contrib["gradb"]
        + contrib["diamagnetic"]
        + contrib["collisions"]
        + contrib["hypercollisions"]
        + contrib["hyperdiffusion"]
        + contrib["end_damping"]
    )

    np.testing.assert_allclose(np.asarray(rhs_terms), np.asarray(rhs_total), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(rhs_sum), np.asarray(rhs_total), rtol=1.0e-12, atol=1.0e-12)
