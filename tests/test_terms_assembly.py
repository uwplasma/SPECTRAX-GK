import numpy as np
import jax.numpy as jnp

from spectraxgk.config import GridConfig
from spectraxgk.geometry import SAlphaGeometry, sample_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.terms.assembly import (
    assemble_rhs,
    assemble_rhs_cached,
    assemble_rhs_terms_cached,
)
from spectraxgk.terms.config import TermConfig


def test_assemble_rhs_terms_sum_matches_total() -> None:
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
    Nl, Nm = 4, 4
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    rng = np.random.default_rng(0)
    G0 = rng.normal(size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    )
    G0 = jnp.asarray(G0)
    term_cfg = TermConfig()
    rhs_total, _fields = assemble_rhs_cached(G0, cache, params, terms=term_cfg)
    rhs_terms, _fields_terms, contrib = assemble_rhs_terms_cached(G0, cache, params, terms=term_cfg)
    rhs_sum = (
        contrib["streaming"]
        + contrib["mirror"]
        + contrib["curvature"]
        + contrib["gradb"]
        + contrib["diamagnetic"]
        + contrib["collisions"]
        + contrib["hypercollisions"]
        + contrib["end_damping"]
    )
    assert np.allclose(np.asarray(rhs_terms), np.asarray(rhs_total), rtol=1.0e-6, atol=1.0e-8)
    assert np.allclose(np.asarray(rhs_sum), np.asarray(rhs_total), rtol=1.0e-6, atol=1.0e-8)


def test_assemble_rhs_accepts_sampled_geometry_contract() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    sampled = sample_flux_tube_geometry(geom, grid.z)
    params = LinearParams(
        R_over_Ln=0.8,
        R_over_LTi=2.49,
        R_over_LTe=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(sampled.gradpar()),
        nu=0.0,
    )
    Nl, Nm = 4, 4
    rng = np.random.default_rng(1)
    G0 = rng.normal(size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    )
    rhs, fields = assemble_rhs(
        jnp.asarray(G0),
        grid,
        sampled,
        params,
        Nl=Nl,
        Nm=Nm,
    )
    assert rhs.shape == G0.shape
    assert fields.phi.shape == (grid.ky.size, grid.kx.size, grid.z.size)
