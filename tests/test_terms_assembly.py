"""Tests for modular RHS assembly and linear-term adapters."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    linear_rhs_cached,
    linear_terms_to_term_config,
    term_config_to_linear_terms,
)
from spectraxgk.terms.assembly import assemble_rhs_cached, compute_fields_cached


def test_linear_term_adapter_roundtrip() -> None:
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.8,
        curvature=0.7,
        gradb=0.6,
        diamagnetic=0.5,
        collisions=0.4,
        hypercollisions=0.3,
        end_damping=0.2,
        apar=0.1,
        bpar=0.9,
    )
    term_cfg = linear_terms_to_term_config(terms, nonlinear=0.25)
    assert term_cfg.nonlinear == 0.25
    roundtrip = term_config_to_linear_terms(term_cfg)
    assert roundtrip == terms


def test_linear_rhs_cached_matches_modular_assembly() -> None:
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        beta=0.02,
        fapar=1.0,
        nu=0.03,
        nu_hyper=0.02,
        damp_ends_amp=0.01,
    )
    Nl, Nm = 3, 4
    cache = build_linear_cache(grid, geom, params, Nl=Nl, Nm=Nm)
    values = jnp.arange(Nl * Nm * grid.ky.size * grid.kx.size * grid.z.size, dtype=jnp.float32)
    values = values.reshape((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))
    G = (1.0e-3 * values).astype(jnp.complex64) + 1j * (2.0e-4 * jnp.cos(values)).astype(jnp.complex64)
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=1.0,
        hypercollisions=1.0,
        end_damping=1.0,
        apar=1.0,
        bpar=1.0,
    )
    term_cfg = linear_terms_to_term_config(terms)

    dG_linear, phi_linear = linear_rhs_cached(G, cache, params, terms=terms, use_jit=False)
    dG_modular, fields = assemble_rhs_cached(G, cache, params, terms=term_cfg, use_custom_vjp=False)
    dG_modular_vjp, fields_vjp = assemble_rhs_cached(G, cache, params, terms=term_cfg, use_custom_vjp=True)
    fields_only = compute_fields_cached(G, cache, params, terms=term_cfg, use_custom_vjp=False)

    assert jnp.allclose(dG_linear, dG_modular, rtol=1.0e-7, atol=1.0e-7)
    assert jnp.allclose(dG_modular_vjp, dG_modular, rtol=1.0e-7, atol=1.0e-7)
    assert jnp.allclose(phi_linear, fields.phi, rtol=1.0e-7, atol=1.0e-7)
    assert jnp.allclose(phi_linear, fields_vjp.phi, rtol=1.0e-7, atol=1.0e-7)
    assert jnp.allclose(phi_linear, fields_only.phi, rtol=1.0e-7, atol=1.0e-7)


def test_assembly_validates_state_shapes_and_species_alignment() -> None:
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=20.0, Ly=20.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)

    with pytest.raises(ValueError):
        assemble_rhs_cached(jnp.zeros((2, 3, 4), dtype=jnp.complex64), cache, params)

    # cache is single-species; this state is two-species.
    with pytest.raises(ValueError):
        assemble_rhs_cached(
            jnp.zeros((2, 2, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
            cache,
            params,
        )

    with pytest.raises(ValueError):
        compute_fields_cached(jnp.zeros((2, 3, 4), dtype=jnp.complex64), cache, params)
