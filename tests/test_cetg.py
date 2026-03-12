"""Tests for the dedicated GX cETG reduced-model path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from spectraxgk.cetg import build_cetg_model_params, cetg_fields, validate_cetg_runtime_config
from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.geometry import SlabGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.runtime import run_runtime_linear, run_runtime_nonlinear
from spectraxgk.runtime_config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


def _base_cetg_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=8,
            Ny=8,
            Nz=12,
            Lx=12.0,
            Ly=12.0,
            boundary="periodic",
            y0=2.0,
            ntheta=12,
            nperiod=1,
        ),
        time=TimeConfig(
            t_max=0.08,
            dt=0.01,
            method="sspx3",
            use_diffrax=False,
            fixed_dt=True,
            sample_stride=1,
            diagnostics_stride=1,
            gx_real_fft=True,
            nonlinear_dealias=True,
        ),
        geometry=GeometryConfig(model="slab", z0=np.pi, zero_shat=True),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-3,
            init_single=False,
            gaussian_init=False,
            kpar_init=1.0,
        ),
        species=(
            RuntimeSpeciesConfig(
                name="electron",
                charge=-1.0,
                mass=2.7e-4,
                density=1.0,
                temperature=1.0,
                tprim=0.0,
                fprim=0.0,
                kinetic=True,
            ),
        ),
        physics=RuntimePhysicsConfig(
            reduced_model="cetg",
            linear=False,
            nonlinear=True,
            electrostatic=True,
            electromagnetic=False,
            adiabatic_electrons=False,
            adiabatic_ions=True,
            tau_e=1.0,
            tau_fac=1.0,
            z_ion=1.0,
            collisions=False,
            hypercollisions=False,
        ),
        collisions=RuntimeCollisionConfig(D_hyper=5.0e-4, nu_hyper=0.0),
        normalization=RuntimeNormalizationConfig(contract="kinetic", diagnostic_norm="gx"),
        terms=RuntimeTermsConfig(
            streaming=1.0,
            mirror=0.0,
            curvature=0.0,
            gradb=0.0,
            diamagnetic=0.0,
            collisions=0.0,
            hypercollisions=0.0,
            hyperdiffusion=1.0,
            end_damping=0.0,
            apar=0.0,
            bpar=0.0,
            nonlinear=1.0,
        ),
        expert=RuntimeExpertConfig(dealias_kz=True),
    )


def test_build_cetg_model_params_matches_gx_defaults() -> None:
    cfg = _base_cetg_cfg()
    geom = SlabGeometry.from_config(cfg.geometry)

    validate_cetg_runtime_config(cfg, geom, Nl=2, Nm=1)
    params = build_cetg_model_params(cfg, geom, Nl=2, Nm=1)

    assert params.tau_fac == 1.0
    assert params.z_ion == 1.0
    assert params.nu_hyper == 2.0
    assert params.D_hyper == 5.0e-4
    assert params.dealias_kz is True


def test_cetg_field_solve_matches_gx_tau_bar() -> None:
    cfg = _base_cetg_cfg()
    geom = SlabGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    params = build_cetg_model_params(cfg, geom, Nl=2, Nm=1)

    density = np.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    temperature = np.zeros_like(density)
    G = np.stack([density, temperature], axis=0)[None, :, None, :, :, :]

    fields = cetg_fields(G, grid, params)

    mask = np.asarray(grid.dealias_mask, dtype=bool)
    phi = np.asarray(fields.phi)
    assert np.allclose(phi[mask], -1.0)
    assert np.allclose(phi[~mask], 0.0)


def test_runtime_linear_cetg_smoke_uses_model_native_dims() -> None:
    cfg = replace(_base_cetg_cfg(), physics=replace(_base_cetg_cfg().physics, linear=True, nonlinear=False))

    out = run_runtime_linear(cfg, ky_target=1.0 / 2.0, solver="time", steps=4, sample_stride=1, return_state=True)

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)
    assert out.state is not None
    assert out.state.shape[0:3] == (1, 2, 1)


def test_runtime_nonlinear_cetg_smoke_uses_model_native_dims() -> None:
    cfg = _base_cetg_cfg()

    out = run_runtime_nonlinear(cfg, ky_target=1.0 / 2.0, kx_target=0.0, steps=4, sample_stride=1, return_state=True)

    assert out.diagnostics is not None
    assert out.state is not None
    assert out.state.shape[0:3] == (1, 2, 1)
    assert np.all(np.isfinite(np.asarray(out.diagnostics.Wg_t)))
    assert np.allclose(np.asarray(out.diagnostics.Wapar_t), 0.0)
    assert np.allclose(np.asarray(out.diagnostics.particle_flux_t), 0.0)
