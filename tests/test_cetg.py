"""Tests for the dedicated GX cETG reduced-model path."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from spectraxgk.cetg import (
    _cetg_linear_omega_max,
    _project_state,
    _to_internal_state,
    build_cetg_model_params,
    cetg_fields,
    cetg_rhs,
    integrate_cetg_gx_diagnostics_state,
    validate_cetg_runtime_config,
)
from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.geometry import SlabGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.terms.config import TermConfig
from spectraxgk.terms.integrators import _SSPX3_ADT, _SSPX3_W1, _SSPX3_W2, _SSPX3_W3
from spectraxgk.runtime import _build_initial_condition, run_runtime_linear, run_runtime_nonlinear
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
    assert params.z0 == np.pi
    assert params.nu_hyper == 2.0
    assert params.D_hyper == 5.0e-4
    assert params.dealias_kz is True


def test_cetg_linear_omega_max_matches_legacy_gx_formula() -> None:
    cfg = _base_cetg_cfg()
    geom = SlabGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    params = build_cetg_model_params(cfg, geom, Nl=2, Nm=1)

    ky_max = float(grid.ky[(grid.ky.size - 1) // 3])
    assert cfg.geometry.z0 is not None
    kz_max = float(grid.z.size) / 3.0 / float(cfg.geometry.z0) * float(geom.gradpar())
    cfac = 0.5 * float(params.c1) * np.sqrt(1.0 + (float(params.C12) - 1.0))

    assert np.isclose(_cetg_linear_omega_max(grid, params), cfac * np.sqrt(ky_max) * kz_max)


def test_cetg_field_solve_matches_gx_tau_bar_and_kz_dealias_scale() -> None:
    cfg = _base_cetg_cfg()
    geom = SlabGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    params = build_cetg_model_params(cfg, geom, Nl=2, Nm=1)

    density = np.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    temperature = np.zeros_like(density)
    G = np.stack([density, temperature], axis=0)[None, :, None, :, :, :]

    fields = cetg_fields(jnp.asarray(G), grid, params)

    mask = np.asarray(grid.dealias_mask, dtype=bool)
    phi = np.asarray(fields.phi)
    assert np.allclose(phi[mask], -float(grid.z.size))
    assert np.allclose(phi[~mask], 0.0)


def test_runtime_linear_cetg_smoke_uses_model_native_dims() -> None:
    cfg = replace(_base_cetg_cfg(), physics=replace(_base_cetg_cfg().physics, linear=True, nonlinear=False))

    out = run_runtime_linear(cfg, ky_target=1.0 / 2.0, solver="time", steps=4, sample_stride=1, return_state=True)

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)
    assert out.state is not None
    assert out.state.shape[0:3] == (1, 2, 1)


def test_runtime_nonlinear_cetg_smoke_uses_model_native_dims() -> None:
    base = _base_cetg_cfg()
    cfg = replace(
        base,
        time=replace(base.time, dt=1.0e-4, t_max=4.0e-4, fixed_dt=True),
        init=replace(base.init, init_amp=1.0e-6),
    )

    out = run_runtime_nonlinear(cfg, ky_target=1.0 / 2.0, kx_target=0.0, steps=4, sample_stride=1, return_state=True)

    assert out.diagnostics is not None
    assert out.state is not None
    assert out.state.shape[0:3] == (1, 2, 1)
    assert np.all(np.isfinite(np.asarray(out.diagnostics.Wg_t)))
    assert np.allclose(np.asarray(out.diagnostics.Wapar_t), 0.0)
    assert np.allclose(np.asarray(out.diagnostics.particle_flux_t), 0.0)


def test_runtime_nonlinear_cetg_adaptive_steps_to_tmax() -> None:
    base = _base_cetg_cfg()
    cfg = replace(
        base,
        time=replace(base.time, dt=0.01, t_max=0.015, fixed_dt=False, sample_stride=1, diagnostics_stride=1),
        init=replace(base.init, init_amp=1.0e-6),
    )

    out = run_runtime_nonlinear(cfg, ky_target=0.5, kx_target=0.0, steps=None, sample_stride=1, return_state=True)

    assert out.diagnostics is not None
    t = np.asarray(out.diagnostics.t)
    assert t.size >= 2
    assert float(t[-1]) >= float(cfg.time.t_max) - 1.0e-6


def test_cetg_sspx3_scan_matches_manual_one_step_with_carried_startup_field() -> None:
    base = _base_cetg_cfg()
    cfg = replace(base, time=replace(base.time, fixed_dt=False))
    geom = SlabGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    params = build_cetg_model_params(cfg, geom, Nl=2, Nm=1)
    terms = TermConfig(
        streaming=float(cfg.terms.streaming),
        mirror=float(cfg.terms.mirror),
        curvature=float(cfg.terms.curvature),
        gradb=float(cfg.terms.gradb),
        diamagnetic=float(cfg.terms.diamagnetic),
        collisions=float(cfg.terms.collisions),
        hypercollisions=float(cfg.terms.hypercollisions),
        hyperdiffusion=float(cfg.terms.hyperdiffusion),
        end_damping=float(cfg.terms.end_damping),
        apar=float(cfg.terms.apar),
        bpar=float(cfg.terms.bpar),
        nonlinear=float(cfg.terms.nonlinear),
    )

    g0_state = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=1,
            nspecies=1,
        ),
        dtype=np.complex64,
    )

    G0 = _project_state(_to_internal_state(jnp.asarray(g0_state)), grid, gx_real_fft=True)
    fields0 = cetg_fields(G0, grid, params, apply_kz_dealias=False)

    def euler_step(G_state: jnp.ndarray, fields_state) -> jnp.ndarray:
        rhs, _ = cetg_rhs(G_state, grid, params, terms, gx_real_fft=True, fields_override=fields_state)
        return _project_state(G_state + (_SSPX3_ADT * float(cfg.time.dt)) * rhs, grid, gx_real_fft=True)

    G1 = euler_step(G0, fields0)
    G2_euler = euler_step(G1, cetg_fields(G1, grid, params))
    G2 = _project_state(
        (1.0 - _SSPX3_W1) * G0 + (_SSPX3_W1 - 1.0) * G1 + G2_euler,
        grid,
        gx_real_fft=True,
    )
    G3 = euler_step(G2, cetg_fields(G2, grid, params))
    G_manual = _project_state(
        (1.0 - _SSPX3_W2 - _SSPX3_W3) * G0
        + _SSPX3_W3 * G1
        + (_SSPX3_W2 - 1.0) * G2
        + G3,
        grid,
        gx_real_fft=True,
    )

    _t, diag, G_scan, _fields = integrate_cetg_gx_diagnostics_state(
        jnp.asarray(g0_state),
        grid,
        params,
        terms,
        dt=float(cfg.time.dt),
        steps=1,
        method="sspx3",
        sample_stride=1,
        diagnostics_stride=1,
        gx_real_fft=True,
        omega_ky_index=1,
        omega_kx_index=0,
        fixed_dt=False,
        dt_min=float(cfg.time.dt_min),
        dt_max=cfg.time.dt_max,
        cfl=float(cfg.time.cfl),
        cfl_fac=cfg.time.cfl_fac,
    )

    assert np.allclose(np.asarray(diag.dt_t), np.asarray([float(cfg.time.dt)]))
    assert np.allclose(np.asarray(G_scan)[0, :, 0], np.asarray(G_manual), rtol=1.0e-6, atol=1.0e-6)
