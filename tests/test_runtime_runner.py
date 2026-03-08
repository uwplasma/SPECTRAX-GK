"""Tests for unified runtime-configured linear runner."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.diagnostics import GXDiagnostics
from spectraxgk.geometry import SAlphaGeometry, sample_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.runtime import (
    _build_initial_condition,
    build_runtime_linear_params,
    build_runtime_linear_terms,
    run_runtime_linear,
    run_runtime_nonlinear,
    run_runtime_scan,
)
from spectraxgk.runtime_config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


def _base_runtime_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        time=TimeConfig(t_max=0.2, dt=0.01, method="rk2", use_diffrax=False, sample_stride=1),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(init_field="density", init_amp=1.0e-8, gaussian_init=False),
        terms=RuntimeTermsConfig(hypercollisions=0.0, end_damping=0.0),
    )


def test_runtime_linear_cyclone_etg_kbm_smoke() -> None:
    ion = RuntimeSpeciesConfig(
        name="ion", charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8
    )
    electron = RuntimeSpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0 / 3670.0,
        density=1.0,
        temperature=1.0,
        tprim=2.49,
        fprim=0.8,
    )

    cyclone = replace(
        _base_runtime_cfg(),
        species=(ion,),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    etg = replace(
        _base_runtime_cfg(),
        species=(electron,),
        normalization=RuntimeNormalizationConfig(contract="etg", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            adiabatic_ions=True,
            electrostatic=True,
            electromagnetic=False,
        ),
    )
    kbm = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=62.8, Ly=62.8, boundary="periodic"),
        init=InitializationConfig(init_field="all", init_amp=1.0e-8, gaussian_init=False),
        species=(ion, electron),
        normalization=RuntimeNormalizationConfig(contract="kbm", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            electrostatic=False,
            electromagnetic=True,
            use_apar=True,
            beta=0.2,
            hypercollisions=False,
        ),
    )

    for cfg, ky in ((cyclone, 0.2), (etg, 2.0), (kbm, 0.2)):
        res = run_runtime_linear(
            cfg,
            ky_target=ky,
            Nl=4,
            Nm=6,
            solver="krylov",
        )
        assert np.isfinite(res.gamma)
        assert np.isfinite(res.omega)


def test_runtime_terms_and_params_follow_toggles() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="gx"),
        physics=RuntimePhysicsConfig(
            electrostatic=False,
            electromagnetic=True,
            use_apar=True,
            use_bpar=False,
            adiabatic_electrons=True,
            tau_e=1.0,
            beta=0.1,
            collisions=False,
            hypercollisions=False,
        ),
    )
    params = build_runtime_linear_params(cfg, Nm=6)
    terms = build_runtime_linear_terms(cfg)
    assert float(params.beta) == 0.1
    assert float(params.fapar) == 1.0
    assert terms.apar == 1.0
    assert terms.bpar == 0.0
    assert terms.collisions == 0.0
    assert terms.hypercollisions == 0.0


def test_runtime_hypercollision_default_tracks_hermite_count() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        collisions=RuntimeCollisionConfig(),
    )
    params_nm8 = build_runtime_linear_params(cfg, Nm=8)
    params_nm16 = build_runtime_linear_params(cfg, Nm=16)
    assert float(params_nm8.p_hyper_m) == 4.0
    assert float(params_nm16.p_hyper_m) == 8.0


def test_runtime_end_damping_defaults_follow_gx_scaling() -> None:
    base = _base_runtime_cfg()
    cfg = replace(
        base,
        time=replace(base.time, dt=0.2),
        collisions=RuntimeCollisionConfig(),
    )
    params = build_runtime_linear_params(cfg, Nm=8)
    assert float(params.damp_ends_amp) == pytest.approx(0.5)
    assert float(params.damp_ends_widthfrac) == pytest.approx(0.125)


def test_runtime_hypercollision_explicit_override_is_preserved() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        collisions=RuntimeCollisionConfig(p_hyper_m=11.0),
    )
    params = build_runtime_linear_params(cfg, Nm=8)
    assert float(params.p_hyper_m) == 11.0


def test_runtime_scan_returns_arrays() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    scan = run_runtime_scan(
        cfg,
        ky_values=[0.1, 0.2],
        Nl=4,
        Nm=6,
        solver="krylov",
    )
    assert scan.ky.shape == (2,)
    assert scan.gamma.shape == (2,)
    assert scan.omega.shape == (2,)


def test_runtime_scan_batch_matches_serial() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
    )
    ky_vals = [0.1, 0.2]
    serial = run_runtime_scan(
        cfg,
        ky_values=ky_vals,
        Nl=4,
        Nm=6,
        solver="time",
        method="rk2",
        dt=0.01,
        steps=10,
        fit_signal="phi",
    )
    batched = run_runtime_scan(
        cfg,
        ky_values=ky_vals,
        Nl=4,
        Nm=6,
        solver="time",
        method="rk2",
        dt=0.01,
        steps=10,
        fit_signal="phi",
        batch_ky=True,
    )
    assert np.allclose(serial.gamma, batched.gamma, rtol=5.0e-2, atol=1.0e-8)
    assert np.allclose(serial.omega, batched.omega, rtol=5.0e-2, atol=1.0e-8)


def test_runtime_nonlinear_smoke() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.01, steps=3, sample_stride=1)
    assert res.diagnostics is not None
    assert res.diagnostics.t.size == 3


def test_runtime_nonlinear_diagnostics_stride() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        dt=0.01,
        steps=5,
        sample_stride=1,
        diagnostics_stride=2,
    )
    assert res.diagnostics is not None
    assert res.diagnostics.t.size == 3


def test_runtime_nonlinear_disable_diagnostics() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        dt=0.01,
        steps=3,
        diagnostics=False,
    )
    assert res.diagnostics is None
    assert res.phi2 is not None


def test_runtime_nonlinear_adaptive_dt() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.2,
            dt=0.01,
            method="rk2",
            use_diffrax=False,
            sample_stride=1,
            fixed_dt=False,
            dt_min=1.0e-5,
            dt_max=0.02,
            cfl=0.5,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.01, steps=4)
    assert res.diagnostics is not None
    t_arr = np.asarray(res.diagnostics.t)
    assert np.all(np.diff(t_arr) > 0)


def test_runtime_gaussian_init_populates_multiple_modes_when_not_single() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=6,
            Ny=8,
            Nz=16,
            Lx=6.28,
            Ly=6.28,
            boundary="linked",
            y0=10.0,
            ntheta=16,
            nperiod=1,
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-8,
            gaussian_init=True,
            gaussian_width=0.5,
            init_single=False,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )
    )
    amp_kykx = np.max(np.abs(g0[0, 0, 0, ...]), axis=-1)
    nonzero = amp_kykx > 0.0
    assert int(np.count_nonzero(nonzero)) > 1


def test_runtime_nonlinear_dealias_toggle_executes() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.03,
            dt=0.01,
            method="rk2",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            nonlinear_dealias=False,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, steps=3)
    assert res.diagnostics is not None
    assert np.all(np.isfinite(res.diagnostics.Wphi_t))


def test_runtime_init_species_targets_all_vs_electrons_only() -> None:
    cfg_all = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        species=(
            RuntimeSpeciesConfig(name="ion", charge=1.0),
            RuntimeSpeciesConfig(name="electron", charge=-1.0, mass=1.0 / 3670.0),
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-8,
            gaussian_init=False,
            init_single=True,
            init_electrons_only=False,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg_all.geometry)
    grid = build_spectral_grid(cfg_all.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g_all = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg_all,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=2,
        )
    )
    assert np.max(np.abs(g_all[0])) > 0.0
    assert np.max(np.abs(g_all[1])) > 0.0

    cfg_e = replace(
        cfg_all,
        init=replace(cfg_all.init, init_electrons_only=True),
    )
    g_e = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg_e,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=2,
        )
    )
    assert np.max(np.abs(g_e[0])) == 0.0
    assert np.max(np.abs(g_e[1])) > 0.0


def test_runtime_initial_condition_accepts_sampled_geometry_contract() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-8,
            gaussian_init=False,
            init_single=True,
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = sample_flux_tube_geometry(SAlphaGeometry.from_config(cfg.geometry), grid.z)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=1,
        )
    )
    assert g0.shape == (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    assert np.all(np.isfinite(g0))


def test_runtime_linear_accepts_gx_netcdf_geometry(tmp_path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    grid = build_spectral_grid(cfg.grid)
    theta = np.asarray(grid.z, dtype=float)
    path = tmp_path / "geom.out.nc"
    with Dataset(path, "w") as root:
        root.createDimension("theta", theta.size)
        grids = root.createGroup("Grids")
        geom = root.createGroup("Geometry")
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        analytic = SAlphaGeometry.from_config(cfg.geometry)
        sampled = sample_flux_tube_geometry(analytic, grid.z)
        for name, values in {
            "bmag": np.asarray(sampled.bmag_profile),
            "bgrad": np.asarray(sampled.bgrad_profile),
            "gds2": np.asarray(sampled.gds2_profile),
            "gds21": np.asarray(sampled.gds21_profile),
            "gds22": np.asarray(sampled.gds22_profile),
            "cvdrift": np.asarray(sampled.cv_profile),
            "gbdrift": np.asarray(sampled.gb_profile),
            "cvdrift0": np.asarray(sampled.cv0_profile),
            "gbdrift0": np.asarray(sampled.gb0_profile),
            "jacobian": np.asarray(sampled.jacobian_profile),
            "grho": np.asarray(sampled.grho_profile),
        }.items():
            geom.createVariable(name, "f8", ("theta",))[:] = values
        for name, value in {
            "gradpar": sampled.gradpar_value,
            "q": sampled.q,
            "shat": sampled.s_hat,
            "rmaj": sampled.R0,
            "aminor": sampled.epsilon * sampled.R0,
        }.items():
            geom.createVariable(name, "f8", ())[:] = value

    cfg_nc = replace(cfg, geometry=replace(cfg.geometry, model="gx-netcdf", geometry_file=str(path)))
    out = run_runtime_linear(cfg_nc, ky_target=0.2, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_accepts_root_level_gx_eik_geometry(tmp_path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    grid = build_spectral_grid(cfg.grid)
    theta = np.asarray(grid.z, dtype=float)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, grid.z)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.asarray(sampled.bmag_profile)
        root.createVariable("gds2", "f8", ("z",))[:] = np.asarray(sampled.gds2_profile)
        root.createVariable("gds21", "f8", ("z",))[:] = np.asarray(sampled.gds21_profile)
        root.createVariable("gds22", "f8", ("z",))[:] = np.asarray(sampled.gds22_profile)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.asarray(sampled.cv_profile)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.asarray(sampled.gb_profile)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.asarray(sampled.cv0_profile)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.asarray(sampled.gb0_profile)
        root.createVariable("jacob", "f8", ("z",))[:] = np.asarray(sampled.jacobian_profile)
        root.createVariable("grho", "f8", ("z",))[:] = np.asarray(sampled.grho_profile)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, sampled.gradpar_value)
        root.createVariable("q", "f8", ())[:] = sampled.q
        root.createVariable("shat", "f8", ())[:] = sampled.s_hat
        root.createVariable("Rmaj", "f8", ())[:] = sampled.R0
        root.createVariable("kxfac", "f8", ())[:] = sampled.kxfac
        root.createVariable("scale", "f8", ())[:] = sampled.theta_scale
        root.createVariable("nfp", "f8", ())[:] = sampled.nfp
        root.createVariable("alpha", "f8", ())[:] = sampled.alpha

    cfg_nc = replace(cfg, geometry=replace(cfg.geometry, model="gx-netcdf", geometry_file=str(path)))
    out = run_runtime_linear(cfg_nc, ky_target=0.2, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_init_all_applies_gx_moment_scaling_single_mode() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        init=InitializationConfig(
            init_field="all",
            init_amp=1.0,
            gaussian_init=False,
            init_single=True,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=1,
        )
    )[0]

    density = g0[0, 0, ky_index, 0, 0]
    tpar = g0[0, 2, ky_index, 0, 0]
    qpar = g0[0, 3, ky_index, 0, 0]
    assert np.allclose(tpar / density, 1.0 / np.sqrt(2.0))
    assert np.allclose(qpar / density, 1.0 / np.sqrt(6.0))


def test_runtime_init_all_applies_gx_moment_scaling_multimode_gaussian() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=6,
            Ny=8,
            Nz=16,
            Lx=6.28,
            Ly=6.28,
            boundary="linked",
            y0=10.0,
            ntheta=16,
            nperiod=1,
        ),
        init=InitializationConfig(
            init_field="all",
            init_amp=1.0,
            gaussian_init=True,
            gaussian_width=0.5,
            init_single=False,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=1,
        )
    )[0]

    amp_density = np.max(np.abs(g0[0, 0, ...]))
    amp_tpar = np.max(np.abs(g0[0, 2, ...]))
    amp_qpar = np.max(np.abs(g0[0, 3, ...]))
    assert amp_density > 0.0
    assert np.isclose(amp_tpar / amp_density, 1.0 / np.sqrt(2.0))
    assert np.isclose(amp_qpar / amp_density, 1.0 / np.sqrt(6.0))


def test_runtime_init_all_applies_gx_moment_scaling_multimode_random() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="all",
            init_amp=1.0,
            gaussian_init=False,
            init_single=False,
            random_seed=7,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=1,
        )
    )[0]

    amp_density = np.max(np.abs(g0[0, 0, ...]))
    amp_tpar = np.max(np.abs(g0[0, 2, ...]))
    amp_qpar = np.max(np.abs(g0[0, 3, ...]))
    assert amp_density > 0.0
    assert np.isclose(amp_tpar / amp_density, 1.0 / np.sqrt(2.0))
    assert np.isclose(amp_qpar / amp_density, 1.0 / np.sqrt(6.0))


def test_runtime_nonlinear_mode_selection_respects_dealias(monkeypatch) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=3,
            Ny=7,
            Nz=16,
            Lx=62.8,
            Ly=62.8,
            boundary="periodic",
            y0=10.0,
        ),
        time=TimeConfig(
            t_max=0.02,
            dt=0.01,
            method="rk2",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            nonlinear_dealias=True,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    captured: dict[str, int] = {}

    def _fake_integrator(
        G0,
        grid,
        geom,
        params,
        *,
        dt,
        steps,
        method,
        terms,
        sample_stride,
        diagnostics_stride,
        use_dealias_mask,
        z_index=None,
        gx_real_fft=True,
        laguerre_mode="grid",
        omega_ky_index=None,
        omega_kx_index=0,
        flux_scale=1.0,
        wphi_scale=1.0,
        fixed_dt=True,
        dt_min=1.0e-7,
        dt_max=None,
        cfl=0.9,
        cfl_fac=1.0,
        collision_split=True,
        collision_scheme="strang",
        implicit_tol=1.0e-6,
        implicit_maxiter=120,
        implicit_iters=3,
        implicit_relax=0.7,
        implicit_restart=20,
        implicit_solve_method="batched",
        implicit_preconditioner=None,
    ):
        captured["omega_ky_index"] = int(omega_ky_index)
        captured["omega_kx_index"] = int(omega_kx_index)
        t = np.asarray([float(dt)], dtype=float)
        zeros = np.zeros_like(t)
        diag = GXDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(dt),
            gamma_t=zeros,
            omega_t=zeros,
            Wg_t=zeros,
            Wphi_t=zeros,
            Wapar_t=zeros,
            heat_flux_t=zeros,
            particle_flux_t=zeros,
            energy_t=zeros,
        )
        return t, diag

    monkeypatch.setattr("spectraxgk.runtime.integrate_nonlinear_gx_diagnostics", _fake_integrator)
    _res = run_runtime_nonlinear(cfg, ky_target=0.3, Nl=3, Nm=4, steps=1)

    # Ny=7, y0=10 -> ky = [0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1]
    # 2/3 dealias keeps |n|<=2 => ky=0.3 is excluded; nearest kept mode is ky=0.2 (index 2).
    assert captured["omega_ky_index"] == 2
    # Nx=3 FFT order puts kx=0 at index 0.
    assert captured["omega_kx_index"] == 0
