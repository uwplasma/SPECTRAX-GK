"""Tests for unified runtime-configured linear runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.diagnostics import SimulationDiagnostics, ResolvedDiagnostics
from spectraxgk.geometry import SAlphaGeometry, apply_gx_geometry_grid_defaults, sample_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.runtime import (
    _build_initial_condition,
    _gx_centered_random_pairs,
    _gx_init_mode_pairs,
    _gx_periodic_zp,
    _infer_runtime_nonlinear_steps,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
    run_linear_case,
    run_nonlinear_case,
    run_runtime_linear,
    run_runtime_nonlinear,
    run_runtime_scan,
)
from spectraxgk.runtime_config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeExpertConfig,
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


def _gx_c_rand_pairs(seed: int, count: int) -> np.ndarray:
    rand_max = float((1 << 31) - 1)
    seed_use = 1 if int(seed) == 0 else int(seed)
    state = np.zeros(344 + 2 * count, dtype=np.uint64)
    state[0] = np.uint64(seed_use)
    for i in range(1, 31):
        state[i] = np.uint64((16807 * int(state[i - 1])) % int(rand_max))
    for i in range(31, 34):
        state[i] = state[i - 31]
    for i in range(34, state.size):
        state[i] = (state[i - 31] + state[i - 3]) & np.uint64(0xFFFFFFFF)
    rand_vals = (state[344:] >> np.uint64(1)).astype(float, copy=False)
    half = 0.5 * rand_max
    inv = 1.0 / rand_max
    out = np.empty((count, 2), dtype=float)
    for i in range(count):
        out[i, 0] = (rand_vals[2 * i] - half) * inv
        out[i, 1] = (rand_vals[2 * i + 1] - half) * inv
    return out


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


def test_runtime_linear_etg_defaults_to_frequency_targeted_krylov(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectraxgk.runtime as runtime

    electron = RuntimeSpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0 / 3670.0,
        density=1.0,
        temperature=1.0,
        tprim=2.49,
        fprim=0.8,
    )
    cfg = replace(
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

    captured: dict[str, object] = {}

    def fake_dominant_eigenpair(_g0, _cache, _params, terms=None, **kwargs):
        captured["terms"] = terms
        captured.update(kwargs)
        return np.asarray(0.2 + 0.3j, dtype=np.complex64), np.zeros_like(np.asarray(_g0))

    monkeypatch.setattr(runtime, "dominant_eigenpair", fake_dominant_eigenpair)

    out = run_runtime_linear(cfg, ky_target=2.0, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert captured["method"] == "shift_invert"
    assert captured["omega_target_factor"] == pytest.approx(0.4)
    assert captured["omega_sign"] == -1
    assert captured["shift_source"] == "target"
    assert captured["shift_selection"] == "targeted"
    assert captured["mode_family"] == "etg"


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


def test_runtime_end_damping_defaults_follow_gx_rate_contract() -> None:
    base = _base_runtime_cfg()
    cfg = replace(
        base,
        time=replace(base.time, dt=0.2),
        collisions=RuntimeCollisionConfig(),
    )
    params = build_runtime_linear_params(cfg, Nm=8)
    assert float(params.damp_ends_amp) == pytest.approx(0.1)
    assert float(params.damp_ends_widthfrac) == pytest.approx(0.125)


def test_runtime_end_damping_can_explicitly_scale_by_dt() -> None:
    base = _base_runtime_cfg()
    cfg = replace(
        base,
        time=replace(base.time, dt=0.2),
        collisions=RuntimeCollisionConfig(damp_ends_scale_by_dt=True),
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


def test_runtime_linear_time_solver_can_return_state() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
    )
    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        method="sspx3",
        dt=0.01,
        steps=2,
        return_state=True,
    )
    assert res.state is not None
    assert res.state.ndim == 6
    assert res.state.shape[:3] == (1, 3, 4)
    assert res.state.shape[-1] == cfg.grid.Nz


def test_runtime_linear_records_fit_window_metadata() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
    )
    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        method="rk2",
        dt=0.01,
        steps=10,
        fit_signal="phi",
        auto_window=False,
        tmin=0.02,
        tmax=0.08,
    )
    assert res.fit_signal_used == "phi"
    assert res.fit_window_tmin == pytest.approx(0.02)
    assert res.fit_window_tmax == pytest.approx(0.08)


def test_runtime_linear_progress_with_sample_stride_gt_one() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        time=replace(_base_runtime_cfg().time, sample_stride=2),
    )
    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        method="rk2",
        dt=0.01,
        steps=10,
        sample_stride=2,
        show_progress=True,
    )
    assert np.isfinite(res.gamma)
    assert np.isfinite(res.omega)


def test_runtime_linear_gx_time_rejects_return_state() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
    )
    with pytest.raises(ValueError, match="return_state"):
        run_runtime_linear(
            cfg,
            ky_target=0.1,
            Nl=3,
            Nm=4,
            solver="gx_time",
            method="rk4",
            dt=0.01,
            steps=2,
            return_state=True,
        )


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


def test_runtime_nonlinear_em_flux_channels_sum_to_total() -> None:
    ion = RuntimeSpeciesConfig(
        name="ion",
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=1.0,
        fprim=1.0,
    )
    electron = RuntimeSpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0 / 3670.0,
        density=1.0,
        temperature=1.0,
        tprim=1.0,
        fprim=1.0,
    )
    cfg = replace(
        _base_runtime_cfg(),
        species=(ion, electron),
        normalization=RuntimeNormalizationConfig(contract="kbm", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            electrostatic=False,
            electromagnetic=True,
            use_apar=True,
            use_bpar=True,
            nonlinear=True,
            beta=0.02,
            collisions=False,
            hypercollisions=False,
        ),
        init=InitializationConfig(init_field="all", init_amp=1.0e-8, gaussian_init=False),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0, apar=1.0, bpar=1.0),
    )

    res = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.01, steps=2, sample_stride=1, return_state=True)

    assert res.diagnostics is not None
    assert isinstance(res.diagnostics.resolved, ResolvedDiagnostics)
    resolved = res.diagnostics.resolved
    assert resolved is not None
    for total, es, apar, bpar in (
        (resolved.HeatFlux_kxst, resolved.HeatFluxES_kxst, resolved.HeatFluxApar_kxst, resolved.HeatFluxBpar_kxst),
        (resolved.HeatFlux_kyst, resolved.HeatFluxES_kyst, resolved.HeatFluxApar_kyst, resolved.HeatFluxBpar_kyst),
        (resolved.HeatFlux_kxkyst, resolved.HeatFluxES_kxkyst, resolved.HeatFluxApar_kxkyst, resolved.HeatFluxBpar_kxkyst),
        (resolved.HeatFlux_zst, resolved.HeatFluxES_zst, resolved.HeatFluxApar_zst, resolved.HeatFluxBpar_zst),
        (resolved.ParticleFlux_kxst, resolved.ParticleFluxES_kxst, resolved.ParticleFluxApar_kxst, resolved.ParticleFluxBpar_kxst),
        (resolved.ParticleFlux_kyst, resolved.ParticleFluxES_kyst, resolved.ParticleFluxApar_kyst, resolved.ParticleFluxBpar_kyst),
        (resolved.ParticleFlux_kxkyst, resolved.ParticleFluxES_kxkyst, resolved.ParticleFluxApar_kxkyst, resolved.ParticleFluxBpar_kxkyst),
        (resolved.ParticleFlux_zst, resolved.ParticleFluxES_zst, resolved.ParticleFluxApar_zst, resolved.ParticleFluxBpar_zst),
    ):
        assert total is not None
        assert es is not None
        assert apar is not None
        assert bpar is not None
        np.testing.assert_allclose(np.asarray(total), np.asarray(es) + np.asarray(apar) + np.asarray(bpar), rtol=1.0e-5, atol=1.0e-6)

    assert res.diagnostics.turbulent_heating_species_t is not None
    turb_heat_s = np.asarray(res.diagnostics.turbulent_heating_species_t)
    assert resolved.TurbulentHeating_kxst is not None
    assert resolved.TurbulentHeating_kyst is not None
    assert resolved.TurbulentHeating_kxkyst is not None
    assert resolved.TurbulentHeating_zst is not None
    np.testing.assert_allclose(np.asarray(resolved.TurbulentHeating_kxst).sum(axis=2), turb_heat_s, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(resolved.TurbulentHeating_kyst).sum(axis=2), turb_heat_s, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(resolved.TurbulentHeating_kxkyst).sum(axis=(2, 3)), turb_heat_s, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(resolved.TurbulentHeating_zst).sum(axis=2), turb_heat_s, rtol=1.0e-5, atol=1.0e-6)


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


def test_runtime_init_file_replace_mode_scales_loaded_state(tmp_path) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-8,
            gaussian_init=False,
            init_file=str(tmp_path / "restart.bin"),
            init_file_scale=2.5,
            init_file_mode="replace",
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    raw = (1.0 + 2.0j) * np.ones((1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    raw.tofile(cfg.init.init_file)

    g0 = _build_initial_condition(grid, geom, cfg, ky_index=1, kx_index=0, Nl=3, Nm=4, nspecies=1)

    assert np.allclose(np.asarray(g0), 2.5 * raw)


def test_runtime_init_file_add_mode_adds_seed_perturbation(tmp_path) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=0.25,
            gaussian_init=False,
            init_single=True,
            init_file=str(tmp_path / "restart.bin"),
            init_file_scale=3.0,
            init_file_mode="add",
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    base = np.ones((1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    base.tofile(cfg.init.init_file)

    g0 = np.asarray(_build_initial_condition(grid, geom, cfg, ky_index=1, kx_index=0, Nl=3, Nm=4, nspecies=1))

    assert np.allclose(g0[0, 1:, :, :, :, :], 3.0)
    assert np.allclose(g0[0, 0, 1:, :, :, :], 3.0)
    assert np.allclose(g0[0, 0, 0, 0, :, :], 3.0)
    assert np.allclose(g0[0, 0, 0, 1, 1:, :], 3.0)
    expected = 3.0 + 0.25
    assert np.allclose(g0[0, 0, 0, 1, 0, :], expected)


def test_runtime_single_mode_init_matches_gx_real_phase() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        init=InitializationConfig(
            init_field="density",
            init_amp=0.25,
            gaussian_init=False,
            init_single=True,
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))

    g0 = np.asarray(_build_initial_condition(grid, geom, cfg, ky_index=ky_index, kx_index=0, Nl=3, Nm=4, nspecies=1))

    seeded = g0[0, 0, 0, ky_index, 0, :]
    assert np.allclose(seeded.real, 0.25)
    assert np.allclose(seeded.imag, 0.0)


def test_runtime_single_mode_init_applies_gx_kpar_phase() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        init=InitializationConfig(
            init_field="density",
            init_amp=0.25,
            gaussian_init=False,
            init_single=True,
            kpar_init=2.0,
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))

    g0 = np.asarray(_build_initial_condition(grid, geom, cfg, ky_index=ky_index, kx_index=0, Nl=3, Nm=4, nspecies=1))

    z = np.asarray(grid.z, dtype=float)
    z_period = _gx_periodic_zp(z)
    expected = 0.25 * np.cos(2.0 * z / z_period)
    seeded = g0[0, 0, 0, ky_index, 0, :]
    assert np.allclose(seeded.real, expected, atol=1.0e-6)
    assert np.allclose(seeded.imag, 0.0)


def test_runtime_nonlinear_fixed_mode_returns_frozen_state() -> None:
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"))
    ky_fixed = float(np.asarray(grid.ky)[1])
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
        expert=RuntimeExpertConfig(fixed_mode=True, iky_fixed=1, ikx_fixed=0),
    )
    res = run_runtime_nonlinear(
        cfg,
        ky_target=ky_fixed,
        Nl=3,
        Nm=4,
        dt=0.01,
        steps=3,
        sample_stride=1,
        return_state=True,
    )

    assert res.state is not None
    initial = _build_initial_condition(
        build_spectral_grid(cfg.grid),
        SAlphaGeometry.from_config(cfg.geometry),
        cfg,
        ky_index=1,
        kx_index=0,
        Nl=3,
        Nm=4,
        nspecies=1,
    )
    assert np.allclose(np.asarray(res.state)[..., 1, 0, :], np.asarray(initial)[..., 1, 0, :])


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


def test_runtime_nonlinear_adaptive_default_steps_match_integrator_dt_cap() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.2,
            dt=0.01,
            method="rk3",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=False,
            dt_min=1.0e-5,
            dt_max=None,
            cfl=1.0,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    steps_val = _infer_runtime_nonlinear_steps(cfg, dt=0.01, steps=None)

    assert steps_val == 20


def test_runtime_nonlinear_adaptive_default_steps_chunk_until_tmax(monkeypatch) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.25,
            dt=0.1,
            method="rk3",
            use_diffrax=False,
            sample_stride=2,
            diagnostics_stride=2,
            fixed_dt=False,
            dt_min=1.0e-5,
            dt_max=None,
            cfl=1.0,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    calls: list[tuple[int, int, int]] = []

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
        fixed_mode_ky_index=None,
        fixed_mode_kx_index=None,
    ):
        calls.append((int(steps), int(sample_stride), int(diagnostics_stride)))
        t = np.asarray([0.04, 0.08, 0.12], dtype=float)
        dt_t = np.asarray([0.04, 0.04, 0.04], dtype=float)
        gamma_t = np.asarray([1.0, 2.0, 3.0], dtype=float) + 3.0 * (len(calls) - 1)
        zeros = np.zeros_like(t)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=dt_t,
            dt_mean=float(np.mean(dt_t)),
            gamma_t=gamma_t,
            omega_t=zeros,
            Wg_t=gamma_t,
            Wphi_t=zeros,
            Wapar_t=zeros,
            heat_flux_t=zeros,
            particle_flux_t=zeros,
            energy_t=gamma_t,
        )
        return t, diag, np.asarray(G0), None

    monkeypatch.setattr("spectraxgk.runtime.integrate_nonlinear_gx_diagnostics_state", _fake_integrator)

    res = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.1, steps=None)

    assert res.diagnostics is not None
    assert calls == [(3, 1, 1), (3, 1, 1), (3, 1, 1)]
    assert np.allclose(np.asarray(res.diagnostics.t), np.asarray([0.04, 0.12, 0.20, 0.28]))
    assert np.allclose(np.asarray(res.diagnostics.gamma_t), np.asarray([1.0, 3.0, 5.0, 7.0]))
    assert float(np.asarray(res.diagnostics.t)[-1]) >= float(cfg.time.t_max)


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


def test_runtime_gaussian_single_mode_keeps_gx_equal_real_imag_parts() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=4, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="linked", y0=10.0, ntheta=16, nperiod=1),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=True,
            gaussian_width=0.5,
            init_single=True,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))

    g0 = np.asarray(_build_initial_condition(grid, geom, cfg, ky_index=ky_index, kx_index=0, Nl=3, Nm=4, nspecies=1))

    seeded = g0[0, 0, 0, ky_index, 0, :]
    assert np.allclose(seeded.real, seeded.imag)


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


def test_runtime_nonlinear_uses_gx_method_default_cfl_fac(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}

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
        fixed_mode_ky_index=None,
        fixed_mode_kx_index=None,
    ):
        captured["cfl_fac"] = float(cfl_fac)
        t = np.asarray([0.1], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(t[0]),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return t, diag, np.asarray(G0), None

    monkeypatch.setattr("spectraxgk.runtime.integrate_nonlinear_gx_diagnostics_state", _fake_integrator)

    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.1,
            dt=0.1,
            method="rk3",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=False,
            cfl=1.0,
            cfl_fac=None,
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, steps=1)

    assert captured["cfl_fac"] == pytest.approx(1.73)


def test_runtime_nonlinear_preserves_explicit_cfl_fac(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}

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
        fixed_mode_ky_index=None,
        fixed_mode_kx_index=None,
    ):
        captured["cfl_fac"] = float(cfl_fac)
        t = np.asarray([0.1], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(t[0]),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return t, diag, np.asarray(G0), None

    monkeypatch.setattr("spectraxgk.runtime.integrate_nonlinear_gx_diagnostics_state", _fake_integrator)

    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.1,
            dt=0.1,
            method="rk3",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=False,
            cfl=1.0,
            cfl_fac=1.25,
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, steps=1)

    assert captured["cfl_fac"] == pytest.approx(1.25)


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
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
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


def test_runtime_linear_explicit_time_accepts_root_level_gx_eik_geometry(tmp_path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        time=replace(_base_runtime_cfg().time, dt=0.02, t_max=0.08, sample_stride=1),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
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
    out = run_runtime_linear(cfg_nc, ky_target=0.2, Nl=4, Nm=6, solver="explicit_time")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_accepts_vmec_model_via_generated_eik(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / "generated_geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
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

    monkeypatch.setattr("spectraxgk.runtime.generate_runtime_vmec_eik", lambda cfg: path)

    cfg_vmec = replace(
        cfg,
        geometry=replace(
            cfg.geometry,
            model="vmec",
            vmec_file=str(tmp_path / "wout_stub.nc"),
            geometry_file=str(path),
            torflux=0.64,
            npol=1.0,
        ),
    )
    out = run_runtime_linear(cfg_vmec, ky_target=0.2, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_secondary_slab_example_runs() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "examples" / "benchmarks" / "runtime_secondary_slab.toml"
    cfg, _ = load_runtime_from_toml(cfg_path)

    out = run_runtime_linear(cfg, ky_target=0.1, Nl=3, Nm=8, solver="explicit_time")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_accepts_miller_model_via_generated_eik(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = RuntimeConfig(
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="linked", y0=10.0, ntheta=16, nperiod=1),
        time=TimeConfig(t_max=0.2, dt=0.01, method="rk2", use_diffrax=False, sample_stride=1, fixed_dt=True),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(init_field="density", init_amp=1.0e-8, gaussian_init=False),
        terms=RuntimeTermsConfig(hypercollisions=0.0, end_damping=0.0),
        physics=RuntimePhysicsConfig(linear=True, nonlinear=False, adiabatic_electrons=True, collisions=False),
    )
    sampled = sample_flux_tube_geometry(SAlphaGeometry.from_config(cfg.geometry), build_spectral_grid(cfg.grid).z)
    path = tmp_path / "miller.eiknc.nc"
    theta = np.asarray(sampled.theta)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.asarray(sampled.bmag_profile)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.asarray(sampled.gb_profile)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.asarray(sampled.gb0_profile)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.asarray(sampled.cv_profile)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.asarray(sampled.cv0_profile)
        root.createVariable("gds2", "f8", ("z",))[:] = np.asarray(sampled.gds2_profile)
        root.createVariable("gds21", "f8", ("z",))[:] = np.asarray(sampled.gds21_profile)
        root.createVariable("gds22", "f8", ("z",))[:] = np.asarray(sampled.gds22_profile)
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

    monkeypatch.setattr("spectraxgk.runtime.generate_runtime_miller_eik", lambda cfg: path)

    cfg_miller = replace(
        cfg,
        geometry=replace(
            cfg.geometry,
            model="miller",
            geometry_file=str(path),
            rhoc=0.5,
            R_geo=cfg.geometry.R0,
        ),
    )
    out = run_runtime_linear(cfg_miller, ky_target=0.2, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_cetg_reference_example_runs_small_smoke() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "examples" / "nonlinear" / "axisymmetric" / "runtime_cetg_reference.toml"
    cfg, _ = load_runtime_from_toml(cfg_path)

    out = run_runtime_nonlinear(cfg, ky_target=1.0 / 6.366, kx_target=0.0, steps=2, sample_stride=1)

    assert out.diagnostics is not None
    assert np.all(np.isfinite(np.asarray(out.diagnostics.Wg_t)))
    assert np.allclose(np.asarray(out.diagnostics.Wapar_t), 0.0)


def test_runtime_etg_nonlinear_example_runs_small_smoke() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "examples" / "nonlinear" / "axisymmetric" / "runtime_etg_nonlinear.toml"
    cfg, _ = load_runtime_from_toml(cfg_path)

    # Keep the shipped ETG pilot contract intact, but reduce the test problem to
    # a CI-sized smoke so GitHub runners do not materialize the production-scale
    # nonlinear diagnostic buffers.
    cfg = replace(
        cfg,
        grid=GridConfig(
            Nx=4,
            Ny=4,
            Nz=8,
            Lx=6.28,
            Ly=6.28,
            boundary="linked",
            y0=0.2,
            ntheta=8,
            nperiod=1,
        ),
        time=replace(cfg.time, t_max=0.002, dt=0.001, sample_stride=1, diagnostics_stride=1, fixed_dt=True),
    )

    out = run_runtime_nonlinear(cfg, ky_target=5.0, kx_target=0.0, steps=2, sample_stride=1, Nl=2, Nm=2)

    assert out.diagnostics is not None
    assert np.all(np.isfinite(np.asarray(out.diagnostics.Wg_t)))
    assert np.all(np.isfinite(np.asarray(out.diagnostics.Wphi_t)))
    assert np.allclose(np.asarray(out.diagnostics.Wapar_t), 0.0)


def test_runtime_linear_gx_time_root_level_geometry_matches_analytic_reference(tmp_path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=32,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            z_min=-3.0 * np.pi,
            z_max=3.0 * np.pi,
        ),
        time=TimeConfig(
            t_max=0.08,
            dt=0.02,
            method="rk4",
            use_diffrax=False,
            sample_stride=1,
            fixed_dt=True,
        ),
        species=(RuntimeSpeciesConfig(name="ion", tprim=3.0, fprim=1.0),),
        normalization=RuntimeNormalizationConfig(contract="kinetic", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        terms=RuntimeTermsConfig(end_damping=0.0, hypercollisions=0.0),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.asarray(sampled.bmag_profile)
        root.createVariable("gds2", "f8", ("z",))[:] = np.asarray(sampled.gds2_profile)
        root.createVariable("gds21", "f8", ("z",))[:] = np.asarray(sampled.gds21_profile)
        root.createVariable("gds22", "f8", ("z",))[:] = np.asarray(sampled.gds22_profile)
        root.createVariable("cvdrift", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.cv_profile)
        root.createVariable("gbdrift", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.gb_profile)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.cv0_profile)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.gb0_profile)
        root.createVariable("jacob", "f8", ("z",))[:] = np.full(theta.size, 7.0)
        root.createVariable("grho", "f8", ("z",))[:] = np.asarray(sampled.grho_profile)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, sampled.gradpar_value)
        root.createVariable("drhodpsi", "f8", ())[:] = 1.0
        root.createVariable("q", "f8", ())[:] = sampled.q
        root.createVariable("shat", "f8", ())[:] = sampled.s_hat
        root.createVariable("Rmaj", "f8", ())[:] = sampled.R0
        root.createVariable("kxfac", "f8", ())[:] = sampled.kxfac
        root.createVariable("scale", "f8", ())[:] = sampled.theta_scale
        root.createVariable("nfp", "f8", ())[:] = sampled.nfp
        root.createVariable("alpha", "f8", ())[:] = sampled.alpha

    cfg_nc = replace(cfg, geometry=replace(cfg.geometry, model="gx-netcdf", geometry_file=str(path)))

    analytic_out = run_runtime_linear(cfg, ky_target=0.2, Nl=4, Nm=6, solver="gx_time")
    imported_out = run_runtime_linear(cfg_nc, ky_target=0.2, Nl=4, Nm=6, solver="gx_time")

    assert imported_out.gamma == pytest.approx(analytic_out.gamma, rel=2.0e-3, abs=1.0e-6)
    assert imported_out.omega == pytest.approx(analytic_out.omega, rel=2.0e-3, abs=1.0e-6)


def test_runtime_linear_gx_time_root_level_geometry_matches_analytic_reference_from_toml(tmp_path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 33)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, alpha=0.0)
    sampled = sample_flux_tube_geometry(analytic, theta)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.asarray(sampled.bmag_profile)
        root.createVariable("gds2", "f8", ("z",))[:] = np.asarray(sampled.gds2_profile)
        root.createVariable("gds21", "f8", ("z",))[:] = np.asarray(sampled.gds21_profile)
        root.createVariable("gds22", "f8", ("z",))[:] = np.asarray(sampled.gds22_profile)
        root.createVariable("cvdrift", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.cv_profile)
        root.createVariable("gbdrift", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.gb_profile)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.cv0_profile)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.gb0_profile)
        root.createVariable("jacob", "f8", ("z",))[:] = np.full(theta.size, 7.0)
        root.createVariable("grho", "f8", ("z",))[:] = np.asarray(sampled.grho_profile)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, sampled.gradpar_value)
        root.createVariable("drhodpsi", "f8", ())[:] = 1.0
        root.createVariable("q", "f8", ())[:] = sampled.q
        root.createVariable("shat", "f8", ())[:] = sampled.s_hat
        root.createVariable("Rmaj", "f8", ())[:] = sampled.R0
        root.createVariable("kxfac", "f8", ())[:] = sampled.kxfac
        root.createVariable("scale", "f8", ())[:] = sampled.theta_scale
        root.createVariable("nfp", "f8", ())[:] = sampled.nfp
        root.createVariable("alpha", "f8", ())[:] = sampled.alpha

    toml = f"""
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 3.0
fprim = 1.0
kinetic = true

[grid]
Nx = 1
Ny = 8
Nz = 32
Lx = 62.8
Ly = 62.8
boundary = "linked"
y0 = 10.0
z_min = {-3.0 * np.pi}
z_max = {3.0 * np.pi}

[time]
t_max = 0.08
dt = 0.02
method = "rk4"
use_diffrax = false
sample_stride = 1
fixed_dt = true

[geometry]
model = "gx-netcdf"
geometry_file = "{path}"

[physics]
adiabatic_electrons = true
tau_e = 1.0
electromagnetic = false

[terms]
end_damping = 0.0
hypercollisions = 0.0

[normalization]
contract = "kinetic"
diagnostic_norm = "none"
"""
    cfg_path = tmp_path / "runtime_w7x.toml"
    cfg_path.write_text(toml, encoding="utf-8")

    cfg_loaded, _ = load_runtime_from_toml(cfg_path)
    cfg_ref = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=32,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            z_min=-3.0 * np.pi,
            z_max=3.0 * np.pi,
        ),
        time=TimeConfig(
            t_max=0.08,
            dt=0.02,
            method="rk4",
            use_diffrax=False,
            sample_stride=1,
            fixed_dt=True,
        ),
        species=(RuntimeSpeciesConfig(name="ion", tprim=3.0, fprim=1.0),),
        normalization=RuntimeNormalizationConfig(contract="kinetic", diagnostic_norm="none"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        terms=RuntimeTermsConfig(end_damping=0.0, hypercollisions=0.0),
    )

    loaded_out = run_runtime_linear(cfg_loaded, ky_target=0.2, Nl=4, Nm=6, solver="gx_time")
    analytic_out = run_runtime_linear(cfg_ref, ky_target=0.2, Nl=4, Nm=6, solver="gx_time")

    assert loaded_out.gamma == pytest.approx(analytic_out.gamma, rel=2.0e-3, abs=1.0e-6)
    assert loaded_out.omega == pytest.approx(analytic_out.omega, rel=2.0e-3, abs=1.0e-6)


@pytest.mark.parametrize("model", ["vmec-eik", "desc-eik"])
def test_runtime_nonlinear_accepts_imported_eik_geometry_aliases(tmp_path, model: str) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=4,
            Ny=8,
            Nz=16,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            z_min=-3.0 * np.pi,
            z_max=3.0 * np.pi,
        ),
        time=TimeConfig(
            t_max=0.04,
            dt=0.02,
            method="rk3",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=True,
        ),
        species=(RuntimeSpeciesConfig(name="ion", tprim=3.0, fprim=1.0, nu=0.01),),
        normalization=RuntimeNormalizationConfig(contract="kinetic", diagnostic_norm="gx"),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            use_apar=False,
            use_bpar=False,
            collisions=False,
            hypercollisions=True,
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-6,
            gaussian_init=False,
            init_single=False,
        ),
        terms=RuntimeTermsConfig(
            end_damping=1.0,
            hypercollisions=1.0,
            hyperdiffusion=1.0,
            nonlinear=1.0,
        ),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / f"geom_{model}.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.asarray(sampled.bmag_profile)
        root.createVariable("gds2", "f8", ("z",))[:] = np.asarray(sampled.gds2_profile)
        root.createVariable("gds21", "f8", ("z",))[:] = np.asarray(sampled.gds21_profile)
        root.createVariable("gds22", "f8", ("z",))[:] = np.asarray(sampled.gds22_profile)
        root.createVariable("cvdrift", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.cv_profile)
        root.createVariable("gbdrift", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.gb_profile)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.cv0_profile)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = 2.0 * np.asarray(sampled.gb0_profile)
        root.createVariable("jacob", "f8", ("z",))[:] = np.full(theta.size, 7.0)
        root.createVariable("grho", "f8", ("z",))[:] = np.asarray(sampled.grho_profile)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, sampled.gradpar_value)
        root.createVariable("drhodpsi", "f8", ())[:] = 1.0
        root.createVariable("q", "f8", ())[:] = sampled.q
        root.createVariable("shat", "f8", ())[:] = sampled.s_hat
        root.createVariable("Rmaj", "f8", ())[:] = sampled.R0
        root.createVariable("kxfac", "f8", ())[:] = sampled.kxfac
        root.createVariable("scale", "f8", ())[:] = sampled.theta_scale
        root.createVariable("nfp", "f8", ())[:] = 5.0
        root.createVariable("alpha", "f8", ())[:] = sampled.alpha

    cfg_nc = replace(cfg, geometry=replace(cfg.geometry, model=model, geometry_file=str(path)))
    out = run_runtime_nonlinear(cfg_nc, ky_target=0.2, Nl=3, Nm=4, dt=0.02, steps=2, sample_stride=1)

    assert np.all(np.isfinite(out.t))
    assert out.diagnostics is not None
    assert np.isfinite(out.diagnostics.Wg_t[-1])
    assert np.isfinite(out.diagnostics.Wphi_t[-1])
    assert np.isfinite(out.diagnostics.heat_flux_t[-1])
    assert np.isfinite(out.diagnostics.particle_flux_t[-1])


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


def test_runtime_random_multimode_init_matches_gx_c_rand_sequence() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=False,
            init_single=False,
            random_seed=7,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=1,
            kx_index=0,
            Nl=1,
            Nm=1,
            nspecies=1,
        )
    )[0, 0, 0]

    z = np.asarray(grid.z, dtype=float)
    z_period = _gx_periodic_zp(z)
    z_phase = np.cos(float(cfg.init.kpar_init) * z / z_period)
    active_modes = _gx_init_mode_pairs(grid)
    expected = np.zeros_like(g0)
    for (kx_i, ky_i), (ra, rb) in zip(
        active_modes,
        cfg.init.init_amp * _gx_c_rand_pairs(int(cfg.init.random_seed), len(active_modes)),
        strict=True,
    ):
        vals = ((rb + 1j * ra) if kx_i == 0 else (ra + 1j * rb)) * z_phase
        expected[ky_i, kx_i, :] = vals
        if kx_i != 0:
            expected[ky_i, expected.shape[1] - kx_i, :] = (rb + 1j * ra) * z_phase

    assert np.allclose(g0, expected, atol=1.0e-6)


def test_runtime_gx_init_mode_pairs_match_gx_loop_bounds() -> None:
    grid = build_spectral_grid(GridConfig(Nx=96, Ny=96, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"))
    active_modes = _gx_init_mode_pairs(grid)

    assert active_modes[0] == (0, 1)
    assert active_modes[-1] == (31, 31)
    assert len(active_modes) == 32 * 31
    assert all(0 <= kx_i <= 31 for kx_i, _ in active_modes)
    assert all(1 <= ky_i <= 31 for _, ky_i in active_modes)


def test_runtime_gx_centered_random_pairs_match_glibc_reference() -> None:
    vals = _gx_centered_random_pairs(22, 5)
    ref = _gx_c_rand_pairs(22, 5)

    assert np.allclose(vals, ref)


def test_runtime_gx_periodic_zp_uses_discrete_period_not_endpoint_span() -> None:
    cfg, _data = load_runtime_from_toml("examples/nonlinear/axisymmetric/runtime_cetg_reference.toml")
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_gx_geometry_grid_defaults(geom, cfg.grid))
    z = np.asarray(grid.z, dtype=float)

    zp = _gx_periodic_zp(z)

    assert np.isclose(zp, 1.0, atol=2.0e-4)


def test_runtime_random_multimode_zero_kx_matches_gx_overwrite_order() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=False,
            init_single=False,
            random_seed=22,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=1,
            kx_index=0,
            Nl=1,
            Nm=1,
            nspecies=1,
        )
    )[0, 0, 0]

    ra, rb = _gx_c_rand_pairs(22, 1)[0]
    assert np.allclose(g0[1, 0, :], (rb + 1j * ra) * np.ones_like(g0[1, 0, :]))


def test_runtime_random_multimode_init_does_not_depend_on_diagnostic_ky() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=False,
            init_single=False,
            random_seed=7,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    g0_ky0 = np.asarray(_build_initial_condition(grid, geom, cfg, ky_index=0, kx_index=1, Nl=1, Nm=1, nspecies=1))
    g0_ky1 = np.asarray(_build_initial_condition(grid, geom, cfg, ky_index=1, kx_index=1, Nl=1, Nm=1, nspecies=1))

    assert np.allclose(g0_ky0, g0_ky1)


def test_runtime_gaussian_multimode_init_does_not_depend_on_diagnostic_ky() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=True,
            init_single=False,
            gaussian_width=0.35,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    g0_ky0 = np.asarray(_build_initial_condition(grid, geom, cfg, ky_index=0, kx_index=1, Nl=1, Nm=1, nspecies=1))
    g0_ky1 = np.asarray(_build_initial_condition(grid, geom, cfg, ky_index=1, kx_index=1, Nl=1, Nm=1, nspecies=1))

    assert np.allclose(g0_ky0, g0_ky1)


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
        fixed_mode_ky_index=None,
        fixed_mode_kx_index=None,
    ):
        captured["omega_ky_index"] = int(omega_ky_index)
        captured["omega_kx_index"] = int(omega_kx_index)
        t = np.asarray([float(dt)], dtype=float)
        zeros = np.zeros_like(t)
        diag = SimulationDiagnostics(
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
        return t, diag, np.asarray(G0), None

    monkeypatch.setattr("spectraxgk.runtime.integrate_nonlinear_gx_diagnostics_state", _fake_integrator)
    _res = run_runtime_nonlinear(cfg, ky_target=0.3, Nl=3, Nm=4, steps=1)

    # Ny=7, y0=10 -> ky = [0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1]
    # 2/3 dealias keeps |n|<=2 => ky=0.3 is excluded; nearest kept mode is ky=0.2 (index 2).
    assert captured["omega_ky_index"] == 2
    # Nx=3 FFT order puts kx=0 at index 0.
    assert captured["omega_kx_index"] == 0


def test_runtime_nonlinear_mode_selection_honors_kx_target(monkeypatch) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=5,
            Ny=5,
            Nz=8,
            Lx=12.0,
            Ly=62.8,
            boundary="periodic",
            y0=10.0,
        ),
        time=TimeConfig(
            t_max=0.02,
            dt=0.01,
            method="rk3",
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
        fixed_mode_ky_index=None,
        fixed_mode_kx_index=None,
    ):
        captured["omega_ky_index"] = int(omega_ky_index)
        captured["omega_kx_index"] = int(omega_kx_index)
        t = np.asarray([float(dt)], dtype=float)
        zeros = np.zeros_like(t)
        diag = SimulationDiagnostics(
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
        return t, diag, np.asarray(G0), None

    monkeypatch.setattr("spectraxgk.runtime.integrate_nonlinear_gx_diagnostics_state", _fake_integrator)
    _res = run_runtime_nonlinear(cfg, ky_target=0.1, kx_target=-1.1, Nl=3, Nm=4, steps=1)

    assert captured["omega_ky_index"] == 1
    assert captured["omega_kx_index"] == 3


def test_run_linear_case_uses_toml_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import spectraxgk.runtime as runtime

    base = _base_runtime_cfg()
    cfg = replace(base, output=replace(base.output, path=str(tmp_path / "linear_case")))

    def fake_load_runtime_from_toml(_path):
        return cfg, {"run": {"ky": 0.2, "Nl": 4, "Nm": 6, "solver": "krylov"}}

    def fake_run_runtime_linear(*_args, **_kwargs):
        return runtime.RuntimeLinearResult(
            ky=0.2,
            gamma=0.1,
            omega=0.2,
            selection=runtime.ModeSelection(ky_index=0, kx_index=0, z_index=0),
        )

    monkeypatch.setattr("spectraxgk.io.load_runtime_from_toml", fake_load_runtime_from_toml)
    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run_runtime_linear)

    rc = run_linear_case(tmp_path / "dummy.toml", show_progress=False)

    out = capsys.readouterr().out
    assert rc == 0
    assert f"saved {tmp_path / 'linear_case.summary.json'}" in out
    assert (tmp_path / "linear_case.summary.json").exists()


def test_run_nonlinear_case_uses_toml_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import spectraxgk.runtime as runtime

    base = _base_runtime_cfg()
    cfg = replace(base, output=replace(base.output, path=str(tmp_path / "nonlinear_case")))
    t = np.asarray([0.0, 0.1], dtype=float)
    diag = SimulationDiagnostics(
        t=t,
        dt_t=t + 0.1,
        dt_mean=0.1,
        gamma_t=np.asarray([0.0, 0.1], dtype=float),
        omega_t=np.asarray([0.0, 0.2], dtype=float),
        Wg_t=np.asarray([1.0, 1.1], dtype=float),
        Wphi_t=np.asarray([2.0, 2.1], dtype=float),
        Wapar_t=np.asarray([0.0, 0.0], dtype=float),
        heat_flux_t=np.asarray([4.0, 4.2], dtype=float),
        particle_flux_t=np.asarray([5.0, 5.2], dtype=float),
        energy_t=np.asarray([3.0, 3.2], dtype=float),
    )

    def fake_load_runtime_from_toml(_path):
        return cfg, {"run": {"ky": 0.2, "Nl": 4, "Nm": 6}, "time": {"dt": 0.1}}

    captured: dict[str, object] = {}

    def fake_run_runtime_nonlinear_with_artifacts(*_args, **_kwargs):
        captured.update(_kwargs)
        summary = tmp_path / "nonlinear_case.summary.json"
        diag_path = tmp_path / "nonlinear_case.diagnostics.csv"
        summary.write_text("{}\n", encoding="utf-8")
        diag_path.write_text("t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux\n", encoding="utf-8")
        return (
            runtime.RuntimeNonlinearResult(
                t=t,
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=0.0,
            ),
            {"summary": str(summary), "diagnostics": str(diag_path)},
        )

    monkeypatch.setattr("spectraxgk.io.load_runtime_from_toml", fake_load_runtime_from_toml)
    monkeypatch.setattr("spectraxgk.runtime_artifacts.run_runtime_nonlinear_with_artifacts", fake_run_runtime_nonlinear_with_artifacts)

    rc = run_nonlinear_case(tmp_path / "dummy.toml", show_progress=False)

    out = capsys.readouterr().out
    assert rc == 0
    assert captured["out"] == str(tmp_path / "nonlinear_case")
    assert f"saved {tmp_path / 'nonlinear_case.summary.json'}" in out
    assert (tmp_path / "nonlinear_case.summary.json").exists()
    assert (tmp_path / "nonlinear_case.diagnostics.csv").exists()
