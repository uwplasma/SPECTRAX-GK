"""Benchmark utilities and reference data tests."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import spectraxgk.benchmarks as benchmarks
from spectraxgk.analysis import fit_growth_rate
from spectraxgk.benchmarks import (
    compare_cyclone_to_reference,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
    run_cyclone_linear,
    run_cyclone_scan,
    run_etg_linear,
    run_etg_scan,
    run_kbm_linear,
    run_kbm_beta_scan,
    run_kbm_scan,
    run_kinetic_linear,
    run_kinetic_scan,
    run_tem_linear,
    run_tem_scan,
    select_kbm_solver_auto,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GridConfig,
    KBMBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
    TimeConfig,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearTerms
from spectraxgk.linear_krylov import KrylovConfig
from spectraxgk.species import Species, build_linear_params


def test_load_cyclone_reference():
    """Reference CSV must load and match known Cyclone values."""
    ref = load_cyclone_reference()
    assert ref.ky.shape == ref.omega.shape == ref.gamma.shape
    assert ref.ky.size > 0
    idx = int(np.argmin(np.abs(ref.ky - 0.3)))
    assert np.isclose(ref.ky[idx], 0.3)
    assert np.isclose(ref.omega[idx], 0.28199404, rtol=1e-6)
    assert np.isclose(ref.gamma[idx], 0.09302951, rtol=1e-6)


def test_load_cyclone_reference_kinetic():
    """Kinetic-electron reference CSV must load and be finite."""
    ref = load_cyclone_reference_kinetic()
    assert ref.ky.shape == ref.omega.shape == ref.gamma.shape
    assert ref.ky.size > 0
    assert np.isfinite(ref.gamma).all()


def test_load_kbm_reference():
    """KBM reference CSV must load and be finite."""
    ref = load_kbm_reference()
    assert ref.ky.shape == ref.omega.shape == ref.gamma.shape
    assert ref.ky.size > 0
    assert np.isfinite(ref.gamma).all()


def test_load_etg_reference():
    """ETG reference CSV must load and be finite."""
    ref = load_etg_reference()
    assert ref.ky.shape == ref.omega.shape == ref.gamma.shape
    assert ref.ky.size > 0
    assert np.isfinite(ref.gamma).all()


def test_load_tem_reference():
    """TEM reference CSV must load and be finite."""
    ref = load_tem_reference()
    assert ref.ky.shape == ref.omega.shape == ref.gamma.shape
    assert ref.ky.size > 0
    assert np.isfinite(ref.gamma).all()


def test_fit_growth_rate_exact():
    """Exact exponential signal should be recovered by the fitter."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.12
    omega = 0.34
    signal = np.exp((gamma - 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_window():
    """Windowing should not bias the fit for a pure exponential."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.08
    omega = 0.21
    signal = np.exp((gamma - 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal, tmin=5.0)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_tmax():
    """A tmax cut should still recover the correct growth rate."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.06
    omega = 0.18
    signal = np.exp((gamma - 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal, tmax=7.0)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_invalid():
    """The fitter should reject malformed inputs."""
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([1.0 + 0j]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([[0.0, 1.0]]), np.array([1.0 + 0j, 2.0 + 0j]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([[1.0 + 0j, 2.0 + 0j]]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([1.0 + 0j, 2.0 + 0j]), tmin=2.0)


def test_run_cyclone_linear_shapes():
    """Smoke test for the Cyclone linear runner on a tiny grid."""
    grid = GridConfig(Nx=8, Ny=8, Nz=16, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, steps=5, dt=0.1, method="rk4", solver="time")
    assert result.phi_t.shape[0] == 5
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_run_cyclone_linear_defaults():
    """Default cfg/params path should run without error."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, steps=3, dt=0.1, method="rk2", solver="time")
    assert result.phi_t.shape[0] == 3


def test_run_cyclone_linear_manual_window():
    """Manual fit windows should exercise the explicit fit path."""
    grid = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(
        cfg=cfg,
        steps=5,
        dt=0.1,
        method="rk2",
        solver="time",
        gx_reference=False,
        auto_window=False,
        tmin=0.1,
        tmax=0.3,
    )
    assert np.isfinite(result.gamma)


def test_run_cyclone_linear_full_operator_smoke():
    """Full operator path should execute without NaNs on a tiny run."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, steps=3, dt=0.1, method="rk2", solver="time")
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_cyclone_scan_and_compare():
    """Scan helper should return arrays and comparison should report errors."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    ky_values = np.array([0.2, 0.3])
    scan = run_cyclone_scan(ky_values, cfg=cfg, steps=3, dt=0.1, method="euler", solver="time")
    assert scan.ky.shape == ky_values.shape
    ref = load_cyclone_reference()
    result = run_cyclone_linear(cfg=cfg, steps=3, dt=0.1, ky_target=0.3, method="euler", solver="time")
    comparison = compare_cyclone_to_reference(result, ref)
    assert comparison.ky > 0.0
    assert np.isfinite(comparison.rel_gamma)


def test_cyclone_scan_fixed_batch_shape_matches_unpadded():
    """Fixed-shape ky batching should match unpadded batching outputs."""

    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    ky_values = np.array([0.2, 0.3, 0.4])
    kwargs = dict(
        cfg=cfg,
        steps=3,
        dt=0.1,
        method="euler",
        solver="time",
        ky_batch=2,
        tmin=0.0,
        tmax=0.3,
    )
    scan_fixed = run_cyclone_scan(ky_values, fixed_batch_shape=True, **kwargs)
    scan_unpadded = run_cyclone_scan(ky_values, fixed_batch_shape=False, **kwargs)
    assert np.allclose(scan_fixed.ky, scan_unpadded.ky)
    assert np.all(np.isfinite(scan_fixed.gamma))
    assert np.all(np.isfinite(scan_fixed.omega))
    assert np.allclose(scan_fixed.gamma, scan_unpadded.gamma, rtol=1.0e-5, atol=1.0e-6)
    assert np.allclose(scan_fixed.omega, scan_unpadded.omega, rtol=1.0e-5, atol=1.0e-6)


def test_cyclone_scan_manual_window():
    """Manual fit windows should exercise the explicit scan fit path."""
    grid = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid)
    ky_values = np.array([0.2])
    scan = run_cyclone_scan(
        ky_values,
        cfg=cfg,
        steps=5,
        dt=0.1,
        method="rk2",
        solver="time",
        auto_window=False,
        tmin=0.1,
        tmax=0.3,
    )
    assert scan.ky.shape == ky_values.shape


def test_cyclone_physics_regression():
    """Cyclone growth rates should track published values at ky rho_i = 0.3."""
    cfg = CycloneBaseCase()
    result = run_cyclone_linear(
        cfg=cfg,
        ky_target=0.3,
        Nl=6,
        Nm=12,
        steps=800,
        dt=0.01,
        method="imex2",
        solver="time",
        gx_reference=False,
    )
    ref = load_cyclone_reference()
    idx = int(np.argmin(np.abs(ref.ky - 0.3)))
    assert np.isclose(result.gamma, ref.gamma[idx], rtol=0.2)
    assert np.isclose(result.omega, ref.omega[idx], rtol=0.1)


def test_cyclone_scan_regression():
    """Reduced ky scan should remain within reference trends."""
    cfg = CycloneBaseCase()
    ky_values = np.array([0.3])
    scan = run_cyclone_scan(
        ky_values,
        cfg=cfg,
        Nl=6,
        Nm=12,
        steps=800,
        dt=0.01,
        method="imex2",
        solver="time",
    )
    ref = load_cyclone_reference()
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        assert np.isclose(gamma, ref.gamma[idx], rtol=0.25)
        assert np.isclose(omega, ref.omega[idx], rtol=0.1)


def test_cyclone_krylov_smoke():
    """Krylov solver should return finite eigenvalues on a small scan."""
    grid = GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    krylov_cfg = KrylovConfig(method="power", power_iters=40, power_dt=0.05)
    result = run_cyclone_linear(
        cfg=cfg,
        ky_target=0.3,
        Nl=4,
        Nm=8,
        solver="krylov",
        krylov_cfg=krylov_cfg,
    )
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_etg_growth_positive_for_gradients():
    """ETG growth rate should remain positive across R/LTe variations."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    cfg_low = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=4.0))
    cfg_high = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=8.0))
    low = run_etg_linear(
        cfg=cfg_low,
        ky_target=3.0,
        Nl=4,
        Nm=8,
        steps=400,
        dt=0.001,
        method="rk4",
        solver="time",
    )
    high = run_etg_linear(
        cfg=cfg_high,
        ky_target=3.0,
        Nl=4,
        Nm=8,
        steps=400,
        dt=0.001,
        method="rk4",
        solver="time",
    )
    assert low.gamma > 0.0
    assert high.gamma > 0.0
    assert high.gamma > 0.1 * low.gamma


def test_etg_frequency_sign():
    """ETG frequency should align with the electron diamagnetic direction."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    cfg = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=6.0))
    result = run_etg_linear(cfg=cfg, ky_target=3.0, Nl=4, Nm=8, steps=400, dt=0.001, method="rk4")
    assert np.isfinite(result.omega)
    assert result.omega < 0.0


def test_etg_scan_shapes():
    """ETG scan helper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    cfg = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=6.0))
    ky_values = np.array([3.0, 4.0])
    scan = run_etg_scan(ky_values, cfg=cfg, Nl=4, Nm=8, steps=400, dt=0.001, method="rk4")
    assert scan.ky.shape == ky_values.shape
    assert scan.gamma.shape == ky_values.shape


def test_kinetic_linear_smoke():
    """Kinetic-electron ITG/TEM benchmark should run and return finite outputs."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=62.8, Ly=62.8)
    cfg = KineticElectronBaseCase(grid=grid)
    result = run_kinetic_linear(cfg=cfg, ky_target=0.3, Nl=4, Nm=8, steps=200, dt=0.01, method="rk4")
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_kinetic_scan_shapes():
    """Kinetic-electron scan helper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=62.8, Ly=62.8)
    cfg = KineticElectronBaseCase(grid=grid)
    ky_values = np.array([0.3, 0.4])
    scan = run_kinetic_scan(ky_values, cfg=cfg, Nl=4, Nm=8, steps=200, dt=0.01, method="rk4")
    assert scan.ky.shape == ky_values.shape
    assert scan.gamma.shape == ky_values.shape


def test_tem_linear_smoke():
    """TEM benchmark should run and return finite outputs."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=62.8, Ly=62.8)
    cfg = TEMBaseCase(grid=grid)
    result = run_tem_linear(cfg=cfg, ky_target=0.3, Nl=4, Nm=8, steps=200, dt=0.01, method="rk4")
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_tem_scan_shapes():
    """TEM scan helper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=62.8, Ly=62.8)
    cfg = TEMBaseCase(grid=grid)
    ky_values = np.array([0.3, 0.4])
    scan = run_tem_scan(ky_values, cfg=cfg, Nl=4, Nm=8, steps=200, dt=0.01, method="rk4")
    assert scan.ky.shape == ky_values.shape
    assert scan.gamma.shape == ky_values.shape


def test_kbm_beta_scan_shapes():
    """KBM beta scan helper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=8, Nz=32, Lx=62.8, Ly=62.8)
    cfg = KBMBaseCase(grid=grid)
    betas = np.array([1.0e-4, 2.0e-4])
    scan = run_kbm_beta_scan(betas, cfg=cfg, ky_target=0.3, Nl=4, Nm=8, steps=200, dt=0.01)
    assert scan.ky.shape == betas.shape
    assert scan.gamma.shape == betas.shape


def test_kbm_ky_scan_shapes():
    """KBM ky-scan wrapper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    ky_values = np.array([0.2, 0.3])
    scan = run_kbm_scan(
        ky_values,
        cfg=cfg,
        beta_value=cfg.model.beta,
        ky_batch=1,
        Nl=4,
        Nm=8,
        dt=0.01,
        steps=100,
        method="rk2",
        solver="gx_time",
    )
    assert scan.ky.shape == ky_values.shape
    assert scan.gamma.shape == ky_values.shape
    assert np.isfinite(scan.gamma).all()
    assert np.isfinite(scan.omega).all()


def test_run_kbm_linear_gx_time_history():
    """Single-point KBM runs should return a usable field history."""
    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    result = run_kbm_linear(
        ky_target=0.3,
        cfg=cfg,
        Nl=4,
        Nm=8,
        dt=0.01,
        steps=120,
        solver="gx_time",
        sample_stride=2,
    )
    assert result.t.ndim == 1
    assert result.phi_t.ndim == 4
    assert result.phi_t.shape[0] == result.t.size
    assert result.selection.ky_index == 0
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_run_kbm_linear_accepts_gx_netcdf_geometry(tmp_path: Path):
    """KBM linear benchmark entry point should accept imported GX geometry."""

    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, grid.Nz + 1)
    geom_path = tmp_path / "kbm_geom.out.nc"
    with Dataset(geom_path, "w") as root:
        root.createDimension("theta", theta.size)
        grids = root.createGroup("Grids")
        geom = root.createGroup("Geometry")
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        geom.createVariable("bmag", "f8", ("theta",))[:] = np.ones(theta.size)
        geom.createVariable("bgrad", "f8", ("theta",))[:] = 0.05 * np.sin(theta)
        geom.createVariable("gds2", "f8", ("theta",))[:] = 1.0 + 0.1 * theta * theta
        geom.createVariable("gds21", "f8", ("theta",))[:] = -0.02 * theta
        geom.createVariable("gds22", "f8", ("theta",))[:] = np.full(theta.size, 0.64)
        geom.createVariable("cvdrift", "f8", ("theta",))[:] = 0.03 * np.cos(theta)
        geom.createVariable("gbdrift", "f8", ("theta",))[:] = 0.03 * np.cos(theta)
        geom.createVariable("cvdrift0", "f8", ("theta",))[:] = -0.01 * np.sin(theta)
        geom.createVariable("gbdrift0", "f8", ("theta",))[:] = -0.01 * np.sin(theta)
        geom.createVariable("jacobian", "f8", ("theta",))[:] = np.ones(theta.size)
        geom.createVariable("grho", "f8", ("theta",))[:] = np.ones(theta.size)
        geom.createVariable("gradpar", "f8", ())[:] = 1.0 / (1.4 * 2.77778)
        geom.createVariable("q", "f8", ())[:] = 1.4
        geom.createVariable("shat", "f8", ())[:] = 0.8
        geom.createVariable("rmaj", "f8", ())[:] = 2.77778
        geom.createVariable("aminor", "f8", ())[:] = 1.0

    cfg = KBMBaseCase(grid=grid)
    cfg_nc = replace(
        cfg,
        geometry=replace(cfg.geometry, model="gx-netcdf", geometry_file=str(geom_path)),
    )
    result = run_kbm_linear(
        ky_target=0.3,
        cfg=cfg_nc,
        Nl=4,
        Nm=8,
        dt=0.01,
        steps=40,
        solver="gx_time",
        sample_stride=2,
    )
    assert result.t.ndim == 1
    assert result.phi_t.ndim == 4
    assert result.phi_t.shape[0] == result.t.size
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_run_kbm_linear_gx_time_uses_requested_mode_extractor(monkeypatch):
    """GX-time KBM runs should honor the requested post-processing extractor."""

    calls: dict[str, str] = {}

    def _fake_integrate(*_args, mode_method: str, **_kwargs):
        from spectraxgk.diagnostics import GXDiagnostics

        calls["integrate_mode_method"] = mode_method
        t = np.array([0.1, 0.2, 0.3], dtype=float)
        phi_t = np.ones((3, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((3, 1, 1), dtype=float)
        omega_t = np.zeros((3, 1, 1), dtype=float)
        diag = GXDiagnostics(
            t=t,
            dt_t=np.full(t.shape, 0.1, dtype=float),
            dt_mean=np.asarray(0.1),
            gamma_t=gamma_t,
            omega_t=omega_t,
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return t, phi_t, gamma_t, omega_t, diag

    def _fake_growth(phi_t, t, sel, *, navg_fraction: float, mode_method: str, use_last: bool = False):
        del phi_t, t, sel, navg_fraction, use_last
        calls["growth_mode_method"] = mode_method
        return 0.25, 1.5, np.zeros(2), np.zeros(2), np.zeros(2)

    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(benchmarks, "gx_growth_rate_from_phi", _fake_growth)

    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    result = run_kbm_linear(
        ky_target=0.3,
        cfg=cfg,
        Nl=2,
        Nm=2,
        dt=0.01,
        steps=4,
        solver="gx_time",
        mode_method="project",
    )

    assert calls["integrate_mode_method"] == "z_index"
    assert calls["growth_mode_method"] == "project"
    assert np.isclose(result.gamma, 0.25)
    assert np.isclose(result.omega, 1.5)


def test_run_kbm_beta_scan_gx_time_keeps_project_mode(monkeypatch):
    """KBM scan helpers should not downgrade project mode on the GX-time path."""

    calls: list[str] = []

    def _fake_integrate(*_args, mode_method: str, **_kwargs):
        from spectraxgk.diagnostics import GXDiagnostics

        calls.append(f"integrate:{mode_method}")
        t = np.array([0.1, 0.2, 0.3], dtype=float)
        phi_t = np.ones((3, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((3, 1, 1), dtype=float)
        omega_t = np.zeros((3, 1, 1), dtype=float)
        diag = GXDiagnostics(
            t=t,
            dt_t=np.full(t.shape, 0.1, dtype=float),
            dt_mean=np.asarray(0.1),
            gamma_t=gamma_t,
            omega_t=omega_t,
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return t, phi_t, gamma_t, omega_t, diag

    def _fake_growth(phi_t, t, sel, *, navg_fraction: float, mode_method: str, use_last: bool = False):
        del phi_t, t, sel, navg_fraction, use_last
        calls.append(f"growth:{mode_method}")
        return 0.15, 0.9, np.zeros(2), np.zeros(2), np.zeros(2)

    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(benchmarks, "gx_growth_rate_from_phi", _fake_growth)

    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    scan = run_kbm_beta_scan(
        np.array([cfg.model.beta]),
        ky_target=0.3,
        cfg=cfg,
        Nl=2,
        Nm=2,
        dt=0.01,
        steps=4,
        solver="gx_time",
        mode_method="project",
    )

    assert calls == ["integrate:z_index", "growth:project"]
    assert np.isclose(scan.gamma[0], 0.15)
    assert np.isclose(scan.omega[0], 0.9)


def test_run_kbm_linear_krylov_explicit_shift_bypasses_multi_target(monkeypatch):
    """Explicit KBM Krylov shifts should bypass the built-in target sweep."""

    calls: list[dict[str, object]] = []

    def _fake_dominant_eigenpair(v0, *args, **kwargs):
        calls.append(kwargs)
        return 0.2 - 1.1j, np.zeros_like(np.asarray(v0))

    def _fake_compute_fields_cached(vec, cache, params, terms):
        del cache, params, terms
        return SimpleNamespace(phi=np.zeros(np.asarray(vec).shape[-3:], dtype=np.complex64))

    monkeypatch.setattr(benchmarks, "dominant_eigenpair", _fake_dominant_eigenpair)
    monkeypatch.setattr(benchmarks, "compute_fields_cached", _fake_compute_fields_cached)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    cfg = KBMBaseCase(grid=grid)
    krylov_cfg = replace(
        benchmarks.KBM_KRYLOV_DEFAULT,
        shift=complex(0.2, -1.1),
        shift_source="target",
        shift_selection="shift",
        omega_sign=0,
        omega_target_factor=0.0,
    )

    result = run_kbm_linear(
        ky_target=0.3,
        cfg=cfg,
        Nl=4,
        Nm=4,
        solver="krylov",
        krylov_cfg=krylov_cfg,
        gx_reference=False,
        diagnostic_norm="none",
    )

    assert len(calls) == 1
    assert calls[0]["shift"] == complex(0.2, -1.1)
    assert calls[0]["shift_selection"] == "shift"
    assert calls[0]["omega_target_factor"] == 0.0
    assert np.isclose(result.gamma, 0.2)
    assert np.isclose(result.omega, 1.1)


def test_run_kbm_beta_scan_krylov_explicit_shift_bypasses_multi_target(monkeypatch):
    """KBM beta scans should honor explicit Krylov shifts without retargeting."""

    calls: list[dict[str, object]] = []

    def _fake_dominant_eigenpair(v0, *args, **kwargs):
        calls.append(kwargs)
        return 0.2 - 1.1j, np.zeros_like(np.asarray(v0))

    monkeypatch.setattr(benchmarks, "dominant_eigenpair", _fake_dominant_eigenpair)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    cfg = KBMBaseCase(grid=grid)
    krylov_cfg = replace(
        benchmarks.KBM_KRYLOV_DEFAULT,
        shift=complex(0.2, -1.1),
        shift_source="target",
        shift_selection="shift",
        omega_sign=0,
        omega_target_factor=0.0,
    )

    scan = run_kbm_beta_scan(
        np.array([1.0e-4]),
        ky_target=0.3,
        cfg=cfg,
        Nl=4,
        Nm=4,
        solver="krylov",
        krylov_cfg=krylov_cfg,
        gx_reference=False,
        diagnostic_norm="none",
    )

    assert len(calls) == 1
    assert calls[0]["shift"] == complex(0.2, -1.1)
    assert calls[0]["shift_selection"] == "shift"
    assert calls[0]["omega_target_factor"] == 0.0
    assert np.isclose(scan.gamma[0], 0.2)
    assert np.isclose(scan.omega[0], 1.1)


def test_kbm_beta_scan_time_mode_only_phi():
    """KBM mode_only path should match full-field signal extraction."""
    grid = GridConfig(Nx=1, Ny=8, Nz=32, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1)
    cfg = KBMBaseCase(grid=grid)
    kwargs = dict(
        betas=np.array([1.0e-4]),
        cfg=cfg,
        ky_target=0.3,
        Nl=4,
        Nm=4,
        solver="time",
        method="euler",
        dt=0.1,
        steps=8,
        fit_signal="phi",
        mode_method="z_index",
        auto_window=False,
        tmin=0.2,
        tmax=0.6,
    )
    scan_mode_only = run_kbm_beta_scan(
        mode_only=True,
        **kwargs,
    )
    scan_full = run_kbm_beta_scan(
        mode_only=False,
        **kwargs,
    )
    assert np.isfinite(scan_mode_only.gamma[0])
    assert np.isfinite(scan_mode_only.omega[0])
    assert np.isfinite(scan_full.gamma[0])
    assert np.isfinite(scan_full.omega[0])
    assert np.isclose(scan_mode_only.gamma[0], scan_full.gamma[0], rtol=1.0e-6, atol=1.0e-10)
    assert np.isclose(scan_mode_only.omega[0], scan_full.omega[0], rtol=1.0e-6, atol=1.0e-10)


def test_kbm_beta_scan_timecfg_auto_fit_nondiffrax():
    """KBM auto fit path should work with non-diffrax TimeConfig."""
    grid = GridConfig(Nx=1, Ny=8, Nz=32, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1)
    time_cfg = TimeConfig(t_max=0.8, dt=0.1, method="rk2", use_diffrax=False, sample_stride=1)
    cfg = KBMBaseCase(grid=grid, time=time_cfg)
    scan = run_kbm_beta_scan(
        np.array([1.0e-4]),
        cfg=cfg,
        ky_target=0.3,
        Nl=4,
        Nm=4,
        solver="time",
        fit_signal="auto",
        mode_only=False,
        auto_window=True,
    )
    assert np.isfinite(scan.gamma[0])
    assert np.isfinite(scan.omega[0])


def test_select_kbm_solver_auto_lock():
    """KBM auto solver lock should be deterministic at GX-reference anchor ky."""
    assert select_kbm_solver_auto("auto", ky_target=0.1, gx_reference=True) == "gx_time"
    assert select_kbm_solver_auto("auto", ky_target=0.3, gx_reference=True) == "gx_time"
    assert select_kbm_solver_auto("auto", ky_target=0.4, gx_reference=True) == "gx_time"
    assert select_kbm_solver_auto("auto", ky_target=0.22, gx_reference=False) == "time"
    assert select_kbm_solver_auto("krylov", ky_target=0.3, gx_reference=True) == "krylov"


def test_etg_scan_manual_window():
    """Manual window path should be exercised for ETG scans."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    cfg = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=6.0))
    ky_values = np.array([3.0])
    scan = run_etg_scan(
        ky_values,
        cfg=cfg,
        Nl=4,
        Nm=8,
        steps=100,
        dt=0.01,
        method="rk4",
        solver="time",
        auto_window=False,
        tmin=0.05,
        tmax=0.15,
    )
    assert np.isfinite(scan.gamma[0])


def test_etg_manual_window():
    """Manual window path should be exercised for ETG fits."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    cfg = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=6.0))
    result = run_etg_linear(
        cfg=cfg,
        ky_target=3.0,
        Nl=4,
        Nm=8,
        steps=100,
        dt=0.01,
        method="rk4",
        solver="time",
        fit_signal="phi",
        terms=LinearTerms(
            streaming=0.0,
            mirror=0.0,
            curvature=0.0,
            gradb=0.0,
            diamagnetic=0.0,
            collisions=0.0,
            hypercollisions=0.0,
            end_damping=0.0,
            apar=0.0,
            bpar=0.0,
        ),
        auto_window=False,
        tmin=0.05,
        tmax=0.15,
    )
    assert np.isfinite(result.gamma)


def test_tem_run_density_fit():
    """TEM run should support density-based fits without error."""
    grid = GridConfig(Nx=1, Ny=6, Nz=8, Lx=6.28, Ly=6.28, ntheta=8, nperiod=1)
    cfg = TEMBaseCase(grid=grid)
    result = run_tem_linear(
        ky_target=0.3,
        cfg=cfg,
        Nl=4,
        Nm=4,
        dt=0.1,
        steps=5,
        method="euler",
        solver="time",
        fit_signal="density",
        auto_window=False,
        tmin=0.1,
        tmax=0.4,
    )
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_benchmark_krylov_smoke_finite():
    """Krylov solves should return finite gamma/omega for core benchmarks."""
    krylov_cfg = KrylovConfig(
        method="propagator",
        krylov_dim=8,
        restarts=1,
        power_iters=20,
        power_dt=0.01,
        omega_cap_factor=5.0,
    )

    small_grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28, ntheta=8, nperiod=1, y0=2.0)
    cyclone_cfg = CycloneBaseCase(grid=small_grid)
    cyclone = run_cyclone_linear(
        cfg=cyclone_cfg,
        ky_target=0.3,
        Nl=4,
        Nm=4,
        solver="krylov",
        krylov_cfg=krylov_cfg,
    )
    assert np.isfinite(cyclone.gamma)
    assert np.isfinite(cyclone.omega)

    etg_grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28, ntheta=8, nperiod=1, y0=0.2)
    etg_cfg = ETGBaseCase(grid=etg_grid, model=ETGModelConfig(R_over_LTe=6.0))
    etg = run_etg_linear(
        cfg=etg_cfg,
        ky_target=3.0,
        Nl=4,
        Nm=4,
        solver="krylov",
        krylov_cfg=krylov_cfg,
    )
    assert np.isfinite(etg.gamma)
    assert np.isfinite(etg.omega)

    kin_grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    kin_cfg = KineticElectronBaseCase(grid=kin_grid)
    kin = run_kinetic_linear(
        cfg=kin_cfg,
        ky_target=0.3,
        Nl=4,
        Nm=4,
        solver="krylov",
        krylov_cfg=krylov_cfg,
    )
    assert np.isfinite(kin.gamma)
    assert np.isfinite(kin.omega)

    kbm_grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    kbm_cfg = KBMBaseCase(grid=kbm_grid)
    kbm_scan = run_kbm_beta_scan(
        np.array([1.0e-4]),
        cfg=kbm_cfg,
        ky_target=0.3,
        Nl=4,
        Nm=4,
        solver="krylov",
        krylov_cfg=krylov_cfg,
    )
    assert np.isfinite(kbm_scan.gamma[0])
    assert np.isfinite(kbm_scan.omega[0])

    tem_grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    tem_cfg = TEMBaseCase(grid=tem_grid)
    tem = run_tem_linear(
        cfg=tem_cfg,
        ky_target=0.3,
        Nl=4,
        Nm=4,
        solver="krylov",
        krylov_cfg=krylov_cfg,
    )
    assert np.isfinite(tem.gamma)
    assert np.isfinite(tem.omega)


def test_etg_linear_with_params():
    """ETG harness should accept explicit parameter overrides."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    model = ETGModelConfig(R_over_LTe=6.0)
    cfg = ETGBaseCase(grid=grid, model=model)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=model.R_over_LTi, fprim=model.R_over_Ln),
            Species(
                charge=-1.0,
                mass=1.0 / model.mass_ratio,
                density=1.0,
                temperature=model.Te_over_Ti,
                tprim=model.R_over_LTe,
                fprim=model.R_over_Ln,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
        rho_star=1.0,
    )
    result = run_etg_linear(cfg=cfg, params=params, ky_target=3.0, Nl=4, Nm=8, steps=100, dt=0.01)
    assert np.isfinite(result.gamma)


def test_etg_scan_with_params():
    """ETG scan should accept explicit parameter overrides."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    model = ETGModelConfig(R_over_LTe=6.0)
    cfg = ETGBaseCase(grid=grid, model=model)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=model.R_over_LTi, fprim=model.R_over_Ln),
            Species(
                charge=-1.0,
                mass=1.0 / model.mass_ratio,
                density=1.0,
                temperature=model.Te_over_Ti,
                tprim=model.R_over_LTe,
                fprim=model.R_over_Ln,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
        rho_star=1.0,
    )
    scan = run_etg_scan(
        np.array([3.0]),
        cfg=cfg,
        params=params,
        Nl=4,
        Nm=8,
        steps=100,
        dt=0.01,
        method="rk4",
    )
    assert np.isfinite(scan.gamma[0])
