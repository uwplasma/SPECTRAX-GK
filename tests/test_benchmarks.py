"""Benchmark utilities and reference data tests."""

import numpy as np
import pytest

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
    run_kbm_beta_scan,
    run_kinetic_linear,
    run_kinetic_scan,
    run_tem_linear,
    run_tem_scan,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GridConfig,
    KBMBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
)
from spectraxgk.geometry import SAlphaGeometry
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
    grid = GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(
        cfg=cfg,
        ky_target=0.3,
        Nl=6,
        Nm=12,
        steps=800,
        dt=0.01,
        method="imex2",
        solver="time",
    )
    ref = load_cyclone_reference()
    idx = int(np.argmin(np.abs(ref.ky - 0.3)))
    assert np.isclose(result.gamma, ref.gamma[idx], rtol=0.2)
    assert np.isclose(result.omega, ref.omega[idx], rtol=0.1)


def test_cyclone_scan_regression():
    """Reduced ky scan should remain within reference trends."""
    grid = GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    ky_values = np.array([0.3, 0.4])
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
    result = run_cyclone_linear(cfg=cfg, ky_target=0.3, Nl=4, Nm=8, solver="krylov")
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_etg_growth_positive_for_gradients():
    """ETG growth rate should remain positive across R/LTe variations."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    cfg_low = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=4.0))
    cfg_high = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=8.0))
    low = run_etg_linear(cfg=cfg_low, ky_target=3.0, Nl=4, Nm=8, steps=400, dt=0.001, method="rk4")
    high = run_etg_linear(cfg=cfg_high, ky_target=3.0, Nl=4, Nm=8, steps=400, dt=0.001, method="rk4")
    assert low.gamma > 0.0
    assert high.gamma > 0.0
    assert high.gamma > 0.9 * low.gamma


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
        auto_window=False,
        tmin=0.05,
        tmax=0.15,
    )
    assert np.isfinite(result.gamma)


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
