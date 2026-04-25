"""Benchmark utilities and reference data tests."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp

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
    InitializationConfig,
    KBMBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
    TimeConfig,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, LinearTerms
from spectraxgk.linear_krylov import KrylovConfig
from spectraxgk.species import Species, build_linear_params

pytestmark = pytest.mark.integration


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


def test_load_reference_with_header_helper_contract():
    """Header-driven reference loader should preserve ky/gamma/omega column semantics."""
    ref = benchmarks._load_reference_with_header("tem_reference.csv")
    raw = np.genfromtxt("src/spectraxgk/data/tem_reference.csv", delimiter=",", names=True, dtype=float)
    np.testing.assert_allclose(ref.ky[:3], raw["ky"][:3])
    np.testing.assert_allclose(ref.gamma[:3], raw["gamma"][:3])
    np.testing.assert_allclose(ref.omega[:3], raw["omega"][:3])


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


def test_benchmark_small_policy_helpers_cover_branch_contracts() -> None:
    """Fast policy helpers should stay deterministic without launching solvers."""

    params = benchmarks._apply_gx_hypercollisions(LinearParams(), nhermite=None)
    assert params.p_hyper_m == pytest.approx(benchmarks.REFERENCE_P_HYPER_M)
    assert benchmarks._gx_p_hyper_m(1) == pytest.approx(1.0)
    assert benchmarks._gx_linked_end_damping(True) == (
        benchmarks.REFERENCE_DAMP_ENDS_AMP,
        benchmarks.REFERENCE_DAMP_ENDS_WIDTHFRAC,
    )
    assert benchmarks._gx_linked_end_damping(False) == (0.0, 0.0)
    assert benchmarks._midplane_index(SimpleNamespace(z=np.zeros(1))) == 0
    assert benchmarks._midplane_index(SimpleNamespace(z=np.zeros(4))) == 3
    assert select_kbm_solver_auto("auto", ky_target=0.22, gx_reference=True) == "gx_time"

    batches = list(benchmarks._iter_ky_batches(np.array([0.1, 0.2, 0.3]), ky_batch=2, fixed_batch_shape=True))
    assert batches[0][0] == 0
    np.testing.assert_allclose(batches[0][1], [0.1, 0.2])
    assert batches[0][2] == 2
    assert batches[1][0] == 2
    np.testing.assert_allclose(batches[1][1], [0.3, 0.3])
    assert batches[1][2] == 1
    singles = list(benchmarks._iter_ky_batches(np.array([0.4, 0.5]), ky_batch=1, fixed_batch_shape=False))
    assert [(start, valid) for start, _batch, valid in singles] == [(0, 1), (1, 1)]

    assert benchmarks._resolve_streaming_window(10.0, 2.0, 4.0, 0.1, 0.2, 0.8) == (2.0, 4.0)
    assert benchmarks._resolve_streaming_window(10.0, None, None, 0.9, 0.2, 0.8) == (9.0, 10.0)


def test_benchmark_fit_signal_helper_fallbacks(monkeypatch) -> None:
    """Fit-signal selection should prefer the requested observable and fall back safely."""

    sel = benchmarks.ModeSelection(ky_index=0, kx_index=0, z_index=0)
    phi = np.zeros((3, 1, 1, 1), dtype=np.complex128)
    density = np.ones_like(phi)
    valid = np.array([1.0 + 0.0j, 2.0 + 0.1j, 3.0 + 0.2j])
    invalid = np.array([np.nan + 0.0j, np.inf + 0.0j, np.nan + 0.0j])

    def fake_extract(arr, selection, method="z_index"):
        assert selection is sel
        assert method == "z_index"
        return valid if arr is density else invalid

    monkeypatch.setattr(benchmarks, "extract_mode_time_series", fake_extract)
    np.testing.assert_allclose(
        benchmarks._select_fit_signal(phi, density, sel, fit_signal="phi", mode_method="z_index"),
        valid,
    )

    with pytest.warns(RuntimeWarning, match="insufficient finite"):
        zero_signal = benchmarks._select_fit_signal(phi, None, sel, fit_signal="phi", mode_method="z_index")
    np.testing.assert_allclose(zero_signal, np.zeros(3, dtype=np.complex128))

    with pytest.raises(ValueError, match="density_t"):
        benchmarks._select_fit_signal(phi, None, sel, fit_signal="density", mode_method="z_index")

    def fake_density_invalid(arr, selection, method="z_index"):
        return invalid if arr is density else valid

    monkeypatch.setattr(benchmarks, "extract_mode_time_series", fake_density_invalid)
    np.testing.assert_allclose(
        benchmarks._select_fit_signal(phi, density, sel, fit_signal="density", mode_method="z_index"),
        valid,
    )

    monkeypatch.setattr(benchmarks, "extract_mode_time_series", lambda *_args, **_kwargs: invalid)
    with pytest.warns(RuntimeWarning, match="insufficient finite"):
        zero_density = benchmarks._select_fit_signal(phi, density, sel, fit_signal="density", mode_method="z_index")
    np.testing.assert_allclose(zero_density, np.zeros(3, dtype=np.complex128))

    with pytest.raises(ValueError, match="fit_signal"):
        benchmarks._select_fit_signal(phi, density, sel, fit_signal="bad", mode_method="z_index")


def test_benchmark_auto_fit_signal_scoring_rejects_bad_windows(monkeypatch) -> None:
    """Automatic branch scoring should make invalid fits unselectable."""

    kwargs = dict(
        tmin=None,
        tmax=None,
        window_fraction=0.5,
        min_points=2,
        start_fraction=0.0,
        growth_weight=1.0,
        require_positive=True,
        min_amp_fraction=0.0,
        max_amp_fraction=1.0,
        window_method="grid",
        max_fraction=1.0,
        end_fraction=1.0,
        num_windows=4,
        phase_weight=0.1,
        length_weight=0.0,
        min_r2=0.8,
        late_penalty=0.0,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )
    t = np.array([0.0, 1.0, 2.0])
    signal = np.exp(0.1 * t)

    monkeypatch.setattr(
        benchmarks,
        "fit_growth_rate_auto_with_stats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad window")),
    )
    assert benchmarks._score_fit_signal_auto(t, signal, **kwargs) == (0.0, 0.0, -np.inf)

    monkeypatch.setattr(
        benchmarks,
        "fit_growth_rate_auto_with_stats",
        lambda *_args, **_kwargs: (np.nan, 1.0, 0.0, 1.0, 1.0, 1.0),
    )
    gamma, _omega, score = benchmarks._score_fit_signal_auto(t, signal, **kwargs)
    assert np.isnan(gamma)
    assert score == -np.inf

    monkeypatch.setattr(
        benchmarks,
        "fit_growth_rate_auto_with_stats",
        lambda *_args, **_kwargs: (-0.1, 1.0, 0.0, 1.0, 1.0, 1.0),
    )
    assert benchmarks._score_fit_signal_auto(t, signal, **kwargs) == (-0.1, 1.0, -np.inf)

    monkeypatch.setattr(
        benchmarks,
        "fit_growth_rate_auto_with_stats",
        lambda *_args, **_kwargs: (0.1, 1.0, 0.0, 1.0, 0.7, 1.0),
    )
    assert benchmarks._score_fit_signal_auto(t, signal, **kwargs) == (0.1, 1.0, -np.inf)

    monkeypatch.setattr(
        benchmarks,
        "fit_growth_rate_auto_with_stats",
        lambda *_args, **_kwargs: (0.2, 1.0, 0.0, 1.0, 0.9, 0.5),
    )
    gamma, omega, score = benchmarks._score_fit_signal_auto(t, signal, **kwargs)
    assert gamma == pytest.approx(0.2)
    assert omega == pytest.approx(1.0)
    assert score == pytest.approx(0.9 + 0.1 * 0.5 + 0.2)


def test_benchmark_reduced_trace_and_initialization_helpers() -> None:
    """Reduced diagnostics and analytic initializers should handle edge cases deterministically."""

    scalar = np.asarray(2.0 + 1.0j)
    np.testing.assert_allclose(benchmarks._extract_mode_only_signal(scalar, local_idx=0), [2.0 + 1.0j])
    vector = np.array([1.0, 2.0])
    assert benchmarks._extract_mode_only_signal(vector, local_idx=0) is vector
    two_dim = np.arange(6).reshape(3, 2)
    np.testing.assert_allclose(benchmarks._extract_mode_only_signal(two_dim, local_idx=5), [1, 3, 5])
    three_dim = np.arange(24).reshape(3, 2, 4)
    np.testing.assert_allclose(
        benchmarks._extract_mode_only_signal(three_dim, local_idx=3, species_index=1),
        [7, 15, 23],
    )
    four_dim = np.arange(48).reshape(3, 2, 2, 4)
    np.testing.assert_allclose(benchmarks._extract_mode_only_signal(four_dim, local_idx=99), [15, 31, 47])

    grid = build_spectral_grid(GridConfig(Nx=3, Ny=3, Nz=5, Lx=6.0, Ly=6.0, y0=5.0, ntheta=5, nperiod=1))
    geom = SAlphaGeometry.from_config(CycloneBaseCase(grid=GridConfig()).geometry)
    init = InitializationConfig(init_field="all", init_amp=0.25, gaussian_init=False)
    state = benchmarks._build_initial_condition(grid, geom, ky_index=[1], kx_index=0, Nl=2, Nm=4, init_cfg=init)
    assert state.shape == (2, 4, grid.ky.size, grid.kx.size, grid.z.size)
    assert np.count_nonzero(np.asarray(state)) > 0

    with pytest.raises(ValueError, match="init_field"):
        benchmarks._build_initial_condition(
            grid,
            geom,
            ky_index=1,
            kx_index=0,
            Nl=1,
            Nm=1,
            init_cfg=InitializationConfig(init_field="bad"),
        )
    with pytest.raises(ValueError, match="moment exceeds"):
        benchmarks._build_initial_condition(
            grid,
            geom,
            ky_index=1,
            kx_index=0,
            Nl=1,
            Nm=1,
            init_cfg=InitializationConfig(init_field="qpar"),
        )
    with pytest.raises(ValueError, match="gaussian_width"):
        benchmarks._build_gaussian_profile(
            np.asarray(grid.z),
            kx=float(grid.kx[0]),
            ky=float(grid.ky[1]),
            s_hat=geom.s_hat,
            init_cfg=InitializationConfig(gaussian_width=0.0),
        )
    np.testing.assert_allclose(
        benchmarks._build_gaussian_profile(
            np.asarray(grid.z),
            kx=float(grid.kx[0]),
            ky=0.0,
            s_hat=geom.s_hat,
            init_cfg=InitializationConfig(),
        ),
        np.zeros_like(np.asarray(grid.z)),
    )


def test_benchmark_kinetic_parameter_helpers_validate_and_override() -> None:
    """Kinetic benchmark parameter builders should reject unphysical inputs and apply explicit overrides."""

    base = SimpleNamespace(
        mass_ratio=1836.0,
        Te_over_Ti=1.2,
        R_over_Ln=2.0,
        R_over_Lni=1.8,
        R_over_Lne=2.2,
        R_over_LTi=6.0,
        R_over_LTe=7.0,
        nu_i=0.01,
        nu_e=0.02,
        beta=1.0e-4,
    )
    kwargs = dict(kpar_scale=0.7, omega_d_scale=1.1, omega_star_scale=0.9, rho_star=0.01)

    def with_model(**updates):
        return SimpleNamespace(**{**vars(base), **updates})

    with pytest.raises(ValueError, match="mass_ratio"):
        benchmarks._two_species_params(with_model(mass_ratio=0.0), **kwargs)
    with pytest.raises(ValueError, match="Te_over_Ti"):
        benchmarks._two_species_params(with_model(Te_over_Ti=0.0), **kwargs)
    two_species = benchmarks._two_species_params(
        base,
        beta_override=0.0,
        fapar_override=0.25,
        damp_ends_amp=0.3,
        damp_ends_widthfrac=0.4,
        nhermite=8,
        **kwargs,
    )
    assert two_species.fapar == pytest.approx(0.25)
    assert two_species.damp_ends_amp == pytest.approx(0.3)
    assert two_species.damp_ends_widthfrac == pytest.approx(0.4)
    assert two_species.p_hyper_m == pytest.approx(4.0)

    with pytest.raises(ValueError, match="mass_ratio"):
        benchmarks._electron_only_params(with_model(mass_ratio=0.0), **kwargs)
    with pytest.raises(ValueError, match="Te_over_Ti"):
        benchmarks._electron_only_params(with_model(Te_over_Ti=0.0), **kwargs)
    electron_only = benchmarks._electron_only_params(
        base,
        beta_override=0.0,
        fapar_override=0.5,
        damp_ends_amp=0.6,
        damp_ends_widthfrac=0.7,
        nhermite=6,
        **kwargs,
    )
    assert electron_only.fapar == pytest.approx(0.5)
    assert electron_only.damp_ends_amp == pytest.approx(0.6)
    assert electron_only.damp_ends_widthfrac == pytest.approx(0.7)
    assert electron_only.p_hyper_m == pytest.approx(3.0)


def test_benchmark_kbm_multi_target_and_kinetic_init_policy() -> None:
    """KBM branch-continuity and kinetic initialization policies should be explicit."""

    kcfg = benchmarks.KBM_KRYLOV_DEFAULT
    assert benchmarks._kbm_use_multi_target_krylov(kcfg, [0.8, 1.0], shift=None)
    assert not benchmarks._kbm_use_multi_target_krylov(kcfg, None, shift=None)
    assert not benchmarks._kbm_use_multi_target_krylov(replace(kcfg, mode_family="cyclone"), [1.0], shift=None)
    assert not benchmarks._kbm_use_multi_target_krylov(replace(kcfg, method="power"), [1.0], shift=None)
    assert not benchmarks._kbm_use_multi_target_krylov(kcfg, [1.0], shift=1.0 + 0.0j)
    assert not benchmarks._kbm_use_multi_target_krylov(replace(kcfg, shift_selection="shift"), [1.0], shift=None)

    default_init = KineticElectronBaseCase().init
    assert benchmarks._kinetic_reference_init_cfg(default_init, gx_reference=False) is default_init
    explicit = replace(default_init, init_amp=2.0e-3)
    assert benchmarks._kinetic_reference_init_cfg(explicit, gx_reference=True) is explicit
    gx_init = benchmarks._kinetic_reference_init_cfg(default_init, gx_reference=True)
    assert gx_init.init_field == "density"
    assert gx_init.init_amp == pytest.approx(1.0e-3)
    assert not gx_init.gaussian_init


def test_benchmark_linear_entrypoints_reject_invalid_scan_options() -> None:
    """Entry points should reject invalid fit and batching options before launching solves."""

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0, y0=5.0, ntheta=8, nperiod=1))
    with pytest.raises(ValueError, match="fit_signal"):
        run_cyclone_linear(cfg=cfg, ky_target=0.2, Nl=2, Nm=2, fit_signal="bad")
    with pytest.raises(ValueError, match="fit_signal"):
        run_cyclone_scan(np.array([0.2]), cfg=cfg, Nl=2, Nm=2, fit_signal="bad")
    with pytest.raises(ValueError, match="ky_batch"):
        run_cyclone_scan(np.array([0.2]), cfg=cfg, Nl=2, Nm=2, ky_batch=0)


@pytest.mark.slow
def test_run_cyclone_linear_shapes():
    """Smoke test for the Cyclone linear runner on a tiny grid."""
    grid = GridConfig(Nx=8, Ny=8, Nz=16, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, ky_target=0.1, steps=5, dt=0.1, method="rk4", solver="time")
    assert result.phi_t.shape[0] == 5
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


@pytest.mark.slow
def test_run_cyclone_linear_defaults():
    """Default cfg/params path should run without error."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, ky_target=0.1, steps=3, dt=0.1, method="rk2", solver="time")
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


@pytest.mark.slow
def test_run_cyclone_linear_full_operator_smoke():
    """Full operator path should execute without NaNs on a tiny run."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, ky_target=0.1, steps=3, dt=0.1, method="rk2", solver="time")
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


@pytest.mark.slow
def test_cyclone_scan_and_compare():
    """Scan helper should return arrays and comparison should report errors."""
    grid = GridConfig(Nx=6, Ny=12, Nz=8, Lx=62.8, Ly=62.8)
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

    grid = GridConfig(Nx=6, Ny=16, Nz=8, Lx=62.8, Ly=62.8)
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


@pytest.mark.slow
def test_cyclone_physics_regression():
    """Cyclone growth rates should track published values at ky rho_i = 0.3."""
    cfg = CycloneBaseCase()
    krylov_cfg = KrylovConfig(method="power", power_iters=30, power_dt=0.05)
    result = run_cyclone_linear(
        cfg=cfg,
        ky_target=0.3,
        Nl=4,
        Nm=8,
        solver="krylov",
        krylov_cfg=krylov_cfg,
        gx_reference=False,
    )
    ref = load_cyclone_reference()
    idx = int(np.argmin(np.abs(ref.ky - 0.3)))
    assert np.isclose(result.gamma, ref.gamma[idx], rtol=0.35)
    assert np.isclose(result.omega, ref.omega[idx], rtol=0.2)


@pytest.mark.slow
def test_cyclone_scan_regression():
    """Reduced ky scan should remain within reference trends."""
    cfg = CycloneBaseCase()
    krylov_cfg = KrylovConfig(method="power", power_iters=30, power_dt=0.05)
    ky_values = np.array([0.3])
    scan = run_cyclone_scan(
        ky_values,
        cfg=cfg,
        Nl=4,
        Nm=8,
        solver="krylov",
        krylov_cfg=krylov_cfg,
    )
    ref = load_cyclone_reference()
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        assert np.isclose(gamma, ref.gamma[idx], rtol=0.4)
        assert np.isclose(omega, ref.omega[idx], rtol=0.2)


@pytest.mark.slow
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


def test_run_cyclone_linear_auto_can_fallback_to_krylov_after_time_path(monkeypatch):
    import spectraxgk.benchmarks as benchmarks

    def _fake_integrate_linear_diagnostics(*_args, **_kwargs):
        phi_t = np.ones((2, 1, 1, 8), dtype=np.complex64)
        density_t = np.ones_like(phi_t)
        return np.array([0.0, 0.1], dtype=float), phi_t, density_t

    monkeypatch.setattr(
        benchmarks,
        "_select_fit_signal_auto",
        lambda *args, **kwargs: (np.ones_like(args[0]), "phi", np.nan, np.nan),
    )
    monkeypatch.setattr(benchmarks, "integrate_linear_diagnostics", _fake_integrate_linear_diagnostics)
    monkeypatch.setattr(
        benchmarks,
        "integrate_linear_gx",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("seed unavailable in branch test")),
    )
    monkeypatch.setattr(
        benchmarks,
        "dominant_eigenpair",
        lambda G0, *_args, **_kwargs: (0.2 - 0.3j, np.zeros_like(np.asarray(G0))),
    )
    monkeypatch.setattr(
        benchmarks,
        "compute_fields_cached",
        lambda *_args, **_kwargs: SimpleNamespace(phi=np.zeros((1, 1, 8), dtype=np.complex64)),
    )

    grid = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0, y0=5.0, ntheta=8, nperiod=1)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(
        cfg=cfg,
        ky_target=0.1,
        Nl=2,
        Nm=2,
        steps=2,
        dt=0.1,
        method="rk4",
        solver="auto",
    )
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_cyclone_scan_gx_time_falls_back_to_krylov_when_gx_growth_is_unavailable(monkeypatch):
    import spectraxgk.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "build_linear_cache", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        benchmarks,
        "integrate_linear_gx",
        lambda *_args, **_kwargs: (
            np.array([0.0, 0.1], dtype=float),
            np.ones((2, 1, 1, 8), dtype=np.complex64),
            np.zeros((2, 1, 1), dtype=float),
            np.zeros((2, 1, 1), dtype=float),
        ),
    )
    monkeypatch.setattr(
        benchmarks,
        "gx_growth_rate_from_phi",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("No finite GX growth-rate samples available")),
    )
    monkeypatch.setattr(
        benchmarks,
        "dominant_eigenpair",
        lambda *_args, **_kwargs: (0.3 - 0.8j, np.ones((1, 2, 2, 1, 1, 8), dtype=np.complex64)),
    )
    monkeypatch.setattr(
        benchmarks,
        "_normalize_growth_rate",
        lambda gamma, omega, *_args, **_kwargs: (float(gamma), float(omega)),
    )

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=8, Nz=8, Lx=6.0, Ly=6.0, y0=5.0, ntheta=8, nperiod=1))
    scan = run_cyclone_scan(
        np.array([0.3]),
        cfg=cfg,
        Nl=2,
        Nm=2,
        dt=np.array([0.1]),
        steps=np.array([2]),
        method="imex2",
        solver="auto",
    )

    assert np.isclose(scan.gamma[0], 0.3)
    assert np.isclose(scan.omega[0], 0.8)


@pytest.mark.slow
def test_etg_growth_positive_for_gradients():
    """ETG growth rate should remain positive across R/LTe variations."""
    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    cfg_low = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=4.0))
    cfg_high = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=8.0))
    low = run_etg_linear(
        cfg=cfg_low,
        ky_target=3.0,
        Nl=3,
        Nm=6,
        steps=80,
        dt=0.003,
        method="rk4",
        solver="time",
    )
    high = run_etg_linear(
        cfg=cfg_high,
        ky_target=3.0,
        Nl=3,
        Nm=6,
        steps=80,
        dt=0.003,
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
    result = run_etg_linear(cfg=cfg, ky_target=3.0, Nl=3, Nm=6, steps=80, dt=0.003, method="rk4")
    assert np.isfinite(result.omega)
    assert result.omega < 0.0


def test_etg_scan_shapes():
    """ETG scan helper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=12, Nz=24, Lx=6.28, Ly=6.28)
    cfg = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=6.0))
    ky_values = np.array([3.0, 4.0])
    scan = run_etg_scan(ky_values, cfg=cfg, Nl=3, Nm=6, steps=50, dt=0.003, method="rk4")
    assert scan.ky.shape == ky_values.shape
    assert scan.gamma.shape == ky_values.shape


def test_kinetic_linear_smoke():
    """Kinetic-electron ITG/TEM benchmark should run and return finite outputs."""
    grid = GridConfig(Nx=1, Ny=12, Nz=24, Lx=62.8, Ly=62.8)
    cfg = KineticElectronBaseCase(grid=grid)
    result = run_kinetic_linear(cfg=cfg, ky_target=0.3, Nl=3, Nm=6, steps=50, dt=0.02, method="rk4")
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_kinetic_scan_shapes():
    """Kinetic-electron scan helper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=12, Nz=24, Lx=62.8, Ly=62.8)
    cfg = KineticElectronBaseCase(grid=grid)
    ky_values = np.array([0.3, 0.4])
    scan = run_kinetic_scan(ky_values, cfg=cfg, Nl=3, Nm=6, steps=50, dt=0.02, method="rk4")
    assert scan.ky.shape == ky_values.shape
    assert scan.gamma.shape == ky_values.shape


def test_tem_linear_smoke():
    """TEM benchmark should run and return finite outputs."""
    grid = GridConfig(Nx=1, Ny=12, Nz=24, Lx=62.8, Ly=62.8)
    cfg = TEMBaseCase(grid=grid)
    result = run_tem_linear(cfg=cfg, ky_target=0.3, Nl=3, Nm=6, steps=50, dt=0.02, method="rk4")
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_tem_scan_shapes():
    """TEM scan helper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=12, Nz=24, Lx=62.8, Ly=62.8)
    cfg = TEMBaseCase(grid=grid)
    ky_values = np.array([0.3, 0.4])
    scan = run_tem_scan(ky_values, cfg=cfg, Nl=3, Nm=6, steps=50, dt=0.02, method="rk4")
    assert scan.ky.shape == ky_values.shape
    assert scan.gamma.shape == ky_values.shape


@pytest.mark.slow
def test_kbm_beta_scan_shapes():
    """KBM beta scan helper should return arrays of the requested size."""
    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8)
    cfg = KBMBaseCase(grid=grid)
    betas = np.array([1.0e-4, 2.0e-4])
    scan = run_kbm_beta_scan(betas, cfg=cfg, ky_target=0.3, Nl=3, Nm=6, steps=40, dt=0.02)
    assert scan.ky.shape == betas.shape
    assert scan.gamma.shape == betas.shape


@pytest.mark.slow
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
        Nl=3,
        Nm=6,
        dt=0.02,
        steps=40,
        method="rk2",
        solver="gx_time",
    )
    assert scan.ky.shape == ky_values.shape
    assert scan.gamma.shape == ky_values.shape
    assert np.isfinite(scan.gamma).all()
    assert np.isfinite(scan.omega).all()


@pytest.mark.slow
def test_run_kbm_linear_gx_time_history():
    """Single-point KBM runs should return a usable field history."""
    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    result = run_kbm_linear(
        ky_target=0.3,
        cfg=cfg,
        Nl=3,
        Nm=6,
        dt=0.02,
        steps=40,
        solver="gx_time",
        sample_stride=2,
    )
    assert result.t.ndim == 1
    assert result.phi_t.ndim == 4
    assert result.phi_t.shape[0] == result.t.size
    assert result.gamma_t is not None
    assert result.omega_t is not None
    assert np.asarray(result.gamma_t).shape[0] == result.t.size
    assert np.asarray(result.omega_t).shape[0] == result.t.size
    assert result.selection.ky_index == 0
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


@pytest.mark.slow
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
        from spectraxgk.diagnostics import SimulationDiagnostics

        calls["integrate_mode_method"] = mode_method
        t = np.array([0.1, 0.2, 0.3], dtype=float)
        phi_t = np.ones((3, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((3, 1, 1), dtype=float)
        omega_t = np.zeros((3, 1, 1), dtype=float)
        diag = SimulationDiagnostics(
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

    def _fake_extract(phi_t, sel, method: str = "z_index"):
        del phi_t, sel
        calls["extract_mode_method"] = method
        return np.array([1.0 + 0.0j, 1.1 - 0.1j, 1.2 - 0.2j], dtype=np.complex128)

    def _fake_fit_auto(t, signal, **kwargs):
        del t, kwargs
        calls["fit_signal_len"] = str(np.asarray(signal).shape[0])
        return 0.25, 1.5, 0.0, 0.0

    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(benchmarks, "extract_mode_time_series", _fake_extract)
    monkeypatch.setattr(benchmarks, "fit_growth_rate_auto", _fake_fit_auto)
    monkeypatch.setattr(
        benchmarks,
        "gx_growth_rate_from_phi",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected GX ratio fit")),
    )

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
    assert calls["extract_mode_method"] == "project"
    assert calls["fit_signal_len"] == "3"
    assert np.isclose(result.gamma, 0.25)
    assert np.isclose(result.omega, 1.5)


def test_run_kbm_linear_uses_gx_linked_end_damping_by_default(monkeypatch):
    """GX-aligned KBM runs should inherit GX linked-end damping defaults."""

    captured: dict[str, float] = {}

    def _fake_two_species_params(*args, damp_ends_amp: float, damp_ends_widthfrac: float, **kwargs):
        del args, kwargs
        captured["amp"] = float(damp_ends_amp)
        captured["width"] = float(damp_ends_widthfrac)
        return SimpleNamespace(
            damp_ends_amp=float(damp_ends_amp),
            damp_ends_widthfrac=float(damp_ends_widthfrac),
            rho_star=benchmarks.KBM_RHO_STAR,
        )

    def _fake_build_linear_cache(*_args, **_kwargs):
        return object()

    def _fake_integrate(*_args, mode_method: str, **_kwargs):
        from spectraxgk.diagnostics import SimulationDiagnostics

        del mode_method
        t = np.array([0.1, 0.2], dtype=float)
        phi_t = np.ones((2, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((2, 1, 1), dtype=float)
        omega_t = np.zeros((2, 1, 1), dtype=float)
        diag = SimulationDiagnostics(
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

    monkeypatch.setattr(benchmarks, "_two_species_params", _fake_two_species_params)
    monkeypatch.setattr(benchmarks, "build_linear_cache", _fake_build_linear_cache)
    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(
        benchmarks,
        "gx_growth_rate_from_phi",
        lambda *args, **kwargs: (0.1, 0.2, np.zeros(1), np.zeros(1), np.zeros(1)),
    )

    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    run_kbm_linear(ky_target=0.3, cfg=cfg, Nl=2, Nm=2, dt=0.01, steps=4, solver="gx_time")

    assert captured["amp"] == pytest.approx(benchmarks.REFERENCE_DAMP_ENDS_AMP)
    assert captured["width"] == pytest.approx(benchmarks.REFERENCE_DAMP_ENDS_WIDTHFRAC)


def test_run_kbm_linear_gx_time_uses_gx_rk4_cfl_factor_by_default(monkeypatch):
    captured: dict[str, float] = {}

    def _fake_integrate(*args, mode_method: str, **_kwargs):
        from spectraxgk.diagnostics import SimulationDiagnostics

        del mode_method
        time_cfg = args[5]
        captured["cfl_fac"] = float(time_cfg.cfl_fac)
        t = np.array([0.1, 0.2], dtype=float)
        phi_t = np.ones((2, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((2, 1, 1), dtype=float)
        omega_t = np.zeros((2, 1, 1), dtype=float)
        diag = SimulationDiagnostics(
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

    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(
        benchmarks,
        "gx_growth_rate_from_phi",
        lambda *args, **kwargs: (0.1, 0.2, np.zeros(1), np.zeros(1), np.zeros(1)),
    )

    cfg = KBMBaseCase(grid=GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2))
    run_kbm_linear(ky_target=0.3, cfg=cfg, Nl=2, Nm=2, dt=0.01, steps=4, solver="gx_time")

    assert captured["cfl_fac"] == pytest.approx(benchmarks.ExplicitTimeConfig.cfl_fac)


def test_run_kbm_linear_gx_time_preserves_explicit_cfl_factor(monkeypatch):
    captured: dict[str, float] = {}

    def _fake_integrate(*args, mode_method: str, **_kwargs):
        from spectraxgk.diagnostics import SimulationDiagnostics

        del mode_method
        time_cfg = args[5]
        captured["cfl_fac"] = float(time_cfg.cfl_fac)
        t = np.array([0.1, 0.2], dtype=float)
        phi_t = np.ones((2, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((2, 1, 1), dtype=float)
        omega_t = np.zeros((2, 1, 1), dtype=float)
        diag = SimulationDiagnostics(
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

    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(
        benchmarks,
        "gx_growth_rate_from_phi",
        lambda *args, **kwargs: (0.1, 0.2, np.zeros(1), np.zeros(1), np.zeros(1)),
    )

    cfg = KBMBaseCase(grid=GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2))
    time_cfg = TimeConfig(t_max=0.04, dt=0.01, cfl_fac=1.25)
    run_kbm_linear(
        ky_target=0.3,
        cfg=cfg,
        time_cfg=time_cfg,
        Nl=2,
        Nm=2,
        dt=0.01,
        steps=4,
        solver="gx_time",
    )

    assert captured["cfl_fac"] == pytest.approx(1.25)


def test_run_kbm_linear_gx_time_uses_method_default_cfl_factor(monkeypatch):
    captured: dict[str, float] = {}

    def _fake_integrate(*args, mode_method: str, **_kwargs):
        from spectraxgk.diagnostics import SimulationDiagnostics

        del mode_method
        time_cfg = args[5]
        captured["cfl_fac"] = float(time_cfg.cfl_fac)
        t = np.array([0.1, 0.2], dtype=float)
        phi_t = np.ones((2, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((2, 1, 1), dtype=float)
        omega_t = np.zeros((2, 1, 1), dtype=float)
        diag = SimulationDiagnostics(
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

    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(
        benchmarks,
        "gx_growth_rate_from_phi",
        lambda *args, **kwargs: (0.1, 0.2, np.zeros(1), np.zeros(1), np.zeros(1)),
    )

    cfg = KBMBaseCase(grid=GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2))
    time_cfg = TimeConfig(t_max=0.04, dt=0.01, method="rk3", cfl_fac=None)
    run_kbm_linear(
        ky_target=0.3,
        cfg=cfg,
        time_cfg=time_cfg,
        Nl=2,
        Nm=2,
        dt=0.01,
        steps=4,
        solver="gx_time",
    )

    assert captured["cfl_fac"] == pytest.approx(1.73)


def test_run_kbm_linear_disables_gx_linked_end_damping_when_requested(monkeypatch):
    """Non-GX KBM runs should keep linked-end damping disabled by default."""

    captured: dict[str, float] = {}

    def _fake_two_species_params(*args, damp_ends_amp: float, damp_ends_widthfrac: float, **kwargs):
        del args, kwargs
        captured["amp"] = float(damp_ends_amp)
        captured["width"] = float(damp_ends_widthfrac)
        return SimpleNamespace(
            damp_ends_amp=float(damp_ends_amp),
            damp_ends_widthfrac=float(damp_ends_widthfrac),
            rho_star=benchmarks.KBM_RHO_STAR,
        )

    def _fake_integrate(*_args, **_kwargs):
        return np.array([0.0]), np.zeros((1, 1, 1, 4), dtype=np.complex64)

    monkeypatch.setattr(benchmarks, "_two_species_params", _fake_two_species_params)
    monkeypatch.setattr(benchmarks, "build_linear_cache", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(benchmarks, "integrate_linear", _fake_integrate)
    monkeypatch.setattr(benchmarks, "fit_growth_rate_auto", lambda *args, **kwargs: (0.1, 0.2, 0.0, 0.0))

    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    run_kbm_linear(
        ky_target=0.3,
        cfg=cfg,
        Nl=2,
        Nm=2,
        dt=0.01,
        steps=4,
        solver="time",
        gx_reference=False,
        fit_signal="phi",
    )

    assert captured["amp"] == pytest.approx(0.0)
    assert captured["width"] == pytest.approx(0.0)


def test_run_kbm_beta_scan_gx_time_keeps_project_mode(monkeypatch):
    """KBM scan helpers should not downgrade project mode on the GX-time path."""

    calls: list[str] = []

    def _fake_integrate(*_args, mode_method: str, **_kwargs):
        from spectraxgk.diagnostics import SimulationDiagnostics

        calls.append(f"integrate:{mode_method}")
        t = np.array([0.1, 0.2, 0.3], dtype=float)
        phi_t = np.ones((3, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((3, 1, 1), dtype=float)
        omega_t = np.zeros((3, 1, 1), dtype=float)
        diag = SimulationDiagnostics(
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

    def _fake_extract(phi_t, sel, method: str = "z_index"):
        del phi_t, sel
        calls.append(f"extract:{method}")
        return np.array([1.0 + 0.0j, 1.1 - 0.1j, 1.2 - 0.2j], dtype=np.complex128)

    def _fake_fit_auto(t, signal, **kwargs):
        del t, signal, kwargs
        calls.append("fit:auto")
        return 0.15, 0.9, 0.0, 0.0

    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(benchmarks, "extract_mode_time_series", _fake_extract)
    monkeypatch.setattr(benchmarks, "fit_growth_rate_auto", _fake_fit_auto)
    monkeypatch.setattr(
        benchmarks,
        "gx_growth_rate_from_phi",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected GX ratio fit")),
    )

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

    assert calls == ["integrate:z_index", "extract:project", "fit:auto"]
    assert np.isclose(scan.gamma[0], 0.15)
    assert np.isclose(scan.omega[0], 0.9)


def test_run_kbm_beta_scan_uses_gx_linked_end_damping_by_default(monkeypatch):
    """GX-aligned KBM beta scans should inherit GX linked-end damping defaults."""

    captured: dict[str, float] = {}

    def _fake_two_species_params(*args, damp_ends_amp: float, damp_ends_widthfrac: float, **kwargs):
        del args, kwargs
        captured["amp"] = float(damp_ends_amp)
        captured["width"] = float(damp_ends_widthfrac)
        return SimpleNamespace(
            damp_ends_amp=float(damp_ends_amp),
            damp_ends_widthfrac=float(damp_ends_widthfrac),
            rho_star=benchmarks.KBM_RHO_STAR,
        )

    def _fake_integrate(*_args, mode_method: str, **_kwargs):
        from spectraxgk.diagnostics import SimulationDiagnostics

        del mode_method
        t = np.array([0.1, 0.2], dtype=float)
        phi_t = np.ones((2, 1, 1, 4), dtype=np.complex64)
        gamma_t = np.zeros((2, 1, 1), dtype=float)
        omega_t = np.zeros((2, 1, 1), dtype=float)
        diag = SimulationDiagnostics(
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

    monkeypatch.setattr(benchmarks, "_two_species_params", _fake_two_species_params)
    monkeypatch.setattr(benchmarks, "build_linear_cache", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(benchmarks, "integrate_linear_gx_diagnostics", _fake_integrate)
    monkeypatch.setattr(
        benchmarks,
        "gx_growth_rate_from_phi",
        lambda *args, **kwargs: (0.1, 0.2, np.zeros(1), np.zeros(1), np.zeros(1)),
    )

    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=16, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    run_kbm_beta_scan(
        np.array([cfg.model.beta]),
        ky_target=0.3,
        cfg=cfg,
        Nl=2,
        Nm=2,
        dt=0.01,
        steps=4,
        solver="gx_time",
    )

    assert captured["amp"] == pytest.approx(benchmarks.REFERENCE_DAMP_ENDS_AMP)
    assert captured["width"] == pytest.approx(benchmarks.REFERENCE_DAMP_ENDS_WIDTHFRAC)


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
    time_cfg = TimeConfig(t_max=0.4, dt=0.1, method="rk2", use_diffrax=False, sample_stride=1)
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
        Nl=3,
        Nm=6,
        steps=60,
        dt=0.02,
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
        Nl=3,
        Nm=6,
        steps=60,
        dt=0.02,
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


def test_kinetic_linear_defaults_to_gx_reference_contract(monkeypatch):
    """Kinetic benchmark defaults should keep the GX electrostatic contract."""
    captured: dict[str, object] = {}

    def _fake_dominant_eigenpair(_G0, _cache, _params, *, terms=None, **_kwargs):
        captured["terms"] = terms
        captured["params"] = _params
        captured["mode_family"] = _kwargs.get("mode_family")
        captured["omega_sign"] = _kwargs.get("omega_sign")
        captured["shift_source"] = _kwargs.get("shift_source")
        return 0.1 + 0.2j, np.zeros((4, 4, 1, 1, 8), dtype=np.complex64)

    def _fake_compute_fields_cached(_vec, _cache, _params, *, terms=None):
        return type("Fields", (), {"phi": np.zeros((1, 1, 8), dtype=np.complex64)})()

    monkeypatch.setattr(benchmarks, "dominant_eigenpair", _fake_dominant_eigenpair)
    monkeypatch.setattr(benchmarks, "compute_fields_cached", _fake_compute_fields_cached)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    cfg = KineticElectronBaseCase(grid=grid)
    run_kinetic_linear(cfg=cfg, ky_target=0.3, Nl=4, Nm=4, solver="krylov")
    terms = captured["terms"]
    params = captured["params"]
    assert terms is not None
    assert terms.bpar == 0.0
    assert float(params.damp_ends_amp) == pytest.approx(benchmarks.REFERENCE_DAMP_ENDS_AMP)
    assert float(params.damp_ends_widthfrac) == pytest.approx(benchmarks.REFERENCE_DAMP_ENDS_WIDTHFRAC)
    assert float(params.nu_hyper_l) == pytest.approx(benchmarks.REFERENCE_NU_HYPER_L)
    assert float(params.nu_hyper_m) == pytest.approx(benchmarks.REFERENCE_NU_HYPER_M)
    assert captured["mode_family"] == "cyclone"
    assert captured["omega_sign"] == 1
    assert captured["shift_source"] == "history"


def test_kinetic_linear_defaults_to_legacy_reference_seed(monkeypatch):
    """Default kinetic GX-reference helpers should restore the historical density seed."""

    captured: dict[str, object] = {}

    def _fake_build_initial_condition(_grid, _geom, *, init_cfg, **_kwargs):
        captured["init_cfg"] = init_cfg
        return np.zeros((4, 4, 1, 1, 8), dtype=np.complex64)

    def _fake_dominant_eigenpair(_G0, _cache, _params, *, terms=None, **_kwargs):
        return 0.1 + 0.2j, np.zeros((4, 4, 1, 1, 8), dtype=np.complex64)

    def _fake_compute_fields_cached(_vec, _cache, _params, *, terms=None):
        return type("Fields", (), {"phi": np.zeros((1, 1, 8), dtype=np.complex64)})()

    monkeypatch.setattr(benchmarks, "_build_initial_condition", _fake_build_initial_condition)
    monkeypatch.setattr(benchmarks, "dominant_eigenpair", _fake_dominant_eigenpair)
    monkeypatch.setattr(benchmarks, "compute_fields_cached", _fake_compute_fields_cached)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    cfg = KineticElectronBaseCase(grid=grid)
    run_kinetic_linear(cfg=cfg, ky_target=0.3, Nl=4, Nm=4, solver="krylov")

    init_cfg = captured["init_cfg"]
    assert init_cfg.init_field == "density"
    assert init_cfg.init_amp == pytest.approx(1.0e-3)
    assert init_cfg.gaussian_init is False


def test_kinetic_linear_respects_explicit_user_seed(monkeypatch):
    """Explicit kinetic init overrides should not be replaced by the legacy parity seed."""

    captured: dict[str, object] = {}

    def _fake_build_initial_condition(_grid, _geom, *, init_cfg, **_kwargs):
        captured["init_cfg"] = init_cfg
        return np.zeros((4, 4, 1, 1, 8), dtype=np.complex64)

    def _fake_dominant_eigenpair(_G0, _cache, _params, *, terms=None, **_kwargs):
        return 0.1 + 0.2j, np.zeros((4, 4, 1, 1, 8), dtype=np.complex64)

    def _fake_compute_fields_cached(_vec, _cache, _params, *, terms=None):
        return type("Fields", (), {"phi": np.zeros((1, 1, 8), dtype=np.complex64)})()

    monkeypatch.setattr(benchmarks, "_build_initial_condition", _fake_build_initial_condition)
    monkeypatch.setattr(benchmarks, "dominant_eigenpair", _fake_dominant_eigenpair)
    monkeypatch.setattr(benchmarks, "compute_fields_cached", _fake_compute_fields_cached)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    custom_init = InitializationConfig(init_field="density", init_amp=1.0e-7, gaussian_init=True)
    cfg = KineticElectronBaseCase(grid=grid, init=custom_init)
    run_kinetic_linear(cfg=cfg, ky_target=0.3, Nl=4, Nm=4, solver="krylov")

    assert captured["init_cfg"] == custom_init


def test_tem_linear_defaults_to_bpar_disabled_terms(monkeypatch):
    """TEM benchmark defaults should disable bpar coupling by default."""
    captured: dict[str, object] = {}

    def _fake_dominant_eigenpair(_G0, _cache, _params, *, terms=None, **_kwargs):
        captured["terms"] = terms
        return 0.1 + 0.2j, np.zeros((4, 4, 1, 1, 8), dtype=np.complex64)

    def _fake_compute_fields_cached(_vec, _cache, _params, *, terms=None):
        return type("Fields", (), {"phi": np.zeros((1, 1, 8), dtype=np.complex64)})()

    monkeypatch.setattr(benchmarks, "dominant_eigenpair", _fake_dominant_eigenpair)
    monkeypatch.setattr(benchmarks, "compute_fields_cached", _fake_compute_fields_cached)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=62.8, Ly=62.8, ntheta=8, nperiod=1, y0=10.0)
    cfg = TEMBaseCase(grid=grid)
    run_tem_linear(cfg=cfg, ky_target=0.3, Nl=4, Nm=4, solver="krylov")
    terms = captured["terms"]
    assert terms is not None
    assert terms.bpar == 0.0


@pytest.mark.slow
def test_benchmark_krylov_smoke_finite():
    """Krylov solves should return finite gamma/omega for core benchmarks."""
    krylov_cfg = KrylovConfig(
        method="propagator",
        krylov_dim=8,
        restarts=1,
        power_iters=20,
        power_dt=0.01,
        shift=complex(0.05, -0.3),
        shift_source="target",
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
    result = run_etg_linear(cfg=cfg, params=params, ky_target=3.0, Nl=3, Nm=6, steps=60, dt=0.02)
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
        Nl=3,
        Nm=6,
        steps=60,
        dt=0.02,
        method="rk4",
    )
    assert np.isfinite(scan.gamma[0])


def test_etg_linear_defaults_to_electrostatic_terms(monkeypatch):
    from types import SimpleNamespace
    import spectraxgk.benchmarks as benchmarks

    captured = {}

    def fake_dominant_eigenpair(G0, cache, params, terms=None, **_kwargs):
        captured["terms"] = terms
        return jnp.asarray(0.2 - 0.3j, dtype=jnp.complex64), jnp.zeros_like(G0)

    def fake_compute_fields_cached(vec, cache, params, terms=None):
        return SimpleNamespace(phi=jnp.zeros(vec.shape[-3:], dtype=jnp.complex64))

    monkeypatch.setattr(benchmarks, "dominant_eigenpair", fake_dominant_eigenpair)
    monkeypatch.setattr(benchmarks, "compute_fields_cached", fake_compute_fields_cached)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=0.628)
    cfg = ETGBaseCase(
        grid=grid,
        model=ETGModelConfig(
            R_over_LTi=0.0,
            R_over_LTe=2.49,
            R_over_Ln=0.8,
            R_over_Lni=0.0,
            R_over_Lne=0.8,
            adiabatic_ions=False,
            mass_ratio=3670.0,
        ),
    )
    out = run_etg_linear(cfg=cfg, ky_target=3.0, Nl=2, Nm=2, solver="krylov")

    assert np.isfinite(out.gamma)
    assert captured["terms"] == LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)


def test_etg_scan_defaults_to_electrostatic_terms(monkeypatch):
    import spectraxgk.benchmarks as benchmarks

    captured = {}

    def fake_dominant_eigenpair(G0, cache, params, terms=None, **_kwargs):
        idx = int(captured.get("calls", 0))
        captured["calls"] = idx + 1
        captured.setdefault("terms", []).append(terms)
        eig = jnp.asarray((0.1 + idx) - 0.2j, dtype=jnp.complex64)
        return eig, jnp.zeros_like(G0)

    monkeypatch.setattr(benchmarks, "dominant_eigenpair", fake_dominant_eigenpair)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=0.628)
    cfg = ETGBaseCase(
        grid=grid,
        model=ETGModelConfig(
            R_over_LTi=0.0,
            R_over_LTe=2.49,
            R_over_Ln=0.8,
            R_over_Lni=0.0,
            R_over_Lne=0.8,
            adiabatic_ions=False,
            mass_ratio=3670.0,
        ),
    )
    scan = run_etg_scan(np.array([3.0, 4.0]), cfg=cfg, Nl=2, Nm=2, solver="krylov")

    assert np.all(np.isfinite(scan.gamma))
    assert captured["calls"] == 2
    assert captured["terms"] == [
        LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0),
        LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0),
    ]


def test_run_etg_scan_continuation_uses_shift_selection_for_carried_shift(monkeypatch):
    import spectraxgk.benchmarks as benchmarks

    calls: list[dict[str, object]] = []

    def fake_dominant_eigenpair(G0, cache, params, terms=None, **kwargs):
        calls.append(dict(kwargs))
        eig = jnp.asarray((0.2 + 0.1 * len(calls)) - 0.3j, dtype=jnp.complex64)
        vec = jnp.ones_like(G0) * (1.0 + 0.0j)
        return eig, vec

    monkeypatch.setattr(benchmarks, "dominant_eigenpair", fake_dominant_eigenpair)

    grid = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=0.628)
    cfg = ETGBaseCase(
        grid=grid,
        model=ETGModelConfig(
            R_over_LTi=0.0,
            R_over_LTe=2.49,
            R_over_Ln=0.8,
            R_over_Lni=0.0,
            R_over_Lne=0.8,
            adiabatic_ions=False,
            mass_ratio=3670.0,
        ),
    )
    kcfg = KrylovConfig(
        method="shift_invert",
        shift=None,
        shift_source="target",
        shift_selection="targeted",
        mode_family="etg",
        continuation=True,
        continuation_selection="overlap",
    )

    scan = run_etg_scan(
        np.array([3.0, 4.0]),
        cfg=cfg,
        Nl=2,
        Nm=2,
        solver="krylov",
        krylov_cfg=kcfg,
    )

    assert np.all(np.isfinite(scan.gamma))
    assert len(calls) == 2
    assert calls[0]["shift_selection"] == "targeted"
    assert complex(calls[1]["shift"]) == pytest.approx(complex(0.3, -0.3))
    assert calls[1]["shift_selection"] == "shift"
    assert calls[1]["select_overlap"] is True
