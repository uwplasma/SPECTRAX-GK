from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.analysis import ModeSelection
from spectraxgk.benchmarks import (
    InitializationConfig,
    KrylovConfig,
    _apply_gx_hypercollisions,
    _build_gaussian_profile,
    _build_initial_condition,
    _electron_only_params,
    _extract_mode_only_signal,
    _gx_linked_end_damping,
    _gx_p_hyper_m,
    _is_array_like,
    _iter_ky_batches,
    _kbm_use_multi_target_krylov,
    _kinetic_reference_init_cfg,
    _load_reference_with_header,
    _midplane_index,
    _normalize_growth_rate,
    _resolve_streaming_window,
    _score_fit_signal_auto,
    _select_fit_signal,
    _select_fit_signal_auto,
    _two_species_params,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
    select_kbm_solver_auto,
)
from spectraxgk.config import KineticElectronBaseCase as KineticBaseConfig
from spectraxgk.linear import LinearParams


def _linear_params() -> LinearParams:
    return LinearParams(
        charge_sign=np.array([1.0]),
        mass=np.array([1.0]),
        density=np.array([1.0]),
        temp=np.array([1.0]),
        nu=np.array([0.0]),
        tau_e=1.0,
        vth=np.array([1.0]),
        rho=np.array([1.0]),
        kpar_scale=1.0,
        R_over_Ln=np.array([1.0]),
        R_over_LTi=np.array([1.0]),
        R_over_LTe=np.array([1.0]),
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        energy_const=1.0,
        energy_par_coef=1.0,
        energy_perp_coef=1.0,
        nu_hermite=0.0,
        nu_laguerre=0.0,
        rho_star=1.0,
        beta=0.0,
        fapar=0.0,
        apar_beta_scale=0.5,
        ampere_g0_scale=0.5,
        bpar_beta_scale=0.5,
        nu_hyper=1.0,
        nu_hyper_l=0.0,
        nu_hyper_m=0.0,
        nu_hyper_lm=0.0,
        p_hyper=2.0,
        p_hyper_l=2.0,
        p_hyper_m=2.0,
        p_hyper_lm=2.0,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        p_hyper_kperp=2.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        tz=np.array([1.0]),
    )


def test_reference_loaders_return_data() -> None:
    for loader in (
        load_cyclone_reference,
        load_cyclone_reference_kinetic,
        load_kbm_reference,
        load_etg_reference,
        load_tem_reference,
    ):
        ref = loader()
        assert ref.ky.size > 0
        assert ref.gamma.shape == ref.omega.shape == ref.ky.shape


def test_load_reference_with_header_reads_named_columns(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "demo.csv").write_text("ky,gamma,omega\n0.1,0.2,-0.3\n", encoding="utf-8")

    class FakeFiles:
        def joinpath(self, *parts):
            return data_dir / parts[-1]

    monkeypatch.setattr("spectraxgk.benchmarks.resources.files", lambda _pkg: FakeFiles())
    ref = _load_reference_with_header("demo.csv")
    np.testing.assert_allclose(ref.ky, [0.1])
    np.testing.assert_allclose(ref.gamma, [0.2])
    np.testing.assert_allclose(ref.omega, [-0.3])


def test_gx_hypercollision_helpers() -> None:
    params = _apply_gx_hypercollisions(_linear_params(), nhermite=12)
    assert _gx_p_hyper_m(None) == 20.0
    assert _gx_p_hyper_m(1) == 1.0
    assert _gx_p_hyper_m(12) == 6.0
    assert params.nu_hyper == 0.0
    assert params.nu_hyper_m == 1.0
    assert params.hypercollisions_kz == 1.0
    assert params.p_hyper_m == 6.0


def test_gx_linked_end_damping_and_midplane_index() -> None:
    assert _gx_linked_end_damping(True) == (0.1, 0.125)
    assert _gx_linked_end_damping(False) == (0.0, 0.0)
    assert _midplane_index(SimpleNamespace(z=np.array([0.0]))) == 0
    assert _midplane_index(SimpleNamespace(z=np.arange(6))) == 4


def test_select_kbm_solver_auto() -> None:
    assert select_kbm_solver_auto("time", ky_target=0.2, gx_reference=True) == "time"
    assert select_kbm_solver_auto("auto", ky_target=0.3, gx_reference=True) == "gx_time"
    assert select_kbm_solver_auto("auto", ky_target=0.7, gx_reference=False) == "time"


def test_select_fit_signal_and_auto(monkeypatch) -> None:
    phi_t = np.ones((4, 1, 1, 1), dtype=np.complex128)
    density_t = 2.0 * phi_t
    sel = ModeSelection(ky_index=0, kx_index=0)

    queue = [
        np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.complex128),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128),
        np.array([4.0, 3.0, 2.0, 1.0], dtype=np.complex128),
    ]
    monkeypatch.setattr("spectraxgk.benchmarks.extract_mode_time_series", lambda *args, **kwargs: queue.pop(0))
    signal = _select_fit_signal(phi_t, density_t, sel, fit_signal="phi", mode_method="project")
    np.testing.assert_allclose(signal, [1.0, 2.0, 3.0, 4.0])

    queue = [
        np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.complex128),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128),
    ]
    monkeypatch.setattr("spectraxgk.benchmarks.extract_mode_time_series", lambda *args, **kwargs: queue.pop(0))
    signal = _select_fit_signal(density_t, phi_t, sel, fit_signal="density", mode_method="project")
    np.testing.assert_allclose(signal, [1.0, 2.0, 3.0, 4.0])

    queue = [np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.complex128)]
    monkeypatch.setattr("spectraxgk.benchmarks.extract_mode_time_series", lambda *args, **kwargs: queue.pop(0))
    with pytest.warns(RuntimeWarning, match="insufficient finite"):
        signal = _select_fit_signal(phi_t, None, sel, fit_signal="phi", mode_method="project")
    np.testing.assert_allclose(signal, np.zeros(4))

    queue = [np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.complex128)]
    monkeypatch.setattr("spectraxgk.benchmarks.extract_mode_time_series", lambda *args, **kwargs: queue.pop(0))
    with pytest.warns(RuntimeWarning, match="insufficient finite"):
        signal = _select_fit_signal(phi_t, density_t, sel, fit_signal="density", mode_method="project", fallback=False)
    np.testing.assert_allclose(signal, np.zeros(4))

    queue = [np.array([1.0, 2.0], dtype=np.complex128)]
    monkeypatch.setattr("spectraxgk.benchmarks.extract_mode_time_series", lambda *args, **kwargs: queue.pop(0))
    with pytest.raises(ValueError):
        _select_fit_signal(phi_t, None, sel, fit_signal="density", mode_method="project")
    with pytest.raises(ValueError):
        _select_fit_signal(phi_t, density_t, sel, fit_signal="bad", mode_method="project")

    signals = {
        "phi": np.array([1.0, 2.0, 3.0], dtype=np.complex128),
        "density": np.array([3.0, 2.0, 1.0], dtype=np.complex128),
    }

    def fake_extract(arr, _sel, method):
        return signals["density" if arr is density_t else "phi"]

    def fake_score(t, signal, **kwargs):
        assert kwargs["num_windows"] == 4
        if np.allclose(signal, signals["phi"]):
            return 0.1, 0.2, 0.3
        return 0.4, 0.5, 0.8

    monkeypatch.setattr("spectraxgk.benchmarks.extract_mode_time_series", fake_extract)
    monkeypatch.setattr("spectraxgk.benchmarks._score_fit_signal_auto", fake_score)
    signal, name, gamma, omega = _select_fit_signal_auto(
        np.array([0.0, 1.0, 2.0]),
        phi_t,
        density_t,
        sel,
        mode_method="project",
        tmin=None,
        tmax=None,
        window_fraction=0.5,
        min_points=2,
        start_fraction=0.2,
        growth_weight=1.0,
        require_positive=True,
        min_amp_fraction=0.0,
        max_amp_fraction=1.0,
        window_method="rolling",
        max_fraction=1.0,
        end_fraction=1.0,
        num_windows=4,
        phase_weight=0.5,
        length_weight=0.5,
        min_r2=0.0,
        late_penalty=0.0,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )
    assert name == "density"
    np.testing.assert_allclose(signal, signals["density"])
    assert gamma == 0.4
    assert omega == 0.5


def test_score_fit_signal_auto_filters_invalid(monkeypatch) -> None:
    captured: dict[str, object] = {}
    def _fake_fit(*args, **kwargs):
        captured["num_windows"] = kwargs["num_windows"]
        return (0.3, -0.2, 0.0, 1.0, 0.95, 0.9)

    monkeypatch.setattr(
        "spectraxgk.benchmarks.fit_growth_rate_auto_with_stats",
        _fake_fit,
    )
    gamma, omega, score = _score_fit_signal_auto(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 4.0], dtype=np.complex128),
        tmin=None,
        tmax=None,
        window_fraction=0.5,
        min_points=2,
        start_fraction=0.2,
        growth_weight=1.0,
        require_positive=True,
        min_amp_fraction=0.0,
        max_amp_fraction=1.0,
        window_method="rolling",
        max_fraction=1.0,
        end_fraction=1.0,
        num_windows=6,
        phase_weight=0.5,
        length_weight=0.5,
        min_r2=0.8,
        late_penalty=0.0,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )
    assert gamma == 0.3
    assert omega == -0.2
    assert score > 0.0
    assert captured["num_windows"] == 6

    monkeypatch.setattr(
        "spectraxgk.benchmarks.fit_growth_rate_auto_with_stats",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    gamma, omega, score = _score_fit_signal_auto(
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0], dtype=np.complex128),
        tmin=None,
        tmax=None,
        window_fraction=0.5,
        min_points=2,
        start_fraction=0.2,
        growth_weight=1.0,
        require_positive=True,
        min_amp_fraction=0.0,
        max_amp_fraction=1.0,
        window_method="rolling",
        max_fraction=1.0,
        end_fraction=1.0,
        num_windows=6,
        phase_weight=0.5,
        length_weight=0.5,
        min_r2=0.8,
        late_penalty=0.0,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )
    assert score == -np.inf


def test_mode_signal_batch_and_window_helpers() -> None:
    arr = np.arange(12).reshape(3, 4)
    np.testing.assert_allclose(_extract_mode_only_signal(arr, local_idx=2), [2, 6, 10])
    arr3 = np.arange(24).reshape(2, 3, 4)
    np.testing.assert_allclose(_extract_mode_only_signal(arr3, local_idx=1, species_index=1), [5, 17])
    arr4 = np.arange(48).reshape(2, 2, 3, 4)
    np.testing.assert_allclose(_extract_mode_only_signal(arr4, local_idx=4), [4, 28])
    np.testing.assert_allclose(_extract_mode_only_signal(np.array(3.0 + 1.0j), local_idx=0), [3.0 + 1.0j])
    np.testing.assert_allclose(_extract_mode_only_signal(np.array([1.0, 2.0]), local_idx=1), [1.0, 2.0])
    assert _is_array_like([1, 2]) is True
    assert _is_array_like(np.array([1, 2])) is True
    assert _is_array_like(1.0) is False

    single_batches = list(_iter_ky_batches(np.array([0.1, 0.2]), ky_batch=1, fixed_batch_shape=False))
    assert len(single_batches) == 2
    assert single_batches[0][0] == 0
    np.testing.assert_allclose(single_batches[0][1], [0.1])
    assert single_batches[0][2] == 1

    batches = list(_iter_ky_batches(np.array([0.1, 0.2, 0.3]), ky_batch=2, fixed_batch_shape=True))
    assert batches[0][0] == 0
    np.testing.assert_allclose(batches[0][1], [0.1, 0.2])
    np.testing.assert_allclose(batches[1][1], [0.3, 0.3])
    assert batches[1][2] == 1

    ragged_batches = list(_iter_ky_batches(np.array([0.1, 0.2, 0.3]), ky_batch=2, fixed_batch_shape=False))
    np.testing.assert_allclose(ragged_batches[1][1], [0.3])
    assert ragged_batches[1][2] == 1

    assert _resolve_streaming_window(10.0, None, None, 0.2, 0.1, 0.9) == (2.0, 3.0)
    assert _resolve_streaming_window(10.0, 1.0, 4.0, 0.2, 0.1, 0.9) == (1.0, 4.0)
    assert _resolve_streaming_window(10.0, None, None, 0.9, 0.05, 0.2) == (9.0, 10.0)


def test_normalization_and_initial_profiles() -> None:
    gamma, omega = _normalize_growth_rate(0.4, -0.2, _linear_params(), "gx")
    assert np.isfinite(gamma)
    assert np.isfinite(omega)

    init_cfg = InitializationConfig(
        init_field="density",
        init_amp=1.5,
        gaussian_init=True,
        gaussian_width=0.3,
        gaussian_envelope_constant=1.0,
        gaussian_envelope_sine=0.1,
    )
    z = np.linspace(-1.0, 1.0, 5)
    profile = _build_gaussian_profile(z, kx=0.2, ky=0.4, s_hat=0.5, init_cfg=init_cfg)
    assert profile.shape == z.shape
    assert np.max(np.abs(profile)) > 0.0
    with pytest.raises(ValueError):
        _build_gaussian_profile(z, kx=0.2, ky=0.4, s_hat=0.5, init_cfg=SimpleNamespace(**{**init_cfg.__dict__, "gaussian_width": 0.0}))


def test_build_initial_condition_supports_all_and_invalid_fields() -> None:
    grid = SimpleNamespace(
        kx=np.array([0.0, 0.5]),
        ky=np.array([0.0, 0.4]),
        z=np.linspace(-1.0, 1.0, 5),
    )
    geom = SimpleNamespace(s_hat=0.8)
    init_cfg = InitializationConfig(init_field="all", init_amp=2.0, gaussian_init=False)
    G0 = _build_initial_condition(grid, geom, ky_index=[0, 1], kx_index=1, Nl=2, Nm=4, init_cfg=init_cfg)
    assert G0.shape == (2, 4, 2, 2, 5)
    assert np.count_nonzero(np.asarray(G0)[:, :, 0, 1, :]) == 0
    assert np.count_nonzero(np.asarray(G0)[:, :, 1, 1, :]) > 0

    too_small = InitializationConfig(init_field="qpar", init_amp=1.0, gaussian_init=False)
    with pytest.raises(ValueError, match="moment exceeds"):
        _build_initial_condition(grid, geom, ky_index=1, kx_index=1, Nl=1, Nm=1, init_cfg=too_small)

    bad = InitializationConfig(init_field="banana")
    with pytest.raises(ValueError):
        _build_initial_condition(grid, geom, ky_index=1, kx_index=1, Nl=2, Nm=4, init_cfg=bad)


def test_kinetic_init_and_kbm_target_helpers() -> None:
    default_init = KineticBaseConfig().init
    replaced = _kinetic_reference_init_cfg(default_init, gx_reference=True)
    assert replaced.init_amp == pytest.approx(1.0e-3)
    assert replaced.gaussian_init is False
    assert _kinetic_reference_init_cfg(default_init, gx_reference=False) == default_init
    custom = InitializationConfig(init_field="upar", init_amp=2.0)
    assert _kinetic_reference_init_cfg(custom, gx_reference=True) == custom

    kcfg = KrylovConfig(method="shift_invert", mode_family="kbm", shift_selection="target")
    assert _kbm_use_multi_target_krylov(kcfg, [0.1, 0.2], shift=None) is True
    assert _kbm_use_multi_target_krylov(kcfg, None, shift=None) is False
    assert _kbm_use_multi_target_krylov(kcfg, [0.1], shift=1.0 + 0.0j) is False
    assert _kbm_use_multi_target_krylov(KrylovConfig(method="arnoldi", mode_family="kbm"), [0.1], shift=None) is False
    assert _kbm_use_multi_target_krylov(KrylovConfig(method="shift_invert", mode_family="etg"), [0.1], shift=None) is False
    assert (
        _kbm_use_multi_target_krylov(
            KrylovConfig(method="shift_invert", mode_family="kbm", shift_selection="shift"),
            [0.1],
            shift=None,
        )
        is False
    )


def test_species_param_builders() -> None:
    model = SimpleNamespace(
        mass_ratio=1836.0,
        Te_over_Ti=1.2,
        R_over_Ln=1.0,
        R_over_Lni=1.1,
        R_over_Lne=0.9,
        R_over_LTi=2.0,
        R_over_LTe=3.0,
        nu_i=0.01,
        nu_e=0.02,
        beta=2.0e-3,
    )
    params = _two_species_params(
        model,
        kpar_scale=1.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        beta_override=1.0e-3,
        fapar_override=0.5,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=0.2,
        nhermite=10,
        apar_beta_scale=0.7,
        ampere_g0_scale=0.8,
        bpar_beta_scale=0.9,
    )
    assert np.asarray(params.charge_sign).shape == (2,)
    assert params.beta == pytest.approx(1.0e-3)
    assert params.fapar == pytest.approx(0.5)
    assert params.damp_ends_amp == pytest.approx(0.1)

    eparams = _electron_only_params(
        model,
        kpar_scale=1.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        beta_override=0.0,
        fapar_override=0.25,
        damp_ends_amp=0.05,
        damp_ends_widthfrac=0.15,
    )
    assert np.asarray(eparams.charge_sign).shape == (1,)
    assert eparams.tau_e == pytest.approx(model.Te_over_Ti)
    assert eparams.beta == pytest.approx(0.0)
    assert eparams.fapar == pytest.approx(0.25)
    assert eparams.damp_ends_widthfrac == pytest.approx(0.15)

    for bad_mass in (0.0, -1.0):
        with pytest.raises(ValueError):
            _two_species_params(
                SimpleNamespace(**{**model.__dict__, "mass_ratio": bad_mass}),
                kpar_scale=1.0,
                omega_d_scale=1.0,
                omega_star_scale=1.0,
                rho_star=1.0,
            )
    with pytest.raises(ValueError):
        _electron_only_params(
            SimpleNamespace(**{**model.__dict__, "Te_over_Ti": 0.0}),
            kpar_scale=1.0,
            omega_d_scale=1.0,
            omega_star_scale=1.0,
            rho_star=1.0,
        )
    with pytest.raises(ValueError):
        _two_species_params(
            SimpleNamespace(**{**model.__dict__, "Te_over_Ti": -1.0}),
            kpar_scale=1.0,
            omega_d_scale=1.0,
            omega_star_scale=1.0,
            rho_star=1.0,
        )
    with pytest.raises(ValueError):
        _electron_only_params(
            SimpleNamespace(**{**model.__dict__, "mass_ratio": 0.0}),
            kpar_scale=1.0,
            omega_d_scale=1.0,
            omega_star_scale=1.0,
            rho_star=1.0,
        )


def test_score_fit_signal_auto_rejects_nonfinite_and_negative_growth(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.benchmarks.fit_growth_rate_auto_with_stats",
        lambda *args, **kwargs: (np.nan, -0.2, 0.0, 1.0, 0.95, 0.9),
    )
    gamma, omega, score = _score_fit_signal_auto(
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0], dtype=np.complex128),
        tmin=None,
        tmax=None,
        window_fraction=0.5,
        min_points=2,
        start_fraction=0.2,
        growth_weight=1.0,
        require_positive=True,
        min_amp_fraction=0.0,
        max_amp_fraction=1.0,
        window_method="rolling",
        max_fraction=1.0,
        end_fraction=1.0,
        num_windows=4,
        phase_weight=0.5,
        length_weight=0.5,
        min_r2=0.8,
        late_penalty=0.0,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )
    assert np.isnan(gamma)
    assert omega == pytest.approx(-0.2)
    assert score == -np.inf

    monkeypatch.setattr(
        "spectraxgk.benchmarks.fit_growth_rate_auto_with_stats",
        lambda *args, **kwargs: (-0.1, -0.2, 0.0, 1.0, 0.95, 0.9),
    )
    gamma, omega, score = _score_fit_signal_auto(
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0], dtype=np.complex128),
        tmin=None,
        tmax=None,
        window_fraction=0.5,
        min_points=2,
        start_fraction=0.2,
        growth_weight=1.0,
        require_positive=True,
        min_amp_fraction=0.0,
        max_amp_fraction=1.0,
        window_method="rolling",
        max_fraction=1.0,
        end_fraction=1.0,
        num_windows=4,
        phase_weight=0.5,
        length_weight=0.5,
        min_r2=0.8,
        late_penalty=0.0,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )
    assert gamma == pytest.approx(-0.1)
    assert omega == pytest.approx(-0.2)
    assert score == -np.inf
