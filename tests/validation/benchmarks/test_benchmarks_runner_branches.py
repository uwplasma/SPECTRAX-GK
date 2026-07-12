from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.config import (
    KBMBaseCase,
)
from spectraxgk.linear import LinearTerms
from spectraxgk.benchmarks import (
    KrylovConfig,
    compare_cyclone_to_reference,
    run_kbm_beta_scan,
    run_kbm_linear,
    run_kbm_scan,
)


def _grid_full():
    return SimpleNamespace(
        ky=np.array([0.0, 0.3], dtype=float),
        kx=np.array([0.0], dtype=float),
        z=np.array([-1.0, 0.0, 1.0], dtype=float),
        dealias_mask=np.array([[True], [True]], dtype=bool),
    )


def _grid_sel():
    return SimpleNamespace(
        ky=np.array([0.3], dtype=float),
        kx=np.array([0.0], dtype=float),
        z=np.array([-1.0, 0.0, 1.0], dtype=float),
        dealias_mask=np.array([[True]], dtype=bool),
    )


def _select_grid_dynamic(grid, idx):
    ky_idx = np.atleast_1d(np.asarray(idx, dtype=int))
    return SimpleNamespace(
        ky=np.asarray(grid.ky)[ky_idx],
        kx=np.asarray(grid.kx),
        z=np.asarray(grid.z),
        dealias_mask=np.ones((ky_idx.size, np.asarray(grid.kx).size), dtype=bool),
    )


def _fake_initial_condition(grid, *args, **kwargs):
    nl = int(kwargs.get("Nl", 2))
    nm = int(kwargs.get("Nm", 2))
    return np.zeros(
        (
            nl,
            nm,
            np.asarray(grid.ky).size,
            np.asarray(grid.kx).size,
            np.asarray(grid.z).size,
        ),
        dtype=np.complex64,
    )


def _benchmark_module_attr(module: str, attr: str) -> str:
    if module in {
        "cyclone_linear",
        "cyclone_scan",
        "cyclone_scan_branches",
        "etg_linear",
        "etg_scan",
        "kinetic_linear",
        "kinetic_scan",
        "kbm_beta",
        "kbm_linear",
        "tem",
    }:
        return f"spectraxgk.benchmarks.{attr}"
    return f"spectraxgk.benchmarks.{attr}"


def _patch_attr(monkeypatch, module: str, attr: str, value) -> None:
    monkeypatch.setattr(_benchmark_module_attr(module, attr), value)


def _fake_salpha_geometry(cfg):
    return SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8)


def _fake_cache(*args, **kwargs):
    return SimpleNamespace()


def _fake_fields(*args, **kwargs):
    return SimpleNamespace(phi=np.ones((1, 1, 3), dtype=np.complex64))


def _identity_normalization(gamma, omega, params, norm):
    return gamma, omega


def _patch_salpha_scaffold(
    monkeypatch,
    module: str,
    *,
    select_grid=lambda grid, idx: _grid_sel(),
    init=_fake_initial_condition,
    with_cache: bool = True,
    with_normalization: bool = True,
) -> None:
    _patch_attr(monkeypatch, module, "build_spectral_grid", lambda cfg: _grid_full())
    _patch_attr(monkeypatch, module, "select_ky_grid", select_grid)
    _patch_attr(
        monkeypatch, module, "SAlphaGeometry.from_config", _fake_salpha_geometry
    )
    _patch_attr(monkeypatch, module, "_build_initial_condition", init)
    if with_cache:
        _patch_attr(monkeypatch, module, "build_linear_cache", _fake_cache)
    if with_normalization:
        _patch_attr(
            monkeypatch, module, "_normalize_growth_rate", _identity_normalization
        )


def _patch_krylov_output(
    monkeypatch,
    module: str,
    eig: complex,
    *,
    shape=(2, 2, 1, 1, 3),
) -> None:
    _patch_attr(
        monkeypatch,
        module,
        "dominant_eigenpair",
        lambda *args, **kwargs: (eig, np.ones(shape, dtype=np.complex64)),
    )
    _patch_attr(monkeypatch, module, "compute_fields_cached", _fake_fields)
    _patch_attr(
        monkeypatch, module, "linear_terms_to_term_config", lambda terms: object()
    )


def test_run_kbm_linear_explicit_time_uses_omega_series_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_flux_tube_geometry",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.apply_geometry_grid_defaults",
        lambda geom, grid: grid,
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_spectral_grid",
        lambda cfg: _grid_full(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.select_ky_grid",
        lambda grid, idx: _grid_sel(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.integrate_linear_explicit_diagnostics",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0]),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            np.array([[[0.1, 0.2]]]),
            np.array([[[0.0, -0.3]]]),
            None,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no fit")),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.windowed_growth_rate_from_omega_series",
        lambda *args, **kwargs: (0.35, -0.22, None, None),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_kbm_linear(
        cfg=KBMBaseCase(),
        solver="explicit_time",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        mode_method="z_index",
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
        ky_target=0.3,
    )

    assert result.gamma == 0.35
    assert result.omega == -0.22


def test_run_kbm_beta_scan_multi_target_resolves_near_marginal_branch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_spectral_grid",
        lambda cfg: _grid_full(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.select_ky_grid",
        lambda grid, idx: _grid_sel(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._two_species_params",
        lambda *args, **kwargs: SimpleNamespace(rho_star=1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._build_initial_condition",
        _fake_initial_condition,
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    target_calls: list[float] = []
    eigs = iter([0.0 + 0.2j, 0.12 + 1.1j])

    def _fake_eigenpair(G0, *args, **kwargs):
        target_calls.append(float(kwargs["omega_target_factor"]))
        return next(eigs), np.zeros_like(np.asarray(G0))

    monkeypatch.setattr("spectraxgk.benchmarks.dominant_eigenpair", _fake_eigenpair)

    scan = run_kbm_beta_scan(
        np.array([0.015]),
        cfg=KBMBaseCase(),
        solver="krylov",
        krylov_cfg=KrylovConfig(
            method="shift_invert",
            mode_family="kbm",
            shift_selection="targeted",
            omega_sign=-1,
        ),
        kbm_target_factors=(0.7, 1.5),
        kbm_beta_transition=float("nan"),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=1,
    )

    assert target_calls == [0.7, 1.5]
    np.testing.assert_allclose(scan.ky, [0.015])
    np.testing.assert_allclose(scan.gamma, [0.12])
    np.testing.assert_allclose(scan.omega, [-1.1])


def test_compare_cyclone_to_reference_handles_zero_reference() -> None:
    result = SimpleNamespace(gamma=0.2, omega=-0.1, ky=0.3)
    reference = SimpleNamespace(
        ky=np.array([0.3]),
        gamma=np.array([0.0]),
        omega=np.array([0.0]),
    )
    comparison = compare_cyclone_to_reference(result, reference)
    assert np.isnan(comparison.rel_gamma)
    assert np.isnan(comparison.rel_omega)





def test_run_kbm_scan_forwards_per_mode_arrays(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_kbm_beta_scan(**kwargs):
        calls.append(kwargs)
        ky = float(kwargs["ky_target"])
        return SimpleNamespace(gamma=np.array([ky + 1.0]), omega=np.array([-ky - 2.0]))

    monkeypatch.setattr(
        "spectraxgk.benchmarks.run_kbm_beta_scan",
        _fake_run_kbm_beta_scan,
    )

    scan = run_kbm_scan(
        np.array([0.2, 0.4]),
        beta_value=1.0e-4,
        dt=np.array([0.1, 0.2]),
        steps=np.array([3, 4]),
        tmin=np.array([0.0, 1.0]),
        tmax=np.array([2.0, 3.0]),
    )

    assert len(calls) == 2
    assert calls[0]["dt"] == 0.1
    assert calls[1]["dt"] == 0.2
    assert calls[0]["steps"] == 3
    assert calls[1]["steps"] == 4
    assert calls[0]["tmin"] == 0.0
    assert calls[1]["tmin"] == 1.0
    assert calls[0]["tmax"] == 2.0
    assert calls[1]["tmax"] == 3.0
    np.testing.assert_allclose(scan.gamma, [1.2, 1.4])
    np.testing.assert_allclose(scan.omega, [-2.2, -2.4])


def test_run_kbm_scan_uses_cfg_beta_and_sequence_pick(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_kbm_beta_scan(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(gamma=np.array([3.0]), omega=np.array([-4.0]))

    cfg = replace(KBMBaseCase(), model=replace(KBMBaseCase().model, beta=2.5e-3))
    monkeypatch.setattr(
        "spectraxgk.benchmarks.run_kbm_beta_scan",
        _fake_run_kbm_beta_scan,
    )

    scan = run_kbm_scan(
        np.array([0.15, 0.35]),
        cfg=cfg,
        dt=[0.05, 0.1],
        steps=(5, 6),
        tmin=[0.2, 0.4],
        tmax=(0.8, 1.2),
    )

    assert len(calls) == 2
    assert calls[0]["betas"][0] == pytest.approx(2.5e-3)
    assert calls[1]["dt"] == 0.1
    assert calls[0]["steps"] == 5
    assert calls[1]["tmin"] == 0.4
    assert calls[0]["tmax"] == 0.8
    np.testing.assert_allclose(scan.gamma, [3.0, 3.0])
    np.testing.assert_allclose(scan.omega, [-4.0, -4.0])


def test_run_kbm_beta_scan_rejects_invalid_species_indices() -> None:
    with pytest.raises(ValueError):
        run_kbm_beta_scan(np.array([1.0e-4]), init_species_index=-1)
    with pytest.raises(ValueError):
        run_kbm_beta_scan(np.array([1.0e-4]), density_species_index=2)


def test_run_kbm_linear_rejects_invalid_fit_and_species_indices(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_flux_tube_geometry",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.apply_geometry_grid_defaults",
        lambda geom, grid: grid,
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_spectral_grid",
        lambda cfg: _grid_full(),
    )
    with pytest.raises(ValueError, match="fit_signal"):
        run_kbm_linear(
            cfg=KBMBaseCase(),
            fit_signal="bad",
            params=SimpleNamespace(rho_star=1.0),
            terms=LinearTerms(),
            Nl=2,
            Nm=2,
        )
    with pytest.raises(ValueError, match="init_species_index"):
        run_kbm_linear(
            cfg=KBMBaseCase(),
            init_species_index=2,
            params=SimpleNamespace(rho_star=1.0),
            terms=LinearTerms(),
            Nl=2,
            Nm=2,
        )
    with pytest.raises(ValueError, match="density_species_index"):
        run_kbm_linear(
            cfg=KBMBaseCase(),
            density_species_index=-1,
            params=SimpleNamespace(rho_star=1.0),
            terms=LinearTerms(),
            Nl=2,
            Nm=2,
        )


def test_run_kbm_beta_scan_auto_krylov_invalid_growth_falls_back_to_time(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_spectral_grid",
        lambda cfg: _grid_full(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.select_ky_grid",
        lambda grid, idx: _grid_sel(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._two_species_params",
        lambda *args, **kwargs: SimpleNamespace(rho_star=1.0, nu=0.0),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.select_kbm_solver_auto",
        lambda *args, **kwargs: "krylov",
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.dominant_eigenpair",
        lambda *args, **kwargs: (
            -0.1 + 0.2j,
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.integrate_linear_diagnostics",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            2.0 * np.ones((2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._select_fit_signal",
        lambda *args, **kwargs: np.array([1.0 + 0.0j, 2.0 + 0.0j]),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.15, -0.07, 0.0, 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_kbm_beta_scan(
        np.array([1.0e-4]),
        solver="auto",
        fit_signal="density",
        reference_aligned=False,
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
    )

    np.testing.assert_allclose(scan.gamma, [0.15])
    np.testing.assert_allclose(scan.omega, [-0.07])


def test_run_kbm_beta_scan_explicit_time_diagnostic_fallback_ladder(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_spectral_grid",
        lambda cfg: _grid_full(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.select_ky_grid",
        lambda grid, idx: _grid_sel(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._two_species_params",
        lambda *args, **kwargs: SimpleNamespace(rho_star=1.0, nu=0.0),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.select_kbm_solver_auto",
        lambda *args, **kwargs: "explicit_time",
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    cases = iter(
        [
            {
                "t": np.array([0.0, 1.0]),
                "growth": (0.21, -0.08, None, None, 0.5),
                "omega_series": None,
                "fit": None,
                "method": "z_index",
                "expected": (0.21, -0.08),
            },
            {
                "t": np.array([0.0, 1.0]),
                "growth": ValueError("phi fit failed"),
                "omega_series": (0.22, -0.09, None, None),
                "fit": None,
                "method": "max",
                "expected": (0.22, -0.09),
            },
            {
                "t": np.array([0.0, 1.0]),
                "growth": ValueError("phi fit failed"),
                "omega_series": ValueError("omega fit failed"),
                "fit": (0.23, -0.10, 0.0, 1.0),
                "method": "z_index",
                "expected": (0.23, -0.10),
            },
            {
                "t": np.array([0.0, 1.0]),
                "growth": None,
                "omega_series": None,
                "fit": (0.24, -0.11, 0.0, 1.0),
                "method": "project",
                "expected": (0.24, -0.11),
            },
            {
                "t": np.array([0.0]),
                "growth": None,
                "omega_series": None,
                "fit": None,
                "method": "z_index",
                "expected": (np.nan, np.nan),
            },
        ]
    )

    def _fake_integrate(*args, **kwargs):
        case = current["case"]
        return (
            case["t"],
            np.ones((len(case["t"]), 1, 1, 3), dtype=np.complex64),
            np.ones((1, 1, len(case["t"])), dtype=float),
            -np.ones((1, 1, len(case["t"])), dtype=float),
            None,
        )

    def _fake_growth(*args, **kwargs):
        value = current["case"]["growth"]
        if isinstance(value, Exception):
            raise value
        return value

    def _fake_omega_series(*args, **kwargs):
        value = current["case"]["omega_series"]
        if isinstance(value, Exception):
            raise value
        return value

    def _fake_fit(*args, **kwargs):
        return current["case"]["fit"]

    current: dict[str, object] = {}
    monkeypatch.setattr(
        "spectraxgk.benchmarks.integrate_linear_explicit_diagnostics",
        _fake_integrate,
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.instantaneous_growth_rate_from_phi",
        _fake_growth,
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.windowed_growth_rate_from_omega_series",
        _fake_omega_series,
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarks.extract_mode_time_series",
        lambda *args, **kwargs: np.ones(2, dtype=np.complex64),
    )
    monkeypatch.setattr("spectraxgk.benchmarks.fit_growth_rate_auto", _fake_fit)

    observed_gamma = []
    observed_omega = []
    for case in cases:
        current["case"] = case
        scan = run_kbm_beta_scan(
            np.array([1.0e-4]),
            solver="explicit_time",
            fit_signal="phi",
            mode_method=case["method"],
            reference_aligned=True,
            Nl=2,
            Nm=2,
            dt=0.1,
            steps=2,
        )
        observed_gamma.append(scan.gamma[0])
        observed_omega.append(scan.omega[0])

    np.testing.assert_allclose(observed_gamma[:4], [0.21, 0.22, 0.23, 0.24])
    np.testing.assert_allclose(observed_omega[:4], [-0.08, -0.09, -0.10, -0.11])
    assert np.isnan(observed_gamma[-1])
    assert np.isnan(observed_omega[-1])
