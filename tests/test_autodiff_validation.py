from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import spectraxgk
import spectraxgk.autodiff_validation as adv
from spectraxgk.autodiff_validation import (
    autodiff_finite_difference_report,
    central_finite_difference_jacobian,
    covariance_diagnostics,
    explicit_complex_operator_matrix,
    implicit_eigenpair_observable_sensitivity_report,
    isolated_eigenpair_observable_sensitivity_report,
    isolated_eigenvalue_sensitivity_report,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, linear_rhs_cached
from spectraxgk.diagnostics import gx_volume_factors
from spectraxgk.quasilinear import effective_kperp2, quasilinear_feature_objective, shape_aware_power_law_objective


def test_covariance_diagnostics_reports_uq_and_sensitivity_metadata() -> None:
    assert spectraxgk.covariance_diagnostics is covariance_diagnostics
    jac = np.array([[1.0, 0.2], [0.1, 0.8], [0.4, -0.3]])
    residual = np.array([1.0e-3, -2.0e-3, 1.5e-3])

    out = covariance_diagnostics(jac, residual, regularization=1.0e-8)
    cov = np.asarray(out["covariance"])
    corr = np.asarray(out["covariance_correlation"])
    eig = np.asarray(out["covariance_eigenvalues"])

    assert out["sensitivity_map_rank"] == 2
    assert np.isfinite(float(out["jacobian_condition_number"]))
    assert np.allclose(cov, cov.T)
    assert np.all(eig > 0.0)
    assert np.all(np.asarray(out["covariance_std"]) > 0.0)
    assert np.allclose(np.diag(corr), 1.0)
    assert float(out["uq_ellipse_area_1sigma"]) > 0.0


def test_covariance_diagnostics_matches_closed_form_gauss_newton() -> None:
    jac = np.eye(2)
    residual = np.array([2.0e-3, -4.0e-3])

    out = covariance_diagnostics(jac, residual, regularization=0.0)
    sigma2 = float(np.mean(residual**2) + 1.0e-12)

    np.testing.assert_allclose(np.asarray(out["covariance"]), sigma2 * np.eye(2))
    np.testing.assert_allclose(np.asarray(out["covariance_std"]), np.sqrt(sigma2) * np.ones(2))
    np.testing.assert_allclose(np.asarray(out["covariance_correlation"]), np.eye(2))
    np.testing.assert_allclose(float(out["uq_ellipse_area_1sigma"]), np.pi * sigma2)
    assert out["sensitivity_map_rank"] == 2
    assert float(out["jacobian_condition_number"]) == 1.0


def test_covariance_diagnostics_flags_rank_deficient_sensitivity_map() -> None:
    jac = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    residual = np.array([1.0e-3, 2.0e-3, -1.0e-3])

    out = covariance_diagnostics(jac, residual, regularization=1.0e-6)

    assert out["sensitivity_map_rank"] == 1
    assert np.isinf(float(out["jacobian_condition_number"]))
    cov = np.asarray(out["covariance"])
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) >= 0.0)
    assert np.asarray(out["covariance_std"])[1] > np.asarray(out["covariance_std"])[0]


def test_covariance_diagnostics_handles_scalar_and_empty_parameter_maps() -> None:
    scalar = covariance_diagnostics(np.ones((3, 1)), np.array([1.0e-3, -1.0e-3, 0.0]))
    assert scalar["sensitivity_map_rank"] == 1
    assert np.asarray(scalar["covariance"]).shape == (1, 1)
    assert float(scalar["uq_ellipse_area_1sigma"]) == 0.0

    with pytest.raises(ValueError, match="at least one parameter column"):
        covariance_diagnostics(np.empty((2, 0)), np.array([1.0e-3, -1.0e-3]))


def test_covariance_diagnostics_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2, 1)), np.ones(2))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2)), np.ones(3))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2)), np.ones(2), regularization=-1.0)


def test_autodiff_finite_difference_report_matches_closed_form_jacobian() -> None:
    assert spectraxgk.autodiff_finite_difference_report is autodiff_finite_difference_report
    assert spectraxgk.central_finite_difference_jacobian is central_finite_difference_jacobian

    def fn(x):
        return jnp.asarray([x[0] ** 2 + 3.0 * x[1], x[0] * x[1]])

    p = jnp.asarray([0.4, -0.2])
    report = autodiff_finite_difference_report(fn, p, step=1.0e-3, rtol=5.0e-4, atol=5.0e-6)
    parallel_report = autodiff_finite_difference_report(
        fn,
        p,
        step=1.0e-3,
        rtol=5.0e-4,
        atol=5.0e-6,
        workers=2,
    )

    assert report["passed"] is True
    assert parallel_report["passed"] is True
    assert parallel_report["finite_difference_parallel"]["requested_workers"] == 2
    jac_ad = np.asarray(report["jacobian_ad"])
    np.testing.assert_allclose(jac_ad, np.asarray([[0.8, 3.0], [-0.2, 0.4]]), rtol=1.0e-6)
    np.testing.assert_allclose(parallel_report["jacobian_fd"], report["jacobian_fd"])
    assert float(report["tangent_max_abs_error"]) < 1.0e-4


def test_central_finite_difference_handles_empty_parameter_vector() -> None:
    jac = central_finite_difference_jacobian(lambda x: jnp.asarray([1.0, 2.0]), jnp.asarray([]))
    assert jac.shape == (2, 0)
    assert jac.dtype == jnp.asarray([]).dtype


def test_quasilinear_feature_objective_derivative_gate() -> None:
    features = jnp.asarray([0.2, 0.8, 1.5])
    report = autodiff_finite_difference_report(
        lambda x: quasilinear_feature_objective(x, csat=0.7),
        features,
        step=1.0e-3,
        rtol=1.0e-4,
        atol=1.0e-5,
    )

    assert report["passed"] is True
    jac = np.asarray(report["jacobian_ad"]).reshape(3)
    expected = np.asarray([0.7 * 1.5 / 0.8, -0.7 * 1.5 * 0.2 / 0.8**2, 0.7 * 0.2 / 0.8])
    np.testing.assert_allclose(jac, expected, rtol=1.0e-6)


def test_quasilinear_sweep_rule_objectives_have_fd_checked_derivatives() -> None:
    features = jnp.asarray([-0.25, 0.8, 1.5])
    for rule in ("linear_weight", "absolute_growth_mixing_length"):
        report = autodiff_finite_difference_report(
            lambda x, rule=rule: quasilinear_feature_objective(x, rule=rule, csat=0.7),
            features,
            step=1.0e-3,
            rtol=2.0e-4,
            atol=1.0e-5,
        )
        assert report["passed"] is True


def test_shape_aware_power_law_objective_has_fd_checked_derivatives() -> None:
    ky = jnp.asarray([0.1, 0.2, 0.4])

    def objective(x):
        features = jnp.stack(
            [
                jnp.asarray([0.1, 0.2, 0.3]),
                jnp.asarray([0.5, 0.6, 0.7]),
                x[:3],
            ],
            axis=-1,
        )
        return jnp.sum(shape_aware_power_law_objective(features, ky, exponent=x[3], csat=0.8))

    x0 = jnp.asarray([1.0, 1.5, 2.0, -0.3])
    report = autodiff_finite_difference_report(objective, x0, step=1.0e-3, rtol=2.0e-4, atol=1.0e-5)

    assert report["passed"] is True


def test_isolated_eigenvalue_sensitivity_report_tracks_branch_derivatives() -> None:
    assert spectraxgk.isolated_eigenvalue_sensitivity_report is isolated_eigenvalue_sensitivity_report

    def matrix_fn(x):
        return jnp.asarray(
            [
                [0.30 + x[0], 0.02],
                [0.00, -0.40 + 0.50 * x[1]],
            ]
        )

    report = isolated_eigenvalue_sensitivity_report(
        matrix_fn,
        jnp.asarray([0.1, -0.2]),
        step=1.0e-3,
        rtol=1.0e-4,
        atol=1.0e-5,
    )

    assert report["passed"] is True
    assert report["branch_isolated"] is True
    assert report["selected_index"] == 0
    jac = np.asarray(report["jacobian_ad"])
    np.testing.assert_allclose(jac, np.asarray([[1.0, 0.0], [0.0, 0.0]]), rtol=1.0e-6)


def test_actual_linear_rhs_eigenvalue_derivative_gate() -> None:
    """Gate AD through a tiny SPECTRAX-GK linear RHS dense fixture."""

    assert spectraxgk.explicit_complex_operator_matrix is explicit_complex_operator_matrix
    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=4, Lx=6.0, Ly=6.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    n_laguerre = 1
    n_hermite = 1
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    base_params = LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    cache = build_linear_cache(grid, geom, base_params, n_laguerre, n_hermite)
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def matrix_fn(x):
        params = LinearParams(
            R_over_Ln=x[0],
            R_over_LTi=x[1],
            nu=0.0,
            nu_hyper=0.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=0.0,
            D_hyper=0.0,
            beta=0.0,
            fapar=0.0,
        )
        return explicit_complex_operator_matrix(
            lambda state: linear_rhs_cached(
                state,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
            )[0],
            state_shape,
        )

    report = isolated_eigenvalue_sensitivity_report(
        matrix_fn,
        jnp.asarray([2.2, 6.9]),
        step=1.0e-3,
        rtol=2.0e-2,
        atol=2.0e-4,
        gap_floor=1.0e-6,
    )

    assert report["passed"] is True
    assert report["branch_isolated"] is True
    jac_ad = np.asarray(report["jacobian_ad"])
    jac_fd = np.asarray(report["jacobian_fd"])
    np.testing.assert_allclose(jac_ad, jac_fd, rtol=2.0e-2, atol=2.0e-4)


def test_actual_linear_rhs_branch_objective_derivative_gate() -> None:
    """Gate a phase-invariant reduced quasilinear objective on the RHS branch."""

    assert spectraxgk.isolated_eigenpair_observable_sensitivity_report is (
        isolated_eigenpair_observable_sensitivity_report
    )
    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    n_laguerre = 2
    n_hermite = 1
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    base_params = LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    cache = build_linear_cache(grid, geom, base_params, n_laguerre, n_hermite)
    vol_fac, _flux_fac = gx_volume_factors(geom, grid)
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def params_from_features(x):
        return LinearParams(
            R_over_Ln=x[0],
            R_over_LTi=x[1],
            nu=0.0,
            nu_hyper=0.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=0.0,
            D_hyper=0.0,
            beta=0.0,
            fapar=0.0,
        )

    def rhs_with_params(state, params):
        return linear_rhs_cached(
            state,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x):
        params = params_from_features(x)
        return explicit_complex_operator_matrix(lambda state: rhs_with_params(state, params)[0], state_shape)

    def objective_fn(eigenvalue, eigenvector, x):
        params = params_from_features(x)
        state = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_with_params(state, params)
        kperp_eff = effective_kperp2(phi, cache, vol_fac)
        gamma = jnp.real(eigenvalue)
        return jnp.asarray([gamma, kperp_eff, gamma / jnp.maximum(kperp_eff, 1.0e-12)])

    report = isolated_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        jnp.asarray([2.2, 6.9]),
        step=1.0e-3,
        rtol=2.5e-2,
        atol=5.0e-4,
        gap_floor=1.0e-6,
    )

    assert report["passed"] is False
    assert report["ad_supported"] is False
    assert report["branch_isolated"] is True
    assert "non-symmetric eigenvectors" in str(report["failure_reason"])


def test_implicit_eigenpair_observable_gate_matches_closed_form_branch() -> None:
    assert spectraxgk.implicit_eigenpair_observable_sensitivity_report is (
        implicit_eigenpair_observable_sensitivity_report
    )

    def matrix_fn(x):
        return jnp.asarray(
            [
                [0.7 + x[0] + 0.2j, 0.2 + 0.1j * x[1]],
                [0.0, -0.4 + 0.3 * x[1] - 0.1j],
            ],
            dtype=jnp.complex64,
        )

    def observable_fn(eigenvalue, eigenvector, x):
        norm = jnp.sum(jnp.abs(eigenvector) ** 2)
        participation = jnp.abs(eigenvector[0]) ** 2 / norm
        return jnp.asarray([jnp.real(eigenvalue), jnp.imag(eigenvalue), participation + 0.1 * x[0]])

    report = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        observable_fn,
        jnp.asarray([0.2, -0.1]),
        step=1.0e-3,
        rtol=1.0e-3,
        atol=2.0e-5,
    )

    assert report["passed"] is True
    assert report["ad_supported"] is True
    assert report["sensitivity_method"] == "implicit_left_right_eigenpair"
    assert report["observable_chain_rule"] == "split_eigenpair_and_explicit_parameter"
    jac_impl = np.asarray(report["jacobian_implicit"])
    jac_fd = np.asarray(report["jacobian_fd"])
    np.testing.assert_allclose(jac_impl, jac_fd, rtol=1.0e-3, atol=2.0e-5)


def test_implicit_eigenpair_observable_gate_handles_complex_observables() -> None:
    def matrix_fn(x):
        return jnp.asarray(
            [
                [0.8 + x[0] + 0.1j, 0.05 + 0.03j * x[1]],
                [0.0, -0.2 + 0.2 * x[1] - 0.05j],
            ],
            dtype=jnp.complex64,
        )

    def observable_fn(eigenvalue, eigenvector, x):
        norm = jnp.sum(jnp.abs(eigenvector) ** 2)
        phase_invariant_weight = jnp.abs(eigenvector[0]) ** 2 / norm
        return jnp.asarray([eigenvalue + 0.1 * x[0] + 1j * phase_invariant_weight])

    report = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        observable_fn,
        jnp.asarray([0.1, -0.2]),
        step=1.0e-3,
        rtol=1.0e-3,
        atol=3.0e-5,
    )

    assert report["passed"] is True
    assert report["observable_chain_rule"] == "split_eigenpair_and_explicit_parameter"
    assert np.asarray(report["jacobian_implicit"]).shape == (2, 2)
    np.testing.assert_allclose(report["jacobian_implicit"], report["jacobian_fd"], rtol=1.0e-3, atol=3.0e-5)


def test_actual_linear_rhs_branch_objective_implicit_derivative_gate() -> None:
    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    n_laguerre = 2
    n_hermite = 1
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    base_params = LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    cache = build_linear_cache(grid, geom, base_params, n_laguerre, n_hermite)
    vol_fac, _flux_fac = gx_volume_factors(geom, grid)
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def params_from_features(x):
        return LinearParams(
            R_over_Ln=x[0],
            R_over_LTi=x[1],
            nu=0.0,
            nu_hyper=0.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=0.0,
            D_hyper=0.0,
            beta=0.0,
            fapar=0.0,
        )

    def rhs_with_params(state, params):
        return linear_rhs_cached(
            state,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x):
        params = params_from_features(x)
        return explicit_complex_operator_matrix(lambda state: rhs_with_params(state, params)[0], state_shape)

    def objective_fn(eigenvalue, eigenvector, x):
        params = params_from_features(x)
        state = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_with_params(state, params)
        kperp_eff = effective_kperp2(phi, cache, vol_fac)
        gamma = jnp.real(eigenvalue)
        return jnp.asarray([gamma, kperp_eff, gamma / jnp.maximum(kperp_eff, 1.0e-12)])

    report = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        jnp.asarray([2.2, 6.9]),
        step=1.0e-3,
        rtol=3.0e-2,
        atol=7.5e-4,
        gap_floor=1.0e-6,
    )

    assert report["passed"] is True
    assert report["branch_isolated"] is True
    assert report["observable_chain_rule"] == "split_eigenpair_and_explicit_parameter"
    jac_impl = np.asarray(report["jacobian_implicit"])
    jac_fd = np.asarray(report["jacobian_fd"])
    np.testing.assert_allclose(jac_impl, jac_fd, rtol=3.0e-2, atol=7.5e-4)


def test_isolated_eigenvalue_sensitivity_report_flags_small_gaps() -> None:
    def matrix_fn(x):
        return jnp.diag(jnp.asarray([x[0], x[0] + 1.0e-9]))

    report = isolated_eigenvalue_sensitivity_report(
        matrix_fn,
        jnp.asarray([0.2]),
        step=1.0e-3,
        gap_floor=1.0e-6,
    )

    assert report["branch_isolated"] is False
    assert report["passed"] is False
    with pytest.raises(ValueError):
        isolated_eigenvalue_sensitivity_report(matrix_fn, jnp.asarray([0.2]), selector="index:4")


def test_eigen_sensitivity_reports_validate_selectors_and_scalar_branches() -> None:
    def scalar_matrix_fn(x):
        return jnp.asarray([[0.5 + x[0] + 0.2j * x[1]]], dtype=jnp.complex64)

    value_report = isolated_eigenvalue_sensitivity_report(
        scalar_matrix_fn,
        jnp.asarray([0.1, -0.2]),
        selector="index:0",
        step=1.0e-3,
        rtol=1.0e-3,
        atol=2.0e-5,
    )
    assert value_report["passed"] is True
    assert value_report["eigenvalue_gap"] == float("inf")

    pair_report = isolated_eigenpair_observable_sensitivity_report(
        scalar_matrix_fn,
        lambda eigenvalue, eigenvector, x: jnp.asarray([jnp.real(eigenvalue) + jnp.abs(eigenvector[0]) ** 2]),
        jnp.asarray([0.1, -0.2]),
        selector="index:0",
        step=1.0e-3,
        rtol=1.0e-3,
        atol=2.0e-5,
    )
    assert pair_report["ad_supported"] is False
    assert pair_report["eigenvalue_gap"] == float("inf")

    with pytest.raises(ValueError):
        isolated_eigenvalue_sensitivity_report(scalar_matrix_fn, jnp.asarray([[0.1]]))
    with pytest.raises(ValueError):
        isolated_eigenvalue_sensitivity_report(scalar_matrix_fn, jnp.asarray([0.1]), selector="min_abs")
    with pytest.raises(ValueError):
        isolated_eigenpair_observable_sensitivity_report(scalar_matrix_fn, lambda *_: jnp.asarray([1.0]), jnp.asarray([[0.1]]))
    with pytest.raises(ValueError):
        isolated_eigenpair_observable_sensitivity_report(
            scalar_matrix_fn,
            lambda *_: jnp.asarray([1.0]),
            jnp.asarray([0.1]),
            selector="index:3",
        )


def test_eigen_sensitivity_reports_cover_empty_matrices_and_ad_fallbacks(monkeypatch) -> None:
    def empty_matrix(x):
        return jnp.zeros((0, 0), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="at least one eigenvalue"):
        isolated_eigenvalue_sensitivity_report(empty_matrix, jnp.asarray([0.1]))
    with pytest.raises(ValueError, match="at least one eigenvalue"):
        isolated_eigenpair_observable_sensitivity_report(empty_matrix, lambda *_: jnp.asarray([1.0]), jnp.asarray([0.1]))
    with pytest.raises(ValueError, match="at least one eigenvalue"):
        implicit_eigenpair_observable_sensitivity_report(empty_matrix, lambda *_: jnp.asarray([1.0]), jnp.asarray([0.1]))

    def matrix_fn(x):
        return jnp.asarray([[1.0 + x[0], 0.0], [0.0, -0.5 + x[0]]], dtype=jnp.complex64)

    def raise_not_implemented(*_args, **_kwargs):
        raise NotImplementedError("forced derivative fallback")

    monkeypatch.setattr(adv, "autodiff_finite_difference_report", raise_not_implemented)
    fallback = adv.isolated_eigenvalue_sensitivity_report(matrix_fn, jnp.asarray([0.1]))
    assert fallback["ad_supported"] is False
    assert "forced derivative fallback" in str(fallback["failure_reason"])


def test_isolated_eigenpair_report_realifies_complex_observable(monkeypatch) -> None:
    def matrix_fn(x):
        return jnp.asarray([[1.0 + x[0] + 0.1j, 0.0], [0.0, -0.5 + 0.2j]], dtype=jnp.complex64)

    captured = {}

    def fake_fd_report(fn, params, **kwargs):
        values = fn(params)
        captured["values"] = np.asarray(values)
        return {
            "passed": True,
            "step": kwargs["step"],
            "rtol": kwargs["rtol"],
            "atol": kwargs["atol"],
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "tangent_max_abs_error": 0.0,
            "jacobian_ad": [[1.0], [0.0]],
            "jacobian_fd": [[1.0], [0.0]],
            "tangent_ad": [1.0, 0.0],
            "tangent_fd": [1.0, 0.0],
        }

    monkeypatch.setattr(adv, "autodiff_finite_difference_report", fake_fd_report)
    report = adv.isolated_eigenpair_observable_sensitivity_report(
        matrix_fn,
        lambda eigenvalue, eigenvector, x: jnp.asarray([eigenvalue + 0.1j * jnp.abs(eigenvector[0]) ** 2]),
        jnp.asarray([0.2]),
    )

    assert report["passed"] is True
    assert report["ad_supported"] is True
    assert captured["values"].shape == (2,)


def test_implicit_eigenpair_observable_report_validates_inputs() -> None:
    def matrix_fn(x):
        return jnp.asarray([[1.0 + x[0], 0.0], [0.0, -1.0 + x[0]]], dtype=jnp.complex64)

    def observable(eigenvalue, eigenvector, x):
        return jnp.asarray([jnp.real(eigenvalue)])

    with pytest.raises(ValueError):
        implicit_eigenpair_observable_sensitivity_report(matrix_fn, observable, jnp.asarray([[0.1]]))
    with pytest.raises(ValueError):
        implicit_eigenpair_observable_sensitivity_report(lambda x: jnp.ones((2, 3)), observable, jnp.asarray([0.1]))
    with pytest.raises(ValueError):
        implicit_eigenpair_observable_sensitivity_report(matrix_fn, observable, jnp.asarray([0.1]), selector="min_abs")
    with pytest.raises(ValueError):
        implicit_eigenpair_observable_sensitivity_report(matrix_fn, observable, jnp.asarray([0.1]), selector="index:9")


def test_autodiff_finite_difference_report_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones((2, 1)))
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones(2), step=0.0)
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones(2), workers=0)
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones(2), workers=2, parallel_executor="process")
    with pytest.raises(ValueError):
        autodiff_finite_difference_report(lambda x: x, jnp.ones((2, 1)))
    with pytest.raises(ValueError):
        autodiff_finite_difference_report(lambda x: x, jnp.ones(2), direction=jnp.ones(3))
    with pytest.raises(ValueError):
        autodiff_finite_difference_report(lambda x: x, jnp.ones(2), workers=0)
    with pytest.raises(ValueError):
        explicit_complex_operator_matrix(lambda x: x, (0,))
    with pytest.raises(ValueError):
        explicit_complex_operator_matrix(lambda x: jnp.zeros((2,), dtype=x.dtype), (1,))
