from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.objectives.portfolio_artifacts import (
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)
from spectraxgk.objectives.portfolio_contracts import (
    aggregate_objective_portfolio,
    portfolio_objective_weight_vector,
    portfolio_sample_weight_tensor,
    validate_objective_portfolio_contract,
)
from spectraxgk.objectives.portfolio_sensitivity import objective_portfolio_sensitivity_report


def test_weighted_objective_portfolio_matches_manual_reduction() -> None:
    rows = jnp.arange(1.0, 1.0 + 2 * 2 * 2 * 3, dtype=jnp.float32).reshape((2, 2, 2, 3))
    surface_weights = jnp.asarray([2.0, 1.0])
    alpha_weights = jnp.asarray([1.0, 3.0])
    ky_weights = jnp.asarray([1.0, 2.0])
    objective_weights = jnp.asarray([0.5, 1.5, 1.0])

    value = aggregate_objective_portfolio(
        rows,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
    )

    surface = np.asarray(surface_weights / jnp.sum(surface_weights))
    alpha = np.asarray(alpha_weights / jnp.sum(alpha_weights))
    ky = np.asarray(ky_weights / jnp.sum(ky_weights))
    objective = np.asarray(objective_weights / jnp.sum(objective_weights))
    sample = surface[:, None, None] * alpha[None, :, None] * ky[None, None, :]
    expected = np.sum(np.asarray(rows) * sample[..., None] * objective)

    np.testing.assert_allclose(float(value), float(expected), rtol=1.0e-6)
    np.testing.assert_allclose(float(jnp.sum(portfolio_sample_weight_tensor(rows, surface_weights=surface_weights))), 1.0)
    np.testing.assert_allclose(float(jnp.sum(portfolio_objective_weight_vector(rows, objective_weights=objective_weights))), 1.0)

    contract = validate_objective_portfolio_contract(
        rows,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
    )
    assert contract.row_shape == (2, 2, 2, 3)
    assert contract.sample_shape == (2, 2, 2)
    assert contract.n_samples == 8
    assert contract.uses_separable_sample_weights is True
    assert contract.uses_objective_weights is True
    assert contract.to_dict()["row_shape"] == [2, 2, 2, 3]


def test_objective_portfolio_gradient_jvp_and_finite_difference_parity() -> None:
    surface_weights = jnp.asarray([1.0, 2.0])
    alpha_weights = jnp.asarray([1.0, 3.0])
    ky_weights = jnp.asarray([2.0, 1.0, 1.5])
    objective_weights = jnp.asarray([0.75, 1.25])

    surface = jnp.asarray([0.2, 0.7])[:, None, None, None]
    alpha = jnp.asarray([-0.35, 0.45])[None, :, None, None]
    ky = jnp.asarray([0.15, 0.55, 0.9])[None, None, :, None]
    objective = jnp.asarray([0.8, 1.6])[None, None, None, :]

    def objective_fn(params: jnp.ndarray) -> jnp.ndarray:
        rows = (
            objective * params[0] ** 2
            + jnp.sin(params[1] + alpha) * (1.0 + surface)
            + params[2] * ky
            + 0.1 * params[0] * params[2] * surface * objective
        )
        return aggregate_objective_portfolio(
            rows,
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            objective_weights=objective_weights,
        )

    params = jnp.asarray([0.42, -0.18, 0.31])
    direction = jnp.asarray([0.25, -0.40, 0.15])
    grad = jax.grad(objective_fn)(params)
    _value, tangent = jax.jvp(objective_fn, (params,), (direction,))
    step = 1.0e-3
    finite_difference = (objective_fn(params + step * direction) - objective_fn(params - step * direction)) / (
        2.0 * step
    )

    np.testing.assert_allclose(float(tangent), float(jnp.vdot(grad, direction)), rtol=2.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(float(tangent), float(finite_difference), rtol=1.5e-3, atol=1.5e-3)


def test_objective_portfolio_sensitivity_report_checks_fd_and_conditioning() -> None:
    surface = jnp.asarray([-0.4, 0.7])[:, None, None]
    alpha = jnp.asarray([-0.5, 0.3])[None, :, None]
    ky = jnp.asarray([0.2, 0.6, 1.0])[None, None, :]

    def row_fn(params: jnp.ndarray) -> jnp.ndarray:
        gamma = 0.15 + 0.08 * params[0] + 0.04 * alpha + 0.03 * ky
        kperp = 0.45 + 0.05 * params[1] ** 2 + 0.08 * ky + 0.02 * surface
        flux = 0.30 + 0.09 * params[2] + 0.04 * jnp.sin(params[1] + alpha) + 0.02 * surface * ky
        ql_flux = gamma * flux / kperp
        return jnp.stack(
            [
                gamma + jnp.zeros_like(ql_flux),
                kperp + jnp.zeros_like(ql_flux),
                ql_flux,
            ],
            axis=-1,
        )

    params = jnp.asarray([0.12, -0.20, 0.35])
    step = 1.0e-4 if bool(jax.config.jax_enable_x64) else 2.0e-3
    rtol = 5.0e-4 if bool(jax.config.jax_enable_x64) else 2.0e-2
    atol = 1.0e-5 if bool(jax.config.jax_enable_x64) else 2.0e-4

    report = objective_portfolio_sensitivity_report(
        row_fn,
        params,
        surface_weights=jnp.asarray([1.0, 2.0]),
        alpha_weights=jnp.asarray([2.0, 1.0]),
        ky_weights=jnp.asarray([1.0, 2.0, 3.0]),
        objective_weights=jnp.asarray([0.2, 0.2, 1.0]),
        step=step,
        rtol=rtol,
        atol=atol,
        min_rank=3,
        condition_number_limit=1.0e4,
        workers=2,
    )

    assert report["passed"] is True
    assert report["portfolio_contract"]["row_shape"] == [2, 2, 3, 3]
    assert report["scalar_gradient_gate"]["passed"] is True
    assert report["row_jacobian_gate"]["passed"] is True
    assert report["conditioning_gate"]["passed"] is True
    assert report["conditioning_gate"]["sensitivity_map_rank"] == 3
    assert report["scalar_gradient_gate"]["finite_difference_parallel"]["requested_workers"] == 2
    assert report["covariance"]["source"] == "objective_portfolio_rows"


def test_objective_portfolio_sensitivity_report_fails_rank_deficient_rows() -> None:
    sample_axis = jnp.arange(4.0).reshape((1, 1, 4))

    def rank_deficient_row_fn(params: jnp.ndarray) -> jnp.ndarray:
        row = 0.2 + params[0] * (1.0 + sample_axis)
        return row[..., None]

    report = objective_portfolio_sensitivity_report(
        rank_deficient_row_fn,
        jnp.asarray([0.1, -0.2]),
        reduction="mean",
        step=1.0e-3,
        rtol=2.0e-2,
        atol=2.0e-4,
        min_rank=2,
        condition_number_limit=1.0e4,
    )

    assert report["passed"] is False
    assert report["scalar_gradient_gate"]["passed"] is True
    assert report["row_jacobian_gate"]["passed"] is True
    assert report["conditioning_gate"]["passed"] is False
    assert report["conditioning_gate"]["rank_deficiency"] == 1


def test_objective_portfolio_rejects_invalid_shape_and_weights() -> None:
    rows = jnp.ones((2, 2, 2, 2))

    with pytest.raises(ValueError, match="objective_rows"):
        aggregate_objective_portfolio(jnp.ones((2, 2, 2)))

    with pytest.raises(ValueError, match="sample_weights"):
        aggregate_objective_portfolio(rows, sample_weights=jnp.ones((2, 2)))

    with pytest.raises(ValueError, match="either sample_weights"):
        aggregate_objective_portfolio(rows, sample_weights=jnp.ones((2, 2, 2)), surface_weights=jnp.ones(2))

    with pytest.raises(ValueError, match="surface_weights"):
        aggregate_objective_portfolio(rows, surface_weights=jnp.asarray([1.0, -0.2]))

    with pytest.raises(ValueError, match="alpha_weights"):
        aggregate_objective_portfolio(rows, alpha_weights=jnp.asarray([0.0, 0.0]))

    with pytest.raises(ValueError, match="ky_weights"):
        aggregate_objective_portfolio(rows, ky_weights=jnp.asarray([1.0, jnp.nan]))

    with pytest.raises(ValueError, match="objective_weights"):
        aggregate_objective_portfolio(rows, objective_weights=jnp.ones(3))

    with pytest.raises(ValueError, match="mean reduction"):
        aggregate_objective_portfolio(rows, surface_weights=jnp.ones(2), reduction="mean")

    with pytest.raises(ValueError, match="max reduction"):
        aggregate_objective_portfolio(rows, sample_weights=jnp.ones((2, 2, 2)), reduction="max")

    with pytest.raises(TypeError, match="real numeric"):
        aggregate_objective_portfolio(jnp.ones((1, 1, 1, 1), dtype=jnp.complex64))

    with pytest.raises(ValueError, match="params"):
        objective_portfolio_sensitivity_report(lambda _p: rows, jnp.ones((1, 1)))


def test_objective_portfolio_mean_and_max_reductions_are_explicit() -> None:
    rows = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    ).reshape((2, 2, 1, 2))

    mean_contract = validate_objective_portfolio_contract(rows, reduction="mean")
    max_contract = validate_objective_portfolio_contract(
        rows,
        objective_weights=jnp.asarray([1.0, 3.0]),
        reduction="max",
    )

    assert mean_contract.reduction == "mean"
    assert max_contract.reduction == "max"
    np.testing.assert_allclose(float(aggregate_objective_portfolio(rows, reduction="mean")), 4.5)
    np.testing.assert_allclose(
        float(
            aggregate_objective_portfolio(
                rows,
                objective_weights=jnp.asarray([1.0, 3.0]),
                reduction="max",
            )
        ),
        7.75,
    )


def test_objective_portfolio_helpers_are_exported_at_package_top_level() -> None:
    import spectraxgk as sgk

    rows = jnp.ones((1, 1, 2, 2))
    contract = sgk.validate_objective_portfolio_contract(rows)

    assert isinstance(contract, sgk.StellaratorObjectivePortfolioContract)
    assert isinstance(sgk.ReducedPortfolioArtifactGuardConfig(), ReducedPortfolioArtifactGuardConfig)
    np.testing.assert_allclose(float(sgk.aggregate_objective_portfolio(rows)), 1.0)
    assert sgk.objective_portfolio_sensitivity_report is objective_portfolio_sensitivity_report
    assert sgk.reduced_portfolio_artifact_guard_report is reduced_portfolio_artifact_guard_report
