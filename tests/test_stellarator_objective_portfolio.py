from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.stellarator_objective_portfolio import (
    aggregate_objective_portfolio,
    portfolio_objective_weight_vector,
    portfolio_sample_weight_tensor,
    validate_objective_portfolio_contract,
)


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


def test_objective_portfolio_helpers_are_exported_at_package_top_level() -> None:
    import spectraxgk as sgk

    rows = jnp.ones((1, 1, 2, 2))
    contract = sgk.validate_objective_portfolio_contract(rows)

    assert isinstance(contract, sgk.StellaratorObjectivePortfolioContract)
    np.testing.assert_allclose(float(sgk.aggregate_objective_portfolio(rows)), 1.0)
