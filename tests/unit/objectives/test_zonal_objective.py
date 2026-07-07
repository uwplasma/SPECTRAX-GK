from __future__ import annotations

import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.objectives.zonal import (
    ZONAL_FLOW_OBJECTIVE_NAMES,
    ZonalFlowObjectiveConfig,
    zonal_flow_objective_artifact_from_records,
    zonal_flow_objective_rows,
    zonal_flow_objective_sensitivity_report,
    zonal_flow_reduced_objective,
)
from spectraxgk.objectives import zonal_records
from spectraxgk.objectives.zonal_records import _finite_metric_tensor_from_records


def test_zonal_record_helpers_have_single_canonical_owner() -> None:
    assert inspect.getmodule(_finite_metric_tensor_from_records) is zonal_records


def test_zonal_flow_objective_prefers_large_residual_and_low_damping() -> None:
    residual_weak = jnp.asarray([[[0.18, 0.22], [0.20, 0.24]]])
    residual_strong = residual_weak + 0.18
    damping_high = jnp.asarray([[[0.12, 0.10], [0.11, 0.09]]])
    damping_low = damping_high * 0.35
    growth = jnp.asarray([[[0.30, 0.34], [0.28, 0.32]]])
    recurrence = jnp.asarray([[[0.04, 0.05], [0.03, 0.04]]])
    cfg = ZonalFlowObjectiveConfig(
        residual_weight=2.0,
        damping_weight=1.0,
        growth_over_residual_weight=0.5,
        recurrence_weight=0.25,
    )

    weak = zonal_flow_reduced_objective(
        residual_level=residual_weak,
        damping_rate=damping_high,
        linear_growth_rate=growth,
        recurrence_amplitude=recurrence,
        config=cfg,
    )
    strong = zonal_flow_reduced_objective(
        residual_level=residual_strong,
        damping_rate=damping_low,
        linear_growth_rate=growth,
        recurrence_amplitude=recurrence * 0.5,
        config=cfg,
    )

    assert float(strong) < float(weak)
    rows = zonal_flow_objective_rows(
        residual_level=residual_strong,
        damping_rate=damping_low,
        linear_growth_rate=growth,
        recurrence_amplitude=recurrence,
        config=cfg,
    )
    assert tuple(rows.shape) == (1, 2, 2, len(ZONAL_FLOW_OBJECTIVE_NAMES))
    np.testing.assert_allclose(
        np.asarray(rows[..., 0]), 1.0 / np.asarray(residual_strong)
    )


def test_zonal_flow_objective_sensitivity_report_checks_ad_fd_and_conditioning() -> (
    None
):
    surface = jnp.asarray([0.3, 0.7])[:, None, None]
    alpha = jnp.asarray([-0.4, 0.2])[None, :, None]
    kx = jnp.asarray([0.05, 0.11, 0.23])[None, None, :]

    def metric_fn(params: jnp.ndarray) -> dict[str, jnp.ndarray]:
        residual = (
            0.34
            + 0.05 * params[0] * (1.0 + surface)
            + 0.03 * jnp.sin(params[1] + alpha)
            - 0.02 * params[2] * kx
        )
        damping = 0.06 + 0.025 * params[1] ** 2 + 0.015 * surface + 0.01 * kx
        growth = 0.22 + 0.04 * params[2] + 0.02 * alpha + 0.01 * kx
        recurrence = 0.025 + 0.01 * params[0] ** 2 + 0.004 * surface * kx
        return {
            "residual_level": residual,
            "damping_rate": damping,
            "linear_growth_rate": growth,
            "recurrence_amplitude": recurrence,
        }

    cfg = ZonalFlowObjectiveConfig(
        residual_weight=1.5,
        damping_weight=1.0,
        growth_over_residual_weight=0.75,
        recurrence_weight=0.25,
    )
    params = jnp.asarray([0.12, -0.18, 0.26])
    step = 1.0e-4 if bool(jax.config.read("jax_enable_x64")) else 2.0e-3
    rtol = 7.5e-4 if bool(jax.config.read("jax_enable_x64")) else 2.5e-2
    atol = 1.0e-5 if bool(jax.config.read("jax_enable_x64")) else 3.0e-4

    report = zonal_flow_objective_sensitivity_report(
        metric_fn,
        params,
        config=cfg,
        surface_weights=jnp.asarray([1.0, 2.0]),
        alpha_weights=jnp.asarray([1.5, 1.0]),
        ky_weights=jnp.asarray([2.0, 1.0, 1.5]),
        step=step,
        rtol=rtol,
        atol=atol,
        min_rank=3,
        condition_number_limit=1.0e5,
        workers=2,
    )

    assert report["kind"] == "zonal_flow_objective_sensitivity_report"
    assert report["passed"] is True
    assert report["claim_level"].startswith("reduced_zonal_flow_objective")
    assert report["portfolio_contract"]["row_shape"] == [2, 2, 3, 4]
    assert report["scalar_gradient_gate"]["passed"] is True
    assert report["row_jacobian_gate"]["passed"] is True
    assert report["conditioning_gate"]["passed"] is True
    assert report["covariance"]["source"] == "objective_portfolio_rows"
    assert report["objective_config"]["objective_names"] == list(
        ZONAL_FLOW_OBJECTIVE_NAMES
    )
    json.dumps(report, allow_nan=False)


def test_zonal_flow_objective_rejects_invalid_contracts() -> None:
    good = jnp.ones((1, 1, 2)) * 0.3

    with pytest.raises(ValueError, match="at least one"):
        ZonalFlowObjectiveConfig(residual_weight=0.0, damping_weight=0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ZonalFlowObjectiveConfig(residual_weight=-1.0)
    with pytest.raises(ValueError, match="residual_floor"):
        ZonalFlowObjectiveConfig(residual_floor=0.0)
    with pytest.raises(ValueError, match="residual_level"):
        zonal_flow_objective_rows(residual_level=jnp.ones((1, 2)), damping_rate=good)
    with pytest.raises(ValueError, match="dimensions"):
        zonal_flow_objective_rows(
            residual_level=jnp.ones((0, 1, 1)), damping_rate=jnp.ones((0, 1, 1))
        )
    with pytest.raises(TypeError, match="real numeric"):
        zonal_flow_objective_rows(
            residual_level=jnp.ones((1, 1, 1), dtype=jnp.complex64),
            damping_rate=jnp.ones((1, 1, 1)),
        )
    with pytest.raises(ValueError, match="finite"):
        zonal_flow_objective_rows(
            residual_level=jnp.asarray([[[jnp.nan]]]),
            damping_rate=jnp.asarray([[[0.1]]]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        zonal_flow_objective_rows(
            residual_level=jnp.asarray([[[0.0]]]), damping_rate=jnp.asarray([[[0.1]]])
        )
    with pytest.raises(ValueError, match="broadcast-compatible"):
        zonal_flow_objective_rows(residual_level=good, damping_rate=jnp.ones((1, 1, 3)))
    with pytest.raises(ValueError, match="residual_level and damping_rate"):
        zonal_flow_objective_sensitivity_report(
            lambda _p: {"residual_level": good}, jnp.ones(1)
        )


def test_zonal_flow_objective_artifact_from_records_is_strict_and_ranked() -> None:
    records = [
        {
            "surface": 0.25,
            "alpha": 0.0,
            "kx_target": 0.05,
            "residual_level": 0.22,
            "gam_damping_rate": 0.07,
            "linear_growth_rate": 0.30,
            "tail_std_ratio": 1.8,
        },
        {
            "surface": 0.25,
            "alpha": 0.0,
            "kx_target": 0.10,
            "residual_level": 0.44,
            "gam_damping_rate": 0.03,
            "linear_growth_rate": 0.28,
            "tail_std_ratio": 0.8,
        },
    ]
    payload = zonal_flow_objective_artifact_from_records(
        records,
        config=ZonalFlowObjectiveConfig(
            residual_weight=1.0,
            damping_weight=1.0,
            growth_over_residual_weight=0.5,
            recurrence_weight=0.25,
        ),
        source_paths=["docs/_static/example.csv"],
    )

    assert payload["promotion_ready"] is True
    assert payload["missing_damping_count"] == 0
    assert payload["axes"] == {"surface": [0.25], "alpha": [0.0], "kx": [0.05, 0.1]}
    assert payload["sample_count"] == 2
    assert np.asarray(payload["objective_rows"]).shape == (
        1,
        1,
        2,
        len(ZONAL_FLOW_OBJECTIVE_NAMES),
    )
    assert (
        payload["row_table"][1]["sample_objective"]
        < payload["row_table"][0]["sample_objective"]
    )
    assert payload["source_paths"] == ["docs/_static/example.csv"]
    json.dumps(payload, allow_nan=False)


def test_zonal_flow_objective_artifact_missing_damping_policy_and_shape_guards() -> (
    None
):
    records = [
        {"kx": 0.05, "residual_level": 0.25, "tail_std_ratio": 1.0},
        {"kx": 0.10, "residual_level": 0.35, "tail_std_ratio": 0.7},
    ]

    with pytest.raises(ValueError, match="missing finite damping_rate"):
        zonal_flow_objective_artifact_from_records(records)

    payload = zonal_flow_objective_artifact_from_records(
        records, missing_damping_policy="zero"
    )
    assert payload["promotion_ready"] is False
    assert payload["missing_damping_count"] == 2
    np.testing.assert_allclose(np.asarray(payload["metrics"]["damping_rate"]), 0.0)

    with pytest.raises(ValueError, match="duplicate"):
        zonal_flow_objective_artifact_from_records(
            [records[0], records[0]],
            missing_damping_policy="zero",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        zonal_flow_objective_artifact_from_records(
            [{"kx": 0.05, "residual_level": 0.0, "damping_rate": 0.1}],
        )

    recurrence_missing = zonal_flow_objective_artifact_from_records(
        [{"kx": 0.05, "residual_level": 0.25, "damping_rate": 0.1}],
    )
    assert recurrence_missing["missing_recurrence_count"] == 1

    with pytest.raises(ValueError, match="missing_damping_policy"):
        zonal_flow_objective_artifact_from_records(
            [{"kx": 0.05, "residual_level": 0.25, "damping_rate": 0.1}],
            missing_damping_policy="skip",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="at least one"):
        zonal_flow_objective_artifact_from_records([], missing_damping_policy="zero")

    with pytest.raises(ValueError, match="missing finite kx"):
        zonal_flow_objective_artifact_from_records(
            [{"residual_level": 0.25, "damping_rate": 0.1}],
        )

    with pytest.raises(ValueError, match="numeric"):
        zonal_flow_objective_artifact_from_records(
            [{"kx": "not-a-number", "residual_level": 0.25, "damping_rate": 0.1}],
        )

    with pytest.raises(ValueError, match="missing finite residual_level"):
        zonal_flow_objective_artifact_from_records(
            [{"kx": 0.05, "residual_level": "nan", "damping_rate": 0.1}],
        )

    with pytest.raises(ValueError, match="complete finite tensor"):
        zonal_flow_objective_artifact_from_records(
            [
                {
                    "surface": 0.0,
                    "kx": 0.05,
                    "residual_level": 0.25,
                    "damping_rate": 0.1,
                },
                {
                    "surface": 1.0,
                    "kx": 0.10,
                    "residual_level": 0.35,
                    "damping_rate": 0.1,
                },
            ],
        )
