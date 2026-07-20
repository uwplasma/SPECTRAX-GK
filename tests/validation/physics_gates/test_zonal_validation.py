from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gkx.diagnostics.zonal_validation import (
    kx_token,
    load_w7x_combined_trace_csv,
    load_w7x_trace_csv,
    normalize_trace,
    reference_mean_trace,
    reference_residual_table,
    reference_time_limits,
    tail_trace_metrics,
    w7x_trace_path,
)


def test_kx_token_and_trace_path_contract() -> None:
    assert kx_token(0.05) == "050"
    assert kx_token(0.1) == "100"
    assert w7x_trace_path(Path("out"), 0.3).as_posix() == "out/w7x_test4_kx300.csv"


def test_normalize_trace_sorts_filters_and_uses_first_nonzero() -> None:
    t = np.array([2.0, 0.0, 1.0, np.nan, 3.0])
    y = np.array([4.0, 0.0, 2.0, 8.0, np.inf])

    t_norm, y_norm = normalize_trace(t, y)

    np.testing.assert_allclose(t_norm, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(y_norm, [0.0, 1.0, 2.0])


def test_normalize_trace_rejects_bad_scale() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_trace(np.array([np.nan]), np.array([1.0]))
    with pytest.raises(ValueError, match="normalization level"):
        normalize_trace(np.array([0.0]), np.array([1.0]), initial_level=0.0)


def test_w7x_trace_loader_accepts_t_or_t_reference(tmp_path: Path) -> None:
    path_ref = tmp_path / "trace_ref.csv"
    path_t = tmp_path / "trace_t.csv"
    pd.DataFrame({"t_reference": [0.0, 1.0], "phi_zonal_real": [1.0, 2.0]}).to_csv(
        path_ref, index=False
    )
    pd.DataFrame({"t": [0.0, 2.0], "phi_zonal_real": [3.0, 5.0]}).to_csv(
        path_t, index=False
    )

    t_ref, y_ref = load_w7x_trace_csv(path_ref)
    t_t, y_t = load_w7x_trace_csv(path_t)

    np.testing.assert_allclose(t_ref, [0.0, 1.0])
    np.testing.assert_allclose(y_ref, [1.0, 2.0])
    np.testing.assert_allclose(t_t, [0.0, 2.0])
    np.testing.assert_allclose(y_t, [3.0, 5.0])


def test_w7x_trace_loader_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_trace.csv"
    pd.DataFrame({"t": [0.0, 1.0], "wrong": [1.0, 2.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="phi_zonal_real"):
        load_w7x_trace_csv(path)


def test_w7x_combined_trace_loader_selects_kx_and_normalization(tmp_path: Path) -> None:
    path = tmp_path / "combined.csv"
    pd.DataFrame(
        {
            "kx_target": [0.05, 0.10, 0.10],
            "t_reference": [0.0, 1.0, 0.0],
            "phi_zonal_real": [2.0, 6.0, 4.0],
            "response_normalized": [1.0, 1.5, 1.0],
        }
    ).to_csv(path, index=False)

    t_raw, y_raw = load_w7x_combined_trace_csv(path, 0.10)
    t_norm, y_norm = load_w7x_combined_trace_csv(path, 0.10, normalized=True)

    np.testing.assert_allclose(t_raw, [0.0, 1.0])
    np.testing.assert_allclose(y_raw, [4.0, 6.0])
    np.testing.assert_allclose(t_norm, [0.0, 1.0])
    np.testing.assert_allclose(y_norm, [1.0, 1.5])


def test_w7x_combined_trace_loader_rejects_missing_and_absent_kx(
    tmp_path: Path,
) -> None:
    missing_required = tmp_path / "missing_required.csv"
    missing_value = tmp_path / "missing_value.csv"
    no_kx = tmp_path / "no_kx.csv"
    pd.DataFrame({"kx_target": [0.1], "phi_zonal_real": [1.0]}).to_csv(
        missing_required, index=False
    )
    pd.DataFrame({"kx_target": [0.1], "t_reference": [0.0]}).to_csv(
        missing_value, index=False
    )
    pd.DataFrame(
        {"kx_target": [0.2], "t_reference": [0.0], "phi_zonal_real": [1.0]}
    ).to_csv(no_kx, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_w7x_combined_trace_csv(missing_required, 0.1)
    with pytest.raises(ValueError, match="missing column"):
        load_w7x_combined_trace_csv(missing_value, 0.1)
    with pytest.raises(ValueError, match="no trace"):
        load_w7x_combined_trace_csv(no_kx, 0.1)


def test_reference_tables_and_mean_trace(tmp_path: Path) -> None:
    residuals = tmp_path / "residuals.csv"
    pd.DataFrame(
        {
            "kx_rhoi": [0.05, 0.05, 0.10],
            "code": ["stella", "GENE", "stella"],
            "residual_median": [0.1, 0.12, 0.2],
        }
    ).to_csv(residuals, index=False)
    traces = pd.DataFrame(
        {
            "kx_rhoi": [0.05, 0.05, 0.05, 0.05],
            "code": ["stella", "GENE", "stella", "GENE"],
            "t_vti_over_a": [0.0, 0.0, 2.0, 2.0],
            "response": [1.0, 1.2, 0.2, 0.4],
        }
    )

    residual_table = reference_residual_table(residuals)
    limits = reference_time_limits(traces)
    ref_t, ref_y = reference_mean_trace(traces, 0.05)

    assert list(residual_table["kx"]) == [0.05, 0.10]
    assert np.isclose(float(residual_table.loc[0, "reference_residual"]), 0.11)
    assert np.isclose(float(residual_table.loc[0, "reference_code_spread"]), 0.01)
    assert np.isclose(float(limits.loc[0, "reference_tmax"]), 2.0)
    np.testing.assert_allclose(ref_t, [0.0, 2.0])
    np.testing.assert_allclose(ref_y, [1.1, 0.3])


def test_reference_tables_reject_missing_columns_and_missing_trace(
    tmp_path: Path,
) -> None:
    residuals = tmp_path / "bad_residuals.csv"
    pd.DataFrame({"kx_rhoi": [0.1], "code": ["stella"]}).to_csv(residuals, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        reference_residual_table(residuals)

    with pytest.raises(ValueError, match="reference trace table missing columns"):
        reference_time_limits(pd.DataFrame({"kx_rhoi": [0.1]}))

    with pytest.raises(ValueError, match="missing reference trace"):
        reference_mean_trace(
            pd.DataFrame(
                {
                    "kx_rhoi": [0.2],
                    "code": ["stella"],
                    "t_vti_over_a": [0.0],
                    "response": [1.0],
                }
            ),
            0.1,
        )


def test_tail_trace_metrics_uses_late_reference_window() -> None:
    t_ref = np.linspace(0.0, 10.0, 6)
    y_ref = np.array([1.0, 0.8, 0.4, 0.2, 0.1, 0.1])
    t_obs = np.linspace(0.0, 10.0, 11)
    y_obs = np.interp(t_obs, t_ref, y_ref)

    metrics = tail_trace_metrics(
        t_obs=t_obs, y_obs=y_obs, t_ref=t_ref, y_ref=y_ref, tail_fraction=0.4
    )

    assert metrics["tail_mean_abs_error"] <= 1.0e-12
    assert metrics["tail_max_abs_error"] <= 1.0e-12
    assert metrics["tail_std"] is not None
    assert metrics["reference_tail_std"] is not None


def test_tail_trace_metrics_returns_none_when_observed_tail_is_missing() -> None:
    metrics = tail_trace_metrics(
        t_obs=np.array([0.0, 1.0]),
        y_obs=np.array([1.0, 0.9]),
        t_ref=np.array([5.0, 6.0, 7.0]),
        y_ref=np.array([0.5, 0.4, 0.3]),
        tail_fraction=0.5,
    )

    assert metrics == {
        "tail_std": None,
        "reference_tail_std": None,
        "tail_mean_abs_error": None,
        "tail_max_abs_error": None,
    }


def test_tail_trace_metrics_reports_late_window_envelope_errors() -> None:
    t_ref = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    y_ref = np.asarray([1.0, 0.7, 0.35, 0.18, 0.11, 0.09])
    t_obs = np.asarray([0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0])
    tail_offsets = np.asarray([0.03, -0.01, 0.02, -0.04])
    y_obs = np.interp(t_obs, t_ref, y_ref)
    y_obs[2:6] += tail_offsets
    y_obs[-1] = 999.0

    metrics = tail_trace_metrics(
        t_obs=t_obs, y_obs=y_obs, t_ref=t_ref, y_ref=y_ref, tail_fraction=0.3
    )

    tail_start = 10.0 - 0.3 * 10.0
    mask = (t_obs >= tail_start) & (t_obs <= 10.0)
    ref_interp = np.interp(t_obs[mask], t_ref, y_ref)
    obs_tail = y_obs[mask]
    diff = obs_tail - ref_interp
    ref_tail = y_ref[t_ref >= tail_start]

    assert mask.tolist() == [False, False, True, True, True, True, False]
    assert metrics["tail_mean_abs_error"] == pytest.approx(float(np.mean(np.abs(diff))))
    assert metrics["tail_max_abs_error"] == pytest.approx(float(np.max(np.abs(diff))))
    assert metrics["tail_std"] == pytest.approx(float(np.std(obs_tail)))
    assert metrics["reference_tail_std"] == pytest.approx(float(np.std(ref_tail)))
