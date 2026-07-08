from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk import StellaratorITGSampleSet, VMECJAXTransportObjectiveConfig
from spectraxgk.objectives.core import SOLVER_OBJECTIVE_NAMES


ROOT = REPO_ROOT
SCRIPT = ROOT / "tools" / "campaigns" / "evaluate_vmec_jax_spectrax_transport_metric.py"
mod = load_campaign_tool("evaluate_vmec_jax_spectrax_transport_metric")


def test_float_tuple_rejects_empty_or_nonfinite_values() -> None:
    assert mod._float_tuple("0.45, 0.64") == (0.45, 0.64)
    with pytest.raises(argparse.ArgumentTypeError):
        mod._float_tuple(",,,")
    with pytest.raises(argparse.ArgumentTypeError):
        mod._float_tuple("0.5,nan")


def test_build_report_is_history_compatible_and_json_safe() -> None:
    samples = StellaratorITGSampleSet(
        surfaces=(0.45, 0.64, 0.78),
        alphas=(0.0, 0.7853981633974483),
        ky_values=(0.1, 0.3, 0.5),
    )
    config = VMECJAXTransportObjectiveConfig(
        kind="growth",
        sample_set=samples,
        ntheta=24,
        mboz=21,
        nboz=21,
        n_laguerre=2,
        n_hermite=3,
        objective_transform="log1p",
    )

    report = mod.build_report(
        input_path=Path("input.final"),
        max_mode=5,
        min_vmec_mode=7,
        transport_kind="growth",
        sample_set=samples,
        config=config,
        metric=0.067,
        solver_device="cpu",
        inner_max_iter=120,
        inner_ftol=1.0e-9,
        trial_max_iter=120,
        trial_ftol=1.0e-9,
        wout_path=Path("wout.final.nc"),
    )

    assert report["kind"] == "vmec_jax_spectrax_transport_metric_eval"
    assert report["transport_objective_final"] == 0.067
    assert report["spectrax_objective_final"] == 0.067
    assert report["transport_metric_final"] == 0.067
    assert report["sample_set"]["n_samples"] == 18
    assert report["spectrax_config"]["gradient_scope"] == "eigenvalue_growth_ad"
    assert report["spectrax_config"]["surface_chunk_size"] == 0
    assert report["wout_path"] == "wout.final.nc"
    assert "not an optimization" in report["claim_scope"]
    assert "long-window" in report["next_action"]
    json.dumps(mod._json_safe(report), allow_nan=False)


def test_sample_statistics_from_state_reports_weighted_reduced_spread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = StellaratorITGSampleSet(
        surfaces=(0.45, 0.64),
        alphas=(0.0,),
        ky_values=(0.1, 0.2),
    )
    config = VMECJAXTransportObjectiveConfig(
        kind="growth",
        sample_set=samples,
        ntheta=8,
        mboz=21,
        nboz=21,
        n_laguerre=1,
        n_hermite=1,
    )
    table = np.zeros((2, 1, 2, len(SOLVER_OBJECTIVE_NAMES)))
    table[..., SOLVER_OBJECTIVE_NAMES.index("gamma")] = np.asarray(
        [[[1.0, 2.0]], [[3.0, 4.0]]]
    )
    monkeypatch.setattr(
        mod,
        "_static_grid_options_from_ky_values",
        lambda ky_values, *, min_ny: {
            "selected_ky_indices": (1, 2),
            "ny": 6,
            "ly": 2.0,
        },
    )
    monkeypatch.setattr(
        mod, "_transport_feature_table_from_state", lambda *args, **kwargs: table
    )

    stats = mod.sample_statistics_from_state(
        ctx=SimpleNamespace(static=object(), indata=object()),
        state=object(),
        config=config,
        include_rows=True,
    )

    assert stats["n_samples"] == 4
    assert stats["weighted_mean"] == pytest.approx(2.5)
    assert stats["weighted_std"] == pytest.approx(np.sqrt(1.25))
    assert stats["weighted_standard_error"] == pytest.approx(np.sqrt(1.25) / 2.0)
    assert len(stats["rows"]) == 4
    assert stats["rows_included"] is True
    assert "not stochastic" in stats["claim_scope"]


def test_sample_statistics_from_objective_table_reuses_batched_table() -> None:
    samples = StellaratorITGSampleSet(
        surfaces=(0.45, 0.64),
        alphas=(0.0,),
        ky_values=(0.1, 0.2),
    )
    config = VMECJAXTransportObjectiveConfig(
        kind="quasilinear_flux",
        sample_set=samples,
        ntheta=8,
        mboz=21,
        nboz=21,
        n_laguerre=1,
        n_hermite=1,
    )
    objective_table = np.asarray([[[[0.5], [1.0]]], [[[1.5], [2.0]]]])

    stats = mod.sample_statistics_from_objective_table(
        objective_table=objective_table,
        config=config,
        include_rows=True,
    )

    assert stats["n_samples"] == 4
    assert stats["weighted_mean"] == pytest.approx(1.25)
    assert stats["weighted_std"] == pytest.approx(np.sqrt(0.3125))
    assert stats["weighted_standard_error"] == pytest.approx(np.sqrt(0.3125) / 2.0)
    assert [row["value"] for row in stats["rows"]] == [0.5, 1.0, 1.5, 2.0]


def test_objective_table_exposes_all_quasilinear_landscape_methods() -> None:
    samples = StellaratorITGSampleSet(
        surfaces=(0.64,),
        alphas=(0.0,),
        ky_values=(0.1, 0.4),
    )
    config = VMECJAXTransportObjectiveConfig(
        kind="quasilinear_flux",
        sample_set=samples,
        ntheta=8,
        mboz=21,
        nboz=21,
        n_laguerre=1,
        n_hermite=1,
    )
    table = np.zeros((1, 1, 2, len(SOLVER_OBJECTIVE_NAMES)))
    table[..., SOLVER_OBJECTIVE_NAMES.index("gamma")] = np.asarray([[[0.2, -0.3]]])
    table[..., SOLVER_OBJECTIVE_NAMES.index("kperp_eff2")] = np.asarray([[[0.5, 2.0]]])
    table[..., SOLVER_OBJECTIVE_NAMES.index("linear_heat_flux_weight")] = np.asarray(
        [[[10.0, 20.0]]]
    )
    table[..., SOLVER_OBJECTIVE_NAMES.index("mixing_length_heat_flux_proxy")] = (
        np.asarray([[[4.0, 5.0]]])
    )

    linear_weight = mod._objective_table_from_feature_table(
        table,
        config,
        transport_kind="quasilinear_flux_linear_weight",
        ky_values=samples.ky_values,
    )
    mixing_length = mod._objective_table_from_feature_table(
        table,
        config,
        transport_kind="quasilinear_flux_mixing_length",
        ky_values=samples.ky_values,
    )
    lapillonne = mod._objective_table_from_feature_table(
        table,
        config,
        transport_kind="quasilinear_flux_lapillonne_2011",
        ky_values=samples.ky_values,
    )
    absolute_growth = mod._objective_table_from_feature_table(
        table,
        config,
        transport_kind="quasilinear_flux_absolute_growth_mixing_length",
        ky_values=samples.ky_values,
    )
    shape_aware = mod._objective_table_from_feature_table(
        table,
        config,
        transport_kind="quasilinear_flux_shape_aware_power_law",
        ky_values=samples.ky_values,
        shape_aware_exponent=1.0,
    )

    np.testing.assert_allclose(linear_weight[..., 0], [[[10.0, 20.0]]])
    np.testing.assert_allclose(mixing_length[..., 0], [[[4.0, 5.0]]])
    np.testing.assert_allclose(lapillonne[..., 0], [[[4.0, 5.0]]])
    np.testing.assert_allclose(absolute_growth[..., 0], [[[4.0, 3.0]]])
    np.testing.assert_allclose(shape_aware[..., 0], [[[5.0, 40.0]]])


def test_parse_args_defaults_to_multisample_admission_set(tmp_path: Path) -> None:
    args = mod.parse_args(
        [
            "--input",
            "input.final",
            "--out-json",
            str(tmp_path / "metric.json"),
        ]
    )

    assert args.surfaces == mod.DEFAULT_SURFACES
    assert args.alphas == mod.DEFAULT_ALPHAS
    assert args.ky_values == mod.DEFAULT_KY_VALUES
    assert args.transport_kind == "growth"
    assert args.mboz == 21
    assert args.nboz == 21
    assert args.surface_chunk_size == 0
    assert args.ql_shape_aware_exponent == 0.5


def test_parse_args_accepts_batched_metric_output_dir(tmp_path: Path) -> None:
    args = mod.parse_args(
        [
            "--input",
            "input.final",
            "--out-json",
            str(tmp_path / "batch.json"),
            "--transport-kind",
            "all",
            "--out-json-dir",
            str(tmp_path / "metrics"),
            "--out-wout",
            str(tmp_path / "wout.nc"),
        ]
    )

    assert args.transport_kind == "all"
    assert args.out_json_dir == tmp_path / "metrics"
    assert args.out_wout == tmp_path / "wout.nc"
