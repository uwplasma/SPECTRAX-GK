from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk import StellaratorITGSampleSet, VMECJAXTransportObjectiveConfig
from spectraxgk.solver_objective_gradients import SOLVER_OBJECTIVE_NAMES


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "evaluate_vmec_jax_spectrax_transport_metric.py"
spec = importlib.util.spec_from_file_location("evaluate_vmec_jax_spectrax_transport_metric", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


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
    )

    assert report["kind"] == "vmec_jax_spectrax_transport_metric_eval"
    assert report["transport_objective_final"] == 0.067
    assert report["spectrax_objective_final"] == 0.067
    assert report["transport_metric_final"] == 0.067
    assert report["sample_set"]["n_samples"] == 18
    assert report["spectrax_config"]["gradient_scope"] == "eigenvalue_growth_ad"
    assert report["spectrax_config"]["surface_chunk_size"] == 0
    assert "not an optimization" in report["claim_scope"]
    assert "long-window" in report["next_action"]
    json.dumps(mod._json_safe(report), allow_nan=False)


def test_sample_statistics_from_state_reports_weighted_reduced_spread(monkeypatch: pytest.MonkeyPatch) -> None:
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
    table[..., SOLVER_OBJECTIVE_NAMES.index("gamma")] = np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]])
    monkeypatch.setattr(
        mod,
        "_static_grid_options_from_ky_values",
        lambda ky_values, *, min_ny: {"selected_ky_indices": (1, 2), "ny": 6, "ly": 2.0},
    )
    monkeypatch.setattr(mod, "_transport_feature_table_from_state", lambda *args, **kwargs: table)

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
