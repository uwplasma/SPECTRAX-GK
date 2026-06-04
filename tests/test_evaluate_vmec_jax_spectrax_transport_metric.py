from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from spectraxgk import StellaratorITGSampleSet, VMECJAXTransportObjectiveConfig


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
    assert "not an optimization" in report["claim_scope"]
    assert "long-window" in report["next_action"]
    json.dumps(mod._json_safe(report), allow_nan=False)


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
