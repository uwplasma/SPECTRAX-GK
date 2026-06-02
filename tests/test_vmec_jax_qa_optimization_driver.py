from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import spectraxgk
from spectraxgk.vmec_jax_candidate_gate import build_solved_vmec_candidate_gate


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "examples" / "optimization" / "vmec_jax_qa_low_turbulence_optimization.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("vmec_jax_qa_low_turbulence_optimization", DRIVER)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_solved_wout_candidate_gate_passes_valid_qa_branch() -> None:
    assert spectraxgk.build_solved_vmec_candidate_gate is build_solved_vmec_candidate_gate
    result = SimpleNamespace(
        history={"aspect_final": 5.999233, "iota_final": 0.427011, "qs_final": 2.604013e-2},
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.410131, 0.414]),
            np.asarray([0.410706, 0.414]),
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is True
    assert report["checks"]["iota_profile"]["passed"] is True
    json.dumps(report, allow_nan=False)


def test_solved_wout_candidate_gate_rejects_transport_branch_that_breaks_constraints() -> None:
    result = SimpleNamespace(
        history={"aspect_final": 5.996817, "iota_final": 0.425028, "qs_final": 1.091236e-1},
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.402043, 0.414]),
            np.asarray([0.402493, 0.414]),
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["iota_profile"]["passed"] is False
    assert "do not promote" in report["next_action"]
    json.dumps(report, allow_nan=False)


def test_driver_transport_metric_from_result_uses_final_state_context() -> None:
    mod = _load_driver()

    class FakeTransport:
        config = SimpleNamespace(kind="growth", objective_transform="log1p", objective_scale=3.0)

        def J(self, ctx, state):
            assert state == "final-state"
            assert ctx.indata == "indata"
            assert ctx.signgs == -1
            assert ctx.flux == "flux"
            return np.asarray(0.125)

    result = SimpleNamespace(
        final_state="final-state",
        final_optimizer=SimpleNamespace(
            _static=SimpleNamespace(s=np.asarray([0.0, 1.0])),
            _indata="indata",
            _signgs=-1,
            _flux="flux",
        ),
    )

    metric = mod._transport_metric_from_result(FakeTransport(), result)

    assert metric["transport_objective_final"] == 0.125
    assert metric["spectrax_objective_final"] == 0.125
    assert metric["transport_metric_final"] == 0.125
    assert metric["transport_objective_source"] == "final_vmec_jax_state"
    assert metric["transport_metric_kind"] == "growth"
    json.dumps(metric, allow_nan=False)


def test_driver_updates_history_with_transport_metric(tmp_path: Path) -> None:
    mod = _load_driver()
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps({"objective_final": 1.0, "history": [{"objective": 1.0}]}),
        encoding="utf-8",
    )

    mod._update_history_with_transport_metric(
        path,
        {
            "transport_objective_final": 0.2,
            "spectrax_objective_final": 0.2,
            "transport_metric_final": 0.2,
            "transport_objective_error": None,
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["transport_objective_final"] == 0.2
    assert payload["spectrax_objective_final"] == 0.2
    assert payload["transport_metric_final"] == 0.2
    assert "transport_objective_error" not in payload
    assert payload["history"][-1]["transport_objective_final"] == 0.2
