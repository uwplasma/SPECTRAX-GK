from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from spectraxgk.vmec_jax_candidate_gate import (
    build_solved_vmec_candidate_gate,
    final_iota_profiles_from_vmec_result,
)


POLICY = {
    "target_aspect": 6.0,
    "aspect_atol": 5.0e-2,
    "min_abs_mean_iota": 0.41,
    "qs_residual_max": 5.0e-2,
    "iota_profile_floor": 0.41,
}


def test_candidate_gate_rejects_bad_history_without_nonfinite_json() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": "bad", "iota_final": np.inf, "qs_final": np.nan},
        **POLICY,
        iota_profiles=(np.asarray([0.0, 0.412]), np.asarray([0.413])),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["value"] is None
    assert report["checks"]["aspect"]["passed"] is False
    assert report["checks"]["mean_iota"]["value"] is None
    assert report["checks"]["quasisymmetry"]["value"] is None
    json.dumps(report, allow_nan=False)


def test_candidate_gate_requires_iota_profiles_when_floor_is_enabled() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": 6.0, "iota_final": 0.42, "qs_final": 0.02},
        **POLICY,
    )

    assert report["passed"] is False
    assert report["checks"]["iota_profile"]["source"] == "missing"
    assert report["checks"]["iota_profile"]["passed"] is False


def test_candidate_gate_can_disable_profile_floor_for_fast_diagnostic_use() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": 6.0, "iota_final": 0.42, "qs_final": 0.02},
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is True
    assert report["checks"]["iota_profile"]["floor"] is None
    assert report["checks"]["iota_profile"]["passed"] is True


def test_final_iota_profiles_from_vmec_result_returns_none_without_solved_state() -> None:
    assert final_iota_profiles_from_vmec_result(SimpleNamespace(history={})) is None


def test_candidate_gate_extracts_iota_profiles_from_vmec_jax_state(monkeypatch) -> None:
    calls: list[tuple[object, object, object, int]] = []

    def fake_profiles_from_state(*, state, static, indata, signgs):
        calls.append((state, static, indata, signgs))
        return None, np.asarray([0.0, 0.411, 0.415]), np.asarray([0.412, 0.416])

    fake_vmec_jax = SimpleNamespace(equilibrium_iota_profiles_from_state=fake_profiles_from_state)
    monkeypatch.setitem(sys.modules, "vmec_jax", fake_vmec_jax)
    optimizer = SimpleNamespace(_static="static", _indata="indata", _signgs=1)
    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": -0.42, "qs_final": 0.02},
        final_state="state",
        final_optimizer=optimizer,
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert calls == [("state", "static", "indata", 1)]
    assert report["passed"] is True
    assert report["checks"]["mean_iota"]["value"] == 0.42
    assert report["checks"]["iota_profile"]["source"] == "vmec_jax_state"
    assert report["checks"]["iota_profile"]["minimum_iotas_excluding_axis"] == 0.411


def test_candidate_gate_prefers_independent_state_qs_over_history(monkeypatch) -> None:
    def fake_profiles_from_state(*, state, static, indata, signgs):
        return None, np.asarray([0.0, 0.411, 0.415]), np.asarray([0.412, 0.416])

    fake_vmec_jax = SimpleNamespace(equilibrium_iota_profiles_from_state=fake_profiles_from_state)
    monkeypatch.setitem(sys.modules, "vmec_jax", fake_vmec_jax)

    class FakeOptimizer:
        _static = "static"
        _indata = "indata"
        _signgs = 1

        def _evaluate_residuals_from_state(self, state):
            assert state == "state"
            return {"combined": 99.0}

        def _qs_total_from_state(self, state, residuals):
            assert state == "state"
            assert residuals == {"combined": 99.0}
            return 0.013

    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_optimizer=FakeOptimizer(),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.013
    assert report["checks"]["quasisymmetry"]["source"] == "vmec_jax_state"


def test_candidate_gate_state_qs_falls_back_to_optimizer_method(monkeypatch) -> None:
    def fake_profiles_from_state(*, state, static, indata, signgs):
        return None, np.asarray([0.0, 0.411, 0.415]), np.asarray([0.412, 0.416])

    fake_vmec_jax = SimpleNamespace(equilibrium_iota_profiles_from_state=fake_profiles_from_state)
    monkeypatch.setitem(sys.modules, "vmec_jax", fake_vmec_jax)

    class FakeOptimizer:
        _static = "static"
        _indata = "indata"
        _signgs = 1

        def _evaluate_residuals_from_state(self, _state):
            raise RuntimeError("state residual unavailable")

        def quasisymmetry_objective(self, params):
            assert params == (1.0, 2.0)
            return 0.017

    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_params=(1.0, 2.0),
        final_optimizer=FakeOptimizer(),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.017
    assert report["checks"]["quasisymmetry"]["source"] == "vmec_jax_state"


def test_final_iota_profiles_from_vmec_result_handles_vmec_jax_failure(monkeypatch) -> None:
    def fake_profiles_from_state(**_kwargs):
        raise RuntimeError("not converged")

    fake_vmec_jax = SimpleNamespace(equilibrium_iota_profiles_from_state=fake_profiles_from_state)
    monkeypatch.setitem(sys.modules, "vmec_jax", fake_vmec_jax)
    result = SimpleNamespace(
        final_state=object(),
        final_optimizer=SimpleNamespace(_static=None, _indata=None, _signgs=-1),
    )

    assert final_iota_profiles_from_vmec_result(result) is None
