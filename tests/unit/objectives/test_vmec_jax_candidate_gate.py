from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.objectives.vmec_candidate_admission import (
    build_authoritative_wout_candidate_gate,
    build_solved_vmec_candidate_gate,
    build_wout_reproducibility_gate,
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


def test_final_iota_profiles_from_vmec_result_returns_none_without_solved_state() -> (
    None
):
    assert final_iota_profiles_from_vmec_result(SimpleNamespace(history={})) is None


def test_candidate_gate_extracts_iota_profiles_from_vmec_jax_state(monkeypatch) -> None:
    calls: list[tuple[object, object, object, int]] = []

    def fake_profiles_from_state(*, state, static, indata, signgs):
        calls.append((state, static, indata, signgs))
        return None, np.asarray([0.0, 0.411, 0.415]), np.asarray([0.412, 0.416])

    fake_vmec_jax = SimpleNamespace(
        equilibrium_iota_profiles_from_state=fake_profiles_from_state
    )
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

    fake_vmec_jax = SimpleNamespace(
        equilibrium_iota_profiles_from_state=fake_profiles_from_state
    )
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


def test_candidate_gate_uses_standalone_qs_not_assembled_transport_block(
    monkeypatch,
) -> None:
    def fake_profiles_from_state(*, state, static, indata, signgs):
        return None, np.asarray([0.0, 0.411, 0.415]), np.asarray([0.412, 0.416])

    class FakeQS:
        def __init__(self, *, helicity_m, helicity_n, surfaces):
            assert helicity_m == 1
            assert helicity_n == 0
            assert np.asarray(surfaces).shape[0] == 11

        def total(self, ctx, state):
            assert state == "state"
            assert ctx.signgs == 1
            return 0.009

    fake_vmec_jax = SimpleNamespace(
        equilibrium_iota_profiles_from_state=fake_profiles_from_state,
        QuasisymmetryRatioResidual=FakeQS,
    )
    monkeypatch.setitem(sys.modules, "vmec_jax", fake_vmec_jax)

    class FakeOptimizer:
        _static = SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0]))
        _indata = "indata"
        _signgs = 1
        _flux = "flux"
        _helicity_m = 1
        _helicity_n = 0

        def _evaluate_residuals_from_state(self, _state):
            return {"transport_contaminated_block": 99.0}

        def _qs_total_from_state(self, _state, _residuals):
            return 99.0

    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_optimizer=FakeOptimizer(),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.009
    assert report["checks"]["quasisymmetry"]["source"] == "vmec_jax_state"


def test_candidate_gate_state_qs_falls_back_to_optimizer_method(monkeypatch) -> None:
    def fake_profiles_from_state(*, state, static, indata, signgs):
        return None, np.asarray([0.0, 0.411, 0.415]), np.asarray([0.412, 0.416])

    fake_vmec_jax = SimpleNamespace(
        equilibrium_iota_profiles_from_state=fake_profiles_from_state
    )
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


def test_final_iota_profiles_from_vmec_result_handles_vmec_jax_failure(
    monkeypatch,
) -> None:
    def fake_profiles_from_state(**_kwargs):
        raise RuntimeError("not converged")

    fake_vmec_jax = SimpleNamespace(
        equilibrium_iota_profiles_from_state=fake_profiles_from_state
    )
    monkeypatch.setitem(sys.modules, "vmec_jax", fake_vmec_jax)
    result = SimpleNamespace(
        final_state=object(),
        final_optimizer=SimpleNamespace(_static=None, _indata=None, _signgs=-1),
    )

    assert final_iota_profiles_from_vmec_result(result) is None


def test_wout_reproducibility_gate_rejects_iota_drift() -> None:
    report = build_wout_reproducibility_gate(
        {
            "source": "optimizer_state_wout",
            "aspect": 5.000154,
            "mean_iota": 0.41020,
            "min_iotas_excluding_axis": 0.40567,
            "min_iotaf": 0.40550,
        },
        {
            "source": "input_final_rerun_wout",
            "aspect": 5.000154,
            "mean_iota": 0.40851,
            "min_iotas_excluding_axis": 0.39598,
            "min_iotaf": 0.39581,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        iota_profile_floor=None,
        mean_iota_repro_atol=5.0e-4,
    )

    assert report["passed"] is False
    assert report["checks"]["rerun_mean_iota_admission"]["passed"] is False
    assert report["checks"]["mean_iota_reproducibility"]["passed"] is False
    assert report["checks"]["mean_iota_reproducibility"][
        "absolute_drift"
    ] == pytest.approx(0.00169)
    json.dumps(report, allow_nan=False)


def test_wout_reproducibility_gate_accepts_matching_rerun() -> None:
    report = build_wout_reproducibility_gate(
        {
            "source": "optimizer_state_wout",
            "aspect": 5.000154,
            "mean_iota": 0.41020,
            "min_iotas_excluding_axis": 0.40567,
            "min_iotaf": 0.40550,
        },
        {
            "source": "input_final_rerun_wout",
            "aspect": 5.0001542,
            "mean_iota": 0.41010,
            "min_iotas_excluding_axis": 0.40561,
            "min_iotaf": 0.40545,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        iota_profile_floor=None,
        mean_iota_repro_atol=5.0e-4,
        aspect_repro_atol=5.0e-7,
        profile_repro_atol=5.0e-4,
    )

    assert report["passed"] is True
    assert report["checks"]["rerun_mean_iota_admission"]["passed"] is True


def test_authoritative_wout_candidate_gate_accepts_mapping_with_qs() -> None:
    report = build_authoritative_wout_candidate_gate(
        {
            "source": "deterministic_rerun_wout",
            "aspect": 5.0001,
            "mean_iota": -0.411,
            "min_iotas_excluding_axis": 0.405,
            "min_iotaf": 0.404,
            "qs_residual": 2.0e-3,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["value"] == pytest.approx(0.411)
    assert report["checks"]["quasisymmetry"]["source"] == "mapping"
    json.dumps(report, allow_nan=False)


def test_authoritative_wout_candidate_gate_rejects_missing_qs() -> None:
    report = build_authoritative_wout_candidate_gate(
        {
            "source": "deterministic_rerun_wout",
            "aspect": 5.0001,
            "mean_iota": 0.411,
            "min_iotas_excluding_axis": 0.405,
            "min_iotaf": 0.404,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is False
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["quasisymmetry"]["error"] == "missing_qs_residual"


def test_authoritative_wout_candidate_gate_reads_wout_file_with_profile_floor(
    tmp_path, monkeypatch
) -> None:
    class FakeVar:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, _key):
            return np.asarray(self.value)

    class FakeDataset:
        def __init__(self, path):
            assert path == tmp_path / "wout_final_rerun.nc"
            self.variables = {
                "aspect": FakeVar(5.0002),
                "iotas": FakeVar([0.0, 0.412, 0.418]),
                "iotaf": FakeVar([0.413, 0.419]),
            }

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_load_wout(path):
        assert path == tmp_path / "wout_final_rerun.nc"
        return "loaded-wout"

    def fake_qs_from_wout(wout, *, surfaces, helicity_m, helicity_n, ntheta, nphi):
        assert wout == "loaded-wout"
        assert tuple(np.asarray(surfaces, dtype=float)) == (0.0, 0.5, 1.0)
        assert (helicity_m, helicity_n, ntheta, nphi) == (1, 0, 31, 32)
        return {"total": 0.003}

    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(
        sys.modules,
        "vmec_jax",
        SimpleNamespace(
            load_wout=fake_load_wout,
            quasisymmetry_ratio_residual_from_wout=fake_qs_from_wout,
        ),
    )

    report = build_authoritative_wout_candidate_gate(
        tmp_path / "wout_final_rerun.nc",
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.411,
        qs_surfaces=(0.0, 0.5, 1.0),
        qs_ntheta=31,
        qs_nphi=32,
    )

    assert report["passed"] is True
    assert report["authoritative_wout"]["mean_iota"] == pytest.approx(0.415)
    assert report["checks"]["iota_profile"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["source"] == "vmec_jax_wout"


def test_authoritative_wout_candidate_gate_reports_wout_load_errors(
    tmp_path, monkeypatch
) -> None:
    def broken_dataset(_path):
        raise OSError("missing variable")

    def broken_load_wout(_path):
        raise RuntimeError("bad wout")

    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=broken_dataset))
    monkeypatch.setitem(
        sys.modules, "vmec_jax", SimpleNamespace(load_wout=broken_load_wout)
    )

    report = build_authoritative_wout_candidate_gate(
        tmp_path / "bad_wout.nc",
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
    )

    assert report["passed"] is False
    assert report["authoritative_wout"]["aspect"] is None
    assert report["checks"]["iota_profile"]["passed"] is False
    assert report["checks"]["quasisymmetry"]["source"] == "vmec_jax_wout_error"
