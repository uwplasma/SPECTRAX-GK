from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_jax_transport_gradient_diagnostic.py"
spec = importlib.util.spec_from_file_location("build_vmec_jax_transport_gradient_diagnostic", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_gradient_diagnostic_defaults_to_multisample_transport_contract(tmp_path: Path) -> None:
    args = mod._parse_args(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
        ]
    )

    sample_set = mod._sample_set_from_args(args)
    summary = mod.transport_objective_sample_summary(sample_set)

    assert args.surfaces == mod.DEFAULT_TRANSPORT_SURFACES
    assert args.alphas == mod.DEFAULT_TRANSPORT_ALPHAS
    assert args.ky_values == mod.DEFAULT_TRANSPORT_KY_VALUES
    assert summary["passed"] is True
    assert summary["sample_count"] == 18


def test_gradient_diagnostic_fails_closed_for_underresolved_sample_set(tmp_path: Path, monkeypatch) -> None:
    def unexpected_stage(_args):
        raise AssertionError("under-resolved sample set should fail before VMEC-JAX stage construction")

    monkeypatch.setattr(mod, "_build_stage", unexpected_stage)
    with pytest.raises(ValueError, match="under-resolved transport-gradient sample set"):
        mod.main(
            [
                "--input",
                str(tmp_path / "input.final"),
                "--out-json",
                str(tmp_path / "gradient.json"),
                "--surfaces",
                "0.64",
                "--alphas",
                "0.0",
                "--ky-values",
                "0.3",
            ]
        )


def test_gradient_diagnostic_records_sample_coverage(tmp_path: Path, monkeypatch) -> None:
    fake_stage = SimpleNamespace(specs=[object(), object()], optimizer=object())

    def fake_stage_builder(_args):
        return fake_stage, {"setup_key": "setup_value"}

    def fake_report(_optimizer, **_kwargs):
        return {
            "kind": "vmec_jax_transport_gradient_diagnostic",
            "finite": True,
            "transport_sensitivity_detected": True,
        }

    def fake_write(report, out_json):
        Path(out_json).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return Path(out_json)

    monkeypatch.setattr(mod, "_build_stage", fake_stage_builder)
    monkeypatch.setattr(mod, "build_boundary_transport_gradient_report", fake_report)
    monkeypatch.setattr(mod, "write_boundary_transport_gradient_report", fake_write)

    rc = mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["setup"] == {"setup_key": "setup_value"}
    assert payload["objective_sample_summary"]["passed"] is True
    assert payload["objective_sample_summary"]["sample_count"] == 18
    assert payload["nonlinear_audit_policy"]["recommended_ky_values"] == [0.1, 0.3, 0.5]


def test_gradient_diagnostic_fd_consistency_passes_for_matching_reverse_gradient(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class MatchingOptimizer:
        _specs = (SimpleNamespace(name="rc01"), SimpleNamespace(name="zs01"))

        def residual_fun(self, params):
            params_array = np.asarray(params, dtype=float)
            return np.asarray([1.25 + 2.0 * params_array[0] - 3.0 * params_array[1]], dtype=float)

        def objective_and_gradient_fun(self, params):
            residual = float(self.residual_fun(params)[0])
            return 0.5 * residual**2, residual * np.asarray([2.0, -3.0], dtype=float)

    def fake_stage_builder(_args):
        return (
            SimpleNamespace(
                specs=MatchingOptimizer._specs,
                optimizer=MatchingOptimizer(),
            ),
            {"setup_key": "setup_value"},
        )

    monkeypatch.setattr(mod, "_build_stage", fake_stage_builder)

    rc = mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
            "--fd-check-indices",
            "0,1",
            "--require-fd-consistency",
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    fd = payload["finite_difference_consistency"]
    assert rc == 0
    assert fd["enabled"] is True
    assert fd["passed"] is True
    assert fd["blockers"] == []
    assert fd["rows"][0]["name"] == "rc01"
    assert fd["rows"][1]["name"] == "zs01"


def test_gradient_diagnostic_fd_consistency_fails_for_disconnected_reverse_gradient(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class DisconnectedOptimizer:
        _specs = (SimpleNamespace(name="rc01"), SimpleNamespace(name="zs01"))

        def residual_fun(self, params):
            params_array = np.asarray(params, dtype=float)
            return np.asarray([1.25 + 2.0 * params_array[0] - 3.0 * params_array[1]], dtype=float)

        def objective_and_gradient_fun(self, params):
            residual = float(self.residual_fun(params)[0])
            return 0.5 * residual**2, np.zeros(2, dtype=float)

    def fake_stage_builder(_args):
        return (
            SimpleNamespace(
                specs=DisconnectedOptimizer._specs,
                optimizer=DisconnectedOptimizer(),
            ),
            {"setup_key": "setup_value"},
        )

    monkeypatch.setattr(mod, "_build_stage", fake_stage_builder)

    rc = mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
            "--fd-check-indices",
            "0,1",
            "--require-fd-consistency",
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    fd = payload["finite_difference_consistency"]
    assert rc == 3
    assert fd["enabled"] is True
    assert fd["passed"] is False
    assert "ad_fd_mismatch" in fd["blockers"]
    assert fd["max_abs_fd_cost_gradient"] > 0.0
    assert fd["rows"][0]["fd_cost_gradient"] == pytest.approx(2.5)
    assert fd["rows"][1]["fd_cost_gradient"] == pytest.approx(-3.75)


def test_gradient_diagnostic_surface_chunking_aggregates_raw_weighted_gradient(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeOptimizer:
        _specs = (SimpleNamespace(name="rc01"), SimpleNamespace(name="zs01"))

        def __init__(self, residual: float, residual_gradient: np.ndarray) -> None:
            self.residual = float(residual)
            self.residual_gradient = np.asarray(residual_gradient, dtype=float)

        def residual_fun(self, _params):
            return np.asarray([self.residual], dtype=float)

        def objective_and_gradient_fun(self, _params):
            return 0.5 * self.residual**2, self.residual * self.residual_gradient

    by_surface = {
        0.45: (1.0, np.asarray([2.0, 0.0])),
        0.64: (3.0, np.asarray([4.0, 0.0])),
        0.78: (5.0, np.asarray([6.0, 0.0])),
    }
    seen_surfaces: list[float] = []

    def fake_stage_builder(args):
        assert args.spectrax_objective_transform == "raw"
        assert args.transport_weight == 1.0
        assert len(args.surfaces) == 1
        surface = round(float(args.surfaces[0]), 2)
        seen_surfaces.append(surface)
        residual, gradient = by_surface[surface]
        return (
            SimpleNamespace(
                specs=FakeOptimizer._specs,
                optimizer=FakeOptimizer(residual, gradient),
            ),
            {"surfaces": [surface]},
        )

    monkeypatch.setattr(mod, "_build_stage", fake_stage_builder)

    rc = mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
            "--surface-gradient-chunk-size",
            "1",
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    expected_raw = (1.0 + 3.0 + 5.0) / 3.0
    expected_raw_gradient = (np.asarray([2.0, 0.0]) + np.asarray([4.0, 0.0]) + np.asarray([6.0, 0.0])) / 3.0
    expected_residual = np.log1p(expected_raw)
    expected_residual_gradient = expected_raw_gradient / (1.0 + expected_raw)
    expected_cost_gradient = expected_residual * expected_residual_gradient

    assert rc == 0
    assert seen_surfaces == [0.45, 0.64, 0.78]
    assert payload["chunked_gradient"]["enabled"] is True
    assert payload["chunked_gradient"]["chunk_count"] == 3
    assert payload["chunked_gradient"]["raw_weighted_residual"] == pytest.approx(expected_raw)
    assert payload["chunked_gradient"]["raw_weighted_gradient_norm_l2"] == pytest.approx(
        np.linalg.norm(expected_raw_gradient)
    )
    assert payload["residual_norm_l2"] == pytest.approx(expected_residual)
    assert payload["gradient_norm_l2"] == pytest.approx(np.linalg.norm(expected_cost_gradient))
    assert payload["top_gradient_components"][0]["name"] == "rc01"
