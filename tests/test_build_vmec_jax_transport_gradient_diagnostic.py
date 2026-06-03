from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
