from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "write_vmec_jax_projected_transport_line_search_inputs.py"
spec = importlib.util.spec_from_file_location("write_vmec_jax_projected_transport_line_search_inputs", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _gradient_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_transport_gradient_diagnostic",
                "parameter_count": 4,
                "top_gradient_components": [
                    {"parameter_index": 1, "gradient": -3.0, "name": "zs10"},
                    {"parameter_index": 3, "gradient": 4.0, "name": "rc11"},
                    {"parameter_index": 0, "gradient": 2.0, "name": "rc01"},
                    {"parameter_index": 2, "gradient": -1.0, "name": "zs11"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_projected_writer_defaults_to_multisample_transport_contract(tmp_path: Path) -> None:
    args = mod._parse_args(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(tmp_path / "gradient.json"),
            "--outdir",
            str(tmp_path / "out"),
        ]
    )

    sample_set = mod._sample_set_from_args(args)
    summary = mod.transport_objective_sample_summary(sample_set)

    assert args.surfaces == mod.DEFAULT_TRANSPORT_SURFACES
    assert args.alphas == mod.DEFAULT_TRANSPORT_ALPHAS
    assert args.ky_values == mod.DEFAULT_TRANSPORT_KY_VALUES
    assert summary["passed"] is True
    assert summary["sample_count"] == 18


def test_projected_writer_manifest_records_sample_coverage(tmp_path: Path, monkeypatch) -> None:
    gradient = tmp_path / "gradient.json"
    _gradient_report(gradient)
    saved: list[tuple[Path, object]] = []

    class FakeOptimizer:
        def save_input(self, path, delta):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("! projected input\n", encoding="utf-8")
            saved.append((Path(path), delta))

    fake_stage = SimpleNamespace(specs=[object(), object(), object(), object()], optimizer=FakeOptimizer())
    monkeypatch.setattr(mod, "_build_stage", lambda _args: fake_stage)

    rc = mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(gradient),
            "--outdir",
            str(tmp_path / "out"),
            "--steps",
            "1e-3,2e-3",
            "--top-n",
            "4",
        ]
    )

    manifest_path = tmp_path / "out" / "projected_line_search_inputs.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert len(saved) == 2
    assert payload["objective_sample_summary"]["passed"] is True
    assert payload["objective_sample_summary"]["sample_count"] == 18
    assert payload["transport_objective_sample_set"]["surfaces"] == [0.45, 0.64, 0.78]
    assert payload["transport_objective_sample_set"]["alphas"] == [0.0, 0.7853981633974483]
    assert payload["transport_objective_sample_set"]["ky_values"] == [0.1, 0.3, 0.5]
    command = payload["rows"][0]["replay_command"]
    assert "--surfaces" in command
    assert "0.45,0.64,0.78" in command
    assert "--alphas" in command
    assert "0.0,0.7853981633974483" in command
    assert "--ky-values" in command
    assert "0.1,0.3,0.5" in command


def test_projected_writer_fails_closed_for_underresolved_sample_set(tmp_path: Path, monkeypatch) -> None:
    gradient = tmp_path / "gradient.json"
    _gradient_report(gradient)

    def unexpected_stage(_args):
        raise AssertionError("under-resolved sample set should fail before VMEC-JAX stage construction")

    monkeypatch.setattr(mod, "_build_stage", unexpected_stage)
    with pytest.raises(ValueError, match="under-resolved transport objective sample set"):
        mod.main(
            [
                "--input",
                str(tmp_path / "input.final"),
                "--gradient-json",
                str(gradient),
                "--outdir",
                str(tmp_path / "out"),
                "--surfaces",
                "0.64",
                "--alphas",
                "0.0",
                "--ky-values",
                "0.3",
            ]
        )
